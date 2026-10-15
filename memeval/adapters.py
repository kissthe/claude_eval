"""Adapters: ingest a LoCoMo conversation into a memory system (snapshot on disk) and
retrieve read-only from a copy of that snapshot.

Every retrieved unit is {"text": serialized text shown to the reader,
"source_turn_ids": [...] (possibly empty), "kind": unit type}.
"""
import importlib.util
import json
import pickle
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

import common
from common import EMB_MODEL, LLM_MODEL, API_KEY, BASE_URL, embed_many, embed_one, load_conversation

VENDOR = common.ROOT / "vendor"
ID_RE = re.compile(r"\[(conv-\d+:D\d+:\d+)\]")


def ids_in(text):
    return list(dict.fromkeys(ID_RE.findall(text or "")))


def _stub_hf():
    """A-MEM and MemoryOS import sentence_transformers/transformers at module load but we
    route all embeddings through the shared API embedder, so stub the heavy packages."""
    for name in ("sentence_transformers", "transformers"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.SentenceTransformer = object
            m.AutoModel = m.AutoTokenizer = object
            sys.modules[name] = m


def _tmp_copy(src):
    d = Path(tempfile.mkdtemp(prefix="memeval_"))
    shutil.copytree(src, d / "snap")
    return d / "snap", d


# ======================================================================= A-MEM

def _amem():
    _stub_hf()
    sys.path.insert(0, str(VENDOR / "A-mem"))
    import chromadb
    from chromadb import EmbeddingFunction
    import agentic_memory.retrievers as R

    class ApiEmbedding(EmbeddingFunction):
        def __init__(self, *a, **k):
            pass

        def __call__(self, input):
            return [np.array(v, dtype=np.float32) for v in embed_many(list(input))]

        @staticmethod
        def name():
            return "memeval-api"

    R.SentenceTransformerEmbeddingFunction = ApiEmbedding
    from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
    return AgenticMemorySystem, MemoryNote


class AMem:
    name = "amem"

    def ingest(self, conv_id, target, outdir):
        AgenticMemorySystem, _ = _amem()
        sa, sb, turns = load_conversation(conv_id)
        ams = AgenticMemorySystem(llm_backend="openai", llm_model=LLM_MODEL, api_key=API_KEY)
        stats = {"analyze_fail": 0}
        for i, t in enumerate(turns):
            content = f"[{t['turn_id']}] {t['speaker']}: {t['text']}"
            a = ams.analyze_content(content)  # note construction: keywords / context / tags
            if not a.get("keywords") and a.get("context", "General") == "General":
                stats["analyze_fail"] += 1
            ams.add_note(content, time=t["time"].strftime("%Y%m%d%H%M"),
                         keywords=a.get("keywords", []), context=a.get("context", "General"),
                         tags=a.get("tags", []))
            if (i + 1) % 50 == 0:
                print(f"[amem {conv_id}] {i + 1}/{len(turns)}", flush=True)
        outdir.mkdir(parents=True, exist_ok=True)
        notes = {k: vars(v) for k, v in ams.memories.items()}
        (outdir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=0))
        return {**stats, "n_units": len(notes)}

    def dump(self, snap):
        notes = json.loads((snap / "notes.json").read_text())
        return [self._unit(n) for n in notes.values()]

    @staticmethod
    def _unit(n, neighbor=False):
        text = (f"{n['content']} | time: {n['timestamp']} | context: {n['context']} | "
                f"keywords: {', '.join(n['keywords'])} | tags: {', '.join(n['tags'])}")
        return {"text": text, "source_turn_ids": ids_in(n["content"]), "kind": "neighbor" if neighbor else "note"}

    def retrieve(self, snap, query, target, k=30):
        AgenticMemorySystem, MemoryNote = _amem()
        notes = json.loads((snap / "notes.json").read_text())
        ams = AgenticMemorySystem(llm_backend="openai", llm_model=LLM_MODEL, api_key=API_KEY)
        ams.memories = {nid: MemoryNote(**n) for nid, n in notes.items()}
        ams.consolidate_memories()  # rebuild vector index (embeddings come from the shared cache)
        res = ams.retriever.search(query, k)
        units, seen = [], set()
        # Same composition as A-MEM's find_related_memories_raw: each hit followed by its linked notes.
        for nid in res["ids"][0]:
            if nid in seen or nid not in notes:
                continue
            seen.add(nid)
            units.append(self._unit(notes[nid]))
            for link in notes[nid].get("links", [])[:k]:
                if link in notes and link not in seen:
                    seen.add(link)
                    units.append(self._unit(notes[link], neighbor=True))
        return units


# ======================================================================= MemoryOS

def _memoryos():
    _stub_hf()
    if "memoryos_pkg" in sys.modules:
        return sys.modules["memoryos_pkg"]
    path = VENDOR / "MemoryOS" / "memoryos-pypi"
    spec = importlib.util.spec_from_file_location("memoryos_pkg", path / "__init__.py",
                                                  submodule_search_locations=[str(path)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["memoryos_pkg"] = pkg
    spec.loader.exec_module(pkg)

    def get_embedding(text, *a, **k):
        return np.array(embed_one(text), dtype=np.float32)

    for mod in ("utils", "mid_term", "long_term", "updater"):
        m = sys.modules.get(f"memoryos_pkg.{mod}")
        if m is not None and hasattr(m, "get_embedding"):
            m.get_embedding = get_embedding
    return pkg


def memoryos_pairs(turns, target):
    """MemoryOS stores (user_input, agent_response) pairs and drops pairs with an empty side.
    Target speaker = user, other speaker = agent (as in MemoryOS's LoCoMo eval, speaker_a=user).
    Consecutive turns of the same speaker are merged into one block."""
    blocks = []
    for t in turns:
        line = f"[{t['turn_id']}] {t['speaker']}: {t['text']}"
        if blocks and blocks[-1]["speaker"] == t["speaker"] and blocks[-1]["session"] == t["session"]:
            blocks[-1]["text"] += "\n" + line
        else:
            blocks.append({"speaker": t["speaker"], "session": t["session"], "time": t["time"], "text": line})
    pairs = []
    for b in blocks:
        if b["speaker"] == target:
            pairs.append({"user_input": b["text"], "agent_response": None, "time": b["time"], "session": b["session"]})
        elif pairs and pairs[-1]["agent_response"] is None and pairs[-1]["session"] == b["session"]:
            pairs[-1]["agent_response"] = b["text"]
        else:
            pairs.append({"user_input": f"({target} says nothing)", "agent_response": b["text"],
                          "time": b["time"], "session": b["session"]})
    for p in pairs:
        if p["agent_response"] is None:
            p["agent_response"] = "(no reply)"
    return pairs


class MemoryOS:
    name = "memoryos"

    def _open(self, conv_id, path):
        pkg = _memoryos()
        return pkg.Memoryos(user_id=conv_id, openai_api_key=API_KEY, openai_base_url=BASE_URL,
                            data_storage_path=str(path), llm_model=LLM_MODEL,
                            embedding_model_name=EMB_MODEL)

    def ingest(self, conv_id, target, outdir):
        sa, sb, turns = load_conversation(conv_id)
        outdir.mkdir(parents=True, exist_ok=True)
        m = self._open(conv_id, outdir)
        pairs = memoryos_pairs(turns, target)
        for i, p in enumerate(pairs):
            m.add_memory(user_input=p["user_input"], agent_response=p["agent_response"],
                         timestamp=p["time"].strftime("%Y-%m-%d %H:%M:%S"))
            if (i + 1) % 25 == 0:
                print(f"[memoryos {conv_id}] {i + 1}/{len(pairs)}", flush=True)
        (outdir / "meta.json").write_text(json.dumps({"conv_id": conv_id, "target": target}))
        return {"n_pairs": len(pairs), **m.get_memory_stats()}

    def _units(self, m, conv_id, r):
        units = []
        for p in r["retrieved_pages"]:
            body = f"{p.get('user_input', '')}\n{p.get('agent_response', '')}"
            units.append({"text": f"{body}\n(conversation chain overview: {p.get('meta_info', '')})",
                          "source_turn_ids": ids_in(body), "kind": "page"})
        prof = m.user_long_term_memory.get_raw_user_profile(conv_id)
        if prof and prof.lower() != "none":
            units.append({"text": f"[User profile of the target speaker] {prof}", "source_turn_ids": [], "kind": "profile"})
        for e in r["retrieved_user_knowledge"]:
            units.append({"text": f"[User knowledge] {e['knowledge']}", "source_turn_ids": ids_in(e["knowledge"]), "kind": "user_knowledge"})
        for e in r["retrieved_assistant_knowledge"]:
            units.append({"text": f"[Other-speaker knowledge] {e['knowledge']}", "source_turn_ids": ids_in(e["knowledge"]), "kind": "assistant_knowledge"})
        for qa in m.short_term_memory.get_all():
            body = f"{qa.get('user_input', '')}\n{qa.get('agent_response', '')}"
            units.append({"text": f"[Recent] {body}", "source_turn_ids": ids_in(body), "kind": "short_term"})
        return units

    def retrieve(self, snap, query, target):
        conv_id = json.loads((snap / "meta.json").read_text())["conv_id"]
        work, tmp = _tmp_copy(snap)
        try:
            m = self._open(conv_id, work)
            r = m.retriever.retrieve_context(user_query=query, user_id=conv_id)
            return self._units(m, conv_id, r)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def dump(self, snap):
        conv_id = json.loads((snap / "meta.json").read_text())["conv_id"]
        m = self._open(conv_id, snap)
        units = []
        for s in m.mid_term_memory.sessions.values():
            units.append({"text": f"[Mid-term summary] {s.get('summary', '')}", "source_turn_ids": [], "kind": "summary"})
            for p in s.get("details", []):
                body = f"{p.get('user_input', '')}\n{p.get('agent_response', '')}"
                units.append({"text": body, "source_turn_ids": ids_in(body), "kind": "page"})
        r = {"retrieved_pages": [], "retrieved_user_knowledge": [{"knowledge": e["knowledge"]} for e in m.user_long_term_memory.get_user_knowledge()],
             "retrieved_assistant_knowledge": [{"knowledge": e["knowledge"]} for e in m.assistant_long_term_memory.get_assistant_knowledge()]}
        return units + self._units(m, conv_id, r)


# ======================================================================= Mem0

def _mem0_config(path):
    return {
        "llm": {"provider": "openai", "config": {"model": LLM_MODEL, "api_key": API_KEY, "openai_base_url": BASE_URL}},
        "embedder": {"provider": "openai", "config": {"model": EMB_MODEL, "api_key": API_KEY,
                                                       "openai_base_url": BASE_URL, "embedding_dims": 1536}},
        "vector_store": {"provider": "qdrant", "config": {"path": str(path / "qdrant"), "collection_name": "mem",
                                                           "embedding_model_dims": 1536, "on_disk": True}},
        "history_db_path": str(path / "history.db"),
    }


class Mem0:
    name = "mem0"

    def ingest(self, conv_id, target, outdir, batch=2):
        from mem0 import Memory
        sa, sb, turns = load_conversation(conv_id)
        outdir.mkdir(parents=True, exist_ok=True)
        m = Memory.from_config(_mem0_config(outdir))
        sessions = {}
        for t in turns:
            sessions.setdefault(t["session"], []).append(t)
        n_add = n_err = 0
        for s, ts in sessions.items():
            # As in Mem0's LoCoMo eval: messages in batches of 2, target speaker = user role.
            for i in range(0, len(ts), batch):
                chunk = ts[i:i + batch]
                msgs = [{"role": "user" if t["speaker"] == target else "assistant",
                         "content": f"{t['speaker']}: {t['text']}"} for t in chunk]
                meta = {"turn_ids": ",".join(t["turn_id"] for t in chunk),
                        "session_time": chunk[0]["time"].strftime("%Y-%m-%d")}
                try:
                    m.add(msgs, user_id=target, metadata=meta)
                except Exception as e:
                    n_err += 1
                    print(f"[mem0 {conv_id}] add error: {e}", flush=True)
                n_add += 1
            print(f"[mem0 {conv_id}] session {s} done", flush=True)
        (outdir / "meta.json").write_text(json.dumps({"conv_id": conv_id, "target": target}))
        n_units = len(m.get_all(filters={"user_id": target}, top_k=10000).get("results", []))
        return {"n_add_calls": n_add, "n_add_errors": n_err, "n_units": n_units}

    @staticmethod
    def _unit(r):
        md = r.get("metadata") or {}
        ids = [x for x in (md.get("turn_ids") or "").split(",") if x]
        when = md.get("session_time", "")
        return {"text": f"({when}) {r['memory']} [source turns: {', '.join(ids)}]",
                "source_turn_ids": ids, "kind": "fact"}

    def retrieve(self, snap, query, target, k=60):
        from mem0 import Memory
        work, tmp = _tmp_copy(snap)
        try:
            m = Memory.from_config(_mem0_config(work))
            res = m.search(query, filters={"user_id": target}, top_k=k).get("results", [])
            return [self._unit(r) for r in res]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def dump(self, snap):
        from mem0 import Memory
        work, tmp = _tmp_copy(snap)
        try:
            target = json.loads((snap / "meta.json").read_text())["target"]
            m = Memory.from_config(_mem0_config(work))
            return [self._unit(r) for r in m.get_all(filters={"user_id": target}, top_k=10000).get("results", [])]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================= raw-turn RAG baselines

def _turn_unit(t):
    return {"text": common.turn_line(t), "source_turn_ids": [t["turn_id"]], "kind": "turn"}


class RawBM25:
    name = "raw_bm25"

    def retrieve_conv(self, conv_id, query, target):
        from rank_bm25 import BM25Okapi
        _, _, turns = load_conversation(conv_id)
        tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())
        bm = BM25Okapi([tok(f"{t['speaker']} {t['text']}") for t in turns])
        scores = bm.get_scores(tok(query))
        return [_turn_unit(turns[i]) for i in np.argsort(-scores)]


class RawDense:
    name = "raw_dense"

    def retrieve_conv(self, conv_id, query, target):
        _, _, turns = load_conversation(conv_id)
        M = np.array(embed_many([f"{t['speaker']}: {t['text']}" for t in turns]))
        q = np.array(embed_one(query))
        sims = (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-9)
        return [_turn_unit(turns[i]) for i in np.argsort(-sims)]


MEMORY_SYSTEMS = {"amem": AMem, "memoryos": MemoryOS, "mem0": Mem0}
RAW = {"raw_bm25": RawBM25, "raw_dense": RawDense}
