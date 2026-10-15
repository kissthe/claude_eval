"""Shared utilities: LoCoMo loading, instrumented OpenAI-compatible client, cached embeddings."""
import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "locomo10.json"
PILOT = ROOT.parent / "pilot30"

BASE_URL = os.environ.get("MEMEVAL_BASE_URL", "https://aihubmix.com/v1")
API_KEY = os.environ.get("AIHUBMIX_API_KEY", "")
LLM_MODEL = os.environ.get("MEMEVAL_LLM", "Qwen3.7-flash")
EMB_MODEL = os.environ.get("MEMEVAL_EMB", "text-embedding-3-small")
# Extra request body for every chat call (e.g. '{"enable_thinking": false}'), set after probing the model.
EXTRA_BODY = json.loads(os.environ.get("MEMEVAL_EXTRA_BODY", "{}"))

# The SDKs used by the three systems read these.
os.environ.setdefault("OPENAI_API_KEY", API_KEY)
os.environ.setdefault("OPENAI_BASE_URL", BASE_URL)
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


# ---------------------------------------------------------------- LoCoMo

def parse_locomo_time(s):
    """'4:04 pm on 20 January, 2023' -> datetime."""
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def load_conversation(conv_id):
    """Return (speaker_a, speaker_b, turns). Only original turns plus image captions;
    LoCoMo observations, summaries, QA and image search queries are never read."""
    for c in json.loads(DATA.read_text()):
        if c["sample_id"] != conv_id:
            continue
        cv = c["conversation"]
        n_sessions = max(int(k.split("_")[1]) for k in cv if k.startswith("session_") and k.count("_") == 1)
        turns = []
        for i in range(1, n_sessions + 1):
            key = f"session_{i}"
            if key not in cv:
                continue
            when = parse_locomo_time(cv[f"{key}_date_time"])
            for t in cv[key]:
                text = t["text"]
                media = None
                if t.get("blip_caption"):
                    text = f"{text} [image caption: {t['blip_caption']}]"
                    media = "image_caption"
                turns.append({
                    "turn_id": f"{conv_id}:{t['dia_id']}",
                    "speaker": t["speaker"],
                    "session": i,
                    "time": when,
                    "text": text,
                    "media_type": media,
                })
        smoke = int(os.environ.get("MEMEVAL_SMOKE_TURNS", "0"))  # plumbing tests only
        return cv["speaker_a"], cv["speaker_b"], turns[:smoke] if smoke else turns
    raise KeyError(conv_id)


def turn_line(t):
    return f"[{t['turn_id']}] ({t['time']:%Y-%m-%d}) {t['speaker']}: {t['text']}"


def load_pilot():
    qs = json.loads((PILOT / "pilot_questions.json").read_text())
    ans = {a["sample_id"]: a for a in json.loads((PILOT / "pilot_answers.json").read_text())}
    return qs, ans


# ---------------------------------------------------------------- call logging

_log_lock = threading.Lock()
_log_path = None
_component = "unknown"


def set_call_log(path, component):
    global _log_path, _component
    _log_path = Path(path)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _component = component


def _log(rec):
    if _log_path is None:
        return
    with _log_lock, open(_log_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def instrument_openai():
    """Patch the OpenAI SDK so every chat call (from any memory system) is logged with
    usage/latency/errors, gets EXTRA_BODY, and is retried on transport errors (max 2)."""
    import openai
    from openai.resources.chat.completions import Completions

    if getattr(Completions.create, "_memeval", False):
        return
    orig = Completions.create

    def create(self, *args, **kwargs):
        if EXTRA_BODY:
            kwargs["extra_body"] = {**EXTRA_BODY, **(kwargs.get("extra_body") or {})}
        last = None
        for attempt in range(3):
            t0 = time.time()
            try:
                r = orig(self, *args, **kwargs)
                u = getattr(r, "usage", None)
                content = r.choices[0].message.content if r.choices else None
                _log({"component": _component, "kind": "chat", "model": kwargs.get("model"),
                      "attempt": attempt, "latency": round(time.time() - t0, 3),
                      "prompt_tokens": getattr(u, "prompt_tokens", None),
                      "completion_tokens": getattr(u, "completion_tokens", None),
                      "empty": not content, "error": None})
                return r
            except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                    openai.InternalServerError) as e:
                last = e
                _log({"component": _component, "kind": "chat", "attempt": attempt,
                      "latency": round(time.time() - t0, 3), "error": f"{type(e).__name__}: {e}"[:300]})
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                _log({"component": _component, "kind": "chat", "attempt": attempt,
                      "latency": round(time.time() - t0, 3), "error": f"{type(e).__name__}: {e}"[:300]})
                raise
        raise last

    create._memeval = True
    Completions.create = create


# ---------------------------------------------------------------- embeddings

_EMB_DB = Path(os.environ.get("MEMEVAL_CACHE", ROOT / "cache" / "embeddings.sqlite"))
_emb_local = threading.local()


def _emb_conn():
    c = getattr(_emb_local, "conn", None)
    if c is None:
        _EMB_DB.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(_EMB_DB, timeout=60)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v TEXT)")
        _emb_local.conn = c
    return c


def _key(model, text):
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=3, timeout=120)
    return _client


def embed_many(texts, model=EMB_MODEL):
    """Embed texts with a persistent cache shared by all processes."""
    texts = [t if t.strip() else " " for t in texts]
    conn = _emb_conn()
    out = [None] * len(texts)
    missing = []
    for i, t in enumerate(texts):
        row = conn.execute("SELECT v FROM e WHERE k=?", (_key(model, t),)).fetchone()
        if row:
            out[i] = json.loads(row[0])
        else:
            missing.append(i)
    for s in range(0, len(missing), 64):
        idx = missing[s:s + 64]
        t0 = time.time()
        r = client().embeddings.create(model=model, input=[texts[i] for i in idx])
        _log({"component": _component, "kind": "embed", "n": len(idx), "latency": round(time.time() - t0, 3),
              "prompt_tokens": getattr(r.usage, "prompt_tokens", None)})
        for i, d in zip(idx, r.data):
            out[i] = d.embedding
            conn.execute("INSERT OR REPLACE INTO e VALUES (?,?)", (_key(model, texts[i]), json.dumps(d.embedding)))
        conn.commit()
    return out


def embed_one(text, model=EMB_MODEL):
    return embed_many([text], model)[0]


# ---------------------------------------------------------------- budget

BUDGET_TOKENS = int(os.environ.get("MEMEVAL_BUDGET", "2048"))


def n_tokens(text):
    # Deterministic approximation (tokenizer files are not downloadable here); same for all conditions.
    return len(text) // 4 + 1


def pack(items, budget=BUDGET_TOKENS):
    """Keep serialized items in rank order until the token budget is reached."""
    kept, used = [], 0
    for it in items:
        n = n_tokens(it) + 1
        if used + n > budget:
            break
        kept.append(it)
        used += n
    return kept, used
