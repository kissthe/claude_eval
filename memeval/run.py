"""Pilot30 memory-system comparison runner.

  python run.py ingest   --run R --system amem --conv conv-30
  python run.py contexts --run R            # build every condition's reader context
  python run.py read     --run R [--workers 8]
  python run.py score    --run R
"""
import argparse
import json
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
from common import load_conversation, load_pilot, pack, turn_line, BUDGET_TOKENS, LLM_MODEL

RUNS = common.ROOT / "runs"
PROMPT = (common.ROOT / "prompts" / "prompt_mem_v1.txt")
CONDITIONS = ["current_only", "full_history", "oracle", "raw_bm25", "raw_dense", "amem", "memoryos", "mem0"]


def targets():
    qs, _ = load_pilot()
    return {q["conversation_id"]: q["target_speaker"] for q in qs}


def query_text(q):
    return f"{q['target_speaker']}: {q['current_input']['text']}"


# ------------------------------------------------------------------ ingest

def cmd_ingest(a):
    import adapters
    run = RUNS / a.run
    out = run / "snapshots" / a.system / a.conv
    if (out / "_done.json").exists():
        print("already done", out)
        return
    common.set_call_log(run / "calls" / f"ingest_{a.system}_{a.conv}.jsonl", f"ingest:{a.system}:{a.conv}")
    common.instrument_openai()
    random.seed(a.seed)
    t0 = time.time()
    stats = adapters.MEMORY_SYSTEMS[a.system]().ingest(a.conv, targets()[a.conv], out)
    stats.update({"system": a.system, "conv": a.conv, "seed": a.seed, "wall_s": round(time.time() - t0, 1)})
    (out / "_done.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, default=str))


# ------------------------------------------------------------------ contexts

def build_context(cond, q, ans, cache):
    conv, target = q["conversation_id"], q["target_speaker"]
    if cond == "current_only":
        return [], 0
    if cond == "full_history":
        _, _, turns = load_conversation(conv)
        units = [{"text": turn_line(t), "source_turn_ids": [t["turn_id"]], "kind": "turn"} for t in turns]
        return units, None  # no budget: full history is the reference condition
    if cond == "oracle":
        _, _, turns = load_conversation(conv)
        gold = set(ans["evidence_turn_ids"])
        units = [{"text": turn_line(t), "source_turn_ids": [t["turn_id"]], "kind": "turn"} for t in turns if t["turn_id"] in gold]
        return units, None
    import adapters
    if cond in adapters.RAW:
        units = adapters.RAW[cond]().retrieve_conv(conv, query_text(q), target)
    else:
        snap = RUNS / cache["run"] / "snapshots" / cond / conv
        units = adapters.MEMORY_SYSTEMS[cond]().retrieve(snap, query_text(q), target)
    return units, BUDGET_TOKENS


def cmd_contexts(a):
    run = RUNS / a.run
    common.set_call_log(run / "calls" / "contexts.jsonl", "contexts")
    common.instrument_openai()
    qs, ans = load_pilot()
    conds = a.conditions.split(",") if a.conditions else CONDITIONS
    for cond in conds:
        d = run / "contexts" / cond
        d.mkdir(parents=True, exist_ok=True)
        for q in qs:
            f = d / f"{q['sample_id']}.json"
            if f.exists():
                continue
            t0 = time.time()
            try:
                units, budget = build_context(cond, q, ans[q["sample_id"]], {"run": a.run})
                n_candidates = len(units)
                if budget is not None:
                    texts, used = pack([u["text"] for u in units], budget)
                    units = units[:len(texts)]
                rec = {"sample_id": q["sample_id"], "condition": cond, "budget": budget,
                       "n_candidates": n_candidates, "units": units,
                       "context_tokens": sum(common.n_tokens(u["text"]) for u in units),
                       "retrieval_s": round(time.time() - t0, 2), "error": None}
            except Exception as e:
                traceback.print_exc()
                rec = {"sample_id": q["sample_id"], "condition": cond, "units": [], "error": repr(e)[:500]}
            f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print("contexts done:", cond, flush=True)


# ------------------------------------------------------------------ read

def reader_message(q, ctx):
    hist = "\n\n".join(u["text"] for u in ctx["units"]) if ctx["units"] else "(none provided)"
    cue = "\n".join(f"- {c['cue_id']}: {c['name']}" for c in q["current_input"]["cue_options"])
    return (f"sample_id: {q['sample_id']}\ntarget_speaker: {q['target_speaker']}\n\n"
            f"<earlier_history>\n{hist}\n</earlier_history>\n\n"
            f"<current_input>\n{q['current_input']['text']}\n</current_input>\n\n"
            f"cue_options:\n{cue}")


def parse(text):
    import re
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def cmd_read(a):
    run = RUNS / a.run
    common.set_call_log(run / "calls" / "reader.jsonl", "reader")
    common.instrument_openai()
    qs, _ = load_pilot()
    system = PROMPT.read_text()
    conds = a.conditions.split(",") if a.conditions else CONDITIONS
    jobs = []
    for cond in conds:
        for q in qs:
            out = run / "reader" / cond / f"{q['sample_id']}.json"
            if not out.exists():
                jobs.append((cond, q, out))

    def go(job):
        cond, q, out = job
        ctx = json.loads((run / "contexts" / cond / f"{q['sample_id']}.json").read_text())
        msg = reader_message(q, ctx)
        rec = {"sample_id": q["sample_id"], "condition": cond, "model": LLM_MODEL, "input_chars": len(system) + len(msg)}
        t0 = time.time()
        try:
            r = common.client().chat.completions.create(
                model=LLM_MODEL, temperature=0, max_tokens=a.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}],
                **({"response_format": {"type": "json_object"}} if a.json_mode else {}))
            text = r.choices[0].message.content
            rec.update({"raw": text, "usage": r.usage.model_dump() if r.usage else None,
                        "finish_reason": r.choices[0].finish_reason, "parsed": parse(text), "error": None})
        except Exception as e:
            rec.update({"raw": None, "parsed": None, "error": repr(e)[:500]})
        rec["latency_s"] = round(time.time() - t0, 2)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(go, jobs)):
            if (i + 1) % 20 == 0:
                print(f"read {i + 1}/{len(jobs)}", flush=True)
    print("read done", len(jobs))


def cmd_score(a):
    import score
    score.main(RUNS / a.run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("--run", required=True); s.add_argument("--system", required=True)
    s.add_argument("--conv", required=True); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("contexts"); s.add_argument("--run", required=True); s.add_argument("--conditions")
    s = sub.add_parser("read"); s.add_argument("--run", required=True); s.add_argument("--conditions")
    s.add_argument("--workers", type=int, default=8); s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--json-mode", action="store_true")
    s = sub.add_parser("score"); s.add_argument("--run", required=True)
    a = p.parse_args()
    {"ingest": cmd_ingest, "contexts": cmd_contexts, "read": cmd_read, "score": cmd_score}[a.cmd](a)
