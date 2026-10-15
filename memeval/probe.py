"""Probe the API before a run: model availability, latency, thinking output, JSON modes, embeddings.
Usage: AIHUBMIX_API_KEY=... python probe.py"""
import json
import time

import common
from common import LLM_MODEL, EMB_MODEL, client


def chat(extra=None, **kw):
    t0 = time.time()
    try:
        r = client().chat.completions.create(
            model=LLM_MODEL, max_tokens=300, temperature=0,
            messages=[{"role": "user", "content": 'Return the JSON object {"keywords": ["a"], "ok": true} and nothing else.'}],
            **({"extra_body": extra} if extra else {}), **kw)
        m = r.choices[0].message
        return {"latency": round(time.time() - t0, 2), "content": (m.content or "")[:200],
                "reasoning_chars": len(getattr(m, "reasoning_content", None) or ""),
                "usage": r.usage.model_dump() if r.usage else None}
    except Exception as e:
        return {"latency": round(time.time() - t0, 2), "error": repr(e)[:300]}


if __name__ == "__main__":
    out = {"model": LLM_MODEL}
    out["plain"] = chat()
    out["enable_thinking_false"] = chat({"enable_thinking": False})
    out["json_object"] = chat(response_format={"type": "json_object"})
    out["json_schema_strict"] = chat(response_format={"type": "json_schema", "json_schema": {
        "name": "r", "strict": True, "schema": {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}, "ok": {"type": "boolean"}},
                                               "required": ["keywords", "ok"], "additionalProperties": False}}})
    t0 = time.time()
    try:
        e = client().embeddings.create(model=EMB_MODEL, input=["hello", "world"])
        out["embedding"] = {"dims": len(e.data[0].embedding), "latency": round(time.time() - t0, 2)}
    except Exception as ex:
        out["embedding"] = {"error": repr(ex)[:300]}
    # parallel latency at the concurrency we plan to use
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(20) as ex:
        lat = [r.get("latency") for r in ex.map(lambda _: chat(json.loads(__import__("os").environ.get("MEMEVAL_EXTRA_BODY", "{}")) or None), range(20))]
    out["parallel20"] = {"wall": round(time.time() - t0, 2), "latencies": lat}
    print(json.dumps(out, indent=1, ensure_ascii=False))
