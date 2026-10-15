"""Offline scoring for a memeval run directory (no network needed)."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import common

LABELS = ["history_supported", "present_cause_sufficient", "insufficient_evidence"]
SHORT = {"history_supported": "H", "present_cause_sufficient": "P", "insufficient_evidence": "I"}
CONDITIONS = ["current_only", "full_history", "oracle", "raw_bm25", "raw_dense", "amem", "memoryos", "mem0"]
MEMORY = ["amem", "memoryos", "mem0"]


def valid(p, sid):
    if not isinstance(p, dict) or set(p) != {"sample_id", "label", "cue_id", "evidence_turn_ids"}:
        return False
    if p["label"] not in LABELS or p.get("sample_id") != sid or not isinstance(p["evidence_turn_ids"], list):
        return False
    if p["label"] != "history_supported" and (p["cue_id"] != "none" or p["evidence_turn_ids"]):
        return False
    return True


def ev_scores(pred, gold_sets):
    best = (0.0, 0.0, 0.0, 0)
    P = set(pred)
    for g in gold_sets:
        G = set(g)
        if not P:
            continue
        tp = len(P & G)
        pr, rc = tp / len(P), tp / len(G)
        f1 = 2 * pr * rc / (pr + rc) if tp else 0.0
        em = int(P == G)
        if (f1, em) > (best[2], best[3]):
            best = (pr, rc, f1, em)
    return best


def label_metrics(rows):
    """rows: list of (gold_label, pred_label or None, i_subtype)."""
    n = len(rows)
    acc = sum(g == p for g, p, _ in rows) / n
    f1s = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p, _ in rows)
        fp = sum(g != c and p == c for g, p, _ in rows)
        fn = sum(g == c and p != c for g, p, _ in rows)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[SHORT[c]] = {"P": pr, "R": rc, "F1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    rate = lambda sel: (sum(p == "history_supported" for g, p, s in rows if sel(g, s)) /
                        max(1, sum(1 for g, p, s in rows if sel(g, s))))
    return {
        "accuracy": acc,
        "macro_f1": sum(v["F1"] for v in f1s.values()) / 3,
        "per_class": f1s,
        "P_to_H": rate(lambda g, s: g == "present_cause_sufficient"),
        "I_to_H": rate(lambda g, s: g == "insufficient_evidence"),
        "NR_to_H": rate(lambda g, s: s == "no_reaction"),
        "NH_to_H": rate(lambda g, s: s == "no_history_link"),
        "confusion": {SHORT[g]: {SHORT.get(p, "invalid") if p else "invalid": 0 for p in LABELS + [None]} for g in LABELS},
    }


def confusion(rows):
    m = {SHORT[g]: Counter() for g in LABELS}
    for g, p, _ in rows:
        m[SHORT[g]][SHORT.get(p, "invalid")] += 1
    return {k: dict(v) for k, v in m.items()}


def anchors():
    rs = json.loads((common.PILOT / "review_sheet.json").read_text())
    return {r["sample_id"]: r["anchor_id"] for r in rs}


def call_costs(run):
    out = defaultdict(lambda: Counter())
    for f in (run / "calls").glob("*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            comp = r["component"]
            key = comp.split(":")[1] if comp.startswith("ingest:") else comp
            c = out[key]
            if r["kind"] == "chat":
                c["chat_calls"] += 1
                c["chat_errors"] += bool(r.get("error"))
                c["chat_empty"] += bool(r.get("empty"))
                c["prompt_tokens"] += r.get("prompt_tokens") or 0
                c["completion_tokens"] += r.get("completion_tokens") or 0
                c["chat_latency_s"] += r.get("latency") or 0
            else:
                c["embed_texts"] += r.get("n", 0)
                c["embed_tokens"] += r.get("prompt_tokens") or 0
    walls = defaultdict(list)
    for f in (run / "snapshots").glob("*/*/_done.json"):
        d = json.loads(f.read_text())
        walls[d["system"]].append(d["wall_s"])
    res = {k: dict(v) for k, v in out.items()}
    for s, w in walls.items():
        res.setdefault(s, {})["ingest_wall_max_s"] = max(w)
        res[s]["ingest_wall_sum_s"] = round(sum(w), 1)
    return res


def main(run):
    run = Path(run)
    qs, ans = common.load_pilot()
    anc = anchors()
    results, per_item = {}, []
    for cond in CONDITIONS:
        d = run / "reader" / cond
        if not d.exists() or len(list(d.glob("*.json"))) < len(qs):
            continue
        rows, h_items, n_valid, n_json = [], [], 0, 0
        ctx_tokens, prov_cov = [], []
        for q in qs:
            sid = q["sample_id"]
            g = ans[sid]
            rec = json.loads((d / f"{sid}.json").read_text())
            ctx = json.loads((run / "contexts" / cond / f"{sid}.json").read_text())
            p = rec.get("parsed")
            n_json += p is not None
            ok = valid(p, sid)
            n_valid += ok
            pred = p["label"] if ok else None  # invalid outputs count as wrong
            rows.append((g["label"], pred, g.get("i_subtype")))
            units = ctx.get("units", [])
            ctx_tokens.append(ctx.get("context_tokens", 0) or 0)
            if units:
                prov_cov.append(sum(bool(u["source_turn_ids"]) for u in units) / len(units))
            ctx_ids = {i for u in units for i in u["source_turn_ids"]}
            item = {"sample_id": sid, "condition": cond, "anchor": anc.get(sid), "gold": SHORT[g["label"]],
                    "i_subtype": g.get("i_subtype"), "pred": SHORT.get(pred, "invalid"),
                    "context_tokens": ctx.get("context_tokens"), "n_units": len(units)}
            if g["label"] == "history_supported":
                sets = g.get("acceptable_evidence_sets") or [g["evidence_turn_ids"]]
                gold_ids = {i for s in sets for i in s}
                pe = p.get("evidence_turn_ids", []) if ok else []
                pr, rc, f1, em = ev_scores(pe, sets) if pred == "history_supported" else (0, 0, 0, 0)
                cue_ok = ok and pred == "history_supported" and p["cue_id"] == g["cue_id"]
                item.update({
                    "retrieved_any_gold": bool(ctx_ids & gold_ids),
                    "retrieved_full_gold_set": any(set(s) <= ctx_ids for s in sets),
                    "cue_ok": cue_ok, "ev_f1": f1, "ev_em": em,
                    "strict_joint": bool(cue_ok and em),
                    "cited_ids_not_in_context": [i for i in pe if i not in ctx_ids] if cond != "current_only" else pe,
                })
                h_items.append(item)
            per_item.append(item)
        m = label_metrics(rows)
        m["confusion"] = confusion(rows)
        nh = len(h_items)
        predH = [x for x in h_items if x["pred"] == "H"]
        m.update({
            "n": len(qs), "valid_json_rate": n_json / len(qs), "schema_ok_rate": n_valid / len(qs),
            "H_recall": len(predH) / nh,
            "cue_acc_conditional": sum(x["cue_ok"] for x in predH) / max(1, len(predH)),
            "cue_acc_e2e": sum(x["cue_ok"] for x in h_items) / nh,
            "evidence_em": sum(x["ev_em"] for x in h_items) / nh,
            "evidence_f1": sum(x["ev_f1"] for x in h_items) / nh,
            "strict_joint": sum(x["strict_joint"] for x in h_items) / nh,
            "retrieval_hit_any_gold": sum(x["retrieved_any_gold"] for x in h_items) / nh,
            "retrieval_hit_full_set": sum(x["retrieved_full_gold_set"] for x in h_items) / nh,
            "hallucinated_evidence_ids": sum(len(x["cited_ids_not_in_context"]) for x in h_items),
            "mean_context_tokens": sum(ctx_tokens) / len(ctx_tokens),
            "provenance_coverage": sum(prov_cov) / len(prov_cov) if prov_cov else None,
        })
        m["ci95"] = bootstrap(rows, [anc.get(q["sample_id"]) for q in qs])
        results[cond] = m

    # paired differences vs references, clustered by anchor
    diffs = {}
    for ref in ("raw_dense", "full_history", "current_only"):
        for cond in MEMORY:
            if cond in results and ref in results:
                diffs[f"{cond}-{ref}"] = paired_diff(per_item, cond, ref, qs, anc)
    funnel = funnels(run, per_item, qs, ans)
    out = {"results": results, "paired_macro_f1_diff": diffs, "funnel": funnel, "costs": call_costs(run)}
    (run / "scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    with open(run / "per_item.jsonl", "w") as f:
        for it in per_item:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print_table(out)
    return out


def bootstrap(rows, clusters, B=5000, seed=0):
    rnd = random.Random(seed)
    by = defaultdict(list)
    for r, c in zip(rows, clusters):
        by[c].append(r)
    keys = list(by)
    vals = []
    for _ in range(B):
        sample = [r for k in (rnd.choice(keys) for _ in keys) for r in by[k]]
        if len({g for g, _, _ in sample}) < 3:
            continue
        vals.append(label_metrics(sample)["macro_f1"])
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def paired_diff(per_item, a, b, qs, anc, B=5000, seed=0):
    idx = {(x["condition"], x["sample_id"]): x for x in per_item}
    gold = {x["sample_id"]: x for x in per_item if x["condition"] == a}
    inv = {"H": "history_supported", "P": "present_cause_sufficient", "I": "insufficient_evidence"}
    def mf1(sids, cond):
        rows = [(inv[idx[(cond, s)]["gold"]], inv.get(idx[(cond, s)]["pred"]), idx[(cond, s)]["i_subtype"]) for s in sids]
        return label_metrics(rows)["macro_f1"]
    sids = [q["sample_id"] for q in qs]
    point = mf1(sids, a) - mf1(sids, b)
    by = defaultdict(list)
    for s in sids:
        by[anc.get(s)].append(s)
    keys, rnd, vals = list(by), random.Random(seed), []
    for _ in range(B):
        ss = [s for k in (rnd.choice(keys) for _ in keys) for s in by[k]]
        if len({gold[s]["gold"] for s in ss}) < 3:
            continue
        vals.append(mf1(ss, a) - mf1(ss, b))
    vals.sort()
    disc = Counter()
    for s in sids:
        ca = idx[(a, s)]["pred"] == idx[(a, s)]["gold"]
        cb = idx[(b, s)]["pred"] == idx[(b, s)]["gold"]
        disc[f"{int(ca)}{int(cb)}"] += 1
    return {"diff": point, "ci95": [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]],
            "discordant_a_right_b_wrong": disc["10"], "discordant_a_wrong_b_right": disc["01"]}


def funnels(run, per_item, qs, ans):
    """H-item failure funnel per memory system:
    1 not stored (no stored unit cites a gold turn) -> 2 stored not retrieved -> 4 retrieved, label wrong
    -> 5 label H, cue wrong -> 6 cue ok, evidence not exact -> 7 all correct."""
    import adapters
    out = {}
    for cond in MEMORY:
        items = [x for x in per_item if x["condition"] == cond and x["gold"] == "H"]
        if not items:
            continue
        stored_cache = {}
        f = Counter()
        detail = []
        for x in items:
            conv = next(q["conversation_id"] for q in qs if q["sample_id"] == x["sample_id"])
            if conv not in stored_cache:
                snap = run / "snapshots" / cond / conv
                try:
                    stored_cache[conv] = {i for u in adapters.MEMORY_SYSTEMS[cond]().dump(snap) for i in u["source_turn_ids"]}
                except Exception as e:
                    stored_cache[conv] = None
            g = ans[x["sample_id"]]
            gold_ids = {i for s in (g.get("acceptable_evidence_sets") or [g["evidence_turn_ids"]]) for i in s}
            stored = stored_cache[conv]
            if stored is None:
                stage = "0_snapshot_unavailable"
            elif not (stored & gold_ids):
                stage = "1_not_stored"
            elif not x["retrieved_any_gold"]:
                stage = "2_stored_not_retrieved"
            elif x["pred"] != "H":
                stage = "4_retrieved_label_wrong"
            elif not x["cue_ok"]:
                stage = "5_cue_wrong"
            elif not x["ev_em"]:
                stage = "6_evidence_not_exact"
            else:
                stage = "7_all_correct"
            f[stage] += 1
            detail.append({"sample_id": x["sample_id"], "stage": stage, "pred": x["pred"]})
        out[cond] = {"counts": dict(sorted(f.items())), "items": detail}
    return out


def print_table(out):
    hdr = f"{'condition':14} {'Acc':>5} {'MacF1':>6} {'CI95':>13} {'H-R':>5} {'P->H':>5} {'NR->H':>6} {'NH->H':>6} {'EvF1':>5} {'Joint':>6} {'Hit':>5} {'ctxTok':>7} {'valid':>6}"
    print(hdr)
    for c, m in out["results"].items():
        ci = m["ci95"]
        print(f"{c:14} {m['accuracy']:5.2f} {m['macro_f1']:6.3f} {('[%.2f,%.2f]' % tuple(ci)) if ci else '':>13} "
              f"{m['H_recall']:5.2f} {m['P_to_H']:5.2f} {m['NR_to_H']:6.2f} {m['NH_to_H']:6.2f} {m['evidence_f1']:5.2f} "
              f"{m['strict_joint']:6.2f} {m['retrieval_hit_any_gold']:5.2f} {m['mean_context_tokens']:7.0f} {m['schema_ok_rate']:6.2f}")
    for k, v in out["paired_macro_f1_diff"].items():
        print(f"{k:28} diff={v['diff']:+.3f} CI95=[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] "
              f"(a>b {v['discordant_a_right_b_wrong']}, a<b {v['discordant_a_wrong_b_right']})")
    for c, f in out["funnel"].items():
        print(c, f["counts"])


if __name__ == "__main__":
    import sys
    main(common.ROOT / "runs" / sys.argv[1])
