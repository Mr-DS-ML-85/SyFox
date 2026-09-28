#!/usr/bin/env python3
"""banking77 manual benchmark — per-row error anatomy on the cal split.

Loads a SyFox fabric through the ctypes bridge, decides EVERY row with full
probabilities, and reports the structure the aggregate number hides:
  top-1/3/5 accuracy, gold-rank histogram, margin (p1-p2) distribution,
  confidence-when-wrong vs confidence-when-correct, worst confusion pairs,
  per-class accuracy (worst 15), and deferral anatomy.
Model/config selection stays on cal; hidden is only run for the final report.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402


def rank_of(gold: str, probs: dict) -> int:
    order = sorted(probs.items(), key=lambda kv: -kv[1])
    for i, (k, _) in enumerate(order, 1):
        if k == gold:
            return i
    return len(order) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/bank77_cal.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--dump", default="", help="write per-row JSONL here")
    ap.add_argument("--arbitrate", action="store_true",
                    help="product-of-experts: flat decide, then re-rank the "
                         "top-5 candidates by bare-anchor readout energy "
                         "(independent second view of the same field)")
    ap.add_argument("--hierarchical", action="store_true",
                    help="two-stage readout: stage 1 = choice over group "
                         "anchor bundles (member c## anchors), stage 2 = "
                         "re-decide within the winning group. Caller-side "
                         "orchestration only; same substrate, same physics.")
    args = ap.parse_args()

    eng = SyFoxEngine(args.model)
    rows = [json.loads(l) for l in open(args.eval)]
    if args.limit:
        rows = rows[: args.limit]

    hier = None
    if args.hierarchical:
        cat = json.load(open("data/bank77_categories.json"))
        lmap = cat["label_of"]                      # class name -> cNN
        groups: dict[str, list[str]] = {}
        for cls in cat["classes"]:
            groups.setdefault(cls.split("_")[0], []).append(lmap[cls])
        hier = groups
        print(f"hierarchical: {len(groups)} groups, sizes "
              f"{sorted((len(v) for v in groups.values()), reverse=True)[:6]}+")
        hier_rows = []
        for r in rows:
            crit = r["questions"]["intent"]["criteria"]
            gq = {"intent": {"type": "choice", "instructions": "",
                             "criteria": {g: ", ".join(ms) for g, ms in groups.items()}}}
            hier_rows.append(gq)

    recs, defers, hier_ok, hier_scored = [], 0, 0, 0
    for i, r in enumerate(rows):
        answers, usage, _ = eng.decide(r["state"], r["questions"],
                                       {"energy_norm": True})
        if args.arbitrate and not answers["intent"].get("deferred"):
            a0 = answers["intent"]
            top5 = [k for k, _ in sorted(a0["probabilities"].items(),
                                         key=lambda kv: -kv[1])[:5]]
            crit = r["questions"]["intent"]["criteria"]
            aq = {"intent": {"type": "choice", "instructions": "",
                             "criteria": {m: "" for m in top5}}}
            aa, _, _ = eng.decide(r["state"], aq, {"energy_norm": True})
            if not aa["intent"].get("deferred"):
                pa = aa["intent"]["probabilities"]
                pf = a0["probabilities"]
                combo = {m: pf.get(m, 0) * pa.get(m, 0) for m in top5}
                answers["intent"]["choice"] = max(combo, key=combo.get)
        if hier is not None:
            ga, _, _ = eng.decide(r["state"], hier_rows[i], {"energy_norm": True})
            g = ga["intent"].get("choice")
            members = hier.get(g, [])
            if len(members) == 1:
                sub_pred = members[0]
            elif members:
                crit = r["questions"]["intent"]["criteria"]
                sq = {"intent": {"type": "choice", "instructions": "",
                                 "criteria": {m: crit[m] for m in members}}}
                sa, _, _ = eng.decide(r["state"], sq, {"energy_norm": True})
                sub_pred = sa["intent"].get("choice")
            else:
                sub_pred = None
            gold_c = r["labels"]["intent"]
            hier_scored += 1
            hier_ok += (sub_pred == gold_c)
        a = answers["intent"]
        gold = r["labels"]["intent"]
        if a.get("deferred"):
            defers += 1
            recs.append({"i": i, "gold": gold, "deferred": True})
            continue
        probs = a["probabilities"]
        top = sorted(probs.items(), key=lambda kv: -kv[1])
        p1, p2 = top[0][1], (top[1][1] if len(top) > 1 else 0.0)
        recs.append({
            "i": i, "state": r["state"], "gold": gold,
            "pred": a.get("choice"), "correct": a.get("choice") == gold,
            "gold_p": round(probs.get(gold, 0.0), 5),
            "gold_rank": rank_of(gold, probs),
            "margin": round(p1 - p2, 5), "conf": a.get("confidence", 0),
            "top3": [k for k, _ in top[:3]],
        })

    scored = [r for r in recs if not r.get("deferred")]
    n = len(scored)
    if n == 0:
        print(f"ALL {len(recs)} rows deferred — nothing to score")
        return
    acc1 = sum(r["correct"] for r in scored) / n
    acc3 = sum(r["gold_rank"] <= 3 for r in scored) / n
    acc5 = sum(r["gold_rank"] <= 5 for r in scored) / n
    cw = [r["conf"] for r in scored if r["correct"]]
    ww = [r["conf"] for r in scored if not r["correct"]]
    mg_c = [r["margin"] for r in scored if r["correct"]]
    mg_w = [r["margin"] for r in scored if not r["correct"]]

    print(f"rows={len(rows)} deferred={defers} ({defers/len(rows):.1%}) scored={n}")
    if hier is not None:
        print(f"HIERARCHICAL top-1: {hier_ok}/{hier_scored} = {hier_ok/hier_scored:.3f}")
    print(f"top-1 {acc1:.3f}   top-3 {acc3:.3f}   top-5 {acc5:.3f}   (chance {1/77:.3f})")
    if cw:
        print(f"conf when correct {sum(cw)/len(cw):.3f}  vs wrong {sum(ww)/len(ww):.3f}")
    if mg_c:
        print(f"margin when correct {sum(mg_c)/len(mg_c):.4f}  vs wrong {sum(mg_w)/len(mg_w):.4f}")

    hist = Counter(min(r["gold_rank"], 78) for r in scored)
    bands = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 77)]
    print("gold-rank bands:", "  ".join(
        f"{a}-{b}:{sum(v for k, v in hist.items() if a <= k <= b)/n:.1%}" for a, b in bands))

    conf = Counter((r["gold"], r["pred"]) for r in scored if not r["correct"])
    print("worst confusion pairs:")
    for (g, p), c in conf.most_common(10):
        print(f"  {g:28s} -> {p:28s} {c}")

    per_class = {}
    for r in scored:
        t, c = per_class.get(r["gold"], (0, 0))
        per_class[r["gold"]] = (t + 1, c + (1 if r["correct"] else 0))
    worst = sorted(per_class.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0]))[:15]
    print("worst classes:", ", ".join(
        f"{k}:{c}/{t}" for k, (t, c) in worst))

    if args.dump:
        with open(args.dump, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"per-row dump -> {args.dump}")


if __name__ == "__main__":
    main()
