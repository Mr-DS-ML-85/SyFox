#!/usr/bin/env python3
"""b77_apply_policy.py — score a derived defer-threshold policy on a FRESH
dump (the hidden split), never the split the policy was derived on (v3.9.1).

Usage:
  syfox bench --model model-b77-sem --eval data/bank77_hidden.jsonl ... \
      --dump-perrow /tmp/hidden_dump.jsonl   # scoring pass (read-only)
  python3 tools/b77_margin_sweep.py --dump cal_dump --mode per-intent \
      --out cal_thresholds.json              # derivation (CAL only)
  python3 tools/b77_apply_policy.py --dump hidden_dump \
      --policy cal_thresholds.json           # THIS TOOL

The policy maps the PREDICTED intent (never gold) to a raw-margin threshold:
answer iff raw_margin >= thr(pred); rows below defer. Reports the overall
honest total and the per-intent table with natural names.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tag", default="", help="label for the report line")
    args = ap.parse_args()

    pol = json.loads(Path(args.policy).read_text())
    thr_map = pol["thresholds"] if "thresholds" in pol else pol
    thr_of = lambda pred: float(thr_map.get(pred, {}).get("threshold", 0.0))

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    answered = [r for r in rows if r["raw_margin"] >= thr_of(r["pred"])]
    correct = sum(1 for r in answered if r["pred"] == r["gold"])
    n = len(rows)
    overall = {
        "n": n, "answered": len(answered), "correct": correct,
        "acc_answered": round(correct / len(answered), 4) if answered else 0.0,
        "honest_total": round(correct / n, 4) if n else 0.0,
        "defer_rate": round(1 - len(answered) / n, 4) if n else 0.0,
    }

    cats = json.loads((ROOT / "data" / "bank77_categories.json").read_text())
    natural = {v: k for k, v in cats["label_of"].items()}
    by_gold = {}
    for r in rows:
        by_gold.setdefault(r["gold"], []).append(r)
    print(f"policy {args.tag or args.policy}  on {args.dump}")
    print(f"overall: answered {overall['answered']}/{n}  correct {overall['correct']}  "
          f"acc_answered {overall['acc_answered']}  honest_total {overall['honest_total']}  "
          f"defer {overall['defer_rate']}")
    print(f"{'gold intent':46s} {'n':>4s} {'ans':>4s} {'correct':>7s} {'honest':>6s} {'acc_ans':>7s}")
    out_rows = []
    for g, rs in sorted(by_gold.items()):
        ans = [r for r in rs if r["raw_margin"] >= thr_of(r["pred"])]
        c = sum(1 for r in ans if r["pred"] == g)
        ht = c / len(rs)
        aa = c / len(ans) if ans else 0.0
        out_rows.append((ht, g, len(rs), len(ans), c, aa))
    out_rows.sort(key=lambda t: (t[0], t[1]))
    for ht, g, ntot, nans, c, aa in out_rows:
        print(f"{g+'='+natural.get(g,'?'):46s} {ntot:4d} {nans:4d} {c:7d} {ht:6.3f} {aa:7.3f}")
    print("json:", json.dumps({"tag": args.tag, "overall": overall}))


if __name__ == "__main__":
    main()
