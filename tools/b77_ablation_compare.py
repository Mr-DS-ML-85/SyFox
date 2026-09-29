#!/usr/bin/env python3
"""b77_ablation_compare.py — per-intent A/B of two bench per-row dumps
(v3.9.1, e.g. semantics ON vs --no-semantics) sharing the same eval file.

Joins on the dump "row" field (the eval row index), reports per-intent
accuracy and honest flips (A wrong -> B correct and vice versa), plus the
confusion-pair migration between the two runs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-a", required=True, help="config A dump")
    ap.add_argument("--dump-b", required=True, help="config B dump")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    args = ap.parse_args()

    A = {json.loads(l)["row"]: json.loads(l) for l in open(args.dump_a) if l.strip()}
    B = {json.loads(l)["row"]: json.loads(l) for l in open(args.dump_b) if l.strip()}
    common = sorted(set(A) & set(B))
    if not common:
        print("no common rows — same eval file?", file=sys.stderr)
        raise SystemExit(2)

    acc_a = sum(A[i]["pred"] == A[i]["gold"] for i in common) / len(common)
    acc_b = sum(B[i]["pred"] == B[i]["gold"] for i in common) / len(common)
    print(f"rows {len(common)}  {args.name_a} acc {acc_a:.4f}  "
          f"{args.name_b} acc {acc_b:.4f}  delta {acc_b - acc_a:+.4f}")

    a_ok = {i: A[i]["pred"] == A[i]["gold"] for i in common}
    b_ok = {i: B[i]["pred"] == B[i]["gold"] for i in common}
    fixed = [i for i in common if not a_ok[i] and b_ok[i]]
    broken = [i for i in common if a_ok[i] and not b_ok[i]]
    print(f"flips: {args.name_a} wrong -> {args.name_b} correct: {len(fixed)}   "
          f"{args.name_a} correct -> {args.name_b} wrong: {len(broken)}")

    cats = json.loads((ROOT / "data" / "bank77_categories.json").read_text())
    natural = {v: k for k, v in cats["label_of"].items()}
    by_gold = {}
    for i in common:
        by_gold.setdefault(A[i]["gold"], []).append(i)
    print(f"\n{'gold intent':46s} {'n':>4s} {'A acc':>6s} {'B acc':>6s} {'delta':>7s}")
    rows = []
    for g, idxs in sorted(by_gold.items()):
        aa = sum(a_ok[i] for i in idxs) / len(idxs)
        bb = sum(b_ok[i] for i in idxs) / len(idxs)
        rows.append((bb - aa, g, len(idxs), aa, bb))
    rows.sort()
    for d, g, n, aa, bb in rows:
        print(f"{g+'='+natural.get(g,'?'):46s} {n:4d} {aa:6.3f} {bb:6.3f} {d:+7.3f}")

    mig = Counter((A[i]["pred"], B[i]["pred"]) for i in common
                  if A[i]["pred"] != B[i]["pred"])
    print("\nprediction migrations (A pred -> B pred, n):")
    for (pa, pb), n in mig.most_common(10):
        print(f"  {pa}={natural.get(pa,'?'):44s} -> {pb}={natural.get(pb,'?'):44s} {n}")


if __name__ == "__main__":
    main()
