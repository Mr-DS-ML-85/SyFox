#!/usr/bin/env python3
"""b77_freq_cut.py — the class-frequency cut (v3.9.1 Task 0.5).

Per-intent accuracy vs per-intent TRAIN ROW COUNT. High correlation would
point at a data problem (thin classes fail); low correlation points at a
model problem (confusable classes fail at any count — banking77 is nearly
balanced, so this is the expected outcome; MEASURE, do not assume).

Counts come from the train split file (gold labels), accuracy from a bench
per-row dump. Reports Pearson r and Spearman rho with permutation-free
t-statistics, plus the ranked per-intent table.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0:
        return 0.0
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, t


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    return pearson(ranks(x), ranks(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="bench per-row dump")
    ap.add_argument("--train", default="data/bank77_train.jsonl")
    ap.add_argument("--label-key", default="labels")
    args = ap.parse_args()

    counts = Counter()
    with open(ROOT / args.train) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lab = r.get(args.label_key, {})
            if isinstance(lab, dict) and lab:
                counts[next(iter(lab.values()))] += 1

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    by_gold = {}
    for r in rows:
        by_gold.setdefault(r["gold"], []).append(r)

    xs, ys, table = [], [], []
    for g, rs in sorted(by_gold.items()):
        n_train = counts.get(g, 0)
        n_tot = len(rs)
        correct = sum(1 for r in rs if r["pred"] == g)
        acc = correct / n_tot
        xs.append(float(n_train))
        ys.append(acc)
        table.append((g, n_train, n_tot, correct, acc))

    r, t_r = pearson(xs, ys)
    rho, t_rho = spearman(xs, ys)
    print(f"intents scored: {len(xs)}   train counts: min {min(xs):.0f} "
          f"max {max(xs):.0f} mean {sum(xs)/len(xs):.1f}")
    print(f"Pearson r  (accuracy vs train count): {r:+.3f}  (t={t_r:+.2f}, "
          f"df={len(xs)-2})")
    print(f"Spearman rho                        : {rho:+.3f}  (t={t_rho:+.2f})")
    print(f"accuracy spread: min {min(ys):.3f} max {max(ys):.3f}  "
          f"train-count spread: {max(xs)-min(xs):.0f} rows")
    verdict = ("DATA problem (thin classes fail)" if abs(r) > 0.5
               else "MODEL problem (accuracy does not track class frequency)")
    print(f"verdict (|r|>0.5 cut): {verdict}")
    print(f"\n{'intent':6s} {'train':>6s} {'n':>4s} {'correct':>7s} {'acc':>6s}")
    for g, ntr, ntot, c, acc in sorted(table, key=lambda t: t[4]):
        print(f"{g:6s} {ntr:6d} {ntot:4d} {c:7d} {acc:6.3f}")


if __name__ == "__main__":
    main()
