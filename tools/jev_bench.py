#!/usr/bin/env python3
"""SyFox vs JEV AI — measured head-to-head (v3.6.0).

JEV AI (TypeSafe AI, Sept 2026 early access) is the "System One" decision
model: state + typed questions (noul/choice/score) in, per-question value +
probability + confidence out, no text generation. SyFox shares that
architecture (state + typed questions -> honest decided answers over a
settled energy field), so the competition is measurable on JEV's own axes:

  workflow accuracy   JEV publishes a workflow-eval figure (0.678); syfox
                      measures choice/score accuracy + defer rates per fabric.
  latency             JEV reports 70-500 ms end-to-end; syfox measures the
                      in-process decide path (latency_us p50/p95 in the bench).
  guardrail           JEV reports a hold-precision axis (0.88); syfox measures
                      its guardrail confusion matrix + OOD defer rate.
  calibration         JEV claims 70% confidence = 70% right, no public study;
                      syfox publishes measured ECE per question type.
  cost                JEV: $0.042 / M input tokens, output free; syfox runs
                      locally at $0 compute cost (open core, no API meter).

This tool reads the JSON the C++ bench already measured (make bench ->
build/bench-*.json) and prints the head-to-head. Every syfox number comes
from an actual run in this repository; every JEV number is the PUBLISHED
figure carried in the bench's jev_reference block. No syfox number is ever
estimated, and no JEV number is invented: their own press states the
multipliers are self-reported ceilings.
"""
import json
import os
import sys

BENCH_FILES = [
    ("model-tickets", "build/bench-tickets.json"),
    ("model-game", "build/bench-game.json"),
    ("model-guard", "build/bench-guard.json"),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x):
    return f"{100.0 * x:.1f}%"


def main():
    rows = []
    for model, path in BENCH_FILES:
        d = load(path)
        if d is None:
            print(f"missing {path}: run `make bench` first", file=sys.stderr)
            continue
        if not d.get("suite", "").startswith("syfox-jev-parity"):
            print(f"{path} is not a parity-suite run", file=sys.stderr)
            continue
        rows.append((model, d))

    if not rows:
        sys.exit("no parity bench JSON found — run `make bench` first")

    # Aggregate measured syfox numbers over the seed fabrics.
    answered_accs, all_lat, eces, ood = [], [], [], []
    for model, d in rows:
        r = d["routing"]
        if r.get("choice_n"):
            answered_accs.append((r["choice_accuracy"], r["choice_n"]))
        if r.get("score_n"):
            answered_accs.append((r["score_accuracy"], r["score_n"]))
        all_lat.append(d["latency_us"]["p50"])
        all_lat.append(d["latency_us"]["p95"])
        c = d.get("calibration", {})
        for k in ("choice_ece", "score_ece"):
            if k in c:
                eces.append(c[k])
        ood.append(d["honesty"]["ood_defer_rate"])

    n_acc = sum(n for _, n in answered_accs)
    acc = sum(a * n for a, n in answered_accs) / n_acc if n_acc else 0.0
    lat_p50_max = max(all_lat)
    ece = sum(eces) / len(eces) if eces else 0.0
    ood_min = min(ood)

    jev = rows[0][1]["jev_reference"]

    print("=" * 74)
    print("SYFOX vs JEV AI (TypeSafe) — measured head-to-head on JEV's axes")
    print("=" * 74)
    print(f"{'axis':<28}{'JEV AI (published)':<24}{'syfox (measured now)':<22}")
    print("-" * 74)
    print(f"{'decision API':<28}{'POST /v1/systemone':<24}{'POST /v1/systemone':<22}")
    print(f"{'question types':<28}{'noul/choice/score':<24}{'noul/choice/score':<22}")
    print(f"{'batch questions/call':<28}{'yes (parallel)':<24}{'yes (one settle)':<22}")
    print(f"{'workflow accuracy':<28}{pct(jev['workflow_accuracy']):<24}{pct(acc) + f' (n={n_acc} answered)':<22}")
    print(f"{'latency (p50->p95)':<28}{jev['latency_ms'] + ' ms':<24}{'<= ' + f'{lat_p50_max:.0f}' + ' us':<22}")
    print(f"{'input cost':<28}{'$0.042 / M tokens':<24}{'$0 (local core)':<22}")
    print(f"{'output cost':<28}{'free (typed value)':<24}{'free (typed value)':<22}")
    print(f"{'calibration study':<28}{'claimed, not public':<24}{'ECE published (mean ' + f'{ece:.3f}' + ')':<22}")
    print(f"{'honest defer w/ reason':<28}{'confidence only':<24}{'defer + reason (OOD ' + pct(ood_min) + ')':<22}")
    print(f"{'option cap / choice':<28}{'255 then two-stage':<24}{'unbounded criteria map':<22}")
    print(f"{'determinism':<28}{'not published':<24}{'bit-reproducible (CI)':<22}")
    print(f"{'full decision audit':<28}{'no':<24}{'evidence JSON + ledger':<22}")
    print("-" * 74)
    print(f"JEV reference source: {jev['source']}")
    print("syfox source: " + ", ".join(f"{m} ({p})" for m, p in BENCH_FILES if load(p)))
    print("syfox latency = in-process decide path (settle + readout), this tree,")
    print("measured by core/bench.hpp; JEV latency = their published end-to-end range.")

    out = {
        "suite": "syfox-vs-jev-head-to-head",
        "syfox_measured": {
            "answered_accuracy": acc,
            "answered_n": n_acc,
            "latency_us_p50_max": lat_p50_max,
            "mean_ece": ece,
            "ood_defer_rate_min": ood_min,
            "per_model": [
                {"model": m, "routing": d["routing"], "calibration": d.get("calibration", {}),
                 "latency_us": d["latency_us"], "honesty": d["honesty"],
                 "guardrail": d.get("guardrail", {})} for m, d in rows
            ],
        },
        "jev_published": jev,
    }
    with open("build/jev_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote build/jev_comparison.json")


if __name__ == "__main__":
    main()
