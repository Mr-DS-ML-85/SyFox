#!/usr/bin/env python3
"""Runtime-flag experiment: do the decide-side flags change WHAT the fabric
answers, and at what cost?

The model-xl zero-shot report (Snake A-E) measured: winner flips across flag
configs while settled_energy stayed at 83.928, and the Miller window reached
0.805 confidence on an ILLEGAL move. This tool re-measures the same five
configs on repo fabrics under the TRAINED schema (real heldout rows), so the
zero-shot observation can be separated from the trained-domain effect:

  A baseline        {}
  B energy-norm     {energy_norm}
  C energy+salience {energy_norm, salience_gating}
  D energy+miller   {energy_norm, miller_window}
  E all flags       {energy_norm, salience_gating, miller_window}

Reported per config: top-1 accuracy, mean margin (p1-p2), mean confidence,
deferrals, high-confidence-wrong rows (conf >= 0.5 while wrong).
Reported across configs: winner-change rate vs baseline, settled-energy
spread per row (max-min over the five configs).

Physics untouched: all three knobs are decide-time injection/source modes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402

CONFIGS = [
    ("A-baseline", {}),
    ("B-energy", {"energy_norm": True}),
    ("C-energy+salience", {"energy_norm": True, "salience_gating": True}),
    ("D-energy+miller", {"energy_norm": True, "miller_window": True}),
    ("E-all", {"energy_norm": True, "salience_gating": True, "miller_window": True}),
]


def run_fabric(model: str, eval_path: str, limit: int) -> dict:
    eng = SyFoxEngine(model)
    rows = [json.loads(l) for l in open(eval_path)]
    if limit:
        rows = rows[:limit]
    qkey = next(iter(rows[0]["questions"]))
    print(f"\n== {model}  eval={eval_path}  rows={len(rows)}  question='{qkey}'")

    per_cfg = {name: [] for name, _ in CONFIGS}
    for i, r in enumerate(rows):
        gold = r["labels"][qkey]
        for name, opts in CONFIGS:
            call = {"energy_norm": False, "salience_gating": False,
                    "miller_window": False, **opts}
            answers, usage, _ = eng.decide(r["state"], r["questions"], call)
            a = answers[qkey]
            if a.get("deferred"):
                per_cfg[name].append({"i": i, "gold": gold, "deferred": True,
                                      "energy": usage.get("settled_energy", 0.0)})
                continue
            probs = a.get("probabilities", {})
            top = sorted(probs.items(), key=lambda kv: -kv[1]) if probs else []
            p1 = top[0][1] if top else 0.0
            p2 = top[1][1] if len(top) > 1 else 0.0
            per_cfg[name].append({
                "i": i, "gold": gold, "deferred": False,
                "pred": a.get("choice"), "correct": a.get("choice") == gold,
                "conf": a.get("confidence", 0.0), "p1": p1, "margin": p1 - p2,
                "energy": usage.get("settled_energy", 0.0),
            })
        if (i + 1) % 300 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    print(f"{'config':22s} {'acc':>6s} {'margin':>7s} {'conf':>6s} "
          f"{'defer':>6s} {'hiConfWrong':>11s} {'flipVsA':>8s}")
    base = per_cfg["A-baseline"]
    for name, _ in CONFIGS:
        recs = per_cfg[name]
        scored = [r for r in recs if not r["deferred"]]
        n = len(scored)
        acc = sum(r["correct"] for r in scored) / max(1, n)
        mg = sum(r["margin"] for r in scored) / max(1, n)
        cf = sum(r["conf"] for r in scored) / max(1, n)
        df = sum(r["deferred"] for r in recs)
        hcw = sum(1 for r in scored if not r["correct"] and r["conf"] >= 0.5)
        if name == "A-baseline":
            flip = "-"
        else:
            bmap = {r["i"]: r for r in base}
            flip = sum(1 for r in scored
                       if r["i"] in bmap and not bmap[r["i"]]["deferred"]
                       and r["pred"] != bmap[r["i"]]["pred"]) / max(1, len(scored))
            flip = f"{flip:7.1%}"
        print(f"{name:22s} {acc:6.3f} {mg:7.4f} {cf:6.3f} "
              f"{df:5d} ({df/len(recs):4.1%}) {hcw:11d} {flip:>8s}")

    print("settled-energy spread across the 5 configs (same rows):")
    print("  A-vs-B (injection-dose gain only):")
    _spread_report(per_cfg, rows, ["A-baseline", "B-energy"])
    print("  B-vs-E (energy-norm constant, source modes varied — the model-xl"
          " Snake A-E condition):")
    _spread_report(per_cfg, rows, ["B-energy", "C-energy+salience",
                                   "D-energy+miller", "E-all"])
    return per_cfg


def _spread_report(per_cfg, rows, names) -> None:
    spreads, zero = [], 0
    for i in range(len(rows)):
        es = [per_cfg[name][i]["energy"] for name in names
              if not per_cfg[name][i]["deferred"]]
        if len(es) < 2:
            continue
        s = max(es) - min(es)
        spreads.append(s)
        zero += (s <= 0.001)
    if spreads:
        print(f"  mean {sum(spreads)/len(spreads):.4f}   max {max(spreads):.4f}"
              f"   rows with spread <= 0.001: {zero}/{len(spreads)}"
              f" ({zero/len(spreads):.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap rows per fabric")
    ap.add_argument("--dump", default="", help="write per-row records JSONL")
    args = ap.parse_args()

    pairs = [("model-tickets-big-latin", "data/big/tickets_en_hidden.jsonl"),
             ("model-sms-natural", "data/sms_hidden.jsonl")]
    dump = []
    for model, ev in pairs:
        if not (Path(model).exists() and Path(ev).exists()):
            print(f"SKIP {model}/{ev} (missing; fresh checkouts need make all)")
            continue
        per_cfg = run_fabric(model, ev, args.limit)
        for name, _ in CONFIGS:
            for r in per_cfg[name]:
                dump.append({"fabric": model, "config": name, **r})
    if args.dump and dump:
        with open(args.dump, "w") as f:
            for r in dump:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nper-row dump -> {args.dump}")


if __name__ == "__main__":
    main()
