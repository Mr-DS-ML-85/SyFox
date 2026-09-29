#!/usr/bin/env python3
"""b77_margin_sweep.py — defer-threshold sweep + per-intent threshold
derivation, OFFLINE from one bench per-row dump (v3.9.1).

The dump (bench --dump-perrow) carries, per choice question, the RAW
pre-temperature top-2 energy gap ("raw_margin") and the prediction. The raw
gap is the physical quantity a threshold policy can act on: the shipped
bank77 fabric's fitted calibration temperature (0.0019, measured) saturates
the post-temperature probability gap at ~1, so a probability-margin sweep
would be inert (this is itself a documented finding).

Discipline:
  * derive on CAL only; score the chosen policy on HIDDEN with
    b77_apply_policy.py (never tune and score on the same split).
  * thresholds are keyed by the PREDICTED intent (at inference the policy
    sees pred, never gold).

Modes:
  --mode global      sweep one global threshold over the grid.
  --mode per-intent  per-intent threshold = argmax of per-intent honest total
                     (correct / class total) on the sweep split; intents with
                     fewer than --min-rows keep the global best.
Outputs JSON with the sweep table and (per-intent mode) the threshold map.
"""
import argparse
import json
from pathlib import Path

GRID = [round(0.01 * i, 2) for i in range(0, 21)]     # 0.00 .. 0.20


def honest(rows, thr_of):
    """rows: list of dump records. thr_of(pred) -> threshold.
    Policy: answer iff raw_margin >= thr(pred); correct iff pred == gold.
    Deferred wrong rows cost nothing; honest total = correct / n."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "answered": 0, "correct": 0, "acc_answered": 0.0,
                "honest_total": 0.0, "defer_rate": 0.0}
    answered = [r for r in rows if r["raw_margin"] >= thr_of(r["pred"])]
    correct = sum(1 for r in answered if r["pred"] == r["gold"])
    return {"n": n, "answered": len(answered), "correct": correct,
            "acc_answered": round(correct / len(answered), 4) if answered else 0.0,
            "honest_total": round(correct / n, 4),
            "defer_rate": round(1 - len(answered) / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--mode", choices=["global", "per-intent"], default="global")
    ap.add_argument("--min-rows", type=int, default=8,
                    help="per-intent rows below this keep the global best")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]

    if args.mode == "global":
        sweep = []
        for t in GRID:
            h = honest(rows, lambda pred, t=t: t)
            sweep.append({"threshold": t, **h})
        best = max(sweep, key=lambda s: (s["honest_total"], s["answered"], -s["threshold"]))
        doc = {"dump": args.dump, "mode": "global", "sweep": sweep, "best": best}
        print(f"{'thr':>5s} {'answered':>9s} {'correct':>8s} {'acc_ans':>8s} {'honest':>7s} {'defer':>6s}")
        for s in sweep:
            mark = "  <- best" if s is best else ""
            print(f"{s['threshold']:5.2f} {s['answered']:9d} {s['correct']:8d} "
                  f"{s['acc_answered']:8.4f} {s['honest_total']:7.4f} {s['defer_rate']:6.3f}{mark}")
    else:
        # global best first (fallback for thin intents)
        best_g, best_h = None, -1.0
        for t in GRID:
            h = honest(rows, lambda pred, t=t: t)
            if h["honest_total"] > best_h or (h["honest_total"] == best_h and best_g is not None and t < best_g):
                best_g, best_h = t, h["honest_total"]
        by_pred = {}
        for r in rows:
            by_pred.setdefault(r["pred"], []).append(r)
        per, n_custom = {}, 0
        for pred, rs in sorted(by_pred.items()):
            if len(rs) < args.min_rows:
                per[pred] = {"threshold": best_g, "rows": len(rs),
                             "honest_total": honest(rs, lambda _, t=best_g: t)["honest_total"],
                             "source": "global_fallback"}
                continue
            best_t, best_ht, best_aa = best_g, -1.0, -1.0
            for t in GRID:
                h = honest(rs, lambda _, t=t: t)
                if (h["honest_total"], h["acc_answered"], -t) > (best_ht, best_aa, -best_t):
                    best_t, best_ht, best_aa = t, h["honest_total"], h["acc_answered"]
            if best_t != best_g:
                n_custom += 1
            per[pred] = {"threshold": best_t, "rows": len(rs),
                         "honest_total": round(best_ht, 4),
                         "acc_answered": round(best_aa, 4),
                         "source": "per_intent"}
        overall = honest(rows, lambda pred: per[pred]["threshold"])
        doc = {"dump": args.dump, "mode": "per-intent", "min_rows": args.min_rows,
               "global_best": best_g, "thresholds": per,
               "policy_overall": overall, "custom_intents": n_custom}
        print(f"global best threshold {best_g} (honest {best_h}); "
              f"{n_custom} intents got a custom threshold "
              f"(min_rows {args.min_rows})")
        print(f"policy overall: answered {overall['answered']}/{overall['n']} "
              f"correct {overall['correct']}  acc_ans {overall['acc_answered']}  "
              f"honest_total {overall['honest_total']}  defer {overall['defer_rate']}")

    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"json -> {args.out}")


if __name__ == "__main__":
    main()
