#!/usr/bin/env python3
"""BUG #2 diagnosis harness: confidence per ablation config on real rows.

Loads a SyFox fabric ONCE through the ctypes bridge, decides every eval row
under 4 configs (default / --no-retrieval / --no-semantics / both-off) and
reports per config: n, accuracy, mean/max/min confidence (answered rows),
deferred count, plus mean confidence among CORRECT vs WRONG answers.
The config that raises mean confidence isolates the confidence diluter; if
none moves, the fabric itself is diluting (shared-fabric crowding) and the
fix is a dedicated fabric + recalibration under v3.2 physics.

Usage:
  python3 tools/confidence_ablation.py --model model-b77-sem \
      --eval data/bank77_cal.jsonl [--limit 200] [--qkey intent] [--dump out.json]

Rows may be trained-schema ({"state","questions","labels"}) or handcrafted
probes lacking "labels" (confidence distribution only, no accuracy).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402

CONFIGS = [
    ("default", {}),
    ("no-retrieval", {"retrieval": False}),
    ("no-semantics", {"semantics": False}),
    ("baseline(both off)", {"retrieval": False, "semantics": False}),
]

# Bridge opts are STICKY per engine handle: every decide must re-state every
# knob, otherwise a later config inherits the previous config's switch state
# (measured artifact: "no-semantics" silently also ran with retrieval off).
FULL_OPTS = {"semantics": True, "retrieval": True, "hierarchy": True}


def per_call(opts: dict) -> dict:
    o = dict(FULL_OPTS)
    o.update(opts)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--qkey", default="intent")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--energy-norm", action="store_true",
                    help="measure under the M1 gain (match a calibrate run "
                         "done with --energy-norm; opts are per-call so the "
                         "mode is part of the measurement)")
    ap.add_argument("--dump", default="", help="write per-config summary JSON")
    a = ap.parse_args()

    if a.energy_norm:
        FULL_OPTS["energy_norm"] = True

    rows = [json.loads(l) for l in open(a.eval) if l.strip()][: a.limit]
    eng = SyFoxEngine(a.model)

    results = {}
    for name, opts in CONFIGS:
        recs = []
        for r in rows:
            q = r.get("questions") or {a.qkey: r.get("question")}
            ans_a, _u, _e = eng.decide(r["state"], q, per_call(opts))
            ans = ans_a.get(a.qkey, {})
            gold = ""
            if "labels" in r and r["labels"]:
                gold = r["labels"].get(a.qkey) or next(iter(r["labels"].values()))
                if isinstance(gold, dict):
                    gold = gold.get("choice", "")
            recs.append({"choice": ans.get("choice", ""),
                         "conf": ans.get("confidence", 0.0),
                         "deferred": ans.get("deferred", False),
                         "gold": gold})
        answered = [x for x in recs if not x["deferred"]]
        confs = [x["conf"] for x in answered]
        scored = [x for x in recs if x["gold"]]
        correct = [x for x in scored if not x["deferred"] and x["choice"] == x["gold"]]
        wrong = [x for x in scored if not x["deferred"] and x["choice"] != x["gold"]]
        results[name] = {
            "n": len(recs),
            "answered": len(answered),
            "deferred": len(recs) - len(answered),
            "acc": round(len(correct) / len(answered), 4) if answered else None,
            "mean_conf": round(statistics.mean(confs), 4) if confs else 0.0,
            "max_conf": round(max(confs), 4) if confs else 0.0,
            "min_conf": round(min(confs), 4) if confs else 0.0,
            "median_conf": round(statistics.median(confs), 4) if confs else 0.0,
            "mean_conf_correct": round(statistics.mean([x["conf"] for x in correct]), 4) if correct else None,
            "mean_conf_wrong": round(statistics.mean([x["conf"] for x in wrong]), 4) if wrong else None,
        }

    print(f"model={a.model} eval={a.eval} rows={len(rows)} qkey={a.qkey}")
    print(f"{'config':<22}{'acc':>7}{'defers':>8}{'meanC':>8}{'medC':>8}"
          f"{'maxC':>8}{'minC':>8}{'C|ok':>8}{'C|bad':>8}")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "  -  "
    for name, _ in CONFIGS:
        r = results[name]
        print(f"{name:<22}{fmt(r['acc']) if r['acc'] is not None else '  -  ':>7}"
              f"{r['deferred']:>8}{fmt(r['mean_conf']):>8}{fmt(r['median_conf']):>8}"
              f"{fmt(r['max_conf']):>8}{fmt(r['min_conf']):>8}"
              f"{fmt(r['mean_conf_correct']) if r['mean_conf_correct'] is not None else '  -  ':>8}"
              f"{fmt(r['mean_conf_wrong']) if r['mean_conf_wrong'] is not None else '  -  ':>8}")

    d, nr, ns, bo = (results[k] for k, _ in CONFIGS)
    print("\ndiagnosis:")
    if nr["mean_conf"] > d["mean_conf"] * 1.25:
        print(f"  retrieval priming dilutes confidence ({d['mean_conf']} -> "
              f"{nr['mean_conf']} with --no-retrieval): lower --retrieval-dose "
              f"(0.30 -> 0.10/0.05) or --retrieval-topk, then recalibrate.")
    else:
        print("  retrieval dose is not the diluter.")
    if ns["mean_conf"] > d["mean_conf"] * 1.25:
        print(f"  semantic leak dilutes confidence ({d['mean_conf']} -> "
              f"{ns['mean_conf']} with --no-semantics): recalibrate under v3.2 "
              f"physics or lower sem_coupling.")
    else:
        print("  semantic leak is not the diluter.")
    if bo["mean_conf"] <= d["mean_conf"] * 1.25 and d["mean_conf"] < 0.20:
        print("  neither knob moves it -> shared-fabric dilution or stale "
              "calibration: recalibrate under v3.2 physics (calibrate), or "
              "move the domain to a dedicated fabric.")

    if a.dump:
        with open(a.dump, "w") as f:
            json.dump({"model": a.model, "eval": a.eval, "rows": len(rows),
                       "configs": results}, f, indent=1)
        print(f"\nwrote {a.dump}")


if __name__ == "__main__":
    main()
