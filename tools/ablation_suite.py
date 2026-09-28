#!/usr/bin/env python3
"""Ablation suite (v3.2.1 BUG #1 verification): do the kill switches matter?

Runs the five paper configs on one eval set through the ctypes bridge:

  default | --no-semantics | --no-retrieval | --no-semantics --no-retrieval
          | --no-hierarchy

and reports per config: accuracy, mean/median confidence, deferrals, plus an
answer-signature distinctness count over the full candidate vector (the paper
criterion is >=4 distinct outputs; hierarchy counts only when the model arms
it with floor < 1.0 — b77-sem ships floor 1.0 = measured-off, disclosed).

Inertness is disclosed loudly: a pre-v3.2 fabric (no SEM4 tail) or a missing
memories.jsonl/hierarchy.json makes switches no-ops by the replay contract;
the tool says so, names the rebuild path, and exits non-zero so a silent
null-ablation can never reach the paper.

Usage:
  python3 tools/ablation_suite.py --model model-b77-sem \
      --eval data/bank77_cal.jsonl [--qkey intent] [--limit 100]
      [--energy-norm] [--dump out.json]
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
    ("no-semantics", {"semantics": False}),
    ("no-retrieval", {"retrieval": False}),
    ("both off", {"semantics": False, "retrieval": False}),
    ("no-hierarchy", {"hierarchy": False}),
]


def model_features(model_dir: Path) -> dict:
    import struct
    feats = {"sem_tail": False, "memories": False, "hierarchy": False,
             "hier_armed": False}
    sub = model_dir / "substrate.bin"
    if sub.exists():
        b = sub.read_bytes()
        feats["sem_tail"] = b.find(bytes([0x34, 0x4D, 0x45, 0x53])) >= 0
    feats["memories"] = (model_dir / "memories.jsonl").exists()
    h = model_dir / "hierarchy.json"
    if h.exists():
        try:
            j = json.loads(h.read_text())
            feats["hierarchy"] = True
            feats["hier_armed"] = float(j.get("floor", 1.0)) < 1.0
        except (ValueError, OSError):
            pass
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--qkey", default="intent")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--energy-norm", action="store_true")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    mdir = Path(a.model)
    feats = model_features(mdir)
    print(f"model={a.model} eval={a.eval} rows<={a.limit} "
          f"energy_norm={bool(a.energy_norm)}")
    print(f"features: sem_tail={feats['sem_tail']} "
          f"memories={feats['memories']} hierarchy={feats['hierarchy']} "
          f"(armed={feats['hier_armed']})")

    blockers = []
    if not feats["sem_tail"]:
        blockers.append(
            "--no-semantics is inert: no v3.2 SEM4 tail in substrate.bin — "
            "rebuild first: ./build/rebuild_sem SRC DST (or retrain under v3.2)")
    if not feats["memories"]:
        blockers.append("--no-retrieval is inert: no memories.jsonl in the "
                        "model dir (retrieval has nothing to prime)")
    if not feats["hier_armed"]:
        blockers.append("--no-hierarchy cannot move the readout: hierarchy.json "
                        "absent or floor=1.0 (gate shipped off by measurement)")
    for b in blockers:
        print(f"  NOTE: {b}")

    rows = [json.loads(l) for l in open(a.eval) if l.strip()][: a.limit]
    eng = SyFoxEngine(a.model)
    base = {"semantics": True, "retrieval": True, "hierarchy": True}
    if a.energy_norm:
        base["energy_norm"] = True

    results, sigs = {}, set()
    for name, off in CONFIGS:
        opts = dict(base)
        opts.update(off)
        acc_c = conf_c = n_ans = 0
        confs, sig = [], ""
        for r in rows:
            q = r.get("questions") or {a.qkey: r.get("question")}
            ans_a, _u, _e = eng.decide(r["state"], q, dict(opts))
            ans = ans_a.get(a.qkey, {})
            sig += f"{ans.get('choice','')}|{ans.get('confidence',0):.6f};"
            gold = ""
            if "labels" in r and r["labels"]:
                gold = r["labels"].get(a.qkey) or next(iter(r["labels"].values()))
                if isinstance(gold, dict):
                    gold = gold.get("choice", "")
            if not ans.get("deferred"):
                n_ans += 1
                confs.append(ans.get("confidence", 0.0))
                if gold and ans.get("choice") == gold:
                    acc_c += 1
            conf_c += 1 if gold and not ans.get("deferred") \
                and ans.get("choice") == gold else 0
        results[name] = {
            "acc_with_defer": round(acc_c / len(rows), 4) if rows else None,
            "acc_answered": round(acc_c / n_ans, 4) if n_ans else None,
            "answered": n_ans,
            "mean_conf": round(statistics.mean(confs), 4) if confs else 0.0,
            "median_conf": round(statistics.median(confs), 4) if confs else 0.0,
        }
        sigs.add(sig)
        print(f"  {name:<14} acc={results[name]['acc_with_defer']} "
              f"meanC={results[name]['mean_conf']} "
              f"medC={results[name]['median_conf']}")

    armed = len([c for c in CONFIGS if c[0] != "no-hierarchy"])
    target = 5 if feats["hier_armed"] else armed
    print(f"\ndistinct output signatures: {len(sigs)} / {len(CONFIGS)} configs "
          f"(criterion >= {target}"
          f"{'' if feats['hier_armed'] else ' — hierarchy ships off, 4 configs count'})")
    if len(sigs) < target:
        print("ABLATION BLOCKED: configs are byte-identical — the switches "
              "moved nothing on this model+input combination.")
        if not feats["sem_tail"]:
            print("Root cause: pre-v3.2 fabric (replay contract makes every "
                  "switch a no-op). Rebuild with ./build/rebuild_sem SRC DST.")
        sys.exit(2)
    print("ablation OK: the kill switches separate on this model.")
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump({"model": a.model, "eval": a.eval, "rows": len(rows),
                       "features": feats, "configs": results,
                       "distinct": len(sigs)}, f, indent=1)
        print(f"wrote {a.dump}")


if __name__ == "__main__":
    main()
