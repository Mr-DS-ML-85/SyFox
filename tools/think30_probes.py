#!/usr/bin/env python3
"""think30 — the physics-native thinking suite (v3.8.0).

30 tests across 6 categories x 5, scored for answered accuracy and honesty:
  A choice   — baseline classification competence (5)
  B why      — causal CAUSE?>EFFECT routing, novel surface forms (5)
  C how      — instrumental MEANS!>GOAL routing (5)
  D xdomain  — cross-domain causal transfer (5)
  E chains   — multi-hop sequences 2-3 steps (5)
  F honesty  — unanswerable states MUST defer (5)

Two configs are measured on the SAME probe files:
  baseline : v3.7-style — fabric learned with causal lanes OFF, plain decide
  thinking : v3.8.0 — fabric learned with --causal-lanes on --typed-lanes on,
             decide with --causal-lanes on --adaptive-depth --self-verify
Multi-vector (needs --distvec) is measured as a THIRD config to keep the
baseline/decide-flag axes separable.

Usage: think30_probes.py [--json-out PATH]
Every number in the output comes from an actual subprocess run in this tree.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "syfox")
DATA = os.path.join(ROOT, "data", "think30")


def run(args):
    p = subprocess.run([BIN] + args, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"syfox {' '.join(args[:3])}... rc={p.returncode}: {p.stderr[:300]}")
    return json.loads(p.stdout)


def learn(model, examples, extra):
    if os.path.exists(model):
        for f in os.listdir(model):
            os.remove(os.path.join(model, f))
        os.rmdir(model)
    # 3-epoch consolidation with the mass-guarded augment re-teach (the
    # bank77 lesson): re-exposures strengthen lanes, never re-deposit mass.
    # Calibrate fits the temperature/Platt scalars on the TRAIN rows
    # (disclosed: in-sample for the scalars, argmax-neutral by monotonicity;
    # without it the honest tie gate reads the raw ~0.02 energy margins of
    # small fabrics and defers everything).
    run(["learn", "--model", model, "--examples", examples,
         "--epochs", "3", "--augment"] + extra)
    run(["calibrate", "--model", model, "--examples", examples] + extra)


def decide(model, state, questions, extra):
    out = run(["decide", "--model", model, "--state", state,
               "--questions", json.dumps(questions)] + extra)
    return out["answers"], out.get("usage", {})


def score_probe(ans, gold):
    """returns (correct, deferred)"""
    qid = list(ans.keys())[0]
    a = ans[qid]
    deferred = bool(a.get("deferred"))
    if gold == "DEFER":
        return deferred, deferred            # correct iff deferred
    if deferred:
        return False, True
    return a.get("choice") == gold, False


def run_suite(config_name, learn_extra, decide_extra):
    tmp = tempfile.mkdtemp(prefix=f"think30-{config_name}-")
    fabrics = {
        "A": (os.path.join(DATA, "choice_train.jsonl"), []),
        "BCD": (os.path.join(DATA, "causal_train.jsonl"), []),
        "D2": (os.path.join(DATA, "analogy_train.jsonl"), []),
        "E": (os.path.join(DATA, "chains_train.jsonl"), []),
    }
    models = {}
    for key, (train, lextra) in fabrics.items():
        model = os.path.join(tmp, f"m{key}")
        learn(model, train, learn_extra + lextra)
        models[key] = model

    tests = []          # (category, id, model_key, probe)
    for line in open(os.path.join(DATA, "choice_probes.jsonl")):
        tests.append(("A", None, "A", json.loads(line)))
    why = [json.loads(l) for l in open(os.path.join(DATA, "causal_probes.jsonl"))]
    for t in why:
        tests.append(("B" if t.get("kind") == "why" else "C", t["id"], "BCD", t))
    for line in open(os.path.join(DATA, "analogy_probes.jsonl")):
        tests.append(("D", None, "D2", json.loads(line)))
    for line in open(os.path.join(DATA, "chains_probes.jsonl")):
        tests.append(("E", None, "E", json.loads(line)))
    for line in open(os.path.join(DATA, "honesty_probes.jsonl")):
        tests.append(("F", None, "A", json.loads(line)))

    rows, per_cat = [], {}
    for cat, tid, mk, probe in tests:
        qs = probe["questions"]
        ans, usage = decide(models[mk], probe["state"], qs, decide_extra)
        gold = probe.get("gold", probe.get("labels", {}).get(list(qs.keys())[0]))
        correct, deferred = score_probe(ans, gold)
        per_cat.setdefault(cat, [0, 0, 0])
        per_cat[cat][0] += 1
        per_cat[cat][1] += 1 if correct else 0
        per_cat[cat][2] += 1 if deferred else 0
        rows.append({"cat": cat, "id": tid, "gold": gold, "correct": correct,
                     "deferred": deferred,
                     "choice": ans[list(ans.keys())[0]].get("choice"),
                     "reason": ans[list(ans.keys())[0]].get("reason")})
    total = sum(v[0] for v in per_cat.values())
    correct = sum(v[1] for v in per_cat.values())
    return {"config": config_name, "accuracy": round(correct / total, 3),
            "correct": correct, "total": total,
            "per_category": {k: {"n": v[0], "correct": v[1], "deferred": v[2]}
                             for k, v in sorted(per_cat.items())},
            "rows": rows, "fabric_dir": tmp}


def main():
    json_out = None
    if "--json-out" in sys.argv:
        json_out = sys.argv[sys.argv.index("--json-out") + 1]

    print("== think30: physics-native thinking suite (30 tests) ==")
    baseline = run_suite("baseline",
                         learn_extra=["--causal-lanes", "off", "--typed-lanes", "off"],
                         decide_extra=[])
    print(f"baseline (v3.7-style fabric + plain decide): "
          f"accuracy {baseline['accuracy']} ({baseline['correct']}/{baseline['total']})")
    for c, v in baseline["per_category"].items():
        print(f"  {c}: {v['correct']}/{v['n']} (defer {v['deferred']})")

    thinking = run_suite("thinking",
                         learn_extra=["--causal-lanes", "on", "--typed-lanes", "on"],
                         decide_extra=["--causal-lanes", "on", "--typed-lanes", "on",
                                       "--adaptive-depth", "--self-verify",
                                       "--perturb-check"])
    print(f"thinking (--causal-lanes --typed-lanes --adaptive-depth --self-verify): "
          f"accuracy {thinking['accuracy']} ({thinking['correct']}/{thinking['total']})")
    for c, v in thinking["per_category"].items():
        print(f"  {c}: {v['correct']}/{v['n']} (defer {v['deferred']})")

    mv = run_suite("multivector",
                   learn_extra=["--causal-lanes", "on", "--typed-lanes", "on", "--distvec"],
                   decide_extra=["--causal-lanes", "on", "--typed-lanes", "on",
                                 "--adaptive-depth", "--self-verify", "--perturb-check",
                                 "--multi-vector"])
    print(f"multivector (thinking flags + --multi-vector on a --distvec fabric): "
          f"accuracy {mv['accuracy']} ({mv['correct']}/{mv['total']})")
    for c, v in mv["per_category"].items():
        print(f"  {c}: {v['correct']}/{v['n']} (defer {v['deferred']})")

    print(f"DELTA thinking - baseline: "
          f"{round(thinking['accuracy'] - baseline['accuracy'], 3)}")

    if json_out:
        doc = {"baseline": baseline, "thinking": thinking, "multivector": mv,
               "comment": "every number from an actual syfox run in this tree; "
                          "A choice / B why / C how / D xdomain / E chains / F honesty(DEFER gold)",
               }
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"json -> {json_out}")


if __name__ == "__main__":
    main()
