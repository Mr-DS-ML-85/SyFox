#!/usr/bin/env python3
"""v3.7.0 probe suite — measured tables for the four representation moves.

Every number written to data/v37_tables.json comes from an actual run of the
C++ CLI in this repository, via subprocess. Nothing is estimated.

Probes
------
1. role_reversal (Move 3, typed ordered lanes)
   Three fabrics learn the SAME 6 labeled rows:
     bag    : --bigrams off --typed-lanes off
     bigram : --bigrams on  --typed-lanes off   (the v3.6 default fabric)
     typed  : --bigrams on  --typed-lanes on    (v3.7)
   The probe set is 4 reversal pairs with GROWING insertion distance between
   the subject and the verb (0/1/3/6 inserted tokens) — the multiset theorem
   says the bag cannot see role at all, bigrams dilute as material is
   inserted, typed lanes carry the S and O relations across the inserts.
   Reported per distance: answered-accuracy and mean top-2 margin per fabric.

2. ood_bridge (Moves 1+2, pretrain + distvec)
   A small labeled billing/technical fabric; OOD probe rows use words the
   LABELED rows never carry but the UNLABELED corpus does (by-construction
   golds disclosed). Variants:
     plain        : labeled rows only
     + pretrain   : labeled rows, then pretrain over data/pretrain_corpus.txt
     + pretrain+distvec : pretrain with --distvec (PPMI+SVD resonance edges)
   Reported: defer rate and answered accuracy per variant. The claim under
   test: "the OOD defer stops firing on words the corpus KNOWS".

3. pretrain_loss (Move 2)
   The masked-basin loss curve (mean per epoch) of the pretrain run itself,
   plus vocabulary growth. Determinism: the run is repeated twice and the
   curves compared bit for bit.
"""
import json
import os
import subprocess
import sys

BIN = "./build/syfox"
OUT = "data/v37_tables.json"

QUESTIONS = json.dumps({
    "q1": {"type": "choice", "instructions": "who is the giver",
           "criteria": {"alice": "alice is the giver", "bob": "bob is the giver"}}
})


def run(args):
    r = subprocess.run([BIN] + args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"probe failed: {' '.join(args)}\n{r.stderr}")
    return json.loads(r.stdout)


def rm_rf(path):
    subprocess.run(["rm", "-rf", path], check=True)


def build_reversal_fabric(tag, extra):
    model = f"build/model-v37-rev-{tag}"
    rm_rf(model)
    run(["learn", "--model", model, "--examples", "data/role_reversal_train.jsonl"] + extra)
    return model


def margin_of(reply):
    a = reply["answers"]["q1"]
    if a.get("deferred"):
        return None
    probs = sorted(a["probabilities"].values(), reverse=True)
    return probs[0] - probs[1]


def probe_reversal():
    fabrics = {
        "bag": build_reversal_fabric("bag", ["--bigrams", "off", "--typed-lanes", "off"]),
        "bigram": build_reversal_fabric("bigram", ["--bigrams", "on", "--typed-lanes", "off"]),
        "typed": build_reversal_fabric("typed", ["--bigrams", "on", "--typed-lanes", "on"]),
    }
    probes = [json.loads(l) for l in open("data/role_reversal_probe.jsonl") if l.strip()]
    table = {}
    for tag, model in fabrics.items():
        by_dist = {}
        for p in probes:
            rep = run(["decide", "--model", model, "--state", p["state"],
                       "--questions", QUESTIONS, "--no-defer",
                       "--bigrams", "on" if tag != "bag" else "off",
                       "--typed-lanes", "on" if tag == "typed" else "off"])
            a = rep["answers"]["q1"]
            m = margin_of(rep)
            d = by_dist.setdefault(p["insertions"], {"n": 0, "correct": 0, "margins": []})
            d["n"] += 1
            if not a.get("deferred") and a.get("choice") == p["gold"]:
                d["correct"] += 1
            if m is not None:
                d["margins"].append(round(m, 6))
        table[tag] = {
            str(dist): {
                "n": d["n"],
                "answered_correct": d["correct"],
                "accuracy": round(d["correct"] / d["n"], 4),
                "mean_margin": round(sum(d["margins"]) / len(d["margins"]), 6) if d["margins"] else None,
            } for dist, d in sorted(by_dist.items())
        }
    return table


OOD_PROBES = [json.loads(l) for l in open("data/ood_bridge_probes.jsonl") if l.strip()]
OOD_QUESTIONS = json.dumps({
    "q1": {"type": "choice", "instructions": "which team handles this",
           "criteria": {"billing": "charges and refunds", "technical": "crashes and bugs"}}
})


def build_ood_fabric(tag, pretrain, distvec):
    model = f"build/model-v37-ood-{tag}"
    rm_rf(model)
    rows = []
    for st in ["refund the duplicate charge on my card",
               "the invoice charged my card twice",
               "the card statement shows a duplicate charge",
               "the refund for the extra charge is pending"]:
        rows.append(json.dumps({"state": st,
                                "questions": {"q1": {"type": "choice", "instructions": "",
                                                     "criteria": {"billing": "charge or refund issue",
                                                                  "technical": "app or login issue"}}},
                                "labels": {"q1": "billing"}}))
    for st in ["the app crash after the login screen",
               "login fails and the session crashes",
               "the app froze on the dashboard after login",
               "the crash report names the login module"]:
        rows.append(json.dumps({"state": st,
                                "questions": {"q1": {"type": "choice", "instructions": "",
                                                     "criteria": {"billing": "charge or refund issue",
                                                                  "technical": "app or login issue"}}},
                                "labels": {"q1": "technical"}}))
    with open(f"{model}-rows.jsonl", "w") as f:
        f.write("\n".join(rows) + "\n")
    os.makedirs(model, exist_ok=True)
    run(["learn", "--model", model, "--examples", f"{model}-rows.jsonl"])
    if pretrain:
        args = ["pretrain", "--model", model, "--corpus", "data/pretrain_corpus.txt",
                "--epochs", "12", "--mask-stride", "1", "--eta-scale", "2"]
        if distvec:
            args += ["--distvec", "--distvec-dims", "64"]
        run(args)
    # temperature calibration on the LABELED rows (the v2.1 tool, unchanged):
    # the OOD energies are real but small; the fit maps energy GAPS to
    # probabilities honestly. Disclosed in the table.
    run(["calibrate", "--model", model, "--examples", f"{model}-rows.jsonl"])
    return model


def probe_ood():
    variants = {
        "plain": build_ood_fabric("plain", False, False),
        "pretrain": build_ood_fabric("pretrain", True, False),
        "pretrain_distvec": build_ood_fabric("pretrain_distvec", True, True),
    }
    table = {}
    for tag, model in variants.items():
        n = correct = deferred = 0
        for p in OOD_PROBES:
            # --energy-norm on ALL variants (disclosed): the decide-side
            # M1 dose gain. On the plain fabric the probe words are dark, so
            # the gain cannot manufacture answers there; on the pretrain
            # fabrics it compensates the sqrt(mass) damping of the grown
            # vocabulary. Same knob, same dose, every variant.
            rep = run(["decide", "--model", model, "--state", p["state"],
                       "--energy-norm", "--questions", OOD_QUESTIONS])
            a = rep["answers"]["q1"]
            n += 1
            if a.get("deferred"):
                deferred += 1
            elif a.get("choice") == p["gold"]:
                correct += 1
        table[tag] = {
            "probes": n,
            "deferred": deferred,
            "defer_rate": round(deferred / n, 4),
            "answered_correct": correct,
            "answered_accuracy": round(correct / max(1, n - deferred), 4),
            "note": "probe golds are BY CONSTRUCTION (the pretrain lines bind each "
                    "probe word to one class's vocabulary); disclosed, not hidden",
        }
    return table


def probe_pretrain_loss():
    model = "build/model-v37-loss"
    rm_rf(model)
    os.makedirs(model, exist_ok=True)
    r1 = run(["pretrain", "--model", model, "--corpus", "data/pretrain_corpus.txt",
              "--epochs", "12", "--mask-stride", "1", "--eta-scale", "2"])["report"]
    rm_rf(model)
    os.makedirs(model, exist_ok=True)
    r2 = run(["pretrain", "--model", model, "--corpus", "data/pretrain_corpus.txt",
              "--epochs", "12", "--mask-stride", "1", "--eta-scale", "2"])["report"]
    return {
        "epoch_mean_loss": r1["epoch_mean_loss"],
        "deterministic": r1["epoch_mean_loss"] == r2["epoch_mean_loss"],
        "lines": r1["lines"],
        "masked_per_epoch": r1["masked"] // 12,
        "vocab_growth": r1["vocab_after"],
        "interned_new": r1["interned_new"],
    }


def main():
    print("measuring role_reversal (Move 3) ...", file=sys.stderr)
    reversal = probe_reversal()
    print("measuring ood_bridge (Moves 1+2) ...", file=sys.stderr)
    ood = probe_ood()
    print("measuring pretrain_loss (Move 2) ...", file=sys.stderr)
    loss = probe_pretrain_loss()
    out = {
        "suite": "syfox-v3.7.0-representation-moves",
        "probe_files": ["data/role_reversal_train.jsonl", "data/role_reversal_probe.jsonl",
                        "data/pretrain_corpus.txt"],
        "role_reversal": reversal,
        "ood_bridge": ood,
        "pretrain_loss": loss,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
