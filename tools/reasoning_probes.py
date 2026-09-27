#!/usr/bin/env python3
"""Reasoning probe suite (v3.3): question-context selection before/after the
question-conditioned readout gate.

Builds a small QA-style fabric (the same shape as the user's dense benchmark:
statements + QA pairs as lessons), then runs the reported failure probes:

  Test 1  who found the radio        -> expect Tariq, was radio (salient noun)
  Test 2  same, {Tariq,Salma} only   -> tie broken by criteria order, was Salma
  Test 4  unknown options xyz/abc/def -> winner by noise, exact-tie case
  Test 10 A->B->C chain              -> transitive propagation
  Test 16 latest temperature         -> temporal ordering
  Test 17 purple elephant / submarine -> OOD abstention

Usage:
  python3 tools/reasoning_probes.py [--rebuild] [--json-out F]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = str(ROOT / "build" / "syfox")
MODEL = str(ROOT / "model-reasoning-probe")

LESSONS = [
    # --- Test 1/2 fabric: Tariq / Salma / radio ---
    ("tariq found the radio in the lab", "tariq"),
    ("the radio was found by tariq", "tariq"),
    ("tariq picked up the radio", "tariq"),
    ("salma was there in the room", "salma"),
    ("salma saw the radio on the table", "salma"),
    ("who found the radio tariq did", "tariq"),
    ("who saw the radio salma did", "salma"),
    # --- Test 10 chain fabric: A -> B -> C (as event words) ---
    ("the alarm rang then the lights turned on", "lights"),
    ("the lights turned on then the door opened", "door"),
    ("what happens after the alarm the lights", "lights"),
    # --- Test 16 temporal fabric ---
    ("the temperature was twenty degrees in the morning", "twenty"),
    ("the temperature rose to twenty five degrees at noon", "twentyfive"),
    ("the temperature fell to twenty two degrees in the evening", "twentytwo"),
    ("the latest temperature reading was twenty two degrees", "twentytwo"),
    # --- OOD block: no submarine / favorite-color lessons anywhere ---
]

PROBES = [
    {
        "name": "Test 1: who found the radio",
        "state": "The radio was found by Tariq. Salma was there.",
        "q": {"i": {"type": "choice", "instructions": "Who found the radio?",
                    "criteria": {"Tariq": "the person who found it",
                                 "Salma": "a person present",
                                 "radio": "the object found"}}},
        "expect": "Tariq", "reported_failure": "radio (0.399) — salient noun",
    },
    {
        "name": "Test 2: two-person tie",
        "state": "The radio was found by Tariq. Salma was there.",
        "q": {"i": {"type": "choice", "instructions": "Who found the radio?",
                    "criteria": {"Tariq": "the person who found it",
                                 "Salma": "a person present"}}},
        "expect": "Tariq", "reported_failure": "Salma 0.5 tie (criteria order)",
    },
    {
        "name": "Test 4: unknown options",
        "state": "The radio was found by Tariq. Salma was there.",
        "q": {"i": {"type": "choice", "instructions": "Who found the radio?",
                    "criteria": {"xyz": "", "abc": "", "def": ""}}},
        "expect": "defer or tie disclosure", "reported_failure": "abc 0.344 by noise",
    },
    {
        "name": "Test 10: alarm-lights-door chain",
        "state": "The alarm rang.",
        "q": {"i": {"type": "choice", "instructions": "What turned on next?",
                    "criteria": {"alarm": "the alarm", "lights": "the lights",
                                 "door": "the door", "unknown": "nothing"}}},
        "expect": "lights (2-hop)", "reported_failure": "near-uniform",
    },
    {
        "name": "Test 16: latest temperature",
        "state": "The temperature was twenty degrees then rose to twenty five degrees then fell to twenty two degrees.",
        "q": {"i": {"type": "choice", "instructions": "What is the latest temperature?",
                    "criteria": {"twenty": "20 degrees", "twentyfive": "25 degrees",
                                 "twentytwo": "22 degrees"}}},
        "expect": "twentytwo", "reported_failure": "chose 20",
    },
    {
        "name": "Test 17: OOD submarine color",
        "state": "The purple elephant boarded the spaceship.",
        "q": {"i": {"type": "choice", "instructions": "What was the submarine's favorite color?",
                    "criteria": {"purple": "purple", "blue": "blue", "green": "green"}}},
        "expect": "defer (unanswerable)", "reported_failure": "purple 0.472",
    },
]


def decide(state, q, extra):
    cmd = [BIN, "decide", "--model", MODEL, "--energy-norm",
           "--state", state, "--questions", json.dumps(q)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip()}
    return json.loads(p.stdout)


def build_model():
    subprocess.run(["rm", "-rf", MODEL], check=True)
    rows = []
    for i, (state, label) in enumerate(LESSONS):
        rows.append({"state": state,
                     "questions": {"i": {"type": "choice", "instructions": "",
                                         "criteria": {label: label}}},
                     "labels": {"i": label}})
    path = ROOT / "data" / "reasoning_probe_lessons.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    subprocess.run([BIN, "learn", "--model", MODEL, "--examples", str(path)],
                   check=True, cwd=str(ROOT))
    subprocess.run([BIN, "calibrate", "--model", MODEL,
                    "--examples", str(path), "--energy-norm"],
                   check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    if a.rebuild or not Path(MODEL, "substrate.bin").exists():
        build_model()

    configs = [
        ("gate OFF (default)", []),
        ("gate ON floor 0.10", ["--question-gate", "--question-gate-floor", "0.10"]),
        ("gate ON floor 0.25", ["--question-gate"]),
        ("gate ON floor 0.50", ["--question-gate", "--question-gate-floor", "0.50"]),
    ]
    table = {}
    for cname, extra in configs:
        recs = []
        for pr in PROBES:
            out = decide(pr["state"], pr["q"], extra)
            ans = out.get("answers", {}).get("i", {})
            recs.append({
                "choice": ans.get("choice", ""),
                "conf": ans.get("confidence", 0),
                "deferred": ans.get("deferred", False),
                "tied": ans.get("tied", False),
                "gate": ans.get("question_gate", []),
                "probs": ans.get("probabilities", {}),
            })
        table[cname] = recs

    for i, pr in enumerate(PROBES):
        print(f"\n== {pr['name']}  [expect: {pr['expect']}; reported: {pr['reported_failure']}]")
        for cname, _ in configs:
            r = table[cname][i]
            tag = "DEFER" if r["deferred"] else f"{r['choice']} @ {r['conf']}"
            tie = " TIED" if r["tied"] else ""
            gt = f" gate={r['gate']}" if r["gate"] else ""
            print(f"   {cname:<30} -> {tag}{tie}{gt}")

    if a.json_out:
        json.dump({"probes": [p["name"] for p in PROBES],
                   "configs": table}, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
