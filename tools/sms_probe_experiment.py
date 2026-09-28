#!/usr/bin/env python3
"""The decisive probe experiment: WHY handcrafted probes tie at 50/50.

Hypothesis under test (from the model-xl diagnostic): at learn time the
substrate wires state words -> label + TRAINING-criteria words. At decide
time the probe is label + criteria-AS-GIVEN. If the probe criteria are new
wording the fabric never saw, only the label token survives, and in a dense
fabric the option readouts converge to a near-tie.

Design (real UCI SMS spam, 5,574 rows, stratified 70/15/15):
  1. train ONE fabric (natural labels spam/ham, instructions inline,
     mined criteria) — mirroring the model-xl training style
  2. score the SAME hidden rows under three decide-side criteria:
       (a) matched   — the exact criteria the fabric was taught
       (b) reworded  — handcrafted wording in the style of the model-xl probes
       (c) empty     — labels only
  3. report top-1 accuracy and mean margin (p1-p2) per condition
     + the four model-xl-style handcrafted probes under each condition.

Model/config selection stays on cal; hidden is scored for the final report.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank77_prepare import mine_criteria  # noqa: E402  (same miner)

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = str(ROOT / "server")
HUMAN_PROBES = [
    ("ham",  "Hi John, just confirming our meeting tomorrow at 10 AM. Please let me know if the time still works for you."),
    ("spam", "Congratulations! You have won $5,000,000. Click this link immediately and provide your password to claim your prize."),
    ("ham",  "Your verification code is 482913. It expires in 10 minutes."),
    ("spam", "WINNER! You have been selected for a $1,000 cash prize. Call now to claim your reward before it expires!"),
]
REWORDED = {
    "spam": "unsolicited or deceptive message, scams, phishing, credential theft, fraudulent offers, or malicious requests",
    "ham": "legitimate personal, business, transactional, or routine communication with no deceptive or malicious intent",
}
INSTRUCTIONS = "Classify the message"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def make_row(text: str, cls: str, criteria: dict[str, str], label_map: dict[str, str],
             instructions: str) -> dict:
    return {
        "state": text,
        "questions": {"intent": {
            "type": "choice", "instructions": instructions,
            "criteria": {label_map[c]: criteria[c] for c in criteria}}},
        "labels": {"intent": label_map[cls]},
    }


def vary_criteria(eval_path: Path, out_path: Path, criteria: dict[str, str],
                  label_map: dict[str, str], instructions: str) -> None:
    rows = [json.loads(l) for l in eval_path.open()]
    for r in rows:
        r["questions"]["intent"]["criteria"] = {
            label_map[c]: criteria[c] for c in criteria}
        r["questions"]["intent"]["instructions"] = instructions
    write_jsonl(out_path, rows)


def bench(model: str, eval_file: str) -> dict:
    out = subprocess.run(
        [str(ROOT / "build/syfox"), "bench", "--model", model,
         "--eval", eval_file, "--energy-norm"],
        capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.stdout)


def main() -> None:
    sys.path.insert(0, BRIDGE)
    from syfox_bridge import SyFoxEngine  # noqa: E402

    df = pd.read_parquet("/tmp/sms.parquet").rename(columns={"sms": "text"})
    df["cls"] = df["label"].map({0: "ham", 1: "spam"})
    rows = list(df[["text", "cls"]].itertuples(index=False, name=None))

    # stratified 70/15/15 (seed 7)
    from collections import defaultdict
    import random
    rng = random.Random(7)
    by_class: dict[str, list] = defaultdict(list)
    for text, cls in rows:
        by_class[cls].append((text, cls))
    tr, cal, hid = [], [], []
    for cls in sorted(by_class):
        pool = sorted(by_class[cls])
        rng.shuffle(pool)
        n = len(pool)
        tr += pool[: int(n * 0.70)]
        cal += pool[int(n * 0.70): int(n * 0.85)]
        hid += pool[int(n * 0.85):]
    print(f"sms: train={len(tr)} cal={len(cal)} hidden={len(hid)}")

    texts_by_class: dict[str, list[str]] = defaultdict(list)
    for text, cls in tr:
        texts_by_class[cls].append(text)
    criteria = mine_criteria(texts_by_class, topk=12, min_count=3, purity=0.0,
                             _stoplist=[False])
    print("criteria[spam]:", criteria["spam"][:100])

    label_map = {"ham": "ham", "spam": "spam"}     # natural labels (model-xl style)
    d = ROOT / "data"
    write_jsonl(d / "sms_train.jsonl", [make_row(t, c, criteria, label_map, INSTRUCTIONS) for t, c in tr])
    write_jsonl(d / "sms_cal.jsonl", [make_row(t, c, criteria, label_map, INSTRUCTIONS) for t, c in cal])
    write_jsonl(d / "sms_hidden.jsonl", [make_row(t, c, criteria, label_map, INSTRUCTIONS) for t, c in hid])

    def run(*a):
        r = subprocess.run([str(ROOT / "build/syfox"), *a], capture_output=True,
                           text=True, cwd=ROOT)
        return json.loads(r.stdout) if r.stdout.strip() else {}

    print(run("learn", "--examples", "data/sms_train.jsonl",
              "--model", "model-sms-natural", "--epochs", "3").get("lanes"))
    run("calibrate", "--model", "model-sms-natural", "--examples", "data/sms_cal.jsonl")

    # ---- condition rows ----
    vary_criteria(d / "sms_hidden.jsonl", d / "sms_hidden_match.jsonl",
                  criteria, label_map, INSTRUCTIONS)
    vary_criteria(d / "sms_hidden.jsonl", d / "sms_hidden_reworded.jsonl",
                  REWORDED, label_map, INSTRUCTIONS)
    vary_criteria(d / "sms_hidden.jsonl", d / "sms_hidden_empty.jsonl",
                  {c: "" for c in criteria}, label_map, INSTRUCTIONS)

    print("\n=== decide-side criteria conditions (same fabric, same hidden rows) ===")
    print(f"{'condition':<12} {'top-1':>7} {'scored':>7} {'margin(p1-p2)':>14}")
    for name, f in [("matched", "sms_hidden_match.jsonl"),
                    ("reworded", "sms_hidden_reworded.jsonl"),
                    ("empty", "sms_hidden_empty.jsonl")]:
        r = bench("model-sms-natural", f"data/{f}")
        rt = r["routing"]
        print(f"{name:<12} {rt['choice_accuracy']:>7.3f} {rt['choice_n']:>7} "
              f"{rt.get('mean_margin', 0):>14.4f}")

    # ---- the four handcrafted probes under each condition ----
    print("\n=== handcrafted probes (model-xl style) under each condition ===")
    eng = SyFoxEngine(str(ROOT / "model-sms-natural"))
    for cond_name, crit in [("matched", criteria), ("reworded", REWORDED),
                            ("empty", {c: "" for c in criteria})]:
        hits = 0
        detail = []
        for gold, text in HUMAN_PROBES:
            q = {"intent": {"type": "choice", "instructions": INSTRUCTIONS,
                            "criteria": {label_map[c]: crit[c] for c in crit}}}
            answers, _, _ = eng.decide(text, q, {"energy_norm": True})
            a = answers["intent"]
            pred = a.get("choice")
            p = a.get("probabilities", {})
            margin = abs(p.get("spam", 0) - p.get("ham", 0))
            hits += pred == gold
            detail.append(f"{gold}: {pred} (|p1-p2|={margin:.3f}, conf={a.get('confidence')})")
        print(f"{cond_name:<9} {hits}/4   " + " | ".join(detail))


if __name__ == "__main__":
    main()
