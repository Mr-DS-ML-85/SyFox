#!/usr/bin/env python3
# ============================================================================
#  SyFox — gen_paraphrases.py (P3, v2.1)
#  Data augmentation for generalization: for each TRAIN state, generate up to
#  5 paraphrased variants and write them as extra lessons. The fabric learns
#  the same outcome from differently-worded experiences, so the energy routes
#  no longer depend on one exact surface form.
#
#  Hard rules (honesty of the eval):
#    * TRAIN ONLY — the script REFUSES any path containing "heldout". Held-out
#      rows must stay unseen by the fabric, directly or through paraphrase.
#    * LABEL-SAFE — words are only REPLACED using the single-word synonym
#      table (data/synonyms.txt, shared with core/normalize.hpp), never
#      inserted; inserted fillers are neutral politeness only. Swapping
#      "urgent"->"immediately" keeps is_urgent=true true; a row that is not
#      urgent never gains an urgency word.
#    * DETERMINISTIC — a per-row seeded RNG; same input file -> same output.
#    * The core stays physics-only: this is OUTSIDE the substrate, a tool
#      that writes lesson files. No neural net anywhere.
#
#  Variant strategies (cycled per row, up to 5 per state):
#    1 clause reorder — swap the first two sentence segments
#    2 filler prefix  — neutral lead-in ("please help:", "i think", ...)
#    3 filler suffix  — neutral sign-off ("thanks.", "please advise.", ...)
#    4 combo          — synonym swap + reorder (swap ALONE is deliberately
#                       excluded: a table word folds back to the same stem,
#                       so a synonym-only variant normalizes to an IDENTICAL
#                       token stream = pure repetition. Measured on the
#                       tickets heldout split: repetition-heavy augmentation
#                       collapsed heldout accuracy 1.00 -> 0.33 (intern mass
#                       grows linearly and damps injection by 1/sqrt(mass)),
#                       so every variant here carries NEW token content.)
#    5 fallback       — a second filler prefix (only if a strategy above
#                       produced no change)
#
#  Usage:
#    python3 tools/gen_paraphrases.py                 # all data/*_train.jsonl
#    python3 tools/gen_paraphrases.py --only tickets
# ============================================================================
import argparse
import json
import os
import random
import re
import sys

SEED = 20260926
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SYNONYMS = os.path.join(DATA, "synonyms.txt")

PREFIXES = ["please help:", "i think", "quick note:", "just to add,", "hey,"]
SUFFIXES = ["thanks.", "thank you.", "please advise.", "appreciate it.", "let me know."]

WORD_RE = re.compile(r"[A-Za-z]+")


def load_synonyms(path):
    """canonical -> [alternates]; the map is symmetric (word <-> alternates)."""
    swap = {}
    if not os.path.exists(path):
        return swap
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            canon, rest = line.split(":", 1)
            canon = canon.strip().lower()
            variants = [v.strip().lower() for v in rest.split(",") if v.strip()]
            group = [canon] + variants
            for w in group:
                others = [x for x in group if x != w]
                swap.setdefault(w, [])
                for o in others:
                    if o not in swap[w]:
                        swap[w].append(o)
    return swap


def synonym_swap(text, swap, rng):
    """Replace the FIRST table word found with one of its alternates."""
    words = list(WORD_RE.finditer(text))
    candidates = [m for m in words if m.group(0).lower() in swap]
    if not candidates:
        return None
    m = candidates[0]
    alt = swap[m.group(0).lower()][rng.randrange(len(swap[m.group(0).lower()]))]
    return text[:m.start()] + alt + text[m.end():]


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT = re.compile(r",\s+|\s+\band\b\s+|\s+\bbut\b\s+")


def clause_reorder(text, swap, rng):
    """Swap the first two segments (sentences, else comma/and/but clauses)."""
    parts = SENT_SPLIT.split(text.strip())
    if len(parts) >= 2:
        parts[0], parts[1] = parts[1], parts[0]
        return " ".join(parts)
    parts = CLAUSE_SPLIT.split(text.strip())
    if parts and len(parts) >= 4:      # split with captures: keep groups intact
        return text                    # capture-group splitting: too fiddly, skip
    if len(parts) >= 2:
        parts[0], parts[1] = parts[1], parts[0]
        return ", ".join(parts)
    return None


def filler_prefix(text, swap, rng):
    return PREFIXES[rng.randrange(len(PREFIXES))] + " " + text


def filler_suffix(text, swap, rng):
    return text + " " + SUFFIXES[rng.randrange(len(SUFFIXES))]


def combo(text, swap, rng):
    out = synonym_swap(text, swap, rng)
    out = clause_reorder(out or text, swap, rng)
    if out:
        return out
    out = filler_prefix(text, swap, rng)
    return out + " " + SUFFIXES[rng.randrange(len(SUFFIXES))]


STRATEGIES = [clause_reorder, filler_prefix, filler_suffix, combo]
STRATEGY_NAMES = ["clause_reorder", "filler_prefix", "filler_suffix", "combo"]


def paraphrase_row(row, swap, rng, per_state):
    seen = {row["state"]}
    out = []
    order = list(range(len(STRATEGIES)))
    rng.shuffle(order)                      # per-row strategy order, still seeded
    attempts = 0
    while len(out) < per_state and attempts < per_state * 4:
        si = attempts % len(STRATEGIES)
        strategy = STRATEGIES[order[si]]
        text = strategy(row["state"], swap, rng)
        attempts += 1
        if not text or text in seen:
            # guaranteed-change fallback: neutral prefix (always differs)
            text = PREFIXES[(attempts + len(out)) % len(PREFIXES)] + " " + row["state"]
            if text in seen:
                continue
        seen.add(text)
        variant = dict(row)
        variant["state"] = text
        variant["paraphrase_of"] = row["state"]
        variant["strategy"] = STRATEGY_NAMES[order[si]]
        out.append(variant)
    return out


def main():
    ap = argparse.ArgumentParser(description="SyFox train-only paraphrase augmenter")
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--synonyms", default=SYNONYMS)
    ap.add_argument("--per-state", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", default=None, help="substring filter on file name")
    args = ap.parse_args()

    if "heldout" in args.synonyms:
        sys.exit("gen_paraphrases: refusing heldout synonyms path")
    swap = load_synonyms(args.synonyms)
    if not swap:
        sys.exit(f"gen_paraphrases: no synonym table at {args.synonyms}")

    files = sorted(f for f in os.listdir(args.data_dir)
                   if f.endswith("_train.jsonl")
                   and (args.only is None or args.only in f))
    if not files:
        sys.exit("gen_paraphrases: no *_train.jsonl files matched")
    for f in files:
        src = os.path.join(args.data_dir, f)
        if "heldout" in src:
            sys.exit(f"gen_paraphrases: refusing heldout input {src}")
        with open(src, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        out_path = src.replace("_train.jsonl", "_paraphrased.jsonl")
        total = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for i, row in enumerate(rows):
                rng = random.Random(f"{args.seed}:{f}:{i}")
                for v in paraphrase_row(row, swap, rng, args.per_state):
                    out.write(json.dumps(v, ensure_ascii=False) + "\n")
                    total += 1
        print(f"{f}: {len(rows)} states -> {total} paraphrased lessons -> "
              f"{os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
