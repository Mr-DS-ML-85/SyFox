#!/usr/bin/env python3
# ============================================================================
#  SyFox — gen_variants.py (v2.2)
#  Variant/paraphrase generation for mass-guarded re-teaching.
#
#  Physics contract (measured, v2.1 + v2.2):
#    * v2.1 measured raw re-teaching of near-duplicates REGRESSING held-out
#      accuracy (tickets 1.000 -> 0.500): intern() counts mass linearly and
#      injection damps by 1/sqrt(mass), so re-exposed surface forms re-weight
#      the field toward the training surface. That artifact is now fixed at
#      the ENGINE boundary: `syfox learn --augment` skips intern() for
#      concepts already in the vocabulary, so a variant lesson lays and
#      strengthens LANES only — coverage grows, acoustic mass does not.
#    * This generator therefore never needs to hold back: it produces the
#      variants; the mass guard makes them safe. The A/B stays reproducible
#      (train-only vs train+variants, `make models-vast`).
#
#  Variant ops per taught state (5-10 variants, deterministic — seeded by
#  the row index, never by wall-clock):
#    1. synonym swap      — 1-2 swappable words replaced via data/synonyms.txt
#    2. clause rotation   — comma/and-segments rotated left by one
#    3. adjacent reorder  — two adjacent words swapped (mid-sentence)
#    4. filler addition   — "Please", "I think", "Asap," style fillers
#    5. combo             — reorder + filler + synonym in one
#
#  Discipline (hard-coded, not optional):
#    * refuses *_heldout.jsonl paths — eval data is never transformed,
#      never generated from, never taught from;
#    * input must be a *_train.jsonl file; output is <stem>_expanded.jsonl;
#    * labels and question schemas are copied verbatim from the source row.
# ============================================================================
import json
import random
import re
import sys
from pathlib import Path

SEED = 20260927
WORD_RE = re.compile(r"[A-Za-z]+")
CLAUSE_RE = re.compile(r",\s+|\s+and\s+|\.\s+")


def load_synonyms(path: Path) -> dict:
    """canonical -> [variants]; used both directions for surface swaps."""
    table = {}
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        canon, rest = line.split(":", 1)
        variants = [v.strip().lower() for v in rest.split(",") if v.strip()]
        table[canon.strip().lower()] = variants
    return table


def swap_synonyms(text: str, table: dict, rng: random.Random, n_swaps: int) -> str:
    """Replace up to n_swaps swappable words (canonical->variant or variant->canonical)."""
    words = text.split(" ")
    swappable = []
    for i, w in enumerate(words):
        lw = re.sub(r"[^a-z]", "", w.lower())
        for canon, variants in table.items():
            if lw == canon or lw in variants:
                swappable.append((i, canon, variants))
                break
    if not swappable:
        return text
    rng.shuffle(swappable)
    for i, canon, variants in swappable[:n_swaps]:
        choice = rng.choice(variants + [canon])
        words[i] = re.sub(r"[A-Za-z]+", choice, words[i], count=1)
    return " ".join(words)


def rotate_clauses(text: str) -> str:
    parts = CLAUSE_RE.split(text)
    if len(parts) < 2:
        return text
    parts = parts[1:] + parts[:1]
    return ", ".join(p.strip(" .") for p in parts) + "."


def swap_adjacent(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    if len(words) < 4:
        return text
    lo, hi = 1, len(words) - 2                    # never the first/last word
    i = rng.randint(lo, hi - 1)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


FILLERS_PRE = ["Please help:", "I think", "Quick one,", "Hey,"]
FILLERS_POST = ["Please.", "Thanks.", "Asap please.", "Appreciate it."]


def add_filler(text: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return f"{rng.choice(FILLERS_PRE)} {text[0].lower()}{text[1:]}"
    return f"{text} {rng.choice(FILLERS_POST)}"


def variants_for(state: str, table: dict, rng: random.Random, count: int,
                 with_fillers: bool) -> list:
    out, seen = [], {state.lower()}
    ops = [
        lambda s: swap_synonyms(s, table, rng, 1),
        lambda s: swap_synonyms(s, table, rng, 2),
        rotate_clauses,
        lambda s: swap_adjacent(s, rng),
    ]
    if with_fillers:
        ops += [lambda s: add_filler(swap_adjacent(s, rng), rng),
                lambda s: add_filler(swap_synonyms(s, table, rng, 1), rng)]
    for _ in range(count * 4):                    # retry budget for dedup
        if len(out) >= count:
            break
        op = rng.choice(ops)
        v = op(state)
        if v.lower() in seen or not v.strip():
            continue
        seen.add(v.lower())
        out.append(v)
    return out


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_fillers = "--full" in sys.argv
    if not args:
        print("usage: gen_variants.py FILE_train.jsonl [--full]  (refuses *_heldout*)")
        return 2
    src = Path(args[0])
    if "heldout" in src.name:
        print(f"REFUSED: {src.name} is held-out eval data; never transformed, never taught from")
        return 2
    if not src.name.endswith("_train.jsonl"):
        print(f"REFUSED: {src.name} must be a *_train.jsonl file (train-side discipline)")
        return 2
    count = 8 if not with_fillers else 10
    for a in sys.argv[1:]:
        if a.startswith("--count="):
            count = int(a.split("=", 1)[1])

    table = load_synonyms(src.parent / "synonyms.txt")
    stem = src.name.replace("_train.jsonl", "")
    out_path = src.parent / (f"{stem}_expanded_full.jsonl" if with_fillers
                             else f"{stem}_expanded.jsonl")
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            rng = random.Random(SEED + idx)      # deterministic per row
            state = row.get("state", "")
            for v in variants_for(state, table, rng, count, with_fillers):
                f.write(json.dumps({"state": v, "questions": row["questions"],
                                    "labels": row["labels"]}, ensure_ascii=False) + "\n")
                written += 1
    print(f"{src.name}: {len(rows)} taught states -> {written} variants -> {out_path.name}"
          f" ({'full' if with_fillers else 'reorder+synonym'} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
