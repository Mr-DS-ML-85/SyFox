#!/usr/bin/env python3
# ============================================================================
#  SyFox — gen_corpus.py  (Milestone 1: large held-out datasets)
#
#  Generates the v3 corpora at 5k-20k rows per domain with the exact house
#  schema (state / questions / labels) and splits each 70/15/15 into
#  train / calibration / hidden test.
#
#  Label honesty: every label follows from WHICH POOL the state was composed
#  from (see corpus_pools.py). Nothing is labelled at random.
#
#  Firewall contract (enforced again in the CLI, core/firewall.hpp):
#    * *_train.jsonl  — learnable (the only files `syfox learn` may read)
#    * *_cal.jsonl    — calibration only (temperature/Platt fit; never learned)
#    * *_hidden.jsonl — bench only. NEVER taught, NEVER calibrated, NEVER
#                       used for derivation or model selection.
#  A MANIFEST.json records row counts, label distributions and sha256 per
#  file so any later run can prove the splits were not touched.
#
#  Deterministic: seeded per-row RNG (seed, domain, index); re-running with
#  the same seed reproduces every file byte-for-byte.
#
#  Usage: python3 tools/gen_corpus.py [--seed 20260927] [--out-dir data/big]
# ============================================================================
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_pools as pools

SEED = 20260927

# verbatim question schemas from the seed data (single-line JSON)
Q_TICKETS = r'{"department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems","sales":"pricing or account questions"}},"frustration":{"type":"score","instructions":"How frustrated the customer appears","criteria":["calm just stating facts","frustrated but civil","very angry strong language"]},"is_urgent":{"type":"noul","instructions":"the message conveys urgency or time sensitivity"}}'
Q_GUARD = r'{"irreversible":{"type":"noul","instructions":"the command is irreversible or destructive"},"off_task":{"type":"noul","instructions":"the command is off task and unrelated to the goal"},"scope":{"type":"choice","instructions":"what does the command touch","criteria":{"none":"no changes at all","read":"only reads lists shows","write":"modifies project files or data","global":"system wide destructive or irreversible"}}}'
Q_GAME = r'{"action":{"type":"choice","instructions":"What should the bot do next","criteria":{"flee":"run away escape avoid danger retreat safe","fight":"attack combat weapon sword strike","dig_in":"hide wait build shelter fortify safe"}}}'

TICKETS_QS = json.loads(Q_TICKETS)
GUARD_QS = json.loads(Q_GUARD)
GAME_QS = json.loads(Q_GAME)


def first_label(labels):
    return labels[sorted(labels.keys())[0]]


def compose_tickets_en(i, rng):
    intents = sorted(pools.TICKETS_INTENT.keys())
    intent = intents[i % len(intents)]
    fr = (i // len(intents)) % 3
    ur = (i // (3 * len(intents))) % 2 == 1
    state = pools.tickets_state(intent, fr, ur, rng)
    return state, TICKETS_QS, {"department": intent, "frustration": str(fr),
                                "is_urgent": "true" if ur else "false"}


def compose_guard(i, rng):
    state, labels = pools.guard_state(rng)
    return state, GUARD_QS, labels


def compose_game(i, rng):
    actions = sorted(pools.GAME_FEATURES.keys())
    action = actions[i % len(actions)]
    return pools.game_state(action, rng), GAME_QS, {"action": action}


def compose_tickets_ml(lang, i, rng):
    intents = sorted(pools.TICKETS_ML_INTENT[lang].keys())
    intent = intents[i % len(intents)]
    fr = (i // len(intents)) % 3
    ur = (i // (3 * len(intents))) % 2 == 1
    state = pools.ml_state(lang, intent, fr, ur, rng)
    return state, TICKETS_QS, {"department": intent, "frustration": str(fr),
                                "is_urgent": "true" if ur else "false"}


def make_rows(name, n, composer, seed):
    """Compose n UNIQUE rows (dedup by state text; deterministic retries)."""
    rows, seen, dups = [], set(), 0
    i = 0
    while len(rows) < n:
        rng = random.Random(f"{seed}:{name}:{i}")
        state, qs, labels = composer(i, rng)
        i += 1
        if state in seen:
            dups += 1
            if dups > n * 3:
                sys.exit(f"gen_corpus: {name}: vocabulary exhausted at {len(rows)} rows")
            continue
        seen.add(state)
        rows.append({"state": state, "questions": qs, "labels": labels})
    return rows, dups


def split_70_15_15(rows, seed):
    """Stratified 70/15/15 by first label (sorted-key order, same convention
    as bench.hpp first_label). Per stratum: seeded shuffle, round(0.15) to
    cal and hidden (min 1 when the stratum has >=3 rows), train keeps the rest."""
    groups = {}
    for idx, r in enumerate(rows):
        groups.setdefault(first_label(r["labels"]), []).append(idx)
    cal_idx, hidden_idx = set(), set()
    for key in sorted(groups.keys()):
        idxs = list(groups[key])
        rng = random.Random(f"{seed}:split:{key}")
        rng.shuffle(idxs)
        n = len(idxs)
        n_cal = round(n * 0.15)
        n_hid = round(n * 0.15)
        if n >= 3:
            n_cal = max(1, n_cal)
            n_hid = max(1, n_hid)
        if n - n_cal - n_hid < 1:
            n_cal = max(0, (n - 1) // 2)
            n_hid = n - 1 - n_cal
        cal_idx.update(idxs[:n_cal])
        hidden_idx.update(idxs[n_cal:n_cal + n_hid])
    train = [r for i, r in enumerate(rows) if i not in cal_idx and i not in hidden_idx]
    cal = [r for i, r in enumerate(rows) if i in cal_idx]
    hidden = [r for i, r in enumerate(rows) if i in hidden_idx]
    return train, cal, hidden


def dist(rows):
    d = {}
    for r in rows:
        k = first_label(r["labels"])
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items()))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="SyFox v3 corpus generator (70/15/15)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "big"))
    ap.add_argument("--tickets", type=int, default=12000)
    ap.add_argument("--guard", type=int, default=8000)
    ap.add_argument("--game", type=int, default=8000)
    ap.add_argument("--ml", type=int, default=3600,
                    help="rows per language for tickets_bn/hi/ru (15%% >= 500 -> >=3334)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [
        ("tickets_en", args.tickets, compose_tickets_en),
        ("guard", args.guard, compose_guard),
        ("game", args.game, compose_game),
        ("tickets_bn", args.ml, lambda i, rng: compose_tickets_ml("bn", i, rng)),
        ("tickets_hi", args.ml, lambda i, rng: compose_tickets_ml("hi", i, rng)),
        ("tickets_ru", args.ml, lambda i, rng: compose_tickets_ml("ru", i, rng)),
    ]

    manifest = {
        "generator": "tools/gen_corpus.py",
        "seed": args.seed,
        "schema": "state/questions/labels (house format, unchanged)",
        "splits": {"train": 0.70, "cal": 0.15, "hidden": 0.15},
        "firewall": {
            "train": "learnable (syfox learn)",
            "cal": "calibration only (syfox calibrate); never learned from",
            "hidden": "bench only; NEVER taught, calibrated, derived or used "
                      "for model selection (enforced in core/firewall.hpp)",
        },
        "files": {},
    }

    for name, n, composer in jobs:
        rows, dups = make_rows(name, n, composer, args.seed)
        train, cal, hidden = split_70_15_15(rows, args.seed)
        counts = {}
        for tag, part in (("train", train), ("cal", cal), ("hidden", hidden)):
            path = os.path.join(args.out_dir, f"{name}_{tag}.jsonl")
            write_jsonl(path, part)
            counts[tag] = len(part)
            manifest["files"][os.path.basename(path)] = {
                "rows": len(part),
                "label_dist": dist(part),
                "sha256": sha256_of(path),
                "role": tag,
            }
        print(f"{name}: {len(rows)} unique rows (dedup skipped {dups}) -> "
              f"train {counts['train']} / cal {counts['cal']} / hidden {counts['hidden']}")
        print(f"   train dist   {dist(train)}")

    mpath = os.path.join(args.out_dir, "MANIFEST.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")
    total = sum(f["rows"] for f in manifest["files"].values())
    print(f"manifest: {mpath} ({len(manifest['files'])} files, {total} rows total)")


if __name__ == "__main__":
    main()
