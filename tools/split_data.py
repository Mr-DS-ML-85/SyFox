#!/usr/bin/env python3
# ============================================================================
#  SyFox — split_data.py
#  P1 of the evaluation-methodology fix: turn each data/<domain>_train.jsonl
#  (which despite its name holds the FULL dataset, used for both training and
#  eval until v2.1) into a real 70/30 train/held-out split.
#
#  Honesty properties:
#    * deterministic: fixed seed, stratified by first label (sorted-key order,
#      same convention as bench.hpp first_label), so re-running on the same
#      input reproduces the same split byte-for-byte.
#    * refuses to double-split: if the *_heldout.jsonl file already exists the
#      script exits with an error unless --force. The current *_train.jsonl is
#      treated as the FULL dataset and rewritten in place to its 70% share.
#    * the pre-split full files remain recoverable from git history.
#    * held-out rows are NEVER paraphrased or re-taught (see gen_paraphrases.py,
#      which refuses heldout paths, and the Makefile, which only learns on
#      *_train.jsonl + *_paraphrased.jsonl).
#
#  Usage:  python3 tools/split_data.py [--seed 20260926] [--force]
# ============================================================================
import argparse
import json
import os
import random
import sys

SEED = 20260926
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def first_label(row):
    """Same convention as bench.hpp first_label(): first label value in
    sorted-key order (labels is a JSON object -> python dict preserves the
    parsed order; sort explicitly to match JVObj = std::map)."""
    labels = row.get("labels") or {}
    if not isinstance(labels, dict) or not labels:
        return ""
    return labels[sorted(labels.keys())[0]]


def split_rows(rows, seed):
    """Stratified 70/30: group by first label, shuffle each group with a
    per-group seeded RNG, round(0.3*n) rows -> heldout. Train keeps the
    original relative order; heldout keeps the original relative order."""
    groups = {}
    for idx, row in enumerate(rows):
        groups.setdefault(first_label(row), []).append(idx)
    heldout_idx = set()
    for key in sorted(groups.keys()):
        idxs = groups[key]
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(idxs)
        n_hold = round(len(idxs) * 0.30)
        if len(idxs) >= 2 and n_hold == 0:
            n_hold = 1                       # every stratum with >=2 rows gets a voice
        if len(idxs) - n_hold < 1:
            n_hold = len(idxs) - 1           # train must keep at least one row
        heldout_idx.update(idxs[:n_hold])
    train = [r for i, r in enumerate(rows) if i not in heldout_idx]
    hold = [r for i, r in enumerate(rows) if i in heldout_idx]
    return train, hold


def dist(rows):
    d = {}
    for r in rows:
        k = first_label(r)
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items()))


def main():
    ap = argparse.ArgumentParser(description="SyFox 70/30 stratified data split")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing *_heldout.jsonl files")
    ap.add_argument("--data-dir", default=DATA)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.data_dir)
                   if f.endswith("_train.jsonl"))
    if not files:
        sys.exit("split_data: no *_train.jsonl files found")
    # A file is "already split" when its matching heldout file exists: rerunning
    # would split the 70% again, silently shrinking train. Refuse by default.
    todo = []
    for f in files:
        hold_path = os.path.join(args.data_dir, f.replace("_train", "_heldout"))
        if os.path.exists(hold_path) and not args.force:
            sys.exit(f"split_data: {hold_path} exists — refusing to double-split "
                     f"(use --force only if you restored the full file from git)")
        todo.append(f)

    total_train = total_hold = 0
    for f in todo:
        path = os.path.join(args.data_dir, f)
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        train, hold = split_rows(rows, args.seed)
        with open(path, "w", encoding="utf-8") as fh:
            for r in train:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        hold_path = path.replace("_train", "_heldout")
        with open(hold_path, "w", encoding="utf-8") as fh:
            for r in hold:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_train += len(train)
        total_hold += len(hold)
        print(f"{f}: {len(rows)} rows -> train {len(train)} / heldout {len(hold)}")
        print(f"    train labels   {dist(train)}")
        print(f"    heldout labels {dist(hold)}")
    print(f"total: train {total_train} / heldout {total_hold} "
          f"({total_hold / (total_train + total_hold):.1%} held out)")


if __name__ == "__main__":
    main()
