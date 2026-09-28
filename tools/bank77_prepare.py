#!/usr/bin/env python3
"""banking77 → SyFox data pipeline (M1 discipline).

Real dataset: PolyAI banking77 (10,003 train / 3,080 test CSV rows, 77 intents).

Splits (hidden-test firewall applies by filename suffix):
  data/bank77_train.jsonl   90% of the official train split   -> learn may read
  data/bank77_cal.jsonl     10% of the official train split   -> calibrate only
  data/bank77_hidden.jsonl  the official test split (3,080)   -> bench ONLY

Per-class criteria are MINED FROM THE TRAIN SPLIT ONLY (never cal, never
hidden) — distinctive-term log-ratio with smoothing, unigrams + bigrams,
min in-class count. The same mined string is written into every row's
criteria object, so learn (teaches label + criteria text) and decide (probes
label + criteria text) see identical anchors.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9']+")

# readout noise control: stopword fragments steal direct-term hits from every
# row (measured: class wrong_exchange_rate mined "was different", "and bit",
# "known" -> it absorbed wrong predictions from ALL 76 other classes)
STOP = set("a an and are as at be been but by can could did do does for from "
           "had has have how i in is it its me my no not of on or our so than "
           "that the their them then there these they this to was we were what "
           "when which who why will with would you your".split())


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


NGRAM_MAX = 2


def ngrams(toks: list[str]) -> list[str]:
    out = list(toks)
    if NGRAM_MAX >= 2:
        out += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return out


def mine_criteria(rows_by_class: dict[str, list[str]], topk: int,
                  min_count: int, purity: float = 0.0,
                  _stoplist=None, _word_purity=None) -> dict[str, str]:
    args_free_stoplist = _stoplist or [False]
    word_purity = _word_purity or [0.0]
    """Distinctive terms per class: smoothed log-ratio of in-class rate vs
    out-of-class rate, computed over unigrams+bigrams of the TRAIN rows."""
    doc_freq: dict[str, int] = Counter()
    class_tf: dict[str, Counter] = {}
    word_doc_freq: dict[str, int] = Counter()
    word_class_tf: dict[str, Counter] = {}
    for cls2, texts2 in rows_by_class.items():
        tf2 = Counter()
        for t2 in texts2:
            tf2.update(set(tokens(t2)))
        word_class_tf[cls2] = tf2
        word_doc_freq.update(tf2)
    for cls, texts in rows_by_class.items():
        tf = Counter()
        for t in texts:
            tf.update(set(ngrams(tokens(t))))          # per-document presence
        class_tf[cls] = tf
        doc_freq.update(tf)   # add per-class DOCUMENT frequencies (not class counts)
    total_docs = sum(len(v) for v in rows_by_class.values())
    criteria = {}
    for cls, tf in class_tf.items():
        n_cls = len(rows_by_class[cls])
        n_out = total_docs - n_cls
        scored = []
        for term, c in tf.items():
            if c < min_count:
                continue
            word_share_global = False
            if word_purity[0] > 0:
                for wt in term.split():
                    c_w = word_class_tf.get(cls, Counter()).get(wt, 0)
                    c_w_out = word_doc_freq.get(wt, 0) - c_w
                    if c_w_out / max(1, n_out) > word_purity[0]:
                        word_share_global = True
                        break
            if word_share_global:
                continue
            if args_free_stoplist[0]:
                words_t = term.split()
                if words_t[0] in STOP or words_t[-1] in STOP:
                    continue                   # stopword-anchored fragment
            c_out = doc_freq[term] - c
            if purity > 0 and c_out > purity * n_out:
                continue                       # purity cap (0 = off)
            p_in = (c + 0.5) / (n_cls + 1.0)
            p_out = (c_out + 0.5) / (n_out + 1.0)
            scored.append((math.log(p_in / p_out), c, term))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        criteria[cls] = ", ".join(t for _, _, t in scored[:topk])
    return criteria


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="/tmp/b77train.csv")
    ap.add_argument("--test-csv", default="/tmp/b77test.csv")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--cal-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--word-purity", type=float, default=0.0,
                    help="drop terms whose WORDS are global hubs (closes the "
                         "pure-bigram harvest loophole; 0 = off)")
    ap.add_argument("--stoplist", action="store_true",
                    help="drop stopword-anchored terms (measured: HURTS — "
                         "verb+stop combos like 'was taken' are the signal)")
    ap.add_argument("--purity", type=float, default=0.0,
                    help="max out-of-class doc share for a mined term")
    ap.add_argument("--max-ngram", type=int, default=2, choices=[1, 2],
                    help="1 = unigram criteria only (cleaner direct-term signal)")
    ap.add_argument("--opaque-labels", action="store_true",
                    help="label classes c00..c76: dedicated per-class anchor "
                         "nodes instead of shared English-word hubs")
    ap.add_argument("--no-instructions", action="store_true",
                    help="empty question instructions: learn_example folds "
                         "instructions into the state side, and identical "
                         "per-row instructions are hub pollution (they are "
                         "never injected at decide)")
    args = ap.parse_args()
    globals()["NGRAM_MAX"] = args.max_ngram

    def read_csv(p):
        with open(p, newline="", encoding="utf-8") as f:
            return [(r["text"].strip(), r["category"].strip()) for r in csv.DictReader(f)]

    train = read_csv(args.train_csv)
    hidden = read_csv(args.test_csv)
    classes = sorted({c for _, c in train})
    assert len(classes) == 77, f"expected 77 classes, got {len(classes)}"
    print(f"train={len(train)} hidden={len(hidden)} classes={len(classes)}")

    # stratified 90/10 carve of the official train split (cal never touches hidden)
    rng = random.Random(args.seed)
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text, cls in train:
        by_class[cls].append((text, cls))
    train_rows, cal_rows = [], []
    for cls in sorted(by_class):
        pool = sorted(by_class[cls])
        rng.shuffle(pool)
        n_cal = max(1, round(len(pool) * args.cal_frac))
        cal_rows += pool[:n_cal]
        train_rows += pool[n_cal:]

    texts_by_class: dict[str, list[str]] = defaultdict(list)
    for text, cls in train_rows:
        texts_by_class[cls].append(text)
    criteria = mine_criteria(texts_by_class, args.topk, args.min_count,
                             args.purity, [args.stoplist], [args.word_purity])

    label_of = {c: f"c{i:02d}" for i, c in enumerate(classes)} \
        if args.opaque_labels else {c: c for c in classes}

    def make_row(text: str, cls: str) -> dict:
        crit = {label_of[c]: criteria[c] for c in classes}
        # keep a side-record of the mapping for opaque runs (decode later)
        return {
            "state": text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "" if args.no_instructions else
                    "Which banking intent does this message belong to",
                    "criteria": crit,
                }
            },
            "labels": {"intent": label_of[cls]},
        }

    out = Path("data")
    files = {
        "bank77_train.jsonl": [make_row(t, c) for t, c in train_rows],
        "bank77_cal.jsonl": [make_row(t, c) for t, c in cal_rows],
        "bank77_hidden.jsonl": [make_row(t, c) for t, c in hidden],
    }
    manifest: dict[str, dict] = {}
    for name, rows in files.items():
        p = out / name
        write_jsonl(p, rows)
        manifest[name] = {"rows": len(rows), "sha256": sha256(p)}
        print(f"{name}: {len(rows)} rows")

    Path("data/bank77_criteria.json").write_text(
        json.dumps({"topk": args.topk, "min_count": args.min_count,
                    "source": "train split only (90% carve)", "criteria": criteria},
                   indent=1, ensure_ascii=False))
    Path("data/bank77_categories.json").write_text(json.dumps(
        {"classes": classes, "opaque": args.opaque_labels,
         "label_of": {c: f"c{i:02d}" for i, c in enumerate(classes)} if args.opaque_labels else None},
        indent=1))
    manifest["bank77_criteria.json"] = {"note": "mined per-class anchors (train-only)"}
    Path("data/bank77_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"split sizes: train={len(train_rows)} cal={len(cal_rows)} hidden={len(hidden)}")
    print(f"example criteria[card_arrival]: {criteria['card_arrival'][:110]}")


if __name__ == "__main__":
    main()
