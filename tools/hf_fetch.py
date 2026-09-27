#!/usr/bin/env python3
"""Fetch real HF datasets for the giant model via parquet shards.

For each dataset+split: GET /api/datasets/{id}/parquet -> shard URLs, download
up to 3 evenly-spaced shards (cached under data/hf_cache/parquet/), then sample
rows EVENLY over the concatenated shard rows (numpy linspace) so class-ordered
corpora still give proportional class coverage. Output: data/hf_raw/{dom}_{split}.json
with verbatim dataset text (truncation happens later in giant_prepare.py).

No synthetic text is added here.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hf_cache" / "parquet"
OUT = ROOT / "data" / "hf_raw"
UA = {"User-Agent": "syfox-giant-prepare/1.0"}


def http_get(url: str, dest: Path) -> None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    retry {attempt + 1} for {dest.name}: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def shard_urls(ds: str) -> dict:
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{ds}/parquet", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def pick(urls: list, n: int) -> list:
    if len(urls) <= n:
        return urls
    idx = np.linspace(0, len(urls) - 1, n).round().astype(int)
    return [urls[i] for i in dict.fromkeys(idx.tolist())]


def fetch_split(ds: str, cfg: str, split: str, n_sample: int,
                textf: str, labf: str, tag: str) -> list:
    tree = shard_urls(ds)
    urls = pick(tree[cfg][split], 3 if split == "train" else 2)
    frames = []
    for u in urls:
        name = "__".join(u.split("/datasets/")[1].split("/"))
        dest = CACHE / name
        if not dest.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            print(f"  downloading {name}")
            http_get(u, dest)
        cols = ([textf] if isinstance(textf, str) else list(textf)) + [labf]
        frames.append(pd.read_parquet(dest, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    if len(df) > n_sample:
        sel = np.linspace(0, len(df) - 1, n_sample).round().astype(int)
        df = df.iloc[sel]
    cols = [textf] if isinstance(textf, str) else list(textf)
    rows = [{"text": " ".join(str(x) for x in t), "label": l}
            for t, l in zip(df[cols].to_numpy().tolist(), df[labf].tolist())]
    out = OUT / f"{tag}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False))
    labs = sorted({str(r["label"]) for r in rows})
    print(f"{tag}: {len(rows)} rows, {len(labs)} labels "
          f"e.g. {labs[:6]} | sample {rows[0]['text'][:50]!r}")
    return rows


PLAN = [
    ("clinc", ("text", "intent"), {
        "train": ("clinc/clinc_oos", "plus", "train", 12000),
        "cal": ("clinc/clinc_oos", "imbalanced", "validation", 3300),
        "hidden": ("clinc/clinc_oos", "plus", "test", 4600)}),
    ("enron", ("text", "label"), {
        "train": ("SetFit/enron_spam", "default", "train", 8000),
        "cal": ("SetFit/enron_spam", "default", "test", 800),
        "hidden": ("SetFit/enron_spam", "default", "test", 2600)}),
    ("emotion", ("text", "label"), {
        "train": ("dair-ai/emotion", "split", "train", 8000),
        "cal": ("dair-ai/emotion", "split", "validation", 700),
        "hidden": ("dair-ai/emotion", "split", "test", 2100)}),
    ("hate", ("tweet", "class"), {
        "train": ("tdavidson/hate_speech_offensive", "default", "train", 9000)}),
    ("agnews", ("text", "label"), {
        "train": ("fancyzhx/ag_news", "default", "train", 8000),
        "hidden": ("fancyzhx/ag_news", "default", "test", 2000)}),
    ("polarity", ("content", "label"), {
        "train": ("fancyzhx/amazon_polarity", "amazon_polarity", "train", 8000),
        "hidden": ("fancyzhx/amazon_polarity", "amazon_polarity", "test", 2000)}),
    ("dbpedia", ("content", "label"), {
        "train": ("fancyzhx/dbpedia_14", "dbpedia_14", "train", 8000),
        "hidden": ("fancyzhx/dbpedia_14", "dbpedia_14", "test", 2000)}),
    ("boolq", (("question", "passage"), "answer"), {
        "train": ("google/boolq", "default", "train", 7500),
        "cal": ("google/boolq", "default", "validation", 3300)}),
]


def main() -> None:
    for dom, fields, splits in PLAN:
        for tag, (ds, cfg, split, n) in splits.items():
            fetch_split(ds, cfg, split, n, fields[0], fields[1], f"{dom}_{tag}")


if __name__ == "__main__":
    main()
