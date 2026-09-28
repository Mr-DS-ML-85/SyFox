#!/usr/bin/env python3
# ============================================================================
#  router_prepare.py — v3.2 two-stage architecture, stage 1.
#  Builds the DEDICATED ROUTER FABRIC: a small (500-node class) substrate
#  whose only job is domain routing — 16 anchors, one per domain, physics
#  only (no classifier, no ML; the same SI substrate decides).
#
#  Source: data/giant/train.jsonl rows already carry the router question
#  (qkey "domain": criteria for rt00..rt15). We carve a balanced sample:
#    data/router16_train.jsonl  — train rows for the router fabric
#    data/router16_cal.jsonl    — small cal carve (router accuracy check)
#    data/router16_router.json  — ships as router.json in the model dir:
#        {"anchors": {rtNN: criteria}, "models": {rtNN: model-dir},
#         "domains": {rtNN: domain-name}}
#
#  The two-stage contract (CLI: decide --router model-router16 ...):
#    stage 1: model-router16 settles the state -> picks the domain anchor
#    stage 2: the mapped domain fabric decides the actual questions
#  Domain fabrics stay isolated; the router fabric never sees domain labels.
# ============================================================================
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/giant/train.jsonl")
    ap.add_argument("--schemas", default="data/giant/schemas.json")
    ap.add_argument("--out-train", default="data/router16_train.jsonl")
    ap.add_argument("--out-cal", default="data/router16_cal.jsonl")
    ap.add_argument("--out-router", default="data/router16_router.json")
    ap.add_argument("--per-domain", type=int, default=60)
    ap.add_argument("--cal-per-domain", type=int, default=10)
    ap.add_argument("--bank77-model", default="model-b77-sem")
    ap.add_argument("--default-model", default="model-giant-latin")
    args = ap.parse_args()

    schemas = json.loads(Path(args.schemas).read_text())
    # labels.domain carries the ANCHOR (rt00..rt15); schemas.router maps domain -> anchor
    domain_of = {a: d for d, a in schemas["router"].items() if not d.startswith("_")}
    anchors = sorted(domain_of)                     # deterministic order
    print(f"domains ({len(anchors)}):",
          ", ".join(f"{domain_of[a]}->{a}" for a in anchors))

    # one pass: bucket rows by router anchor, stride-sample inside each bucket.
    # Rows are REWRITTEN to carry ONLY the domain question: learn teaches every
    # question in a row, and the router fabric must never see domain labels.
    buckets = defaultdict(list)
    with open(args.train) as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            a = row.get("labels", {}).get("domain")
            if a is None:
                continue
            slim = {
                "state": row["state"],
                "questions": {"domain": row["questions"]["domain"]},
                "labels": {"domain": a},
            }
            buckets[a].append(json.dumps(slim, ensure_ascii=False))

    train_rows, cal_rows = [], []
    for a in anchors:
        rows = buckets.get(a, [])
        if not rows:
            print(f"  WARNING: no rows for anchor {a} ({domain_of[a]})")
            continue
        stride = max(1, len(rows) // max(1, args.per_domain))
        picked = rows[::stride][: args.per_domain]
        train_rows.extend(picked)
        # deterministic cal carve: the row after each picked stride boundary
        cal_idx = [min(i * stride + 1, len(rows) - 1) for i in range(args.cal_per_domain)]
        seen = set()
        for ci in cal_idx:
            if rows[ci] not in seen and rows[ci] not in picked:
                cal_rows.append(rows[ci])
                seen.add(rows[ci])
    print(f"train rows: {len(train_rows)}  cal rows: {len(cal_rows)}")

    Path(args.out_train).write_text("\n".join(train_rows) + "\n")
    Path(args.out_cal).write_text("\n".join(cal_rows) + "\n")

    # router.json: anchors = the modal criteria text per anchor across rows
    crit_votes = defaultdict(Counter)
    for line in train_rows:
        row = json.loads(line)
        crit = row["questions"]["domain"]["criteria"]
        items = crit.obj.items() if hasattr(crit, "obj") else crit.items()
        for tgt, text in items:
            crit_votes[tgt][text] += 1
    keep = set(anchors)
    anchors_out = {a: votes.most_common(1)[0][0]
                   for a, votes in sorted(crit_votes.items()) if a in keep}
    models = {}
    for a in anchors:
        d = domain_of[a]
        models[a] = args.bank77_model if d == "bank77" else args.default_model
    router = {
        "anchors": anchors_out,
        "models": models,
        "domains": {a: domain_of[a] for a in anchors},
    }
    Path(args.out_router).write_text(json.dumps(router, indent=1))
    print(f"wrote {args.out_router} ({len(anchors_out)} anchors)")


if __name__ == "__main__":
    main()
