#!/usr/bin/env python3
# ============================================================================
#  bank77_hier.py — v3.2 Stage 3 (semantic hierarchy) + Stage 4 prep for the
#  DEDICATED bank77 fabric.
#
#  What it emits (deterministic, seed-free):
#    data/bank77_hier_train.jsonl — the v3.1.0 opaque/empty-instructions rows
#        PLUS one category lesson per row (interleaved intent/category), so
#        BOTH levels of the hierarchy are trained anchors in one fabric.
#    data/bank77_hierarchy.json   — ships into the model dir as hierarchy.json:
#        {"intents": {cNN: h_<cat>}, "categories": {h_<cat>: {"criteria": ...}},
#         "floor": 0.35}
#    data/bank77_memories.jsonl   — stride-sampled lived experiences for
#        retrieval-by-default ({"label": cNN, "state": text}).
#
#  Category assignment is prefix/keyword rules over the NATURAL intent names
#  (Banking77's own naming), applied in priority order — top_up wins over
#  transfer because "top_up_by_bank_transfer_charge" contains both words.
#  No ML anywhere: this is a deterministic labeling function + the criteria
#  miner already shipped in v3.1.0.
# ============================================================================
import argparse
import json
from pathlib import Path

RULES = [
    ("top_up", lambda n: "top_up" in n or "topping_up" in n),
    ("withdrawal", lambda n: "withdrawal" in n or "cash_received" in n
        or n == "atm_support"),
    ("transfer", lambda n: "transfer" in n
        or n in ("beneficiary_not_allowed", "receiving_money")),
    ("card_payment", lambda n: "card_payment" in n
        or n in ("transaction_charged_twice", "extra_charge_on_statement",
                 "exchange_charge", "reverted_card_payment?",
                 "declined_card_payment", "pending_card_payment")),
    ("exchange_rate", lambda n: "exchange" in n
        or n in ("fiat_currency_support", "supported_cards_and_currencies",
                 "visa_or_mastercard")),
    ("card_management", lambda n: "card" in n
        or n in ("contactless_not_working", "passcode_forgotten", "pin_blocked",
                 "change_pin", "apple_pay_or_google_pay", "lost_or_stolen_card",
                 "activate_my_card")),
    ("refund", lambda n: "refund" in n),
    ("verify_identity", lambda n: "identity" in n or "verify" in n
        or n in ("age_limit", "country_support", "edit_personal_details",
                 "terminate_account", "lost_or_stolen_phone")),
]


def category_of(natural: str) -> str:
    n = natural.lower()
    for name, rule in RULES:
        if rule(n):
            return name
    return "account_other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/bank77_train.jsonl")
    ap.add_argument("--criteria", default="data/bank77_criteria.json")
    ap.add_argument("--categories", default="data/bank77_categories.json")
    ap.add_argument("--out-train", default="data/bank77_hier_train.jsonl")
    ap.add_argument("--out-hierarchy", default="data/bank77_hierarchy.json")
    ap.add_argument("--out-memories", default="data/bank77_memories.jsonl")
    ap.add_argument("--memories", type=int, default=256,
                    help="stride-sampled memories for retrieval-by-default")
    ap.add_argument("--cat-words-per-intent", type=int, default=8)
    ap.add_argument("--floor", type=float, default=0.35)
    args = ap.parse_args()

    cats = json.loads(Path(args.categories).read_text())
    label_of = cats["label_of"]                     # natural -> cNN
    anchor_of = {natural: label_of[natural] for natural in cats["classes"]}

    # intent -> category (assert full coverage)
    cat_of_intent, members = {}, {}
    for natural in cats["classes"]:
        cat = category_of(natural)
        cat_of_intent[anchor_of[natural]] = cat
        members.setdefault(cat, []).append(natural)
    covered = sum(len(v) for v in members.values())
    assert covered == len(cats["classes"]), f"category rules lost {covered} != 77"
    print(f"categories ({len(members)}):")
    for cat in sorted(members):
        print(f"  {cat:16s} {len(members[cat]):2d} intents")

    # category criteria: union of member intents' mined criteria words
    crit = json.loads(Path(args.criteria).read_text())
    crit = crit["criteria"] if isinstance(crit.get("criteria"), dict) else crit
    # keyed by NATURAL intent name ("w, w, ..." comma text)
    def words_of(natural: str) -> list:
        v = crit.get(natural, [])
        if isinstance(v, str):
            return [w.strip() for w in v.split(",") if w.strip()]
        return list(v)

    cat_criteria = {}
    for cat, naturals in members.items():
        seen, words = set(), []
        for natural in naturals:
            for w in words_of(natural)[: args.cat_words_per_intent]:
                if w.lower() not in seen:
                    seen.add(w.lower())
                    words.append(w)
        cat_criteria["h_" + cat] = " ".join(words[: 3 * args.cat_words_per_intent])

    hierarchy = {
        "intents": {a: "h_" + c for a, c in cat_of_intent.items()},
        "categories": {"h_" + c: {"criteria": cat_criteria["h_" + c]}
                       for c in members},
        "floor": args.floor,
    }
    Path(args.out_hierarchy).write_text(json.dumps(hierarchy, indent=1))
    print(f"wrote {args.out_hierarchy}")

    # training rows: original (intent) rows + category rows, interleaved
    n_intent = n_cat = 0
    with open(args.train) as src, open(args.out_train, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_intent += 1
            # category twin of the same state: labels.hier = h_<cat>
            try:
                anchor = row["labels"]["intent"]
            except KeyError:
                continue
            cat = cat_of_intent.get(anchor)
            if cat is None:
                continue
            crow = {
                # "hier " prefix: the category twin must NOT collide with the
                # intent lesson in the contradiction detector (same state text
                # + a different outcome = false dispute; the audit ledger would
                # fill with counter-evidence against the intent binding).
                "state": "hier " + row["state"],
                "questions": {"hier": {
                    "type": "choice",
                    "instructions": "",
                    "criteria": {"h_" + cat: cat_criteria["h_" + cat]},
                }},
                "labels": {"hier": "h_" + cat},
            }
            dst.write(json.dumps(crow, ensure_ascii=False) + "\n")
            n_cat += 1
    print(f"wrote {args.out_train}: {n_intent} intent + {n_cat} category lessons")

    # retrieval memories: stride sample (deterministic)
    rows = []
    with open(args.train) as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    stride = max(1, len(rows) // max(1, args.memories))
    picked = rows[::stride][: args.memories]
    with open(args.out_memories, "w") as dst:
        for row in picked:
            dst.write(json.dumps({
                "label": row["labels"]["intent"],
                "state": row["state"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_memories}: {len(picked)} memories (stride {stride})")


if __name__ == "__main__":
    main()
