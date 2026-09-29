#!/usr/bin/env python3
"""b77_confusion_report.py — render a bench --confusion JSON into the
intra-category diagnostics views (v3.9.1).

Inputs
  --bench  bench JSON (from `syfox bench ... --confusion`)
  --dump   optional per-row dump (bench --dump-perrow) for reason anatomy
Views
  1. full 77x77 matrix: per-class row table (n, answered, acc, honest,
     worst confusion target) with natural intent names.
  2. card_management 22x22 + OTHER row/col: the intra-category binding
     constraint view (category membership = the shipped bank77_hier.py
     card_management rule, applied over data/bank77_categories.json).
  3. top confusion pairs over the whole matrix.

Every number comes from the bench JSON / dump produced by an actual run;
nothing is simulated here.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# card_management membership — the SAME priority-ordered rules bank77_hier.py
# ships (top_up > withdrawal > transfer > card_payment > exchange_rate >
# card_management > ...). Duplicating the rule here keeps the diagnostic
# view consistent with the fabric's own category lessons.
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
    for cat, fn in RULES:
        if fn(natural):
            return cat
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--card-only", action="store_true",
                    help="print only the card_management view")
    args = ap.parse_args()

    d = json.loads(Path(args.bench).read_text())
    conf = d.get("confusion")
    if not conf:
        print("bench JSON has no confusion block (run bench with --confusion)",
              file=sys.stderr)
        sys.exit(2)
    labels = conf["labels"]
    mat = conf["matrix"]
    cats = json.loads((ROOT / "data" / "bank77_categories.json").read_text())
    natural = {v: k for k, v in cats["label_of"].items()}   # cNN -> name

    # ---- per-class full table ----------------------------------------------
    routing = d["routing"]
    acc_all = routing["choice_accuracy"]
    if not args.card_only:
        print(f"model {d['model']}  eval {d['eval_source']}  rows {d['rows']}  "
              f"acc(answered) {acc_all}  defer {routing['choice_defer_rate']}  "
              f"honest_total {round(acc_all * (1 - routing['choice_defer_rate']), 4)}")
        print(f"{'gold':42s} {'n':>4s} {'ans':>4s} {'acc':>6s} {'honest':>6s}  worst confusion")
        rows_out = []
        for i, g in enumerate(labels):
            if g == "DEFER":
                continue
            row = mat[i]
            n_tot = sum(row)
            n_ans = n_tot - row[labels.index("DEFER")] if "DEFER" in labels else n_tot
            correct = row[i]
            acc_ans = correct / n_ans if n_ans else 0.0
            honest = correct / n_tot if n_tot else 0.0
            worst_n, worst_j = 0, -1
            for j, v in enumerate(row):
                if j != i and j < len(labels) and labels[j] != "DEFER" and v > worst_n:
                    worst_n, worst_j = v, j
            worst = f"{labels[worst_j]}={natural.get(labels[worst_j],'?')}({worst_n})" if worst_j >= 0 else "-"
            rows_out.append((honest, g, n_tot, n_ans, acc_ans, worst))
        rows_out.sort(key=lambda t: (t[0], t[1]))
        for honest, g, n_tot, n_ans, acc_ans, worst in rows_out:
            print(f"{g+'='+natural.get(g,'?'):42s} {n_tot:4d} {n_ans:4d} "
                  f"{acc_ans:6.3f} {honest:6.3f}  {worst}")

    # ---- card_management 22x22 + OTHER -------------------------------------
    card = [c for c in cats["classes"] if category_of(c) == "card_management"]
    card_ids = sorted(cats["label_of"][c] for c in card)
    view = card_ids + ["OTHER"]
    idx = {l: i for i, l in enumerate(labels)}
    print(f"\ncard_management view ({len(card_ids)} intents + OTHER) "
          f"rows=gold cols=pred:")
    hdr = "gold\\pred".ljust(38) + "".join(c.replace("c", "").rjust(5) for c in card_ids) + "OTH".rjust(6)
    print(hdr)
    agg = {}
    for g in card_ids + ["OTHER"]:
        golds = card_ids if g == "OTHER" else [g]
        rowsum = [0] * (len(card_ids) + 1)
        n_tot = 0
        for gg in golds:
            if gg not in idx:
                continue
            i = idx[gg]
            for j, l in enumerate(labels):
                v = mat[i][j]
                if l == "DEFER":
                    continue
                col = card_ids.index(l) if l in card_ids else len(card_ids)
                rowsum[col] += v
                if l in card_ids or gg in card_ids:
                    pass
            if gg in card_ids:
                n_tot += sum(x for j, x in enumerate(mat[i])
                             if labels[j] != "DEFER")
        agg[g] = (rowsum, n_tot)
        name = (natural.get(g, "ALL_OTHER") if g != "OTHER" else "ALL_OTHER")
        print((g + "=" + name).ljust(38)[:38]
              + "".join(str(v).rjust(5) for v in rowsum[:len(card_ids)])
              + str(rowsum[-1]).rjust(6))
    print("\ncard_management per-intent honest totals:")
    print(f"{'intent':44s} {'n':>4s} {'correct':>7s} {'honest':>6s} {'acc_ans':>7s}")
    tot_c = tot_n = 0
    for g in card_ids:
        rowsum, n_tot = agg[g]
        correct = rowsum[card_ids.index(g)]
        acc_ans = correct / n_tot if n_tot else 0.0
        tot_c += correct
        tot_n += n_tot
        print(f"{g+'='+natural.get(g,'?'):44s} {n_tot:4d} {correct:7d} "
              f"{(correct / n_tot if n_tot else 0):6.3f} {acc_ans:7.3f}")
    print(f"{'card_management TOTAL':44s} {tot_n:4d} {tot_c:7d} "
          f"{(tot_c / tot_n if tot_n else 0):6.3f}")

    # ---- top pairs ----------------------------------------------------------
    if not args.card_only:
        print("\ntop confusion pairs (gold -> pred, n):")
        for p in conf.get("pairs_top", [])[:15]:
            print(f"  {p['gold']}={natural.get(p['gold'],'?'):44s} -> "
                  f"{p['pred']}={natural.get(p['pred'],'?'):44s} {p['n']}")

    # ---- defer reason anatomy from the dump ---------------------------------
    if args.dump:
        from collections import Counter
        rows = [json.loads(l) for l in open(args.dump)]
        reasons = Counter(r["reason"] for r in rows if r["deferred"])
        print("\ndefer reasons (dump):", dict(reasons) if reasons else
              "none — all rows answered")


if __name__ == "__main__":
    main()
