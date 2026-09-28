#!/usr/bin/env python3
"""Router-criteria wording sweep — selection ON CAL ONLY (never hidden).

The router question's criteria text is a decide-side probe: it selects which
nodes are read (docs/PROBING.md). Three wording variants per domain (short 4
words / mid 8 / rich 12) are scored on a deterministic cal subsample; the
winner is written to data/giant/router_criteria.json for pass-2 training and
final evaluation.

Usage: python3 tools/router_sweep.py --model-slug giant-latin
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402

GIANT = ROOT / "data" / "giant"

# word lists per latin domain: [short4, mid8, rich12]
VARIANTS = {
    "tickets_en": ["support billing sales technical",
                   "invoice refund subscription plan billing login error sales",
                   "invoice refund duplicate charge subscription seats login error page spins pricing quote annual plan"],
    "bank77": ["banking cards payments",
               "card payment transfer refund transaction atm balance fee",
               "card payment declined transfer not arrived refund pending top up exchange rate atm withdrawal fee charged exchange exchange rate balance inquiry"],
    "sms": ["text messages spam",
            "text message prize claim stop reply short code",
            "text message prize won claim free stop reply unsubscribe mobile offer urgent £ call now"],
    "clinc": ["assistant requests weather",
              "weather alarm timer restaurant booking translate flight card",
              "weather forecast alarm set timer restaurant reservation book table translate language flight status credit card limit change"],
    "enron": ["email inbox messages",
              "email inbox attachment forward meeting memo unsubscribe",
              "email inbox attachment forward meeting agenda memo attached file unsubscribe click verify password mailbox"],
    "emotion": ["feelings emotions reactions",
                "feeling happy sad angry scared love surprised",
                "feeling so happy sad angry scared terrified love surprised grateful crying tears over the moon"],
    "hate": ["social offensive posts",
             "offensive insult hateful abusive comment tweet species group",
             "offensive insult hateful abusive comment tweet targeted group species disgusting women people should not be allowed"],
    "agnews": ["news headlines sports",
               "headline world politics sports business technology reuters",
               "headline world politics sports business technology reuters oil prices election team quarter back climate summit"],
    "polarity": ["product reviews opinions",
                 "product review stars purchase quality recommend disappointed",
                 "product review stars purchase quality recommend disappointed broke after two days worth every cent sturdy elegant"],
    "dbpedia": ["encyclopedia entities topics",
                "company athlete film album village animal plant building",
                "company athlete footballer film album village animal plant building educational institution office holder river dolphin tower"],
    "boolq": ["boolean question passage",
              "passage question answer true false fact wikipedia",
              "passage question answer yes no true false fact wikipedia highest peak capital deposed first"],
    "game": ["game commands actions",
             "zombie skeleton turret sword pickaxe retreat attack",
             "zombie skeleton horde turret wall sword pickaxe health low retreat toward hall fight doorway"],
    "guard": ["safety destructive irreversible",
              "command destructive irreversible delete wipe safe operation",
              "command destructive irreversible delete wipe drop table production database audit logs safe monthly summary"],
}
ORDER = sorted(VARIANTS)  # rt anchor order must match schemas (sorted domains)


ABSTRACT = {
    "tickets_en": "customer support conversations about orders and accounts",
    "bank77": "banking service requests from customers",
    "sms": "brief personal mobile messages",
    "clinc": "spoken assistant requests for everyday tasks",
    "enron": "longer workplace email correspondence",
    "emotion": "expressions of personal feelings",
    "hate": "user comments that need moderation review",
    "agnews": "brief journalistic news summaries",
    "polarity": "written consumer opinions about purchases",
    "dbpedia": "factual encyclopedia style descriptions",
    "boolq": "reading comprehension with factual passages",
    "game": "interactive game situation commands",
    "guard": "system safety operation checks",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", default="giant-latin")
    ap.add_argument("--per-domain", type=int, default=250)
    args = ap.parse_args()

    schemas = json.load(open(GIANT / "schemas.json"))
    rt_of = schemas["router"]
    eng = SyFoxEngine(f"model-{args.model_slug}")
    router_q = {"type": "choice", "instructions": "",
                "criteria": schemas["domains"]["_router"]["criteria"]}

    # deterministic cal subsample across latin domains
    latin_cal = [json.loads(l) for l in open(GIANT / "cal_latin.jsonl")]
    by_dom = {d: [r for r in latin_cal if r["labels"]["domain"] == rt_of[d]]
              for d in ORDER}
    rows = []
    rng = random.Random(7)
    for dom in ORDER:
        cal = by_dom[dom]
        rng.shuffle(cal)
        rows.extend(cal[: args.per_domain])
    print(f"cal subsample: {len(rows)} rows x {len(VARIANTS['bank77'])} variants")

    results = []
    for level in range(3):
        rq = {"type": "choice", "instructions": "",
              "criteria": {rt_of[d]: VARIANTS[d][level] for d in ORDER}}
        ok = 0
        for r in rows:
            answers, _, _ = eng.decide(r["state"], {"domain": rq},
                                       {"energy_norm": True})
            a = answers["domain"]
            ok += (not a.get("deferred")
                   and a.get("choice") == r["labels"]["domain"])
        results.append(ok / len(rows))
        print(f"variant {level} ({['short4', 'mid8', 'rich12'][level]}): "
              f"router cal acc {ok/len(rows):.3f}")

    # variant 3: abstract register words (harvest-suppressed), no retrain
    rq = {"type": "choice", "instructions": "",
          "criteria": {rt_of[d]: ABSTRACT[d] for d in ORDER}}
    ok = 0
    for r in rows:
        answers, _, _ = eng.decide(r["state"], {"domain": rq},
                                   {"energy_norm": True})
        a = answers["domain"]
        ok += (not a.get("deferred") and a.get("choice") == r["labels"]["domain"])
    results.append(ok / len(rows))
    print(f"variant 3 (abstract): router cal acc {ok/len(rows):.3f}")

    best = max(range(4), key=lambda i: results[i])
    names = ["short4", "mid8", "rich12", "abstract"]
    print(f"winner: variant {best} ({names[best]})")
    if best == 3:
        winner = {rt_of[d]: ABSTRACT[d] for d in ORDER}
    else:
        winner = {rt_of[d]: VARIANTS[d][best] for d in ORDER}
    (GIANT / "router_criteria.json").write_text(json.dumps({
        "winner_level": best,
        "cal_acc": results,
        "criteria": winner,
        "note": "selected on cal subsample only; hidden untouched",
    }, indent=1))


if __name__ == "__main__":
    main()
