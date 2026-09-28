#!/usr/bin/env python3
"""Probe-condition experiment for the wrapper: router asked ALONE vs router +
ALL domain class questions in one decide (the Jev-style 'answer everything'
call). Measured on the same cal subsample as the wording sweep (never hidden).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402

G = ROOT.parent / "syfox-hf" / "datasets"
schemas = json.load(open(G / "schemas.json"))
D = schemas["domains"]
rt_of = schemas["router"]
eng = SyFoxEngine(str(ROOT.parent / "syfox-hf" / "model" / "giant-latin"))
eng.set_energy_norm(True)

rq = {"type": "choice", "instructions": "", "criteria": D["_router"]["criteria"]}
all_q = {"domain": rq}
for dom in sorted(D):
    if dom.startswith("_") or D[dom].get("type") == "noul_multi":
        continue
    all_q[f"q_{dom}"] = {"type": "choice", "instructions": "",
                         "criteria": D[dom]["criteria"]}

latin_cal = [json.loads(l) for l in open(G / "cal_latin.jsonl")]
rows = latin_cal[:: max(1, len(latin_cal) // 1500)][:1500]

import os
if os.environ.get("USE_HIDDEN"):   # blind-router hidden measurement
    rows = []
    for f in sorted(G.glob("eval/*_hidden.jsonl")):
        dom = f.name.replace("_hidden.jsonl", "")
        if dom in ("tickets_bn", "tickets_hi", "tickets_ru", "guard"):
            continue
        for l in open(f):
            r = json.loads(l)
            r["labels"]["__dom__"] = dom
            rows.append(r)
    print(f"hidden blind-router rows: {len(rows)}")
    ok = n = 0
    for r in rows:
        answers, _, _ = eng.decide(r["state"], {"domain": rq},
                                   {"energy_norm": True})
        a = answers["domain"]
        n += 1
        ok += (not a.get("deferred")
               and a.get("choice") == r["labels"]["domain"])
    print(f"hidden BLIND router acc: {ok/n:.3f}  (n={n})")
    raise SystemExit(0)

for cond in ("alone", "all"):
    ok = n = 0
    for r in rows:
        if cond == "alone":
            questions = {"domain": rq}
        else:
            questions = all_q
        answers, _, _ = eng.decide(r["state"], questions, {"energy_norm": True})
        a = answers["domain"]
        n += 1
        ok += (not a.get("deferred") and a.get("choice") == r["labels"]["domain"])
    print(f"{cond:6s}: router cal acc {ok/n:.3f}  (n={n})")

# class accuracy under the 'all' condition for the probe's own domain
ok = n = 0
for r in rows:
    dom = next(d for d in rt_of if rt_of[d] == r["labels"]["domain"])
    answers, _, _ = eng.decide(r["state"], all_q, {"energy_norm": True})
    a = answers.get(f"q_{dom}", {})
    if a.get("deferred") or not a:
        continue
    n += 1
    ok += a.get("choice") == r["labels"][next(k for k in r["labels"] if k != "domain")]
print(f"all-cond class acc (own domain): {ok/max(1,n):.3f}  (n={n})")
