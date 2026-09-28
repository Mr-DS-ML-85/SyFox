#!/usr/bin/env python3
"""Router route distribution across the 16 domains (v3.2.1 verification).

Stage-1 only: every row's state is settled on model-router16 with the 16
anchors from router.json; the winning anchor is tabulated per true domain.
Read-only evaluation of the shipped router (no selection, no training) —
same discipline as the v3.1.3 blind-router pass.

Usage: python3 tools/router_routes.py [--per-domain N] [--json-out F]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="model-router16")
    ap.add_argument("--router-json", default="data/router16_router.json")
    ap.add_argument("--giant-dir", default="data/giant")
    ap.add_argument("--per-domain", type=int, default=15)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    rj = json.load(open(a.router_json))
    anchors = rj["anchors"]                      # rt00..rt15 -> anchor text
    models = rj.get("models", {})
    eng = SyFoxEngine(a.router)
    q = {"route": {"type": "choice", "instructions":
                   "which domain does this state belong to",
                   "criteria": anchors}}

    files = sorted(Path(a.giant_dir).glob("*_hidden.jsonl"))
    table, confs = {}, {}
    for f in files:
        dom = f.name.replace("_hidden.jsonl", "")
        hits, cs = Counter(), []
        n = 0
        for line in open(f):
            if n >= a.per_domain:
                break
            r = json.loads(line)
            if not r.get("state"):
                continue
            ans_a, _u, _e = eng.decide(r["state"], q,
                                       {"semantics": True, "retrieval": True,
                                        "hierarchy": True})
            ans = ans_a.get("route", {})
            if not ans.get("deferred") and ans.get("choice"):
                hits[ans["choice"]] += 1
                cs.append(ans.get("confidence", 0.0))
            n += 1
        table[dom] = dict(hits.most_common(4))
        confs[dom] = round(sum(cs) / len(cs), 3) if cs else 0.0
        print(f"{dom:<10} n={n:<3} top={dict(hits.most_common(3))} meanC={confs[dom]}")

    # distribution shape: how concentrated is each domain's routing?
    total = sum(sum(v.values()) for v in table.values())
    top_agree = sum(next(iter(v.values())) for v in table.values() if v)
    print(f"\nrouted rows: {total}, routed-to-modal-anchor agreement: "
          f"{top_agree}/{total} = {top_agree/total:.3f}")
    if a.json_out:
        json.dump({"table": table, "mean_conf": confs},
                  open(a.json_out, "w"), indent=1)
        print(f"wrote {a.json_out}")

if __name__ == "__main__":
    main()
