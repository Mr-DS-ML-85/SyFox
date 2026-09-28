#!/usr/bin/env python3
"""Giant-model manual benchmark: per-domain hidden scoring + router +
end-to-end + handcrafted probes, through the ctypes bridge.

Everything is measured with energy_norm ON (mandatory at corpus scale —
v3.1.0 finding). Router accuracy: the 'domain' question asked on every
hidden row. End-to-end: router correct AND class correct in one decide call
(the bridge returns answers for every question in one shot).

Usage:
  python3 tools/giant_bench.py --model-slug giant-latin --domains \
      tickets_en,bank77,sms,clinc,... [--limit N] [--dump out.jsonl]
  python3 tools/giant_bench.py --probes --model-slug giant-latin
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from syfox_bridge import SyFoxEngine  # noqa: E402

GIANT = Path(os.environ.get("GIANT_DIR", str(ROOT / "data" / "giant")))


def rank_of(gold: str, probs: dict) -> int:
    order = sorted(probs.items(), key=lambda kv: -kv[1])
    for i, (k, _) in enumerate(order, 1):
        if k == gold:
            return i
    return len(order) + 1


def load_schemas() -> dict:
    return json.load(open(GIANT / "schemas.json"))


def build_questions(r: dict, schemas: dict) -> dict:
    """Full-schema rows pass through; schema-free rows (state/domain/labels,
    as shipped in the HF repo) get their questions rebuilt from schemas."""
    if "questions" in r:
        return r["questions"]
    dom = r["domain"]
    spec = schemas["domains"][dom]
    if spec.get("type") == "noul_multi":
        return r["questions"]          # noul keeps its questions inline
    rq = {"type": "choice", "instructions": "",
          "criteria": schemas["domains"]["_router"]["criteria"]}
    cq = {"type": "choice", "instructions": "", "criteria": spec["criteria"]}
    return {"domain": rq, spec["qkey"]: cq}


def hidden_path(dom: str) -> Path:
    for cand in (GIANT / f"{dom}_hidden.jsonl", GIANT / "eval" / f"{dom}_hidden.jsonl"):
        if cand.exists():
            return cand
    return GIANT / f"{dom}_hidden.jsonl"


def run_domain(eng: SyFoxEngine, dom: str, limit: int, dump: list,
               schemas: dict) -> dict:
    rows = [json.loads(l) for l in open(hidden_path(dom))]
    if limit:
        rows = rows[:limit]
    rows = [{**r, "questions": build_questions(r, schemas)} for r in rows]
    qkeys = [k for k in rows[0]["questions"] if k != "domain"]
    qkey = qkeys[0]
    rt_ok = cl_ok = e2e_ok = defer = 0
    n = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        answers, usage, _ = eng.decide(r["state"], r["questions"],
                                       {"energy_norm": True})
        n += 1
        da = answers.get("domain", {})
        if da.get("deferred"):
            defer += 1
            continue
        rt_hit = da.get("choice") == r["labels"]["domain"]
        rt_ok += rt_hit
        ca = answers.get(qkey, {})
        gold = r["labels"].get(qkey)
        if ca.get("deferred"):
            continue
        if gold == "true" or gold == "false":
            cl_hit = (ca.get("choice") == gold)
        else:
            cl_hit = (ca.get("choice") == gold)
        cl_ok += cl_hit
        e2e_ok += (rt_hit and cl_hit)
        if dump is not None:
            probs = ca.get("probabilities", {})
            top = sorted(probs.items(), key=lambda kv: -kv[1])
            dump.append(json.dumps({
                "domain": dom, "i": i, "gold_router": r["labels"]["domain"],
                "pred_router": da.get("choice"),
                "gold": gold, "pred": ca.get("choice"),
                "correct": bool(cl_hit), "conf": ca.get("confidence", 0),
                "gold_rank": rank_of(gold, probs) if probs else 0,
                "margin": round(top[0][1] - (top[1][1] if len(top) > 1 else 0),
                                5) if top else 0,
            }, ensure_ascii=False))
        if (i + 1) % 1000 == 0:
            print(f"  {dom}: {i + 1}/{len(rows)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    secs = time.time() - t0
    return {"domain": dom, "n": n, "router_acc": rt_ok / n,
            "class_acc": cl_ok / max(1, n - defer),
            "e2e_acc": e2e_ok / n, "deferred": defer, "secs": round(secs, 1),
            "ms_per_row": round(1000 * secs / max(1, n), 2),
            "qkey": qkey,
            "note": "noul_multi" if qkey == "irreversible" else ""}


def run_probes(model_slug: str) -> None:
    eng = SyFoxEngine(f"model-{model_slug}")
    schemas = json.load(open(GIANT / "schemas.json"))
    inv = {d: {v: k for k, v in schemas["domains"][d]["classes"].items()}
           for d in schemas["domains"]
           if schemas["domains"][d].get("classes")}
    rows = [json.loads(l) for l in open(GIANT / "probes.jsonl")]
    hits_r = hits_c = scored = 0
    print(f"{'id':5s} {'domain':11s} {'router':7s} {'class':7s} "
          f"{'conf':>5s}  state")
    for p in rows:
        answers, _, _ = eng.decide(p["state"], p["questions"],
                                   {"energy_norm": True})
        da = answers.get("domain", {})
        dom = p["domain"]
        qkey = schemas["domains"][dom]["qkey"]
        ca = answers.get(qkey, {})
        r_ok = (not da.get("deferred")
                and da.get("choice") == p["expected_router"])
        hits_r += r_ok
        line = f"{p['id']:5s} {dom:11s} {'ok' if r_ok else 'MISS':7s}"
        if p["expected_class"] is None:
            d = ca.get("deferred", da.get("deferred", False))
            line += (f" {'defer' if d else 'ANSWERED':7s} "
                     f"{ca.get('confidence', 0):5.3f}  {p['state'][:48]}")
        else:
            scored += 1
            c_ok = (not ca.get("deferred")
                    and ca.get("choice") == p["expected_class"])
            hits_c += c_ok
            name = schemas["domains"][dom]["classes"].get(
                ca.get("choice", ""), ca.get("choice", "?"))
            line += (f" {'ok' if c_ok else 'MISS':7s} "
                     f"{ca.get('confidence', 0):5.3f}  -> {name[:26]}")
        print(line, flush=True)
    print(f"\nrouter {hits_r}/{len(rows)} = {hits_r/len(rows):.3f}   "
          f"class {hits_c}/{scored} = {hits_c/max(1, scored):.3f}   "
          f"(OOD probes: answered-instead-of-deferred counted as MISS on router)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", default="giant-latin",
                    help="model dir suffix, e.g. giant-latin")
    ap.add_argument("--domains",
                    default="tickets_en,game,guard,bank77,sms,clinc,enron,"
                            "emotion,hate,agnews,polarity,dbpedia,boolq")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    if args.probes:
        run_probes(args.model_slug)
        return

    eng = SyFoxEngine(f"model-{args.model_slug}")
    info = eng.info()
    schemas = load_schemas()
    print(f"model-{args.model_slug}: {info.get('nodes')} nodes / "
          f"{info.get('lanes')} lanes, energy_norm on")
    dump = [] if args.dump else None
    results = []
    for dom in args.domains.split(","):
        dom = dom.strip()
        f = hidden_path(dom)
        if not f.exists():
            print(f"skip {dom} (no hidden file)")
            continue
        r = run_domain(eng, dom, args.limit, dump, schemas)
        results.append(r)
        print(f"{r['domain']:11s} n={r['n']:5d} router={r['router_acc']:.3f} "
              f"class={r['class_acc']:.3f} e2e={r['e2e_acc']:.3f} "
              f"defer={r['deferred']:4d} ({r['ms_per_row']} ms/row)"
              f"{'  [' + r['note'] + ']' if r['note'] else ''}", flush=True)

    tot_n = sum(r["n"] for r in results)
    if tot_n:
        print(f"\nTOTAL rows={tot_n} "
              f"router={sum(r['router_acc']*r['n'] for r in results)/tot_n:.3f} "
              f"class={sum(r['class_acc']*(r['n']-r['deferred']) for r in results)/tot_n:.3f} "
              f"e2e={sum(r['e2e_acc']*r['n'] for r in results)/tot_n:.3f}")
    if args.dump:
        out = Path(args.dump)
        out.write_text("\n".join(dump) + "\n")
        print(f"per-row dump -> {out} ({len(dump)} rows)")
        (GIANT / "bench_hidden_summary.json").write_text(
            json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
