#!/usr/bin/env python3
# ============================================================================
#  SyFox HTTP bridge v3 — a Jev-compatible System One API over the SI substrate.
#
#  Endpoints:
#    POST   /v1/systemone   { "state": str|obj, "questions": {qid: {...}},
#                             "lang": "auto"|"<slug>",            (optional)
#                             "options": { ... CLI-equal knobs }  (optional) }
#    GET    /v1/models      model dir discovery (meta.json, no fabric load)
#    GET    /v1/health      core version, flags, uptime, deferral count
#    GET    /v1/deferrals   v2.2 active-learning: logged deferrals
#    DELETE /v1/deferrals   clear the deferral buffer
#
#  options (per request, CLI-equal semantics):
#    evidence         bool        M3: machine-auditable evidence JSON
#    energy_norm      bool        M1: measurement gain for corpus-scale fabrics
#    salience_gating  bool        SI selection mode
#    miller_window    bool        SI selection mode (Miller [cap-4,cap] window)
#    ngrams           "on"|"off"  v2.2 trigram lane policy
#
#  Server flags mirror the CLI: --lang off|auto|<slug>, --ngrams auto|on|off,
#  --energy-norm, --salience-gating, --miller-window, --threads N,
#  --log-deferrals FILE.
#
#  The Python layer is a TOOL: it speaks HTTP and JSON. Every decision is
#  made by the C++ SI substrate core (energy injection -> settle -> resonance
#  readout -> honest silence). No transformer, no classifier lives here.
# ============================================================================
import argparse
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from syfox_bridge import SyFoxEngine, core_version, detect_script, _LIB

app = FastAPI(title="SyFox System One API",
              version=f"{core_version()}.api3")

STATE = {
    "base_model": "model-tickets",
    "lang": "off",            # off | auto | <slug>
    "ngrams": "auto",         # auto | on | off  (auto = CLI routing policy)
    "log_deferrals": None,
    "started": time.time(),
}
ENGINES: dict[str, SyFoxEngine] = {}   # model_dir -> loaded engine cache
DEFERRALS: list[dict] = []             # in-memory ring (mirrored to file)
GLOBAL_LOCK = threading.Lock()         # serializes routing + decide (ngrams is
                                       # a process-global policy in the core)

SERVER_FLAGS: dict[str, bool] = {
    "energy_norm": False, "salience_gating": False,
    "miller_window": False, "parallel_settle": False, "threads": 0,
}


def _norm_questions(questions) -> tuple[dict | None, JSONResponse | None]:
    """Validate + normalize the question schema (tool-side, v0.1-compatible)."""
    if not isinstance(questions, dict) or not questions:
        return None, JSONResponse(status_code=400, content={
            "error": {"message": "questions must be a non-empty JSON object"}})
    norm = {}
    for qid, q in questions.items():
        if not isinstance(q, dict) or "type" not in q:
            return None, JSONResponse(status_code=400, content={
                "error": {"message": f"question '{qid}' needs a type: choice|score|noul"}})
        t = q["type"]
        if t not in ("choice", "score", "noul"):
            return None, JSONResponse(status_code=400, content={
                "error": {"message": f"question '{qid}': unknown type '{t}'"}})
        nq = {"type": t, "instructions": str(q.get("instructions", ""))}
        if t in ("choice", "score"):
            if "criteria" not in q:
                return None, JSONResponse(status_code=400, content={
                    "error": {"message": f"question '{qid}' needs criteria"}})
            nq["criteria"] = q["criteria"]
        norm[qid] = nq
    return norm, None


def _get_engine(model_dir: str) -> SyFoxEngine:
    eng = ENGINES.get(model_dir)
    if eng is None:
        eng = SyFoxEngine(model_dir)
        eng.set_energy_norm(SERVER_FLAGS["energy_norm"])
        eng.set_source_modes(SERVER_FLAGS["salience_gating"],
                             SERVER_FLAGS["miller_window"])
        eng.set_parallel_settle(SERVER_FLAGS["parallel_settle"],
                                SERVER_FLAGS["threads"])
        ENGINES[model_dir] = eng
    return eng


def _route(state: str, lang_req: str | None) -> tuple[str, str]:
    """CLI-equal routing: returns (model_dir, note).
    off  -> base model, no detection (v2.1 behavior)
    auto -> detect script; fall back to base with an honest note
    slug -> force the family; same fallback
    """
    lang = (lang_req or STATE["lang"] or "off").lower()
    if lang in ("", "off"):
        return STATE["base_model"], ""
    sc = detect_script(state)
    slug = lang if lang not in ("auto",) else (sc if sc != "unknown" else "")
    if not slug or slug == "unknown":
        return STATE["base_model"], "script=unknown; using base model"
    cand = f"{STATE['base_model']}-{slug}"
    if Path(cand).is_dir():
        return cand, f"routed to {cand} (script={slug})"
    return STATE["base_model"], f"no {cand} substrate; using base model (honest fallback)"


def _ngrams_policy(ngrams_req, script_slug: str) -> None:
    """v2.2 trigram policy: explicit wins; 'auto' = non-Latin routed -> on."""
    g = ngrams_req if ngrams_req in ("on", "off") else STATE["ngrams"]
    if g == "on":
        _LIB.syfox_engine_set_ngrams(None, 1)
    elif g == "off":
        _LIB.syfox_engine_set_ngrams(None, 0)
    else:  # auto: non-Latin routing is load-bearing, Latin keeps word stream
        _LIB.syfox_engine_set_ngrams(
            None, 1 if (script_slug and script_slug != "latin"
                        and STATE["lang"] != "off") else 0)


@app.get("/v1/models")
async def models():
    """Discover model dirs from meta.json (no fabric load)."""
    out = []
    for d in sorted(Path(".").glob("model-*")):
        if not d.is_dir():
            continue
        meta = {}
        mp = d / "meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {}
        out.append({
            "id": d.name,
            "object": "model",
            "owned_by": "Mr-DS-ML-85",
            "core": meta.get("core", "si-substrate"),
            "version": meta.get("version"),
            "nodes": meta.get("nodes"),
            "lanes": meta.get("lanes"),
            "evidence_records": meta.get("evidence_records"),
            "contradictions": meta.get("contradictions"),
            "loaded": d.name in {Path(k).name for k in ENGINES},
        })
    return {"object": "list", "core_version": core_version(), "data": out}


@app.get("/v1/health")
async def health():
    info = {}
    if ENGINES:
        try:
            info = ENGINES[STATE["base_model"]].info()
        except Exception:
            info = {}
    return {
        "status": "ok",
        "core_version": core_version(),
        "model": STATE["base_model"],
        "lang": STATE["lang"],
        "ngrams": STATE["ngrams"],
        "flags": SERVER_FLAGS,
        "fabric": info,
        "deferrals": len(DEFERRALS),
        "uptime_s": round(time.time() - STATE["started"], 1),
    }


@app.get("/v1/deferrals")
async def deferrals():
    """v2.2 active-learning loop, step 1: rows the fabric deferred on."""
    return {"object": "list", "count": len(DEFERRALS), "data": DEFERRALS}


@app.delete("/v1/deferrals")
async def deferrals_clear():
    n = len(DEFERRALS)
    DEFERRALS.clear()
    return {"cleared": n}


@app.post("/v1/systemone")
async def systemone(req: Request):
    body = await req.json()
    state = body.get("state", "")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    norm, err = _norm_questions(body.get("questions", {}))
    if err:
        return err
    opts = body.get("options", {}) or {}
    if not isinstance(opts, dict):
        return JSONResponse(status_code=400, content={
            "error": {"message": "options must be a JSON object"}})

    with GLOBAL_LOCK:  # field is single-threaded per engine; ngrams is global
        model_dir, note = _route(state, body.get("lang"))
        script_slug = ""
        lang_req = (body.get("lang") or STATE["lang"] or "off").lower()
        if lang_req not in ("", "off"):
            sc = detect_script(state)
            script_slug = lang_req if lang_req != "auto" else sc
        _ngrams_policy(opts.get("ngrams"), script_slug)

        try:
            eng = _get_engine(model_dir)
            answers, usage, evidence = eng.decide(state, norm, opts)
        except RuntimeError as e:
            return JSONResponse(status_code=422, content={
                "error": {"message": str(e)}})

        # options.defer_margin P: honest-uncertainty disclosure at the decision
        # layer (CLI-equal). The substrate still decided; a p1-p2 below P is
        # disclosed as a deferral (reason "low_margin") instead of a
        # confident-looking label on tied candidates.
        dm = opts.get("defer_margin")
        if isinstance(dm, (int, float)) and dm > 0:
            for a in answers.values():
                if a.get("deferred"):
                    continue
                vals = sorted(a.get("probabilities", {}).values(), reverse=True)
                if len(vals) >= 2 and vals[0] - vals[1] < dm:
                    a["deferred"] = True
                    a["reason"] = "low_margin"

        # v2.2 active-learning: log deferrals (state + full question schema)
        deferred = {qid: a for qid, a in answers.items() if a.get("deferred")}
        if deferred:
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "state": state, "questions": norm,
                   "reasons": {qid: a.get("reason", "silence") for qid, a in deferred.items()}}
            DEFERRALS.append(row)
            if STATE["log_deferrals"]:
                with open(STATE["log_deferrals"], "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = {"answers": answers, "usage": usage,
           "model": body.get("model", Path(STATE["base_model"]).name)}
    if note:
        out["lang_note"] = note
        out["routed_model"] = Path(model_dir).name
    if evidence is not None:
        out["evidence"] = evidence
    return out


def main():
    ap = argparse.ArgumentParser(description="SyFox System One server (v3 API)")
    ap.add_argument("--model", default="model-tickets", help="base model directory")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lang", default="off",
                    help="off | auto | <script-slug>  (v2.2 per-script routing)")
    ap.add_argument("--ngrams", default="auto", choices=["auto", "on", "off"],
                    help="trigram lane policy (v2.2; auto = CLI routing policy)")
    ap.add_argument("--energy-norm", action="store_true",
                    help="M1 decide-side measurement gain (corpus-scale fabrics)")
    ap.add_argument("--salience-gating", action="store_true",
                    help="SI salience selection mode (off by default)")
    ap.add_argument("--miller-window", action="store_true",
                    help="SI Miller [cap-4,cap] window mode (off by default)")
    ap.add_argument("--threads", type=int, default=0,
                    help="M5 OMP parallel settle (needs an OMP build: make omp)")
    ap.add_argument("--log-deferrals", default=None,
                    help="append every deferral to FILE.jsonl (active learning)")
    args = ap.parse_args()

    if not Path(args.model).is_dir():
        raise SystemExit(f"syfox: model dir not found: {args.model}")
    STATE["base_model"] = args.model
    STATE["lang"] = args.lang
    STATE["ngrams"] = args.ngrams
    STATE["log_deferrals"] = args.log_deferrals
    SERVER_FLAGS.update(energy_norm=args.energy_norm,
                        salience_gating=args.salience_gating,
                        miller_window=args.miller_window,
                        parallel_settle=args.threads > 0, threads=args.threads)

    _get_engine(args.model)  # load the base fabric up front
    print(f"SyFox v{core_version()} serving on http://{args.host}:{args.port} "
          f"(model: {args.model}, core: si-substrate, lang: {args.lang}, "
          f"ngrams: {args.ngrams})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
