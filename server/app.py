#!/usr/bin/env python3
# ============================================================================
#  SyFox HTTP bridge — a Jev-compatible System One API over the SI substrate.
#
#  Endpoints:
#    POST /v1/systemone   { "state": str|obj, "questions": { qid: {...} } }
#    GET  /v1/models      engine discovery
#
#  The Python layer is a TOOL: it speaks HTTP and JSON. Every decision is
#  made by the C++ SI substrate core (energy injection -> settle -> resonance
#  readout -> honest silence). No transformer, no classifier lives here.
# ============================================================================
import argparse
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from syfox_bridge import SyFoxEngine  # ctypes bridge to build/syfox_core

app = FastAPI(title="SyFox System One API", version="0.1.0")
ENGINE = None
LOCK = threading.Lock()


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "syfox-latest",
                "object": "model",
                "owned_by": "Mr-DS-ML-85",
                "core": "si-substrate",
                "description": "System One decision engine: state in, typed "
                               "decisions out. No text generation.",
            }
        ],
    }


@app.post("/v1/systemone")
async def systemone(req: Request):
    body = await req.json()
    state = body.get("state", "")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    questions = body.get("questions", {})
    if not isinstance(questions, dict) or not questions:
        return JSONResponse(status_code=400, content={
            "error": {"message": "questions must be a non-empty JSON object"}})

    # validate + normalize question schema (tool-side)
    norm = {}
    for qid, q in questions.items():
        if not isinstance(q, dict) or "type" not in q:
            return JSONResponse(status_code=400, content={
                "error": {"message": f"question '{qid}' needs a type: choice|score|noul"}})
        t = q["type"]
        norm[qid] = {
            "type": t,
            "instructions": str(q.get("instructions", "")),
        }
        if t in ("choice", "score"):
            if "criteria" not in q:
                return JSONResponse(status_code=400, content={
                    "error": {"message": f"question '{qid}' needs criteria"}})
            norm[qid]["criteria"] = q["criteria"]
        if t not in ("choice", "score", "noul"):
            return JSONResponse(status_code=400, content={
                "error": {"message": f"question '{qid}': unknown type '{t}'"}})

    with LOCK:  # the substrate field is single-threaded per engine instance
        answers, usage = ENGINE.decide(state, norm)

    return {
        "answers": answers,
        "usage": usage,
        "model": body.get("model", "syfox-latest"),
    }


def main():
    global ENGINE
    ap = argparse.ArgumentParser(description="SyFox System One server")
    ap.add_argument("--model", default="model-tickets", help="model directory")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    ENGINE = SyFoxEngine(args.model)
    print(f"SyFox serving on http://{args.host}:{args.port} "
          f"(model: {args.model}, core: si-substrate)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
