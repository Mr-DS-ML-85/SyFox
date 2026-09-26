# 🦊 SyFox — the open System One decision engine

**State in. Typed decisions out. No text generation. Ever.**

SyFox is an open, dependency-free decision engine in the spirit of TypeSafe AI's
[Jev](https://www.firecrawl.dev/blog/what-is-jev) — but with a fundamentally
different core: instead of a trained transformer, SyFox runs on the
**Synthetic-Intelligence physics substrate** — an energy field where decisions
*collapse out of dynamics*, not out of token probabilities.

> Core = SI physics only. No transformer, no neural network, no
> pattern-matching classifier. Tools around the core (JSON, CLI, HTTP) are
> ordinary engineering.

```
state tokens ──inject──> ConceptField (nodes with acoustic mass)
                │
                ├── settle: dissipative diffusion along Hebbian lanes,
                │           energy-gated source cap (bounded working set), K_settle
                │
                ├── per-question READOUT on the same settled field:
                │     Choice → specific resonance  → typed option + probabilities
                │     Score  → specific resonance  → weighted level + confidence
                │     Noul   → support resonance   → probability 0..1
                │
                ├── Hebbian learn: labelled lessons rewire lanes
                │   (fire together → wire together; counter-evidence dissolves)
                └── honest silence: nothing settled? → defer, never guess
```

## Why this matters

| | LLM (System 2) | Jev | **SyFox** |
|---|---|---|---|
| Output | prose to parse | typed values | **typed values** |
| Compute | token by token | one parallel pass | **one settle pass** |
| Can hallucinate text | yes | no (schema-bound) | **no (schema-bound + honest silence)** |
| Core | transformer | trained foundation model | **energy-field physics (SI)** |
| Weights | billions, opaque | opaque | **your lanes, inspectable, yours** |
| Runs offline | maybe | no | **yes, zero dependencies** |
| Cost per decision | $$ | $0.000000042 | **~0 (CPU microseconds)** |

SyFox will never generalize like Jev — a trained foundation model sees billions
of examples; SyFox sees *your* examples, and its knowledge is exactly what you
taught it, inspectable lane by lane. That is the trade: **narrow but fully
ownable, auditable, offline, and free.**

## Quickstart

```bash
make            # builds build/syfox, build/syfox-test, build/libsyfox_core.so
make test       # unit tests (physics, learning, honest silence)
make models     # trains the three seed domains from data/*.jsonl
make demo       # tickets + game bot + coding-agent guardrail
```

Decide from the CLI:

```bash
./build/syfox decide --model model-tickets \
  --state "You charged me twice for the same invoice, please refund" \
  --questions '{
    "department": {"type":"choice","instructions":"Which team","criteria":
      {"billing":"payment or subscription issues","technical":"bugs","sales":"pricing"}},
    "is_urgent": {"type":"noul","instructions":"the message conveys urgency or time sensitivity"}
  }'
```

```json
{"answers":{
  "department":{"choice":"billing","probabilities":{"billing":0.609,...},"confidence":0.204,"deferred":false},
  "is_urgent":{"noul":0.177,"confidence":0.646,"deferred":false}},
 "usage":{"core":"si-substrate","settled_energy":0.83,"lanes":5786}}
```

Serve it (Jev-compatible call shape, `POST /v1/systemone`):

```bash
make serve      # http://127.0.0.1:8010  (needs: pip install fastapi uvicorn)
```

```bash
curl -X POST localhost:8010/v1/systemone -H 'Content-Type: application/json' \
  -d @examples/support_ticket.json
```

## The three question primitives

| Type | Asks | Returns |
|---|---|---|
| `choice` | which of these options? | option + probability per option + confidence |
| `score` | where on this rubric? | weighted level value + per-level probabilities + confidence |
| `noul` | is this true? | probability 0..1 + confidence |

Every question is evaluated **in isolation against the same settled field** —
adding questions does not re-settle the substrate (flat marginal latency,
inherited structurally from the one-pass design).

## Train your own domain

Training = teaching the substrate with labelled lessons (JSONL):

```json
{"state":"...","questions":{...schema...},"labels":{"department":"billing","is_urgent":"false"}}
```

```bash
./build/syfox learn     --model model-mine --examples my_data.jsonl
./build/syfox calibrate --model model-mine --examples my_data.jsonl
```

* `learn` — Hebbian lane binding: state concepts ↔ outcome concepts wire
  together; for `noul` false-labels the same routes actively dissolve
  (anti-Hebbian), so the field discriminates instead of accumulating.
* `calibrate` — a tool-side post-processor fits temperature (choice/score) and
  Platt scaling (noul) on readout energies so probabilities mean what they say.
  The core physics is untouched.

## The honest-silence contract

If the field settles below the silence floor — unknown vocabulary, empty
state, nothing corroborated — every question returns:

```json
{"deferred": true, "reason": "unknown_vocabulary"}
```

SyFox never guesses from nothing. For guardrails, combine this with a defer
band: route answers with confidence below your threshold to a human.

## Known limits (v0.1 — honest list)

* Tone/sentiment questions need much more data than routing questions; with 22
  seed rows the argmax is usually right but probabilities stay near-uniform.
* Word-level substrate: heavy typos fall outside the vocabulary (character-n
  gram lanes are on the roadmap).
* English/Bengali-style token folding is deliberately minimal (deterministic
  string normalization, no NLP pipeline).
* Every answer is only as good as the lessons you taught — the substrate
  cannot exceed its data. Audit `model-*/substrate.bin` before trusting it.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full mechanism
specification and the Synthetic-Intelligence lineage, and
[ROADMAP.md](ROADMAP.md) for where this is going.

## License & credit

MIT © 2026 Mr-DS-ML-85. Core mechanics ported from the author's
[Synthetic-Intelligence](https://github.com/Mr-DS-ML-85/Synthetic-Intelligence)
research substrate. Inspired by the *interface* of TypeSafe AI's Jev — not
affiliated with, nor derived from, any TypeSafe model or weights.
