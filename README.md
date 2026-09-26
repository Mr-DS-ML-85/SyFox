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
                │           + SI salience integrator (tanh decay/gain·motion;
                │           optional salience ranking + Miller [cap-4,cap] window)
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

## The SI salience lineage — and the two selection modes

SyFox's bounded working set is not an arbitrary batch size. The
Synthetic-Intelligence substrate it descends from runs two verified
mechanisms around the source cap, and both are ported here:

* **Cavity salience integrator** (SI `physics.hpp`,
  `s = tanh(s·0.995 + 0.05·|velocity|)`): every settle pass, each node
  integrates its own motion. A node at rest is an *exact* fixed point
  (`tanh(0) = 0`), so salience sparsely marks **what moved** — observable
  per concept via `node_salience()`. `inject()` spikes a touched node to
  the ceiling (TSDA spike semantics).
* **TSDA Miller window** (SI `salience.hpp`): the working cap is not fixed —
  SI samples it per decision from `[cap−4, cap]`, i.e. `[5,9]` at cap 9
  (Miller's 7±2). SyFox keeps the same lineage at its default cap 24:
  `--miller-window` draws the live cap from `[20,24]` each decision.

Selection itself stays **energy-gated by default** because that benchmarks
best on SyFox's narrow domains. The SI-faithful alternatives ship as CLI
flags, off by default:

```bash
./build/syfox decide --model model-tickets --miller-window --salience-gating \
  --state "You charged me twice for the same invoice, please refund" \
  --questions '{"department":{"type":"choice",...}}'
```

* `--miller-window` — live cap from `[source_cap−4, source_cap]` per
  decision. One deliberate adaptation: SI draws from a seeded `mt19937`
  stream; SyFox derives the draw from the decision's state fingerprint, so
  **the same state settles the same field, bit for bit**.
* `--salience-gating` — rank propagation sources by salience (motion
  history) instead of raw energy.

Benchmark (10 in-domain decisions × 3 seed domains): default energy gating
10/10 with calibrated-shaped confidences; salience ranking alone 4/10 at the
wide cap (motion history is a poor selector when the cap rarely trims);
salience + literal `[5,9]` 10/10 argmax but probabilities saturate at 1.0
(calibration collapse — verdict deferred to the v0.2 eval harness); Miller
`[20,24]` 9/10. Modes are runtime-only: `substrate.bin` is mode-neutral, so
the same saved model replays under any mode on any machine.

## The derivation layer — knowledge the field grows itself

The substrate derives answers through lane diffusion (see Known limits). The
derivation layer goes further: it lets the operator grow **new lanes** from
the field's own fabric — the kernel of what the upstream SI stack does with
explicit rule machinery, rebuilt here as physics (weight algebra + settle
dynamics; no classifiers). Four offline, explicit commands; `decide` stays
read-only:

```bash
./build/syfox derive  --model model-mine                        # compose two-hop evidence
./build/syfox derive  --model model-mine --examples data.jsonl  # harvest co-activation
./build/syfox dream   --model model-mine --steps 200            # emergent-resonance candidates
./build/syfox promote --model model-mine                        # apply human-validated lines only
./build/syfox analogs --model model-mine --concept refund       # structural analogs (read-only)
```

* **derive (compose)** — a two-hop path A→B→C is evidence for A↔C, damped by
  the same `1/√mass` law the field applies to energy, corroborated by at most
  the top-3 paths, and re-verified against the live fabric on every run.
* **derive (harvest)** — replay states and record what genuinely
  **co-activates** during settle; dissipation filters cross-context noise for
  free. Works on unlabelled text.
* **dream + promote** — the field is probed with random energy; undriven
  concepts that light up *without any direct lane* to the probe become
  candidate facts in a ledger. **A human must validate each line** before
  `promote` may lay it. Dreaming never modifies the substrate.
* **analogs** — concepts with the same structural role, ranked by novelty
  potential `Phi = I·exp(k·d)`; a transfer hypothesis, never auto-applied.

Every derived lane carries a generation counter (experience = 0), persisted
in `substrate.bin`; observed lanes can never be weakened by derivation.
Honest verdict on the 22-row seed models: derive sharpens well-separated
routing domains (billing 0.604→0.912 after harvest) but flips close-call
scenarios — **do not derive models trained on <100 rows** until the v0.2 eval
harness says otherwise. Details and benchmark: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §10.

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
* The substrate **derives, it doesn't retrieve**. A state you never taught
  still gets a computed answer: energy diffuses across the lane fabric and
  evidence composes across lessons. Verified in a two-lesson probe (only
  `alpha beta`→x and `beta gamma`→y taught): querying `alpha` alone — a
  state never taught, from a lesson that never mentioned y — puts y at
  0.494 through the two-hop path alpha→beta→y. What the field cannot exceed
  is the vocabulary and lane fabric of its experience: concepts you never
  taught don't exist in it, and unknown words defer (honest silence) instead
  of being hallucinated. Audit `model-*/substrate.bin` — the lanes ARE the
  knowledge, every one inspectable.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full mechanism
specification and the Synthetic-Intelligence lineage, and
[ROADMAP.md](ROADMAP.md) for where this is going.

## License & credit

MIT © 2026 Mr-DS-ML-85. Core mechanics ported from the author's
[Synthetic-Intelligence](https://github.com/Mr-DS-ML-85/Synthetic-Intelligence)
research substrate. Inspired by the *interface* of TypeSafe AI's Jev — not
affiliated with, nor derived from, any TypeSafe model or weights.
