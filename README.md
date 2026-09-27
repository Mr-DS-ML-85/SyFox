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

The v3 server exposes the same decide-side surface as the CLI. Server flags
mirror the CLI (`--lang auto|<slug>`, `--ngrams on|off|auto`, `--energy-norm`,
`--salience-gating`, `--miller-window`, `--threads N`, `--log-deferrals FILE`);
per-request knobs ride in the body (`"lang"`, `"options": {"evidence": true, ...}`):

| Endpoint | Purpose |
|---|---|
| `POST /v1/systemone` | decide (Jev-compatible shape; optional `options.evidence` → M3 evidence JSON, `lang` → v2.2 per-script routing with honest fallback notes) |
| `GET /v1/models` | model dir discovery from `meta.json` — nodes, lanes, evidence records, contradiction count (no fabric load) |
| `GET /v1/health` | core version, active flags, fabric stats, uptime |
| `GET /v1/deferrals` | v2.2 active-learning loop, step 1: everything the fabric deferred on (also mirrored to `--log-deferrals` file) |

```bash
# M3 evidence over HTTP: supporting lanes + provenance + contradictions
curl -X POST localhost:8010/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":"Refund the duplicate charge","options":{"evidence":true},
       "questions":{"department":{"type":"choice","instructions":"team",
         "criteria":{"billing":"payments","technical":"bugs"}}}}'
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

## The no-regression gate — derivation that cannot lie

v0.1's honest verdict was: derive sharpens separated domains but flips
close-call scenarios on small fabrics, so models shipped un-derived. v0.2
turns that verdict into a **mechanism**: `syfox derive --gate FILE.jsonl` is
transactional.

```bash
./build/syfox derive --model model-mine --gate data.jsonl                      # compose, gated
./build/syfox derive --model model-mine --examples data.jsonl --gate data.jsonl # harvest, gated
```

The gate replays every labelled row plus generated close-call probes
(cross-domain state mixtures) before and after the derivation, snapshots the
fabric bit-exactly (per-node lane order included — settle() sums in lane
order), and applies two rules refined by a full strength-scan experiment:

1. **taught rows never flip** — a derive that breaks a lesson is destroying
   knowledge; any argmax flip on labelled rows reverts the fabric to the
   snapshot and the model on disk is untouched;
2. **no manufactured certainty** — close-call probes are ambiguous by
   construction, so their argmax may re-resolve, but if the mean confidence
   on them RISES the fabric is feigning sureness about coin flips → revert.

Per-model verdict under the v2.1 held-out gate (`make heldout-gate`, P5 — the
gate rows are now the HELD-OUT split, so the verdict measures generalization
safety, not memorized behavior; gold-labelled accuracy before/after is in the
report):

| model | verdict | heldout flips | reason / outcome |
|---|---|---|---|
| tickets | **REVERT** | 9 | fabric restored bit-for-bit; ships un-derived |
| game | **PASS** | 0 | 253 derived lanes committed; heldout accuracy 0.667 → 0.667 |
| guard | **REVERT** | 1 | fabric restored bit-for-bit; ships un-derived |

Ungated `derive` still works and prints a warning; the gate is the safe path.

## The Jev-parity benchmark

`syfox bench` measures the axes the System One model class is judged on —
same evaluation type as TypeSafe's published Jev numbers and the community
classification benchmarks (accuracy, latency, calibration, confidence,
honesty, guardrail precision):

| axis | Jev published reference | `syfox bench` measures |
|---|---|---|
| decision accuracy | 67.8% on a 4-workflow suite (≈ GPT-5.6 Terra 67.9%) | choice/score argmax on eval rows |
| latency | 70–500 ms one-pass | p50/p95 µs per decision |
| calibration | "calibrated Bayesian confidence" (RLCD) | ECE (10 bins, top-prob) + conf gap |
| honesty | "never hallucinates" (schema-bound) | OOD defer rate (untaught vocabulary must defer) |
| guardrails | pi-warden 88% hold precision | noul hold precision/recall + confusion |
| determinism | non-autoregressive one-pass | full replay, byte-identical decision signatures |

### Evaluation methodology v2.1 — held-out, not resubstitution

Every data domain is split **70/30 (train / held-out), stratified by label**,
deterministic seed (`tools/split_data.py`). The fabric learns only from
`*_train.jsonl`; `*_heldout.jsonl` is touched exactly twice: the temperature/
Platt fit sets 2–3 scalars on it (P6 — monotone, so argmax is unaffected),
and the derivation gate replays it read-only (P5). **The headline numbers
below are held-out accuracy, not resubstitution.** In-sample numbers are kept
only as a regression contrast (`make bench`).

`make bench-heldout` (held-out split; the derivation state is post-gate —
game derived, tickets/guard un-derived):

| model | held-out choice acc (n) | score acc (n) | choice ECE | OOD defer | p50 latency | deterministic |
|---|---|---|---|---|---|---|
| tickets | **1.000** (6) | 0.667 (6) | **0.012** | 1.00 | 64 µs | yes |
| game (derived) | 0.667 (3) | — | 0.329 | 1.00 | 29 µs | yes |
| guard | **0.750** (4) | — | 0.200 | 1.00 | 34 µs | yes |

**Held-out accuracy, not resubstitution.** The v0.2 README reported
1.000 / 1.000 / 0.846 — that was the model grading its own homework (the
eval rows were the training rows). The honest held-out numbers are above;
in-sample contrast (`make bench`): tickets 1.000 / game 1.000 / guard 0.778.
The gap between the two is the actual generalization measure, and it is now
measured, printed, and impossible to confuse.

### Coverage vs accuracy — the headline metric

Top-1 accuracy at 100% coverage is the worst corner of the trade-off the
field actually operates in. `syfox bench --split heldout --coverage-curve`
sweeps a confidence threshold τ over 0.0→1.0 and reports accuracy *within*
each coverage level (`make coverage` reproduces; ASCII plot included in the
output). Tickets, held-out, 12 labelled choice/score questions:

| coverage ≥ | τ | coverage | accuracy within |
|---|---|---|---|
| 100% | 0.45 | 100% | 83.3% |
| 90% | 0.45 | 100% | 83.3% |
| 70% | 0.55 | 83.3% | 90.0% |
| 50% | 0.95 | 50% | **100%** |

Read that last row the SI way: when the field is confident, it is *right* —
the substrate's honest silence and the calibration together give a selective
predictor whose emitted answers are trustworthy, and the curve tells you
exactly what each confidence threshold buys. Game/guard curves are flat at
their 0.667–0.75 held-out level (3–4 questions, no headroom — more held-out
data would give the curve room to separate).

### Calibration on held-out (P6)

Temperature + Platt are fitted on the held-out split — **2–3 scalars only;
the fabric itself never learns from held-out rows**, and temperature is
monotone so argmax decisions are untouched. The fit objective is multi-class
NLL over the full candidate energy vector (the textbook definition), and a
1-bit adoption guard keeps T = 1 when the fitted T would worsen held-out ECE:

| model | ECE before (T=1) | ECE after | adopted |
|---|---|---|---|
| tickets | 0.665 | **0.012** | T = 0.0005 |
| game | 0.329 | 0.329 | fit rejected (would worsen to 0.503) |
| guard | 0.496 | 0.200 | T = 0.0107 |

Tickets meets the calibrated-confidence bar (ECE < 0.1). Game and guard
cannot cross it at full coverage **with their current single held-out error**:
with 1 wrong answer among 3–4 emitted, ECE ≥ (1/n)·|0 − conf| is
mathematically ≥ 0.2 no matter the temperature — the remedy is more held-out
rows and fewer errors, which `make coverage` now measures directly. This is
the same discipline as the derivation gate: the harness decides adoption, not
hope.

### Augmentation dose-response (P3, measured)

`tools/gen_paraphrases.py` generates 5 paraphrased lessons per taught state
(`data/*_paraphrased.jsonl`: clause reorders, neutral fillers, combos —
never touching held-out files). The held-out harness then **measured the
augmentation backfiring**: the substrate counts occurrences linearly
(`intern()` mass) and damps re-exposed tokens by 1/√mass, so re-teaching
near-duplicate lessons re-weights the field toward the training surface
forms. Held-out choice accuracy:

| recipe | tickets | game | guard |
|---|---|---|---|
| train only (shipped default) | **1.000** | 0.667 | **0.750** |
| train + 5 paraphrases/state | 0.500 | 0.667 | 0.667 |
| train + 1 paraphrase/state | 0.500 | — | — |
| train doubled (pure repeat) | 0.667 | — | — |

A lesson in this substrate is a physical deposition, not a data point: the
same experience re-deposits mass and *damps* the pathway. So the shipped
default builds un-augmented (`make models`), the augmented recipe stays
reproducible (`make models-augmented`), and the synonym-folding table
(`data/synonyms.txt` + Porter stemming, applied at the token boundary) gives
the generalization P3 was after — "reimbursement" lands on the same node as
"refund" — without re-deposition.

### Variant lessons, mass-guarded (v2.2, measured — still negative)

v2.2 asked the obvious follow-up: was the v2.1 regression ONLY the mass
re-deposition? `tools/gen_variants.py` now generates 8–10 variants per
taught state (synonym swaps, clause rotation, adjacent reorder, fillers;
deterministic, refuses held-out paths), and `learn --augment` re-teaches
them **mass-guarded** — concepts already in the vocabulary are not
`intern()`ed again, so acoustic mass never changes; only lanes are laid and
strengthened. Held-out choice accuracy (`make models-vast`):

| recipe | tickets | game | guard |
|---|---|---|---|
| train only | **1.000** | 0.667 | **0.750** |
| train + 199 mass-guarded variants | 0.500 | 0.000 | 0.500 |

Still negative — worse than plain re-teaching on game. The mechanism is now
measurable with the mass confound removed: variant lessons lay NEW lanes
from the same words to MULTIPLE labels (the seed domains share vocabulary
across labels — "refund" appears under billing AND sales), and reorder
fillers bind noise nodes to everything. On tiny multi-label fabrics the
fabric smears. **The physics verdict is unchanged: coverage comes from
distinct labelled experience, not from re-phrasing 16 rows** — which is
exactly what the active-learning loop (below) collects. `make models-vast`
stays reproducible so this A/B can be re-run as fabrics grow.

### Multilingual boundary and typo robustness (v2.2, measured)

**The language barrier is real and was at the tokenizer**: v2.1 scanned bytes
with `std::isalnum`, which in the C locale tears apart every multi-byte UTF-8
sequence — a Bengali or Hindi query produced **zero tokens** and deferred
forever. v2.2 rebuilds the boundary (`core/script.hpp` + `core/normalize.hpp`
+ `core/ngram.hpp`):

1. **Script detection** — 35 Unicode script families (Latin, Cyrillic, Greek,
   Arabic, Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
   Kannada, Malayalam, Sinhala, Thai, Lao, Tibetan, Myanmar, Georgian, Khmer,
   Mongolian, Ethiopic, Cherokee, Coptic, Vai, Yi, Bopomofo, Han, Kana,
   Hangul, Hebrew, Armenian, Syriac, Thaana, Nko) by block-range majority
   vote. Deterministic: ties break by table order, digits are neutral,
   malformed bytes are separators. These families carry **100+ languages**
   (Latin alone 60+: English, Spanish, French, German, Portuguese, Vietnamese,
   Turkish, Swahili, Indonesian…; Cyrillic ~15: Russian, Ukrainian, Bulgarian,
   Kazakh…; Arabic ~10: Arabic, Persian, Urdu, Pashto…; Devanagari ~8: Hindi,
   Marathi, Nepali…; Bengali, Punjabi, Gujarati, Odia, Tamil, Telugu, Kannada,
   Malayalam, Sinhala, Thai, Lao, Khmer, Burmese, Tibetan, Georgian, Armenian,
   Greek, Hebrew, Amharic, Tigrinya, Chinese, Japanese, Korean, Dhivehi,
   Mongolian, Cherokee, Coptic, Vai, Nuosu…).
2. **Per-script substrates** — `--lang auto` detects the script of every
   query/lesson and routes to `<model>-<script>`: one SI substrate per script
   family (Latin included). A routed substrate is isolated fabric — no
   cross-script interference. Missing substrate → honest fallback note.
3. **Character trigram lanes** — taught words deposit `g3:ref`-style
   sub-word lanes (literal strings, not hashes — every lane inspectable).
   At decide time, an UNKNOWN word injects its trigrams **only if ≥25% of
   them already exist in the fabric** (the signature of a corrupted form:
   `refnd` keeps `g3:ref` of `refund`). A legitimate unseen word shares
   nothing and stays dark — honest silence, no manufactured evidence.
   Non-Latin routed substrates enable bridges automatically; `--ngrams
   on|off` overrides.

Measured (all reproducible; `make models` teaches the trigram lanes, the
default decide path stays word-level):

| measurement | result |
|---|---|
| Bengali tokens (v2.1) | **0 tokens** — every non-Latin query deferred |
| Bengali tokens (v2.2) | full tokenization, script detected |
| Bengali held-out routing (n=3, `model-tickets-bengali`) | choice acc **1.000**, OOD defer 1.00 |
| Bengali train routing (n=11, in-sample) | choice acc 1.000, score acc 1.000 |
| Hindi (Devanagari) + Russian (Cyrillic) demos | routed to `-devanagari` / `-cyrillic`, correct billing choice |
| Latin held-out with lanes in fabric, word-level decide | tickets 1.000 / game 0.667 / guard 0.750 — **byte-identical to v2.1** |
| Latin held-out, bridges on (`--ngrams on`) | same clean numbers (traction gate) |
| Deterministic corruption 40% (`bench --typos 40`) | tickets clean 1.000 → 0.500 (bridges neutral at n=6; the sweep is deterministic and scales with fabric) |

### Active learning loop (v2.2) — your fine-tuning

A substrate can only know what it was taught, so the growth path is the
loop — deploy, log deferrals, label, re-learn:

```bash
./build/syfox decide --model model-tickets --log-deferrals logs/deferrals.jsonl \
    --state "..." --questions '{...}'        # step 1: honest silence leaves a trail
./build/syfox active --deferrals logs/deferrals.jsonl --out worksheet.jsonl
                                             # step 2: dedup + rank by deferral count
# step 3: a human fills "labels" in the worksheet (schema already embedded)
./build/syfox learn --model model-tickets --examples worksheet.jsonl
                                             # step 4: re-teach
```

Each deferral log row carries the state AND the full question schema, so a
labeled worksheet row is directly teachable — no reconstruction step. This
is the mechanism answer to "train it vastly": the field grows by **distinct
labelled experience collected from its own uncertainty**, not by re-phrasing
the same 16 rows (measured twice — see the dose-response sections above).

### Latency and cost

Measured p50 per decision: 29–64 µs on a single CPU core (p95 ≤ 80 µs) —
three to four orders of magnitude under the Jev reference band (70–500 ms
per call), with the same one-pass, non-autoregressive shape and byte-identical
replay determinism. Cost per decision is electricity: the engine runs
offline on CPU with no API call, no token billing, and no per-request
infrastructure; memory footprint is a binary model directory
(substrate.bin + calibration.json), not a GPU-resident network.

`jev_reference` in each report carries the published figures for side-by-side
reading; no parity is claimed beyond sharing the axes.

## v3 milestones — scale, distinct experience, provenance, adversarial, multicore

All numbers below come from real runs on this repo (targets in the Makefile,
raw reports in `build/`, corpus manifest with SHA-256 per file in
`data/big/MANIFEST.json`). Generator: `tools/gen_corpus.py`, seed 20260927.

### Hidden-test discipline, enforced in code (M1)

`core/firewall.hpp` refuses the splits that must never steer a build:
`learn` rejects `_hidden`/`_cal` files, `calibrate` rejects `_hidden`,
`derive --gate` rejects `_hidden`. `make firewall-check` proves all four
refusals on the live CLI. The corpus (`make models-big`): tickets_en 12,000
rows, game 8,000, guard 8,000 — each split 70/15/15 into
train/calibration/hidden, calibration fitting 2–3 scalars on `_cal` only,
hidden scored once by bench.

Hidden-test results (`make bench-big`, `--energy-norm`):

| domain | hidden accuracy | choice ECE | note |
|---|---|---|---|
| tickets_en | **0.971** (n=1800) | 0.005 | p95 441 µs |
| game | **0.763** (n=1200) | 0.049 | 3-way game verbs |
| guard | **0.638** (n=1200) | 0.037 | hold precision 0.782, recall 0.910 |

Multilingual hidden splits ≥500 rows each (`make bench-big-ml`):
**Bengali 0.969** (n=540, ECE 0.015), **Hindi 0.965** (n=539, ECE 0.020),
**Russian 0.970** (n=539, ECE 0.011) — routed to per-script substrates.

Coverage-vs-accuracy on hidden (`make coverage-big`), all operating points
feasible: tickets 96.4%@53% / 90.1%@73% / 83.0%@93%; game 94.3%@52% /
86.7%@76% / 80.2%@91%; guard 76.2%@54% / 68.1%@80% / 65.8%@90%.

### Distinct experience > repeated phrasings — the M2 A/B (measured)

At EQUAL lesson counts (25,200 each; `make ab-distinct`):

| arm | rows | lessons | hidden acc | lanes | nodes |
|---|---|---|---|---|---|
| A distinct | 8,400 | 25,200 | **0.825** | 39,158 | 1,898 |
| B repeated ×8 | 1,050 | 25,200 | 0.782 | 18,633 | 530 |
| B + `--dedup` | 1,050 | 3,150 | 0.609 | 18,736 | 530 |

Repetition adds almost no lanes (same words, same bridges) — distinct
experience builds a 2.1× denser fabric and wins by +4.3 pts. B-dedup skips
exactly 22,050 duplicate lessons (`dedup_skipped` in the report) and its
accuracy matches the dose-response curve at ~1,000 rows — two independent
measurements agreeing. `--novelty` weighting measured neutral (0.827 vs
0.825 at floor 0.90) and ships OFF; dedup is the mechanism that pays.

### Evidence and contradictions (M3)

`decide --evidence` prints a machine-auditable ledger for every channel:
each supporting lane with weight, generation, support/counter events,
first/last teach seq, and the substrate that owns it. Example on the big
tickets model — `department` resolved from 62 supporting lanes, top lane
`the -> problem` (w=0.2896, 10,870 support events, seq window [1, 25198]).

Contradictions cannot silently override: re-teaching a state with a
different outcome surfaces `contradictions: N` in the learn report, records
a contested flag with the old/new outcomes and both seqs, marks counter
lanes (anti-Hebbian dissolution), and the field keeps whichever binding the
PHYSICS supports — verified live: teach `a` ×3 then `b` ×3, decide still
answers `a` with the dispute on the record. The ledger is read-only; lane
weights drive the field, the ledger never does.

### Adversarial / OOD suite (M4)

`bench --adversarial` (`make adv-big`) runs deterministic perturbation
families over the hidden tests and reports accuracy, defer rate,
false-confidence and confidence-when-wrong per family. On tickets:
reorder 0.971, padding 0.968, typos 0.968, intensifiers 0.976,
self-contradiction 0.969, negation 0.974, **double-negation 0.942
(weakest family — known limit)**, near-miss 0.933, cross-domain 0.908 —
and **unknown concepts defer 100%** (accuracy 0 BY DESIGN: the model
refuses rather than guesses; false-confidence 0.000).

The conflicting-lessons attack teaches 40 real contradictions into a copy
of the model, with a control arm re-teaching the same states at gold (same
teach-event count, no dispute) to isolate contradiction-specific damage:
**silent_override = false on all three domains**, contested 40/40,
decisions bit-deterministic afterwards. Untouched-row accuracy on tickets
moves 0.7193 → 0.6517 under attack vs 0.7027 control — the anti-Hebbian
law's blast radius is now a measured number (~5.1 pts), not a guess.

### Multicore and the GPU gate (M5)

- CSR mirror: the settle hot path reads contiguous offsets/targets/weights
  buffers, rebuilt lazily and order-preserving — bit-identical floats.
- `make omp` builds the same sources with `-fopenmp`. Parallel settle
  partitions sources statically and combines per-thread scatter buffers in
  fixed thread order. `make omp-identity` proves sequential == `--threads 2`
  on the full report (routing/calibration/honesty/guardrail identical at
  0.971). `syfox-test-omp` passes the whole suite.
- Honest scaling numbers on this 2-core sandbox: deterministic parallel
  settle is SLOWER at this fabric size (p50 620 µs vs 415 µs sequential —
  coordination beats the gain below ~10k nodes); batched throughput with
  private per-worker engines is FASTER: 2,390 → 3,714 decisions/sec
  (**1.554×**, `bench --throughput 2`, checksum non-zero).
- GPU gate (`make density`): tickets fabric 3,518 nodes / 82,655 lanes /
  density 0.0067 / degree 23.5; game 0.0458; guard 0.0485. The SoA/CSR
  layout is GPU-shaped; a port happens only when substrates are dense
  enough to pay for it. Design only — no GPU claims.

## Recall — associative memory in energy space

`syfox recall` retrieves stored experiences the way the physics allows: settle
the query into an energy fingerprint, settle each stored memory, rank by
cosine of the two settled fields. Hopfield-style content-addressable memory,
but similarity lives in the substrate's own state space — **no token
comparison, no n-grams, no embedding table, no transformer**.

```bash
./build/syfox recall --model model-game --state "zombies at night, health dropping" \
                     --memories data/game_train.jsonl --topk 5
```

On the derived game model: the zombie query recalls the flee lessons at
resonance 0.9696–0.9421, fight lessons clearly below at 0.856; untaught
vocabulary resonates with nothing (empty hits — the same honest silence as
decide). Read-only and bit-deterministic.

Details and mechanism specs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §10–13.

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
  Platt scaling (noul) on the held-out split's readout energies so
  probabilities mean what they say. Multi-class NLL objective; a 1-bit ECE
  guard rejects the fit when it would hurt. The core physics is untouched.
* Tokenization — one deterministic pipeline for teach + decide
  (`core/normalize.hpp`): lowercase alnum runs → Porter (1980) stemming →
  synonym folding via `data/synonyms.txt` (override with `--synonyms`; the
  file ships with the model). No neural net, no pattern matching in the core
  — this is input encoding at the boundary, and the energy physics downstream
  is unchanged. Extend `data/synonyms.txt` to teach the boundary new
  equivalences; keep it conservative, a wrong merge is a wrong lane.

## The honest-silence contract

If the field settles below the silence floor — unknown vocabulary, empty
state, nothing corroborated — every question returns:

```json
{"deferred": true, "reason": "unknown_vocabulary"}
```

SyFox never guesses from nothing. For guardrails, combine this with a defer
band: route answers with confidence below your threshold to a human.

## Known limits (v2.2 — honest list)

* Tone/sentiment questions need much more data than routing questions; with 22
  seed rows the argmax is usually right but probabilities stay near-uniform.
  More genuinely distinct labelled rows help; re-teaching near-duplicates does
  not (measured twice — see the augmentation dose-response sections).
* ~~Word-level substrate: heavy typos fall outside the vocabulary~~ **fixed in
  v2.2**: character trigram lanes now exist (see "Multilingual boundary and
  typo robustness"). Measured honestly: on the 6-row tickets held-out set the
  bridges are neutral at the operating point (clean and corrupted accuracy
  unchanged) — the mechanism is load-bearing for non-Latin typo routing, where
  the script barrier makes vocabularies tiny.
* ~~English/Bengali-style token folding is deliberately minimal~~ **updated in
  v2.2**: tokenization is now codepoint-aware for any script (UTF-8 decode,
  script-homogeneous runs, Cyrillic/Greek/Latin lowercase maps) — still a
  deterministic string boundary, still no NLP pipeline, still no neural net.
* **The substrate derives AND retrieves.** A state you never taught still gets
  a computed answer: energy diffuses across the lane fabric and evidence
  composes across lessons. Verified in a two-lesson probe (only
  `alpha beta`→x and `beta gamma`→y taught): querying `alpha` alone — a
  state never taught, from a lesson that never mentioned y — puts y at
  0.494 through the two-hop path alpha→beta→y. And since v0.2 it also
  **retrieves**: `syfox recall` (core/recall.hpp) is Hopfield-style
  associative memory — settle the query and each stored memory into
  L2-normalized energy fingerprints, rank by cosine. Zero token comparison,
  zero pattern matching; untaught vocabulary resonates with nothing. What
  the field cannot exceed is the vocabulary and lane fabric of its
  experience: concepts you never taught don't exist in it, and unknown words
  defer (honest silence) instead of being hallucinated. Audit
  `model-*/substrate.bin` — the lanes ARE the knowledge, every one
  inspectable.

## v3.2 — the semantic layer, retrieval by default, the two-stage router

The substrate now carries a SECOND field: a 64-dim semantic vector per
concept (signed character-trigram hashing + two fabric-grounding passes —
deterministic, no ML), from which **resonance edges** leak a small,
energy-conserving share of a source's energy to its semantically similar
neighbours during settle, and readout gains a semantic-neighbour term.
**Context-sensitive lanes** learn required/forbidden context words from the
lessons that laid them and carry less when the decision's own tokens do not
match — the same "card" routes toward card_arrival or
card_delivery_estimate by context, physically. **Retrieval is default-on**:
a `memories.jsonl` in the model dir primes every decide with the outcomes of
the most resonating lived experiences (Hopfield-style settled-field
cosine), deterministic and disclosed in the output. A dedicated
**bank77 fabric** (1,648 nodes, semantic field + 141,920 context-signed
lanes + 256 memories) takes 77-way banking intents to **0.158 hidden**
(vs 0.081 v3.1.0 dedicated / 0.023 shared), and a small dedicated
**router fabric** (16 domain anchors, cal 0.444 at 16-way vs chance 0.063)
powers the two-stage physics router: `decide --router model-router16 ...`
routes, then the mapped domain fabric decides. Pre-v3.2 models replay
bit-for-bit (the semantic state lives in a magic-guarded file tail);
`--no-semantics`, `--no-retrieval`, `--no-hierarchy` are the kill switches.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §16 for the full mechanism
specification, [docs/ZEROSHOT.md](docs/ZEROSHOT.md) for the zero-shot probe
reports (Parts 1 and 4), the runtime-flag mechanism (settled energy is
invariant to the source modes; confidence is concentration, not
correctness), [docs/PROBING.md](docs/PROBING.md) for probe methodology, and
[ROADMAP.md](ROADMAP.md) for where this is going.

## v3.2.1 — the paper-blocking bug fixes

Three ablation-study blockers were root-caused and fixed (full story in the
[CHANGELOG](CHANGELOG.md), every number archived in
`data/v321_bugfix_tables.json`):

- **Ablation kill switches** were silently inert on pre-v3.2 fabrics (the
  replay contract: no SEM4 tail, no memories, no hierarchy => nothing to
  switch off). `decide` now discloses every inert switch on stderr,
  `tools/ablation_suite.py` refuses to pass a null ablation, and a unit test
  pins >= 4 distinct outputs across the five paper configs on an armed
  fabric. On a v3.2 fabric the switches were already live (accuracy moves
  0.9533 -> 0.9733 with `--no-semantics` on tickets-cal).
- **Confidence collapse** (96.6% accuracy at mean confidence 0.0016) was a
  calibration-path bug: the fit skipped the retrieval-priming block and ran
  in a different dose regime than decide. `harvest_rows` now shares the
  exact decide-time `prime_field` prologue, and the shipped fabrics are
  recalibrated in their benched mode: tickets-cal mean confidence
  **0.0016 -> 0.8736** (median 0.98, C|ok 0.898 vs C|bad 0.373, accuracy
  held), easy cases 0.64-0.97, router16 ECE 0.368 -> 0.099.
- **`--router model-router16`** failed with a misleading "lacks router.json
  anchors" whenever the fabric dir wasn't found in the caller's layout —
  a missing substrate used to load as a silent empty fabric. Model loads now
  fail loudly naming the dir, `resolve_model_dir` accepts both repo layouts
  (`model-router16` and `model/router16`), and `make ci` asserts the route
  field end-to-end. Route distribution across all 16 hidden domain files:
  language anchors 12/12, modal agreement 0.663.

Also fixed on the way: `save()` iterated unordered maps, so every
load->save cycle permuted `substrate.bin` (never byte-stable, settled-field
sum order drifted) — persistence is now sorted-key and provably
roundtrip-stable (unit-test enforced; all shipped fabrics rebuilt through
the converged path). `make ci` is the pre-commit gate: unit suite +
ablation distinctness + router route assertion.

## License & credit

MIT © 2026 Mr-DS-ML-85. Core mechanics ported from the author's
[Synthetic-Intelligence](https://github.com/Mr-DS-ML-85/Synthetic-Intelligence)
research substrate. Inspired by the *interface* of TypeSafe AI's Jev — not
affiliated with, nor derived from, any TypeSafe model or weights.
