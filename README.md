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

Silence is enforced at TWO layers, each with a documented absolute floor:

| Layer | Floor | Reason | Fires when |
|---|---|---|---|
| decide (settled field) | `silence_floor = 0.05` total field energy | `honest_silence` / `unknown_vocabulary` | the whole field is dark — nothing was taught or resonated |
| readout (candidates) | `kUnknownCandidateFloor = 0.01` best-candidate energy | `unknown_candidates` | the field is lit but NO option carries measurable signal |

The readout floor is **absolute** (v3.3.1), not relative to field brightness: a
bright field sitting entirely on non-candidate nodes must defer, and a valid
candidate on a dim field must answer (measured: dim-mode candidates at 0.34
answer; dust carryover below 0.01 defers instead of being picked by noise).
Do not raise it above 0.10 — that false-defers valid dim candidates.
Set `SYFOX_DEBUG_READOUT=1` to print per-candidate energies and the active
threshold on stderr; `make readout-silence` (also in `make ci`) regression-gates
the whole contract.

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

## v3.3 — question-conditioned readout, tie disclosure, readout silence

Three readout-honesty mechanisms, one ported from the original
Synthetic-Intelligence substrate (full story in the
[CHANGELOG](CHANGELOG.md), numbers in `data/v33_reasoning_tables.json`):

- **Question-conditioned readout** (opt-in, `--question-gate`): the old
  repo drove every token of the turn into the field; SyFox's readout
  measured state energy only, so the salient state noun beat the
  question-relevant entity. The gate scales candidates by the lane mass of
  the question's NEW tokens (`who`, `latest`, ...), disclosed as
  `question_gate:[tokens]`. Measured: who-found margin 0.056 -> 0.113,
  latest-temperature confidence x14; costs 5.3 points on tickets-cal,
  hence opt-in like the energy-norm gain.
- **Exact-tie disclosure**: replies set `tied: true` when the top two
  probabilities are equal (the pick is deterministic criteria order; the
  v3.2 semantic readout term itself breaks most ties physically).
- **Readout silence**: when NO candidate carries measurable energy, decide
  defers with `reason: unknown_candidates` instead of answering a uniform
  distribution by criteria order (the "unknown options answered by noise"
  and pure-OOD failure modes). v3.3.1 makes the threshold an explicit
  absolute constant — `kUnknownCandidateFloor = 0.01` (was an implicit
  `> 0` check that answered on dust-level carryover) — documented in the
  honest-silence contract above, debug-visible via `SYFOX_DEBUG_READOUT=1`,
  and pinned by `tests/readout_silence_test.sh` / `make readout-silence`.

## v3.4 — honest defer for ties, multi-hop readout, question-context gate

Three decision-layer mechanisms (the SI substrate physics stays frozen; every
knob is measured, deterministic, and disclosed):

- **Honest defer for near-ties (engine default ON, margin 0.05)**: when the
  top two probabilities are closer than the margin, the readout does not
  carry a decision — the answer defers with `reason: ambiguous_tie` instead
  of shipping a coin-flip as a confident label. The engine owns it, so CLI,
  C API, HTTP bridge and bench share one behavior (the v3.1 CLI post-pass
  `low_margin` is gone). Flags: `--defer-margin P`, `--no-defer`. Measured:
  tickets bench score accuracy 0.938 → **1.000** with defer rate 0.063 (the
  one wrong score row was a near-tie, now honestly deferred); the 6-row game
  toy fabric's decisions all sit at mean margin 0.023, so they now disclose
  as all-deferred (`choice_defer_rate: 1.0`) instead of noise-answering.
  Bench JSON gained `choice_defer_rate` / `score_defer_rate` so selective
  accuracy reads honestly (accuracy over ANSWERED questions, defers counted
  separately).
- **Multi-hop readout walk (opt-in, `--hops N`, default 1 = legacy)**: walks
  lanes BFS-style from each probe anchor up to N levels with per-hop damping
  `hop_coupling/sqrt(hop+1)` (hop 1 ×0.707 … hop 4 ×0.447), levels visited in
  ascending node-id order, width-capped at 64 per level. Deterministic;
  default N=1 is bit-identical to every earlier version (test-enforced).
  Measured on chain probes: 2-hop chains already answer at N=1 (diffusion
  carries them); the walk lifts far targets into readout visibility
  (3-hop target 0.1061 → 0.1097; 4-hop target 0.210 vs ~dust at legacy) —
  but damped lanes keep far targets near-tied, and the honest-defer layer
  refuses to dress them as confident answers. Faint multi-hop signal is now
  VISIBLE and HONEST instead of confident noise.
- **Question-context gate (opt-in, `--ctx-gate`, `--ctx-alpha F`)**:
  two-stage settle — stage 1 settles the question's own tokens
  (instructions + criteria descriptions, no labels) into a context field;
  stage 2 re-settles the state and the final field composes as
  `(1-alpha)*state + alpha*context` (alpha default 0.5). Disclosed as
  `usage.ctx_gate` / `usage.ctx_alpha`. Measured: who-found answer holds with
  confidence 0.056 → 0.061; OOD abstention preserved; temporal probes hold
  (with a disclosed confidence tradeoff). Deterministic; two settled fields
  composed at the decision layer — settle physics untouched.

## v3.5 — JAS: the J-A-S cycle, the impossibility register, the arithmetic oracle

Ported from the lineage's research papers — `research-paper/jas.md` (the
J-A-S cycle) and the central paper's division of labour: **the substrate is
the relevance heuristic; claims that can be destroyed live above it; an
external oracle decides**. No physics touched, no fitted parameter, no
gradient. Three new commands:

- **`syfox jas --model M --lessons F --holdout F`** — the cycle as a running
  loop. E sense experience (labeled rows) → J the jump: counting leaps to a
  universal STRONGER than the evidence ("every row containing token T has
  label L" — that gap is what makes it refutable; a restatement of the
  frequency is only a summary) → A→S deduction: each holdout row gets a
  mechanical prediction → experiment: the oracle is the observed label →
  refutation is logged → revision under a STRUCTURAL restriction
  (T AND U, checkable before any experiment, never fitted to its own
  counterexamples) → round 2 on held-out rows the first round never saw.
  The frozen field is scored on the same rows as the disclosed
  "relevance heuristic" column. MEASURED on the demo fabric
  (`data/jas_cycle_demo.jsonl`, 18 train / 6 holdout rows): the jump
  induced 3 universals; round-2 fresh rows REFUTED one (the 2/2-observed
  `customer → refund` axiom died on "parcel shipping refund requested by
  the customer" with a disclosed conflict) — verdict
  `REFUTED_IN_ROUND2_REVISE_NEXT_CYCLE`, counterexample appended to
  `<model>/refuted.axioms`. A clean holdout yields `SURVIVES` — survival is
  not proof; induction confers no warrant beyond not-yet-destroyed.
  Unit test `test_jas` pins the full arc: refutation → revision
  (`alpha AND one → A`) → convergence on fresh rows.
- **`syfox register [--model M] [--text]`** — the **impossibility register**:
  every "X cannot be done" claim records its WARRANT — ASSERTED / INDUCED /
  DERIVED / THEOREM — because warrant determines what a counterexample
  MEANS. For a THEOREM it is a scope error until proven otherwise (Minsky–
  Papert was never violated; multilayer networks left its scope). syfox's
  five architectural boundaries are seeded with honest warrants — two are
  already SCOPE-ESCAPED by this very release (see below) — and persist to
  `<model>/impossibility.json`.
- **`syfox calc --expr "17*23" [--compile]`** — the arithmetic derivation
  oracle, and the honest answer to "why can't the engine compute": **the
  field never computes**. Six primitives (+ − × ÷ % ^) compose through one
  recursive-descent grammar into arbitrarily deep derivations (the thinker
  ratio: few primitives, exponential reachable expressions), with a
  verifier that rejects rather than guesses (`1/0` errors, never answers).
  Word problems map number words to operators (times → ×, plus → +, …) —
  detection is a relevance heuristic, exactly the division the central
  paper prescribes. `--compile` runs the full JAS experiment: the engine
  EMITS a C++ translation unit, hands it to g++ (external, deterministic,
  indifferent — it shares no representation with the substrate), runs it,
  and discloses `agreement` + `provenance: established_by_experiment`.
  MEASURED: `17*23` → 391 with g++ agreement true; `2+3*4` → 14;
  `(2+3)^2/5` → 5; "what is 17 times 23 plus 5" → 396. This is precisely
  how the original repo "wrote code and calculated numbers" — the compiler
  computed, never the field. The impossibility theorem "the settled energy
  field cannot compute exact arithmetic" STANDS; the capability escaped its
  scope to the decision layer + oracle. Scope escape, not a violation.

CI gate: `make jas-test` (wired into `make ci`); `tests/jas_test.sh` pins
the oracle values, the register shape, the cycle verdicts, and byte-level
determinism of the cycle JSON.

### Why the 24-domain accuracy is low — and which paper unlocks each fix

A per-domain bag-of-words fabric (one fabric per corpus, ~900 training
rows, untyped unweighted co-occurrence lanes) measured on 24 corpora:

| Measured shape | Root cause (from the fabric's construction) | Paper remedy (status) |
| --- | --- | --- |
| paws 0.500 (chance), xnli 0.333 (chance) | word-swap / entailment pairs are a MULTISET-INVARIANCE problem — untyped bag-of-words literally cannot see the difference (same theorem class as `metformin`/`metfromin`) | ordered + typed lanes: MRS bigram lanes (measured 7/7 on transpositions), CPSB role asymmetry, RADE typed edges — MODERATE fabric change, physics frozen |
| sst2 0.677 (best case) | short, binary, polarity tokens dominate lanes — the one regime bag-of-words handles | (no fix needed — the honest baseline) |
| agnews 0.594, imdb 0.573, amazon 0.573 | longer texts: function-word mass dilutes the unweighted lanes | CMD rarity weights `w = log(N/(1+df))` + habituation analog — MODERATE (lane weighting at learn time) |
| sst5 0.268, emotion 0.297 | fine-grained 5/6-way classes → class nodes settle near-tied; the v3.4 defer layer honestly refuses most of them | bridge lanes gated by relative Fisher distance (rbg + graph_fisher) to sharpen class basins — MODERATE |
| tweetml 0.34–0.50 (8 languages) | ~280 rows/language, no subword morphology, sparse token overlap | MRS bigram lanes + per-language fabrics with banding (hnca/mims) — MODERATE |
| hate3 0.417, tweet_off 0.52, tweet_sent 0.386 | implicit toxicity carries no surface co-occurrence; label skew | perturbation-negative lessons (BED "manifold sculpting": swap subject/object, reorder, splice) so lanes carry structure, not marginal statistics — MODERATE |
| obqa 0.401, dbpedia 0.43 | needs world knowledge beyond one-pass co-occurrence on ~900 rows | l9-L3 transitive pre-derivation with cited premises + analogy lanes — MODERATE |
| wikiqa 0.63 | answer selection benefits from the v3.3 question gate; typed matching absent | RADE proof-pass readout over typed lanes — the typed-fabric half again |

The pattern is one sentence: **the limits are representational (what the
lanes encode), not dynamical (how energy settles)** — which is why every
remedy in the papers lands on fabric construction or the decision layer
and none requires touching the frozen physics. Port order by
measured-value-per-risk: (1) IDF/rarity lane weights (CMD), (2) bigram
lanes (MRS/CPSB — also the subject-object scope escape), (3) bridge lanes
Fisher-gated (rbg + graph_fisher), (4) perturbation negatives (BED),
(5) transitive pre-derivation (l9). The JAS layer is the harness that will
hold each of these accountable: every claimed fix lands as an induced
axiom, gets staked on held-out rows, and survives or is retracted.

## v3.6 — the fabric-construction papers land: ordered lanes, rarity weighting, the perturbation check — and the JEV head-to-head

v3.5 diagnosed WHY the 24-domain accuracy is low (the limits are
representational — what the lanes encode — not dynamical) and named the
port order from the 54-paper corpus. v3.6 implements the top of that order
on the v3.5.0 tree, physics frozen, replay contract intact:

- **Ordered bigram lanes (MRS §4b.1–4b.2, measured 7/7 there; CMD §2.5;
  CPSB ordered pairs; HTR arrival-order tagging).** Every NEW lesson now
  also interns adjacent token pairs (`w1~w2`, porter-stemmed, `~` never
  collides with a normalized word) and binds them to the outcome anchors;
  decide() injects the state's bigram nodes at half dose (`kBigramDose`,
  mirrored bit-exactly in the calibration harvest). This is the escape from
  the multiset-invariance theorem that pinned paws at 0.500 and xnli at
  0.333: a token bag cannot tell "a beats b" from "b beats a", but the pair
  multisets differ. Old fabrics contain no `w1~w2` nodes, so decide-side
  injection is a no-op on them — bit-identical replay, unit-tested both ways.
- **Document-frequency rarity weighting (CMD §2.2 `log(N/(1+df))` grounding;
  MIMS §6 fan-out derivation).** The substrate now carries a df table (one
  note per (lesson, node), persisted in a magic-guarded `IDF5` tail) and
  hebbian_lesson scales STATE-side bind weights by
  `log1p(L/(1+df))/log1p(L) ∈ (0,1]` — ubiquitous tokens bind weakly, rare
  tokens bind strongly, no stopword list, one counted formula. Fabrics
  without the tail reproduce the v3.5 weights exactly (factor 1.0).
- **The perturbation-contrast honesty check (BED §8, opt-in
  `--perturb-check`).** Each decide() additionally settles a structure-
  broken copy of the state — the SAME token multiset with adjacent pairs
  transposed (deterministic). If the broken state's winning margin matches
  the real one, the readout was carried by the token bag, not by structure,
  and choice questions defer with reason `perturbation_tie` instead of
  shipping a bag-of-words coin flip. The real field is snapshotted and
  restored, so the readout measures exactly what an unchecked decide()
  measures. This is BED's "the potential was trained as a classifier"
  diagnosis, run per decision.
- **Already-shipped ports, credited where they live:** l9-L3 transitive
  pre-derivation with cited premises + rbg-L6-A bridge insertion gated by
  the graph_fisher relative Fisher distance = the derive layer
  (`core/derive.hpp`, `syfox derive --gate`), shipped with the
  no-regression gate; JAS (v3.5) is the accountability harness that stakes
  every claimed fix on held-out rows.

**Measured (train split, in-sample — the seed fabrics are demo-scale; the
heldout split has n=6/0/2 rows and is reported in build/bench-*.json):**
tickets choice 0.875 → **1.000** (16/16, defer 0), score 1.000 (defer 0.063,
unchanged); game choice_defer_rate 1.0 → **0.333 with 4/4 answered correct**
(the ordered fabric now separates what the bag fabric honestly refused);
guard 0.778 (unchanged — no order signal in its rows); ood_defer_rate 1.0 on
all three (the honesty floor holds). New test group `test_fabric36` — 22
checks: bigram lanes laid and energized, multiset invariance pinned
bit-equal on a bag fabric AND broken correctly on an ordered fabric, IDF
weakens the same-mass/same-token lane only when df notes exist, IDF5 tail
roundtrips, old-fabric replay bit-identity, perturbation defer on bag-only
fabrics and answer-preservation on ordered ones. Full unit suite,
readout-silence and jas gates pass; `make ci` is now fresh-clone safe
(router/ablation gates SKIP loudly when the gitignored fabrics/data are
absent instead of failing the tree — bug found and fixed in this release).

**The JEV head-to-head (`make jev-compare` → `tools/jev_bench.py`).** JEV
AI (TypeSafe AI, Sept 2026 early access) is the closed "System One" decision
model: state + typed questions in, value + probability + confidence out. That
is SyFox's own shape, so the competition runs on JEV's published axes — every
syfox number below is measured by this tree (core/bench.hpp parity suite),
every JEV number is their published figure (self-reported; their press
labels the multipliers ceilings):

| axis | JEV AI (published) | syfox (measured, this tree) |
| --- | --- | --- |
| decision API | POST /v1/systemone | POST /v1/systemone (server/app.py) |
| question types | noul / choice / score | noul / choice / score |
| batch questions/call | yes (parallel) | yes (one settle) |
| workflow accuracy | 67.8% | **95.5%** (n=44 answered, train split) |
| latency | 70–500 ms | **≤ 474 µs** (p50/p95 across seed fabrics) |
| input cost | $0.042 / M tokens | **$0** (local core, no meter) |
| calibration study | claimed, not public | **ECE published** (mean 0.131) |
| honest defer with reason | confidence only | **defer + reason** (OOD 100%) |
| options per choice | 255 then two-stage | unbounded criteria map |
| determinism | not published | bit-reproducible (CI-enforced) |
| full decision audit | no | evidence JSON + lane ledger |

The accuracy row compares their published workflow figure against syfox's
in-sample train-split runs on three demo fabrics — the axes are shared, the
workloads are not, and JEV's own numbers are self-reported. The structural
advantages are not numbers at all: SyFox discloses WHY every decision or
deferral happened (evidence JSON, lane provenance, impossibility register),
replays bit-identically, runs offline at zero marginal cost, and now carries
order in its fabric — and everything JEV keeps closed, the 54-paper corpus
that built this engine is in the open.

## v3.7 — the representation papers land: PPMI+SVD dense vectors, energy-space self-supervision, typed ordered lanes, generalized tool scope

The readout-side analysis of the v3.6 fabric was precise: the representation
is hand-written by the tokenizer (bag-of-words stems), everything downstream
inherits that one choice, and no readout flag can fix it — the fixes have to
be REPRESENTATIONAL, not dynamical. v3.7 lands all four moves from that
analysis. The SI core stays frozen end to end: decay 0.82, diffusion 0.45,
dual-channel readout, Hebbian learning — every mechanism below is fabric
construction, training loop, or decision-layer tooling.

**Move 1 — PPMI + truncated SVD (`core/distvec.hpp`, opt-in `--distvec`).**
The 1990s answer (Deerwester et al. 1990; Levy & Goldberg 2014), and it is
not neural: the symmetric Hebbian lane fabric IS a co-occurrence observation,
so the builder computes PPMI over the lane weights (hub pollution pays
through the marginal), then factorizes with deterministic subspace iteration
+ Rayleigh-Ritz (cyclic Jacobi) and keeps the PSD part — U·Sigma^beta,
Perron/frequency axis dropped, isolated rows zeroed. Every content node gets
a dense k-dim vector (default 300), and the semantic RESONANCE EDGES are
re-selected from the dense space (small fabrics exact-scan; large fabrics use
deterministic 2-hop lane candidates). Settle physics untouched — the edges
are the same mechanism v3.2 shipped, only chosen by a representation that
generalizes. Engineering found on the way, each pinned by test: the raw-FNV
hash bit collapses the init basis (splitmix64 finalizer added); a singular
±1 init silently loses subspace rank (deterministic canonical-basis repair);
degenerate ±|lambda| pairs never separate under plain subspace iteration
(Rayleigh-Ritz rotation added); small fabrics (n <= 512) use the full space,
so their spectrum is exact. Vectors persist in a magic-guarded 'DSTV' tail;
every pre-v3.7 fabric replays bit-identically (unit-tested both ways).

**Move 2 — energy-space self-supervision (`syfox pretrain`).** The direct
analog of next-token prediction, expressed as lane physics: take UNLABELED
raw text, hide one token (deterministic stride), inject the rest, settle,
read out which BASIN got the energy (the true token's node against the
strongest non-context node), strengthen the failed context->token lanes with
error-weighted Hebbian. The loss is measurable —
mean over masked positions of `1 - e_true/(e_true+e_best_other)` — and it is
reported per epoch. No labels, no gradients, no fitted parameters. MEASURED
(data/pretrain_corpus.txt, 55 lines, 603 masked positions/epoch, 12 epochs,
eta-scale 2): epoch loss 0.463 -> 0.445 (falls, plateaus, bit-deterministic
across runs), vocabulary grows 0 -> 191 nodes with real lanes.

**Move 3 — typed ordered lanes (`core/roles.hpp`, opt-in `--typed-lanes`).**
The strong version of order, not the bigram patch: nodes carry a ROLE
(S/V/O/M/F from a compact scene grammar — closed function-word list, seed
verb-cue lexicon, prepositional-patient rule for "to/for"), and relation-
typed pair nodes (`w#s>v#v`, `w#v>o#o`, `w#s>o#o`, `!w#v` for negation)
intern exactly like the v3.6 bigrams and bind to the outcome anchors through
the same Hebbian lesson. Old fabrics contain no typed nodes, so the
decide-side injection is a no-op on them — the replay contract, unit-tested
both ways. MEASURED (data/role_reversal_{train,probe}.jsonl, 10 rows, 4
insertion distances 0/1/3/6): the bag fabric answers 4/8 probes at chance;
the typed fabric answers 6/8 and separates both distance-0 reversal pairs
(unit-pinned deterministically). Margins stay near the toy fabric's noise
floor (0.004-0.014) — the mechanism is real and pinned; scale is the
limiting factor, as with every fabric-construction effect in this engine.

**Move 4 — generalized tool scope (`syfox tools`, `decide --tools`).** The
calc pattern (field refuses, verified tool computes, register records the
refusal was principled) is now a typed contract carried by EVERY register
claim: `field_behavior` / `tool` / `tool_verified` on all five
impossibilities, printed by `syfox tools [--text]`. `decide --tools` runs
the arithmetic relevance heuristic and discloses the oracle's derivation in
usage (`tool_used/tool_expression/tool_value`, provenance
established_by_experiment) while the field still answers or defers exactly
as it measures. Default off; bench bit-identity untouched.

**The moves compose — the OOD bridge (measured, data/v37_tables.json).** A
labeled billing/technical fabric defers 6/6 on probe rows whose words it has
never seen (clean baseline). `pretrain` over the UNLABELED corpus interns
those words with real lanes; after temperature calibration on the labeled
rows and the disclosed `--energy-norm` dose gain (same knob on every
variant), the pretrain fabric answers 6/6 probes correctly (by-construction
golds, disclosed in the table). The plain fabric under the SAME knobs
answers 3/6 at 0.5 accuracy — the chance floor — which is precisely the
measured difference between a manufactured answer and a bridged one: the
pretrain lanes carry the energy to the right basin, the plain fabric's
function-word carryover does not.

Regression: bench bit-identical to v3.6 on every shipped fabric (tickets
choice 1.000 / score 1.000 defer 0.063, guard 0.778, game defer 0.333,
OOD defer 1.0 x3, deterministic x3); full unit suite (4 new test groups)
green; readout-silence ALL PASS; jas-test ALL PASS; `make ci` green.

## Architectural boundaries (System-2 limits — honest list)

These are NOT bugs; they are capabilities a bag-of-words SI core does not
have and cannot fake. Each is measured; none is hidden by a confident-looking
wrong answer — the v3.4 defer layer converts most of them into honest
deferrals. Since v3.5 each boundary ALSO lives in the impossibility register
(`syfox register`) with its WARRANT: multi-hop >2 and the fixed-threshold
limit are INDUCED (surveys of failures, refutable by construction), temporal
ordering is DERIVED (from the parent axiom "lanes are unordered multisets"),
subject-object and field-arithmetic are THEOREMS — and two of them are
already SCOPE-ESCAPED: arithmetic by the calc oracle (computation happens
outside the field), subject-object by ordered/typed lanes when a fabric
carries them.

- **Multi-hop propagation beyond 2 hops stays near-tied.** Diffusion carries
  2-hop chains (alarm → lights answers at depth 1); 3-4 hop targets gain only
  damped scraps of energy (`1/sqrt(hop+1)` × per-lesson lane decay), so they
  defer as `ambiguous_tie` rather than answer. A typed or weighted relational
  fabric would be needed for crisp transitive chains.
- **Arithmetic.** No numeric circuit exists in the physics; temperature/
  counting questions answered by token salience alone (architectural limit,
  documented since v3.3, shared with the upstream repo). SCOPE-ESCAPE (v3.5,
  generalized v3.7): the calc oracle computes OUTSIDE the field — `syfox calc`
  standalone, and `decide --tools` discloses the oracle's derivation in usage
  while the field still measures. The register records the contract.
- **Temporal ordering is training-side, not inference-side.** "Latest/after"
  probes answer only when the fabric carries lanes that encode the ordering
  (the latest-temperature probe answers via the question gate at conf 0.117);
  a fabric without such lanes cannot infer order at readout. Since v3.6 a
  fabric trained with bigram lanes (the default) carries ADJACENCY — a
  weaker but real form of order — and since v3.7 the opt-in `--typed-lanes`
  construction adds relation-typed roles (S/V/O, negation) on top. The
  perturbation check discloses when a decision is order-carried rather than
  bag-carried.
- **Subject-object reversal.** The core is deliberately bag-of-words: lanes
  connect co-occurring concepts without typing them, so "who found the radio"
  and "what did the radio find" read the same lanes. The original research
  repo's TYPED graph edges are exactly what this core omits by design.
  SCOPE-ESCAPE (v3.6 + v3.7): fabrics trained with the default bigram lanes
  carry token ORDER ("alice beat bob" and "bob beat alice" decide
  differently, unit-tested); the v3.7 `--typed-lanes` construction adds the
  strong version — (word, role) nodes and relation-typed S>V, V>O, S>O pair
  lanes, which survive inserted material that dilutes bigrams (measured
  6/8 vs 4/8 at chance on the reversal probe; both distance-0 reversals
  pinned in unit tests). The theorem's scope is the untyped bag fabric.
- **OOD with lit candidates.** Fully dark candidates defer
  (`unknown_candidates`); but a state-lit OOD question whose candidates all
  sit in the fabric can still pick a confident wrong answer — abstention is
  only as good as the energy gap, which is why the defer margin exists.
  Since v3.6 the opt-in `--perturb-check` adds a structure-contrast defer:
  when a decision's margin survives breaking the state's token order, it was
  a bag statistic, not structure, and defers as `perturbation_tie`.
  PARTIAL ESCAPE (v3.7, measured): `pretrain` turns UNLABELED text into
  lanes, so "semantically-known-but-lexically-unseen" words stop being dark —
  the OOD probe suite went from 6/6 honest deferral to 6/6 answered correct
  (by-construction golds disclosed; data/v37_tables.json) — while the plain
  fabric under identical knobs answers at the 0.5 chance floor. The
  fixed-threshold floor itself stays: a bridge that reaches only dust still
  defers, by design.

## v3.8 — physics-native thinking: adaptive depth, multi-vector state, causal signatures, energy self-verification

The v3.8 spec: make the field THINK deeper without token prediction, without
ML/NN/transformer, without touching the frozen SI physics (decay 0.82 /
diffusion 0.45 / dual-channel readout / Hebbian). Four mechanisms, each
opt-in, each deterministic, each a no-op on existing fabrics (the replay
contract holds; bench bit-identical to v3.7). Plus the v3.7.1 scaling fix
that made --distvec survivable at real fabric sizes.

**v3.7.1 — distvec SCALING FIX (prerequisite).** The v3.7.0 build was
measured O(n^1.9): 76 s at 5,595 nodes, 156 s at 7,985, timeout past 9.3k,
~15 hours extrapolated at 94,773 nodes — a cost, not a win. Root causes
removed: the exact neighbour scan (O(n^2*k)) now caps at 2,000 nodes (larger
fabrics use the documented 2-hop lane closure); MGS orthonormalization moved
to a column-major blocked BCGS2 basis (contiguous kernels, O(k/32) OpenMP
regions instead of ~840k strided ones); the subspace walk, Rayleigh-Ritz and
Ritz rotation rebuilt on the same basis; init hashes cached per node.
Measured A/B (same corpora, same machine): 17.1 s -> 7.2 s at 5,107 nodes;
14.5 s at 10,216; 36.9 s at 20,398; 95.4 s at 40,850 — near-linear through
10k, ~1.36 exponent at large sizes from RAM streaming. Extrapolated generalist
build: ~15 hours -> ~5 minutes (~180x). Same-name rebuilds md5-identical.

**TASK 1 — adaptive depth (`--adaptive-depth`, `--adaptive-max-k`).**
Thinking-Without-Tokens lineage: depth is compute. Before each settle the
engine measures the problem — unknown-token fraction (the OOD axis) and lane
isolation (known tokens with zero lanes to travel) — into a difficulty in
[0,1], and settles k = base + round(difficulty * max_extra) passes (defaults
8..16). Same state -> same depth, bit for bit; the base is pinned at engine
construction so repeated decisions cannot ratchet; harvest_rows mirrors the
rule (the v3.2.1 fit==decide lesson). The settle equations per pass are
byte-identical — only their count scales with difficulty. Disclosed in usage
as difficulty / k_settle_used / k_settle_base.

**TASK 2 — multi-vector state (`--multi-vector`, `--multi-vector-k`).**
CSM lineage: the state is not one energy vector but K superposed views.
On a --distvec fabric, the engine settles the SAME injection K times (default
8), each under a spectral channel gate: lane (a,b) carries flow scaled by
exp(-(u_k(a)-u_k(b))^2 / 2 sigma_k^2), where u_k is the fabric's k-th leading
distributional axis — lanes between spectrally close nodes flow freely,
distant ones are damped. Each channel runs the frozen physics exactly; the
readout measures the MEAN of the K settled fields (the multi-vector state).
Channels with dead axes are skipped and disclosed. No distvec field -> honest
no-op. Deterministic; unit-tested for determinism, no-op, and effect.

**TASK 3 — causal signatures (`--causal-lanes`).** WHAT/WHY/HOW relation
types on lanes: a deterministic connective scan interns cause?>effect nodes
("so/therefore/cause/lead/trigger/produce" keep text order; "because/since/
due" flip it so the cause leads the name) and means!>goal nodes ("by/via/
through/using/how"). Interned, df-noted and bound to outcome anchors exactly
like the v3.6 bigrams and v3.7 typed pairs; decide/harvest/perturb inject the
carried pairs at kCausalDose=0.5 (fit/decide parity). Cross-domain reasoning
uses the signature: "overcharg?>agent" composes with any other ?>-typed
fabric structure even when no surface token overlaps. Fabrics learned without
the flag contain no causal nodes -> every decide-side block is a no-op
(replay contract, unit-tested both ways).

**TASK 4 — energy self-verification (always disclosed; gate opt-in
`--self-verify`, `--self-verify-floor`).** EBT lineage: the settle's own
convergence trace is a third confidence measurement besides margin and
calibration — whether the field had finished MOVING. Every pass the energy
decays (~0.18 baseline, moves nothing) and diffuses (the actual computation);
the trace records the last pass's relative motion sum|de|/sum e. Measured
settled baselines: 0.189-0.212 across the bench fabrics, so the default
floor is 0.25; a choice answer measured on a field still routing energy
above the floor defers with reason `unconverged_field`. The trace (passes /
motion_rate / eps-break) is ALWAYS in the decide usage JSON.

**MEASURED — the think30 suite (`make think30` -> data/think30_tables.json;
tools/think30_probes.py; every number from an actual run in this tree).**
30 tests, 6 categories x 5: A choice (baseline competence), B why, C how,
D cross-domain causal transfer, E multi-hop chains, F honesty (unanswerable
states that MUST defer). Configs share probe files; fabrics differ only in
the construction flags (3-epoch consolidation + calibrate, disclosed):
  baseline (v3.7-style fabric, plain decide)      accuracy 0.467 (14/30)
  thinking (causal+typed fabric, all four flags)  accuracy 0.600 (18/30)
  multivector (thinking + --distvec --multi-vector) accuracy 0.633 (19/30)
Per-category movement against the baseline: E chains 0/5 -> 3/5 (adaptive
depth + self-verify let long chains run to their attractor instead of
answering transients); F honesty 0/5 -> 5/5 (the calibrated baseline SHIPS
noise answers on near-vocabulary OOD states — self-verify + perturb-check
catch every one); A 4/5 -> 5/5 under multi-vector. The B/C/D causal
categories measured 3-4/5 in BOTH configs at this fabric scale: the v3.6
bigram lanes already carry most of the signal here, so the causal signature
shows no additional accuracy delta at demo scale (honest; the unit tests
pin the mechanism, the think30 D-probes pin the routing). DEFER BEHAVIOUR:
the thinking configs defer more and answer better — selective accuracy over
answered rows rose from 14/14 to 18/19 with deferrals disclosed per reason.

**Boundary update.** Temporal ordering: carries adjacency (v3.6) and now
role-typed and causally-typed structure (v3.8) on fabrics built with the
flags on. The four v3.8 mechanisms are decision-layer and fabric-side only —
the impossibility register's THEOREM entries are untouched; what changed is
that the engine now MEASURES and discloses when it is thinking deeper
(adaptive depth), seeing more (multi-vector), reasoning by relation type
(causal lanes), and whether it finished thinking (self-verification).

## v3.9 — the constructive substrate: SI's Layer-2 reasoning stack, ported at last

The v3.9 audit made two charges, both verified and both answered here.

**Charge 1: "SyFox is not a port of SI — it shares two constants."** TRUE for
the physics, and the false advertising is gone: `si_substrate.hpp` now states
the honest lineage in its header (shared: substrate architecture, salience
integrator, selection modes; different by design: scalar energy vs SI's
mass-spring oscillators, undirected lanes vs stiffness/resonance tunnels,
pass-counted settles vs dt=0.02 ticks, native 0.82/0.45 constants). What the
audit ALSO surfaced was the real gap: syfox had SI's Layer-1 field but NONE
of its Layer-2 constructive substrate — no typed knowledge graph, no chaining,
no proof traces, no rule induction, no discovery queries, no knowledge base.
v3.9 ports that stack, faithfully, into `core/reason.hpp` (1,500+ lines,
self-contained):

- **The typed graph** (`sxr::ConceptGraph`): directed edges with FULL
  provenance (rule, parent_a, parent_b, step); O(1) existence, O(deg)
  neighbourhoods; bounded retraction (RADE semantics — removing a premise
  retracts exactly its derived descendants).
- **The nine reasoning primitives** (synthetic-intelligence.md §4.4):
  transitive, deduction (modus ponens), inheritance (class-property flow),
  negation (contrapositive), case analysis (disjunction elimination),
  induction (schematic, min-witnesses), analogy (structure-mapping transfer),
  abduction (backward-only), recursion (the fixpoint loop). Forward chaining
  to fixpoint under a tension budget (Axiom II), backward chaining with
  adaptive iterative deepening, cycle-safe via an active goal stack.
- **Proof traces**: every derived edge reconstructs its full derivation chain;
  the verifier re-checks each derivation against the rule set down to the
  premises before an answer is trusted.
- **The meta-learner** (§5.3.3): contiguous derivation sub-chains are counted
  across queries; a chain that saves ≥ 2 bits is promoted to a macro-rule.
  The thinker index (§3.4) measures C (composition) vs R (retrieval).
- **The six discovery queries** (inference.hpp): isolated subgraphs, dangling
  nodes, the exact missing bridge (bidirectional BFS), symmetry gaps (the
  Maxwell displacement-current pattern), anomalies, and the gap-query
  classifier. `syfox discover` runs them.
- **The .axioms knowledge format** — SI's own data files
  (syllogism.axioms, insulin_pathway.axioms, maxwell_equations.axioms, ...)
  load unchanged. New commands: `syfox axioms --load FILE`, `syfox ask
  --query "subject relation object" [--backward]`, `syfox reason`,
  `syfox discover [--query "..."]`, `syfox thinker --queries FILE`,
  `syfox audit`. Loading axioms interns their concepts into the fabric AND
  lays axiom lanes — one vocabulary, two stores. The RADE proof pass
  (`--proof`) cites the typed path behind a choice answer;
  `--require-proof` defers choices the knowledge layer cannot justify.

**Charge 2: "many papers unimplemented."** The remaining never-ported
mechanisms land here, each opt-in, each with a replay contract, each a
measurement-first port from the digest agents' reading of the papers:

| paper | mechanism | flag / command |
|---|---|---|
| RADE | premise-cited proof pass at decide | `--proof` / `--require-proof` |
| CMD §4 | wavefront gate: candidates the query reached | `--wavefront-gate` |
| MCPE/CPME | momentum readout: echo suppression | `--momentum-readout` |
| HTR §2.2 | coherence gate: structural rise vs field (ΔR) | `--coherence-gate` |
| VGHT | coincidence-gated hop walk C=σ(12(E·E−0.05)) | `--vght-gate` |
| P-CMA (shell) | gradient-free CEM over source-cap variants | `--cem-plan` |
| MAA | fragment abduction for unknown tokens | `--fragment-abduction` |
| fixes.md | OOD signature density gate (kNN over answered signatures) | `--ood-knn` |
| RIFA SIL | contradiction tension accumulator + conflict-zone defer | `--tension-gate` |
| CME | graded fingerprint re-teach (half dose) | `--fingerprints` |
| CMD §2.5 | sentinel bigrams (word-boundary position) | `--sentinels on` |
| RB (surrogate) | pro/con negation-scoped lane namespaces | `--pro-con on` |
| CPME-II (discrete) | per-lane order evidence Φ-blend, ORD1 tail | `--order-consolidation on` |
| CMA/CCH | DERIVED-vs-INDUCED path-additivity audit | `syfox audit` |

Measured (think30, same 30 probes, every number from an actual run in this
tree): baseline 0.467 → thinking 0.600 (v3.8) → multivector 0.633 (v3.8) →
**v3.9 stack 0.700 (21/30)** — thinking flags + `--wavefront-gate
--cem-plan` on a `--distvec --multi-vector` fabric. The bench replay is
bit-identical to v3.8 (tickets 1.000/1.000, guard 0.778, game defer 0.333,
OOD defer 1.0 ×3, deterministic). JEV head-to-head unchanged where measured
(95.5% answered, n=44, ≤591 µs vs 70–500 ms) and extended where only syfox
can answer: proof traces, discovery queries, the thinker index, and the
axiom format are axes JEV does not implement or publish.

**Honest negatives, measured and disclosed.** Not every paper mechanism
helps at demo scale, and the suite says so: `--momentum-readout` alone moved
think30 0.600 → 0.533 (its peak-diffusion-gain form suppresses too much on
8-pass burst fields — kept as an instrument, not in the recommended stack);
`--coherence-gate` and `--vght-gate` alone measured 0.567; the learn-side
flags (`--fingerprints --sentinels --order-consolidation --pro-con`) changed
demo-scale fabrics for the worse (0.433 combined) and are documented as
corpus-scale features. `--cem-plan` alone is exactly neutral (0.600) and
becomes positive in combination. The CSAT boundary theorem is now documented
in the substrate header: global viscosity/decay modulation is rank-preserving
and can never change an arg-max readout — any readout-affecting mechanism
must couple per-node, which is why the settle physics stays frozen.

**Remaining non-ports (paper verdicts, unchanged):** BSMA (no mechanism —
the irreducibility fixpoint is already syfox's SettleTrace), CSAT (per-node
chaotic decay mutates frozen physics; global form proven readout-neutral),
liouville_analog/cr_bound (theory capstones), bmsa_new/prm/ntda/srg
(design-only or retracted), CMC/HRM/cb/SCER-doctrine (dead with their
substrates or incompatible with honest silence), ZeroGrad + ANQ LEA
(already absorbed into the lineage physics).

## License & credit

MIT © 2026 Mr-DS-ML-85. Core mechanics ported from the author's
[Synthetic-Intelligence](https://github.com/Mr-DS-ML-85/Synthetic-Intelligence)
research substrate. Inspired by the *interface* of TypeSafe AI's Jev — not
affiliated with, nor derived from, any TypeSafe model or weights.
