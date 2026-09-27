## v3.1.1 — the probe experiment: criteria wording is a first-class input; honest ties via --defer-margin

The model-xl diagnostic (handcrafted Bangla/English/SMS probes tying at
~0.50/0.50 with confidence 0 and deferred:false) reproduced and explained on
real data. Experiment: one fabric trained on real UCI SMS spam (5,574 rows,
stratified 70/15/15), the SAME hidden rows decided under three criteria
wordings:

| decide-side criteria | hidden top-1 | mean margin |
|---|---|---|
| matched (trained schema) | 0.160 | 0.444 |
| reworded (handcrafted, abstract) | 0.842 | 0.043 |
| empty (labels only) | 0.422 | 0.017 |

Findings (all measured, `tools/sms_probe_experiment.py`):
- The criteria text at decide selects which nodes the question probes.
  Matched-trained criteria can HURT: their state-overlapping words harvest
  direct hits (anchor-hub noise) and land worse than the majority baseline,
  with huge misleading margins. Abstract reworded criteria suppress the
  harvest and let the anchor one-hop (class-typicality) carry the decision —
  at tie-sized margins. Same fabric: accuracy swings 0.16 <-> 0.84.
- The model-xl probe pattern (near-tied probabilities, confidence 0,
  deferred:false) is the reworded/empty condition of this mechanism: the
  probes were measuring a DIFFERENT system than the benchmark number
  (SMS 0.937 was measured under the trained schema). Handcrafted probes
  bound; held-out runs conclude. Methodology written up in docs/PROBING.md.
- Decision-layer honesty: new `decide --defer-margin P` (and server
  `options.defer_margin`) discloses p1-p2 below P as `deferred: true,
  reason: "low_margin"` instead of a confident-looking label on tied
  candidates. The substrate still decides; physics untouched. Measured live:
  a 0.509/0.491 tie now returns a deferral, not a spam label.
- Packaging note (from the model-xl diagnostic): a fresh checkout has no
  build/ directory (gitignored) — run `make all` first; that also produces
  libsyfox_core.so for the HTTP server.

## v3.1.0 — banking77: real 77-way benchmark, the manual error anatomy, and the hub-pollution fixes

Real dataset, not synthetic: PolyAI banking77 — 10,003 official train rows /
3,080 official test rows / 77 intents (`tools/bank77_prepare.py` pulls the
CSVs). The official test split becomes the hidden test (firewall-enforced
naming); 10% of train is carved stratified (seed 77) for calibration.

### Measured fix ladder (selection on cal ONLY, hidden scored once)
| config change | cal top-1 |
|---|---|
| naive: generic criteria, instructions inline, natural labels | 0.029 |
| + mined per-class criteria (log-ratio, unigrams+bigrams, top-20) | (baseline run) |
| + empty instructions (learn folds instructions into the state side; identical per-row instructions are hub pollution — they are never injected at decide) | 0.041 |
| + opaque anchors `c00..c76` (natural labels like `card_arrival` tokenize into shared hubs; "card" spans many classes. Opaque ids give every class a DEDICATED anchor node) | 0.073 |
| + consolidation epochs 3 (reproduces the model-xl recipe on the clean schema) | 0.081–0.085 |
| + top-20 terms + anchor arbitration (flat top-5 re-ranked by bare-anchor readout — product of two readout views) | **0.091** |

HIDDEN (official test, 3,080 rows, scored once): **top-1 0.081, top-3 0.221,
top-5 0.326** vs chance 0.013 — 6.3× chance at top-1, 25× at top-5.
No cal overfit: cal predicted 0.091, hidden gave 0.081.

### What the manual benchmark exposed (the point of doing it by hand)
- Without `--energy-norm`, corpus-scale masses silence the readout: all 77
  options sit at exactly 1/77 and argmax degrades to alphabetical order.
  The M1 measurement gain is MANDATORY at this scale (0% -> 100% coverage).
- The error anatomy (`tools/bank77_analysis.py`): gold sits at ranks 2-20 for
  ~60% of rows; margin-when-correct is ~2x margin-when-wrong; a single class
  (`wrong_exchange_rate_for_cash_withdrawal`) absorbs wrong predictions from
  every other class via hub-word bigrams ("cash abroad" is class-pure as a
  bigram while "cash" is global as a word — the readout tokens are words).
- Measured NEGATIVE (kept off, documented): stopword filtering (0.036 —
  verb+stop combos like "was taken" are the signal), term-level purity caps
  (0.035), word-level purity caps (0.031-0.053), first-token hierarchical
  two-stage (0.051), shorter criteria under the clean schema, unigram-only
  criteria (0.045-0.052). The hub phrases are simultaneously the noise and
  the signal.
- Honest gap note: 0.081 hidden here is not model-xl's 0.284 — model-xl is a
  14-domain consolidated fabric (45,892 nodes / 1.9M lanes) whose mining
  recipe this sandbox rebuild does not replicate. What transfers is the
  SCHEMA (opaque anchors + empty instructions + mined criteria + energy-norm)
  and the measurement tooling: point `tools/bank77_analysis.py --model
  model-xl` at any hidden file to re-run the same anatomy there.

### Infrastructure
- `tools/bank77_prepare.py`: CSV -> train/cal/hidden JSONL (firewall naming),
  stratified carve, deterministic criteria mining, SHA-256 manifest.
- `tools/bank77_analysis.py`: the manual benchmark — per-row probabilities
  through the ctypes bridge; top-1/3/5, gold-rank bands, margin and
  confidence-when-wrong, worst confusion pairs, per-class accuracy, deferral
  anatomy, per-row JSONL dump. `--arbitrate` = anchor product-of-experts.
- Makefile: `bank77` (prepare + learn + cal-bench), `bank77-report` (hidden
  run). Bulk splits are gitignored (167 MB inline criteria); the tool
  regenerates them deterministically.

## v3.0.0 — milestones 1–5: scale discipline, distinct experience, provenance, adversarial suite, multicore

Every claim below is a number from a real run on this repo (commands in the
Makefile; raw JSON in `build/`). Nothing is projected or estimated.

### Milestone 1 — large held-out datasets, hidden-test firewall
- `tools/gen_corpus.py` (+`tools/corpus_pools.py`): deterministic generator,
  seed 20260927. Per domain — tickets_en 12,000 rows (8,400/1,800/1,800),
  game 8,000 (5,600/1,200/1,200), guard 8,000 (5,600/1,200/1,200); Bengali /
  Hindi / Russian tickets 3,600-ish each with hidden splits of 540 / 539 / 539
  (all ≥ the 500 the milestone requires). 70/15/15 train/cal/hidden, label
  distributions in `data/big/MANIFEST.json` with a SHA-256 per file.
- `core/firewall.hpp` (new): the hidden-test contract ENFORCED in code, not
  prose. `learn` refuses `_hidden`/`_cal`; `calibrate` refuses `_hidden`;
  `derive --gate` refuses `_hidden`. `make firewall-check` proves all four
  refusals on the real CLI.
- Hidden-test reports (`make bench-big`, `--energy-norm`, calibration fit on
  the `_cal` split only): **tickets_en 0.971** (ECE 0.005), **game 0.763**
  (ECE 0.049), **guard 0.638** (ECE 0.037; guardrail hold precision 0.782,
  recall 0.910). p95 latency 126–441 µs, decisions bit-deterministic.
- Multilingual hidden (`make bench-big-ml`, per-script routing):
  **Bengali 0.969** (ECE 0.015), **Hindi 0.965** (ECE 0.020),
  **Russian 0.970** (ECE 0.011).
- Coverage-vs-accuracy on the hidden tests (`make coverage-big`), all three
  operating points feasible: tickets 96.4%@53% / 90.1%@73% / 83.0%@93%;
  game 94.3%@52% / 86.7%@76% / 80.2%@91%; guard 76.2%@54% / 68.1%@80% /
  65.8%@90%. Dose-response (`make scale-probe`): 0.603@250 → 0.825@8,400
  distinct rows — distinct experience keeps paying at scale.

### Milestone 2 — distinct experience beats repeated phrasings (measured)
- `learn --dedup`: exact-duplicate lessons skipped, auditable via
  `dedup_skipped` in the learn report.
- The A/B at EQUAL lesson counts (25,200 each; `make ab-distinct`):
  **8,400 distinct rows 0.825** (39,158 lanes, 1,898 nodes) vs **1,050 rows
  re-taught 8× 0.782** (18,633 lanes, 530 nodes). Distinct experience wins
  and builds a 2.1× denser fabric. B-dedup (duplicates skipped) lands at
  0.609 with dedup_skipped=22,050 — consistent with the dose-response curve
  at ~1,000 rows, cross-checking both measurements.
- `--novelty` per-lesson novelty weighting: measured NEUTRAL at floor 0.90
  (0.827 vs 0.825 baseline) — ships OFF; dedup is the mechanism that pays.

### Milestone 3 — contradiction handling and machine-auditable evidence
- Every lane carries an evidence ledger: support events, counter events
  (anti-Hebbian dissolutions), generation, first/last teach seq, substrate
  context. `decide --evidence` prints it per channel — e.g. `department`
  resolved with 62 supporting lanes, top lane `the -> problem` w=0.2896,
  10,870 support events, seq window [1, 25198].
- Contradictions (same state+question taught with a different outcome) are
  DETECTED, counted in the learn report, and recorded — never silently
  overridden. Live test: teach `a` ×3 then `b` ×3 → decide keeps `a`,
  `contested: true`, contradiction record carries old/new outcome, both seqs,
  state hash; lanes show counter events. The ledger is read-only: lane
  weights drive the field, the ledger never does.

### Milestone 4 — adversarial/OOD suite (`bench --adversarial`, `make adv-big`)
Deterministic families on the hidden tests, honest per-family metrics:
- tickets: reorder 0.971, padding 0.968, typos 0.968, intensifiers 0.976,
  self-contradiction 0.969, negation 0.974, double-negation 0.942 (weakest),
  near-miss 0.933, cross-domain 0.908; **unknown concepts defer 100%**
  (zero false confidence — honest silence under attack).
- Conflicting-lessons attack (40 real contradictions taught post-hoc, with a
  control arm re-taught at gold to isolate the teach-event cost):
  silent_override = FALSE on all three domains, contested 40/40, decisions
  bit-deterministic after. Contradiction-specific damage on untouched rows
  (tickets): 0.7193 → 0.6517 vs control 0.7027 — the anti-Hebbian law costs
  ~5.1 pts beyond generic re-teaching, now measured instead of guessed.

### Milestone 5 — multicore settle, batched throughput, the GPU gate
- CSR mirror of the lane fabric (contiguous offsets/targets/weights, lazily
  rebuilt, order-preserving — every float sum keeps its exact operand order).
- OpenMP build (`make omp`, same sources, `-fopenmp`): parallel settle with
  static partitioning + per-thread scatter buffers combined in fixed thread
  order. **Bit-identity proven**: `make omp-identity` — sequential and
  `--threads 2` produce identical routing/calibration/honesty/guardrail
  blocks at 0.971 on the hidden test. `syfox-test-omp` all-pass.
- Honest performance note: at this fabric size on a 2-core sandbox the
  deterministic parallel settle is SLOWER (p50 620 µs vs 415 µs) — thread
  coordination exceeds the gain below ~10k nodes. It exists for dense
  fabrics; the numbers say when to use it.
- Batched multicore throughput (`bench --throughput N`, private engine per
  worker, checksum proves real work): 2,390 → 3,714 decisions/sec
  (**1.554×** on 2 cores) on the 3,518-node tickets fabric.
- GPU gate measured (`make density`): per-fabric density 0.0067 (tickets,
  3,518 nodes / 82,655 lanes, degree 23.5) — the SoA/CSR layout is in place;
  a GPU port stays DESIGN-ONLY until substrates are dense enough to justify
  it. No GPU claims.

### Fixed in v3 (found by running the milestones, not by reading)
- `si_substrate.hpp`: Substrate class was left unclosed by an interrupted
  edit (build broken); CSR cache members needed `mutable`.
- Parallel settle segfault: per-thread scatter buffers were constructed
  EMPTY (n vectors of size 0) → out-of-bounds combine. Now sized n.
- `omp-identity`/`throughput`/`density` targets silently measured an EMPTY
  fabric (missing `--lang auto` on a per-script-routed model) — the "ok"
  was vacuous. All three now load real substrates; identity re-proven at
  0.971, throughput checksum non-zero.

## v2.2.0 — multilingual boundary, trigram bridges, active learning

The language barrier was at the tokenizer: v2.1's byte-level `std::isalnum`
scan produced ZERO tokens for any multi-byte UTF-8 query — Bengali, Hindi,
Russian, every non-ASCII language deferred forever. v2.2 rebuilds the input
boundary (si_substrate.hpp still byte-identical to v0.2):

### Multilingual boundary
- `core/script.hpp` (new): UTF-8 decode + 35 Unicode script families by
  block-range majority vote (deterministic; ties by table order; digits
  neutral; malformed bytes are separators). Families cover 100+ languages.
- `core/normalize.hpp`: codepoint-aware tokenization. Pure-ASCII input keeps
  the byte-identical v2.1 path (test-guarded); non-ASCII decodes UTF-8 into
  script-homogeneous runs; Cyrillic/Greek/Latin lowercase maps; Porter +
  synonyms apply to pure-ASCII tokens only (Porter 1980 is English-only).
- `--lang auto|<slug>`: per-script routing. learn routes lessons into
  `<model>-<script>` substrates (one fabric per script family — Latin gets
  its own); decide/bench/calibrate/recall route queries, with an honest
  fallback note when the routed substrate is missing.
- Seed data: `data/tickets_bn_{train,heldout}.jsonl`, `tickets_hi_train.jsonl`,
  `tickets_ru_train.jsonl`, plus a disjoint-vocabulary `tickets_eval_extra.jsonl`
  (unseen words, known intents). `make models-multilingual && make bench-multilingual`.
- Measured: Bengali held-out routing 1.000 (n=3); Bengali train 1.000/1.000
  (in-sample); Hindi/Russian demos route and answer billing; OOD defer 1.00.

### Character trigram bridges (typo robustness, traction-gated)
- `core/ngram.hpp` (new): literal `g3:`-prefixed trigram lanes (no hashing —
  every lane inspectable), taught word<->gram fabric + gram<->outcome lanes.
- Decide-time protocol: an unknown word injects its trigrams only when >=25%
  already exist in the fabric (corrupted-form signature: "refnd" keeps
  g3:ref of "refund"); legitimate unseen words stay dark — honest silence,
  no manufactured evidence. Non-Latin routed substrates enable bridges
  automatically; `--ngrams on|off` overrides.
- Measured A/B: with the traction gate, clean Latin held-out numbers stay
  byte-identical to v2.1 (tickets 1.000 / game 0.667 / guard 0.750); the
  deterministic corruption sweep (`bench --typos P`) quantifies robustness
  honestly. bench OOD probes now also skip sub-word-entangled nonsense
  (no fake silence violations).

### Mass-guarded variant lessons (measured — still negative, documented)
- `learn --augment` + `tools/gen_variants.py` (8-10 variants/state,
  deterministic, refuses held-out paths): re-teaching skips intern() for
  known concepts, so lanes strengthen but acoustic mass never re-deposits.
- Held-out verdict: STILL regresses (tickets 1.000 -> 0.500, game 0.667 ->
  0.000, guard 0.750 -> 0.500 with 199 variants). With the mass confound
  removed the mechanism is visible: shared vocabulary across labels smears
  tiny multi-label fabrics. Coverage comes from distinct labelled experience
  — which the active-learning loop collects. `make models-vast` keeps the
  A/B reproducible.

### Active learning loop
- `decide --log-deferrals FILE`: every deferral logged with state + the FULL
  question schema (a labeled row is directly teachable).
- `syfox active --deferrals FILE --out FILE [--min-count N]`: dedups and
  ranks the log into a labeling worksheet. Deploy -> log -> label -> re-learn.

### Other
- `bench --typos P`: deterministic word corruption (deletion/swap/duplication
  by word hash; codepoint-level, works on Bengali/Hindi), reports clean vs
  corrupted accuracy + delta.
- VERSION 2.1.0 -> 2.2.0; tests: +5 suites (script table, UTF-8 boundary,
  trigram lanes, mass guard, Bengali e2e + determinism, corruption sweep).

# Changelog

All notable changes to SyFox. Versions follow the eval-methodology
milestones; every number in here is reproducible from the tagged sources
(`make models && make bench-heldout && make coverage && make heldout-gate`).

## v2.1.0 — evaluation-methodology fix + generalization work

The v0.2 eval graded the model on its own training rows (in-domain
resubstitution) and fitted its calibration the same way. v2.1 replaces the
methodology: a real held-out split gates everything, the headline metric is
the coverage-vs-accuracy curve, and the harness decides — by measurement —
what ships.

### P1 — held-out eval split (blocks everything else)

* `tools/split_data.py`: deterministic 70/30 split of every
  `data/*_train.jsonl` (which was the FULL dataset), stratified by first
  label, seeded RNG, refuses to double-split. Pre-split files remain in git
  history. Result: 31 train / 13 held-out rows; every label class present on
  both sides.
* `make models` now teaches the fabric from `*_train.jsonl` only.
* `syfox bench --split train|heldout` scores the right file and labels the
  source in `eval_source` (`held-out 30% split; fabric never taught from
  these rows` vs `train split; in-sample for the fabric`).

### P2 — coverage-vs-accuracy curve (the headline metric)

* `syfox bench --coverage-curve` sweeps a confidence threshold τ over
  0.0→1.0 (21 steps) and reports emitted / correct / coverage /
  accuracy-within-coverage, an ASCII accuracy-vs-coverage plot, explicit
  operating points at ≥50% / ≥70% / ≥90% coverage, and a JSON block.
* Engine-level honest silence (unknown vocabulary) is never emitted at any
  threshold — the curve measures the calibrated-confidence band on top of it.
* Choice + score questions on labelled rows; deferred answers stay out of
  every denominator.
* Tickets, held-out: 100% coverage → 83.3% accuracy; ≥70% coverage → 90.0%;
  50% coverage → 100%. The confident subset is trustworthy, measurably.

### P3 — paraphrase generation (implemented, then gated by measurement)

* `tools/gen_paraphrases.py`: up to 5 variants per taught state — clause
  reorders, neutral-politeness fillers, combos. Deterministic (per-row seeded
  RNG), refuses any held-out path, replaces words only via the single-word
  synonym table (never inserts domain-significant words), so labels stay
  valid by construction. Output: `data/*_paraphrased.jsonl` (155 lessons).
* Measured verdict on the held-out split: augmentation REGRESSES accuracy —
  tickets 1.000 → 0.500, guard 0.750 → 0.667, game 0.667 → 0.667 (1 variant
  per state is equally harmful). Mechanism: `intern()` mass grows linearly
  and injection damps by 1/√mass, so re-teaching near-duplicate lessons
  re-weights the field toward the training surface forms. A lesson here is a
  physical deposition, not a data point.
* Consequence: `make models` ships un-augmented; `make models-augmented`
  reproduces the augmented A/B. Synonym-only variants were dropped from the
  generator entirely (they normalize to identical token streams — pure
  repetition, zero novelty).

### P4 — tokenization improvement

* `core/normalize.hpp`: one deterministic pipeline for teach, decide,
  recall and harvest — lowercase alnum scan → Porter (1980) stemming (faithful
  port, verified against the reference vocabulary) → synonym folding via
  `data/synonyms.txt` (embedded copy as fallback; `--synonyms` overrides).
  "reimbursement" now lands on the same node as "refund" at decide time.
* `si_substrate.hpp` is byte-identical to v0.2: this is input encoding at the
  boundary, not physics. No neural net, no pattern matching added anywhere.
* Label-safety rule for the table: a swap must never be able to change a
  row's labels (no urgency words in fillers, no cross-department drift).

### P5 — derivation gated on held-out

* `syfox derive --gate data/*_heldout.jsonl` (per model, `make heldout-gate`):
  the no-regression gate now replays the HELD-OUT rows, and the report
  carries gold-labelled accuracy before/after plus a verdict and revert
  reason.
* Verdicts: tickets **REVERT** (9 held-out flips), guard **REVERT** (1 flip),
  game **PASS** (253 derived lanes committed, held-out accuracy held
  0.667 → 0.667). The §10-era mitigation ("ship seed models un-derived") is
  now a measured, per-model decision made by the harness.

### P6 — calibration on held-out + ECE

* `syfox calibrate --examples data/*_heldout.jsonl` (wired into
  `make models`): temperature/Platt fitted on held-out rows. The fabric never
  learns from held-out rows; the fit sets 2–3 scalars and is monotone, so
  argmax decisions are unaffected. Disclosed: the reported "after" ECE is
  measured on the same rows the scalars were fitted on (standard for
  temperature scaling; 1–2 parameters).
* Calibration rows now carry the FULL candidate energy vector + gold label;
  the fit objective is multi-class NLL over the true softmax (the textbook
  definition). The old pairwise {correct, worst-wrong} view was structurally
  blind to argmax errors on near-ties and could drive T to its floor,
  manufacturing confidence on wrong answers.
* 1-bit adoption guard per question type: keep T = 1 / default Platt when the
  fitted parameters would worsen held-out multi-class ECE.
* Two real defects fixed along the way: a 0.005 floor clamp inside the NLL
  search flattened the objective below 0.005 and hid the true optimum (the
  max-subtracted softmax is overflow-safe for any T > 0), and the same clamp
  in the decide path silently discarded any fitted T below it. Also fixed:
  array-criteria labels (score levels) were compared to level text and
  silently dropped from every fit.
* Results (held-out, multi-class ECE, top-prob confidence): tickets
  0.665 → **0.012** (T = 0.0005 adopted; the ECE < 0.1 bar is met), guard
  0.496 → 0.200 (T = 0.0107 adopted), game 0.329 kept (fit rejected — the
  fitted T would have worsened ECE to 0.503). With 1 wrong answer among
  3–4 emitted questions, ECE ≥ (1/n)·|0 − conf| ≥ 0.2 is forced by
  arithmetic at any temperature; the remedy is more held-out data, which the
  harness now measures.
* Bench ECE now uses the standard top-probability confidence (the signal
  temperature scaling acts on); the entropy confidence remains in the
  conf-when-correct/wrong fields.

### P7 — documentation

* README: held-out tables replace the resubstitution table (the old numbers
  are kept only as the labeled in-sample contrast), coverage-vs-accuracy
  table at 50/70/90% coverage, calibration-before/after table, augmentation
  dose-response table, per-model gate verdicts, latency + cost section
  (p50 29–64 µs, CPU-only, no per-decision API cost), tokenizer docs.
* `syfox --help` documents the new flags.

### Tests

* New checks: Porter spot-checks against the 1980 reference vocabulary,
  synonym folding through the stemmed table, normalize determinism.
* `test_calibration_tool` updated to the multi-class row shape and now
  asserts the fit sharpens on clean margins.
* Gate test: the harvest-revert branch compares against the pre-derive
  fabric state (a committed compose gate legitimately changes lanes first).
* Full suite: 107 checks, all passing. Substrate physics untouched:
  `git diff f4dba63 -- core/si_substrate.hpp` is empty — every change in
  v2.1 lives at the input boundary (normalize.hpp, new) or in tooling
  (bench / calibrate / gate reporting, Makefile, data split).

### Honesty notes

* The held-out split is small (13 rows total). Every number above carries its
  n. The harness exists precisely so that growing `data/` makes the numbers
  *mean more* without changing any code.
* Held-out rows touch two consumers beyond eval: the calibration fit (2–3
  scalars, disclosed above) and the derivation gate (read-only replay). No
  held-out row ever strengthens a lane. The coverage curve, accuracy, and
  flips are computed from decisions only.
* Jev's published figures remain side-by-side references (`jev_reference`);
  no parity is claimed beyond sharing the measurement axes.

## v0.2.0 — SI salience lineage + Jev-parity bench + gated derivation

* Real SI salience port: cavity integrator (s = tanh(s·0.995 + 0.05·|v|)),
  TSDA-lineage Miller window behind runtime flags (`--salience-gating`,
  `--miller-window`), state-hash-derived draws for bit-exact determinism.
* Jev-parity benchmark suite (accuracy, latency, calibration, honesty,
  guardrail, determinism axes) with published Jev figures carried for
  side-by-side reading.
* Derivation layer: lane induction (compose + harvest), dreamer with
  human-validated promotion ledger, structural analogy (energy fingerprints).
* No-regression derivation gate: transactional derive with bit-exact
  snapshot/restore; any taught-row argmax flip or manufactured certainty on
  ambiguous probes reverts.

## v0.1.0 — System One decision engine on SI substrate physics

* Energy injection → dissipative settling → dual-channel readout
  (Specific mean / Support mass) → Hebbian learning + anti-Hebbian
  dissolution, honest silence below the floor.
* Energy-gated source cap (24 sources/pass), deterministic tie-breaking.
* CLI: learn / calibrate / decide / demo / stats.
