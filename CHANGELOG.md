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
