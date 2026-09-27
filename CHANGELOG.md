## v3.2.0 — the semantic layer (Stages 1-4), retrieval by default, the two-stage physics router, the dedicated bank77 fabric

**The semantic field (core, deterministic, no ML).** Every concept node
carries a 64-dim semantic vector: signed character-trigram hashing of the
string (Stage 1 "concept frequency encoding" generalizes to a full vector)
plus two fabric-grounding passes over the Hebbian lanes — meaning diffuses
along the fabric, nothing is fitted. From the vectors: omega_semantic (fixed
scalar projection), top-6 resonance edges (cos >= 0.50, persisted in a
magic-guarded v4 substrate.bin tail), and typed reads (cos >= 0.82
synonym-grade, >= 0.66 related — Option 2 derived from the vectors rather
than a hand table). At settle, sources leak sem_coupling 0.12 of their
post-diffusion energy along resonance edges, distributed by cos x
frequency-match — ENERGY CONSERVED exactly (test-enforced); readout gains a
semantic-neighbour term (sem_hop 0.10). **Context-sensitive lanes (Stage
2)**: lanes accumulate required/forbidden context words from the lessons
that laid them; at settle a mismatched lane carries less (forbidden x0.20,
missing-required x0.60..1.0 by match count, all-1.0 bit-identical to the
plain path) — contextual disambiguation as dampened coupling, not rules.
**Stage 3 semantic hierarchy** ships as hierarchy.json (intents/categories/
floor) and gates intent candidates by stage-1 category energies at readout.
**Replay contract**: pre-v3.2 files end at the v3 tail, semantics inert,
bit-for-bit replay (test-enforced); intern() invalidates a stale field
(segfault fix caught by the M3 ledger test); --no-semantics is the runtime
kill switch. **Retrieval by default**: a memories.jsonl in the model dir is
fingerprinted once at load; every decide primes the field with the top-5
resonating memories' outcomes at dose 0.30 x inject (deterministic,
disclosed in usage.retrieval); --memories/--retrieval-topk/--retrieval-dose/
--no-retrieval.

**Dedicated bank77 fabric (the v3.0 architecture, now with physics that
match it).** 9,000 opaque-anchor intent lessons + 9,000 category lessons
(12 chunks, zero false contradictions after de-colliding the twins),
1,648 nodes / 217,729 lanes / 7,728 resonance edges / 141,920 context-signed
lanes / 256 retrieval memories. Hidden 3,080 rows scored once (energy-norm):
**0.158 top-1** vs 0.081 (v3.1.0 dedicated+opaque) and 0.023 (77-way in the
shared giant fabric). Ablations on cal (1,003): --no-semantics 0.039 (the
semantic field is the jump), retrieval neutral, hierarchy gate
measured-negative (floor 0.35: 0.111 vs gate-off 0.133; monotone in the
floor sweep) — the shipped model carries floor 1.0 = gate off, documented.

**Two-stage physics router.** tools/router_prepare.py carves a
router-question-only sample (16 domains x 60 rows) into a dedicated small
fabric: model-router16, 4,742 nodes, cal 16-way routing **0.444** (chance
0.0625), ECE 0.380 -> 0.086. `decide --router model-router16 ...` settles
the state on the router fabric, then the mapped domain fabric decides
(router.json anchors/models); the route is disclosed as route:{anchor,
confidence,top,model}. Bank77 queries route to the dedicated fabric; all
other domains to the giant fabric.

**Zero-shot re-run under v3.2** (docs/ZEROSHOT.md Part 4,
tools/zeroshot_suite.py, data/zeroshot_v32_results.json): email spam on the
tickets-only fabric 6/6 plain and 6/6 semantic-rebuild (conf 0.91-0.99);
SMS spam 5/6 both (the error at conf 0.103 — honest low-conf disclosure);
snake 1/8 both; tic-tac-toe picks cell 1 at conf 0 — multi-constraint
composition fails exactly as in Part 1, and the semantic field does not
fake it (honest scope: it moves trained discrimination, not symbolic
reasoning).

**Surface.** CLI: --router, --no-semantics, --no-retrieval, --no-hierarchy,
--retrieval-topk, --retrieval-dose, decide --memories FILE; stats reports
semantics/sem_edges/lane_contexts/retrieval/hierarchy. C API + HTTP bridge:
set_semantics/set_retrieval/set_retrieval_topk/set_retrieval_dose/
set_hierarchy + decide_ex opts keys + usage.retrieval + engine info fields.
Tests: 4 new groups (semantic field determinism/conservation/round-trip/
pre-v3.2 contract, context lanes learn+gate, retrieval-by-default +
kill switch + inert-without-memories, hierarchy load+gate). Tools:
bank77_hier.py, router_prepare.py, zeroshot_suite.py, rebuild_sem.cpp;
Makefile b77-hier-data/b77-sem/b77-sem-bench/b77-sem-hidden/router-data/
router16/router16-bench/zeroshot-sem/zeroshot. Core physics (injection,
settle lane diffusion, Hebbian learning, honest silence) unchanged where
the semantic tail is absent.

## v3.1.3 — routed-learn isolation bugfix; the 16-domain giant model + syfox-hf packaging

**Bugfix (CLI, boundary layer):** `learn --lang auto` taught EVERY script
family the ENTIRE corpus — the routed loop iterated all rows instead of the
family's rows, so each per-script substrate (`<model>-<slug>`) was a full
copy of everything and the "one fabric per script" isolation contract was
broken since v2.2. Fixed: families now teach only their own rows (epochs
applied per family). All tests pass; multilingual fabrics should be
retrained — the giant model below was trained post-fix, and its
bengali/devanagari/cyrillic substrates (546-680 nodes) answer their
held-out tickets at 0.920-0.959 with a trivially perfect in-substrate
router.

**The giant model** (built in the sandbox, packaged in the separate
`syfox-hf` repo): 16 domains — bank77 (9k), UCI SMS (3.9k), the 6 data/big
corpora (27.2k), plus 8 newly downloaded real HF datasets (CLINC-150,
enron_spam, emotion, Davidson hate, ag_news, amazon_polarity, dbpedia_14,
boolq) — 76,142 train rows / 152,284 lessons per epoch, 278 globally unique
opaque anchors, empty instructions everywhere, round-robin domain
interleave, mined top-10 criteria for the new domains, per-script
calibration. One interleaved pass: giant-latin 43,420 nodes / 1,905,170
lanes. Hidden (scored once, energy-norm on): tickets_en 0.904, sms 0.752,
enron 0.752, hate 0.653, agnews 0.604, polarity 0.571, dbpedia 0.475, boolq
0.531 (majority-level), bank77 0.023 (77-way collapses in a shared fabric);
bn 0.943 / hi 0.920 / ru 0.959 on their own substrates. Router measured
under THREE conditions: informed 0.432 (weighted, the bench condition),
blind 0.400 hidden (23,528 rows) / 0.399 cal, all-questions 0.399 cal —
asking everything dilutes the anchoring instead of recovering it. Router
wording sweep (cal only): short4 0.426 > abstract 0.208 > mid8 0.164 >
rich12 0.079 — richer specific criteria HARVEST worse (PROBING.md
mechanism). Tooling: tools/hf_fetch.py (cached parquet fetcher),
tools/giant_prepare.py (corpus builder with the three failure-mode fixes),
tools/giant_bench.py (router+class+e2e per-row bench, slim-format aware),
tools/router_sweep.py, tools/wrapper_condition_experiment.py; Makefile
targets giant-fetch/-prepare/-chunks. data/giant bulk gitignored
(schemas/probes/MANIFEST committed).

## v3.1.2 — zero-shot probe reports archived; the runtime-flag mechanism measured

The model-xl zero-shot experiments (Snake 8-scenario suite + the A–E flag
experiment, Tic-Tac-Toe symbolic composition, file-organizer routing) are
archived in `docs/ZEROSHOT.md`, and the report's central observation —
winner flips at constant settled energy, Miller-window confidence 0.805 on
an illegal move — is now mechanism-explained in code and re-measured on
repo fabrics with real held-out rows (`tools/flag_probe_experiment.py`,
2,638 hidden rows, trained schema, deterministic):

- **Source modes don't touch the field.** With the injection dose held
  constant, settled energy is bit-identical across salience/miller configs
  (spread ≤ 0.001 on 99.5% / 99.8% of rows) because a settle pass conserves
  energy up to uniform decay — source gating only decides which nodes
  propagate ("gated nodes keep their energy (readout-visible) but stay
  silent as sources", `si_substrate.hpp settle()`). The same invariance is
  why the model-xl A–E run reported 83.928 in every amplified config.
- **Energy-norm DOES change the field** (injection dose): 0% of rows keep
  their settled energy vs baseline. On trained schemas it is pure
  amplification — tickets-big flips 0.1% of winners at identical 0.971
  accuracy while margins rise 0.0355 → 0.9430 and confidence 0.002 → 0.923;
  baseline deferrals 35 → 0 (the documented M1 purpose).
- **Flags are diagnostic instruments, not defaults.** Salience gating is
  fabric-dependent: −4.1 points on tickets-big (0.971 → 0.930), +19 points
  on sms-natural (0.160 → 0.360) where the full-source readout is dominated
  by harvest noise. High-confidence-wrong rows (conf ≥ 0.5 while wrong):
  0 at baseline on both fabrics → 15–86 under amplified configs.
- Zero-shot and trained-domain results stay in separate buckets; the
  interpretation rules (confidence = concentration, never correctness;
  `--defer-margin` for honest ties) are in ZEROSHOT.md §Part 3 and
  PROBING.md §6.

Also: the four SMS probe data files (`data/sms_train.jsonl`,
`sms_hidden_match/reworded/empty.jsonl`) are now committed so the v3.1.1
probe experiment and this flag experiment reproduce from a fresh clone.

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
