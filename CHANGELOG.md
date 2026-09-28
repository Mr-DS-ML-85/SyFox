## v3.8.0 — physics-native thinking (no token prediction, no ML, physics frozen)

Four opt-in mechanisms + the v3.7.1 scaling fix. Spec: deepen the field's
thinking without token prediction; SI core untouched; bit-identical replay on
default flags (bench verified against the v3.7 record).

- v3.7.1 DISTVEC SCALING FIX: measured O(n^1.9) build (76 s @ 5,595 nodes,
  156 s @ 7,985, ~15 h extrapolated @ 94,773) root-caused to the exact
  neighbour scan (O(n^2*k), cap 8000 -> 2000), strided row-major MGS with
  ~840k OpenMP regions per call (-> column-major blocked BCGS2), and the
  subspace/Ritz kernels (-> contiguous cm passes, cached init hashes).
  Measured: 17.1 -> 7.2 s @ 5,107 nodes; 95.4 s @ 40,850; ~15 h -> ~5 min
  extrapolated at the generalist size. Determinism md5-verified.
- TASK 1 ADAPTIVE DEPTH (`--adaptive-depth`, `--adaptive-max-k`): difficulty
  = unknown-token fraction + lane isolation, k = base + round(d*max_extra)
  (8..16 defaults), deterministic, ratchet-guarded, mirrored in
  harvest_rows; disclosed as difficulty/k_settle_used/k_settle_base.
- TASK 2 MULTI-VECTOR (`--multi-vector`, `--multi-vector-k`): K=8 spectral
  channel gates from the distvec axes (lane flow scaled by
  exp(-(u_k(a)-u_k(b))^2/2s^2)), frozen physics per channel, readout on the
  mean superposed field; honest no-op without --distvec; dead axes skipped.
- TASK 3 CAUSAL SIGNATURES (`--causal-lanes`): cause?>effect and
  means!>goal pair nodes from a deterministic connective scan (stemmed
  forms), interned/df-noted/bound like the bigrams, injected in
  decide/harvest/perturb at kCausalDose=0.5; replay no-op on old fabrics.
- TASK 4 ENERGY SELF-VERIFICATION: settle trace (passes, final/initial
  energy, relative motion rate) always in the decide usage; opt-in
  `--self-verify` defers choice answers measured on fields still moving
  above the floor (default 0.25, measured settled baseline 0.189-0.212),
  reason `unconverged_field`.
- TESTS: test_thinking group (adaptive depth incl. ratchet guard,
  self-verify gate both sides, multi-vector no-op/determinism/effect,
  causal emission directions + fabric interning + replay both ways); full
  suite green; readout-silence ALL PASS; jas-test ALL PASS; bench
  bit-identical (tickets 1.000/1.000 defer 0.063, game defer 0.333, guard
  0.778, OOD defer 1.0 x3, deterministic x3).
- MEASURED: think30 suite (`make think30` -> data/think30_tables.json) —
  30 tests, 6 categories x 5: baseline 0.467 (14/30) -> thinking flags
  0.600 (18/30) -> +multi-vector 0.633 (19/30). Honesty category 0/5 ->
  5/5 (noise answers on near-vocabulary OOD states now defer); chains
  0/5 -> 3/5. Causal categories showed no additional delta at demo scale
  (bigram lanes already carry the signal; disclosed honestly).

## v3.6.0 — the fabric-construction papers land + the JEV head-to-head

Implements the top of v3.5's paper port order on the v3.5.0 tree (settle
physics frozen; old fabrics replay bit-identically):

- ORDERED BIGRAM LANES (MRS SS4b.1-4b.2 measured 7/7; CMD SS2.5; CPSB; HTR
  arrival-order): learn interns adjacent token pairs ("w1~w2") and binds them
  to the outcome anchors; decide injects the state's bigram nodes at
  kBigramDose=0.5 of the state dose, mirrored bit-exactly in harvest_rows.
  Escape from the multiset-invariance theorem (paws 0.500 / xnli 0.333 are
  chance because a token bag cannot encode order). Old fabrics have no
  "w1~w2" nodes so the decide-side block is a no-op on them (replay
  contract, unit-tested both ways). --bigrams on|off (default on).
- DOCUMENT-FREQUENCY RARITY WEIGHTING (CMD SS2.2 log(N/(1+df)); MIMS SS6
  fan-out): substrate df table (one note per (lesson, node), persisted in a
  magic-guarded 'IDF5' tail) + hebbian_lesson scales STATE-side bind weights
  by log1p(L/(1+df))/log1p(L) in (0,1]. No stopword list; one counted
  formula. Fabrics without the tail reproduce v3.5 weights (factor 1.0).
- PERTURBATION-CONTRAST HONESTY CHECK (BED SS8, opt-in --perturb-check):
  decide() additionally settles a structure-broken copy of the state (same
  token multiset, adjacent pairs transposed, deterministic); when the broken
  state's margin matches the real one the readout was bag-carried, and
  choice questions defer with reason "perturbation_tie". Field snapshotted
  and restored — the readout measures exactly what an unchecked decide()
  measures. Usage discloses perturb_margin_real / perturb_margin_broken.
- MEASURED (train split, in-sample; heldout n=6/0/2): tickets choice
  0.875 -> 1.000 (16/16, defer 0), score 1.000 defer 0.063 unchanged; game
  choice_defer_rate 1.0 -> 0.333 with 4/4 answered correct; guard 0.778
  unchanged; ood_defer_rate 1.0 x3. test_fabric36: 22 new checks (bigram
  lay/inject, multiset-invariance pinned bit-equal on a bag fabric and
  broken on an ordered one, IDF same-token weaker-lane isolation, IDF5
  roundtrip, old-fabric replay bit-identity, perturbation defer/preserve).
  Full unit suite, readout-silence, jas gates ALL PASS.
- BUGFIX (fresh clones): `make ci` failed because its router/ablation gates
  need model-router16 / model-b77-sem / data/bank77_cal.jsonl — all
  gitignored. Those gates now SKIP loudly when absent; core gates (unit
  tests, readout-silence, jas — now wired explicitly) always run.
- JEV HEAD-TO-HEAD (tools/jev_bench.py, make jev-compare ->
  build/jev_comparison.json): JEV AI (TypeSafe AI, Sept 2026) is the closed
  "System One" decision model on the same API shape. Measured on JEV's
  published axes: workflow accuracy 95.5% (n=44 answered, train split) vs
  67.8% published; latency <= 474 us vs 70-500 ms published; $0 local vs
  $0.042/M input; plus the axes JEV does not publish — ECE (mean 0.131),
  honest defer with reasons (OOD 100%), bit-reproducibility, evidence JSON.
  Every syfox number measured by core/bench.hpp in this tree; every JEV
  number is their published self-reported figure (source cited in JSON).
- README v3.6 section + boundary updates (subject-object scope-escape by
  bigram lanes; temporal adjacency; perturbation defer). VERSION 3.5.0 ->
  3.6.0.

## v3.5.0 — JAS: the J-A-S cycle, the impossibility register, the arithmetic oracle

Ported from research-paper/jas.md + the central paper's substrate/thinker
division of labour; the SI settle physics stays frozen and decide() is
bit-identical (bench verified). Three additive CLI commands, one new test
group, one new CI gate.

- `core/jas.hpp` — the cycle: E labeled rows -> J the jump (counting leaps
  to a universal STRONGER than the evidence; axioms ranked
  confidence/support/token, deterministic) -> A->S deduction per holdout
  row -> experiment (oracle = observed label) -> refutation -> revision
  under a structural restriction (T AND U, fit on TRAIN only, never on its
  own counterexamples) -> round 2 on fresh rows. Verdicts:
  SURVIVES | REFUTED_AND_REVISED_SURVIVES_HELDOUT |
  REFUTED_AND_REVISION_ALSO_REFUTED | REFUTED_AND_REVISION_UNTESTED |
  REFUTED_IN_ROUND2_REVISE_NEXT_CYCLE | NO_TESTABLE_AXIOMS. Refutations
  append to <model>/refuted.axioms (with the counterexample row);
  survivors to <model>/verified.axioms (provenance INDUCED). The frozen
  field is scored on the same rows as the disclosed "relevance heuristic"
  column. Per-row experiment disclosure in JSON (predicted/actual/
  CONSISTENT|REFUTES|NO_PREDICTION, conflicts flagged).
- `syfox register [--model M] [--text]` — the impossibility register:
  five architectural boundaries seeded with warrants
  (2 THEOREM / 2 INDUCED / 1 DERIVED), each with parent + status + bypass;
  two THEOREM claims are SCOPE-ESCAPED (arithmetic -> calc oracle;
  subject-object -> ordered/typed lanes); persists
  <model>/impossibility.json.
- `core/calc.hpp` + `syfox calc --expr ... [--compile]` — the arithmetic
  derivation oracle: 6 primitives, one recursive-descent grammar, a
  verifier that rejects (1/0 errors), overflow-guarded exact integer path,
  word-problem mapping (times->*, plus->+, ...). --compile EMITS a C++
  translation unit, compiles with g++, runs it, discloses
  agreement + provenance established_by_experiment. MEASURED: 17*23=391
  (g++ agrees), 2+3*4=14, (2+3)^2/5=5, 2^10=1024, word problem -> 396,
  1/0 rejected. The field never computes — the theorem stands, the
  capability escaped its scope.
- Tests: unit group test_jas (oracle exactness, detection, register
  invariants, full cycle arc incl. refutation->revision->convergence,
  survival-is-not-proof); CI gate tests/jas_test.sh (make jas-test, wired
  into make ci; pins oracle values, register shape, verdict space,
  byte-determinism of cycle JSON x2); demo fabric data/jas_cycle_demo.jsonl.
- README: v3.5 section + "Why the 24-domain accuracy is low" table mapping
  each measured failure shape to its fabric-construction root cause and the
  paper that unlocks the fix (MRS/CPSB/RADE ordered+typed lanes, CMD IDF
  weights, rbg+graph_fisher Fisher-gated bridge lanes, BED perturbation
  negatives, l9 transitive pre-derivation); Architectural boundaries now
  cross-reference the register.
- MEASURED regression: bench bit-identical to v3.4.0 (tickets choice
  0.875 / score 1.000 defer 0.063; guard 0.778; game all-deferred
  choice_defer_rate 1.0; ood_defer_rate 1.0; deterministic x3). Full unit
  suite passes. VERSION 3.4.0 -> 3.5.0

## v3.4.0 — honest defer for ties, multi-hop readout walk, question-context gate

Three decision-layer mechanisms; the SI substrate physics (decay 0.82,
diffusion 0.45, K_settle 8, silence floors) stays frozen. Everything is
measured, deterministic, and disclosed.

**HONEST DEFER FOR NEAR-TIES (engine default ON, margin 0.05).** When the
top two probabilities are closer than the margin the readout does not carry
a decision: the answer defers with `reason: ambiguous_tie` instead of
shipping a coin-flip as a confident label. Engine-level (members
`defer_margin_`, `set_defer_margin`), so CLI, C API, HTTP bridge and bench
share ONE behavior — the v3.1 CLI post-pass (`low_margin`, default OFF) is
removed; `--defer-margin P` now sets the engine knob and `--no-defer`
restores the v3.3 answer-always behavior (both verified end-to-end on a
measured gap-0.186 fabric: default answers, `--defer-margin 0.25` defers
`ambiguous_tie`, `--no-defer` overrides). Score questions defer on the same
margin over level probabilities. MEASURED bench impact (train-split,
v3.3.1 -> v3.4.0): tickets choice 0.875 unchanged (defer 0), score 0.938 ->
1.000 with defer 0.063 (the one wrong score row was a near-tie, now honestly
deferred); guard 0.778 unchanged (defer 0); the 6-row game toy fabric
discloses as all-deferred (`choice_defer_rate` 1.0 — its decisions sit at
mean margin 0.023, i.e. they never carried signal; v3.3's 1.000 was
in-sample noise-picking that happened to be right). Bench JSON gained
`choice_defer_rate`/`score_defer_rate` (accuracy over ANSWERED questions,
defers counted separately — selective accuracy reads honestly). Unit tests:
5 legacy groups pin `set_defer_margin(0)` (bengali argmax, M3 contradiction
evidence, M3 ledger persistence, hierarchy, question-gate readout contract),
new group `test_defer_ties` pins the defer (near-tie defers with
probabilities intact, decisive row answers, margin 0 answers the same
near-tie, score flavor defers).

**MULTI-HOP READOUT WALK (opt-in, `--hops N`, default 1 = legacy).**
`Substrate::set_hop_depth` (clamped 1..8): depths > 1 walk lanes BFS-style
from each probe anchor up to N levels with per-hop damping
`hop_coupling/sqrt(hop+1)` (hop 1 x0.707 ... hop 4 x0.447), levels collected
in ascending NodeId order (std::map), width-capped at 64 nodes per level
(`kHopWidthCap`), never revisited — deterministic and bounded. Default
depth 1 keeps the legacy single-hop code path BIT-IDENTICAL (test-enforced:
default == explicit 1). MEASURED on chain probes (2-hop already answers at
N=1 — diffusion carries alarm->lights at probs 0.314 top, unchanged at N=4;
3-hop sparrow->animal target 0.1061 -> 0.1097; 4-hop relay far target echo
visible at 0.210 at N=4) — damped lanes keep far targets near-tied, so the
reported conf-0.019 noise answers now surface as honest `ambiguous_tie`
deferrals instead. Probe suite extended with the Test 18 (sparrow->bird->
animal) and Test 21 (alpha..echo 4-hop relay) analogs + hops configs;
unit group `test_multi_hop` pins default/clamp/bit-identity/determinism.

**QUESTION-CONTEXT GATE (opt-in, `--ctx-gate`, `--ctx-alpha F`).** Two-stage
settle: stage 1 settles the question's own tokens (instructions + criteria
descriptions, NO labels) into a context field; stage 2 re-settles the state
(prime + state + bridges, unchanged) and `Substrate::blend_field` composes
`e = (1-alpha)*state + alpha*context` (alpha default 0.5, clamped [0,1]).
Two ALREADY-SETTLED fields composed at the decision layer — settle() physics
untouched, deterministic, disclosed as `usage.ctx_gate`/`usage.ctx_alpha`.
MEASURED on the probe fabric: who-found answer holds (Tariq), confidence
0.056 -> 0.061; OOD submarine probe still abstains (`unknown_candidates`);
latest-temperature holds with a disclosed confidence tradeoff (0.007 ->
0.003). Unit group `test_ctx_gate` pins OFF-unchanged / ON-disclosure /
answer-holds / determinism / OOD-preserved.

**Surface.** CLI `--defer-margin P` (engine knob), `--no-defer`, `--hops N`
(1..8), `--ctx-gate`, `--ctx-alpha F`; C API decide opts `defer_margin`,
`no_defer`, `hops`, `ctx_gate`, `ctx_alpha`; bridge opts docs; usage JSON
`ctx_gate`/`ctx_alpha` disclosure. README: v3.4 section + "Architectural
boundaries (System-2 limits)" honest list. Verified: full unit suite passes
(incl. 3 new groups), `make readout-silence` ALL PASS, bench deterministic
x3 with OOD defer 1.0 x3. VERSION 3.3.1 -> 3.4.0.

## v3.3.1 — readout-silence threshold made explicit (BLOCKER fix)

**The reported false defer, root-caused on the shipped tree.** The report
said `{Tariq, Salma}` false-defers with `unknown_candidates` because the
readout threshold (~0.35) sits above a valid candidate's energy (0.301).
On the shipped v3.3.0 code the check is an implicit `> 0` — there is no
0.35 constant — and the case does NOT reproduce: on the QA probe fabric
(`model-reasoning-probe`), Test 2 `{Tariq, Salma}` ANSWERS `Tariq`
(max candidate 0.8759, Salma 0.4811, settled field 2.4594, all measured
with `SYFOX_DEBUG_READOUT=1`). The reported threshold matches a local
mid-v3.4 worktree state, not this tree. The hardening below landed
regardless, because the implicit `> 0` form has a real defect: a best
candidate carrying dust-level diffusion carryover (`0 < e < 0.01`)
was ANSWERED — a pick by noise — instead of deferred.

**Fix (Option B, absolute floor).** `kUnknownCandidateFloor = 0.01f` —
a named, documented, ABSOLUTE constant in `core/syfox.hpp`: defer with
`reason: unknown_candidates` when the BEST candidate's readout energy is
below 0.01. Deliberately absolute, not relative-to-field: a bright field
sitting entirely on non-candidate nodes must defer, and a valid candidate
on a dim field must answer. Measured guardrails: probe-fabric candidates
0.19–0.88 all answer (0.19 measured WITHOUT energy-norm — do not raise
the floor above 0.10); dark candidates 0.0000 still defer; the lit-field/
dark-candidates OOD shape (`purple`/`submarine`) still defers with
`unknown_candidates`; a fully dark field (`xyz abc def`) defers at the
decide level with `unknown_vocabulary` (floor 0.05) before readout.

**Debug visibility.** `SYFOX_DEBUG_READOUT=1` prints, on stderr: the
decide-level field (`settled_energy`, `silence_floor`) and, per question,
`max_candidate` + the active threshold + every candidate energy — a false
defer is now diagnosable from one run.

**Regression gate.** `tests/readout_silence_test.sh` pins 7 checks
(2-candidate and 3-candidate valid cases answer; lowercase answers; dim
energy-norm-OFF case answers; dark field defers with the decide-level
reason; lit-field/dark-candidates defers with the readout-level reason).
New `make readout-silence` target, wired into `make ci` (the script
builds its own probe fabric from the committed
`data/reasoning_probe_lessons.jsonl`). Verified: unit suite all pass;
`make bench` bit-identical to the pre-fix snapshot (tickets 0.875/0.938,
game 1.000, guard 0.778, `ood_defer_rate` 1.0 on all three,
`deterministic` true on all three). README honest-silence section now
documents both silence layers and their floors. VERSION 3.3.0 -> 3.3.1.

## v3.3.0 — question-conditioned readout, tie disclosure, readout silence

**The `--evidence` "empty model" bug — root-caused, already cured, better
hint.** The reported symptom (lanes 0, vocabulary 0, calibrated false, all
`unknown_vocabulary` with `--evidence`, healthy without) is reproduced
exactly on a v3.2.0 build with `decide --evidence --state ... --questions ...`
and NO `--model`: the harness that emitted the flag treated it as
value-taking and swallowed the following `--model` token, and v3.2.0's
silent load then decided on an empty default fabric — with the evidence
block in the output, which made the flag look guilty. The flag itself is a
plain boolean in both versions. v3.2.1's loud load failure already exits
with `syfox: decide model "model": cannot read model/substrate.bin`;
v3.3 adds a stderr hint when the failed dir is the default `model` value,
naming the swallow-the-argument failure pattern explicitly.

**Question-conditioned readout (opt-in) — ported from the original
Synthetic-Intelligence repo.** There, EVERY token of the turn drove the
activation field (`si_main.cpp`: `for (int id : matched)
physics.drive(id, 3.0f, 6.0f, 20)`) — the question was never a bystander.
SyFox's contract keeps state and question separate, and the readout only
measured state energy: the question's instructions gated nothing, so the
salient state noun beat the question-relevant entity ("Who found the
radio?" → `radio` over `Tariq`). The portable form gates the READOUT:
question tokens that are NEW relative to the state (the `who` in "Who found
the radio?") distribute lane mass over the candidate anchors, and each
candidate is scaled by its share (`floor + (1-floor) x r/max_r`), floor
0.25 default. Read-only over the settled field — same shape as the
hierarchy gate; disclosed as `question_gate: [tokens]` in every reply
path. **Measured (tools/reasoning_probes.py, QA-style probe fabric):**
gate ON doubles the who-found margin (0.056 -> 0.113 at floor 0.10), lifts
the latest-temperature confidence 14x (0.008 -> 0.117), and never flips a
correct answer off. **Measured regression that sets the default:** on
tickets-cal the gate costs 5.3 points (0.9533 -> 0.9000 — "which team"
lanes mislead), so the gate ships OPT-IN exactly like the M1 energy-norm:
`--question-gate` / `--question-gate-floor F` / opts `question_gate`,
`question_gate_floor`.

**Exact-tie disclosure + readout silence.** The tie-break truth (measured):
criteria iterate in `std::map` key order and `max_element` keeps the first
max, so exact ties go to the alphabetically-first candidate — the reported
"Salma wins (last position)" was S < T, not a `>=` bug. Two structural
tie-breakers were measured along the way: the v3.2 sem_hop term
deterministically separates differently-spelled labels (aaa/bbb with
IDENTICAL lane sums 0.679428518 still end at 0.500078 vs 0.499921 through
their different trigram vectors), and per-lesson lane decay makes
same-structure candidates from separate lessons near-ties rather than
exact ones. What was missing was HONESTY about ties: every reply now sets
`tied: true` when the top two probabilities are equal (near-ties remain the
`--defer-margin` knob's business). And when the readout measures ZERO
energy on every candidate, decide now defers with
`reason: "unknown_candidates"` instead of answering a uniform distribution
by criteria order — the same honest-silence contract the dark-field case
already followed (measured: the reported unknown-options test answered
`abc` at 0.344 by noise; it now defers; the OOD "submarine's favorite
color" probe defers in a fabric that never saw the entities).

**The five reported reasoning failure modes — status.** (1) multi-hop
chaining: diffusion already propagates through learned lanes (the
alarm→lights probe answers `lights`), 3+ hop composition stays weak;
(2) temporal ordering: the gate carries `latest`/`after` signal when the
fabric has those lanes — a training-side fix, tool-supported; (3)
arithmetic: architectural limit, unchanged in both repos; (4) OOD
abstention: the all-dark-readout defer fixes the pure case; the
state-lit-but-unanswerable case (purple 0.472) remains open — the state
salience override needs a mechanism the gate does not provide; (5)
subject-object relations: the old repo solves it with TYPED graph edges
(`Tariq -found-> radio`), which the bag-of-words core deliberately lacks —
architectural limit, documented.

Tools: reasoning_probes.py (probe fabric builder + before/after suite);
data/v33_reasoning_tables.json archives every number. Tests: 9 new checks
(gate on/off/disclosure/determinism, tie disclosure, criteria-order pick,
readout silence). VERSION 3.2.1 -> 3.3.0.

## v3.2.1 — the three paper-blocking bug fixes (ablation kill switches, confidence calibration, router config)

**BUG #1 — ablation kill switches were silently inert (root-caused, disclosed,
test-pinned).** The switches themselves were wired correctly all the way into
settle() and readout() — on a v3.2 fabric (model-b77-sem) the five paper
configs produce 4 distinct outputs. The reported byte-identical behavior is
the replay contract on a PRE-v3.2 fabric: no SEM4 tail => --no-semantics has
nothing to switch off, no memories.jsonl => --no-retrieval primes nothing, no
hierarchy.json (or floor 1.0 = the measured-off default) => --no-hierarchy
moves nothing. Fixes: (1) `decide` now prints a loud stderr note per inert
switch naming the missing feature and the rebuild path
(./build/rebuild_sem SRC DST); (2) tools/ablation_suite.py runs the five
configs over an eval set, reports per-config accuracy/meanC/medC plus an
answer-signature distinctness count, DISCLOSES inertness, and exits non-zero
when a v3.2-featured fabric collapses — a null ablation can never reach the
paper silently again; (3) unit test test_ablation_distinctness builds a fabric
with all three layers ARMED (hierarchy floor 0.35) and asserts >= 4 distinct
outputs across default/--no-semantics/--no-retrieval/--no-hierarchy on the
same input (got 4); (4) the semantic effect is measured-visible (tickets-cal
accuracy 0.9533 -> 0.9733 with --no-semantics), so no leak-fraction change.

**BUG #2 — confidence collapse (96.6% accuracy at mean confidence 0.0016)
root-caused: the calibration fit ran on a different physics composition and
dose regime than decide.** Two concrete engine bugs and one data fix:
(1) harvest_rows() re-implemented injection inline and SKIPPED the retrieval
priming block — every fabric WITH memories was calibrated on a no-prime field
while decide primes; the priming prologue is now a shared prime_field() used
by both paths, bit-identical to the old decide behavior; (2) the mode match:
calibrate must run under the same --energy-norm regime the fabric is benched
in (readout gaps are dose-dependent at fixed temperature — the v3.1.2
measurement of conf 0.002 -> 0.923 under energy-norm is the same lever);
(3) recalibration of the shipped v3.2 fabrics under the fixed paths.
Measured after: tickets-big-latin-sem mean confidence 0.0016 -> 0.8736
(median 0.9835, C|ok 0.898 vs C|bad 0.373, accuracy held 0.9533), easy cases
0.636-0.965, b77-sem C|ok/C|bad separation 2.3x with the semantic leak
verified as signal (accuracy collapses without it). Router16 recalibrated on
its anchor-worded cal rows (ECE 0.368 -> 0.099, in-distribution mean
confidence 0.298). Acceptance criteria met: avg >= 0.20 on a well-trained
domain, >= 0.50 on easy cases. tools/confidence_ablation.py produces the
per-config ablation table and the diagnosis (retrieval dose and semantic
leak are NOT the diluters — the stale fit was); tables archived in
data/v321_bugfix_tables.json.

**BUG #3 — `decide --router model-router16` failed with "lacks router.json
anchors".** Root cause: the router dir didn't exist in the caller's layout
(the HF package nests fabrics at model/<name>, the syfox repo at
model-<name>) and a missing substrate loaded SILENTLY as an empty fabric, so
the failure surfaced as the misleading router.json message. Fixes:
(1) Substrate::load and Engine::load_model report failure; every CLI command
now fails loudly naming the model dir it could not read (first learn stays
tolerant — a fresh fabric is intended there; the ctypes bridge returns a
null handle so Python raises); (2) resolve_model_dir() tries as-given,
model/<name>, model-<name>, model/model-<name> — one CLI invocation works in
both repo layouts, and router.json models mapping targets resolve the same
way; (3) the router.json error now names the expected file and the builder
tool; (4) router.json models mapping updated to the v3.2 -sem fabrics;
(5) router16 rebuilt (sorted save) and recalibrated; (6) `make ci` gate
includes a route-field assertion (tools/assert_route.py). Verified
end-to-end: "Why am I missing my refund" -> route rt01 -> model-b77-sem ->
c00 answered; route distribution across all 16 hidden domain files measured
(language anchors 12/12 perfect; bank77 rows route to the support anchor
rt13 11/12 — honest limitation, disclosed; modal agreement 0.663).

**PERSISTENCE DETERMINISM (found during the ablation investigation).**
save() iterated unordered_map members directly, so every load->save cycle
PERMUTED substrate.bin — measured: three different md5s over two rebuild
passes on one fabric with an identical lane multiset. Same-lane physics per
file, but the settled field's floating-point sum order drifted across save
cycles: model dirs were never byte-stable and an ablation re-run after a
re-save was silently a different experiment. Fix: every map-backed section
(out_, derived_gen_, lane_evidence_, lane_ctx_) is written in SORTED key
order; load rebuilds out_ in file order, so one sorted save converges the
format and every later roundtrip is bit-identical (rebuild_sem x3:
md5 06011b36... stable; unit-test enforced). All shipped v3.2 fabrics were
rebuilt through the converged path.

Tools: ablation_suite.py, confidence_ablation.py, router_routes.py,
assert_route.py, sub_diff.py; Makefile: `ci` gate + rebuild_sem now in
`all` (the stale-binary trap that masked the first sorted-save attempt).
Data: v321_bugfix_tables.json (every number above, archived).

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
