# SyFox — System One decision engine on an SI substrate core
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD := build

all: build/syfox build/syfox-test build/libsyfox_core.so build/rebuild_sem

build:
	mkdir -p build

CORE_HDRS := core/syfox.hpp core/si_substrate.hpp core/normalize.hpp core/ngram.hpp core/script.hpp core/derive.hpp core/bench.hpp core/gate.hpp core/recall.hpp core/json.hpp core/firewall.hpp

build/syfox: src/syfox_cli.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

build/syfox-test: src/syfox_test.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

# v3 Milestone 5: deterministic parallel-settle build (OpenMP, fixed
# reduction order). Bit-identical to the sequential build by construction and
# by test (syfox-test-omp proves it on the probe fabric).
build/syfox-omp: src/syfox_cli.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -fopenmp -I. $< -o $@

build/syfox-test-omp: src/syfox_test.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -fopenmp -I. $< -o $@

build/libsyfox_core.so: src/syfox_core.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -fPIC -shared -I. $< -o $@

# ---------------------------------------------------------------------------
# v2.1 model build (P1 + P6 of the eval-methodology fix):
#   * the fabric learns ONLY from data/*_train.jsonl — never from
#     *_heldout.jsonl (P1), and (measured) never from the paraphrase
#     augmentation by default: see P3 note below and README "augmentation
#     dose-response"
#   * temperature/Platt scalars are fitted on the HELD-OUT split (P6) — the
#     fabric sees no heldout state; the fit sets 2-3 scalars and is monotone,
#     so argmax decisions are unaffected
#   * derivation is NOT run here: `make heldout-gate` runs it per model under
#     the no-regression gate against the heldout rows (P5)
# Deterministic: every model is rebuilt from its seed data, not grown in place.
# ---------------------------------------------------------------------------
models: build/syfox
	rm -rf model-tickets model-game model-guard
	@# v2.2: taught WITH trigram lanes (fabric carries the sub-word bridges).
	@# Decide-time default is word-level: `make bench-heldout` reproduces the
	@# v2.1 numbers exactly; bridges activate via --ngrams on / --lang auto.
	./build/syfox learn --ngrams on --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox learn --ngrams on --model model-game --examples data/game_train.jsonl
	./build/syfox learn --ngrams on --model model-guard --examples data/guard_train.jsonl
	@# P6: calibration fit on held-out rows (scalars only; argmax unaffected)
	./build/syfox calibrate --ngrams on --model model-tickets --examples data/tickets_heldout.jsonl
	./build/syfox calibrate --ngrams on --model model-game --examples data/game_heldout.jsonl
	./build/syfox calibrate --ngrams on --model model-guard --examples data/guard_heldout.jsonl

# ---------------------------------------------------------------------------
# P3 reproduction: the same build WITH the paraphrase augmentation applied to
# training (train + *_paraphrased.jsonl, 5 variants per taught state). The
# heldout harness MEASURED this recipe regressing accuracy on every domain
# (tickets 1.000 -> 0.500, guard 0.750 -> 0.667, game 0.667 -> 0.667): the
# substrate damps re-exposed tokens by 1/sqrt(mass) and intern() counts
# occurrences linearly, so re-teaching near-duplicate lessons re-weights the
# field toward the training surface forms. Kept as a target so the A/B is
# reproducible; NOT the default build. Details: README "augmentation
# dose-response".
# ---------------------------------------------------------------------------
models-augmented: build/syfox
	rm -rf model-tickets model-game model-guard
	./build/syfox learn --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox learn --model model-tickets --examples data/tickets_paraphrased.jsonl
	./build/syfox learn --model model-game --examples data/game_train.jsonl
	./build/syfox learn --model model-game --examples data/game_paraphrased.jsonl
	./build/syfox learn --model model-guard --examples data/guard_train.jsonl
	./build/syfox learn --model model-guard --examples data/guard_paraphrased.jsonl
	./build/syfox calibrate --model model-tickets --examples data/tickets_heldout.jsonl
	./build/syfox calibrate --model model-game --examples data/game_heldout.jsonl
	./build/syfox calibrate --model model-guard --examples data/guard_heldout.jsonl

# ---------------------------------------------------------------------------
# P5: transactional derivation, gated on the HELD-OUT rows. Pass => derived
# lanes committed (accuracy_before/after reported); revert => the fabric is
# restored bit-for-bit and the model ships un-derived with a logged reason.
# ---------------------------------------------------------------------------
# Order matters for honest measurement: `make heldout-gate` rebuilds fresh
# models and applies the gates; `bench-heldout` / `coverage` deliberately do
# NOT depend on `models` (a rebuild would wipe the committed derivation) —
# they measure the CURRENT on-disk model state.
heldout-gate: build/syfox models
	./build/syfox derive --model model-tickets --gate data/tickets_heldout.jsonl --examples data/tickets_train.jsonl | tee build/gate-tickets.json
	./build/syfox derive --model model-game --gate data/game_heldout.jsonl --examples data/game_train.jsonl | tee build/gate-game.json
	./build/syfox derive --model model-guard --gate data/guard_heldout.jsonl --examples data/guard_train.jsonl | tee build/gate-guard.json

# ---------------------------------------------------------------------------
# v2.2 variant-augmented build ("vast data"): train lessons first, then the
# generated variant rows re-taught with --augment (mass-guarded: lanes
# strengthen, acoustic mass unchanged — the v2.1 re-deposition artifact is
# structurally impossible in this recipe). Eval data is untouched: calibrate
# still fits only on the held-out split, bench still scores it read-only.
# ---------------------------------------------------------------------------
models-vast: build/syfox
	rm -rf model-tickets model-game model-guard
	./build/syfox learn --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox learn --model model-tickets --examples data/tickets_expanded.jsonl --augment
	./build/syfox learn --model model-game --examples data/game_train.jsonl
	./build/syfox learn --model model-game --examples data/game_expanded.jsonl --augment
	./build/syfox learn --model model-guard --examples data/guard_train.jsonl
	./build/syfox learn --model model-guard --examples data/guard_expanded.jsonl --augment
	./build/syfox calibrate --model model-tickets --examples data/tickets_heldout.jsonl
	./build/syfox calibrate --model model-game --examples data/game_heldout.jsonl
	./build/syfox calibrate --model model-guard --examples data/guard_heldout.jsonl

models-vast-full: build/syfox
	rm -rf model-tickets model-game model-guard
	./build/syfox learn --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox learn --model model-tickets --examples data/tickets_expanded_full.jsonl --augment
	./build/syfox learn --model model-game --examples data/game_train.jsonl
	./build/syfox learn --model model-game --examples data/game_expanded_full.jsonl --augment
	./build/syfox learn --model model-guard --examples data/guard_train.jsonl
	./build/syfox learn --model model-guard --examples data/guard_expanded_full.jsonl --augment
	./build/syfox calibrate --model model-tickets --examples data/tickets_heldout.jsonl
	./build/syfox calibrate --model model-game --examples data/game_heldout.jsonl
	./build/syfox calibrate --model model-guard --examples data/guard_heldout.jsonl

# ---------------------------------------------------------------------------
# v2.2 multilingual: one SI substrate per script family, built by routing.
# Latin lessons route to model-tickets-latin, Bengali to -bengali, Hindi
# (Devanagari) to -devanagari, Russian (Cyrillic) to -cyrillic.
# ---------------------------------------------------------------------------
models-multilingual: build/syfox
	rm -rf model-tickets-latin model-tickets-bengali model-tickets-devanagari model-tickets-cyrillic
	./build/syfox learn --model model-tickets --lang auto --examples data/tickets_train.jsonl
	./build/syfox learn --model model-tickets --lang auto --examples data/tickets_bn_train.jsonl
	./build/syfox learn --model model-tickets --lang auto --examples data/tickets_hi_train.jsonl
	./build/syfox learn --model model-tickets --lang auto --examples data/tickets_ru_train.jsonl
	./build/syfox calibrate --model model-tickets-latin --examples data/tickets_train.jsonl
	./build/syfox calibrate --model model-tickets-bengali --examples data/tickets_bn_train.jsonl
	./build/syfox calibrate --model model-tickets-devanagari --examples data/tickets_hi_train.jsonl
	./build/syfox calibrate --model model-tickets-cyrillic --examples data/tickets_ru_train.jsonl

# Held-out discipline for Bengali (train + heldout exist); hi/ru are demos.
bench-multilingual: build/syfox models-multilingual
	./build/syfox bench --model model-tickets --lang auto --eval data/tickets_bn_heldout.jsonl | tee build/bench-bn-heldout.json
	./build/syfox bench --model model-tickets --lang auto --eval data/tickets_bn_train.jsonl | tee build/bench-bn-train.json

# ---------------------------------------------------------------------------
# v2.2 typo-robustness sweep: deterministic corruption, grams ON vs OFF
# (the A/B that measures what the trigram lanes actually buy).
# ---------------------------------------------------------------------------
typos: build/syfox models
	./build/syfox bench --ngrams on --model model-tickets --split heldout --typos 40 | tee build/typos-tickets-grams.json
	./build/syfox bench --ngrams off --model model-tickets --split heldout --typos 40 | tee build/typos-tickets-nograms.json
	./build/syfox bench --model model-tickets --lang auto --eval data/tickets_bn_heldout.jsonl --typos 40 | tee build/typos-bengali.json

# Disjoint-vocabulary generalization eval (unseen words, known intents).
bench-extra: build/syfox
	./build/syfox bench --model model-tickets --eval data/tickets_eval_extra.jsonl | tee build/bench-extra-tickets.json
	./build/syfox bench --model model-tickets --ngrams off --eval data/tickets_eval_extra.jsonl | tee build/bench-extra-tickets-nograms.json

test: build/syfox-test
	./build/syfox-test

# v3.3.1 readout-silence regression: valid candidates answer, true unknowns
# defer with the layer-specific reason (builds the QA probe fabric on demand).
readout-silence: build/syfox
	bash tests/readout_silence_test.sh

# Jev-parity eval suite on the three seed models.
#   make bench           -> train split (in-sample; labelled as such)
#   make bench-heldout   -> held-out split (the honest headline number)
#   make coverage        -> coverage-vs-accuracy curves on the held-out split
bench: build/syfox models
	./build/syfox bench --model model-tickets --split train | tee build/bench-tickets.json
	./build/syfox bench --model model-game --split train | tee build/bench-game.json
	./build/syfox bench --model model-guard --split train | tee build/bench-guard.json

bench-heldout: build/syfox
	./build/syfox bench --model model-tickets --split heldout | tee build/bench-heldout-tickets.json
	./build/syfox bench --model model-game --split heldout | tee build/bench-heldout-game.json
	./build/syfox bench --model model-guard --split heldout | tee build/bench-heldout-guard.json

coverage: build/syfox
	./build/syfox bench --model model-tickets --split heldout --coverage-curve | tee build/coverage-tickets.txt
	./build/syfox bench --model model-game --split heldout --coverage-curve | tee build/coverage-game.txt
	./build/syfox bench --model model-guard --split heldout --coverage-curve | tee build/coverage-guard.txt

demo: models
	@echo "=========== TICKETS ==========="
	./build/syfox demo --domain tickets --model model-tickets
	@echo "============ GAME ============="
	./build/syfox demo --domain game --model model-game
	@echo "=========== GUARD ============="
	./build/syfox demo --domain guard --model model-guard

serve: build/syfox models
	python3 server/app.py --model model-tickets --port 8010

clean:
	rm -rf build

.PHONY: all models models-augmented models-vast models-vast-full models-multilingual bench-multilingual typos bench-extra heldout-gate test bench bench-heldout coverage demo serve omp adv-big throughput density clean

# ---------------------------------------------------------------------------
# v3 Milestone 1 — big-corpus models (data/big, 70/15/15 splits).
#   learn    reads ONLY *_train.jsonl   (firewall refuses _cal/_hidden)
#   calibrate fits on *_cal.jsonl       (firewall refuses _hidden)
#   hidden   scored by bench-big only   (never taught/calibrated/derived)
# Milestone-2 policy: --dedup skips exact duplicate lessons (auditable via
# dedup_skipped; 0 on the distinct corpus). --novelty measured slightly
# negative at floor 0.25 (-1.0pt), neutral at 0.90 — ships OFF; the A/B
# target below reproduces the measurement.
# ---------------------------------------------------------------------------
models-big: build/syfox
	rm -rf model-tickets-big model-tickets-big-latin model-tickets-big-bengali \
	        model-tickets-big-devanagari model-tickets-big-cyrillic \
	        model-game-big model-game-big-latin model-guard-big model-guard-big-latin
	./build/syfox learn --ngrams on --dedup --lang auto --model model-tickets-big --examples data/big/tickets_en_train.jsonl
	./build/syfox learn --ngrams on --dedup --lang auto --model model-game-big --examples data/big/game_train.jsonl
	./build/syfox learn --ngrams on --dedup --lang auto --model model-guard-big --examples data/big/guard_train.jsonl
	./build/syfox calibrate --energy-norm --lang auto --model model-tickets-big --examples data/big/tickets_en_cal.jsonl
	./build/syfox calibrate --energy-norm --lang auto --model model-game-big --examples data/big/game_cal.jsonl
	./build/syfox calibrate --energy-norm --lang auto --model model-guard-big --examples data/big/guard_cal.jsonl

# Bengali / Hindi / Russian substrates (routed per script family); each
# hidden split carries 540 rows (>= the 500 the milestone requires).
models-big-ml: models-big
	./build/syfox learn --lang auto --dedup --model model-tickets-big --examples data/big/tickets_bn_train.jsonl
	./build/syfox learn --lang auto --dedup --model model-tickets-big --examples data/big/tickets_hi_train.jsonl
	./build/syfox learn --lang auto --dedup --model model-tickets-big --examples data/big/tickets_ru_train.jsonl
	./build/syfox calibrate --energy-norm --lang auto --model model-tickets-big --examples data/big/tickets_bn_cal.jsonl
	./build/syfox calibrate --energy-norm --lang auto --model model-tickets-big --examples data/big/tickets_hi_cal.jsonl
	./build/syfox calibrate --energy-norm --lang auto --model model-tickets-big --examples data/big/tickets_ru_cal.jsonl

# Milestone-1 headline: HIDDEN-TEST report, full timing/determinism settings.
bench-big: models-big
	./build/syfox bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl | tee build/bench-hidden-tickets-en.json
	./build/syfox bench --energy-norm --lang auto --model model-game-big --eval data/big/game_hidden.jsonl | tee build/bench-hidden-game.json
	./build/syfox bench --energy-norm --lang auto --model model-guard-big --eval data/big/guard_hidden.jsonl | tee build/bench-hidden-guard.json

bench-big-ml: models-big-ml
	./build/syfox bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_bn_hidden.jsonl | tee build/bench-hidden-tickets-bn.json
	./build/syfox bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_hi_hidden.jsonl | tee build/bench-hidden-tickets-hi.json
	./build/syfox bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_ru_hidden.jsonl | tee build/bench-hidden-tickets-ru.json

# Coverage-vs-accuracy on the hidden tests (the headline metric).
coverage-big: models-big
	./build/syfox bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl --coverage-curve | tee build/coverage-hidden-tickets-en.txt
	./build/syfox bench --energy-norm --lang auto --model model-game-big --eval data/big/game_hidden.jsonl --coverage-curve | tee build/coverage-hidden-game.txt
	./build/syfox bench --energy-norm --lang auto --model model-guard-big --eval data/big/guard_hidden.jsonl --coverage-curve | tee build/coverage-hidden-guard.txt

# ---------------------------------------------------------------------------
# v3 Milestone 2 — the distinct-vs-repeated A/B (equal lesson counts).
#   A: 8,400 distinct rows (25,200 lessons)      -> hidden choice acc
#   B: 1,050 rows re-taught 8x (25,200 lessons)  -> hidden choice acc
#   B-dedup: the same file with duplicate lessons skipped (dedup_skipped
#   reports 7/8 of B's lessons as zero-information).
# ---------------------------------------------------------------------------
bank77: build/syfox
	python3 tools/bank77_prepare.py --topk 20 --opaque-labels --no-instructions
	./build/syfox learn --examples data/bank77_train.jsonl --model model-bank77-best --epochs 3
	./build/syfox calibrate --model model-bank77-best --examples data/bank77_cal.jsonl
	./build/syfox bench --model model-bank77-best --eval data/bank77_cal.jsonl --energy-norm

bank77-report: bank77
	python3 tools/bank77_analysis.py --model model-bank77-best --eval data/bank77_hidden.jsonl --arbitrate --dump data/bank77_hidden_predictions.jsonl | tee data/bank77_report_hidden.txt

ab-distinct: build/syfox
	@head -n 1050 data/big/tickets_en_train.jsonl > build/ab-repeat-src.jsonl
	@for i in 1 2 3 4 5 6 7 8; do cat build/ab-repeat-src.jsonl; done > build/ab-repeat.jsonl
	rm -rf model-ab
	./build/syfox learn --ngrams on --dedup --model model-ab --examples data/big/tickets_en_train.jsonl | tee build/ab-A-learn.json
	./build/syfox calibrate --model model-ab --examples data/big/tickets_en_cal.jsonl > /dev/null
	./build/syfox bench --model model-ab --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 | tee build/ab-A-distinct.json
	rm -rf model-ab
	./build/syfox learn --ngrams on --model model-ab --examples build/ab-repeat.jsonl | tee build/ab-B-learn.json
	./build/syfox calibrate --model model-ab --examples data/big/tickets_en_cal.jsonl > /dev/null
	./build/syfox bench --model model-ab --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 | tee build/ab-B-repeated.json
	rm -rf model-ab
	./build/syfox learn --ngrams on --dedup --model model-ab --examples build/ab-repeat.jsonl | tee build/ab-Bd-learn.json
	./build/syfox calibrate --model model-ab --examples data/big/tickets_en_cal.jsonl > /dev/null
	./build/syfox bench --model model-ab --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 | tee build/ab-B-dedup.json
	rm -rf model-ab build/ab-repeat-src.jsonl
	@echo "A/B complete: build/ab-A-distinct.json vs ab-B-repeated.json vs ab-B-dedup.json"

# Milestone-1b: the dose-response probe (does distinct experience scale?).
scale-probe: build/syfox
	tools/scale_probe.sh

# Milestone-1: prove the firewall refuses hidden/calibration reads.
firewall-check: build/syfox
	@! ./build/syfox learn --model model-x --examples data/big/tickets_en_hidden.jsonl 2>/dev/null
	@echo "ok  learn refuses _hidden"
	@! ./build/syfox learn --model model-x --examples data/big/tickets_en_cal.jsonl 2>/dev/null
	@echo "ok  learn refuses _cal"
	@! ./build/syfox calibrate --model model-x --examples data/big/tickets_en_hidden.jsonl 2>/dev/null
	@echo "ok  calibrate refuses _hidden"
	@! ./build/syfox derive --model model-tickets --gate data/big/game_hidden.jsonl 2>/dev/null
	@echo "ok  derive --gate refuses _hidden"

# ---------------------------------------------------------------------------
# v3 Milestone 4 — the adversarial suite on the big hidden tests.
#   Deterministic stress families + conflicting-lessons attack; honest
#   per-family metrics (accuracy / defer / conf-when-wrong / false-conf).
# ---------------------------------------------------------------------------
adv-big: build/syfox
	./build/syfox bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl --adversarial | tee build/adv-tickets.json
	./build/syfox bench --energy-norm --lang auto --model model-game-big --eval data/big/game_hidden.jsonl --adversarial | tee build/adv-game.json
	./build/syfox bench --energy-norm --lang auto --model model-guard-big --eval data/big/guard_hidden.jsonl --adversarial | tee build/adv-guard.json

# ---------------------------------------------------------------------------
# v3 Milestone 5 — multicore: deterministic OMP settle + batched throughput.
#   omp          builds the -fopenmp binaries (same sources, no code forks)
#   omp-identity proves parallel settle == sequential settle BIT-IDENTICAL
#   throughput   batched decisions/sec, 1 worker vs N workers (private
#                substrates per worker; performance axis, not accuracy)
#   density      the measured GPU gate (nodes / lanes / density / degree)
# ---------------------------------------------------------------------------
omp: build/syfox-omp build/syfox-test-omp

omp-identity: omp models-big
	@./build/syfox     bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 | python3 -c "import json,sys; print(json.load(sys.stdin)['routing']['choice_accuracy'])" > build/omp-seq-acc.txt
	@./build/syfox-omp bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 --threads 2 | python3 -c "import json,sys; print(json.load(sys.stdin)['routing']['choice_accuracy'])" > build/omp-par-acc.txt
	@./build/syfox-omp bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 --threads 2 > build/omp-par-full.json
	@./build/syfox     bench --energy-norm --lang auto --model model-tickets-big --eval data/big/tickets_en_hidden.jsonl --latency-reps 1 --replays 0 > build/omp-seq-full.json
	@python3 tools/check_omp_identity.py

throughput: build/syfox models-big
	@# NOTE: the BASE model dir holds only routing state (--lang auto teaches
	@# into <model>-<script> dirs), so throughput must load a real substrate:
	@# model-tickets-big-latin. workers own private engine copies; the metric
	@# is decisions/sec (checksum non-zero proves real decisions ran).
	./build/syfox bench --energy-norm --model model-tickets-big-latin --eval data/big/tickets_en_hidden.jsonl --throughput 2 | tee build/throughput-tickets.json

density: build/syfox models-big
	@echo "GPU gate (design only — no GPU in this sandbox):"
	@echo "per-script substrate fabrics (--lang auto routes lessons into"
	@echo "<model>-<script>; the base dir holds only routing state):"
	@ls -d model-*-big-* 2>/dev/null | while read d; do \
	        ./build/syfox stats --model "$$d" | python3 -c "import json,sys; d=json.load(sys.stdin); print('%6d nodes %7d lanes density=%.5f degree=%.2f  %s' % (d.get('nodes',0), d.get('lanes',0), d.get('fabric_density',0), d.get('mean_out_degree',0), sys.argv[1]))" "$$d"; \
	done
giant-fetch:
	python3 tools/hf_fetch.py

giant-prepare:
	python3 tools/giant_prepare.py

giant-chunks:
	mkdir -p data/giant/chunks
	split -n l/12 -d -a 2 data/giant/train.jsonl data/giant/chunks/chunk_ --additional-suffix=.jsonl

# -- v3.2 semantic layer: dedicated bank77 fabric + 16-domain router ---------
# Dedicated high-cardinality substrate: opaque anchors, empty instructions
# (the three RCA fixes), PLUS the v3.2 semantic field, context-sensitive
# lanes, retrieval memories and the (gate-off) semantic hierarchy.
b77-hier-data:
	python3 tools/bank77_hier.py

b77-sem: build/syfox b77-hier-data
	bash scripts/b77_train_chunked.sh 0 11
	cp data/bank77_hierarchy.json model-b77-sem/hierarchy.json
	cp data/bank77_memories.jsonl model-b77-sem/memories.jsonl
	./build/syfox calibrate --model model-b77-sem --examples data/bank77_cal.jsonl

b77-sem-bench: build/syfox
	./build/syfox bench --model model-b77-sem --eval data/bank77_cal.jsonl --energy-norm --replays 1 --latency-reps 1

b77-sem-hidden: build/syfox
	./build/syfox bench --model model-b77-sem --eval data/bank77_hidden.jsonl --energy-norm --replays 1 --latency-reps 2

# Two-stage physics router: small dedicated fabric (one anchor per domain),
# trained on router-question-only rows; decide --router model-router16 routes
# into the mapped domain fabric (router.json holds anchors + models).
router-data:
	python3 tools/router_prepare.py

router16: build/syfox router-data
	./build/syfox learn --model model-router16 --examples data/router16_train.jsonl --epochs 3
	cp data/router16_router.json model-router16/router.json
	./build/syfox calibrate --model model-router16 --examples data/router16_cal.jsonl

router16-bench: build/syfox
	./build/syfox bench --model model-router16 --eval data/router16_cal.jsonl --energy-norm --replays 1 --latency-reps 1

# Zero-shot probe suite (email/sms spam on the tickets fabric, snake +
# tic-tac-toe on the giant fabric), plain vs semantic rebuilds.
zeroshot-sem: build/rebuild_sem
	./build/rebuild_sem model-game-big-latin model-game-big-latin-sem
	./build/rebuild_sem model-tickets-big-latin model-tickets-big-latin-sem
	./build/rebuild_sem model-giant-latin model-giant-latin-sem

zeroshot: build/syfox
	python3 tools/zeroshot_suite.py

build/rebuild_sem: src/rebuild_sem.cpp $(CORE_HDRS)
	$(CXX) $(CXXFLAGS) -fopenmp -I. src/rebuild_sem.cpp -o build/rebuild_sem

# --- v3.2.1 CI gate: the three bug-fix regressions ---------------------------
# Runs the unit suite, the ablation kill-switch distinctness check on the
# dedicated bank77 fabric, and the two-stage router end-to-end. Every step
# exits non-zero on regression, so `make ci` is the pre-commit gate.
ci: build/syfox test
	./build/syfox decide --router model-router16 --model model-b77-sem \
		--energy-norm --state "Why am I missing my refund" \
		--questions '{"intent":{"type":"choice","instructions":"","criteria":{"c00":"refund in my account","c33":"money back","c61":"app crash"}}}' \
		| python3 tools/assert_route.py
	python3 tools/ablation_suite.py --model model-b77-sem \
		--eval data/bank77_cal.jsonl --limit 40 --energy-norm
	bash tests/readout_silence_test.sh
