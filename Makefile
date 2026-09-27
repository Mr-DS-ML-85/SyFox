# SyFox — System One decision engine on an SI substrate core
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD := build

all: build/syfox build/syfox-test build/libsyfox_core.so

build:
	mkdir -p build

CORE_HDRS := core/syfox.hpp core/si_substrate.hpp core/normalize.hpp core/ngram.hpp core/script.hpp core/derive.hpp core/bench.hpp core/gate.hpp core/recall.hpp core/json.hpp

build/syfox: src/syfox_cli.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

build/syfox-test: src/syfox_test.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

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

.PHONY: all models models-augmented models-vast models-vast-full models-multilingual bench-multilingual typos bench-extra heldout-gate test bench bench-heldout coverage demo serve clean
