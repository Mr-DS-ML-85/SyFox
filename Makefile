# SyFox â System One decision engine on an SI substrate core
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD := build

all: build/syfox build/syfox-test build/libsyfox_core.so

build:
	mkdir -p build

CORE_HDRS := core/syfox.hpp core/si_substrate.hpp core/derive.hpp core/bench.hpp core/gate.hpp core/recall.hpp core/json.hpp

build/syfox: src/syfox_cli.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

build/syfox-test: src/syfox_test.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

build/libsyfox_core.so: src/syfox_core.cpp $(CORE_HDRS) | build
	$(CXX) $(CXXFLAGS) -fPIC -shared -I. $< -o $@

# Deterministic: every model is rebuilt from its seed data, not grown in place.
models: build/syfox
	rm -rf model-tickets model-game model-guard
	./build/syfox learn --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox calibrate --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox learn --model model-game --examples data/game_train.jsonl
	./build/syfox calibrate --model model-game --examples data/game_train.jsonl
	@# gated derivation: the no-regression gate decides; game passes at the
	@# conservative strength (0 taught flips, no manufactured certainty)
	./build/syfox derive --model model-game --gate data/game_train.jsonl
	./build/syfox learn --model model-guard --examples data/guard_train.jsonl
	./build/syfox calibrate --model model-guard --examples data/guard_train.jsonl

test: build/syfox-test
	./build/syfox-test

# Jev-parity eval suite on the three seed models (in-domain resubstitution â
# the honest label is in the JSON output).
bench: build/syfox models
	./build/syfox bench --model model-tickets --eval data/tickets_train.jsonl | tee build/bench-tickets.json
	./build/syfox bench --model model-game --eval data/game_train.jsonl | tee build/bench-game.json
	./build/syfox bench --model model-guard --eval data/guard_train.jsonl | tee build/bench-guard.json

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

.PHONY: all models test bench demo serve clean
