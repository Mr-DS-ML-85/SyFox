# SyFox — System One decision engine on an SI substrate core
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
BUILD := build

all: build/syfox build/syfox-test build/libsyfox_core.so

build:
	mkdir -p build

build/syfox: src/syfox_cli.cpp core/syfox.hpp core/si_substrate.hpp core/derive.hpp core/json.hpp | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

build/syfox-test: src/syfox_test.cpp core/syfox.hpp core/si_substrate.hpp core/derive.hpp core/json.hpp | build
	$(CXX) $(CXXFLAGS) -I. $< -o $@

build/libsyfox_core.so: src/syfox_core.cpp core/syfox.hpp core/si_substrate.hpp core/derive.hpp core/json.hpp | build
	$(CXX) $(CXXFLAGS) -fPIC -shared -I. $< -o $@

# Deterministic: every model is rebuilt from its seed data, not grown in place.
models: build/syfox
	rm -rf model-tickets model-game model-guard
	./build/syfox learn --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox calibrate --model model-tickets --examples data/tickets_train.jsonl
	./build/syfox learn --model model-game --examples data/game_train.jsonl
	./build/syfox calibrate --model model-game --examples data/game_train.jsonl
	./build/syfox learn --model model-guard --examples data/guard_train.jsonl
	./build/syfox calibrate --model model-guard --examples data/guard_train.jsonl

test: build/syfox-test
	./build/syfox-test

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

.PHONY: all models test demo serve clean
