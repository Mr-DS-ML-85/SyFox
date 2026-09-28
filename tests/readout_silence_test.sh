#!/usr/bin/env bash
# tests/readout_silence_test.sh — v3.3.1 regression gate for the readout-silence
# threshold (kUnknownCandidateFloor = 0.01, ABSOLUTE).
#
# Pins the measured contract:
#   - valid candidates (energy >= 0.01) must ANSWER, never false-defer:
#       {Tariq,Salma}, {Tariq,Salma,radio}, lowercase, and the dim
#       energy-norm-OFF case (max candidate 0.341 on the probe fabric)
#   - true unknowns must DEFER, with the layer-specific reason:
#       dark field            -> unknown_vocabulary  (decide-level floor 0.05)
#       lit field, dark cands -> unknown_candidates  (readout-level floor 0.01)
#
# Usage: bash tests/readout_silence_test.sh   (MODEL=dir overrides the fabric;
# the QA probe fabric is built on first run from data/reasoning_probe_lessons.jsonl)
set -uo pipefail

cd "$(dirname "$0")/.."
BIN=build/syfox
MODEL=${MODEL:-model-reasoning-probe}
FAIL=0

[ -x "$BIN" ] || { echo "FAIL: $BIN missing — run make first"; exit 1; }
command -v jq >/dev/null || { echo "FAIL: jq required"; exit 1; }

if [ ! -f "$MODEL/substrate.bin" ]; then
  echo "building probe fabric $MODEL ..."
  rm -rf "$MODEL"
  ./"$BIN" learn --model "$MODEL" --examples data/reasoning_probe_lessons.jsonl >/dev/null
  ./"$BIN" calibrate --model "$MODEL" --examples data/reasoning_probe_lessons.jsonl --energy-norm >/dev/null
fi

STATE="The radio was found by Tariq. Salma was there."
INSTR="Who found the radio?"

# run <jq-filter> <state> <questions> [extra flags...]
run() {
  local filt="$1" state="$2" questions="$3"; shift 3
  ./"$BIN" decide --model "$MODEL" "$@" --state "$state" --questions "$questions" \
    2>/dev/null | jq -r "$filt"
}

check() {
  local name="$1" got="$2" want="$3"
  if [ "$got" = "$want" ]; then
    echo "  ok  $name (got $got)"
  else
    echo "  FAIL $name: got [$got], want [$want]"; FAIL=1
  fi
}

echo "[readout silence — valid candidates must answer]"
for CRIT in '{"Tariq":"Tariq","Salma":"Salma"}' '{"Tariq":"Tariq","Salma":"Salma","radio":"radio"}'; do
  check "criteria=$CRIT deferred" \
    "$(run '.answers.q.deferred' "$STATE" "{\"q\":{\"type\":\"choice\",\"instructions\":\"$INSTR\",\"criteria\":$CRIT}}" --energy-norm)" \
    "false"
done
check "lowercase deferred" \
  "$(run '.answers.q.deferred' "the radio was found by tariq. salma was there." \
     '{"q":{"type":"choice","instructions":"who found the radio?","criteria":{"tariq":"tariq","salma":"salma"}}}' --energy-norm)" \
  "false"
check "energy-norm OFF (dim candidate 0.341) deferred" \
  "$(run '.answers.q.deferred' "$STATE" \
     "{\"q\":{\"type\":\"choice\",\"instructions\":\"$INSTR\",\"criteria\":{\"Tariq\":\"Tariq\",\"Salma\":\"Salma\"}}}")" \
  "false"

echo "[readout silence — true unknowns must defer]"
check "dark field defers" \
  "$(run '.answers.q.deferred' "xyz abc def" \
     '{"q":{"type":"choice","criteria":{"xyz":"xyz","abc":"abc"}}}' --energy-norm)" \
  "true"
check "dark field reason (decide-level floor 0.05)" \
  "$(run '.answers.q.reason' "xyz abc def" \
     '{"q":{"type":"choice","criteria":{"xyz":"xyz","abc":"abc"}}}' --energy-norm)" \
  "unknown_vocabulary"
check "lit field + dark candidates reason (readout floor 0.01)" \
  "$(run '.answers.q.reason' "$STATE" \
     "{\"q\":{\"type\":\"choice\",\"instructions\":\"$INSTR\",\"criteria\":{\"purple\":\"purple elephant\",\"submarine\":\"submarine\"}}}" --energy-norm)" \
  "unknown_candidates"

if [ "$FAIL" -eq 0 ]; then
  echo "readout-silence regression: ALL PASS"
else
  echo "readout-silence regression: FAILURES PRESENT"; exit 1
fi
