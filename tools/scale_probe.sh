#!/usr/bin/env bash
# ============================================================================
#  SyFox — scale_probe.sh  (Milestone 1b: the dose-response curve)
#  Learns on growing prefixes of the big train split, calibrates on the
#  calibration split, and scores the SAME hidden test each time. Answers two
#  questions with measurements, not hopes:
#    * does "more distinct experience" help the substrate? (Milestone 2 goal)
#    * what does each doubling COST in wall-clock? (Milestone 5 planning)
#  Fast-probe settings: --latency-reps 1 --replays 0 (the hidden headline
#  runs in the Makefile use the full defaults).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:-data/big/tickets_en_train.jsonl}
HID=${2:-data/big/tickets_en_hidden.jsonl}
CAL=${3:-data/big/tickets_en_cal.jsonl}
M=model-probe

TOTAL=$(wc -l < "$SRC")
echo "scale probe: source=$SRC ($TOTAL rows) hidden=$HID"
printf '%8s %10s %12s %10s %10s\n' N learn_s bench_s choice_acc choice_ece
for N in 250 500 1000 2000 4000 "$TOTAL"; do
    if [ "$N" -gt "$TOTAL" ]; then continue; fi
    rm -rf "$M"
    head -n "$N" "$SRC" > build/probe-rows.jsonl
    T0=$(date +%s.%N)
    ./build/syfox learn     --model "$M" --examples build/probe-rows.jsonl > /dev/null
    ./build/syfox calibrate --model "$M" --examples "$CAL"                  > /dev/null
    T1=$(date +%s.%N)
    OUT=$(./build/syfox bench --model "$M" --eval "$HID" \
            --latency-reps 1 --replays 0)
    T2=$(date +%s.%N)
    echo "$OUT" | python3 -c '
import json, sys
j = json.load(sys.stdin)
# argv: [-c, N, T0, T1, T1, T2]
print("%8s %10.1f %12.1f %10.4f %10.4f" % (
    sys.argv[1], float(sys.argv[3]) - float(sys.argv[2]),
    float(sys.argv[5]) - float(sys.argv[4]),
    j["routing"]["choice_accuracy"], j["calibration"]["choice_ece"]))' \
        "$N" "$T0" "$T1" "$T1" "$T2"
done
rm -rf "$M" build/probe-rows.jsonl
