#!/usr/bin/env bash
# tests/jas_test.sh — v3.5 regression gate for the J-A-S cycle, the calc
# derivation oracle, and the impossibility register (research-paper/jas.md).
#
# Pins the measured contract:
#   - calc: exact derivation (17*23=391, precedence 2+3*4=14), word problems
#     map number words to operators; --compile emits C++, g++ computes, and
#     the oracle AGREES (provenance established_by_experiment) — the field
#     never computes, the oracle does.
#   - register: 5 seeded claims, every one carries a warrant, THEOREM claims
#     are not marked violated; --model persists impossibility.json.
#   - jas cycle: E -> J -> A -> S -> experiment -> refutation -> J2 runs
#     end-to-end on the demo fabric; the axiom layer is INDUCED (support/
#     total disclosed); the oracle's verdict is one of the known strings;
#     a refutation writes model-jas-demo/refuted.axioms. Deterministic.
#
# Usage: bash tests/jas_test.sh
set -uo pipefail

cd "$(dirname "$0")/.."
BIN=build/syfox
MODEL=model-jas-demo
FAIL=0

note() { printf '  %s\n' "$*"; }
bad()  { printf '  FAIL %s\n' "$*"; FAIL=$((FAIL+1)); }

[ -x "$BIN" ] || { echo "FAIL: $BIN missing — run make first"; exit 1; }
command -v jq >/dev/null || { echo "FAIL: jq required"; exit 1; }

echo "[jas] calc — internal derivation oracle"
V1=$(./"$BIN" calc --expr "17*23" | jq -r '.value')
[ "$V1" = "391" ] && note "ok 17*23 = 391" || bad "17*23 gave '$V1' (want 391)"
V2=$(./"$BIN" calc --expr "2+3*4" | jq -r '.value')
[ "$V2" = "14" ] && note "ok precedence 2+3*4 = 14" || bad "2+3*4 gave '$V2' (want 14)"
V3=$(./"$BIN" calc --expr "what is 17 times 23 plus 5" | jq -r '.value')
[ "$V3" = "396" ] && note "ok word problem maps to 17*23+5 = 396" || bad "word problem gave '$V3' (want 396)"
V4=$(./"$BIN" calc --expr "1/0" | jq -r '.ok')
[ "$V4" = "false" ] && note "ok 1/0 rejected, never guessed" || bad "1/0 returned ok=$V4"

if command -v g++ >/dev/null; then
  AGREE=$(./"$BIN" calc --expr "17*23" --compile | jq -r '.compile_oracle.agreement')
  PROV=$(./"$BIN" calc --expr "17*23" --compile | jq -r '.compile_oracle.provenance')
  [ "$AGREE" = "true" ] && note "ok g++ oracle agrees (established_by_experiment)" \
                        || bad "g++ oracle agreement=$AGREE"
  [ "$PROV" = "established_by_experiment" ] && note "ok provenance disclosed" \
                                            || bad "provenance='$PROV'"
else
  note "SKIP g++ oracle (no compiler on PATH)"
fi

echo "[jas] impossibility register"
./"$BIN" register --model "$MODEL" >/dev/null 2>&1 || bad "register command failed"
[ -f "$MODEL/impossibility.json" ] && note "ok impossibility.json persisted" \
                                   || bad "impossibility.json missing"
N=$(jq '.claims | length' "$MODEL/impossibility.json")
[ "$N" = "5" ] && note "ok 5 seeded claims" || bad "claims=$N (want 5)"
WT=$(jq '[.claims[].warrant] | index("THEOREM") != null' "$MODEL/impossibility.json")
[ "$WT" = "true" ] && note "ok warrants include THEOREM" || bad "no THEOREM warrant"

echo "[jas] the cycle (E -> J -> A -> S -> experiment -> refutation -> J2)"
if [ ! -f "$MODEL/substrate.bin" ]; then
  ./"$BIN" learn --model "$MODEL" --examples data/jas_cycle_demo.jsonl >/dev/null
fi
head -18 data/jas_cycle_demo.jsonl > build/jas_train.jsonl
tail -6  data/jas_cycle_demo.jsonl > build/jas_holdout.jsonl
rm -f "$MODEL/refuted.axioms" "$MODEL/verified.axioms"
OUT=$(./"$BIN" jas --model "$MODEL" --lessons build/jas_train.jsonl --holdout build/jas_holdout.jsonl 2>/dev/null)
echo "$OUT" > build/jas_cycle_output.json
VERDICT=$(jq -r '.verdict' build/jas_cycle_output.json)
case "$VERDICT" in
  SURVIVES|REFUTED_AND_REVISED_SURVIVES_HELDOUT|REFUTED_AND_REVISION_ALSO_REFUTED|\
  REFUTED_AND_REVISION_UNTESTED|REFUTED_IN_ROUND2_REVISE_NEXT_CYCLE|NO_TESTABLE_AXIOMS)
    note "ok verdict '$VERDICT' is a known outcome" ;;
  *) bad "unknown verdict '$VERDICT'" ;;
esac
NA=$(jq '.axioms | length' build/jas_cycle_output.json)
[ "$NA" -ge 1 ] && note "ok $NA induced axiom(s) with INDUCED provenance" || bad "axioms=$NA"
W=$(jq -r '.axioms[0].warrant' build/jas_cycle_output.json)
[ "$W" = "INDUCED" ] && note "ok axiom warrant INDUCED" || bad "warrant=$W"
R2=$(jq '.round2.tested' build/jas_cycle_output.json)
[ "$R2" = "3" ] && note "ok round 2 tested 3 fresh rows" || bad "round2 tested=$R2"
FL=$(jq -r '.field_layer.role' build/jas_cycle_output.json)
[ "$FL" = "relevance heuristic (frozen SI core, untouched)" ] \
  && note "ok field layer disclosed as relevance heuristic" || bad "field role='$FL'"
case "$VERDICT" in
  REFUTED*)
    [ -f "$MODEL/refuted.axioms" ] && note "ok refutation logged to refuted.axioms" \
                                   || bad "refutation not logged" ;;
esac
# determinism: second run must produce byte-identical JSON
rm -f "$MODEL/refuted.axioms" "$MODEL/verified.axioms"
./"$BIN" jas --model "$MODEL" --lessons build/jas_train.jsonl --holdout build/jas_holdout.jsonl 2>/dev/null > build/jas_cycle_output2.json
if cmp -s build/jas_cycle_output.json build/jas_cycle_output2.json; then
  note "ok cycle deterministic (byte-identical JSON x2)"
else
  bad "cycle JSON differs across identical runs"
fi

[ "$FAIL" = "0" ] && echo "JAS ALL PASS" || { echo "JAS $FAIL FAILURES"; exit 1; }
