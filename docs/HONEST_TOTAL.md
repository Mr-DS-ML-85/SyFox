# Honest totals — acc × answered/total for every measured config

Every number below is `correct / total_rows` (the selective-accuracy
headline: accuracy over ANSWERED rows multiplied by the answered fraction —
a deferred row scores as what it is: not answered). Accuracies over answered
rows alone flatter any system that defers aggressively; this table is the
anti-flattery view. All rows come from actual `syfox bench` runs in this
tree (v3.9.1 diagnostics: `--confusion` / `--dump-perrow`).

## The bank77 fabric (model-b77-sem, dedicated 19,690-node fabric)

Construction (disclosed): `bank77_hier_train.jsonl` = 9,000 intent + 9,000
category lessons, taught chunked (12 × 1,500 rows × 3 epochs) with v3.9
learn defaults (ordered bigram lanes ON, document-frequency IDF weighting
ON, CMD sentinels ON); hierarchy + 256 memories shipped; calibrated on the
1,003-row cal split (fit adopted T=0.0019 — see the calibration finding
below). Eval: energy-norm, replays 1, latency-reps 1.

| # | config | split | acc (answered) | answered frac | **honest total** |
|---|--------|-------|---------------:|--------------:|-----------------:|
| 1 | baseline (diffusion 0.45) | cal 1003 | 0.401 | 1.000 | **0.4008** |
| 2 | baseline | hidden 3080 | 0.424 | 0.999 | **0.4237** |
| 3 | `--no-semantics` | cal 1003 | 0.340 | 1.000 | **0.3400** |
| 4 | global margin sweep, best = 0.00 | cal 1003 | 0.401 | 1.000 | **0.4008** (= baseline) |
| 5 | global margin 0.20 | cal 1003 | 0.527 | 0.583 | 0.3071 |
| 6 | per-intent thresholds (cal-derived, 33 custom) | cal 1003 | 0.482 | 0.831 | 0.4008 (= baseline) |
| 7 | per-intent thresholds applied to hidden | hidden 3080 | 0.481 | 0.840 | **0.4042** (−2.0 pts vs #2) |
| 8 | **diffusion 0.02** | cal 1003 | 0.537 | 1.000 | **0.5374** |
| 9 | **diffusion 0.02** | hidden 3080 | 0.571 | 1.000 | **0.5711** (+14.7 pts vs #2) |
| 10 | diffusion 0.10 | hidden 3080 | 0.526 | 1.000 | **0.5260** (+10.2 pts vs #2) |

## The reduced-protocol fabric (learn-side sweep family)

Construction: `bank77_train.jsonl` only (9,000 intent lessons, NO category
lessons, no hierarchy/memories), ONE process, 1 epoch, lane-cap 256. All
sweep points share this construction so univariate deltas are comparable;
absolute numbers are NOT comparable to the table above.

| config | split | acc (answered) | answered frac | **honest total** |
|--------|-------|---------------:|--------------:|-----------------:|
| baseline (diffusion 0.45, eta 0.10, cap 256) | cal 1003 | 0.587 | 0.674 | **0.3956** |
| lane-cap 64 | cal 1003 | 0.553 | 0.681 | 0.3766 |
| lane-cap 1024 | cal 1003 | 0.596 | 0.691 | **0.4118** (+1.6) |
| diffusion 0.10 on this family | cal 1003 | 0.546 | 0.942 | **0.5143** (+11.9) |
| diffusion 0.02 on this family | cal 1003 | 0.667 | 0.839 | **0.5596** (+16.4) |

## Reading the table honestly

1. **The shipped defer-margin axis is inert on this fabric.** The cal fit
   adopted T=0.0019; post-temperature probability gaps saturate at ~1.0, so
   the engine's probability-gap defer (default 0.05) can never fire
   (measured: defer 0/1003). Rows 4-7 simulate thresholds on the RAW
   pre-temperature energy gap instead (dumped per row by v3.9.1).
2. **No global threshold improves the honest total** (row 5: every positive
   margin loses). The per-intent policy is honest-total-neutral on cal
   (row 6) and LOSES on hidden (row 7) — 33 thresholds fit on 1,003 rows
   overfit; honest negative, per-intent deferral is NOT the lever here.
3. **The physics knob (diffusion) is the lever** (rows 8-10): chosen on cal
   only, validated on hidden once. See INTRACATEGORY_ATTACK.md for the full
   sweep and the mechanism reading.
