# Intra-category attack — the card_management binding constraint (v3.9.1)

The tester's audit: "within-category separation among 22 confusable
intents — 9-way 0.535 ≈ 77-way 0.581 — the fabric understands banking but
can't distinguish fine-grained intents." This report runs the revised plan
(diagnostics first, then intervention), every number from an actual run in
this tree, tuning on cal only, hidden touched only for final validation.

**Headline: the binding constraint is not the intents — it is the diffusion
constant. Lowering lane diffusion 0.45 → 0.02 on the SAME frozen fabric
lifts the hidden honest total 0.424 → 0.571 (+14.7 pts), and card_management
0.389 → 0.632 (+24.3 pts). No physics equation changed, no retrain, no
global defer default shipped.**

---

## 0. Honest-total table

See [HONEST_TOTAL.md](HONEST_TOTAL.md) — the acc × answered/total table for
every config. Definition: honest total = correct / ALL rows (a deferred row
scores as not-answered). Accuracy-over-answered alone is not reported anywhere
as a headline in this document.

**Calibration finding (measured, first-class):** the cal fit adopted
choice temperature T=0.0019. Post-temperature probability gaps saturate at
~1.0, so (a) the engine's probability-gap defer axis is inert on this
fabric (0 deferrals in 1,003 cal rows at the shipped 0.05 margin), and
(b) bench-measured ECE is 0.597 (cal) / 0.573 (hidden) even though the
calibrate report printed 0.136 on its own fit rows. The shipped margin
sweep was therefore re-anchored on the RAW pre-temperature energy gap,
which v3.9.1 now discloses per row (`Answer.raw_margin`, dump field
`raw_margin`). Disclosure only — nothing in the engine consumes it.

## 0.5 Class-frequency cut

Per-intent accuracy vs per-intent train-row count (31–168, mean 117 —
banking77 is nearly balanced):

| split | Pearson r | Spearman rho | verdict (|r|>0.5 cut) |
|-------|----------:|-------------:|------------------------|
| cal 1003 | **−0.139** (t=−1.22, df=75) | −0.123 | MODEL problem |
| hidden 3080 | **−0.043** (t=−0.37) | −0.052 | MODEL problem |

Accuracy does not track class frequency (the sign is even negative). The
binding constraint is NOT thin classes — it is confusability. Tool:
`tools/b77_freq_cut.py`.

## 1. Confusion matrix

`syfox bench --confusion` now emits, in the bench JSON: `labels` (sorted
union of golds + preds + the DEFER pseudo-class), the full N×N `matrix`
(rows=gold, cols=pred, row i == labels[i] always — a v3.9.1 row-alignment
bug that shifted every row when DEFER sorted first was caught by
cross-checking the hidden dump against the matrix and is pinned by
`test_bench_diagnostics`), `deferred_total`, and `pairs_top`.

Most-confused pairs (hidden, 3,080 rows):

| gold → pred | n | natural names |
|---|--:|---|
| c18 → c76 | 21 | card_payment_wrong_exchange_rate → wrong_exchange_rate_for_cash_withdrawal |
| c68 → c74 | 15 | unable_to_verify_identity → why_verify_identity |
| c34 → c33 | 14 | exchange_via_app → exchange_rate |
| c13 → c67 | 13 | card_delivery_estimate → transfer_timing |
| c15 → c72 | 13 | card_not_working → virtual_card_not_working |
| c27 → c04 | 13 | declined_cash_withdrawal → atm_support |
| c20 → c76 | 12 | cash_withdrawal_charge → wrong_exchange_rate_for_cash_withdrawal |
| c62 → c65 | 12 | topping_up_by_card → transfer_into_account |
| c69 → c74 | 12 | verify_my_identity → why_verify_identity |

The structure is semantic, not random: exchange-rate twins, fee twins,
verification twins, withdrawal-vs-ATM. Where card_management golds land
(hidden baseline): c72 virtual_card_not_working (82), c04 atm_support (69),
c73 visa_or_mastercard (59), c76 (47), c65 transfer_into_account (43).

**card_management (22 intents) view** (from the same matrix, "OTHER"
aggregated; `tools/b77_confusion_report.py`):

| split | config | card_management honest | rest honest | overall |
|---|---|---:|---:|---:|
| hidden | baseline | **0.389** (342/880) | 0.438 (963/2200) | 0.424 |
| hidden | diffusion 0.02 | **0.632** (556/880) | 0.547 (1203/2200) | 0.571 |
| hidden | diffusion 0.10 | 0.552 (486/880) | 0.516 (1134/2200) | 0.526 |

The intra-category binding constraint is real (card −4.9 pts vs rest at
baseline) and the diffusion intervention relieves it MORE than it relieves
the rest (+24.3 vs +10.9) — fine-grained within-category separation is
exactly what the smeared field was destroying.

## 2. Per-intent thresholds from the cal sweep (honest negative)

Global sweep on the cal dump (simulated offline from `raw_margin`; the
shipped probability-margin axis is inert, see §0):

| threshold | answered | acc (answered) | honest total |
|---:|---:|---:|---:|
| 0.00 (= baseline) | 1003 | 0.401 | **0.4008** |
| 0.05 | 879 | 0.432 | 0.3789 |
| 0.10 | 753 | 0.468 | 0.3509 |
| 0.20 | 585 | 0.527 | 0.3071 |

Every positive global threshold LOSES honest total: accuracy-when-answered
rises but never pays for the coverage lost.

Per-intent thresholds (grid 0.00–0.20, argmax per-intent honest on cal,
min-rows 8 → 33 custom thresholds): cal honest 0.4008 (neutral, the fit
removes only wrong rows in-sample), acc(answered) 0.401 → 0.482.
**Applied to hidden: honest 0.4042 vs baseline 0.4237 (−2.0 pts).** 33
thresholds fit on 1,003 rows overfit; the cal-argmax did not generalize.
This is a measured honest negative: per-intent deferral is not the lever on
this fabric. No global defer-margin default was shipped (the constraint
held; nothing in the engine changed). Tools: `tools/b77_margin_sweep.py`,
`tools/b77_apply_policy.py`.

## 3. Semantic field ablation (per-intent)

`--no-semantics` on the same cal rows: honest total 0.4008 → **0.3400**
(−6.1 pts). Flips: 106 correct→wrong vs 45 wrong→correct. Biggest
per-intent losses: c19 card_swallowed −0.333, c38 get_disposable_virtual_card
−0.300, c60 top_up_limits −0.300, c57 top_up_by_card_charge −0.273,
c24 contactless_not_working −0.250, c34 exchange_via_app −0.250. The
semantic field is load-bearing for exactly the fine-grained card cluster
the audit flagged. Tool: `tools/b77_ablation_compare.py`.

## 4. Physics constant sweep (the lever)

Decide-side knobs sweep on the frozen headline fabric (cal; scalars held —
disclosed; one knob at a time):

| knob | points (honest total on cal) |
|---|---|
| decay | 0.75 → 0.4002, 0.82 → 0.4008, 0.88 → 0.4010 — **flat** |
| k_settle | 4 → 0.4008, 8 → 0.4008, 16 → 0.4008 — **inert** (the ε early-stop already converges before 8 on this fabric) |
| hop_coupling | 0.25 → 0.3966, 0.35 → 0.4008, 0.50 → 0.3982 — **flat** |
| **diffusion** | 0.02 → **0.5374**, 0.05 → 0.5170, 0.10 → 0.4990, 0.20 → 0.4950, 0.25 → 0.4920, 0.30 → 0.4855, 0.35 → 0.4680, 0.40 → 0.4250, 0.45 → 0.4008, 0.55 → 0.3536 — **monotone, boundary optimum** |

Learn-side knobs (reduced protocol, disclosed: train-only fabric, 1 epoch,
no category lessons/hierarchy/memories; all points share construction so
univariate deltas are comparable; absolute numbers differ from the headline
family):

| knob | points (honest total on cal) |
|---|---|
| lane_cap | 64 → 0.3766, 256 → 0.3956, 1024 → **0.4118** (+1.6) |
| learn_eta | 0.05 → 0.3986, 0.10 → 0.3956, 0.20 → 0.3959 — **flat** in this range |
| diffusion on this family | 0.45 → 0.3956 (acc_ans 0.587, defer 0.326), 0.10 → 0.5143, 0.02 → **0.5596** |

**Mechanism reading.** Diffusion is the fraction of retained energy that
flows along lanes per settle pass (SI core frozen — this is a CONFIG value,
`SubstrateConfig::diffusion`, not an equation change). At 0.45 on a
1.31M-lane bigram-era fabric, the injected state diffuses into a broad
background; readout energies become context-smeared and fine-grained
intents that share vocabulary (every card intent contains "card") collapse
into each other. Low diffusion keeps energy concentrated on the direct
injection + its strongest lanes → separation. Three corroborations:
(1) the effect replicates on a second, independently built fabric family
(reduced protocol: +16.4 pts); (2) lane_cap 1024 (denser lane retention)
helps at the margin while the decide-side diffusion cut helps massively —
the interaction is fabric density × diffusion; (3) the per-intent pattern
of the gain concentrates on the confusable card cluster.

**Hidden validation (no hidden tuning: 0.02 and 0.10 were chosen from the
cal curve, then measured once each):**

| config | hidden honest total | delta |
|---|---:|---:|
| baseline (0.45) | 0.4237 | — |
| diffusion 0.10 | 0.5260 | +10.2 pts |
| diffusion 0.02 | **0.5711** | **+14.7 pts** |

## Recommendation

1. **Per-fabric diffusion disclosure** (follow-up work): record the measured
   diffusion optimum in `meta.json` at calibration time and let `decide`
   honor a per-fabric value — NOT a new global default. The shipped 0.45
   was validated on small fabrics in the v2 era; the bigram-era fabric
   population is 30–100× denser and the measured optimum moved to the
   boundary. The default stays 0.45 until a per-fabric mechanism ships
   (changing the global default would also break replay contracts for
   every existing fabric).
2. **Do not chase per-intent defer thresholds** — measured negative here.
3. **Do not claim 77-way parity** — the honest total is 0.571 with the
   intervention, 0.424 without; the card_management cluster is 0.632/0.389.
   Every headline in this repo must cite the honest total.
4. The calibration fit's adoption guard needs a hidden-consistency check
   (it adopted T=0.0019 on its own fit rows while measured ECE is 0.573–
   0.597); worth its own task.

## Reproduce

```bash
make all
./build/syfox bench --model model-b77-sem --eval data/bank77_cal.jsonl \
    --energy-norm --replays 1 --latency-reps 1 --confusion \
    --dump-perrow data/b77_diag/cal_dump.jsonl
python3 tools/b77_confusion_report.py --bench data/b77_diag/hidden_bench.json
python3 tools/b77_margin_sweep.py --dump data/b77_diag/cal_dump.jsonl --mode global
python3 tools/b77_freq_cut.py --dump data/b77_diag/hidden_dump.jsonl
./build/syfox bench --model model-b77-sem --eval data/bank77_hidden.jsonl \
    --energy-norm --replays 1 --latency-reps 1 --diffusion 0.02
```

Artifacts: `data/b77_diag/` (bench JSONs + per-row dumps + threshold map).
Unit tests: `test_bench_diagnostics` (confusion/dump/knobs/alignment).
