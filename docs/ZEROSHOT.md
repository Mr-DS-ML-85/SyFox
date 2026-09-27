# Zero-shot probes vs trained benchmarks (v3.1.2)

SyFox is evaluated on two separate dimensions that must never be conflated:

| dimension | what it measures | instrument |
|---|---|---|
| learned-domain performance | trained/held-out accuracy, calibration, OOD deferral, contradiction handling, latency | `bench --split hidden`, real held-out rows, trained schema |
| zero-shot composition | whether the settled SI field can compose relations it was never taught (constraints, topology, semantic chains) | handcrafted probes on untrained tasks |

Part 1 archives the model-xl zero-shot experiments (user-run, no
task-specific training; diagnostic only). Part 2 gives the code-level
mechanism of the runtime flags and re-measures them on repo fabrics with
real held-out rows (`tools/flag_probe_experiment.py`). Part 3 states the
interpretation rules. See also `docs/PROBING.md` for the criteria-wording
methodology these probes follow.

---

## Part 1 — model-xl zero-shot experiments (user-run, archived)

**Provenance:** `model-xl/model-xl-latin`, 45,892 nodes / ~1.89676 M lanes,
SyFox 3.0.0 SI substrate. No Snake, Tic-Tac-Toe, or file-organization
training data was used in that fabric. All numbers below are the user's
runs, transcribed unmodified. Two run batches exist for a few scenarios and
disagree in exact confidence (same winner, same direction); both are kept
rather than smoothed.

### Snake — zero-shot decision test (8 scenarios)

| # | Scenario | Expected | Baseline | Conf | Energy-norm | Conf |
|---|---|---|---|---:|---|---:|
| 1 | Food ahead | Right | Right | 0.001 | Right | 0.924 |
| 2 | Wall ahead | Up/Down | Right | 0 | Right | 0.536 |
| 3 | Body blocks right | Up/Down/Left | Up | 0 | Up | 0.289 |
| 4 | Food behind while moving right | Up/Down | Right | 0.001 | Right | 0.952 |
| 5 | Food at wall; right is lethal | Up/Down | Right | 0.004 | Right | 0.499 |
| 6 | Long-term survival | Up | Up | 0.004 | Up | 0.556 |
| 7 | Only safe move | Down | Down | 0.005 | Down | 0.510 |
| 8 | Shortest safe route | Up | Right | 0.002 | Right | 0.525 |

Earlier run batch (same scenarios, independently run): food directly ahead
→ right 97.8%; body blocking right → up 56.3%; long-term survival → up
69.4%; only legal move → down 71.4%.

Observations (measured):

- Energy normalization dramatically increased separation of the EXISTING
  attractors (conf 0.001 → 0.924 on #1) without guaranteeing correct
  constraint reasoning: #2, #4, #5 kept selecting the unsafe direction,
  now with high confidence.
- The two scenarios with the clearest trained-style signal (#1 food ahead,
  #7 only safe move) were selected under both configurations.
- Baseline confidence ~0.001-0.005 is the near-uniform readout of a fabric
  with no trained signal for the task — the honest-silence regime, not a
  dead fabric.

### Snake — runtime flag experiment (forced safe move)

| Config | Flags | Selected | Confidence |
|---|---|---|---:|
| A | baseline | Down (correct) | 0.005 |
| B | `--energy-norm` | Down (correct) | 0.510 |
| C | `--energy-norm --salience-gating` | Up (illegal) | 0.515 |
| D | `--energy-norm --miller-window` | Left (illegal) | 0.805 |
| E | all three | Up (illegal) | 0.516 |

The energy-normalized variants (B–E) all reported the same settled energy
(83.928) while the selected answer changed. An earlier batch measured the
same qualitative outcome with different confidences (B 0.714, D 0.924).
This is the single most important observation in the report and it is
mechanism-explained in Part 2.

### Tic-Tac-Toe — zero-shot symbolic composition

The probe explicitly supplied the 3×3 topology, position numbering, the win
rule, and the full board (X: 1,2 · O: 4,5 · X to move; winning move = 3).
Result — baseline: selected 8, conf 0; `--energy-norm`: selected 8, conf
0.011, with the full distribution:

```
1=0.085  2=0.123  3=0.081  4=0.113  5=0.089
6=0.108  7=0.118  8=0.170  9=0.112
```

Position 3 (the actual winning move) received the LOWEST mass of all nine
options (0.081). The fabric did not compose `X + X + empty → winning move`
even with every fact supplied. This is a zero-shot symbolic-composition
failure, not evidence about a trained SyFox.

### File-organizer routing — zero-shot semantic mapping

Folders and meanings supplied; file `annual_report.pdf` explicitly
described as "an office report" and "not a newly downloaded uncategorized
file". Expected: Documents. Result with `--energy-norm`:

```
Documents=0.004  Pictures=0.016  Music=0.563 (selected)
Videos=0.084     Downloads=0.333        confidence 0.386
```

The intended chain `annual_report.pdf → office report → document →
Documents` did not form; the readout settled on Music, with Downloads
second (0.333) despite the explicit exclusion.

### What Part 1 establishes

- Energy normalization amplifies whatever ordering the field already has;
  amplification is not reasoning.
- Multi-condition constraints (lethal wall, forbidden reversal, exclusions)
  are not composed on unseen tasks.
- Confidence is signal concentration, not correctness: config D reached
  0.805 (earlier batch 0.924) on an ILLEGAL move.

---

## Part 2 — what the runtime flags actually touch (code + measurement)

### Injection dose vs source selection

The three flags act at two different points:

- `--energy-norm` (M1) changes the decide-side injection dose
  (`core/syfox.hpp`, `set_energy_norm`): every probe word injects more
  energy, so the SETTLED FIELD ITSELF is different.
- `--salience-gating` and `--miller-window` change which nodes may
  PROPAGATE energy during the settle passes (`core/si_substrate.hpp`,
  `settle()`): "gated nodes keep their energy (readout-visible) but stay
  silent as sources." The Miller cap is drawn from the decision's state
  hash (`[source_cap-4, source_cap]`), so it is deterministic per state.
- Because a settle pass conserves energy up to uniform per-node decay
  (propagation only redistributes energy between nodes), the total settled
  energy is INVARIANT to source selection. Source modes change WHERE energy
  sits, never HOW MUCH exists — and therefore change the readout, not the
  field.

### Measured on repo fabrics (this release)

`tools/flag_probe_experiment.py`, real held-out rows, TRAINED schema,
five configs A–E as above. Deterministic: repeated runs reproduce every
number.

**model-tickets-big-latin** (1,800 hidden rows):

| config | top-1 | margin | conf | defer | hiConfWrong | flip vs A |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | 0.971 | 0.0355 | 0.002 | 35 | 0 | – |
| B energy | 0.971 | 0.9430 | 0.923 | 0 | 21 | 0.1% |
| C energy+salience | 0.930 | 0.9567 | 0.944 | 0 | 86 | 6.1% |
| D energy+miller | 0.975 | 0.9556 | 0.940 | 0 | 20 | 1.5% |
| E all | 0.923 | 0.9493 | 0.937 | 0 | 83 | 6.9% |

**model-sms-natural** (838 hidden rows):

| config | top-1 | margin | conf | defer | hiConfWrong | flip vs A |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | 0.170 | 0.0086 | 0.000 | 68 | 0 | – |
| B energy | 0.160 | 0.1668 | 0.060 | 2 | 0 | 0.1% |
| C energy+salience | 0.360 | 0.3819 | 0.234 | 2 | 19 | 22.6% |
| D energy+miller | 0.161 | 0.1688 | 0.063 | 2 | 0 | 3.0% |
| E all | 0.379 | 0.3941 | 0.253 | 2 | 15 | 23.9% |

**Settled-energy spread per row (same rows across configs):**

| comparison | tickets-big | sms-natural |
|---|---|---|
| A vs B (injection dose changes) | mean 7.324, max 20.185, 0% of rows within 0.001 | mean 7.865, max 47.273, 0% |
| B vs C/D/E (source modes vary, dose constant) | mean 0.0000, max 0.0010, **99.5% bit-identical** | mean 0.0000, max 0.0010, **99.8% bit-identical** |

Findings, all measured:

1. **The model-xl Snake observation reproduces exactly.** With the
   injection dose held constant (B–E), settled energy is bit-identical
   (spread ≤ 0.001 on 99.5–99.8% of rows; the exceptions are float
   rounding) while the winner changes on 1.5–23.9% of rows. Source modes
   are readout-side selectors over one unchanged field.
2. **Energy-norm on a trained schema is amplification, not
   re-decision:** flipping 0.1% of winners at unchanged accuracy (0.971 →
   0.971) while margins rise 26× (0.0355 → 0.9430) and confidence 0.002 →
   0.923. It also lifts the 35 baseline deferrals to 0 — its documented M1
   purpose. On zero-shot probes (near-uniform baseline) the same mechanism
   CAN flip winners, because there the field has no stable ordering to
   preserve.
3. **Salience gating re-reads the field and is fabric-dependent:**
   −4.1 points on tickets-big (0.971 → 0.930) but +19 points on
   sms-natural (0.160 → 0.360), where the full-source readout is dominated
   by harvest noise (see PROBING.md) and salience-ranked sources suppress
   it. Same field in both cases.
4. **High-confidence-wrong rows (conf ≥ 0.5 while wrong):** baseline 0 on
   both fabrics (confidence 0.000-0.002 never crosses the bar) → 15–86
   rows under amplified configs. Amplified confidence without accuracy
   gain is exactly the D-config risk the model-xl report demonstrated with
   0.805 on an illegal move.
5. **Benchmark numbers are measured under their documented config.** Flags
   re-shape the readout; any flagged operating point is a different
   measurement and must be re-benched before it is quoted.

---

## Part 3 — interpretation rules

1. **Confidence = concentration, never correctness.** Use
   `decide --defer-margin P` (or server `options.defer_margin`) so
   sub-threshold separations are disclosed as `deferred: true, reason:
   low_margin` instead of looking like decisions.
2. **Flags are diagnostic instruments, not defaults.** They answer "what
   does the field look like through a different readout" — they do not
   improve the field. Default-config numbers remain the reference.
3. **Zero-shot probes bound, they never establish.** A probe can
   demonstrate a failure (Snake #5, Tic-Tac-Toe position 3, file routing)
   and motivate a held-out run; it cannot estimate accuracy. Training on a
   task and re-probing is the only path to a number.
4. **Sweep the wording, not just the flags** — the matched/reworded/empty
   sweep of PROBING.md applies to every zero-shot probe here; near-uniform
   distributions are the probe-condition family, not a dead fabric.

## Data sources

- **banking77**: PolyAI task-specific dialogues (`data/bank77_*`; bulk
  splits gitignored, regenerable via `tools/bank77_prepare.py`).
- **SMS**: UCI SMS Spam Collection (free-for-research corpus;
  `data/sms_*.jsonl` committed so the v3.1.1 probe experiment and this
  flag experiment reproduce from a fresh clone).
- **tickets/game/guard big splits**: synthetic corpora (`data/big/`,
  SHA-256 `MANIFEST.json`).
