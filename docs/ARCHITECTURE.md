# SyFox Architecture — the SI substrate core

This document specifies exactly what happens between "state in" and "decision
out", and which mechanisms are ported from the Synthetic-Intelligence (SI)
research substrate. The rule that governs this codebase:

> **The core is SI physics only.** No transformer, no neural network, no
> pattern-matching classifier. Everything below is field dynamics over an
> explicit concept graph. Tools (JSON, CLI, HTTP, calibration math) live
> outside the core and never rewrite it.

## 1. Concepts and acoustic mass

Every lowercase folded token interned by the engine becomes a **concept node**
(`core/si_substrate.hpp`, `Substrate::intern`). Nodes carry:

* **mass** — a raw occurrence count: `intern()` adds exactly `1.0` each time
  the concept appears in any lesson. It is **linear, not log-compressed**.
  *Lineage:* SI "acoustic mass" — heavier units respond less to the same
  stimulus. The sub-linear behavior lives at the **use sites**, not in the
  counter: injection deposits `energy / sqrt(mass)` and Hebbian bind strength
  scales by `1/sqrt(mass)`, so frequent concepts ("the", "please") cannot
  flood the field.
* **energy** — transient; injected and dissipated per decision.

(An earlier revision also carried a `salience` field. Audit found it was
write-only — incremented at injection, read nowhere; gating selects sources
by energy — so it was removed. Dead state in the core is doc debt.)

Token folding (`fold`) is deterministic string normalization (plural / `-ed` /
`-ing` / trailing `e`), applied identically to states, options, and
instructions. It is I/O hygiene, not linguistics and not pattern matching.

## 2. Lanes (Hebbian bindings)

Lanes are weighted edges between concepts, stored sparsely per node with a
per-node lane cap (weakest lane evicted, bounded memory).

* **Bind** (`hebbian_lesson`): for a labelled lesson, every active state
  concept binds to every outcome concept with
  `η · (1/√mass_a + 1/√mass_b) · scale`. Adjacent state tokens also bind
  (the co-occurrence fabric that lets energy diffuse inside a phrase).
  *Lineage:* SI Hebbian lane bindings in `physics.hpp`.
* **Weaken** (`weaken`): counter-evidence subtracts lane weight; a lane whose
  weight reaches zero dissolves. Used by Noul lessons labelled false.
* **Decay** (`lane_decay`): every lesson multiplies all lanes by 0.995 —
  unused routes fade.

Learning is substrate rewiring. There are no fitted weights anywhere in the
core; the calibration tool (below) is external and read-only.

## 3. Injection

A state is tokenized; every token that exists in the vocabulary deposits
`inject_energy / sqrt(mass)` on its node. Unknown tokens stay dark — this is
the first half of the honest-silence contract (a state of nothing but unknown
words settles zero energy and defers).

## 4. Settle (the System One pass)

`settle()` runs up to `k_settle` (default 8) dissipative diffusion passes:

1. Sources are gated by **source cap** (`source_cap`, 24 by default): the
   top-K nodes **by energy** propagate this pass — selection is by energy
   only; no separate salience score exists. *Lineage:* SI working-set
   capping; the SI substrate samples its cap from a Miller window [5,9],
   SyFox pins a fixed 24. Gated nodes keep their energy (they remain
   readout-visible); the cap gates flow, it does not annihilate. (An earlier
   version pruned energy outright and destroyed 97% of the field — the
   gating form is the correct physics and the tests pin it.)
2. Each source retains `(1 − diffusion)` of its decayed energy and flows
   `diffusion` out along its lanes, split proportionally to lane weight:
   `next[b] += e·decay·diffusion·(w_ab / Σw_a)`.
3. Total energy is non-increasing (dissipative); the pass loop stops early
   when the field relaxes (`total_before − total_after < eps`).

One settle per decision — all questions read the same settled field. Adding
questions never re-settles the substrate: flat marginal latency by structure,
not by scheduling.

## 5. Readout (resonance sweep, two lanes)

Probes read the settled field at their own concepts: direct node energy plus a
hop term through lanes. Two readout lanes exist because two question shapes
need two different questions of the field:

* **Specific** (`choice`, `score`): mean lane support —
  `hop · Σ w·e_n / Σ w`. Rewards precise coupling; a node bound diffusely to
  half the vocabulary cannot out-read a node bound exactly to this state.
  Options are then normalized by softmax at the fitted temperature.
* **Support** (`noul`): active lane mass — `hop · Σ w·e_n` (unnormalized).
  A statement may be corroborated broadly; what matters is how much
  lane-connected energy the field offers it. The ratio to total field energy
  is the statement's **support**.

Both are field measurements. No weights are fitted at readout; no classes
exist in the core.

## 6. Honest silence

If settled field energy is below `silence_floor` (0.05):

* empty state → `deferred: true, reason: "empty_state"`
* all-unknown vocabulary → `reason: "unknown_vocabulary"`
* settled but dark → `reason: "honest_silence"`

The engine refuses to guess from nothing. *Lineage:* the SI interface's
honest-silence dispatch — answers only when the readout actually collapsed.

## 7. Calibration (a tool, not core)

`fit_calibration` collects readout energies on labelled rows and fits, purely
outside the core:

* **temperature** per choice/score — golden-section search on NLL of
  `softmax(E_correct/T, E_wrong/T)`.
* **Platt scaling** for noul — logistic regression `sigmoid(a·support + b)`
  on supports, gradient descent to convergence (separable data needs a steep
  slope; the fit runs 4000 iterations with decaying lr).

Calibration changes how energies are *reported* as probabilities, never how
they are produced. Confidence for choice/score is the normalized-entropy
margin of the distribution; for noul it is `|p − 0.5|·2`.

## 8. Persistence

`save_model` writes `substrate.bin` (vocabulary + masses + lanes, binary),
`calibration.json`, and `meta.json`. Loading is exact — the same state settles
to the same readout bit-for-bit (pinned by tests).

## 9. Provenance

| Mechanism | Origin |
|---|---|
| Energy injection + field settle | SI `physics.hpp` substrate dynamics |
| Source capping (top-K by energy) | SI working-set cap (SI samples a Miller window [5,9]; SyFox fixes 24) |
| Hebbian lane bindings + decay | SI lane learning |
| Honest silence | SI dispatch contract |
| Typed Choice/Score/Noul API | interface shape inspired by TypeSafe Jev |
| Temperature/Platt calibration | classical statistics (tool layer) |
