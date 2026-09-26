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
   top-K nodes propagate each pass — ranked **by energy** by default, or by
   **salience** when `salience_gating` is on. Gated nodes keep their energy
   (they remain readout-visible); the cap gates flow, it does not
   annihilate. (An earlier version pruned energy outright and destroyed 97%
   of the field — the gating form is the correct physics and the tests pin
   it.)

   **SI salience lineage, restored after audit.** The SI substrate runs a
   cavity salience integrator, `s = tanh(s·decay + gain·|velocity|)`
   (`physics.hpp`, `salience_gain = 0.05`, `salience_decay = 0.995`), and
   its TSDA layer samples the live working cap from
   `[working_cap−4, working_cap]` = [5,9] (Miller 7 ± 2). SyFox ports both:
   the integrator runs **every pass** on the motion proxy `|ΔE|` (a node at
   rest is an exact fixed point, mirroring SI's sparsity guard), and
   `miller_window = true` samples the live cap from
   `[source_cap−4, source_cap]` each decision. One deliberate adaptation:
   SI draws the sample from a seeded mt19937 stream; SyFox derives it from
   the decision's state hash, so the same state always settles to the same
   field bit-for-bit (the persistence contract outranks stochasticity).
   `inject()` spikes salience to 1.0 on touch (TSDA spike semantics).

   *Benchmark (10 in-domain decisions, 3 domains).* default energy gating
   10/10 with calibrated-shaped confidences; salience ranking at the wide
   cap 4/10 (motion history is a poor selector when the cap rarely trims);
   salience + literal [5,9] window 10/10 argmax but probabilities saturate
   at 1.0 (calibration collapse — needs the v0.2 eval harness before
   adoption); Miller window at wide cap 9/10. Verdict: default stays
   energy-gated at a fixed 24; the SI-faithful modes ship behind config
   flags until a held-out eval says otherwise.
2. Each source retains `(1 − diffusion)` of its decayed energy and flows
   `diffusion` out along its lanes, split proportionally to lane weight:
   `next[b] += e·decay·diffusion·(w_ab / Σw_a)`.
3. Total energy is non-increasing (dissipative); the pass loop stops early
   when the field relaxes (`total_before − total_after < eps`).

One settle per decision — all questions read the same settled field. Adding
questions never re-settles the substrate: flat marginal latency by structure,
not by scheduling.

**Derivation, not retrieval.** Because diffusion is multi-hop, a state that
was never in any lesson still settles to a computed answer: energy crosses
the co-occurrence fabric into outcome lanes laid down by *other* lessons.
Probe (two lessons, `alpha beta`→x and `beta gamma`→y): the untaught state
`alpha` alone settles with y at 0.494 — lesson 2 never mentioned alpha; the
two-hop path alpha→beta→y carried the evidence. The boundary is the taught
vocabulary and lane fabric: concepts never interned have no node, unknown
words settle to zero energy and defer (§6). This is the substrate-level
kernel of the SI stack's explicit derivation layers (meta-learner
macro-rules, dreamer emergent resonances, analogical mapping) — not yet
ported; see ROADMAP v0.3.

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
| Source capping (top-K, energy or salience) | SI working-set cap + TSDA live_cap ([cap−4, cap], Miller 7 ± 2) |
| Salience integrator tanh(s·decay + gain·motion) | SI `physics.hpp` cavity salience (verified live there) |
| Hebbian lane bindings + decay | SI lane learning |
| Honest silence | SI dispatch contract |
| Typed Choice/Score/Noul API | interface shape inspired by TypeSafe Jev |
| Temperature/Platt calibration | classical statistics (tool layer) |
| Derivation layer (§10) | SyFox-native design after studying SI meta_learner / dreamer / analogy as mechanism reference — no code copied; weight algebra on lanes instead of typed relation rules |

## 10. Derivation layer (offline, explicit)

SyFox's lanes so far carried only lived experience (`learn`). The derivation
layer lets the field grow knowledge **derived from** its own fabric — the
substrate-level kernel of what the upstream SI stack does with explicit rule
machinery (meta-learner macro-rules, dreamer emergent resonances, analogical
mapping). The primitives differ by constitution: SI composes typed relation
triples over a fact graph; SyFox composes **weighted lane paths** — weight
algebra and field dynamics, no classifiers. All engines are OFFLINE, explicit
CLI steps; `decide()` stays read-only and byte-identical unless the operator
derives on a model. Provenance: derived lanes carry a **generation counter**
(experienced / human-promoted = 0, each composition step +1, cap 3), persisted
as a versioned tail in `substrate.bin`; v1 files load clean (generation 0).

**1. Lane induction, compose mode** (`syfox derive --model DIR`): a two-hop
path A→B→C is evidence for A↔C, with three guards that each came out of a
real benchmark failure:
* *acoustic-mass damping* — path evidence is damped by `1/√mass(B)`, the same
  law `settle()` applies to energy through B; without it, hub tokens ("the",
  "command") fabricated corroboration between unrelated concepts and two demo
  argmaxes flipped;
* *top-K corroboration* (K=3) — a dense fabric offers hundreds of two-hop
  paths; treating them as independent evidence saturated the field (noul
  collapsed to 0.999 everywhere);
* *ghost-guarded verifier* — every derived lane is re-checked against the
  LIVE fabric (snapshot ghosts of already-dissolved lanes cannot justify
  anything), over-claims are healed down to their justification, unsupported
  ones dissolve. Composition is raise-only: observed lanes are never weakened.

**2. Lane induction, harvest mode** (`syfox derive --model DIR --examples
FILE.jsonl`): static algebra guesses; the settled field knows. Replay states
through inject→settle and record which concept pairs genuinely **co-activate**.
Dissipation does the filtering — cross-scenario energy decays before it
registers. Observed lanes are untouchable; gen-1 lanes must be re-nominated by
every run or they dissolve (justification is recomputable, hence deterministic).
Labels are not needed: the field learns structure from unlabelled exposure.

**3. Dreamer** (`syfox dream --model DIR [--steps N] [--seed S]`): inject
small random energy patterns, settle, and watch for undriven nodes that lit
up **without any direct lane to the driven set** — reached only through the
topology's own interference. Candidates go to `model-*/mutations.jsonl`. The
substrate is never modified by dreaming; a human sets `validated:true` on a
line and `syfox promote` applies it as a premise-grade lane. Same seed → same
dream, bit for bit.

**4. Analogy** (`syfox analogs --model DIR --concept WORD`, read-only): a
concept's signature is its lane neighbourhood (who it touches, both
directions); structural isomorphism = signature Jaccard; novelty potential
follows the L9 formulation `Phi = I·exp(k·d)`. High Phi = structurally aligned
AND fabric-distant: a transfer *hypothesis* for the human, never auto-applied.

*Benchmark (v0.2, honest).* compose on model-tickets: all 3 demo argmaxes held
and sharpened (technical conf 0.251→0.454, sales 0.579→0.812); on model-game
1 argmax flipped; on model-guard confidences collapsed and 1 flipped. harvest
on model-tickets: billing 0.604→0.912 (conf 0.679) but one close-call demo
flipped. Verdict: both modes are real and pinned by tests, but on 22-row seed
fabrics the scenario vocabularies overlap too much for derivation to be safe
as a default — recommend ≥100 rows per domain or well-separated vocabularies,
and gate adoption on the v0.2 eval harness. The seed models ship un-derived;
`make models` is unchanged.
