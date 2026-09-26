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
| No-regression gate (§11) | SyFox-native: transactional derive + bit-exact fabric snapshot/restore; policy refined by a full strength-scan (taught-row integrity + no manufactured certainty) |
| Jev-parity bench (§12) | SyFox-native: axis lineage from TypeSafe's published Jev material + community classification benchmarks; all numbers produced by this repo's binary |
| Recall (§13) | Hopfield-style associative memory reinterpreted as settled-field cosine — SI-physics-native, zero symbol-space similarity |

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
flipped. Verdict (v0.2, after the strength-scan): both modes are real and
pinned by tests, but NO strength — down to 32 lanes — avoids taught-row flips
on the mixed-label tickets/guard fabrics. That verdict is now ENFORCED by the
no-regression gate (§11) instead of shipped as advice; model-game passes and
ships derived.

## 11. The no-regression gate (transactional derivation)

Derivation writes into the fabric; the gate (`core/gate.hpp`, CLI
`syfox derive --gate FILE.jsonl`) makes that write safe by construction:

1. replay every labelled row + generated close-call probes (cross-domain
   state mixtures, deterministic generator shared with `bench`) → baseline
   decision signatures;
2. snapshot the fabric **bit-exactly** — `snapshot_fabric()` copies each
   node's lane vector in insertion order plus the provenance map, because
   `settle()` sums over lanes in vector order (a weight set is not the
   state);
3. run the derivation (conservative presets: harvest co_floor 0.15/gain
   0.35/budgets 8·512, compose gain 0.25/top-2/budget 4);
4. replay the same probe set → compare;
5. **policy**: (a) any argmax flip on a TAUGHT row reverts — labelled
   knowledge must survive derivation; (b) any RISE in mean confidence on the
   MIXED probes reverts — close-call states are ambiguous by construction,
   their argmax may re-resolve, but manufactured certainty about them is the
   exact failure the gate exists to kill;
6. revert = `restore_fabric(snapshot)` → the fabric is bit-identical and
   `decide()` output is byte-identical (pinned by test); commit = the caller
   saves the model.

Strength-scan evidence (scripts/syfox_gate_scan.cpp, all strengths from
default down to 32 lanes): tickets reverts everywhere (8–18 taught flips —
its seed data mixes refund/invoice labels across billing and sales), guard
reverts at every useful strength, game commits at the conservative strength.
Per-model outcome: `model-game` ships derived (668 lanes, 0 taught flips,
mixed mean confidence 0.975→0.886 — the fabric got MORE honest about close
calls; demo argmaxes 3/3 unchanged; `make models` reproduces). `model-tickets`
and `model-guard` ship un-derived because the gate proves it, not because a
human remembered to check.

## 12. The Jev-parity benchmark (`syfox bench`)

`core/bench.hpp` measures the axes the System One model class is judged on
(axis lineage verified against TypeSafe's published Jev material and the
community classification benchmarks, Sept 2026): decision accuracy, latency
percentiles, calibration (ECE, 10 bins, + conf-when-correct vs
conf-when-wrong gap), honesty (OOD probes built from deterministic nonsense
vocabulary checked against the model's own — untaught tokens must defer),
guardrail hold precision/recall (noul confusion at the 0.5 threshold),
determinism (full replay compared byte-wise), and close-call margin
distribution (mixed cross-domain states; the same generator the gate
replays). Accuracy/calibration/guardrail aggregate over LABELLED rows only;
unlabelled mixed probes never pollute accuracy denominators (a real bug the
e2e suite caught). Every report carries a `jev_reference` block with the
PUBLISHED figures (67.8% workflow accuracy, 70–500 ms, pi-warden 88% hold
precision) for side-by-side reading — axes shared, numbers not compared
against toy-data resubstitution.

Seed-model results (in-domain resubstitution; `make bench`):

| model | acc | choice ECE | conf ok/wrong | OOD defer | hold precision | p50 / p95 |
|---|---|---|---|---|---|---|
| tickets | 1.000 (22) | 0.405 | 0.354 / 0.000 | 1.00 | 1.000 (2/2) | 62 / 80 µs |
| game (derived) | 1.000 (9) | 0.001 | 0.999 / — | 1.00 | — | 36 / 45 µs |
| guard | 0.846 (13) | 0.243 | 0.795 / 0.451 | 1.00 | 0.750 (3/4) | 33 / 40 µs |

Reading: honesty is structural (1.00 defer everywhere); wrong answers already
carry near-zero confidence on the mixed-label fabrics; latency is three
orders of magnitude under the Jev band with the same one-pass shape. Score
accuracy on tickets (0.682) is the documented tone-readout data limit, now
measured instead of asserted.

## 13. Recall (associative retrieval in energy space)

`core/recall.hpp`, CLI `syfox recall --state '...' --memories FILE.jsonl`.
Content-addressable memory the only way the constitution allows: settle the
query into an energy fingerprint (one float per node, L2-normalized), settle
each stored memory the same way, rank by cosine of the two settled fields.
No token comparison, no string distance, no n-gram overlap, no embedding
table, no transformer — similarity is measured in the substrate's own state
space, so two states resonate exactly as far as the field routes them
together. Unknown vocabulary resonates with nothing (empty hits — honest
silence, since `inject` skips unknown tokens). Read-only, bit-deterministic;
memory stores accept `{"state","label"}` rows or the training schema
(label = first label value). On the derived game model the zombie query
recalls the flee lessons at 0.9696–0.9421 with fight lessons at ≤0.856.
