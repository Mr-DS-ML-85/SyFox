# SyFox Roadmap

## v0.1 — System One on SI physics (this release)
- [x] Concept field with occurrence mass (1/√mass damping at use sites),
      dissipative settle, energy-gated source capping
- [x] Hebbian lane learning + anti-Hebbian weakening (Noul)
- [x] Choice / Score / Noul typed readouts on one settled field
- [x] Honest silence: defer instead of guess
- [x] Tool-side calibration (temperature + Platt) on readout energies
- [x] CLI (learn / calibrate / decide / demo / stats), unit tests
- [x] Jev-compatible HTTP API (`POST /v1/systemone`) via ctypes bridge
- [x] Three seed domains: support tickets, game bot, coding-agent guardrail

## v0.2 — Depth and robustness
- [x] Character n-gram concept lanes: typo robustness without touching the core physics (v2.2: trigram bridges + traction gate; `bench --typos` measures)
- [x] Physics port (post-audit): SI cavity salience integrator
      tanh(s·decay + gain·|ΔE|) runs every pass; `salience_gating` and
      `miller_window` ([cap−4, cap], TSDA live_cap) ship behind config flags.
      Bench on 10 in-domain decisions: default 10/10 (stays default),
      salience+[5,9] 10/10 but saturated confidence, salience-only 4/10.
      Harness now exists (`syfox bench`); salience-mode benching still to
      do before the verdict is reopened.
- [x] Jev-parity eval harness (`syfox bench --model --eval`): routing
      accuracy, ECE + confidence gap, OOD defer rate, guardrail hold
      precision/recall, latency p50/p95, determinism (byte-identical
      replays), close-call margin distribution. Axis lineage from TypeSafe's
      published Jev material (ARCHITECTURE §12); `make bench` on the seed
      models.
- [x] No-regression derivation gate (`syfox derive --gate`): transactional
      derive — bit-exact fabric snapshot, replay taught rows + close-call
      probes, revert on any taught flip or manufactured certainty on
      ambiguous states. Strength-scan verdict: model-game ships derived at
      the conservative strength, tickets/guard provably cannot (ARCHITECTURE
      §11). This retires the "seed models ship un-derived" advice: the gate
      enforces it per model.
- [x] Recall: associative retrieval via settled-field cosine (Hopfield-style,
      zero symbol-space similarity) — `syfox recall --state --memories`
      (ARCHITECTURE §13)
- [ ] Two-stage choice for >255 options (energy shortlist → resonance rescore)
- [ ] Relational Noul via dual-injection interference readout (command ∩ task overlap)
- [ ] CI regression files (record decisions, diff on retrain) — a libre `jevassert` analog
- [ ] More seed data per domain; tone readout re-evaluated with ~100 rows

## v0.3 — Structure
- [x] Derivation layer port (shipped early, v0.2): lane induction in two
      modes (compose: damped two-hop algebra + ghost-guarded verifier;
      harvest: settle co-activation replay), dreamer emergent-resonance
      ledger with human-validated promote, read-only analogical mapping.
      Lane provenance (generation counter) persisted in substrate.bin v2.
      Strength-scan verdict enforced by the no-regression gate (v0.2):
      model-game ships derived, tickets/guard provably revert at every
      strength (ARCHITECTURE.md §10–11)
- [x] Eval harness (delivered early as `syfox bench` + the derivation gate;
      see v0.2) — gates derive/harvest adoption per model and measures the
      salience flags when that question is reopened
- [ ] Lane graph introspection CLI (`syfox inspect --concept refund`) — audit what the engine knows
- [ ] Multi-label choice (several true options)
- [ ] Streaming batches: many states against one model in a single process
- [ ] Optional Bengali token folding parity with the SI repo's lang layer

## v1.0 — Production posture
- [ ] Concurrency: field-per-thread engines, zero-copy bridge
- [ ] Benchmarks vs a hosted decision API on latency and cost (publish methodology)
- [ ] Security review of the bridge + fuzzing the JSON parser
- [ ] Packaging: pip wheel with prebuilt core, single static binary

## Non-goals (by constitution)
- No transformer, no neural network, no pattern-matching classifier in the core.
- No text generation. SyFox returns typed values or silence — nothing else.
- No cloud dependency. The substrate runs on your CPU, offline, forever.

## v2.1 — evaluation-methodology fix (this release)
- [x] Held-out 70/30 split, stratified, deterministic (`tools/split_data.py`);
      `make models` trains on `*_train.jsonl` only; `syfox bench --split
      heldout` scores rows the fabric never learned from. Resubstitution
      retired as a headline number (in-sample kept as a labeled contrast).
- [x] Coverage-vs-accuracy curve (`syfox bench --coverage-curve`): threshold
      sweep 0.0→1.0, table + ASCII plot, operating points at ≥50/70/90%
      coverage. The headline metric.
- [x] Paraphrase augmentation (`tools/gen_paraphrases.py`, 5 variants/state,
      train-only by construction). Measured on held-out: REGRESSES accuracy
      (intern mass linear, injection damps 1/√mass) — shipped un-augmented by
      default, `make models-augmented` reproduces the A/B.
- [x] Token boundary: Porter (1980) stemming + synonym folding table
      (`data/synonyms.txt`, `--synonyms`), deterministic, no NN, substrate
      physics byte-identical.
- [x] Derivation gate replays held-out rows; per-model verdicts logged
      (game PASS/derived, tickets+guard REVERT) with gold accuracy
      before/after.
- [x] Calibration fitted on held-out rows: multi-class NLL over the full
      candidate vector, 1-bit ECE adoption guard, pairwise-blind-spot and
      0.005-clamp defects fixed. Tickets held-out ECE 0.665 → 0.012.
- [ ] Deeper held-out sets (13 rows today): the harness is ready; more data
      makes every number mean more without code changes.
- [ ] Character n-gram concept lanes (carried from v0.2): typo robustness —
      now measurable on the held-out split.

## v3.0 — milestones 1–5 (this release)
- [x] M1 large held-out datasets: deterministic generator (seed 20260927),
      12k/8k/8k rows per domain at 70/15/15 train/cal/hidden; hidden test
      quarantined by core/firewall.hpp (learn/calibrate/derive--gate refuse
      it — `make firewall-check`); hidden reports tickets 0.971 (ECE 0.005),
      game 0.763 (ECE 0.049), guard 0.638 (ECE 0.037); Bengali 0.969 /
      Hindi 0.965 / Russian 0.970 on hidden splits of 540/539/539; coverage
      curves feasible at all three operating points on every domain.
- [x] M2 distinct experience: `--dedup` (auditable `dedup_skipped`);
      A/B at equal lesson counts — 8,400 distinct rows 0.825 vs 1,050 rows
      ×8 repeats 0.782, with 2.1× the lanes; novelty weighting measured
      neutral and shipped OFF.
- [x] M3 contradiction + provenance: per-lane evidence ledger (support /
      counter events, generation, seq window, context) persisted and
      projected by `decide --evidence`; contradictions detected, recorded,
      never silently overriding (adversarial suite: silent_override=false
      on all domains).
- [x] M4 adversarial/OOD suite: `bench --adversarial` with deterministic
      families (reorder, padding, typos, intensifiers, self-contradiction,
      negation, double negation, unknown concepts, near-miss, cross-domain,
      conflicting lessons) and per-family accuracy / defer / false-conf /
      conf-when-wrong. Weakest family documented (double negation).
- [x] M5 multicore → GPU gate: CSR mirror (order-preserving, bit-identical),
      OpenMP deterministic parallel settle PROVEN bit-identical
      (`make omp-identity` at 0.971), batched throughput 1.554× on 2 cores
      with private per-worker engines, GPU gate measured per fabric
      (`make density`) and a port kept design-only until substrates are
      dense enough. `--deterministic` and `--throughput` modes both
      preserved.
- [ ] GPU port proper: gated on fabric density (see `make density`); the
      SoA/CSR layout is the only GPU-specific preparation in the core.
- [ ] Guard-domain deepening: hold precision 0.782 has headroom; the
      contradiction blast-radius measurement (~5.1 pts on tickets) suggests
      curriculum ordering (teach disputes late) as a measurable lever.
