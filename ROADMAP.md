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
- [ ] Character n-gram concept lanes: typo robustness without touching the core physics
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
