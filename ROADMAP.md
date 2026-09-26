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
      Final adoption verdict waits on the eval harness below.
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
      Benchmarked: sharpens well-separated fabrics, unsafe on 22-row seeds —
      adoption gated on the eval harness below (ARCHITECTURE.md §10)
- [ ] Eval harness: `syfox eval --model --examples` with accuracy + reliability
      diagrams — this gates derive/harvest adoption and the salience flags
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
