// ============================================================================
//  SyFox — derive.hpp
//  The derivation layer: knowledge the field derives from its own fabric.
//
//  Lineage: SyFox-native design after studying the upstream SI stack's
//  derivation mechanisms (meta-learner rule induction, dreamer discovery
//  daemon, analogical mapping) as MECHANISM REFERENCE ONLY — no code copied.
//  The primitives differ: SI composes typed relation triples over a fact
//  graph; SyFox composes weighted lane paths over the substrate fabric,
//  so every rule below is weight algebra, not pattern matching. The core
//  constitution holds: no transformer, no neural network, no classifier.
//
//  Three engines:
//
//  1. LANE INDUCTION (meta-learner analog)
//     A two-hop path A->B->C is evidence for A<->C. compose_pass() turns
//     corroborated paths into derived lanes: weight = noisy-or of path
//     products, damped by compose_gain (derived evidence is weaker than
//     experience). Each derived lane carries a generation counter —
//     composition of generations >= max_generation is refused AT WRITE TIME,
//     so a stale-rule verifier bug (an SI audit finding) cannot occur here
//     by construction: provenance is checked, never cached as an id.
//     prune_stale() is the use-time verifier: a derived lane must still be
//     justified by live parent paths; over-claiming lanes are healed down
//     to their justification, unsupported ones dissolve.
//     Derived lanes NEVER weaken observed lanes: compose only raises.
//
//  2. DREAMER (discovery analog)
//     Inject small random energy patterns (SyFox physics has no drive
//     frequency — the honest adaptation of SI's random cavity drive),
//     settle, and watch for UNDRIVEN nodes that lit up strongly WITHOUT
//     any direct lane to the driven set: reached only through the
//     topology's own interference. Those become candidate facts in the
//     mutation ledger. The substrate is never modified by dreaming; a
//     human validates a ledger line before promote() may apply it.
//
//  3. ANALOGY (structure-mapping analog)
//     A concept's signature is its lane neighbourhood (who it touches, in
//     which direction). Structural isomorphism = signature Jaccard; the
//     novelty potential follows the L9 formulation Phi = I * exp(k * d):
//     structurally aligned AND far apart on the fabric = candidate
//     transfer. v0.2 is read-only introspection — transfers are hypotheses
//     for the human, never auto-applied.
//
//  All engines are OFFLINE, DETERMINISTIC (no RNG except the seeded dreamer)
//  and EXPLICIT: decide() stays read-only and byte-identical unless the
//  operator runs derive/promote on a model.
// ============================================================================
#pragma once
#include "si_substrate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace syfox {
namespace derive {

// ---------------------------------------------------------------------------
struct DeriveConfig {
    float         path_min       = 0.02f;  // min parent weight to count as evidence
    float         compose_gain   = 0.50f;  // derived evidence is weaker than lived evidence
    float         compose_floor  = 0.005f; // below this, a derived lane is not worth laying
    std::uint32_t max_generation = 3;      // derived-of-derived depth cap (verifier bound)
    int           max_paths      = 3;      // corroboration cap: top-K paths per pair.
                                            // Dense fabrics offer hundreds of two-hop
                                            // paths; treating them as independent
                                            // evidence saturates the field (benchmarked:
                                            // argmax flips + noul collapse). Top-K
                                            // keeps derived lanes in evidence range.
    int           max_derived_per_node = 32; // creation budget per source per pass
};

struct DeriveStats {
    long created      = 0;
    long strengthened = 0;   // derived evidence raised an existing lane
    long healed       = 0;   // verifier lowered an over-claiming derived lane
    long dissolved    = 0;   // verifier dissolved an unsupported derived lane
};

// -- 1a. LANE INDUCTION: one composition pass (two-hop) ----------------------
// Deterministic: nodes in id order, candidates accumulated before any write
// (no intra-pass cascade), floating accumulation in a fixed order.
inline DeriveStats compose_pass(si::Substrate& s, const DeriveConfig& c) {
    const std::size_t N = s.node_count();
    // adjacency snapshot with generation: a -> [(b, w, gen)]
    std::vector<std::vector<std::pair<si::NodeId, std::pair<float, std::uint32_t>>>> adj(N);
    s.for_each_lane([&](si::NodeId a, si::NodeId b, float w) {
        adj[a].push_back({b, {w, s.generation_of(a, b)}});
    });

    // candidate accumulation: cand[a][i] = {c, top-K path products, gen}
    struct Cand { si::NodeId c; std::vector<float> paths; std::uint32_t gen; };
    std::vector<std::vector<Cand>> cand(N);

    for (std::size_t a = 0; a < N; ++a) {
        for (const auto& ab : adj[a]) {
            if (ab.second.first < c.path_min) continue;            // A->B too weak
            const si::NodeId b = ab.first;
            if (b >= N) continue;
            for (const auto& bc : adj[b]) {
                if (bc.second.first < c.path_min) continue;        // B->C too weak
                const si::NodeId cc = bc.first;
                if (cc == a) continue;                             // no self-lane
                const std::uint32_t parent_gen = std::max(ab.second.second, bc.second.second);
                if (parent_gen >= c.max_generation) continue;      // verifier bound, at write time
                // acoustic-mass law: evidence through an intermediate is damped
                // by 1/sqrt(mass), the same law settle() applies to energy
                // flowing through that node. Hubs ("the", "and") therefore
                // cannot fabricate corroboration between unrelated concepts.
                const float damp = 1.0f / std::sqrt(std::max(1.0f, s.node_mass(b)));
                const float evidence = ab.second.first * bc.second.first * damp;
                Cand* slot = nullptr;
                for (auto& k : cand[a]) if (k.c == cc) { slot = &k; break; }
                if (!slot) {
                    cand[a].push_back({cc, {evidence}, parent_gen + 1});
                    slot = &cand[a].back();
                } else {
                    slot->paths.push_back(evidence);               // top-K corroboration
                    slot->gen = std::min(slot->gen, parent_gen + 1); // strongest provenance wins
                }
                if (static_cast<int>(slot->paths.size()) > c.max_paths) {
                    auto weakest = std::min_element(slot->paths.begin(), slot->paths.end());
                    slot->paths.erase(weakest);                    // drop the weakest path
                }
            }
        }
    }

    DeriveStats st;
    for (std::size_t a = 0; a < N; ++a) {
        long budget = c.max_derived_per_node;
        for (const auto& k : cand[a]) {
            if (budget <= 0) break;                                // per-source creation budget
            // corroboration over at most the retained top-K paths
            float p = 1.0f;
            for (float e : k.paths) p *= (1.0f - e);
            p = 1.0f - p;
            const float derived = p * c.compose_gain;
            if (derived < c.compose_floor) continue;
            const float existing = s.lane_weight(static_cast<si::NodeId>(a), k.c);
            if (existing <= 0.0f) {
                s.bind_derived(static_cast<si::NodeId>(a), k.c, derived, k.gen);
                ++st.created; --budget;
            } else if (derived > existing) {
                s.bind_derived(static_cast<si::NodeId>(a), k.c, derived, k.gen);
                ++st.strengthened;
            }
        }
    }
    return st;
}

// -- 1b. VERIFIER: use-time re-check of every derived lane -------------------
// A derived lane must be justified by live parent paths under the CURRENT
// fabric. Over-claimed -> healed down to justification; unjustified -> gone.
// Observed lanes (gen 0) are premises: never questioned here.
inline DeriveStats prune_stale(si::Substrate& s, const DeriveConfig& c) {
    const std::size_t N = s.node_count();
    struct Lane { si::NodeId a, b; float w; std::uint32_t gen; };
    std::vector<Lane> lanes;
    s.for_each_lane([&](si::NodeId a, si::NodeId b, float w) {
        lanes.push_back({a, b, w, s.generation_of(a, b)});
    });
    std::vector<std::vector<std::pair<si::NodeId, float>>> adj(N);
    for (const auto& l : lanes) adj[l.a].push_back({l.b, l.w});

    DeriveStats st;
    for (const auto& l : lanes) {
        if (l.gen == 0) continue;                                  // premise: skip
        float best_path = 0.0f;
        for (const auto& ab : adj[l.a]) {
            if (ab.second < c.path_min) continue;
            // live-fabric authority: a lane dissolved/healed earlier in this
            // pass must not justify anything through its snapshot ghost
            if (s.lane_weight(l.a, ab.first) != ab.second) continue;
            if (s.generation_of(l.a, ab.first) >= l.gen) continue; // parents must be senior
            const float damp = 1.0f / std::sqrt(std::max(1.0f, s.node_mass(ab.first)));
            for (const auto& bc : adj[ab.first]) {
                if (bc.first != l.b || bc.second < c.path_min) continue;
                if (s.lane_weight(ab.first, l.b) != bc.second) continue;  // ghost guard
                if (s.generation_of(ab.first, l.b) >= l.gen) continue;
                best_path = std::max(best_path, ab.second * bc.second * damp);
            }
        }
        const float justified = best_path * c.compose_gain;
        if (best_path <= 0.0f) {
            s.dissolve_lane(l.a, l.b);
            ++st.dissolved;
        } else if (justified < l.w - 1e-6f) {
            s.set_derived_weight(l.a, l.b, justified);
            ++st.healed;
        }
    }
    return st;
}

// Full derivation run: verify to convergence, then compose `passes` times.
// Each prune pass can dissolve lanes that were ghost-justifying others; the
// fixed-point loop makes one run() self-consistent.
inline DeriveStats run(si::Substrate& s, const DeriveConfig& c, int passes) {
    DeriveStats st;
    for (int round = 0; round < 4; ++round) {
        DeriveStats p = prune_stale(s, c);
        st.healed += p.healed; st.dissolved += p.dissolved;
        if (p.healed == 0 && p.dissolved == 0) break;              // fixed point
    }
    for (int i = 0; i < passes; ++i) {
        DeriveStats p = compose_pass(s, c);
        st.created += p.created; st.strengthened += p.strengthened;
    }
    return st;
}

// -- 1c. DYNAMIC HARVEST: the field's own dynamics nominate the lanes --------
// Static algebra guesses; the settled field KNOWS. Replay states through
// inject->settle and record which concept pairs genuinely co-activate.
// Dissipation does the filtering: energy that crosses scenario boundaries
// decays before it can register, so the harvest picks up same-context
// bridges and skips cross-scenario noise that static products cannot see.
// This mirrors the upstream SI meta-learner's stance (induce from observed
// derivation traces, not from static structure). Deterministic: replay in
// file order, fixed accumulation order, weight-sorted laying.
struct HarvestConfig {
    float co_floor        = 0.08f;  // min normalized co-activation to count
    float gain            = 0.50f;  // derived evidence is weaker than lived evidence
    int   max_per_node    = 16;     // laying budget per node
    int   max_total       = 4096;   // global laying budget per harvest run
};

struct HarvestStats { long created = 0, refreshed = 0, dissolved = 0; };

inline HarvestStats harvest(si::Substrate& s,
                            const std::vector<std::vector<std::string>>& replay_states,
                            const HarvestConfig& hc) {
    const std::size_t N = s.node_count();
    HarvestStats st;
    if (N == 0 || replay_states.empty()) return st;

    // undirected pair key
    const auto pair_key = [](si::NodeId x, si::NodeId y) {
        return (static_cast<std::uint64_t>(std::min(x, y)) << 32) | static_cast<std::uint32_t>(std::max(x, y));
    };

    std::unordered_map<std::uint64_t, float> acc;
    for (const auto& toks : replay_states) {
        s.reset_field();
        s.inject(toks);
        s.settle();
        std::vector<si::NodeId> act;
        for (std::size_t i = 0; i < N; ++i)
            if (s.node_energy(static_cast<si::NodeId>(i)) > 1e-4f) act.push_back(static_cast<si::NodeId>(i));
        for (std::size_t i = 0; i < act.size(); ++i)
            for (std::size_t j = i + 1; j < act.size(); ++j) {
                const float w = std::min(s.node_energy(act[i]), s.node_energy(act[j]));
                if (w <= 0.0f) continue;
                acc[pair_key(act[i], act[j])] += w;
            }
    }
    float mx = 0.0f;
    for (const auto& kv : acc) mx = std::max(mx, kv.second);
    if (mx <= 0.0f) return st;

    // verifier: a gen>=1 lane must be re-nominated by the CURRENT replay
    std::unordered_set<std::uint64_t> supported;
    for (const auto& kv : acc)
        if (kv.second / mx >= hc.co_floor) supported.insert(kv.first);
    std::vector<std::pair<si::NodeId, si::NodeId>> stale;
    s.for_each_lane([&](si::NodeId a, si::NodeId b, float) {
        if (s.generation_of(a, b) >= 1 && !supported.count(pair_key(a, b)))
            stale.push_back({a, b});
    });
    std::sort(stale.begin(), stale.end());
    for (const auto& p : stale) {
        if (s.lane_weight(p.first, p.second) <= 0.0f) continue;    // mirror already gone
        s.dissolve_lane(p.first, p.second);
        ++st.dissolved;
    }

    // lay in weight order (deterministic), honoring budgets
    std::vector<std::pair<std::uint64_t, float>> ranked(acc.begin(), acc.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& x, const auto& y) { return x.second > y.second; });
    std::unordered_map<si::NodeId, int> per_node;
    long laid = 0;
    for (const auto& kv : ranked) {
        if (laid >= hc.max_total) break;
        const float co = kv.second / mx;
        if (co < hc.co_floor) break;                               // sorted: rest is weaker
        const si::NodeId a = static_cast<si::NodeId>(kv.first >> 32);
        const si::NodeId b = static_cast<si::NodeId>(kv.first & 0xffffffffull);
        if (per_node[a] >= hc.max_per_node || per_node[b] >= hc.max_per_node) continue;
        const float derived = co * hc.gain;
        const float existing = s.lane_weight(a, b);
        if (existing > 0.0f && s.generation_of(a, b) == 0) continue; // observed lane: experience outranks derivation, never touched
        if (existing <= 0.0f) {
            s.bind_derived(a, b, derived, 1);
            ++st.created; ++laid;
        } else if (derived > existing) {
            s.bind_derived(a, b, derived, 1);
            ++st.refreshed;
        } else {
            ++st.refreshed;                                        // re-nominated, current weight stands
        }
        ++per_node[a]; ++per_node[b];
    }
    return st;
}

// ---------------------------------------------------------------------------
// -- 2. DREAMER --------------------------------------------------------------
struct DreamConfig {
    float e_min          = 0.6f;   // per-driven-node energy range (drive-amp analog)
    float e_max          = 1.5f;
    float support_floor  = 0.02f;  // emergent share of total settled energy
    int   min_cluster    = 2;      // >= N independent co-vibrators (interference signature)
    int   max_per_step   = 8;      // record at most this many emergent nodes per step
    int   drive_nodes_max = 3;     // 1..3 random driven nodes per dream step
};

struct DreamCandidate {
    std::vector<si::NodeId> driven;
    si::NodeId emergent;
    float support;                 // emergent_energy / total_settled_energy
};

inline std::vector<DreamCandidate> dream(si::Substrate& s, const DreamConfig& dc,
                                         std::uint64_t seed, int steps) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> energy_dist(dc.e_min, dc.e_max);
    std::vector<DreamCandidate> out;
    const std::size_t N = s.node_count();
    if (N == 0 || steps <= 0) return out;

    for (int step = 0; step < steps; ++step) {
        s.reset_field();
        // pick 1..drive_nodes_max distinct driven nodes (deterministic per seed)
        const int k = std::min<std::size_t>(
            1 + static_cast<std::size_t>(rng() % static_cast<std::uint32_t>(dc.drive_nodes_max)), N);
        std::vector<si::NodeId> order(N);
        for (std::size_t i = 0; i < N; ++i) order[i] = static_cast<si::NodeId>(i);
        for (int i = 0; i < k; ++i) {                              // partial Fisher-Yates
            const std::size_t j = i + static_cast<std::size_t>(rng() % static_cast<std::uint32_t>(N - i));
            std::swap(order[i], order[j]);
        }
        std::vector<si::NodeId> driven(order.begin(), order.begin() + k);
        for (si::NodeId id : driven) s.inject_energy(id, energy_dist(rng));
        s.settle();
        const float total = s.total_energy();
        if (total <= 0.0f) continue;

        // the driven set and its DIRECT lane neighbourhood are not emergent
        std::unordered_set<si::NodeId> excluded(driven.begin(), driven.end());
        for (si::NodeId id : driven) {
            std::vector<std::pair<si::NodeId, float>> lanes;
            s.lanes_of(id, lanes);
            for (const auto& l : lanes) excluded.insert(l.first);
        }
        // emergent: bright, untouched by direct lanes — pure topology reach
        std::vector<si::NodeId> emergent;
        for (std::size_t i = 0; i < N && static_cast<int>(emergent.size()) < dc.max_per_step; ++i) {
            if (excluded.count(static_cast<si::NodeId>(i))) continue;
            if (s.readout_neighbours(static_cast<si::NodeId>(i)) == 0) continue;  // isolated node: trivially dark
            if (s.node_energy(static_cast<si::NodeId>(i)) / total >= dc.support_floor)
                emergent.push_back(static_cast<si::NodeId>(i));
        }
        if (static_cast<int>(emergent.size()) < dc.min_cluster) continue;
        for (si::NodeId id : emergent)
            out.push_back({driven, id, s.node_energy(id) / total});
    }
    return out;
}

// ---------------------------------------------------------------------------
// -- 3. ANALOGY (read-only) ---------------------------------------------------
struct AnalogueMatch {
    std::string concept;
    float iso;                     // structural isomorphism in [0,1]
    int   hops;                    // fabric distance (cap 24; -1 -> treated far)
    float phi;                     // novelty potential I * exp(k * d)
};

inline std::vector<AnalogueMatch> find_analogues(const si::Substrate& s,
                                                 const std::string& source,
                                                 int top_k = 8,
                                                 float min_iso = 0.10f,
                                                 float k = 0.15f) {
    const std::size_t N = s.node_count();
    std::vector<AnalogueMatch> out;
    if (N == 0 || !s.has(source)) return out;
    const si::NodeId A = s.find(source);

    // signatures: sorted (direction, other_concept) multisets, both directions
    std::vector<std::vector<std::pair<int, std::string>>> sig(N);
    s.for_each_lane([&](si::NodeId a, si::NodeId b, float) {
        if (b < N) sig[a].push_back({0, s.concept_of(b)});
        if (a < N) sig[b].push_back({1, s.concept_of(a)});
    });
    for (auto& v : sig) std::sort(v.begin(), v.end());

    const auto iso_of = [](const std::vector<std::pair<int, std::string>>& x,
                           const std::vector<std::pair<int, std::string>>& y) {
        if (x.empty() && y.empty()) return 0.0f;
        std::size_t i = 0, j = 0, inter = 0;
        while (i < x.size() && j < y.size()) {
            if (x[i] == y[j]) { ++inter; ++i; ++j; }
            else if (x[i] < y[j]) ++i;
            else ++j;
        }
        const std::size_t uni = x.size() + y.size() - inter;
        return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
    };

    // BFS hop distance (both directions), cap 24
    const auto hop_distance = [&](si::NodeId from, si::NodeId to) {
        if (from == to) return 0;
        std::vector<int> depth(N, -1);
        std::queue<si::NodeId> q;
        depth[from] = 0; q.push(from);
        while (!q.empty()) {
            si::NodeId u = q.front(); q.pop();
            if (depth[u] >= 24) continue;
            std::vector<std::pair<si::NodeId, float>> lanes;
            s.lanes_of(u, lanes);
            for (const auto& l : lanes) {
                if (depth[l.first] >= 0) continue;
                depth[l.first] = depth[u] + 1;
                if (l.first == to) return depth[l.first];
                q.push(l.first);
            }
        }
        return -1;
    };

    for (std::size_t b = 0; b < N; ++b) {
        if (static_cast<si::NodeId>(b) == A || sig[b].empty()) continue;
        const float iso = iso_of(sig[A], sig[b]);
        if (iso < min_iso) continue;
        const int d = hop_distance(A, static_cast<si::NodeId>(b));
        const float d_eff = (d < 0 ? 24.0f : static_cast<float>(d));
        out.push_back({s.concept_of(static_cast<si::NodeId>(b)), iso, d, iso * std::exp(k * d_eff)});
    }
    std::sort(out.begin(), out.end(), [](const AnalogueMatch& x, const AnalogueMatch& y) {
        return x.phi > y.phi;
    });
    if (static_cast<int>(out.size()) > top_k) out.resize(static_cast<std::size_t>(top_k));
    return out;
}

// ---------------------------------------------------------------------------
// -- PROMOTION (human-validated dream candidates -> premise-grade lanes) -----
// The ledger line must have been validated by a human (CLI enforces the flag);
// promoted lanes are ordinary Hebbian lanes (generation 0, symmetric, subject
// to lane decay) — they are experience once a human has vouched for them.
inline void apply_promotion(si::Substrate& s,
                            const std::vector<std::string>& driven_concepts,
                            const std::string& emergent_concept,
                            float support, float promote_gain) {
    for (const auto& d : driven_concepts) {
        if (!s.has(d) || !s.has(emergent_concept)) continue;
        const float w = std::min(1.5f, std::max(0.05f, support * promote_gain));
        s.bind(s.find(d), s.find(emergent_concept), w);
    }
}

} // namespace derive
} // namespace syfox
