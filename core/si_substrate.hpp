// ============================================================================
//  SyFox — si_substrate.hpp
//  ---------------------------------------------------------------------------
//  THE CORE. A faithful standalone port of the Synthetic-Intelligence
//  physics substrate mechanics (Mr-DS-ML-85/Synthetic-Intelligence, physics.hpp
//  lineage). Nothing here is a transformer, neural network, or pattern-matching
//  classifier. Decisions emerge from field dynamics only:
//
//    1. INJECTION   — state tokens deposit energy on concept nodes,
//                     weighted by acoustic mass (frequent concepts are
//                     "heavier" and carry less per-token energy).
//    2. SETTLE      — energy diffuses along Hebbian lanes with decay,
//                     iterated K_settle passes; an energy-gated source cap
//                     keeps the working set bounded. The SI salience
//                     integrator (physics.hpp cavity lineage:
//                     s = tanh(s·decay + gain·|motion|)) runs every pass and
//                     is observable on every node. Two flags enable the
//                     faithful SI selection modes: salience_gating ranks
//                     sources by salience instead of raw energy; miller_window
//                     samples the live cap from [source_cap-4, source_cap]
//                     per decision (TSDA "7 ± 2"; SI draws it from a seeded
//                     mt19937, SyFox derives it from the state hash so the
//                     same state still settles bit-for-bit). The field stops
//                     when the total energy delta drops below eps.
//    3. READOUT     — a resonance sweep reads the settled field at probe
//                     nodes (options / anchors / statements). Readout is a
//                     measurement of the field, not a classifier.
//    4. HEBBIAN     — labelled experience binds lanes between co-active
//                     state and option concepts (fire together, wire
//                     together), with weight decay and per-node lane caps.
//    5. HONEST      — if the settled field is below the silence floor the
//         SILENCE    substrate reports "no collapse" instead of guessing.
//
//  Zero dependencies. Deterministic. C++17.
// ============================================================================
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace si {

using NodeId = std::uint32_t;

// ---------------------------------------------------------------------------
// v3 Milestone 3 — the lane evidence ledger (audit surface, NOT physics).
// A lane's WEIGHT drives the field (settle/readout) exactly as before; the
// ledger records HOW a lane earned its weight, so every decision can show
// its evidence and every derived relation can be audited:
//   support_events  — lessons that strengthened this lane (Hebbian binds)
//   counter_events  — anti-Hebbian weakenings (disconfirming evidence)
//   first_seq/last_seq — the teach-event window that touched the lane
//                        (provenance; full per-lesson ids live in the
//                        model's conflicts/ledger files)
//   context         — the engine context tag at last touch (model/domain)
// The ledger never feeds back into settle/readout: it is bookkeeping over
// the physics, the way a lab notebook accompanies an experiment.
// ---------------------------------------------------------------------------
struct LaneEvidence {
    std::uint32_t support_events = 0;
    std::uint32_t counter_events = 0;
    std::uint64_t first_seq = 0;
    std::uint64_t last_seq  = 0;
    std::string   context;
};

// ---------------------------------------------------------------------------
struct SubstrateConfig {
    float decay          = 0.82f;  // per-pass energy retention (damping)
    float diffusion      = 0.45f;  // fraction of energy that flows along lanes
    int   k_settle       = 8;      // max settle passes
    float eps            = 1e-4f;  // phase-stability threshold (early stop)
    float source_cap     = 24.0f;  // max propagation sources per pass, top-K BY ENERGY
                                   // (SI-inspired bounded working set; fixed by default)
    // -- SI salience lineage (physics.hpp cavity integrator + TSDA SalienceMap) --
    float salience_gain   = 0.05f;  // SI physics.hpp salience_gain
    float salience_decay  = 0.995f; // SI physics.hpp salience_decay
    bool  salience_gating = false;  // true: rank sources by salience (motion history), not energy
    bool  miller_window   = false;  // true: live cap sampled from [source_cap-4, source_cap]
                                    // per decision (TSDA live_cap; deterministic per state)
    float silence_floor  = 0.05f;  // below this total energy => honest silence
    float inject_energy  = 1.0f;   // energy per fresh token
    float learn_eta      = 0.10f;  // Hebbian learning rate
    float lane_decay     = 0.995f; // per-lesson lane weight decay
    std::size_t lane_cap = 256;    // max lanes per node (weakest evicted)
    float hop_coupling   = 0.35f;  // 1-hop lane contribution at readout
};

inline float fnv1a_hash(const std::string& s) {  // deterministic token hash
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return static_cast<float>(h & 0xFFFFFFull);
}

// ---------------------------------------------------------------------------
// ConceptField — nodes, lanes, and the field dynamics over them.
// ---------------------------------------------------------------------------
class Substrate {
public:
    explicit Substrate(SubstrateConfig cfg = {}) : cfg_(cfg) {}

    // -- vocabulary ---------------------------------------------------------
    NodeId intern(const std::string& concept) {
        auto it = index_.find(concept);
        if (it != index_.end()) { nodes_[it->second].mass += 1.0f; return it->second; }
        NodeId id = static_cast<NodeId>(nodes_.size());
        index_.emplace(concept, id);
        // occurrence mass: a LINEAR count of intern() calls (not log-compressed).
        // The sub-linear behavior lives at the USE sites: injection deposits
        // energy/sqrt(mass) and Hebbian bind strength scales by 1/sqrt(mass),
        // so heavy concepts move less.
        nodes_.push_back(Node{concept, 1.0f, 0.0f, 0.0f});
        csr_dirty_ = true;                       // node count changed (M5 CSR mirror)
        return id;
    }

    bool has(const std::string& c) const { return index_.count(c) != 0; }
    NodeId find(const std::string& c) const { return index_.at(c); }
    std::size_t node_count() const { return nodes_.size(); }
    std::size_t lane_count() const {
        std::size_t n = 0;
        for (const auto& kv : out_) n += kv.second.size();
        return n;
    }
    std::size_t readout_neighbours(NodeId a) const {
        auto it = out_.find(a);
        return it == out_.end() ? 0 : it->second.size();
    }

    // -- Hebbian lane binding (fire together -> wire together) ---------------
    void bind(NodeId a, NodeId b, float w) {
        if (a == b || w <= 0.0f) return;
        add_lane(a, b, w);
        add_lane(b, a, w);
    }

    void add_lane(NodeId a, NodeId b, float w) {
        auto& lanes = out_[a];
        for (auto& l : lanes)
            if (l.first == b) { l.second = std::min(4.0f, std::max(0.0f, l.second + w)); csr_dirty_ = true; return; }
        if (w > 0.0f) {
            lanes.emplace_back(b, w);
            csr_dirty_ = true;
            if (lanes.size() > cfg_.lane_cap) {          // evict weakest lane
                auto weakest = std::min_element(lanes.begin(), lanes.end(),
                    [](const auto& x, const auto& y){ return x.second < y.second; });
                lanes.erase(weakest);
            }
        }
    }

    // subtract weight from a lane; a lane whose weight hits zero dissolves
    void scale_lane(NodeId a, NodeId b, float dw) {
        auto it = out_.find(a);
        if (it == out_.end()) return;
        auto& lanes = it->second;
        for (std::size_t i = 0; i < lanes.size(); ++i)
            if (lanes[i].first == b) {
                lanes[i].second = std::max(0.0f, lanes[i].second + dw);
                if (lanes[i].second <= 0.0f) lanes.erase(lanes.begin() + static_cast<std::ptrdiff_t>(i));
                csr_dirty_ = true;
                return;
            }
    }

    // -- INJECTION ------------------------------------------------------------
    // Fresh tokens carry inject_energy each, damped by acoustic mass so that
    // frequent concepts cannot flood the field.
    void inject(const std::vector<std::string>& tokens, float energy = 0.0f) {
        if (energy <= 0.0f) energy = cfg_.inject_energy;
        for (const auto& t : tokens) {
            if (!has(t)) continue;                        // unknown concepts stay dark
            NodeId id = find(t);
            Node& n = nodes_[id];
            float mass_damp = 1.0f / std::sqrt(n.mass);   // heavy => less per-token
            n.energy += energy * mass_damp;
            n.salience = 1.0f;   // TSDA: spike to ceiling on touch
            state_hash_ = state_hash_ * 1099511628211ull
                        + static_cast<std::uint64_t>(fnv1a_hash(t));  // decision fingerprint
        }
    }

    // salience of a concept (SI integrator state; diagnostic + test surface)
    float node_salience(const std::string& c) const {
        auto it = index_.find(c);
        return it == index_.end() ? 0.0f : nodes_[it->second].salience;
    }

    // the source cap actually used by the last settle() (sampled value when
    // miller_window is on; otherwise always config().source_cap)
    float last_source_cap() const { return last_cap_; }

    // -- SETTLE ---------------------------------------------------------------
    // One pass = dissipative diffusion along lanes + source gating.
    // Energy is CONSERVED per pass up to decay: each node retains
    // (1 - diffusion) of its (decayed) energy and flows `diffusion` out along
    // its lanes, split proportionally to lane weight. The source cap GATES
    // propagation: only the top-`cap` active nodes send energy each pass —
    // ranked by energy by default, or by salience (motion history) when
    // salience_gating is on. Gated nodes keep their energy (readout-visible)
    // but stay silent as sources — bounded working set, nothing is
    // annihilated. Salience itself integrates every pass exactly as the SI
    // cavity physics does: s = tanh(s·decay + gain·|ΔE|); a node at rest is
    // an exact fixed point (tanh(0) = 0), mirroring SI's sparsity guard.
    void settle() {
        std::vector<float> next(nodes_.size(), 0.0f);
        std::vector<std::size_t> active;

        // M5: build/refresh the CSR mirror if the fabric moved. The build is
        // order-preserving (per-source vector copied verbatim), so switching
        // between map iteration and CSR iteration cannot change a single float.
        if (csr_dirty_) build_csr();

        // M5: parallel-settle buffers. Allocated once per call, reused across
        // passes. Determinism contract: static schedule partitions sources
        // into contiguous ascending chunks; per-thread scatter buffers are
        // combined in ascending thread order; therefore every next[i]
        // accumulates contributions in ascending SOURCE order — exactly the
        // sequential order. Bit-identical, test-verified.
        int nthreads = 1;
#if defined(_OPENMP)
        if (parallel_settle_) nthreads = omp_get_max_threads();
        if (nthreads > static_cast<int>(nodes_.size())) nthreads = nodes_.size() > 0 ? static_cast<int>(nodes_.size()) : 1;
#endif
        std::vector<std::vector<float>> thread_bufs(
            static_cast<std::size_t>(nthreads > 1 ? nthreads : 0),
            std::vector<float>(nodes_.size(), 0.0f));  // each buffer is n floats, zeroed

        // Live source cap (TSDA live_cap lineage): when miller_window is on,
        // the working set is drawn from [source_cap-4, source_cap] each
        // decision instead of being pinned — narrow focus some decisions,
        // broad others. SI samples a seeded mt19937 per tick; SyFox derives
        // the draw from the decision's state hash, so reproducibility holds:
        // same state -> same cap -> same field, bit for bit.
        float cap = cfg_.source_cap;
        if (cfg_.miller_window) {
            std::uint32_t hi = static_cast<std::uint32_t>(std::max(1.0f, cfg_.source_cap));
            std::uint32_t lo = (hi >= 5) ? hi - 4 : 1;
            cap = static_cast<float>(lo + (state_hash_ % (hi - lo + 1)));
        }
        last_cap_ = cap;
        const std::size_t cap_n = static_cast<std::size_t>(cap);
        const std::size_t n = nodes_.size();
        for (int pass = 0; pass < cfg_.k_settle; ++pass) {
            float total_before = 0.0f;
            for (const auto& nd : nodes_) total_before += nd.energy;
            if (total_before <= 0.0f) return;             // nothing to settle

            std::fill(next.begin(), next.end(), 0.0f);
            for (auto& tb : thread_bufs) std::fill(tb.begin(), tb.end(), 0.0f);
            active.clear();
            for (std::size_t i = 0; i < n; ++i)
                if (nodes_[i].energy > 1e-7f) active.push_back(i);

            // source gating: at most `cap` sources per pass (ties break by
            // vocabulary order — deterministic)
            if (active.size() > cap_n) {
                std::nth_element(active.begin(),
                                 active.begin() + static_cast<std::ptrdiff_t>(cap_n),
                                 active.end(), [&](std::size_t a, std::size_t b){
                                     return cfg_.salience_gating
                                         ? nodes_[a].salience > nodes_[b].salience
                                         : nodes_[a].energy   > nodes_[b].energy; });
                active.resize(cap_n);
            }
            std::vector<char> is_source(n, 0);
            for (std::size_t i : active) is_source[i] = 1;

            // -- diffusion pass (CSR slices; optional deterministic OMP) ----
#if defined(_OPENMP)
#pragma omp parallel if (nthreads > 1) num_threads(nthreads)
#endif
            {
                float* mybuf = nullptr;
#if defined(_OPENMP)
                const int tid = omp_get_thread_num();
                if (nthreads > 1) mybuf = thread_bufs[static_cast<std::size_t>(tid)].data();
#pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    const Node& nd = nodes_[i];
                    if (nd.energy <= 0.0f) continue;
                    float* scatter = mybuf ? mybuf : next.data();
                    if (!is_source[i]) { scatter[i] += nd.energy * cfg_.decay; continue; }  // gated: decay only
                    const std::size_t b = csr_off_[i], e = csr_off_[i + 1];
                    float out_w = 0.0f;
                    for (std::size_t k = b; k < e; ++k) out_w += csr_w_[k];
                    float retained = (out_w > 0.0f) ? (1.0f - cfg_.diffusion) : 1.0f;
                    scatter[i] += nd.energy * cfg_.decay * retained;   // damped self-retention
                    if (out_w > 0.0f) {
                        const float flow = nd.energy * cfg_.decay * cfg_.diffusion;
                        for (std::size_t k = b; k < e; ++k)
                            scatter[csr_dst_[k]] += flow * (csr_w_[k] / out_w);
                    }
                }
            }
            if (nthreads > 1) {                       // fixed thread-order combine
                for (const auto& tb : thread_bufs)
                    for (std::size_t i = 0; i < n; ++i) next[i] += tb[i];
            }
            // SI cavity salience integrator: s = tanh(s·decay + gain·|ΔE|).
            // Motion proxy is how much energy moved through the node this
            // pass. Updated before the energies are committed, so the next
            // pass's source ranking sees it. Element-wise (no accumulation):
            // parallel-safe without touching the determinism contract.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nthreads > 1)
#endif
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                if (nodes_[i].energy <= 0.0f && next[i] <= 0.0f) continue;
                const float motion = std::fabs(next[i] - nodes_[i].energy);
                nodes_[i].salience = std::tanh(nodes_[i].salience * cfg_.salience_decay
                                             + cfg_.salience_gain * motion);
            }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nthreads > 1)
#endif
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
                nodes_[ii].energy = next[ii];

            float total_after = 0.0f;
            for (const auto& nd : nodes_) total_after += nd.energy;
            if (total_before - total_after < cfg_.eps) break;  // relaxed
        }
    }

    // -- READOUT (resonance sweep, two lanes) -----------------------------------
    // Both lanes are field measurements — no weights fitted, no classes exist.
    //
    //  SPECIFIC — "mean lane support": rewards precise coupling. A node bound
    //    diffusely to half the vocabulary must not read hotter than a node
    //    bound exactly to this state's concepts. Used for Choice/Score where
    //    the engine must pick THE best-matching option.
    //
    //  SUPPORT — "active lane mass": rewards breadth of corroboration. A
    //    statement (Noul) may be supported by many partial lanes; what matters
    //    is how much lane-connected energy the field offers it. Used for Noul.
    enum class ReadoutMode { Specific, Support };

    float readout(const std::vector<std::string>& probe,
                  ReadoutMode mode = ReadoutMode::Specific) const {
        float r = 0.0f;
        for (const auto& t : probe) {
            auto it = index_.find(t);
            if (it == index_.end()) continue;
            const Node& n = nodes_[it->second];
            r += n.energy;
            const auto lane_it = out_.find(it->second);
            if (lane_it == out_.end() || lane_it->second.empty()) continue;
            if (mode == ReadoutMode::Specific) {
                float wsum = 0.0f, flow = 0.0f;
                for (const auto& lane : lane_it->second) {
                    wsum += lane.second;
                    flow += lane.second * nodes_[lane.first].energy;
                }
                if (wsum > 0.0f) r += cfg_.hop_coupling * flow / wsum;
            } else {                                      // Support: active lane mass
                for (const auto& lane : lane_it->second)
                    r += cfg_.hop_coupling * lane.second * nodes_[lane.first].energy;
            }
        }
        return r;
    }

    float total_energy() const {
        float s = 0.0f;
        for (const auto& n : nodes_) s += n.energy;
        return s;
    }

    float node_energy(NodeId id) const {
        return id < nodes_.size() ? nodes_[id].energy : 0.0f;
    }

    // acoustic mass (occurrence count); derivation uses the same 1/sqrt(mass)
    // damping law the field applies at its use sites, so hub-mediated
    // composition is suppressed exactly like hub-mediated energy flow
    float node_mass(NodeId id) const {
        return id < nodes_.size() ? nodes_[id].mass : 1.0f;
    }

    bool silent() const { return total_energy() < cfg_.silence_floor; }

    // -- HONEST SILENCE reset ---------------------------------------------------
    void reset_field() {
        for (auto& n : nodes_) { n.energy = 0.0f; n.salience = 0.0f; }
        state_hash_ = 0;
    }

    // -- ANTI-HEBBIAN WEAKENING --------------------------------------------------
    // Counter-evidence: unbind lanes between active concepts and an outcome
    // (used for Noul "false" lessons so disconfirming routes fade).
    void weaken(const std::vector<std::string>& active_state,
                const std::vector<std::string>& outcome, float eta_scale = 1.0f) {
        for (const auto& ta : active_state) {
            if (!has(ta)) continue;
            for (const auto& tb : outcome) {
                if (!has(tb)) continue;
                NodeId a = find(ta), b = find(tb);
                scale_lane(a, b, -cfg_.learn_eta * eta_scale);
                scale_lane(b, a, -cfg_.learn_eta * eta_scale);
            }
        }
    }

    // -- HEBBIAN LESSON ----------------------------------------------------------
    // Strengthen lanes between currently-active state concepts and the
    // concepts of a rewarded outcome; decay every lane slightly so unused
    // routes fade. This is how SyFox learns: substrate rewiring, not fitting.
    void hebbian_lesson(const std::vector<std::string>& active_state,
                        const std::vector<std::string>& outcome,
                        float eta_scale = 1.0f) {
        std::vector<NodeId> s_ids, o_ids;
        for (const auto& t : active_state)
            if (has(t)) s_ids.push_back(find(t));
        for (const auto& t : outcome) {
            intern(t);                                    // outcome concepts join the field
            o_ids.push_back(find(t));
        }
        for (NodeId a : s_ids)
            for (NodeId b : o_ids) {
                float wa = 1.0f / std::sqrt(nodes_[a].mass);
                float wb = 1.0f / std::sqrt(nodes_[b].mass);
                bind(a, b, cfg_.learn_eta * eta_scale * (wa + wb));
            }
        enforce_lane_decay();
    }

    // -- LANE EVIDENCE LEDGER (v3 Milestone 3; bookkeeping, not physics) ------
    // record_support / record_counter: called by the ENGINE around its teach
    // events. seq = the engine's monotonic teach counter; context = engine tag.
    void record_support(NodeId a, NodeId b, std::uint64_t seq, const std::string& context) {
        LaneEvidence& e = lane_evidence_[lane_key(a, b)];
        ++e.support_events;
        e.last_seq = seq;
        if (e.first_seq == 0) e.first_seq = seq;
        e.context = context;
    }
    void record_counter(NodeId a, NodeId b, std::uint64_t seq, const std::string& context) {
        LaneEvidence& e = lane_evidence_[lane_key(a, b)];
        ++e.counter_events;
        e.last_seq = seq;
        if (e.first_seq == 0) e.first_seq = seq;
        e.context = context;
    }
    bool evidence_of(NodeId a, NodeId b, LaneEvidence& out) const {
        auto it = lane_evidence_.find(lane_key(a, b));
        if (it == lane_evidence_.end()) return false;
        out = it->second;
        return true;
    }
    std::size_t evidence_count() const { return lane_evidence_.size(); }

    // -- persistence (binary, deterministic) -------------------------------------
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        std::uint32_t n = static_cast<std::uint32_t>(nodes_.size());
        f.write(reinterpret_cast<const char*>(&n), 4);
        for (const auto& node : nodes_) {
            std::uint32_t len = static_cast<std::uint32_t>(node.concept.size());
            f.write(reinterpret_cast<const char*>(&len), 4);
            f.write(node.concept.data(), len);
            f.write(reinterpret_cast<const char*>(&node.mass), 4);
        }
        std::uint32_t lanes = 0;
        for (const auto& kv : out_) lanes += static_cast<std::uint32_t>(kv.second.size());
        f.write(reinterpret_cast<const char*>(&lanes), 4);
        for (const auto& kv : out_)
            for (const auto& l : kv.second) {
                f.write(reinterpret_cast<const char*>(&kv.first), 4);
                f.write(reinterpret_cast<const char*>(&l.first), 4);
                f.write(reinterpret_cast<const char*>(&l.second), 4);
            }
        // v2 provenance tail — old (v1) files simply end here; load() detects EOF.
        std::uint32_t dc = static_cast<std::uint32_t>(derived_gen_.size());
        f.write(reinterpret_cast<const char*>(&dc), 4);
        for (const auto& kv : derived_gen_) {
            std::uint32_t a = static_cast<std::uint32_t>(kv.first >> 32);
            std::uint32_t b = static_cast<std::uint32_t>(kv.first & 0xffffffffull);
            f.write(reinterpret_cast<const char*>(&a), 4);
            f.write(reinterpret_cast<const char*>(&b), 4);
            f.write(reinterpret_cast<const char*>(&kv.second), 4);
        }
        // v3 evidence tail — v2 files end here; load() detects EOF.
        std::uint32_t ec = static_cast<std::uint32_t>(lane_evidence_.size());
        f.write(reinterpret_cast<const char*>(&ec), 4);
        for (const auto& kv : lane_evidence_) {
            std::uint32_t a = static_cast<std::uint32_t>(kv.first >> 32);
            std::uint32_t b = static_cast<std::uint32_t>(kv.first & 0xffffffffull);
            f.write(reinterpret_cast<const char*>(&a), 4);
            f.write(reinterpret_cast<const char*>(&b), 4);
            f.write(reinterpret_cast<const char*>(&kv.second.support_events), 4);
            f.write(reinterpret_cast<const char*>(&kv.second.counter_events), 4);
            f.write(reinterpret_cast<const char*>(&kv.second.first_seq), 8);
            f.write(reinterpret_cast<const char*>(&kv.second.last_seq), 8);
            std::uint32_t clen = static_cast<std::uint32_t>(kv.second.context.size());
            f.write(reinterpret_cast<const char*>(&clen), 4);
            if (clen) f.write(kv.second.context.data(), clen);
        }
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return;
        nodes_.clear(); index_.clear(); out_.clear();
        csr_dirty_ = true;
        std::uint32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4);
        nodes_.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t len = 0; f.read(reinterpret_cast<char*>(&len), 4);
            std::string c(len, '\0'); f.read(c.data(), len);
            float mass = 0; f.read(reinterpret_cast<char*>(&mass), 4);
            index_.emplace(c, static_cast<NodeId>(nodes_.size()));
            nodes_.push_back(Node{c, mass, 0.0f, 0.0f});
        }
        std::uint32_t lanes = 0; f.read(reinterpret_cast<char*>(&lanes), 4);
        for (std::uint32_t i = 0; i < lanes; ++i) {
            NodeId a = 0, b = 0; float w = 0;
            f.read(reinterpret_cast<char*>(&a), 4);
            f.read(reinterpret_cast<char*>(&b), 4);
            f.read(reinterpret_cast<char*>(&w), 4);
            out_[a].emplace_back(b, w);
        }
        // v2 provenance tail (optional): v1 files hit EOF here and stay clean.
        std::uint32_t dc = 0;
        if (f.read(reinterpret_cast<char*>(&dc), 4)) {
            for (std::uint32_t i = 0; i < dc; ++i) {
                std::uint32_t a = 0, b = 0, g = 0;
                if (!f.read(reinterpret_cast<char*>(&a), 4)) break;
                if (!f.read(reinterpret_cast<char*>(&b), 4)) break;
                if (!f.read(reinterpret_cast<char*>(&g), 4)) break;
                derived_gen_.emplace(lane_key(static_cast<NodeId>(a), static_cast<NodeId>(b)), g);
            }
        }
        // v3 evidence tail (optional): v2 files hit EOF here and stay clean.
        std::uint32_t ec = 0;
        if (f.read(reinterpret_cast<char*>(&ec), 4)) {
            for (std::uint32_t i = 0; i < ec; ++i) {
                std::uint32_t a = 0, b = 0, sup = 0, cnt = 0, clen = 0;
                std::uint64_t fs = 0, ls = 0;
                if (!f.read(reinterpret_cast<char*>(&a), 4)) break;
                if (!f.read(reinterpret_cast<char*>(&b), 4)) break;
                if (!f.read(reinterpret_cast<char*>(&sup), 4)) break;
                if (!f.read(reinterpret_cast<char*>(&cnt), 4)) break;
                if (!f.read(reinterpret_cast<char*>(&fs), 8)) break;
                if (!f.read(reinterpret_cast<char*>(&ls), 8)) break;
                if (!f.read(reinterpret_cast<char*>(&clen), 4)) break;
                std::string ctx(clen, '\0');
                if (clen && !f.read(ctx.data(), clen)) break;
                LaneEvidence e;
                e.support_events = sup; e.counter_events = cnt;
                e.first_seq = fs; e.last_seq = ls; e.context = ctx;
                lane_evidence_[lane_key(static_cast<NodeId>(a), static_cast<NodeId>(b))] = e;
            }
        }
    }

    const SubstrateConfig& config() const { return cfg_; }

    // -- LANE PROVENANCE + SURGERY (derivation-layer surface, see derive.hpp) --
    // Derived lanes are laid by rule composition over existing fabric, not by
    // direct experience. They carry a generation counter: observed (Hebbian /
    // human-promoted) lanes are premise-grade (generation 0) and never lose
    // that grade; each composition step adds one generation. The generation
    // cap and use-time re-verification live in derive.hpp.
    void bind_derived(NodeId a, NodeId b, float w, std::uint32_t gen) {
        if (a == b || w <= 0.0f) return;
        set_lane_derived(a, b, w, gen);
        set_lane_derived(b, a, w, gen);
    }

    // Verifier-side weight set: assign exactly (down too), dissolve at zero.
    // Only meaningful for derived lanes; observed lanes are never touched by
    // callers (composition uses bind_derived, which is set-never-weaken).
    void set_derived_weight(NodeId a, NodeId b, float w) {
        auto it = out_.find(a);
        if (it == out_.end()) return;
        for (auto& l : it->second) {
            if (l.first == b) {
                if (w <= 0.0f) { dissolve_lane(a, b); return; }
                l.second = std::min(4.0f, w);
                csr_dirty_ = true;
                // keep symmetry: the mirror lane tracks the same evidence
                auto mit = out_.find(b);
                if (mit != out_.end())
                    for (auto& ml : mit->second)
                        if (ml.first == a) { ml.second = l.second; break; }
                return;
            }
        }
    }

    // Remove a lane (both directions) and its provenance.
    void dissolve_lane(NodeId a, NodeId b) {
        auto erase_one = [](std::vector<std::pair<NodeId, float>>& lanes, NodeId t) {
            for (std::size_t i = 0; i < lanes.size(); ++i)
                if (lanes[i].first == t) { lanes.erase(lanes.begin() + static_cast<std::ptrdiff_t>(i)); return; }
        };
        auto it = out_.find(a);
        if (it != out_.end()) erase_one(it->second, b);
        auto mit = out_.find(b);
        if (mit != out_.end()) erase_one(mit->second, a);
        derived_gen_.erase(lane_key(a, b));
        derived_gen_.erase(lane_key(b, a));
        csr_dirty_ = true;
    }

    std::uint32_t generation_of(NodeId a, NodeId b) const {
        auto it = derived_gen_.find(lane_key(a, b));
        return it == derived_gen_.end() ? 0u : it->second;
    }

    float lane_weight(NodeId a, NodeId b) const {
        auto it = out_.find(a);
        if (it == out_.end()) return 0.0f;
        for (const auto& l : it->second) if (l.first == b) return l.second;
        return 0.0f;
    }

    const std::string& concept_of(NodeId id) const { return nodes_[id].concept; }

    // Visitor over every lane (a -> b, weight). Directional view of the
    // symmetric fabric: each direction is visited once, like the save format.
    template <class F>
    void for_each_lane(F&& f) const {
        for (const auto& kv : out_)
            for (const auto& l : kv.second) f(kv.first, l.first, l.second);
    }

    // -- FABRIC SNAPSHOT (transactional derivation surface, see gate.hpp) ------
    // Exact, order-preserving capture of the lane fabric. settle() sums over
    // each node's lane vector in insertion order, so a faithful restore must
    // reproduce that order bit-for-bit: a (a,b,w) set is NOT enough, the
    // per-node vector sequences are the state. Used by the no-regression gate
    // to guarantee "revert" means "decide() output is byte-identical again".
    struct FabricSnapshot {
        std::vector<std::vector<std::pair<NodeId, float>>> adj;   // per node, exact order
        std::unordered_map<std::uint64_t, std::uint32_t> gen;     // lane_key -> generation
        std::size_t lanes = 0;
    };

    FabricSnapshot snapshot_fabric() const {
        FabricSnapshot snap;
        snap.adj.resize(nodes_.size());
        for (const auto& kv : out_) {
            if (kv.first >= snap.adj.size()) continue;
            snap.adj[kv.first] = kv.second;
            snap.lanes += kv.second.size();
        }
        snap.gen = derived_gen_;
        return snap;
    }

    void restore_fabric(const FabricSnapshot& snap) {
        out_.clear();
        for (std::size_t a = 0; a < snap.adj.size(); ++a)
            if (!snap.adj[a].empty()) out_[static_cast<NodeId>(a)] = snap.adj[a];
        derived_gen_ = snap.gen;
        csr_dirty_ = true;
    }

    void lanes_of(NodeId a, std::vector<std::pair<NodeId, float>>& out_lanes) const {
        auto it = out_.find(a);
        if (it == out_.end()) return;
        out_lanes = it->second;
    }

    // node-energy injection by id (dreamer probing; no decision fingerprint)
    void inject_energy(NodeId id, float energy) {
        if (id >= nodes_.size() || energy <= 0.0f) return;
        Node& n = nodes_[id];
        n.energy += energy / std::sqrt(n.mass);
        n.salience = 1.0f;   // TSDA: spike to ceiling on touch
    }

    // Runtime selection-mode overrides (CLI surface for the SI-faithful modes).
    // Physics constants stay fixed; only the two gating/window modes are
    // settable, and neither is persisted into substrate.bin — the saved model
    // is mode-neutral, so every binary can replay it under any mode.
    void set_source_modes(bool salience_gating_mode, bool miller_window_mode) {
        cfg_.salience_gating = salience_gating_mode;
        cfg_.miller_window   = miller_window_mode;
    }

private:
    struct Node { std::string concept; float mass; float energy; float salience; };

    static std::uint64_t lane_key(NodeId a, NodeId b) {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b);
    }

    // Composition-time lane write: raise-only (never weakens existing fabric),
    // and provenance is recorded once — a lane already premise-grade (laid by
    // experience) keeps generation 0; experience dominates derivation.
    void set_lane_derived(NodeId a, NodeId b, float w, std::uint32_t gen) {
        auto& lanes = out_[a];
        for (auto& l : lanes) {
            if (l.first == b) {
                if (w > l.second) { l.second = std::min(4.0f, w); csr_dirty_ = true; }
                derived_gen_.emplace(lane_key(a, b), gen);  // keep first (lowest) grade
                return;
            }
        }
        lanes.emplace_back(b, w);
        csr_dirty_ = true;
        derived_gen_.emplace(lane_key(a, b), gen);
        if (lanes.size() > cfg_.lane_cap) {              // evict weakest lane
            auto weakest = std::min_element(lanes.begin(), lanes.end(),
                [](const auto& x, const auto& y){ return x.second < y.second; });
            derived_gen_.erase(lane_key(a, weakest->first));
            lanes.erase(weakest);
        }
    }

    void enforce_lane_decay() {
        for (auto& kv : out_)
            for (auto& l : kv.second) l.second *= cfg_.lane_decay;
        csr_dirty_ = true;
    }

    SubstrateConfig cfg_;
    std::unordered_map<std::string, NodeId> index_;
    std::vector<Node> nodes_;
    std::unordered_map<NodeId, std::vector<std::pair<NodeId, float>>> out_;
    // Derivation provenance: lane key -> generation (sparse; observed lanes absent = 0)
    std::unordered_map<std::uint64_t, std::uint32_t> derived_gen_;
    // v3 Milestone 3: audit ledger (support/counter events, provenance window)
    std::unordered_map<std::uint64_t, LaneEvidence> lane_evidence_;
    std::uint64_t state_hash_ = 0;   // decision fingerprint for Miller sampling
    float last_cap_ = 0.0f;          // cap used by the last settle()

    // -- M5: CSR MIRROR of the out-lane fabric (settle hot path) -------------
    // Flattened contiguous buffers (offsets + targets + weights) built lazily
    // from out_ and invalidated by every lane/node mutation above. The build
    // copies each source's lane vector VERBATIM, so per-source iteration order
    // is exactly the map order settle() used before: every float sum keeps its
    // exact operand order and results stay bit-identical. This is the
    // cache-friendly layout the GPU design generalizes (ARCHITECTURE.md S15).
    // mutable: a lazy cache — logically const rebuild inside const settle paths.
    mutable std::vector<std::size_t> csr_off_;   // nodes_.size()+1
    mutable std::vector<NodeId>      csr_dst_;
    mutable std::vector<float>       csr_w_;
    mutable bool csr_dirty_ = true;
    bool parallel_settle_ = true;        // OMP builds honor this; sequential builds ignore it

    void build_csr() const {
        const std::size_t n = nodes_.size();
        csr_off_.assign(n + 1, 0);
        for (std::size_t i = 0; i < n; ++i) {
            auto it = out_.find(static_cast<NodeId>(i));
            csr_off_[i + 1] = csr_off_[i] + (it == out_.end() ? 0 : it->second.size());
        }
        csr_dst_.resize(csr_off_[n]);
        csr_w_.resize(csr_off_[n]);
        for (std::size_t i = 0; i < n; ++i) {
            auto it = out_.find(static_cast<NodeId>(i));
            if (it == out_.end()) continue;
            std::size_t k = csr_off_[i];
            for (const auto& lane : it->second) {
                csr_dst_[k] = lane.first;
                csr_w_[k]   = lane.second;
                ++k;
            }
        }
        csr_dirty_ = false;
    }

public:
    // parallel settle toggle (M5): on OMP builds, >1 threads use per-thread
    // scatter buffers combined in fixed thread order — bit-identical to the
    // sequential path (test-verified). Off builds always run sequential.
    void set_parallel_settle(bool on) { parallel_settle_ = on; }
    bool parallel_settle_enabled() const {
#if defined(_OPENMP)
        return parallel_settle_;
#else
        return false;
#endif
    }
    // fabric density for the GPU-gate report (M5): edges / possible directed
    // edges, plus mean out-degree. Pure measurements, no claims.
    double fabric_density() const {
        const double n = static_cast<double>(nodes_.size());
        if (n < 2.0) return 0.0;
        return static_cast<double>(lane_count()) / (n * (n - 1.0));
    }
    double mean_out_degree() const {
        const double n = static_cast<double>(nodes_.size());
        return n > 0.0 ? static_cast<double>(lane_count()) / n : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Tokenizer — plain lexical split (I/O concern, not pattern matching):
// lowercase, split on non-alphanumeric, keep tokens of length 2..24.
// ---------------------------------------------------------------------------
// Light deterministic lexical folding (string normalization, not ML):
//   plurals:    "refunds"->"refund", "invoices"->"invoice"  (len>=4, not "ss")
//   past/verb:  "charged"->"charg", "charge"->"charg"      (len>=5)
//   gerund:     "billing"->"bill", "running"->"runn"       (len>=6)
// Consistency matters, not linguistic truth: the same fold applies to state,
// options, and instructions, so matched forms always meet.
inline std::string fold(const std::string& w) {
    std::string s = w;
    if (s.size() >= 4 && s.compare(s.size() - 2, 2, "ss") != 0 && s.back() == 's') s.pop_back();
    if (s.size() >= 6 && s.compare(s.size() - 3, 3, "ing") == 0) { s.erase(s.size() - 3); return s; }
    if (s.size() >= 5 && s.compare(s.size() - 2, 2, "ed") == 0) s.erase(s.size() - 2);
    if (s.size() >= 5 && s.back() == 'e') s.pop_back();
    return s;
}

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto emit = [&]() {
        if (cur.size() >= 2 && cur.size() <= 24) out.push_back(fold(cur));
        cur.clear();
    };
    for (char raw : text) {
        unsigned char c = static_cast<unsigned char>(raw);
        if (std::isalnum(c)) cur.push_back(static_cast<char>(std::tolower(c)));
        else if (!cur.empty()) emit();
    }
    if (!cur.empty()) emit();
    return out;
}

} // namespace si
