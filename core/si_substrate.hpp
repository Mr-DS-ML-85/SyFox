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
#include <unordered_set>
#include <vector>

namespace si {

using NodeId = std::uint32_t;

// ---------------------------------------------------------------------------
// v3.2 SEMANTIC FIELD — the substrate's second field, deterministic, no ML.
// Every concept node carries a 64-dim semantic vector built in two
// deterministic steps (see build_semantics):
//   1. LEXICAL LAYER   — signed character-trigram hashing of the concept
//      string into 64 buckets ("card_arrival" and "card_delivery" share
//      trigrams, so their base vectors align; unrelated words stay
//      near-orthogonal). A fixed function of the string: no training.
//   2. FABRIC GROUNDING — two smoothing passes over the Hebbian lanes
//      (v_i <- normalize(v_i + beta * lane-weighted mean of v_j)). The
//      fabric's own co-occurrence structure shapes the vectors — a
//      deterministic diffusion of meaning along the lanes, not a fit.
// From the vectors, per node:
//   omega_semantic  — scalar projection onto one fixed direction (Stage 1
//                     "concept frequency encoding"); close frequencies
//                     resonate, like coupled oscillators detuned.
//   resonance edges — top-k cosine neighbours (cos >= sem_theta), weight
//                     = cos * frequency-match (Stage 4 "semantic field":
//                     during settle, sources leak a small fraction of
//                     their retained energy to semantically similar nodes
//                     — energy is CONSERVED, the flow is one more
//                     diffusion channel beside the Hebbian lanes).
//   semantic type   — cos >= 0.82 reads as synonym-grade kin, >= 0.66 as
//                     related (Option 2 typed lanes, derived from the
//                     vectors, not from a hand-written table).
// CONTEXT-SENSITIVE LANES (Stage 2): a lane may carry a context signature
// (required/forbidden context words, learned from the co-occurrence
// statistics of the lessons that laid it). At settle, the lane's flow is
// damped when the current decision's injected tokens do not match the
// signature: all required present -> full flow, missing -> ctx_missing,
// any forbidden present -> ctx_forbidden. The same "card" can then route
// toward card_arrival when arrive/receive are in the state and toward
// card_delivery_estimate when estimate/how-long is — contextual
// disambiguation as dampened coupling, not pattern matching.
// Persistence: a v4 tail in substrate.bin. Models saved before v3.2 have no
// tail: has_semantics() is false and every semantic code path is inert, so
// old fabrics replay bit-for-bit. Runtime kill switch: set_semantics(false).
// ---------------------------------------------------------------------------
inline constexpr std::size_t SEM_DIMS = 64;

// v3.4 multi-hop readout walk: max nodes expanded per hop level (smallest
// NodeIds first, deterministic). Bounds the walk at O(levels x 64).
inline constexpr std::size_t kHopWidthCap = 64;

// v3.2 context signature attached to a lane (read-side of Stage 2).
struct LaneCtx {
    std::vector<NodeId> required;    // all of these should be in the state
    std::vector<NodeId> forbidden;   // any of these suppresses the lane
};

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
    // -- v3.2 semantic field constants ---------------------------------------
    float sem_coupling  = 0.12f;  // resonance outflow fraction of retained energy
    float sem_theta     = 0.50f;  // min cosine for a resonance edge
    float sem_hop       = 0.10f;  // semantic-neighbour contribution at readout
    float sem_beta      = 0.50f;  // fabric-grounding strength (vector smoothing)
    int   sem_neighbors = 6;      // resonance edges per node
    float ctx_missing   = 0.60f;  // lane flow factor when required context is absent
    float ctx_forbidden = 0.20f;  // lane flow factor when forbidden context is present
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
        // v3.2: the semantic field is a property of the WHOLE fabric — a new
        // node changes every top-k neighbourhood, so the stored field is now
        // stale. Invalidate rather than index out of bounds: the next
        // save_model() rebuilds it over the grown fabric (and in-memory
        // engines run lane-physics only until then — honest degradation).
        if (!semvecs_.empty()) {
            semvecs_.clear(); sem_edges_.clear(); sem_off_.clear();
            sem_edge_count_ = 0;
        }
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

    // -- v3.6.0 DOCUMENT-FREQUENCY TABLE (CMD §2.2 rarity grounding; MIMS §6
    // fan-out derivation) -----------------------------------------------
    // The engine calls begin_df_lesson() once per lesson and note_df() once
    // per (lesson, node), so df(n) = number of lessons containing n. The
    // rarity factor log1p(L/(1+df))/log1p(L) ∈ (0,1] then scales the
    // STATE-side bind weights in hebbian_lesson: a token that fires in every
    // lesson distinguishes nothing and binds weakly; a rare token binds
    // strongly. This is CMD's "ubiquitous participants are suppressed by
    // construction, not by a rule" — no stopword list, one counted formula.
    // Active only when the table is non-empty (fresh learns note df; fabrics
    // loaded from a pre-v3.6 file have no 'IDF5' tail and every factor is
    // 1.0, so the replay contract holds bit for bit).
    void begin_df_lesson() { ++df_lessons_; }
    void note_df(NodeId id) { ++df_[id]; }
    std::uint32_t df_of(NodeId id) const {
        auto it = df_.find(id);
        return it == df_.end() ? 0u : it->second;
    }
    std::uint64_t df_lesson_count() const { return df_lessons_; }
    std::size_t df_table_size() const { return df_.size(); }
    bool idf_active() const { return !df_.empty() && df_lessons_ > 0; }
    float idf_factor(NodeId id) const {
        if (!idf_active()) return 1.0f;
        const double L = static_cast<double>(df_lessons_);
        const double d = static_cast<double>(df_of(id));
        return static_cast<float>(std::log1p(L / (1.0 + d)) / std::log1p(L));
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
                drop_lane_ctx(a, weakest->first);        // v3.2: signature dies with its lane
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
                if (lanes[i].second <= 0.0f) {
                    drop_lane_ctx(a, b);                 // v3.2: signature dies with its lane
                    lanes.erase(lanes.begin() + static_cast<std::ptrdiff_t>(i));
                }
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
            present_.insert(id);   // v3.2: context-lane visibility (until next reset)
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
                    if (sem_active() && !lane_ctx_.empty()) {
                        // v3.2 Stage 2: context-sensitive lanes — each lane's
                        // EFFECTIVE weight is w * factor(f). The node's total
                        // outflow scales by (sum w*f / sum w): damped lanes
                        // carry less, the energy stays home. All factors 1.0
                        // reproduces the plain path bit-for-bit (same float
                        // sum order, 1.0f multiplies are exact).
                        float raw_w = 0.0f, eff_w = 0.0f;
                        for (std::size_t k = b; k < e; ++k) {
                            const float f = lane_flow_factor(i, csr_dst_[k]);
                            raw_w += csr_w_[k];
                            eff_w += csr_w_[k] * f;
                        }
                        if (raw_w > 0.0f && eff_w > 0.0f) {
                            const float out_frac = cfg_.diffusion * (eff_w / raw_w);
                            scatter[i] += nd.energy * cfg_.decay * (1.0f - out_frac);
                            const float flow = nd.energy * cfg_.decay * out_frac;
                            for (std::size_t k = b; k < e; ++k) {
                                const float f = lane_flow_factor(i, csr_dst_[k]);
                                if (f > 0.0f)
                                    scatter[csr_dst_[k]] += flow * (csr_w_[k] * f / eff_w);
                            }
                        } else {
                            scatter[i] += nd.energy * cfg_.decay;   // nothing flows
                        }
                        continue;
                    }
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
            // -- v3.2 SEMANTIC RESONANCE (Stage 4) ----------------------------
            // Sequential by design (determinism contract): sources — and only
            // sources, so the working-set cap stays meaningful — leak a small
            // fraction (sem_coupling) of their POST-diffusion energy along
            // their resonance edges, distributed proportionally to
            // cos * frequency-match. Energy is conserved exactly (the pool is
            // subtracted from the source, then distributed); decay applies
            // next pass as always. A node with no resonance edges is an
            // exact fixed point of this pass.
            if (sem_active()) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!is_source[i]) continue;
                    const std::size_t sb = sem_off_[i], se = sem_off_[i + 1];
                    if (sb == se) continue;
                    float wsum = 0.0f;
                    for (std::size_t k = sb; k < se; ++k)
                        wsum += sem_edges_[k].second * freq_match(i, sem_edges_[k].first);
                    if (wsum <= 0.0f) continue;
                    const float pool = next[i] * cfg_.sem_coupling;
                    next[i] -= pool;
                    for (std::size_t k = sb; k < se; ++k) {
                        const float fm = freq_match(i, sem_edges_[k].first);
                        next[sem_edges_[k].first] += pool * (sem_edges_[k].second * fm / wsum);
                    }
                }
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
                if (hop_depth_ <= 1) {
                    // legacy single-hop path — bit-identical to pre-v3.4
                    float wsum = 0.0f, flow = 0.0f;
                    for (const auto& lane : lane_it->second) {
                        wsum += lane.second;
                        flow += lane.second * nodes_[lane.first].energy;
                    }
                    if (wsum > 0.0f) r += cfg_.hop_coupling * flow / wsum;
                } else {
                    // v3.4 multi-hop walk: BFS over lanes from this anchor.
                    // Level h contributes hop_coupling/sqrt(h+1) of the
                    // weight-averaged energy there. Deterministic: candidate
                    // nodes collect in ascending NodeId order (std::map),
                    // width-capped, never revisited.
                    std::vector<char> visited(nodes_.size(), 0);
                    visited[it->second] = 1;
                    std::vector<NodeId> frontier{it->second};
                    for (int hop = 1; hop <= hop_depth_ && !frontier.empty(); ++hop) {
                        std::map<NodeId, float> next_w;
                        for (NodeId u : frontier) {
                            const auto uit = out_.find(u);
                            if (uit == out_.end()) continue;
                            for (const auto& lane : uit->second)
                                if (!visited[lane.first])
                                    next_w[lane.first] += lane.second;
                        }
                        if (next_w.empty()) break;
                        std::vector<NodeId> level;
                        float wsum = 0.0f, flow = 0.0f;
                        for (const auto& kv : next_w) {
                            if (level.size() >= kHopWidthCap) break;
                            level.push_back(kv.first);
                            wsum += kv.second;
                            flow += kv.second * nodes_[kv.first].energy;
                        }
                        for (NodeId v : level) visited[v] = 1;
                        const float damp = cfg_.hop_coupling
                            / std::sqrt(static_cast<float>(hop + 1));
                        if (wsum > 0.0f) r += damp * flow / wsum;
                        frontier.swap(level);
                    }
                }
            } else {                                      // Support: active lane mass
                for (const auto& lane : lane_it->second)
                    r += cfg_.hop_coupling * lane.second * nodes_[lane.first].energy;
            }
            // v3.2 Stage 4: semantic-neighbour term — energy resting on
            // semantically similar nodes counts for this probe even where no
            // lane connects them. Weighted by the edge cosine; zero when the
            // fabric carries no semantic field (pre-v3.2 models, or --no-semantics).
            if (sem_active()) {
                const std::size_t sb = sem_off_[it->second], se = sem_off_[it->second + 1];
                for (std::size_t k = sb; k < se; ++k)
                    r += cfg_.sem_hop * sem_edges_[k].second * nodes_[sem_edges_[k].first].energy;
            }
        }
        return r;
    }

    float total_energy() const {
        float s = 0.0f;
        for (const auto& n : nodes_) s += n.energy;
        return s;
    }

    // v3.4 — decision-layer field composition (question-context gate): blend
    // the settled state field with an ALREADY-SETTLED context field,
    // e = (1-alpha)*e + alpha*ctx. Settle() physics is untouched — this
    // composes two settled fields at the decision layer, so the question can
    // carry part of the readout field without rewriting the dynamics.
    void blend_field(const std::vector<float>& ctx, float alpha) {
        const std::size_t n = std::min(nodes_.size(), ctx.size());
        for (std::size_t i = 0; i < n; ++i)
            nodes_[i].energy = (1.0f - alpha) * nodes_[i].energy + alpha * ctx[i];
    }

    // v3.6.0 — BED §8 perturbation-contrast support: snapshot the settled
    // field (energies only; salience/present_ belong to the composition, not
    // the measurement) and restore it after a diagnostic re-settle. The
    // perturbation check settles a structure-broken copy of the state in a
    // SCRATCH pass, measures its margins, then restores the real field so the
    // readout that follows measures exactly what the un-checked decide()
    // would have measured. Deterministic; no physics touched.
    std::vector<float> snapshot_field() const {
        std::vector<float> e(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) e[i] = nodes_[i].energy;
        return e;
    }
    void restore_field(const std::vector<float>& e) {
        const std::size_t n = std::min(nodes_.size(), e.size());
        for (std::size_t i = 0; i < n; ++i) nodes_[i].energy = e[i];
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
        present_.clear();
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
                // v3.6.0 (CMD §2.2 / MIMS §6): rarity weighting on the STATE
                // side only. Outcome anchors keep full weight — their df is
                // per-class, and a class label seen often is not a stopword.
                // idf_factor is 1.0 whenever the df table is absent (pre-v3.6
                // fabrics), so existing lanes are reproduced exactly.
                bind(a, b, cfg_.learn_eta * eta_scale * (wa + wb) * idf_factor(a));
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

    // =====================================================================
    // v3.2 SEMANTIC FIELD API (deterministic; see the header comment block)
    // =====================================================================

    // Stage 2 learn-side accumulation: ctx_word co-occurred with lane (a,b)
    // in the lesson that laid it. Counts live in a scratch table; only
    // finalize_contexts() (called at save) promotes surviving counts into
    // the lane's signature. Bookkeeping — nothing here touches the field.
    void add_ctx_support(NodeId a, NodeId b, NodeId ctx_word) {
        if (a == b || ctx_word == a || ctx_word == b) return;
        lane_ctx_acc_[lane_key(a, b)][ctx_word] += 1.0f;
    }

    // Promote accumulated co-occurrence counts into per-lane signatures.
    // Deterministic pruning over node-id-ascending iteration:
    //   required  — top-4 context words with count >= 2 for this lane
    //   forbidden — up to 2 words that are STRONG required context of the
    //               same source node's lanes into a DIFFERENT target family
    //               (count >= 3 there, absent-or-weak here). This is the
    //               cross-class diff: the words that discriminate WHERE
    //               this source node's energy should go.
    // Idempotent: the scratch table is consumed; calling twice is a no-op.
    void finalize_contexts() {
        if (lane_ctx_acc_.empty()) return;
        // gather per-source: which targets exist and each target's top context
        for (auto& kv : lane_ctx_acc_) {
            const std::uint64_t key = kv.first;
            const NodeId a = static_cast<NodeId>(key >> 32);
            const NodeId b = static_cast<NodeId>(key & 0xffffffffull);
            // required: top-4 by (count, then node id ascending for ties)
            std::vector<std::pair<float, NodeId>> cand;
            for (const auto& cv : kv.second)
                if (cv.second >= 2.0f) cand.emplace_back(cv.second, cv.first);
            if (cand.empty()) continue;
            std::sort(cand.begin(), cand.end(), [](const auto& x, const auto& y) {
                return x.first != y.first ? x.first > y.first : x.second < y.second;
            });
            LaneCtx sig;
            for (std::size_t i = 0; i < cand.size() && i < 4; ++i)
                sig.required.push_back(cand[i].second);
            lane_ctx_[key] = std::move(sig);
            (void)a; (void)b;
        }
        // forbidden pass: for lane (a,b), words that dominate a sibling lane
        // (a,b') with b' != b and are not in this lane's required set.
        std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> by_source;
        for (const auto& kv : lane_ctx_) {
            const NodeId a = static_cast<NodeId>(kv.first >> 32);
            by_source[a].push_back(kv.first);
        }
        for (auto& sv : by_source) {
            auto& keys = sv.second;
            std::sort(keys.begin(), keys.end());
            for (std::size_t i = 0; i < keys.size(); ++i) {
                // collect the strongest context words of the SIBLING lanes
                std::unordered_map<NodeId, float> sibling_ctx;
                for (std::size_t j = 0; j < keys.size(); ++j) {
                    if (i == j) continue;
                    auto it = lane_ctx_acc_.find(keys[j]);
                    if (it == lane_ctx_acc_.end()) continue;   // consumed
                    for (const auto& cv : it->second)
                        if (cv.second >= 3.0f) sibling_ctx[cv.first] += cv.second;
                }
                if (sibling_ctx.empty()) continue;
                // not already required here
                const LaneCtx& mine = lane_ctx_[keys[i]];
                auto is_req = [&](NodeId w) {
                    for (NodeId r : mine.required) if (r == w) return true;
                    return false;
                };
                std::vector<std::pair<float, NodeId>> fc;
                for (const auto& cv : sibling_ctx)
                    if (!is_req(cv.first)) fc.emplace_back(cv.second, cv.first);
                if (fc.empty()) continue;
                std::sort(fc.begin(), fc.end(), [](const auto& x, const auto& y) {
                    return x.first != y.first ? x.first > y.first : x.second < y.second;
                });
                for (std::size_t k = 0; k < fc.size() && k < 2; ++k)
                    lane_ctx_[keys[i]].forbidden.push_back(fc[k].second);
            }
        }
        lane_ctx_acc_.clear();
    }

    bool has_semantics() const { return !semvecs_.empty(); }
    void set_semantics(bool on) { sem_enabled_ = on; }

    // v3.4 — multi-hop readout walk depth. 1 (default) = the legacy single-hop
    // readout, bit-identical to every earlier version (replay contract).
    // Depths 2..8 walk lanes BFS-style from each probe anchor with per-hop
    // damping hop_coupling/sqrt(hop+1): hop 1 x0.707, hop 2 x0.577, hop 3
    // x0.500, hop 4 x0.447. Levels are visited in ascending NodeId order and
    // capped at 64 nodes per level, so the walk is deterministic and
    // bounded. Readout-layer only: settle() physics untouched.
    void set_hop_depth(int d) { hop_depth_ = std::max(1, std::min(8, d)); }
    int hop_depth() const { return hop_depth_; }
    std::size_t resonance_edge_count() const { return sem_edge_count_; }
    std::size_t lane_context_count() const { return lane_ctx_.size(); }

    // semantic vector of a node (empty vector when semantics absent)
    std::vector<float> sem_vector(NodeId id) const {
        std::vector<float> v;
        if (id >= node_count() || semvecs_.empty()) return v;
        v.assign(semvecs_.begin() + static_cast<std::ptrdiff_t>(id * SEM_DIMS),
                 semvecs_.begin() + static_cast<std::ptrdiff_t>((id + 1) * SEM_DIMS));
        return v;
    }

    // Stage 1: omega_semantic — scalar projection of the semantic vector onto
    // one FIXED deterministic direction, mapped to [0,1]. Fixed point of the
    // frequency axis: same string -> same omega, always.
    float semantic_freq(NodeId id) const {
        if (id >= node_count() || semvecs_.empty()) return 0.0f;
        float d = 0.0f;
        for (std::size_t k = 0; k < SEM_DIMS; ++k)
            d += semvecs_[id * SEM_DIMS + k] * omega_dir()[k];
        return std::max(0.0f, std::min(1.0f, 0.5f + 0.5f * d));
    }

    // Option 2 typed-lane read: cos >= 0.82 synonym-grade kin,
    //             cos >= 0.66 related; 0 = no semantic relation measured.
    static const char* semantic_type(float cos) {
        if (cos >= 0.82f) return "synonym";
        if (cos >= 0.66f) return "related";
        if (cos > 0.0f)   return "resonance";
        return "none";
    }

    // resonance neighbours of a node (id, cosine) — empty when no semantics
    const std::vector<std::pair<NodeId, float>>* resonance_of(NodeId id) const {
        if (semvecs_.empty() || id >= node_count()) return nullptr;
        static const std::vector<std::pair<NodeId, float>> kEmpty;
        const std::size_t b = sem_off_[id], e = sem_off_[id + 1];
        if (b == e) return &kEmpty;
        // slices live in sem_edges_; return pointer into it (stable between builds)
        static thread_local std::vector<std::pair<NodeId, float>> out;
        out.assign(sem_edges_.begin() + static_cast<std::ptrdiff_t>(b),
                   sem_edges_.begin() + static_cast<std::ptrdiff_t>(e));
        return &out;
    }

    const LaneCtx* lane_context(NodeId a, NodeId b) const {
        auto it = lane_ctx_.find(lane_key(a, b));
        return it == lane_ctx_.end() ? nullptr : &it->second;
    }

    // Build the semantic field over the CURRENT fabric. Deterministic.
    // Expensive once (O(n^2 * dims) neighbour search, OMP-parallel with a
    // fixed per-row reduction); stored in the model from then on.
    void build_semantics(int threads = 0);

    // -- persistence (binary, deterministic) -------------------------------------
    // v3.2.1 BUGFIX — save() used to iterate unordered_map members (out_,
    // derived_gen_, lane_evidence_, lane_ctx_) directly, so every load->save
    // cycle permuted the file (measured: three different md5s over two
    // rebuild passes on one fabric; the lane MULTISET is identical, the ORDER
    // is not). Consequences: model dirs were never byte-stable, and the
    // settled field's floating-point sum order drifted across save cycles.
    // Fix: every map-backed section is written in SORTED KEY order. Load
    // rebuilds out_ in file order, so iteration order is a pure function of
    // the file => one sorted save converges the format and every later
    // roundtrip is bit-identical. Same lanes, same weights: no physics change.
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
        std::vector<const std::pair<const NodeId, std::vector<std::pair<NodeId, float>>>*> sorted_out;
        sorted_out.reserve(out_.size());
        for (const auto& kv : out_) sorted_out.push_back(&kv);
        std::sort(sorted_out.begin(), sorted_out.end(),
                  [](const auto* x, const auto* y) { return x->first < y->first; });
        for (const auto* kv : sorted_out)
            for (const auto& l : kv->second) {
                f.write(reinterpret_cast<const char*>(&kv->first), 4);
                f.write(reinterpret_cast<const char*>(&l.first), 4);
                f.write(reinterpret_cast<const char*>(&l.second), 4);
            }
        // v2 provenance tail — old (v1) files simply end here; load() detects EOF.
        std::vector<std::pair<std::uint64_t, std::uint32_t>> sorted_derived;
        sorted_derived.reserve(derived_gen_.size());
        for (const auto& kv : derived_gen_) sorted_derived.emplace_back(kv.first, kv.second);
        std::sort(sorted_derived.begin(), sorted_derived.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        std::uint32_t dc = static_cast<std::uint32_t>(sorted_derived.size());
        f.write(reinterpret_cast<const char*>(&dc), 4);
        for (const auto& kv : sorted_derived) {
            std::uint32_t a = static_cast<std::uint32_t>(kv.first >> 32);
            std::uint32_t b = static_cast<std::uint32_t>(kv.first & 0xffffffffull);
            f.write(reinterpret_cast<const char*>(&a), 4);
            f.write(reinterpret_cast<const char*>(&b), 4);
            f.write(reinterpret_cast<const char*>(&kv.second), 4);
        }
        // v3 evidence tail — v2 files end here; load() detects EOF.
        std::vector<std::uint64_t> sorted_ev;
        sorted_ev.reserve(lane_evidence_.size());
        for (const auto& kv : lane_evidence_) sorted_ev.push_back(kv.first);
        std::sort(sorted_ev.begin(), sorted_ev.end());
        std::uint32_t ec = static_cast<std::uint32_t>(sorted_ev.size());
        f.write(reinterpret_cast<const char*>(&ec), 4);
        for (const auto& key : sorted_ev) {
            const auto& kv = *lane_evidence_.find(key);
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
        // v3.2 SEMANTIC TAIL — vectors + resonance edges + lane contexts.
        // Magic-guarded: a truncated/garbage read leaves semantics off and the
        // rest of the model fully usable (pre-v3.2 files simply end here).
        const std::uint32_t sem_magic = 0x53454D34u;             // 'SEM4'
        f.write(reinterpret_cast<const char*>(&sem_magic), 4);
        const std::uint32_t has_sem = semvecs_.empty() ? 0u : 1u;
        f.write(reinterpret_cast<const char*>(&has_sem), 4);
        if (has_sem) {
            f.write(reinterpret_cast<const char*>(semvecs_.data()),
                    static_cast<std::streamsize>(semvecs_.size() * sizeof(float)));
            const std::size_t n = nodes_.size();
            std::uint32_t total = static_cast<std::uint32_t>(sem_edges_.size());
            f.write(reinterpret_cast<const char*>(&total), 4);
            for (std::size_t i = 0; i <= n; ++i) {
                std::uint32_t off = static_cast<std::uint32_t>(sem_off_[i]);
                f.write(reinterpret_cast<const char*>(&off), 4);
            }
            for (const auto& e : sem_edges_) {
                f.write(reinterpret_cast<const char*>(&e.first), 4);
                f.write(reinterpret_cast<const char*>(&e.second), 4);
            }
        }
        std::vector<std::uint64_t> sorted_ctx;
        sorted_ctx.reserve(lane_ctx_.size());
        for (const auto& kv : lane_ctx_) sorted_ctx.push_back(kv.first);
        std::sort(sorted_ctx.begin(), sorted_ctx.end());
        const std::uint32_t ctxc = static_cast<std::uint32_t>(sorted_ctx.size());
        f.write(reinterpret_cast<const char*>(&ctxc), 4);
        for (const auto& key : sorted_ctx) {
            const auto& kv = *lane_ctx_.find(key);
            f.write(reinterpret_cast<const char*>(&kv.first), 8);
            std::uint32_t rq = static_cast<std::uint32_t>(kv.second.required.size());
            f.write(reinterpret_cast<const char*>(&rq), 4);
            for (NodeId w : kv.second.required) f.write(reinterpret_cast<const char*>(&w), 4);
            std::uint32_t fb = static_cast<std::uint32_t>(kv.second.forbidden.size());
            f.write(reinterpret_cast<const char*>(&fb), 4);
            for (NodeId w : kv.second.forbidden) f.write(reinterpret_cast<const char*>(&w), 4);
        }
        // v3.6.0 IDF TAIL — document-frequency table + lesson counter (CMD §2.2
        // / MIMS §6 rarity grounding). Magic-guarded 'IDF5' after the context
        // block: pre-v3.6 files end here and load() leaves df_ empty, which
        // makes every idf_factor 1.0 — the replay contract, bit for bit.
        const std::uint32_t idf_magic = 0x49444635u;             // 'IDF5'
        f.write(reinterpret_cast<const char*>(&idf_magic), 4);
        const std::uint64_t lesson_count = df_lessons_;
        f.write(reinterpret_cast<const char*>(&lesson_count), 8);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> sorted_df;
        sorted_df.reserve(df_.size());
        for (const auto& kv : df_) sorted_df.emplace_back(kv.first, kv.second);
        std::sort(sorted_df.begin(), sorted_df.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        const std::uint32_t dfc = static_cast<std::uint32_t>(sorted_df.size());
        f.write(reinterpret_cast<const char*>(&dfc), 4);
        for (const auto& kv : sorted_df) {
            f.write(reinterpret_cast<const char*>(&kv.first), 4);
            f.write(reinterpret_cast<const char*>(&kv.second), 4);
        }
    }

    // v3.2.1: reports failure. A missing substrate used to load SILENTLY as an
    // empty fabric (every decide then deferred with honest_silence, and a
    // --router mis-path surfaced as the misleading "lacks router.json anchors").
    // Honest failure: false <=> the file could not be opened.
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
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
        // v3.2 semantic tail (optional, magic-guarded): pre-v3.2 files end at
        // the evidence tail; the failed magic read leaves semantics absent.
        std::uint32_t magic = 0;
        if (f.read(reinterpret_cast<char*>(&magic), 4) && magic == 0x53454D34u) {
            std::uint32_t has_sem = 0;
            if (f.read(reinterpret_cast<char*>(&has_sem), 4) && has_sem == 1) {
                const std::size_t n = nodes_.size();
                semvecs_.assign(n * SEM_DIMS, 0.0f);
                if (f.read(reinterpret_cast<char*>(semvecs_.data()),
                           static_cast<std::streamsize>(semvecs_.size() * sizeof(float)))) {
                    std::uint32_t total = 0;
                    if (f.read(reinterpret_cast<char*>(&total), 4)) {
                        sem_edges_.clear();
                        sem_edges_.reserve(total);
                        sem_off_.assign(n + 1, 0);
                        bool ok = true;
                        for (std::size_t i = 0; i <= n && ok; ++i) {
                            std::uint32_t off = 0;
                            ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&off), 4));
                            sem_off_[i] = off;
                        }
                        for (std::uint32_t k = 0; k < total && ok; ++k) {
                            std::uint32_t id = 0; float w = 0;
                            ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&id), 4));
                            if (ok) ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&w), 4));
                            if (ok) sem_edges_.emplace_back(id, w);
                        }
                        if (ok) sem_edge_count_ = total;
                        else { sem_edges_.clear(); sem_off_.clear(); }
                    }
                }
                if (sem_edges_.empty()) semvecs_.clear();   // torn tail: stay off
            }
            std::uint32_t ctxc = 0;
            if (f.read(reinterpret_cast<char*>(&ctxc), 4)) {
                bool ok = true;
                for (std::uint32_t k = 0; k < ctxc && ok; ++k) {
                    std::uint64_t key = 0; std::uint32_t rq = 0, fb = 0;
                    ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&key), 8));
                    if (ok) ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&rq), 4));
                    LaneCtx sig;
                    for (std::uint32_t j = 0; j < rq && ok; ++j) {
                        std::uint32_t w = 0;
                        ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&w), 4));
                        if (ok) sig.required.push_back(w);
                    }
                    if (ok) ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&fb), 4));
                    for (std::uint32_t j = 0; j < fb && ok; ++j) {
                        std::uint32_t w = 0;
                        ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&w), 4));
                        if (ok) sig.forbidden.push_back(w);
                    }
                    if (ok) lane_ctx_[key] = std::move(sig);
                }
            }
        }
        // v3.6.0 IDF tail (optional, magic-guarded 'IDF5'): pre-v3.6 files end
        // after the context block; the failed magic read leaves df_ empty and
        // every idf_factor at 1.0.
        std::uint32_t idf_magic = 0;
        if (f.read(reinterpret_cast<char*>(&idf_magic), 4) && idf_magic == 0x49444635u) {
            std::uint64_t lesson_count = 0;
            std::uint32_t dfc = 0;
            if (f.read(reinterpret_cast<char*>(&lesson_count), 8) &&
                f.read(reinterpret_cast<char*>(&dfc), 4)) {
                df_lessons_ = lesson_count;
                bool ok = true;
                for (std::uint32_t k = 0; k < dfc && ok; ++k) {
                    std::uint32_t id = 0, c = 0;
                    ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&id), 4));
                    if (ok) ok = static_cast<bool>(f.read(reinterpret_cast<char*>(&c), 4));
                    if (ok) df_[static_cast<NodeId>(id)] = c;
                }
                if (!ok) { df_.clear(); df_lessons_ = 0; }   // torn tail: stay inert
            }
        }
        return true;
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
        drop_lane_ctx(a, b);                          // v3.2: signature dies with its lane
        drop_lane_ctx(b, a);
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
            drop_lane_ctx(a, weakest->first);            // v3.2: signature dies with its lane
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
    // v3.6.0 document-frequency table (node -> lessons containing it) and the
    // lesson counter; persisted in the 'IDF5' tail. Empty table = IDF inert.
    std::unordered_map<NodeId, std::uint32_t> df_;
    std::uint64_t df_lessons_ = 0;
    // v3 Milestone 3: audit ledger (support/counter events, provenance window)
    std::unordered_map<std::uint64_t, LaneEvidence> lane_evidence_;
    // v3.2 semantic field state ------------------------------------------------
    std::vector<float> semvecs_;                       // nodes_ * SEM_DIMS (row-major); empty = absent
    std::vector<std::pair<NodeId, float>> sem_edges_;  // resonance edges, CSR payload
    std::vector<std::size_t> sem_off_;                 // CSR offsets, nodes_.size()+1
    std::size_t sem_edge_count_ = 0;
    std::unordered_map<std::uint64_t, LaneCtx> lane_ctx_;            // final signatures
    std::unordered_map<std::uint64_t, std::unordered_map<NodeId, float>> lane_ctx_acc_; // learn-time scratch
    std::unordered_set<NodeId> present_;               // tokens injected since last reset_field()
    bool sem_enabled_ = true;                          // runtime kill switch (--no-semantics)
    int  hop_depth_ = 1;                               // v3.4 readout walk depth (1 = legacy)
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

private:
    // -- v3.2 semantic helpers --------------------------------------------------
    bool sem_active() const {
        return sem_enabled_ && !semvecs_.empty() && !sem_edges_.empty();
    }

    // fixed deterministic direction for omega_semantic (Stage 1); computed once
    static const float* omega_dir() {
        static const std::vector<float> dir = [] {
            std::vector<float> d(SEM_DIMS, 0.0f);
            for (std::size_t k = 0; k < SEM_DIMS; ++k) {
                std::uint64_t h = 1469598103934665603ull;
                const std::string seed = "syfox-omega-axis-" + std::to_string(k);
                for (unsigned char c : seed) { h ^= c; h *= 1099511628211ull; }
                d[k] = (static_cast<float>(h & 0xFFFFu) / 32767.5f) - 1.0f;  // [-1,1)
            }
            float n2 = 0.0f;
            for (float v : d) n2 += v * v;
            if (n2 > 0.0f) { const float inv = 1.0f / std::sqrt(n2); for (auto& v : d) v *= inv; }
            return d;
        }();
        return dir.data();
    }

    // coupled-oscillator detuning: equal frequencies exchange fully, far ones don't
    float freq_match(NodeId a, NodeId b) const {
        return 1.0f - 0.5f * std::fabs(semantic_freq(a) - semantic_freq(b));
    }

    // Stage 2 settle-side factor: how much of a lane's weight flows this pass
    float lane_flow_factor(NodeId a, NodeId b) const {
        auto it = lane_ctx_.find(lane_key(a, b));
        if (it == lane_ctx_.end()) return 1.0f;
        const LaneCtx& sig = it->second;
        for (const NodeId w : sig.forbidden)
            if (present_.count(w)) return cfg_.ctx_forbidden;
        if (sig.required.empty()) return 1.0f;
        std::size_t matched = 0;
        for (const NodeId w : sig.required)
            if (present_.count(w)) ++matched;
        if (matched == sig.required.size()) return 1.0f;
        return cfg_.ctx_missing
             + (1.0f - cfg_.ctx_missing)
             * (static_cast<float>(matched) / static_cast<float>(sig.required.size()));
    }

    void drop_lane_ctx(NodeId a, NodeId b) {
        lane_ctx_.erase(lane_key(a, b));
        lane_ctx_acc_.erase(lane_key(a, b));
    }

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
// v3.2 build_semantics — defined outside the class for OMP clarity.
// Deterministic end to end: lexical hashing is a pure function of the string;
// the grounding passes iterate node ids ascending; the neighbour search
// parallelizes over ROWS (each row's top-k depends only on that row), with
// ties broken by (cosine desc, neighbour id asc).
// ---------------------------------------------------------------------------
inline void Substrate::build_semantics(int threads) {
    const std::size_t n = node_count();
    semvecs_.clear(); sem_edges_.clear(); sem_off_.clear(); sem_edge_count_ = 0;
    if (n == 0) return;
    // Cost guard: the neighbour search is O(n^2 * dims). Dedicated fabrics are
    // 10^3..10^4 nodes (seconds); a 43k-node fabric is minutes once. Beyond
    // 65k the build is refused — semantics stays absent, the model stays
    // fully usable (honest scope, not silent failure).
    if (n > 65000) return;
    (void)threads;
#if defined(_OPENMP)
    if (threads > 0) omp_set_num_threads(threads);
#endif

    // Step 1 — LEXICAL LAYER: signed character-trigram hashing, L2-normalized.
    semvecs_.assign(n * SEM_DIMS, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        float* v = &semvecs_[i * SEM_DIMS];
        const std::string& s = concept_of(static_cast<NodeId>(i));
        auto bump = [&](const std::string& g, float w) {
            std::uint64_t h = 1469598103934665603ull;
            for (unsigned char c : g) { h ^= c; h *= 1099511628211ull; }
            const std::size_t bucket = static_cast<std::size_t>(h % SEM_DIMS);
            const float sign = ((h >> 40) & 1u) ? 1.0f : -1.0f;
            v[bucket] += sign * w;
        };
        if (s.size() < 3) {
            bump(s, 2.0f);
        } else {
            for (std::size_t p = 0; p + 3 <= s.size(); ++p) bump(s.substr(p, 3), 1.0f);
            bump(s, 1.5f);                       // whole-token identity, too
        }
        float n2 = 0.0f;
        for (std::size_t k = 0; k < SEM_DIMS; ++k) n2 += v[k] * v[k];
        if (n2 > 0.0f) {
            const float inv = 1.0f / std::sqrt(n2);
            for (std::size_t k = 0; k < SEM_DIMS; ++k) v[k] *= inv;
        }
    }

    // Step 2 — FABRIC GROUNDING: two smoothing passes over the Hebbian lanes.
    // v_i <- normalize(v_i + beta * lane-weighted mean of v_j). This is how
    // the fabric's own co-occurrence structure shapes the vectors — meaning
    // diffusing along the lanes, not a fit.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> nx(n * SEM_DIMS, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const float* v = &semvecs_[i * SEM_DIMS];
            float* o = &nx[i * SEM_DIMS];
            for (std::size_t k = 0; k < SEM_DIMS; ++k) o[k] = v[k];
            auto it = out_.find(static_cast<NodeId>(i));
            if (it != out_.end() && !it->second.empty()) {
                float wsum = 0.0f;
                float avg[SEM_DIMS];
                for (std::size_t k = 0; k < SEM_DIMS; ++k) avg[k] = 0.0f;
                for (const auto& l : it->second) {
                    const float* u = &semvecs_[l.first * SEM_DIMS];
                    for (std::size_t k = 0; k < SEM_DIMS; ++k) avg[k] += l.second * u[k];
                    wsum += l.second;
                }
                if (wsum > 0.0f)
                    for (std::size_t k = 0; k < SEM_DIMS; ++k)
                        o[k] = v[k] + cfg_.sem_beta * (avg[k] / wsum);
            }
            float n2 = 0.0f;
            for (std::size_t k = 0; k < SEM_DIMS; ++k) n2 += o[k] * o[k];
            if (n2 > 0.0f) {
                const float inv = 1.0f / std::sqrt(n2);
                for (std::size_t k = 0; k < SEM_DIMS; ++k) o[k] *= inv;
            }
        }
        semvecs_.swap(nx);
    }
    (void)omega_dir();   // materialize the fixed omega axis once

    // Step 3 — RESONANCE EDGES: top-k cosine neighbours with cos >= theta.
    const int kmax = std::max(1, cfg_.sem_neighbors);
    std::vector<std::vector<std::pair<float, NodeId>>> per(n);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        const float* vi = &semvecs_[i * SEM_DIMS];
        float vi2 = 0.0f;
        for (std::size_t d = 0; d < SEM_DIMS; ++d) vi2 += vi[d] * vi[d];
        if (vi2 <= 0.0f) continue;                       // dark node: no resonance
        std::vector<std::pair<float, NodeId>> best;      // kept sorted: cos desc, id asc
        best.reserve(static_cast<std::size_t>(kmax) + 1);
        auto better = [](const std::pair<float, NodeId>& x, const std::pair<float, NodeId>& y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
        };
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const float* vj = &semvecs_[j * SEM_DIMS];
            float dot = 0.0f;
            for (std::size_t d = 0; d < SEM_DIMS; ++d) dot += vi[d] * vj[d];
            if (dot < cfg_.sem_theta) continue;
            if (best.size() < static_cast<std::size_t>(kmax)) {
                best.emplace_back(dot, static_cast<NodeId>(j));
                std::sort(best.begin(), best.end(), better);
            } else if (better({dot, static_cast<NodeId>(j)}, best.back())) {
                best.back() = {dot, static_cast<NodeId>(j)};
                std::sort(best.begin(), best.end(), better);
            }
        }
        per[i] = std::move(best);
    }
    sem_off_.assign(n + 1, 0);
    std::size_t running = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sem_off_[i] = running;
        running += per[i].size();
    }
    sem_off_[n] = running;
    sem_edges_.reserve(running);
    for (std::size_t i = 0; i < n; ++i)
        for (const auto& p : per[i]) sem_edges_.emplace_back(p.second, p.first);
    sem_edge_count_ = running;
}

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
