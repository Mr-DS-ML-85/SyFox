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
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace si {

using NodeId = std::uint32_t;

// ---------------------------------------------------------------------------
// Physics constants — the substrate's constitution.
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
            if (l.first == b) { l.second = std::min(4.0f, std::max(0.0f, l.second + w)); return; }
        if (w > 0.0f) {
            lanes.emplace_back(b, w);
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
        for (int pass = 0; pass < cfg_.k_settle; ++pass) {
            float total_before = 0.0f;
            for (const auto& n : nodes_) total_before += n.energy;
            if (total_before <= 0.0f) return;             // nothing to settle

            std::fill(next.begin(), next.end(), 0.0f);
            active.clear();
            for (std::size_t i = 0; i < nodes_.size(); ++i)
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
            std::vector<char> is_source(nodes_.size(), 0);
            for (std::size_t i : active) is_source[i] = 1;

            for (std::size_t i = 0; i < nodes_.size(); ++i) {
                const Node& n = nodes_[i];
                if (n.energy <= 0.0f) continue;
                if (!is_source[i]) { next[i] += n.energy * cfg_.decay; continue; }  // gated: decay only
                const auto it = out_.find(static_cast<NodeId>(i));
                float out_w = 0.0f;
                if (it != out_.end())
                    for (const auto& lane : it->second) out_w += lane.second;
                float retained = (out_w > 0.0f) ? (1.0f - cfg_.diffusion) : 1.0f;
                next[i] += n.energy * cfg_.decay * retained;   // damped self-retention
                if (it != out_.end() && out_w > 0.0f) {
                    float flow = n.energy * cfg_.decay * cfg_.diffusion;
                    for (const auto& lane : it->second)
                        next[lane.first] += flow * (lane.second / out_w);
                }
            }
            // SI cavity salience integrator: s = tanh(s·decay + gain·|ΔE|).
            // Motion proxy is how much energy moved through the node this
            // pass. Updated before the energies are committed, so the next
            // pass's source ranking sees it.
            for (std::size_t i = 0; i < nodes_.size(); ++i) {
                if (nodes_[i].energy <= 0.0f && next[i] <= 0.0f) continue;
                float motion = std::fabs(next[i] - nodes_[i].energy);
                nodes_[i].salience = std::tanh(nodes_[i].salience * cfg_.salience_decay
                                             + cfg_.salience_gain * motion);
            }
            for (std::size_t i = 0; i < nodes_.size(); ++i) nodes_[i].energy = next[i];

            float total_after = 0.0f;
            for (const auto& n : nodes_) total_after += n.energy;
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
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return;
        nodes_.clear(); index_.clear(); out_.clear();
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
                if (w > l.second) l.second = std::min(4.0f, w);
                derived_gen_.emplace(lane_key(a, b), gen);  // keep first (lowest) grade
                return;
            }
        }
        lanes.emplace_back(b, w);
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
    }

    SubstrateConfig cfg_;
    std::unordered_map<std::string, NodeId> index_;
    std::vector<Node> nodes_;
    std::unordered_map<NodeId, std::vector<std::pair<NodeId, float>>> out_;
    // Derivation provenance: lane key -> generation (sparse; observed lanes absent = 0)
    std::unordered_map<std::uint64_t, std::uint32_t> derived_gen_;
    std::uint64_t state_hash_ = 0;   // decision fingerprint for Miller sampling
    float last_cap_ = 0.0f;          // cap used by the last settle()
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
