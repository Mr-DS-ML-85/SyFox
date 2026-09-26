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
//                     keeps the working set bounded (SI-inspired — the SI
//                     substrate samples a Miller window [5,9], SyFox pins a
//                     fixed cap); the field stops when the total energy delta
//                     drops below eps (phase stability).
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
                                   // (SI-inspired bounded working set; SI samples [5,9], SyFox fixes 24)
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
        nodes_.push_back(Node{concept, 1.0f, 0.0f});
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
        }
    }

    // -- SETTLE ---------------------------------------------------------------
    // One pass = dissipative diffusion along lanes + source gating.
    // Energy is CONSERVED per pass up to decay: each node retains
    // (1 - diffusion) of its (decayed) energy and flows `diffusion` out along
    // its lanes, split proportionally to lane weight. The source cap GATES
    // propagation (only the `source_cap` highest-ENERGY nodes send energy
    // this pass — selection is by energy, nothing else);
    // gated nodes keep their energy (readout-visible) but stay silent as
    // sources — bounded working set, nothing is annihilated.
    void settle() {
        std::vector<float> next(nodes_.size(), 0.0f);
        std::vector<std::size_t> active;
        for (int pass = 0; pass < cfg_.k_settle; ++pass) {
            float total_before = 0.0f;
            for (const auto& n : nodes_) total_before += n.energy;
            if (total_before <= 0.0f) return;             // nothing to settle

            std::fill(next.begin(), next.end(), 0.0f);
            active.clear();
            for (std::size_t i = 0; i < nodes_.size(); ++i)
                if (nodes_[i].energy > 1e-7f) active.push_back(i);

            // source gating: at most `source_cap` (top-K by energy) per pass
            if (static_cast<float>(active.size()) > cfg_.source_cap) {
                std::nth_element(active.begin(),
                                 active.begin() + static_cast<std::ptrdiff_t>(cfg_.source_cap),
                                 active.end(), [&](std::size_t a, std::size_t b){
                                     return nodes_[a].energy > nodes_[b].energy; });
                active.resize(static_cast<std::size_t>(cfg_.source_cap));
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

    bool silent() const { return total_energy() < cfg_.silence_floor; }

    // -- HONEST SILENCE reset ---------------------------------------------------
    void reset_field() {
        for (auto& n : nodes_) n.energy = 0.0f;
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
            nodes_.push_back(Node{c, mass, 0.0f});
        }
        std::uint32_t lanes = 0; f.read(reinterpret_cast<char*>(&lanes), 4);
        for (std::uint32_t i = 0; i < lanes; ++i) {
            NodeId a = 0, b = 0; float w = 0;
            f.read(reinterpret_cast<char*>(&a), 4);
            f.read(reinterpret_cast<char*>(&b), 4);
            f.read(reinterpret_cast<char*>(&w), 4);
            out_[a].emplace_back(b, w);
        }
    }

    const SubstrateConfig& config() const { return cfg_; }

private:
    struct Node { std::string concept; float mass; float energy; };

    void enforce_lane_decay() {
        for (auto& kv : out_)
            for (auto& l : kv.second) l.second *= cfg_.lane_decay;
    }

    SubstrateConfig cfg_;
    std::unordered_map<std::string, NodeId> index_;
    std::vector<Node> nodes_;
    std::unordered_map<NodeId, std::vector<std::pair<NodeId, float>>> out_;
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
