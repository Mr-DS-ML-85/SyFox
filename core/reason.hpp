// ============================================================================
//  SyFox — reason.hpp
// ---------------------------------------------------------------------------
//  v3.9.0 — THE CONSTRUCTIVE SUBSTRATE (SI Layer 2), ported at last.
//
//  This is the faithful port of the Synthetic-Intelligence reasoning stack —
//  src/unified.hpp (graph rewriter + backward chaining + contradiction guard),
//  src/primitives.hpp (the nine reasoning primitives), src/meta_learner.hpp
//  (rule induction, verifier, thinker index), src/inference.hpp (the six
//  discovery queries) and src/axiom_loader.hpp (the .axioms triple format) —
//  closing the gap the v3.9 audit named: syfox had the FIELD (Layer 1 energy
//  physics) but none of SI's Layer 2: no typed knowledge graph, no forward or
//  backward chaining, no proof traces, no rule induction, no discovery.
//
//  What is ported 1:1 (paper: synthetic-intelligence.md §3.4, §4.4, §5.3):
//    - Typed directed edges with FULL provenance (rule, parent_a, parent_b,
//      step) — every derived edge knows exactly which premises produced it,
//      so proofs are traceable and retraction is bounded (RADE semantics).
//    - Forward chaining to fixpoint under a tension budget (Axiom II:
//      |ΔG| <= alpha·T; here tension is an input, fed from the fabric layer).
//    - The NINE primitives: transitive, deduction (modus ponens),
//      inheritance (class-property flow), negation (contrapositive),
//      case analysis (disjunction elimination), induction (schematic
//      generalization with a witness count), analogy (structure-mapping edge
//      transfer), abduction (backward-only), recursion (implicit in the
//      fixpoint loop).
//    - Backward chaining with adaptive iterative deepening (progress-bounded,
//      not depth-bounded) and cycle protection via an active goal stack.
//    - Meta-learner: contiguous derivation sub-chains are counted; a chain
//      that saves >= promotion_threshold bits is promoted to a macro-rule.
//    - Verifier: a candidate answer edge is re-checked against the rule set
//      all the way back to its premises before an answer is emitted.
//    - Thinker index C/R (§3.4): composition rate vs retrieval rate.
//    - The SIX discovery queries: isolated subgraphs, dangling edges, logic
//      gap, symmetry gaps, anomalies, and the gap-query classifier.
//    - The .axioms triple format — SI's own data files (syllogism.axioms,
//      insulin_pathway.axioms, maxwell_equations.axioms, ...) load unchanged.
//
//  What is deliberately DIFFERENT from SI, and why:
//    - No RIFA/oscillator physics dependency. In syfox the Layer-1 physics is
//      the energy field (si_substrate.hpp); the reasoning budget takes
//      `tension` as an injected scalar (set_tension) so a RIFA-style tension
//      accumulator can drive it without coupling the two stores.
//    - Determinism hardening: primitive candidate sets are collected and
//      sorted by a total order before committing (SI iterated hash maps).
//      Same derivations, same budget arithmetic, one canonical order —
//      bit-stable across runs and thread counts by construction.
//    - Base-case edge lookup uses the out_adj index (O(deg)) instead of a
//      full edge scan; semantics identical (first matching edge).
//
//  Nothing here touches the settle physics (decay 0.82 / diffusion 0.45) or
//  the field readout. This file is Layer 2; the fabric is Layer 1; the bridge
//  (Engine::reason_bridge) lays fabric lanes for asserted axioms at learn
//  time and walks typed proof paths over settled fields at decide time.
//  No transformer, no neural network, no token prediction — counting,
//  joining, and traversal only.
// ============================================================================

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sxr {

// ---------------------------------------------------------------------------
// Relations. The base six are SI's unified.hpp set; the extension block is
// primitives.hpp verbatim (mechanistic, symbolic, code-structure,
// commonsense). A syfox axiom file may use ANY of these tokens.
// ---------------------------------------------------------------------------
enum RelId : int {
    REL_IS_A      = 0,   // "socrates is_a man"
    REL_HAS       = 1,   // "man has mortality"
    REL_IMPLIES   = 2,   // "raining implies wet"
    REL_PART_OF   = 3,   // "wheel part_of car"
    REL_EQUALS    = 4,   // "x equals 3"
    REL_CAUSES    = 5,   // "friction causes heat"
    REL_NOT       = 6,   // "socrates NOT is_a stone" (unary; X --not--> P)
    REL_OR        = 7,   // "wet_or_rainy OR (rainy, wet)"
    REL_HYPOTH    = 8,   // hypothesised edge tag (MAA volatile bridges)
    // mechanistic (drug repurposing / pathway reasoning)
    REL_BINDS_TO        = 9,
    REL_INHIBITS        = 10,
    REL_ACTIVATES       = 11,
    REL_TREATS          = 12,
    REL_CAUSES_DISEASE  = 13,
    REL_PART_OF_PATHWAY = 14,
    REL_EXPRESSED_IN    = 15,
    REL_METABOLIZES     = 16,
    REL_UPREGULATES     = 17,
    REL_DOWNREGULATES   = 18,
    // symbolic / physics
    REL_PROPORTIONAL_TO = 19,
    REL_MIRROR_OF       = 20,
    REL_DERIVES_FROM    = 21,
    REL_FUNDAMENTAL     = 22,
    REL_TRIGGERED_BY    = 23,
    // code structure
    REL_CALLS           = 24,
    REL_RETURNS         = 25,
    REL_TAKES_ARG       = 26,
    REL_PRECEDES        = 27,
    REL_IMPORTS         = 28,
    REL_MUTATES         = 29,
    // commonsense (testimonial support, never experimental)
    REL_USED_FOR        = 30,
    REL_CAPABLE_OF      = 31,
    REL_AT_LOCATION     = 32,
    REL_DESIRES         = 33,
    REL_CREATED_BY      = 34,
    REL_MAX             = 35
};

inline const char* rel_name(int r) {
    switch (r) {
        case REL_IS_A:              return "is_a";
        case REL_HAS:               return "has";
        case REL_IMPLIES:           return "implies";
        case REL_PART_OF:           return "part_of";
        case REL_EQUALS:            return "equals";
        case REL_CAUSES:            return "causes";
        case REL_NOT:               return "not";
        case REL_OR:                return "or";
        case REL_HYPOTH:            return "hyp";
        case REL_BINDS_TO:          return "binds_to";
        case REL_INHIBITS:          return "inhibits";
        case REL_ACTIVATES:         return "activates";
        case REL_TREATS:            return "treats";
        case REL_CAUSES_DISEASE:    return "causes_disease";
        case REL_PART_OF_PATHWAY:   return "part_of_pathway";
        case REL_EXPRESSED_IN:      return "expressed_in";
        case REL_METABOLIZES:       return "metabolizes";
        case REL_UPREGULATES:       return "upregulates";
        case REL_DOWNREGULATES:     return "downregulates";
        case REL_PROPORTIONAL_TO:   return "is_proportional_to";
        case REL_MIRROR_OF:         return "mirror_of";
        case REL_DERIVES_FROM:      return "derives_from";
        case REL_FUNDAMENTAL:       return "is_fundamental";
        case REL_TRIGGERED_BY:      return "triggered_by";
        case REL_CALLS:             return "calls";
        case REL_RETURNS:           return "returns";
        case REL_TAKES_ARG:         return "takes_arg";
        case REL_PRECEDES:          return "precedes";
        case REL_IMPORTS:           return "imports";
        case REL_MUTATES:           return "mutates";
        case REL_USED_FOR:          return "used_for";
        case REL_CAPABLE_OF:        return "capable_of";
        case REL_AT_LOCATION:       return "at_location";
        case REL_DESIRES:           return "desires";
        case REL_CREATED_BY:        return "created_by";
        default:                    return "?";
    }
}

inline int rel_from_token(const std::string& t) {
    for (int r = 0; r < REL_MAX; ++r)
        if (t == rel_name(r)) return r;
    return -1;
}

// Node types (unified.hpp).
enum NodeType : int {
    NODE_ENTITY   = 0,
    NODE_CLASS    = 1,
    NODE_PROPERTY = 2,
    NODE_ABSTRACT = 3,   // spawned by phase-space expansion
};

// ---------------------------------------------------------------------------
// Edge with full provenance. A premise has derived_by_rule == -1; a derived
// edge names the rule that fired and the two parent edges, so trace() can
// reconstruct the whole derivation chain and retraction is bounded.
// ---------------------------------------------------------------------------
struct Edge {
    int from;
    int to;
    int rel;
    int derived_by_rule = -1;
    int parent_a = -1;
    int parent_b = -1;
    int step = 0;
};

struct Node {
    int id = -1;
    std::string label;
    int type = NODE_ENTITY;
};

// ---------------------------------------------------------------------------
// The typed graph. O(1) existence via edge_key; O(deg) neighbourhood via
// out_adj / in_adj — the incremental-index design SI built when ConceptNet
// scale turned full scans fatal. All three are maintained in add_edge.
// ---------------------------------------------------------------------------
struct ConceptGraph {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::unordered_map<std::string, int> by_label;

    std::unordered_set<std::uint64_t> edge_key;
    std::vector<std::vector<int>> out_adj;
    std::vector<std::vector<int>> in_adj;

    static std::uint64_t ekey(int from, int to, int rel) {
        // rel < 64 for every relation above; 29 bits per endpoint.
        return ((std::uint64_t)(std::uint32_t)from << 35)
             | ((std::uint64_t)(std::uint32_t)to   << 6)
             | (std::uint64_t)(std::uint32_t)(rel & 0x3f);
    }

    int add_or_get_node(const std::string& label, int type = NODE_ENTITY) {
        auto it = by_label.find(label);
        if (it != by_label.end()) return it->second;
        int id = static_cast<int>(nodes.size());
        nodes.push_back(Node{id, label, type});
        by_label.emplace(label, id);
        return id;
    }

    bool has_edge(int from, int to, int rel) const {
        return edge_key.count(ekey(from, to, rel)) != 0;
    }

    // First matching edge index, or -1. Deterministic: out_adj preserves
    // insertion order, so the first edge added is the one found.
    int find_edge(int from, int to, int rel) const {
        if (from < 0 || from >= static_cast<int>(out_adj.size())) return -1;
        for (int ei : out_adj[static_cast<std::size_t>(from)]) {
            const Edge& e = edges[static_cast<std::size_t>(ei)];
            if (e.to == to && e.rel == rel) return ei;
        }
        return -1;
    }

    int add_edge(int from, int to, int rel,
                 int rule_idx = -1, int pa = -1, int pb = -1, int step = 0) {
        std::uint64_t k = ekey(from, to, rel);
        if (edge_key.count(k)) return -1;
        int id = static_cast<int>(edges.size());
        edges.push_back(Edge{from, to, rel, rule_idx, pa, pb, step});
        edge_key.insert(k);
        const std::size_t need = static_cast<std::size_t>(std::max(from, to)) + 1;
        if (out_adj.size() < need) { out_adj.resize(need); in_adj.resize(need); }
        out_adj[static_cast<std::size_t>(from)].push_back(id);
        in_adj[static_cast<std::size_t>(to)].push_back(id);
        return id;
    }

    // Bounded retraction (RADE persistence semantics): removing one wrong
    // premise edge makes every derived edge citing it unsupported. Returns
    // the number of edges removed. Provenance makes this local, not global.
    std::size_t retract(int edge_idx) {
        if (edge_idx < 0 || edge_idx >= static_cast<int>(edges.size())) return 0;
        std::vector<char> dead(edges.size(), 0);
        dead[static_cast<std::size_t>(edge_idx)] = 1;
        bool grew = true;
        std::size_t killed = 0;
        while (grew) {                     // bounded by the derivation DAG depth
            grew = false;
            for (std::size_t i = 0; i < edges.size(); ++i) {
                if (dead[i]) continue;
                const Edge& e = edges[i];
                if (e.derived_by_rule < 0) continue;
                const bool pa_dead = e.parent_a >= 0 &&
                    dead[static_cast<std::size_t>(e.parent_a)];
                const bool pb_dead = e.parent_b >= 0 &&
                    dead[static_cast<std::size_t>(e.parent_b)];
                if (pa_dead || pb_dead) { dead[i] = 1; grew = true; }
            }
        }
        std::vector<Edge> keep;
        keep.reserve(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) {
            if (dead[i]) { ++killed; continue; }
            keep.push_back(edges[i]);
        }
        edges.swap(keep);
        reindex();
        return killed;
    }

    void reindex() {
        edge_key.clear();
        out_adj.assign(nodes.size(), {});
        in_adj.assign(nodes.size(), {});
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const Edge& e = edges[i];
            edge_key.insert(ekey(e.from, e.to, e.rel));
            const std::size_t need = static_cast<std::size_t>(std::max(e.from, e.to)) + 1;
            if (out_adj.size() < need) { out_adj.resize(need); in_adj.resize(need); }
            out_adj[static_cast<std::size_t>(e.from)].push_back(static_cast<int>(i));
            in_adj[static_cast<std::size_t>(e.to)].push_back(static_cast<int>(i));
        }
    }
};

// ---------------------------------------------------------------------------
// A rewrite rule: (a --rel_ab--> b) + (b --rel_bc--> c) => (a --rel_ac--> c).
// Cost is deducted from the tension budget per successful application.
// ---------------------------------------------------------------------------
struct Rule {
    std::string name;
    int rel_ab;
    int rel_bc;
    int rel_ac;
    float cost = 1.0f;
    bool spawn_intermediate = false;   // Axiom II: phase space grows in NODES
};

// One rewriting pass over a snapshot (rules never trigger themselves within
// a pass). Returns edges added.
inline int apply_rules_once(std::vector<Rule>& rules, ConceptGraph& g,
                            float& budget, int step) {
    int added = 0;
    const std::size_t snap = g.edges.size();
    for (std::size_t ri = 0; ri < rules.size(); ++ri) {
        const Rule& r = rules[ri];
        if (budget < r.cost) continue;
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e1 = g.edges[i];
            if (e1.rel != r.rel_ab) continue;
            if (e1.to < 0 || static_cast<std::size_t>(e1.to) >= g.out_adj.size()) continue;
            const std::size_t kn = g.out_adj[static_cast<std::size_t>(e1.to)].size();
            for (std::size_t k = 0; k < kn; ++k) {
                const std::size_t j = static_cast<std::size_t>(
                    g.out_adj[static_cast<std::size_t>(e1.to)][k]);
                if (j >= snap) continue;
                const Edge& e2 = g.edges[j];
                if (e2.rel != r.rel_bc) continue;
                if (g.has_edge(e1.from, e2.to, r.rel_ac)) continue;
                if (r.spawn_intermediate) {
                    const std::string lbl = std::string("[") + rel_name(r.rel_ac)
                        + ":" + g.nodes[static_cast<std::size_t>(e1.from)].label
                        + "->" + g.nodes[static_cast<std::size_t>(e2.to)].label + "]";
                    const int mid = g.add_or_get_node(lbl, NODE_ABSTRACT);
                    g.add_edge(e1.from, mid, r.rel_ac, static_cast<int>(ri),
                               static_cast<int>(i), static_cast<int>(j), step);
                    g.add_edge(mid, e2.to, r.rel_ac, static_cast<int>(ri),
                               static_cast<int>(i), static_cast<int>(j), step);
                    added += 2;
                } else {
                    g.add_edge(e1.from, e2.to, r.rel_ac, static_cast<int>(ri),
                               static_cast<int>(i), static_cast<int>(j), step);
                    ++added;
                }
                budget -= r.cost;
                if (budget < r.cost) break;
            }
            if (budget < r.cost) break;
        }
    }
    return added;
}

// ---------------------------------------------------------------------------
// The nine primitive kinds (primitives.hpp §4.4). PRIM_ABDUCTION is realized
// by backward chaining (goal-directed) and PRIM_RECURSION by the fixpoint
// loop; both are structural, not separate fire functions.
// ---------------------------------------------------------------------------
enum PrimitiveKind {
    PRIM_TRANSITIVE    = 0,
    PRIM_DEDUCTION     = 1,
    PRIM_INHERITANCE   = 2,
    PRIM_NEGATION      = 3,
    PRIM_CASE_ANALYSIS = 4,
    PRIM_INDUCTION     = 5,
    PRIM_ANALOGY       = 6,
    PRIM_ABDUCTION     = 7,   // backward-chaining side
    PRIM_RECURSION     = 8    // the fixpoint loop itself
};

struct PrimRuleMeta {
    PrimitiveKind kind = PRIM_TRANSITIVE;
    int rel_extra = -1;      // NEGATION: the negated rel; INDUCTION: property
    int min_witnesses = 3;   // INDUCTION only
};

// ---------------------------------------------------------------------------
// The reasoning engine (SI's ExtSI, self-contained).
// ---------------------------------------------------------------------------
struct ReasonCore {
    ConceptGraph graph;
    std::vector<Rule> rules;
    std::vector<PrimRuleMeta> prim_meta;

    float base_budget = 100.0f;    // reasoning-step budget scale
    float tension = 0.0f;          // injected Layer-1 tension (RIFA IC port)
    int   max_steps = 1 << 20;     // runaway guard; engine self-terminates
    bool  bulk_load = false;       // seeding is not lived experience

    // -- rule registration --------------------------------------------------
    void add_rule(const std::string& name, int rel_ab, int rel_bc, int rel_ac,
                  bool spawn = false, float cost = 1.0f) {
        rules.push_back(Rule{name, rel_ab, rel_bc, rel_ac, cost, spawn});
        while (prim_meta.size() < rules.size()) prim_meta.push_back({});
    }
    void add_primitive(const std::string& name, PrimitiveKind kind,
                       int rel_ab, int rel_bc, int rel_ac,
                       int rel_extra = -1, int min_witnesses = 3) {
        add_rule(name, rel_ab, rel_bc, rel_ac);
        // add_rule padded prim_meta with a default slot for this rule —
        // overwrite it with the real kind (keeps prim_meta 1:1 with rules;
        // a first cut pushed a SECOND slot here and shifted every later
        // primitive's kind off by one — the contrapositive and induction
        // never fired, caught by test_reason).
        prim_meta.back() = {kind, rel_extra, min_witnesses};
    }
    void add_transitive(const std::string& name, int a, int b, int c) {
        add_primitive(name, PRIM_TRANSITIVE, a, b, c);
    }
    void add_deduction(const std::string& name) {
        add_primitive(name, PRIM_DEDUCTION, REL_IMPLIES, REL_IMPLIES, REL_IMPLIES);
    }
    void add_inheritance(const std::string& name, int property_rel) {
        add_primitive(name, PRIM_INHERITANCE, REL_IS_A, property_rel, property_rel);
    }
    void add_negation(const std::string& name, int implies_rel = REL_IMPLIES) {
        add_primitive(name, PRIM_NEGATION, implies_rel, REL_NOT, REL_NOT);
    }
    void add_case_analysis(const std::string& name, int consequence = REL_IMPLIES) {
        add_primitive(name, PRIM_CASE_ANALYSIS, REL_OR, consequence, consequence);
    }
    void add_induction(const std::string& name, int property_rel, int witnesses = 3) {
        add_primitive(name, PRIM_INDUCTION, REL_IS_A, property_rel, property_rel,
                      property_rel, witnesses);
    }
    void add_analogy(const std::string& name, int mapping_rel = REL_IS_A) {
        add_primitive(name, PRIM_ANALOGY, mapping_rel, mapping_rel, mapping_rel);
    }

    // SI's default rule set (unified.cpp construction, ported): taxonomy
    // flows down classes, chains compose, causes propagate through parts.
    void add_default_rules() {
        add_transitive("transitive_is_a", REL_IS_A, REL_IS_A, REL_IS_A);
        add_deduction("modus_ponens");
        add_transitive("implies_chain", REL_IMPLIES, REL_IMPLIES, REL_IMPLIES);
        add_transitive("causes_chain", REL_CAUSES, REL_CAUSES, REL_CAUSES);
        add_inheritance("class_has_flows_down", REL_HAS);
        add_inheritance("class_causes_flows_down", REL_CAUSES);
        add_negation("contrapositive");
        add_case_analysis("disjunction_elimination");
        add_induction("schematic_generalization", REL_HAS, 3);
    }

    // -- assertions ----------------------------------------------------------
    int assert_fact(const std::string& subj, int rel, const std::string& obj,
                    int subj_type = NODE_ENTITY, int obj_type = NODE_CLASS) {
        const int a = graph.add_or_get_node(subj, subj_type);
        const int b = graph.add_or_get_node(obj, obj_type);
        return graph.add_edge(a, b, rel);
    }

    // Tension-bounded budget (Axiom II in code): |ΔG| <= alpha·T.
    float compute_budget() const {
        return base_budget * (1.0f + 2.0f * tension);
    }

    // -- forward chaining to fixpoint ----------------------------------------
    struct ReasonReport { int edges_added; int steps_used; float budget_left; };
    ReasonReport reason() {
        float budget = compute_budget();
        int total = 0, step = 0;
        for (; step < max_steps; ++step) {
            const int add = apply_rules_once(rules, graph, budget, step + 1);
            if (add == 0) break;
            total += add;
        }
        return {total, step, budget};
    }

    // -- the nine primitives, per kind ---------------------------------------
    // Provenance-preserving join: each candidate carries its two parent edge
    // indices, so every committed edge is traceable. Candidates are sorted by
    // a total order before committing (determinism hardening over SI's hash
    // iteration); parents always have indices < snap and stay valid because
    // edges are only ever appended.

    struct Cand {
        int from, to;
        int pa, pb;
        std::uint64_t key() const {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32)
                 | static_cast<std::uint32_t>(to);
        }
    };

    // 1+2. DEDUCTION / TRANSITIVE — join on e1.to == e2.from.
    int fire_deduction(std::size_t rule_idx, float& budget, int step) {
        const Rule& r = rules[rule_idx];
        if (budget < r.cost) return 0;
        std::vector<Cand> cands;
        const std::size_t snap = graph.edges.size();
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e1 = graph.edges[i];
            if (e1.rel != r.rel_ab) continue;
            if (e1.to < 0 || static_cast<std::size_t>(e1.to) >= graph.out_adj.size()) continue;
            const std::size_t kn = graph.out_adj[static_cast<std::size_t>(e1.to)].size();
            for (std::size_t k = 0; k < kn; ++k) {
                const std::size_t j = static_cast<std::size_t>(
                    graph.out_adj[static_cast<std::size_t>(e1.to)][k]);
                if (j >= snap) continue;
                const Edge& e2 = graph.edges[j];
                if (e2.rel != r.rel_bc) continue;
                if (graph.has_edge(e1.from, e2.to, r.rel_ac)) continue;
                cands.push_back({e1.from, e2.to, static_cast<int>(i), static_cast<int>(j)});
            }
        }
        return commit_sorted(cands, rule_idx, r, budget, step);
    }

    // 3. INHERITANCE — (X is_a C) + (C prop Y) => (X prop Y): class properties
    //    flow to members for ANY relation, unlike transitive's matched rels.
    int fire_inheritance(std::size_t rule_idx, float& budget, int step) {
        const Rule& r = rules[rule_idx];    // rel_ab=IS_A, rel_bc=rel_ac=prop
        if (budget < r.cost) return 0;
        std::vector<Cand> cands;
        const std::size_t snap = graph.edges.size();
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e_is_a = graph.edges[i];
            if (e_is_a.rel != REL_IS_A) continue;
            if (e_is_a.to < 0 || static_cast<std::size_t>(e_is_a.to) >= graph.out_adj.size()) continue;
            const std::size_t kn = graph.out_adj[static_cast<std::size_t>(e_is_a.to)].size();
            for (std::size_t k = 0; k < kn; ++k) {
                const std::size_t j = static_cast<std::size_t>(
                    graph.out_adj[static_cast<std::size_t>(e_is_a.to)][k]);
                if (j >= snap) continue;
                const Edge& e_prop = graph.edges[j];
                if (e_prop.rel != r.rel_bc) continue;
                if (graph.has_edge(e_is_a.from, e_prop.to, r.rel_ac)) continue;
                cands.push_back({e_is_a.from, e_prop.to, static_cast<int>(i), static_cast<int>(j)});
            }
        }
        return commit_sorted(cands, rule_idx, r, budget, step);
    }

    // 4. NEGATION — contrapositive: (P implies Q) + (X not Q) => (X not P).
    int fire_negation(std::size_t rule_idx, float& budget, int step) {
        const Rule& r = rules[rule_idx];
        if (budget < r.cost) return 0;
        std::vector<Cand> cands;
        const std::size_t snap = graph.edges.size();
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e_imp = graph.edges[i];
            if (e_imp.rel != r.rel_ab) continue;         // (P implies Q)
            const int P = e_imp.from, Q = e_imp.to;
            for (std::size_t j = 0; j < snap; ++j) {
                const Edge& e_not = graph.edges[j];
                if (e_not.rel != REL_NOT) continue;
                if (e_not.to != Q) continue;
                const int X = e_not.from;
                if (graph.has_edge(X, P, REL_NOT)) continue;
                cands.push_back({X, P, static_cast<int>(i), static_cast<int>(j)});
            }
        }
        return commit_sorted(cands, rule_idx, r, budget, step);
    }

    // 5. CASE_ANALYSIS — disjunction elimination: (X or d_i) for all i, and
    //    every (d_i implies R) => (X implies R). ALL disjuncts must reach R.
    int fire_case_analysis(std::size_t rule_idx, float& budget, int step) {
        const Rule& r = rules[rule_idx];
        if (budget < r.cost) return 0;
        // Group OR edges by source X (std::map: deterministic order).
        std::map<int, std::vector<int>> or_by_X;
        const std::size_t snap = graph.edges.size();
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e = graph.edges[i];
            if (e.rel == REL_OR) or_by_X[e.from].push_back(static_cast<int>(i));
        }
        std::vector<Cand> cands;
        for (const auto& kv : or_by_X) {
            const int X = kv.first;
            const auto& or_edges = kv.second;
            if (or_edges.size() < 2) continue;           // need a real disjunction
            std::map<int, int> r_hits;                   // R -> disjunct coverage
            for (int oe : or_edges) {
                const int d = graph.edges[static_cast<std::size_t>(oe)].to;
                for (std::size_t j = 0; j < snap; ++j) {
                    const Edge& e_imp = graph.edges[j];
                    if (e_imp.rel != r.rel_bc) continue;
                    if (e_imp.from != d) continue;
                    r_hits[e_imp.to] += 1;
                }
            }
            for (const auto& rh : r_hits) {
                if (rh.second < static_cast<int>(or_edges.size())) continue;
                if (graph.has_edge(X, rh.first, r.rel_ac)) continue;
                cands.push_back({X, rh.first, or_edges.front(), -1});
            }
        }
        return commit_sorted(cands, rule_idx, r, budget, step);
    }

    // 6. INDUCTION — schematic generalization: >= min_witnesses members X_i
    //    of class C sharing property (X_i prop Y) => (C prop Y).
    int fire_induction(std::size_t rule_idx, float& budget, int step) {
        const Rule& r = rules[rule_idx];
        const auto& meta = prim_meta[rule_idx];
        const int property = meta.rel_extra;
        if (budget < r.cost) return 0;
        std::map<std::uint64_t, int> counts;             // (C<<32|Y) -> witnesses
        std::map<std::uint64_t, int> witness;            // provenance edge
        const std::size_t snap = graph.edges.size();
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e_is_a = graph.edges[i];
            if (e_is_a.rel != REL_IS_A) continue;
            const int Xi = e_is_a.from, C = e_is_a.to;
            if (Xi < 0 || static_cast<std::size_t>(Xi) >= graph.out_adj.size()) continue;
            const std::size_t kn = graph.out_adj[static_cast<std::size_t>(Xi)].size();
            for (std::size_t k = 0; k < kn; ++k) {
                const std::size_t j = static_cast<std::size_t>(
                    graph.out_adj[static_cast<std::size_t>(Xi)][k]);
                if (j >= snap) continue;
                const Edge& e_prop = graph.edges[j];
                if (e_prop.rel != property) continue;
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(C)) << 32)
                  | static_cast<std::uint32_t>(e_prop.to);
                counts[key] += 1;
                witness[key] = static_cast<int>(i);
            }
        }
        std::vector<Cand> cands;
        for (const auto& kv : counts) {
            if (kv.second < meta.min_witnesses) continue;
            const int C = static_cast<int>(kv.first >> 32);
            const int Y = static_cast<int>(kv.first & 0xffffffffull);
            if (graph.has_edge(C, Y, property)) continue;
            cands.push_back({C, Y, witness[kv.first], -1});
        }
        return commit_sorted(cands, rule_idx, r, budget, step);
    }

    // 7. ANALOGY — structure-preserving edge transfer across a mapping edge
    //    (X --mapping--> X'): every outgoing edge of X transfers to X'.
    int fire_analogy(std::size_t rule_idx, float& budget, int step) {
        const Rule& r = rules[rule_idx];
        if (budget < r.cost) return 0;
        const int mapping = r.rel_ab;
        std::vector<std::pair<int,int>> pairs;
        const std::size_t snap = graph.edges.size();
        for (std::size_t i = 0; i < snap; ++i) {
            const Edge& e = graph.edges[i];
            if (e.rel == mapping) pairs.emplace_back(e.from, e.to);
        }
        std::vector<Cand> cands;
        for (const auto& mp : pairs) {
            const int X = mp.first, Xp = mp.second;
            if (X < 0 || static_cast<std::size_t>(X) >= graph.out_adj.size()) continue;
            const std::size_t kn = graph.out_adj[static_cast<std::size_t>(X)].size();
            for (std::size_t k = 0; k < kn; ++k) {
                const std::size_t j = static_cast<std::size_t>(
                    graph.out_adj[static_cast<std::size_t>(X)][k]);
                if (j >= snap) continue;
                const Edge& e = graph.edges[j];
                if (e.rel == mapping) continue;          // never copy the mapping
                if (e.to == Xp) continue;                // no trivial self-loops
                if (graph.has_edge(Xp, e.to, e.rel)) continue;
                cands.push_back({Xp, e.to, static_cast<int>(j), -1});
            }
        }
        return commit_sorted(cands, rule_idx, r, budget, step);
    }

    // Canonical commit path: sort by (from, to), dedup, then add with budget.
    int commit_sorted(std::vector<Cand>& cands, std::size_t rule_idx,
                      const Rule& r, float& budget, int step) {
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.from != b.from) return a.from < b.from;
            return a.to < b.to;
        });
        cands.erase(std::unique(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) {
                         return a.from == b.from && a.to == b.to; }),
                    cands.end());
        int added = 0;
        for (const auto& c : cands) {
            if (budget < r.cost) break;
            if (graph.has_edge(c.from, c.to, r.rel_ac)) continue;
            graph.add_edge(c.from, c.to, r.rel_ac,
                           static_cast<int>(rule_idx), c.pa, c.pb, step);
            ++added;
            budget -= r.cost;
        }
        return added;
    }

    // The extended orchestrator: every primitive kind fires per pass.
    int apply_all_primitives(float& budget, int step) {
        while (prim_meta.size() < rules.size()) prim_meta.push_back({});
        int added = 0;
        for (std::size_t ri = 0; ri < rules.size(); ++ri) {
            switch (prim_meta[ri].kind) {
                case PRIM_TRANSITIVE:
                case PRIM_DEDUCTION:
                    added += fire_deduction(ri, budget, step);   break;
                case PRIM_INHERITANCE:
                    added += fire_inheritance(ri, budget, step); break;
                case PRIM_NEGATION:
                    added += fire_negation(ri, budget, step);    break;
                case PRIM_CASE_ANALYSIS:
                    added += fire_case_analysis(ri, budget, step); break;
                case PRIM_INDUCTION:
                    added += fire_induction(ri, budget, step);   break;
                case PRIM_ANALOGY:
                    added += fire_analogy(ri, budget, step);     break;
                case PRIM_ABDUCTION:
                case PRIM_RECURSION:
                    break;   // structural: backward chaining / the loop itself
            }
        }
        return added;
    }

    ReasonReport reason_ext() {
        float budget = compute_budget();
        int total = 0, step = 0;
        for (; step < max_steps; ++step) {
            const int add = apply_all_primitives(budget, step + 1);
            if (add == 0) break;
            total += add;
        }
        return {total, step, budget};
    }

    // -- PROOF TRACES (§5.3 query side) --------------------------------------
    // Reconstruct a derivation chain by walking parent edges to premises.
    // Premises first, conclusion last, deduplicated, order-stable.
    std::vector<int> trace(int edge_idx) const {
        std::vector<int> out;
        if (edge_idx < 0 || edge_idx >= static_cast<int>(graph.edges.size())) return out;
        std::vector<int> stack{edge_idx};
        while (!stack.empty()) {
            const int e = stack.back(); stack.pop_back();
            out.push_back(e);
            const Edge& ee = graph.edges[static_cast<std::size_t>(e)];
            if (ee.parent_a >= 0) stack.push_back(ee.parent_a);
            if (ee.parent_b >= 0) stack.push_back(ee.parent_b);
        }
        std::reverse(out.begin(), out.end());
        std::vector<int> seen;
        for (int e : out)
            if (std::find(seen.begin(), seen.end(), e) == seen.end())
                seen.push_back(e);
        return seen;
    }

    // -- QUERY ----------------------------------------------------------------
    struct QueryResult {
        std::string translation;
        bool  holds = false;
        float coherence = 0.0f;      // 1 = both concepts known, 0.5 no-proof
        int   proof_edge = -1;
        std::vector<int> proof;
        int   steps_used = 0;
        bool  ood = false;           // subject or object not in the graph
    };

    QueryResult query(const std::string& subj, int rel, const std::string& obj) const {
        QueryResult r;
        const auto it_s = graph.by_label.find(subj);
        const auto it_o = graph.by_label.find(obj);
        if (it_s == graph.by_label.end() || it_o == graph.by_label.end()) {
            r.holds = false; r.coherence = 0.0f; r.ood = true;
            r.translation = std::string("unknown (out of ontology): ") + subj
                          + " " + rel_name(rel) + " " + obj;
            return r;
        }
        const int e_idx = graph.find_edge(it_s->second, it_o->second, rel);
        r.holds = (e_idx >= 0);
        r.proof_edge = e_idx;
        if (r.holds) {
            r.coherence = 1.0f;
            r.proof = trace(e_idx);
            r.translation = "yes: " + subj + " " + rel_name(rel) + " " + obj;
        } else {
            r.coherence = 0.5f;
            r.translation = "no: " + subj + " " + rel_name(rel) + " " + obj
                          + " does not follow";
        }
        return r;
    }

    QueryResult ask(const std::string& subj, int rel, const std::string& obj) {
        const ReasonReport rep = reason_ext();
        QueryResult q = query(subj, rel, obj);
        q.steps_used = rep.steps_used;
        return q;
    }

    // -- BACKWARD CHAINING (§5.3.2 + adaptive deepening) ----------------------
    // Goal-directed: prove (from --rel--> to) either from an existing edge or
    // by finding an intermediate X where both subgoals prove recursively.
    // Cycle protection via the ACTIVE goal stack (RAII pop — a failed subgoal
    // does not block alternative paths). Adaptive wrapper doubles the depth
    // while the last level still produced real derivations.
    struct Goal { int from; int rel; int to; };

    QueryResult ask_backward(const std::string& subj, int rel,
                             const std::string& obj, int max_depth = 8) {
        QueryResult r;
        const auto it_s = graph.by_label.find(subj);
        const auto it_o = graph.by_label.find(obj);
        if (it_s == graph.by_label.end() || it_o == graph.by_label.end()) {
            r.holds = false; r.coherence = 0.0f; r.ood = true;
            r.translation = std::string("unknown (out of ontology): ") + subj
                          + " " + rel_name(rel) + " " + obj;
            return r;
        }
        std::vector<Goal> visited;
        const int e = prove_backward(it_s->second, rel, it_o->second,
                                     max_depth, visited);
        if (e >= 0) {
            r.holds = true; r.coherence = 1.0f;
            r.proof_edge = e;
            r.proof = trace(e);
            r.translation = "yes: " + subj + " " + rel_name(rel) + " " + obj;
        } else {
            r.holds = false; r.coherence = 0.3f;
            r.translation = "no: " + subj + " " + rel_name(rel) + " " + obj
                          + " does not follow";
        }
        return r;
    }

    QueryResult ask_backward_adaptive(const std::string& subj, int rel,
                                      const std::string& obj,
                                      int hard_ceiling = 1 << 20) {
        int prev_count = static_cast<int>(graph.edges.size());
        QueryResult r;
        int stagnant = 0;
        for (int depth = 2; depth <= hard_ceiling; depth <<= 1) {
            r = ask_backward(subj, rel, obj, depth);
            r.steps_used = depth;
            if (r.holds) return r;
            const int now = static_cast<int>(graph.edges.size());
            if (now == prev_count) {
                if (++stagnant >= 2) return r;   // search exhausted itself
            } else {
                stagnant = 0;
            }
            prev_count = now;
            if (depth >= (1 << 18)) return r;    // runaway guard
        }
        return r;
    }

  private:
    int prove_backward(int from, int rel, int to, int depth,
                       std::vector<Goal>& visited) {
        for (const auto& v : visited)
            if (v.from == from && v.rel == rel && v.to == to) return -1;
        const int base = graph.find_edge(from, to, rel);
        if (base >= 0) return base;
        if (depth <= 0) return -1;
        visited.push_back(Goal{from, rel, to});
        struct PopGuard {
            std::vector<Goal>& v;
            ~PopGuard() { if (!v.empty()) v.pop_back(); }
        } pop_guard{visited};

        for (std::size_t ri = 0; ri < rules.size(); ++ri) {
            const Rule& r = rules[ri];
            if (r.rel_ac != rel) continue;
            for (int X = 0; X < static_cast<int>(graph.nodes.size()); ++X) {
                if (X == from || X == to) continue;
                const int p1 = prove_backward(from, r.rel_ab, X, depth - 1, visited);
                if (p1 < 0) continue;
                const int p2 = prove_backward(X, r.rel_bc, to, depth - 1, visited);
                if (p2 < 0) continue;
                const int e_idx = graph.add_edge(from, to, rel,
                                                 static_cast<int>(ri), p1, p2, depth);
                if (e_idx < 0) return graph.find_edge(from, to, rel);
                return e_idx;
            }
        }
        return -1;
    }

  public:
    // -- PATH FINDING (foundation for L3 Transitive + L6 Analogy; l9 paper) ---
    // BFS over outgoing edges; relation-agnostic by default. Returns edge
    // indices tracing from -> ... -> to (empty = unreachable).
    std::vector<int> find_path(const std::string& from, const std::string& to,
                               int rel_mask = -1, int max_hops = 12) const {
        std::vector<int> path;
        const auto it_a = graph.by_label.find(from);
        const auto it_b = graph.by_label.find(to);
        if (it_a == graph.by_label.end() || it_b == graph.by_label.end()) return path;
        const int a = it_a->second, b = it_b->second;
        if (a == b) return path;
        std::vector<int> came_edge(graph.nodes.size(), -1);
        std::vector<int> came_from(graph.nodes.size(), -1);
        std::vector<char> seen(graph.nodes.size(), 0);
        std::vector<int> depth(graph.nodes.size(), 0);
        std::vector<int> q{a};
        seen[static_cast<std::size_t>(a)] = 1;
        std::size_t head = 0;
        while (head < q.size()) {
            const int u = q[head++];
            if (depth[static_cast<std::size_t>(u)] >= max_hops) continue;
            if (u < 0 || static_cast<std::size_t>(u) >= graph.out_adj.size()) continue;
            for (int ei : graph.out_adj[static_cast<std::size_t>(u)]) {
                const Edge& e = graph.edges[static_cast<std::size_t>(ei)];
                if (rel_mask >= 0 && e.rel != rel_mask) continue;
                const int v = e.to;
                if (seen[static_cast<std::size_t>(v)]) continue;
                seen[static_cast<std::size_t>(v)] = 1;
                came_edge[static_cast<std::size_t>(v)] = ei;
                came_from[static_cast<std::size_t>(v)] = u;
                depth[static_cast<std::size_t>(v)] = depth[static_cast<std::size_t>(u)] + 1;
                if (v == b) {
                    std::vector<int> rev;
                    for (int cur = b; cur != a; cur = came_from[static_cast<std::size_t>(cur)])
                        rev.push_back(came_edge[static_cast<std::size_t>(cur)]);
                    for (auto it = rev.rbegin(); it != rev.rend(); ++it)
                        path.push_back(*it);
                    return path;
                }
                q.push_back(v);
            }
        }
        return path;
    }

    // -- CONTRADICTION GUARD (ALTO auto-proxy; LTO Fix 2) ---------------------
    // Would asserting (subj --rel--> obj) contradict the graph? Conservative:
    // explicit NOT edges either direction, or an is_a subject whose class is
    // NOT-separated from obj. O(1) + O(deg).
    // syfox extension (disclosed): the symmetric case — a NOT claim (X not Y)
    // is refused when the graph already holds the positive edge (X is_a/has Y,
    // or Y is_a X). SI's guard only checked the is_a-claim direction; the
    // inverse is the same class of exact-pair conflict and refuses nothing
    // that a teach correction could not express differently.
    bool would_contradict(const std::string& subj, int rel,
                          const std::string& obj) const {
        const auto it_s = graph.by_label.find(subj);
        const auto it_o = graph.by_label.find(obj);
        if (it_s == graph.by_label.end() || it_o == graph.by_label.end()) return false;
        const int a = it_s->second, b = it_o->second;
        if (graph.has_edge(a, b, REL_NOT) || graph.has_edge(b, a, REL_NOT))
            return true;
        if (rel == REL_NOT &&
            (graph.has_edge(a, b, REL_IS_A) || graph.has_edge(a, b, REL_HAS) ||
             graph.has_edge(b, a, REL_IS_A)))
            return true;
        if (rel == REL_IS_A && a >= 0 &&
            static_cast<std::size_t>(a) < graph.out_adj.size()) {
            for (int ei : graph.out_adj[static_cast<std::size_t>(a)]) {
                const Edge& e1 = graph.edges[static_cast<std::size_t>(ei)];
                if (e1.rel != REL_IS_A) continue;
                const int C = e1.to;
                if (graph.has_edge(C, b, REL_NOT) || graph.has_edge(b, C, REL_NOT))
                    return true;
            }
        }
        return false;
    }

    void print_proof(const QueryResult& q, FILE* f = stdout) const {
        if (!q.holds) { std::fprintf(f, "  (no proof — edge does not hold)\n"); return; }
        std::fprintf(f, "  derivation (%zu steps):\n", q.proof.size());
        for (int e : q.proof) {
            const Edge& ee = graph.edges[static_cast<std::size_t>(e)];
            const char* how = (ee.derived_by_rule < 0)
                ? "premise"
                : (ee.derived_by_rule < static_cast<int>(rules.size())
                   ? rules[static_cast<std::size_t>(ee.derived_by_rule)].name.c_str()
                   : "macro");
            std::fprintf(f, "    [%s] %s --%s--> %s\n", how,
                graph.nodes[static_cast<std::size_t>(ee.from)].label.c_str(),
                rel_name(ee.rel),
                graph.nodes[static_cast<std::size_t>(ee.to)].label.c_str());
        }
    }

};

// ---------------------------------------------------------------------------
// META-LEARNER (meta_learner.hpp §5.3.3) — rule induction.
// Contiguous derivation sub-chains are counted across queries; when a chain
// has saved >= promotion_threshold bits (count * log2(length)) it is promoted
// to a macro-rule that composes the chain into one step. This is the
// operational analogue of RECURSION at the rule-set level.
// ---------------------------------------------------------------------------
struct SubChainKey {
    std::vector<int> rule_ids;
    bool operator==(const SubChainKey& o) const { return rule_ids == o.rule_ids; }
};
struct SubChainKeyHash {
    std::size_t operator()(const SubChainKey& k) const {
        std::size_t h = 1469598103934665603ULL;
        for (int r : k.rule_ids) {
            h ^= static_cast<std::size_t>(r + 0x9E3779B97F4A7C15ULL);
            h *= 1099511628211ULL;
        }
        return h;
    }
};
struct SubChainStat {
    int count = 0;
    int length = 0;
    bool promoted = false;
    int macro_rule_idx = -1;
};
struct MacroRule {
    std::string name;
    std::vector<int> component_rule_ids;
    int rel_ab, rel_bc, rel_ac;
    int uses = 0;
};

struct MetaLearner {
    ReasonCore* engine = nullptr;
    std::unordered_map<SubChainKey, SubChainStat, SubChainKeyHash> chain_stats;
    std::vector<MacroRule> macros;
    float promotion_threshold = 2.0f;   // bits (SI conservative default)

    static float bits_saved_per_use(int chain_length) {
        return static_cast<float>(std::log2(static_cast<double>(std::max(1, chain_length))));
    }

    void observe_trace(const std::vector<int>& derivation_edges) {
        if (!engine) return;
        std::vector<int> rule_ids;
        rule_ids.reserve(derivation_edges.size());
        for (int e : derivation_edges) {
            const Edge& ee = engine->graph.edges[static_cast<std::size_t>(e)];
            if (ee.derived_by_rule >= 0) rule_ids.push_back(ee.derived_by_rule);
        }
        const int n = static_cast<int>(rule_ids.size());
        const int max_len = std::min(n, 6);
        for (int len = 2; len <= max_len; ++len) {
            for (int i = 0; i + len <= n; ++i) {
                SubChainKey key;
                key.rule_ids.assign(rule_ids.begin() + i, rule_ids.begin() + i + len);
                auto& stat = chain_stats[key];
                stat.count += 1;
                stat.length = len;
                if (!stat.promoted) {
                    const float saved = static_cast<float>(stat.count) * bits_saved_per_use(len);
                    if (saved >= promotion_threshold) try_promote(key, stat);
                }
            }
        }
    }

    void try_promote(const SubChainKey& key, SubChainStat& stat) {
        if (key.rule_ids.empty()) return;
        const int first = key.rule_ids.front(), last = key.rule_ids.back();
        if (first < 0 || first >= static_cast<int>(engine->rules.size())) return;
        if (last < 0 || last >= static_cast<int>(engine->rules.size())) return;
        const int rel_ab = engine->rules[static_cast<std::size_t>(first)].rel_ab;
        const int rel_bc = engine->rules[static_cast<std::size_t>(last)].rel_bc;
        const int rel_ac = engine->rules[static_cast<std::size_t>(last)].rel_ac;
        std::string name = "macro[";
        for (std::size_t i = 0; i < key.rule_ids.size(); ++i) {
            if (i) name += " x ";
            name += engine->rules[static_cast<std::size_t>(key.rule_ids[i])].name;
        }
        name += "]";
        const int old_size = static_cast<int>(engine->rules.size());
        engine->add_rule(name, rel_ab, rel_bc, rel_ac, false, 1.0f);
        // add_rule padded the meta slot — overwrite it (macros fire as plain
        // transitive compositions through the standard switch path).
        engine->prim_meta.back() = {PRIM_TRANSITIVE, -1, 3};
        stat.promoted = true;
        stat.macro_rule_idx = old_size;
        macros.push_back(MacroRule{name, key.rule_ids, rel_ab, rel_bc, rel_ac, 0});
    }
};

// VERIFIER (§5.3.4): re-check a candidate answer edge against the rule set,
// recursively down to its premises. A stale derivation fails; Layer 3 must
// emit "unknown", not the stale answer.
inline bool verify_edge(const ReasonCore& si, int edge_idx) {
    if (edge_idx < 0 || edge_idx >= static_cast<int>(si.graph.edges.size())) return false;
    const Edge& e = si.graph.edges[static_cast<std::size_t>(edge_idx)];
    if (e.derived_by_rule < 0) return true;   // a premise is trivially valid
    if (e.derived_by_rule >= static_cast<int>(si.rules.size())) return false;
    const Rule& r = si.rules[static_cast<std::size_t>(e.derived_by_rule)];
    if (e.parent_a < 0 || e.parent_b < 0) return false;
    const Edge& pa = si.graph.edges[static_cast<std::size_t>(e.parent_a)];
    const Edge& pb = si.graph.edges[static_cast<std::size_t>(e.parent_b)];
    if (pa.rel != r.rel_ab) return false;
    if (pb.rel != r.rel_bc) return false;
    if (e.rel  != r.rel_ac) return false;
    if (pa.to  != pb.from) return false;
    if (e.from != pa.from) return false;
    if (e.to   != pb.to)   return false;
    return verify_edge(si, e.parent_a) && verify_edge(si, e.parent_b);
}

// THINKER INDEX (§3.4): C = expected derivation length, R = fraction of
// length-1 (premise) retrievals. High C/R = composing; low = retrieving.
struct ThinkerIndex {
    int total_queries = 0;
    int retrieval_queries = 0;
    double sum_length = 0.0;

    void record(const ReasonCore::QueryResult& r) {
        if (!r.holds) return;
        total_queries++;
        const int L = static_cast<int>(r.proof.size());
        sum_length += static_cast<double>(L);
        if (L <= 1) retrieval_queries++;
    }
    float composition_rate() const {
        return total_queries ? static_cast<float>(sum_length / total_queries) : 0.f;
    }
    float retrieval_rate() const {
        return total_queries ? static_cast<float>(retrieval_queries) / total_queries : 0.f;
    }
    float thinker_index() const {
        const float R = retrieval_rate();
        return composition_rate() / std::max(R + 1e-6f, 1e-6f);
    }
    void report(FILE* f = stdout) const {
        std::fprintf(f,
            "thinker index: C=%.3f  R=%.3f  C/R=%.3f  (%d queries, %d retrievals)\n",
            composition_rate(), retrieval_rate(), thinker_index(),
            total_queries, retrieval_queries);
    }
};

// ---------------------------------------------------------------------------
// THE SIX DISCOVERY QUERIES (inference.hpp) — find what the graph does NOT
// say: ungrounded islands, dangling concepts, the exact missing bridge,
// incomplete symmetries (the Maxwell displacement-current pattern), and
// confused zones.
// ---------------------------------------------------------------------------

// Is this node explicitly fundamental (axiomatic grounding)?
inline bool is_fundamental_node(const ReasonCore& si, int nid) {
    if (nid < 0 || static_cast<std::size_t>(nid) >= si.graph.out_adj.size()) return false;
    for (int ei : si.graph.out_adj[static_cast<std::size_t>(nid)]) {
        const Edge& e = si.graph.edges[static_cast<std::size_t>(ei)];
        if (e.rel == REL_FUNDAMENTAL) return true;
        if (e.rel == REL_IS_A) {
            const std::string& tl = si.graph.nodes[static_cast<std::size_t>(e.to)].label;
            if (tl == "fundamental_law" || tl == "axiom" || tl == "law" ||
                tl == "conservation_law" || tl == "invariant")
                return true;
        }
    }
    return false;
}

inline std::vector<std::vector<int>> undirected_adj(const ReasonCore& si) {
    std::vector<std::vector<int>> adj(si.graph.nodes.size());
    for (const auto& e : si.graph.edges) {
        adj[static_cast<std::size_t>(e.from)].push_back(e.to);
        adj[static_cast<std::size_t>(e.to)].push_back(e.from);
    }
    return adj;
}

// 1. isolated subgraphs — connected components with no fundamental node.
struct IsolatedComponent {
    std::vector<int> nodes;
    bool contains_fundamental = false;
};
inline std::vector<IsolatedComponent> find_isolated_subgraphs(const ReasonCore& si) {
    const auto adj = undirected_adj(si);
    std::vector<int> comp(si.graph.nodes.size(), -1);
    std::vector<IsolatedComponent> comps;
    for (int i = 0; i < static_cast<int>(si.graph.nodes.size()); ++i) {
        if (comp[static_cast<std::size_t>(i)] >= 0) continue;
        IsolatedComponent c;
        std::vector<int> q{i};
        comp[static_cast<std::size_t>(i)] = static_cast<int>(comps.size());
        while (!q.empty()) {
            const int u = q.back(); q.pop_back();
            c.nodes.push_back(u);
            if (is_fundamental_node(si, u)) c.contains_fundamental = true;
            for (int v : adj[static_cast<std::size_t>(u)])
                if (comp[static_cast<std::size_t>(v)] < 0) {
                    comp[static_cast<std::size_t>(v)] = comp[static_cast<std::size_t>(i)];
                    q.push_back(v);
                }
        }
        comps.push_back(std::move(c));
    }
    std::vector<IsolatedComponent> floating;
    for (auto& c : comps)
        if (!c.contains_fundamental) floating.push_back(std::move(c));
    std::sort(floating.begin(), floating.end(),
              [](const IsolatedComponent& a, const IsolatedComponent& b) {
                  return a.nodes.size() > b.nodes.size();
              });
    return floating;
}

// 2. dangling edges — connected nodes with no path (<= K hops) to any
//    fundamental node; ranked by degree.
struct DanglingNode { int node_id; int total_degree; };
inline std::vector<DanglingNode>
find_dangling_edges(const ReasonCore& si, int max_hops = 5, int top_k = 10) {
    std::vector<std::vector<int>> out(si.graph.nodes.size());
    std::vector<int> deg(si.graph.nodes.size(), 0);
    for (const auto& e : si.graph.edges) {
        out[static_cast<std::size_t>(e.from)].push_back(e.to);
        deg[static_cast<std::size_t>(e.from)]++;
        deg[static_cast<std::size_t>(e.to)]++;
    }
    std::vector<DanglingNode> dangling;
    for (int i = 0; i < static_cast<int>(si.graph.nodes.size()); ++i) {
        if (deg[static_cast<std::size_t>(i)] == 0) continue;
        std::vector<int> depth(si.graph.nodes.size(), -1);
        depth[static_cast<std::size_t>(i)] = 0;
        std::vector<int> q{i};
        std::size_t head = 0;
        int hit = -1;
        while (head < q.size() && hit < 0) {
            const int u = q[head++];
            if (depth[static_cast<std::size_t>(u)] >= max_hops) continue;
            for (int v : out[static_cast<std::size_t>(u)]) {
                if (depth[static_cast<std::size_t>(v)] >= 0) continue;
                depth[static_cast<std::size_t>(v)] = depth[static_cast<std::size_t>(u)] + 1;
                if (is_fundamental_node(si, v)) { hit = depth[static_cast<std::size_t>(v)]; break; }
                q.push_back(v);
            }
        }
        if (hit < 0) dangling.push_back({i, deg[static_cast<std::size_t>(i)]});
    }
    std::sort(dangling.begin(), dangling.end(),
              [](const DanglingNode& a, const DanglingNode& b) {
                  return a.total_degree > b.total_degree;
              });
    if (static_cast<int>(dangling.size()) > top_k) dangling.resize(static_cast<std::size_t>(top_k));
    return dangling;
}

// 3. logic gap — when A does not reach B: bidirectional BFS returns the two
//    closest fringe nodes, the exact missing bridge.
struct LogicGap {
    int from_side_node = -1, to_side_node = -1;
    int hops_from_a = -1, hops_from_b_reverse = -1;
    bool full_path_exists = false;
};
inline LogicGap find_logic_gap(const ReasonCore& si, const std::string& a_label,
                               const std::string& b_label, int max_hops = 6) {
    LogicGap g;
    const auto ita = si.graph.by_label.find(a_label);
    const auto itb = si.graph.by_label.find(b_label);
    if (ita == si.graph.by_label.end() || itb == si.graph.by_label.end()) return g;
    const int A = ita->second, B = itb->second;
    if (A == B) return g;
    const int n = static_cast<int>(si.graph.nodes.size());
    std::vector<int> fdepth(n, -1), bdepth(n, -1);
    fdepth[static_cast<std::size_t>(A)] = 0;
    std::vector<int> fq{A};
    for (std::size_t h = 0; h < fq.size(); ++h) {
        const int u = fq[h];
        if (fdepth[static_cast<std::size_t>(u)] >= max_hops) continue;
        if (u < 0 || static_cast<std::size_t>(u) >= si.graph.out_adj.size()) continue;
        for (int ei : si.graph.out_adj[static_cast<std::size_t>(u)]) {
            const int v = si.graph.edges[static_cast<std::size_t>(ei)].to;
            if (fdepth[static_cast<std::size_t>(v)] >= 0) continue;
            fdepth[static_cast<std::size_t>(v)] = fdepth[static_cast<std::size_t>(u)] + 1;
            fq.push_back(v);
        }
    }
    bdepth[static_cast<std::size_t>(B)] = 0;
    std::vector<int> bq{B};
    for (std::size_t h = 0; h < bq.size(); ++h) {
        const int u = bq[h];
        if (bdepth[static_cast<std::size_t>(u)] >= max_hops) continue;
        if (u < 0 || static_cast<std::size_t>(u) >= si.graph.in_adj.size()) continue;
        for (int ei : si.graph.in_adj[static_cast<std::size_t>(u)]) {
            const int v = si.graph.edges[static_cast<std::size_t>(ei)].from;
            if (bdepth[static_cast<std::size_t>(v)] >= 0) continue;
            bdepth[static_cast<std::size_t>(v)] = bdepth[static_cast<std::size_t>(u)] + 1;
            bq.push_back(v);
        }
    }
    int best_meet = -1, best_cost = INT32_MAX;
    for (int v = 0; v < n; ++v) {
        if (fdepth[static_cast<std::size_t>(v)] < 0 || bdepth[static_cast<std::size_t>(v)] < 0) continue;
        const int c = fdepth[static_cast<std::size_t>(v)] + bdepth[static_cast<std::size_t>(v)];
        if (c < best_cost) { best_cost = c; best_meet = v; }
    }
    if (best_meet >= 0 && best_cost > 0) { g.full_path_exists = true; return g; }
    int best_u = -1, best_v = -1, best_score = INT32_MAX;
    for (int u = 0; u < n; ++u) {
        if (fdepth[static_cast<std::size_t>(u)] < 0) continue;
        for (int v = 0; v < n; ++v) {
            if (bdepth[static_cast<std::size_t>(v)] < 0 || u == v) continue;
            const int score = fdepth[static_cast<std::size_t>(u)] + bdepth[static_cast<std::size_t>(v)];
            if (score < best_score) { best_score = score; best_u = u; best_v = v; }
        }
    }
    g.from_side_node = best_u; g.to_side_node = best_v;
    if (best_u >= 0) g.hops_from_a = fdepth[static_cast<std::size_t>(best_u)];
    if (best_v >= 0) g.hops_from_b_reverse = bdepth[static_cast<std::size_t>(best_v)];
    return g;
}

// 4. symmetry gaps — asserted edge (X r Y) where both endpoints have
//    mirror_of partners but the mirror edge is NOT asserted.
struct SymmetryGap { int x, y, x_mirror, y_mirror, r; };
inline std::vector<SymmetryGap> find_symmetry_gaps(const ReasonCore& si) {
    std::unordered_map<int, int> mirror;
    for (const auto& e : si.graph.edges) {
        if (e.rel != REL_MIRROR_OF) continue;
        mirror[e.from] = e.to;
        mirror[e.to] = e.from;
    }
    if (mirror.empty()) return {};
    std::vector<SymmetryGap> gaps;
    for (const auto& e : si.graph.edges) {
        if (e.rel == REL_IS_A || e.rel == REL_HAS || e.rel == REL_MIRROR_OF ||
            e.rel == REL_FUNDAMENTAL || e.rel == REL_PART_OF) continue;
        const auto itx = mirror.find(e.from);
        const auto ity = mirror.find(e.to);
        if (itx == mirror.end() || ity == mirror.end()) continue;
        if (!si.graph.has_edge(itx->second, ity->second, e.rel))
            gaps.push_back({e.from, e.to, itx->second, ity->second, e.rel});
    }
    return gaps;
}

// 5. anomalies — high-degree nodes with weak axiomatic grounding, scoped to
//    components that contain at least one fundamental node.
struct AnomalyNode { int node_id; int total_degree; int hops_to_fundamental; float anomaly_score; };
inline std::vector<AnomalyNode>
find_anomalies(const ReasonCore& si, int max_hops = 6, int top_k = 8) {
    std::vector<int> deg(si.graph.nodes.size(), 0);
    for (const auto& e : si.graph.edges) {
        deg[static_cast<std::size_t>(e.from)]++;
        deg[static_cast<std::size_t>(e.to)]++;
    }
    const auto adj = undirected_adj(si);
    std::vector<int> comp(si.graph.nodes.size(), -1);
    std::vector<bool> comp_has_fund;
    for (int i = 0; i < static_cast<int>(si.graph.nodes.size()); ++i) {
        if (comp[static_cast<std::size_t>(i)] >= 0) continue;
        const int cid = static_cast<int>(comp_has_fund.size());
        comp_has_fund.push_back(false);
        std::vector<int> q{i};
        comp[static_cast<std::size_t>(i)] = cid;
        while (!q.empty()) {
            const int u = q.back(); q.pop_back();
            if (is_fundamental_node(si, u)) comp_has_fund[static_cast<std::size_t>(cid)] = true;
            for (int v : adj[static_cast<std::size_t>(u)])
                if (comp[static_cast<std::size_t>(v)] < 0) {
                    comp[static_cast<std::size_t>(v)] = cid;
                    q.push_back(v);
                }
        }
    }
    std::unordered_set<int> category_target;
    for (const auto& e : si.graph.edges)
        if (e.rel == REL_FUNDAMENTAL) category_target.insert(e.to);
    std::vector<AnomalyNode> out;
    for (int i = 0; i < static_cast<int>(si.graph.nodes.size()); ++i) {
        if (!comp_has_fund[static_cast<std::size_t>(comp[static_cast<std::size_t>(i)])]) continue;
        if (category_target.count(i)) continue;
        if (is_fundamental_node(si, i)) continue;
        const std::string& lbl = si.graph.nodes[static_cast<std::size_t>(i)].label;
        if (lbl == "theory" || lbl == "law" || lbl == "axiom" ||
            lbl == "invariant" || lbl == "conservation_law" || lbl == "fundamental_law")
            continue;
        if (deg[static_cast<std::size_t>(i)] < 2) continue;
        std::vector<int> depth(si.graph.nodes.size(), -1);
        depth[static_cast<std::size_t>(i)] = 0;
        std::vector<int> q{i};
        std::size_t head = 0;
        int hit = -1;
        while (head < q.size() && hit < 0) {
            const int u = q[head++];
            if (depth[static_cast<std::size_t>(u)] >= max_hops) continue;
            if (u < 0 || static_cast<std::size_t>(u) >= si.graph.out_adj.size()) continue;
            for (int ei : si.graph.out_adj[static_cast<std::size_t>(u)]) {
                const int v = si.graph.edges[static_cast<std::size_t>(ei)].to;
                if (depth[static_cast<std::size_t>(v)] >= 0) continue;
                depth[static_cast<std::size_t>(v)] = depth[static_cast<std::size_t>(u)] + 1;
                if (is_fundamental_node(si, v)) { hit = depth[static_cast<std::size_t>(v)]; break; }
                q.push_back(v);
            }
        }
        const float remoteness = (hit < 0) ? 1.0f
            : static_cast<float>(hit) / static_cast<float>(max_hops);
        const float score = static_cast<float>(deg[static_cast<std::size_t>(i)]) * remoteness;
        if (score <= 0.0f) continue;
        out.push_back({i, deg[static_cast<std::size_t>(i)], hit, score});
    }
    std::sort(out.begin(), out.end(),
              [](const AnomalyNode& a, const AnomalyNode& b) {
                  return a.anomaly_score > b.anomaly_score;
              });
    if (static_cast<int>(out.size()) > top_k) out.resize(static_cast<std::size_t>(top_k));
    return out;
}

// 6. classify_gap_query — the discovery front door: is this natural-language
//    line a discovery question, and which kind? (inference.hpp classifier.)
struct GapQuery {
    bool is_query = false;
    int kind = -1;   // 0 logic gap, 1 isolated, 2 dangling, 3 symmetry, 4 anomaly
    std::vector<std::string> terms;
};
inline GapQuery classify_gap_query(const std::string& query) {
    GapQuery g;
    std::string q;
    q.reserve(query.size());
    for (char c : query) q += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto has = [&](const char* kw) { return q.find(kw) != std::string::npos; };
    if (has("symmetr")) g.kind = 3;
    else if (has("isolated") || has("floating")) g.kind = 1;
    else if (has("dangling") || has("dead end") || has("dead-end")) g.kind = 2;
    else if (has("anomal") || has("confused") || has("breakthrough")) g.kind = 4;
    else if (has("missing") || has("gap") || has("unlinked") ||
             has("not connected") || has("disconnected")) g.kind = 0;
    const bool interrogative = has("what") || has("where") || has("any") ||
        has("which") || has("find") || has("show") || has("list") ||
        has("?") || has("is there") || has("are there");
    if (g.kind < 0 || !interrogative) return g;
    g.is_query = true;
    static const char* stop[] = {
        "what", "whats", "where", "which", "any", "is", "are", "there", "the",
        "a", "an", "of", "in", "and", "or", "between", "from", "to", "find",
        "show", "list", "me", "tell", "us", "our", "missing", "gap", "gaps",
        "unlinked", "not", "connected", "disconnected", "isolated", "floating",
        "dangling", "dead", "end", "ends", "symmetry", "symmetries",
        "incomplete", "anomalies", "anomaly", "confused", "breakthrough",
        "graph", "concepts", "concept", "parts", "part", "it", "this", "that",
        "look", "should", "we", "for"};
    std::unordered_set<std::string> stopset(std::begin(stop), std::end(stop));
    std::istringstream iss(q);
    std::string w;
    while (iss >> w) {
        while (!w.empty() && !std::isalnum(static_cast<unsigned char>(w.back()))) w.pop_back();
        while (!w.empty() && !std::isalnum(static_cast<unsigned char>(w.front()))) w.erase(w.begin());
        if (w.size() < 2 || stopset.count(w)) continue;
        g.terms.push_back(w);
    }
    return g;
}

// ---------------------------------------------------------------------------
// AXIOM LOADER (axiom_loader.hpp) — SI's .axioms triple format, unchanged:
//   subject relation object [# comment]
// One triple per line; '#' comments; unknown relations skipped with a
// warning. SI's own data files (syllogism.axioms, insulin_pathway.axioms,
// maxwell_equations.axioms, ...) load unchanged.
// ---------------------------------------------------------------------------
struct AxiomWriteReport {
    bool accepted = false;
    bool duplicate = false;
    bool contradiction = false;
    std::string message;
};

inline AxiomWriteReport teach_axiom(ReasonCore& si, const std::string& subj,
                                    int rel, const std::string& obj,
                                    const std::string& source = "user") {
    AxiomWriteReport r;
    if (subj.empty() || obj.empty() || rel < 0) {
        r.message = "malformed triple";
        return r;
    }
    if (si.would_contradict(subj, rel, obj)) {
        r.contradiction = true;
        r.message = "contradicts an existing axiom (from " + source +
                    "); refused. Delete the conflicting premise first if intentional.";
        return r;
    }
    const auto it_s = si.graph.by_label.find(subj);
    const auto it_o = si.graph.by_label.find(obj);
    if (it_s != si.graph.by_label.end() && it_o != si.graph.by_label.end() &&
        si.graph.has_edge(it_s->second, it_o->second, rel)) {
        r.duplicate = true;
        r.accepted = true;
        r.message = "already known";
        return r;
    }
    si.assert_fact(subj, rel, obj);
    r.accepted = true;
    r.message = "added";
    return r;
}

inline int load_axiom_file(ReasonCore& si, const std::string& path, FILE* log = stderr) {
    std::ifstream in(path);
    if (!in) return 0;
    const bool prev_bulk = si.bulk_load;
    si.bulk_load = true;
    struct BulkGuard { ReasonCore& s; bool prev; ~BulkGuard() { s.bulk_load = prev; } }
        guard{si, prev_bulk};
    int accepted = 0, line_no = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++line_no;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream iss(line);
        std::string subj, rel_tok, obj;
        if (!(iss >> subj >> rel_tok >> obj)) continue;
        const int rel = rel_from_token(rel_tok);
        if (rel < 0) {
            if (log) std::fprintf(log,
                "[axiom_loader] %s:%d unknown relation '%s' — skipped\n",
                path.c_str(), line_no, rel_tok.c_str());
            continue;
        }
        const auto rep = teach_axiom(si, subj, rel, obj, path);
        if (rep.accepted && !rep.duplicate) ++accepted;
        else if (rep.contradiction && log)
            std::fprintf(log, "[axiom_loader] %s:%d %s\n",
                         path.c_str(), line_no, rep.message.c_str());
    }
    if (log) std::fprintf(log, "[axiom_loader] %s: +%d accepted\n",
                          path.c_str(), accepted);
    return accepted;
}

inline bool append_axiom_line(const std::string& path, const std::string& subj,
                              int rel, const std::string& obj) {
    if (rel < 0 || rel >= REL_MAX) return false;
    std::ofstream out(path, std::ios::app);
    if (!out) return false;
    out << subj << ' ' << rel_name(rel) << ' ' << obj << '\n';
    return true;
}

} // namespace sxr
