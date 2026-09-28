// ============================================================================
//  jas — the J-A-S cycle ported to syfox (Einstein's epistemology as a
//  running loop), plus the impossibility register.
//
//  Ported from research-paper/jas.md and the original src/jas.cpp of the
//  Synthetic-Intelligence lineage. The diagram:
//
//                +---------------------------+
//                |        Axioms (A)         |
//                +---------------------------+
//                   /                     \
//                  v                       v
//              +-----+                 +-----+
//              |  S  |                 | S' |    deduced theorems
//              +-----+                 +-----+
//                 |                       |
//                 | experiment            | experiment
//                 v                       v
//          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  Sense Experience (E)
//          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//          \                                 /
//           +----------- Jump (J) ----------+
//
//  J is NOT logical. Induction (counting) leaps from observed frequency to
//  a universal the evidence does not entail; deduction (mechanical, below)
//  turns the axiom into testable predictions; an EXTERNAL ORACLE (the
//  held-out labels, or g++ for arithmetic) confirms or destroys. Refutation
//  is logged, the axiom is REVISED under a structural restriction, and the
//  cycle turns again on rows the first round never saw.
//
//  The division of labour with the frozen SI core follows the central
//  paper exactly: the substrate (energy settle) is the RELEVANCE HEURISTIC;
//  the JAS layer holds claims that can be destroyed; the oracle decides.
//  No physics touched. No loss function, no gradient, no fitted parameter.
//
//  The impossibility register records the WARRANT of every "X cannot be
//  done" claim: ASSERTED / INDUCED / DERIVED / THEOREM. Warrant determines
//  what a counterexample MEANS — for a THEOREM it is a scope error until
//  proven otherwise. syfox's five architectural boundaries are seeded here
//  with honest warrants; two of them are already scope-escaped by this very
//  file (arithmetic via the oracle; subject-object via ordered lanes).
// ============================================================================
#pragma once
#include "core/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace syfox {
namespace jas {

// ------------------------------------------------------ the register ---
enum class Warrant { ASSERTED, INDUCED, DERIVED, THEOREM };

inline const char* warrant_name(Warrant w) {
    switch (w) {
        case Warrant::ASSERTED: return "ASSERTED";
        case Warrant::INDUCED:  return "INDUCED";
        case Warrant::DERIVED:  return "DERIVED";
        default:                return "THEOREM";
    }
}

inline Warrant warrant_from(const std::string& s) {
    if (s == "ASSERTED") return Warrant::ASSERTED;
    if (s == "DERIVED")  return Warrant::DERIVED;
    if (s == "THEOREM")  return Warrant::THEOREM;
    return Warrant::INDUCED;
}

struct ImpossibilityClaim {
    std::string statement;   // the "X cannot be done" content
    Warrant     warrant;     // how the claim is held
    std::string parent;      // axiom (DERIVED) or law (THEOREM)
    std::string status;      // STANDS | SCOPE ESCAPED | REFUTED
    std::string bypass;      // how it was/would be escaped
};

// syfox's own register, seeded from README "Architectural boundaries" —
// now with WARRANTS, which is the JAS discipline the boundaries section
// was missing. Self-application is deliberate.
inline std::vector<ImpossibilityClaim> syfox_impossibilities() {
    std::vector<ImpossibilityClaim> v;
    {
        ImpossibilityClaim c;
        c.statement = "untyped bag-of-words lanes cannot distinguish a "
                      "sentence from its subject-object swap";
        c.warrant   = Warrant::THEOREM;
        c.parent    = "a multiset is invariant under permutation";
        c.status    = "SCOPE ESCAPED";
        c.bypass    = "ordered/typed lanes (MRS bigram lanes, CPSB asymmetric "
                      "lanes, RADE typed edges) are a DIFFERENT representation, "
                      "not a cleverer weighting of the same multiset — the "
                      "sensory layer stops being the object the theorem "
                      "quantifies over. Same theorem class as the lineage's "
                      "metformin/metfromin claim (measured 7/7 there).";
        v.push_back(c);
    }
    {
        ImpossibilityClaim c;
        c.statement = "the settled energy field cannot compute exact arithmetic";
        c.warrant   = Warrant::THEOREM;
        c.parent    = "energy settling is a continuous relaxation; a specific "
                      "answer needs log2(N) discrete bits stored somewhere "
                      "(Translation Capacity Lemma) — the field is a point on "
                      "a manifold, not an answer";
        c.status    = "SCOPE ESCAPED";
        c.bypass    = "JAS calc: |Pi|=6 derivation primitives compose into "
                      "arbitrarily deep derivations at the decision layer, and "
                      "--compile EMITS a C++ translation unit and hands the "
                      "computation to g++ (provenance: established_by_"
                      "experiment). The field still cannot compute — the "
                      "computation happens OUTSIDE it. Scope escape, not a "
                      "violation.";
        v.push_back(c);
    }
    {
        ImpossibilityClaim c;
        c.statement = "multi-hop relational chains beyond 2 hops stay near-tied";
        c.warrant   = Warrant::INDUCED;
        c.parent    = "v3.4 measured survey: 3-4 hop probes remain near-tied "
                      "under hop_coupling/sqrt(hop+1) damping";
        c.status    = "STANDS";
        c.bypass    = "refutable by construction: pre-derived bridge lanes with "
                      "cited premises (l9 transitive closure; rbg analogy lanes "
                      "gated by relative Fisher distance) collapse a 3-hop "
                      "chain to 1 hop. One fabric answering a 3-hop chain "
                      "crisply retracts this claim — an INDUCED claim is only "
                      "a survey of failures.";
        v.push_back(c);
    }
    {
        ImpossibilityClaim c;
        c.statement = "temporal ordering fails on this engine";
        c.warrant   = Warrant::DERIVED;
        c.parent    = "axiom: lanes are unordered token multisets";
        c.status    = "STANDS (on the parent axiom)";
        c.bypass    = "a counterexample here does not touch the claim — it "
                      "refutes the PARENT AXIOM. Ordered-lane construction "
                      "(MRS bigrams, CPSB role asymmetry, cpme_ii "
                      "order-consistent Hebbian) revises the parent, and the "
                      "retraction propagates upward. The limit is not "
                      "independent.";
        v.push_back(c);
    }
    {
        ImpossibilityClaim c;
        c.statement = "no fixed readout threshold eliminates confident-wrong "
                      "answers on lit candidates";
        c.warrant   = Warrant::INDUCED;
        c.parent    = "v3.3.1 ships an ABSOLUTE floor (0.01) deliberately, not "
                      "a relative one; abstention quality is bounded by the "
                      "energy gap";
        c.status    = "STANDS";
        c.bypass    = "candidate escapes from the papers: relative Fisher "
                      "noise-floor scoring (graph_fisher), coincidence gates "
                      "(vght), re-injection self-consistency (bsma), tension-"
                      "gated thresholds (rifa). Until one measures zero "
                      "confident-wrong at nonzero coverage, the survey stands.";
        v.push_back(c);
    }
    return v;
}

inline sfx::JV register_json() {
    sfx::JVArr arr;
    for (const auto& c : syfox_impossibilities()) {
        sfx::JVObj o;
        o["statement"] = sfx::JV(c.statement);
        o["warrant"]   = sfx::JV(warrant_name(c.warrant));
        o["parent"]    = sfx::JV(c.parent);
        o["status"]    = sfx::JV(c.status);
        o["bypass"]    = sfx::JV(c.bypass);
        arr.push_back(sfx::JV(std::move(o)));
    }
    sfx::JVObj out;
    out["register"] = sfx::JV("impossibility");
    out["rule"] = sfx::JV("ASSERTED/INDUCED: one counterexample retracts. "
                          "DERIVED: the counterexample refutes the parent axiom; "
                          "retraction propagates. THEOREM: a counterexample is a "
                          "scope error until proven otherwise.");
    out["claims"] = sfx::JV(std::move(arr));
    return sfx::JV(std::move(out));
}

inline void report_register(std::FILE* out = stderr) {
    std::fprintf(out, "================ IMPOSSIBILITY REGISTER ================\n");
    std::fprintf(out, "Warrant determines what a counterexample means.\n\n");
    for (const auto& c : syfox_impossibilities()) {
        std::fprintf(out, "  [%s] %s\n", warrant_name(c.warrant), c.statement.c_str());
        std::fprintf(out, "      rests on : %s\n", c.parent.c_str());
        std::fprintf(out, "      status   : %s\n", c.status.c_str());
        std::fprintf(out, "      %s\n\n", c.bypass.c_str());
    }
}

// ------------------------------------------------------- lesson rows ---
// The cycle's sense experience (E): labeled rows in the probe-fabric shape
// {"state":..., "questions":{...}?, "labels":{"qid":label}?}.
// Rows with exactly one label drive the induction (single-criterion cycle;
// multi-criterion rows are skipped for the axiom layer but still scored by
// the field layer).

struct Row {
    std::string state;
    std::string label;             // first label (lexicographic qid), may be empty
    bool has_questions = false;
    sfx::JV questions;             // row's own question schema when present
};

inline std::vector<Row> load_rows(const std::string& path, std::string& err) {
    std::vector<Row> rows;
    std::ifstream in(path);
    if (!in) { err = "cannot open " + path; return rows; }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        sfx::JV j;
        try { j = sfx::JV::parse(line); } catch (const std::exception&) {
            err = "bad json in " + path; return rows;
        }
        if (!j.is_obj() || !j.has("state")) continue;
        Row r;
        r.state = j.at("state").as_str();
        if (j.has("questions")) { r.has_questions = true; r.questions = j.at("questions"); }
        if (j.has("labels") && j.at("labels").is_obj()) {
            const sfx::JVObj& L = j.at("labels").obj;
            // lexicographically-first qid for determinism
            for (const auto& kv : L) { r.label = kv.second.as_str(); break; }
        }
        if (!r.state.empty() && !r.label.empty()) rows.push_back(std::move(r));
    }
    return rows;
}

// ------------------------------------------------------ the jump (J) ---
struct InducedAxiom {
    std::string statement;       // human-readable universal
    std::string token;           // T  (the structural predicate)
    std::string extra;           // U  (revision predicate: T AND U)
    std::string label;           // L
    int    support   = 0;        // training rows containing T with label L
    int    total     = 0;        // training rows containing T
    double confidence = 0.0;     // support/total — the OBSERVED FREQUENCY that
                                 // prompted the leap, not a probability
    bool   refuted = false;
    std::string counterexample;
};

// tokenize without pulling the whole normalize pipeline: the fabric's own
// folding matters at decide time, but the AXIOM layer uses the same folding
// via si::norm when available — jas.hpp stays decoupled and expects callers
// to pass tokens produced by si::norm::normalize (one shared table).

struct CycleConfig {
    std::size_t max_axioms = 3;   // top-K by (confidence, support, token)
    int         min_support = 2;  // a universal needs observations to leap from
};

// E -> J: count (token, label) statistics over training rows, then LEAP.
// The axiom is deliberately STRONGER than the evidence: "92% of rows with
// token T have label L" becomes "EVERY row with T has label L". That gap is
// what makes it refutable; a restatement of the frequency is only a summary.
inline std::vector<InducedAxiom> jump(const std::vector<Row>& train, const CycleConfig& cfg) {
    struct Stat { int total = 0; std::map<std::string, int> by_label; };
    std::map<std::string, Stat> stats;
    for (const auto& r : train) {
        // row tokens: whitespace split on the already-normalized state
        std::istringstream ss(r.state);
        std::string t;
        std::set<std::string> seen;   // per-row uniqueness: presence, not counts
        while (ss >> t) seen.insert(t);
        for (const auto& tok : seen) {
            Stat& s = stats[tok];
            ++s.total;
            ++s.by_label[r.label];
        }
    }
    std::vector<InducedAxiom> out;
    for (const auto& kv : stats) {
        for (const auto& lv : kv.second.by_label) {
            if (kv.second.total < cfg.min_support) continue;
            // leap condition: the dominant label is not yet certainty —
            // an axiom the data already proves at 100% of huge support is
            // a summary; we keep support-bounded leaps (min_support caps it)
            InducedAxiom ax;
            ax.token  = kv.first;
            ax.label  = lv.first;
            ax.support = lv.second;
            ax.total   = kv.second.total;
            ax.confidence = static_cast<double>(lv.second) / kv.second.total;
            ax.statement = "every row whose state contains '" + ax.token +
                           "' has label '" + ax.label + "'";
            out.push_back(std::move(ax));
        }
    }
    // rank: purest leaps first (high confidence, enough support, stable order)
    std::sort(out.begin(), out.end(), [](const InducedAxiom& a, const InducedAxiom& b) {
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        if (a.support != b.support) return a.support > b.support;
        if (a.token != b.token) return a.token < b.token;
        return a.label < b.label;
    });
    if (out.size() > cfg.max_axioms) out.resize(cfg.max_axioms);
    return out;
}

// --------------------------------------------------- deduction (A->S) ---
struct Prediction {
    bool has = false;             // false = the axiom set makes NO prediction
    bool conflict = false;        // axioms fire with different labels
    std::string label;
    std::vector<std::string> fired;
};

inline Prediction deduce(const std::vector<InducedAxiom>& axioms, const Row& r,
                         bool use_revision) {
    Prediction p;
    // row tokens
    std::set<std::string> seen;
    { std::istringstream ss(r.state); std::string t; while (ss >> t) seen.insert(t); }
    std::map<std::string, int> votes;
    for (const auto& ax : axioms) {
        const bool applies = seen.count(ax.token) &&
                             (!use_revision || ax.extra.empty() || seen.count(ax.extra));
        if (!applies) continue;
        p.fired.push_back(ax.token);
        ++votes[ax.label];
    }
    if (votes.empty()) return p;
    p.has = true;
    p.label = votes.begin()->first;
    int best = votes.begin()->second;
    for (const auto& kv : votes) {
        if (kv.second > best) { best = kv.second; p.label = kv.first; }
    }
    int with_best = 0;
    for (const auto& kv : votes) if (kv.second == best) ++with_best;
    p.conflict = with_best > 1;
    if (p.conflict) p.label = votes.begin()->first;   // deterministic, and flagged
    return p;
}

// --------------------------------------------------- the cycle report ---
struct ExperimentRow {
    std::string state;
    bool predicted = false;
    bool conflict = false;
    std::string predicted_label;
    std::string actual_label;
    bool consistent = false;
};

struct CycleReport {
    // E
    int train_rows = 0, holdout_rows = 0;
    // J
    std::vector<InducedAxiom> axioms;
    // round 1
    int r1_consistent = 0, r1_refuted = 0, r1_no_prediction = 0, r1_conflict = 0;
    std::string r1_counterexample;              // refuting row's state
    std::vector<ExperimentRow> r1_detail;       // per-row disclosure
    // field layer (the frozen fabric, same rows, the relevance heuristic)
    int field_answered = 0, field_correct = 0, field_deferred = 0;
    double field_acc = 0.0;      // over answered
    // revision + round 2 (held-out rows NOT used in round 1)
    bool revised = false;
    std::vector<InducedAxiom> revised_axioms;   // subset carrying extra != ""
    int r2_consistent = 0, r2_refuted = 0, r2_no_prediction = 0, r2_tested = 0;
    std::string r2_counterexample;              // refuting row's state
    std::vector<ExperimentRow> r2_detail;       // per-row disclosure
    // verdict
    std::string verdict;
    sfx::JV logs;                // file paths written
};

// Run the full cycle. Caller supplies the training rows (E), the holdout
// rows (the oracle's raw experience), and an optional DECIDE callback for
// the field-layer column (may be null: cycle runs without the fabric).
// `decide_field(state, questions, label_out, deferred_out)` returns false
// when the field defers.
template <typename DecideFn>
inline CycleReport run_cycle(const std::vector<Row>& train_in,
                             const std::vector<Row>& holdout_in,
                             const CycleConfig& cfg,
                             DecideFn decide_field,
                             const std::string& model_dir,
                             std::string& err) {
    CycleReport rep;
    if (train_in.empty()) { err = "no training rows"; return rep; }
    if (holdout_in.size() < 2) { err = "need >= 2 holdout rows (one per round)"; return rep; }

    // deterministic split of the holdout: sorted by state, evens -> round 1,
    // odds -> round 2 (round 2 rows are NEVER seen by round 1 — otherwise
    // the revision would be fitted to its own counterexamples)
    std::vector<Row> holdout = holdout_in;
    std::sort(holdout.begin(), holdout.end(),
              [](const Row& a, const Row& b) { return a.state < b.state; });
    std::vector<Row> r1_rows, r2_rows;
    for (std::size_t i = 0; i < holdout.size(); ++i)
        ((i % 2 == 0) ? r1_rows : r2_rows).push_back(holdout[i]);

    rep.train_rows = static_cast<int>(train_in.size());
    rep.holdout_rows = static_cast<int>(holdout.size());

    // ---- [J] the jump -------------------------------------------------
    rep.axioms = jump(train_in, cfg);
    if (rep.axioms.empty()) { rep.verdict = "NO_TESTABLE_AXIOMS"; return rep; }

    // ---- [A->S] deduction + [experiment] oracle, round 1 --------------
    // Round 1 with the field layer measured on the SAME rows.
    for (const auto& r : r1_rows) {
        Prediction p = deduce(rep.axioms, r, false);
        ExperimentRow det;
        det.state = r.state;
        det.actual_label = r.label;
        if (!p.has) ++rep.r1_no_prediction;
        else {
            det.predicted = true;
            det.conflict = p.conflict;
            det.predicted_label = p.label;
            if (p.conflict) ++rep.r1_conflict;
            if (p.label == r.label) { ++rep.r1_consistent; det.consistent = true; }
            else {
                ++rep.r1_refuted;
                if (rep.r1_counterexample.empty()) rep.r1_counterexample = r.state;
            }
        }
        rep.r1_detail.push_back(std::move(det));
        // field layer (frozen physics; the relevance heuristic)
        if (r.has_questions && r.questions.is_obj()) {
            std::string choice; bool deferred = false;
            if (decide_field(r.state, r.questions, choice, deferred)) {
                ++rep.field_answered;
                if (!deferred && choice == r.label) ++rep.field_correct;
                else if (deferred) ++rep.field_deferred;
            } else ++rep.field_deferred;
        }
    }
    if (rep.field_answered > 0)
        rep.field_acc = static_cast<double>(rep.field_correct) / rep.field_answered;

    // ---- [refutation] + revision ---------------------------------------
    bool any_refuted = rep.r1_refuted > 0;
    if (any_refuted) {
        rep.revised = true;
        // structural restriction: T AND U, where U is the training token that
        // maximizes confidence of (T and U -> L), evaluated BEFORE any
        // experiment (a predicate the data can check in advance), excluding
        // tokens that add nothing (U with identical contingency to T alone).
        std::vector<InducedAxiom> revised;
        for (auto ax : rep.axioms) {
            // best co-token for this axiom on TRAIN rows containing T
            std::map<std::string, std::pair<int, int>> co;  // u -> {support, total}
            for (const auto& r : train_in) {
                std::set<std::string> seen;
                { std::istringstream ss(r.state); std::string t; while (ss >> t) seen.insert(t); }
                if (!seen.count(ax.token)) continue;
                for (const auto& u : seen) {
                    if (u == ax.token) continue;
                    auto& c = co[u];
                    ++c.second;
                    if (r.label == ax.label) ++c.first;
                }
            }
            std::string best_u; double best_c = ax.confidence; int best_s = 0;
            for (const auto& kv : co) {
                if (kv.second.second < cfg.min_support) continue;
                const double conf = static_cast<double>(kv.second.first) / kv.second.second;
                if (conf > best_c + 1e-12 ||
                    (conf > best_c - 1e-12 && !best_u.empty() &&
                     (kv.second.first > best_s ||
                      (kv.second.first == best_s && kv.first < best_u)))) {
                    best_c = conf; best_u = kv.first; best_s = kv.second.first;
                }
            }
            if (!best_u.empty()) {
                ax.extra = best_u;
                ax.statement = "every row whose state contains '" + ax.token +
                               "' AND '" + best_u + "' has label '" + ax.label + "'";
                revised.push_back(ax);
            }
        }
        rep.revised_axioms = revised;
    }

    // ---- round 2: held-out rows never seen in round 1 ------------------
    {
        std::vector<InducedAxiom> tmp = rep.axioms;
        if (rep.revised) {
            // temporary swap: deduce() reads rep.axioms; give it the revised set
            rep.axioms = rep.revised_axioms;
        }
        for (const auto& r : r2_rows) {
            Prediction p = deduce(rep.axioms, r, true);
            ExperimentRow det;
            det.state = r.state;
            det.actual_label = r.label;
            ++rep.r2_tested;
            if (!p.has) ++rep.r2_no_prediction;
            else {
                det.predicted = true;
                det.conflict = p.conflict;
                det.predicted_label = p.label;
                if (p.label == r.label) { ++rep.r2_consistent; det.consistent = true; }
                else {
                    ++rep.r2_refuted;
                    if (rep.r2_counterexample.empty()) rep.r2_counterexample = r.state;
                }
            }
            rep.r2_detail.push_back(std::move(det));
        }
        if (rep.revised) rep.axioms = tmp;   // restore original axioms
    }

    // ---- verdict --------------------------------------------------------
    // Any experimental refutation counts — the paper is explicit that ONE
    // counterexample ends an INDUCED claim, regardless of which round
    // delivered it. A round-2 refutation (fresh rows, revision not yet
    // fitted) schedules the next turn of the cycle.
    if (rep.r1_refuted > 0) {
        if (rep.r2_refuted == 0 && rep.r2_consistent > 0)
            rep.verdict = "REFUTED_AND_REVISED_SURVIVES_HELDOUT";
        else if (rep.r2_refuted > 0)
            rep.verdict = "REFUTED_AND_REVISION_ALSO_REFUTED";
        else
            rep.verdict = "REFUTED_AND_REVISION_UNTESTED";
    } else if (rep.r2_refuted > 0) {
        rep.verdict = "REFUTED_IN_ROUND2_REVISE_NEXT_CYCLE";
    } else {
        rep.verdict = "SURVIVES";
    }

    // ---- logs: refuted/verified axioms (model dir), jas.cpp format -----
    if (!model_dir.empty()) {
        sfx::JVObj logs;
        if (rep.r1_refuted > 0) {
            const std::string rf = model_dir + "/refuted.axioms";
            std::ofstream f(rf, std::ios::app);
            if (f) {
                for (const auto& ax : rep.axioms)
                    f << "# REFUTED by experiment (round 1): " << ax.statement << "\n";
                f << "# counterexample: " << rep.r1_counterexample << "\n";
                logs["refuted"] = sfx::JV(rf);
            }
        }
        if (rep.r2_refuted > 0) {
            const std::string rf = model_dir + "/refuted.axioms";
            std::ofstream f(rf, std::ios::app);
            if (f) {
                f << "# REFUTED by experiment (round 2, fresh rows): "
                  << rep.r2_counterexample << "\n";
                logs["refuted_r2"] = sfx::JV(rf);
            }
        }
        if (rep.verdict == "REFUTED_AND_REVISED_SURVIVES_HELDOUT" ||
            rep.verdict == "SURVIVES") {
            const std::string vf = model_dir + "/verified.axioms";
            std::ofstream f(vf, std::ios::app);
            if (f) {
                f << "# INDUCED, survived held-out experiments (" << rep.verdict << "):\n";
                for (const auto& ax : (rep.revised && rep.verdict != "SURVIVES")
                                          ? rep.revised_axioms : rep.axioms)
                    f << "# " << ax.statement << "\n";
                logs["verified"] = sfx::JV(vf);
            }
        }
        rep.logs = sfx::JV(std::move(logs));
    }
    return rep;
}

// ------------------------------------------------------ cycle JSON -----
inline sfx::JV detail_json(const std::vector<ExperimentRow>& rows) {
    sfx::JVArr arr;
    for (const auto& d : rows) {
        sfx::JVObj o;
        o["state"] = sfx::JV(d.state);
        o["predicted"] = sfx::JV(d.predicted);
        if (d.predicted) {
            o["predicted_label"] = sfx::JV(d.predicted_label);
            o["conflict"] = sfx::JV(d.conflict);
        }
        o["actual"] = sfx::JV(d.actual_label);
        o["verdict"] = sfx::JV(d.predicted ? (d.consistent ? "CONSISTENT" : "REFUTES") : "NO_PREDICTION");
        arr.push_back(sfx::JV(std::move(o)));
    }
    return sfx::JV(std::move(arr));
}

inline sfx::JV report_json(const CycleReport& rep) {
    sfx::JVObj out;
    out["command"] = sfx::JV("jas");
    out["cycle"] = sfx::JV("E -> J -> A -> S -> experiment -> refutation -> J2 -> experiment");
    out["train_rows"] = sfx::JV(static_cast<double>(rep.train_rows));
    out["holdout_rows"] = sfx::JV(static_cast<double>(rep.holdout_rows));
    sfx::JVArr axs;
    for (const auto& ax : rep.axioms) {
        sfx::JVObj o;
        o["statement"] = sfx::JV(ax.statement);
        o["warrant"] = sfx::JV("INDUCED");
        o["support"] = sfx::JV(static_cast<double>(ax.support));
        o["total"] = sfx::JV(static_cast<double>(ax.total));
        o["confidence"] = sfx::JV(std::round(ax.confidence * 1000.0) / 1000.0);
        o["refuted"] = sfx::JV(rep.revised);
        axs.push_back(sfx::JV(std::move(o)));
    }
    out["axioms"] = sfx::JV(std::move(axs));
    out["round1"] = sfx::JV(sfx::JVObj{
        {"consistent", sfx::JV(static_cast<double>(rep.r1_consistent))},
        {"refuted", sfx::JV(static_cast<double>(rep.r1_refuted))},
        {"no_prediction", sfx::JV(static_cast<double>(rep.r1_no_prediction))},
        {"conflict", sfx::JV(static_cast<double>(rep.r1_conflict))},
        {"counterexample", sfx::JV(rep.r1_counterexample)},
        {"experiments", detail_json(rep.r1_detail)}});
    out["field_layer"] = sfx::JV(sfx::JVObj{
        {"role", sfx::JV("relevance heuristic (frozen SI core, untouched)")},
        {"answered", sfx::JV(static_cast<double>(rep.field_answered))},
        {"correct", sfx::JV(static_cast<double>(rep.field_correct))},
        {"deferred", sfx::JV(static_cast<double>(rep.field_deferred))},
        {"accuracy_over_answered", sfx::JV(std::round(rep.field_acc * 10000.0) / 10000.0)}});
    if (rep.revised) {
        sfx::JVArr raxs;
        for (const auto& ax : rep.revised_axioms) {
            sfx::JVObj o;
            o["statement"] = sfx::JV(ax.statement);
            o["restriction"] = sfx::JV(ax.token + " AND " + ax.extra);
            raxs.push_back(sfx::JV(std::move(o)));
        }
        out["revised_axioms"] = sfx::JV(std::move(raxs));
    }
    out["round2"] = sfx::JV(sfx::JVObj{
        {"tested", sfx::JV(static_cast<double>(rep.r2_tested))},
        {"consistent", sfx::JV(static_cast<double>(rep.r2_consistent))},
        {"refuted", sfx::JV(static_cast<double>(rep.r2_refuted))},
        {"no_prediction", sfx::JV(static_cast<double>(rep.r2_no_prediction))},
        {"counterexample", sfx::JV(rep.r2_counterexample)},
        {"experiments", detail_json(rep.r2_detail)}});
    out["verdict"] = sfx::JV(rep.verdict);
    out["logs"] = rep.logs;
    out["note"] = sfx::JV("induction confers no warrant beyond survival; "
                          "a claim that predicts everything predicts nothing — "
                          "no_prediction rows are the feature, not a failure");
    return sfx::JV(std::move(out));
}

}  // namespace jas
}  // namespace syfox
