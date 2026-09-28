// ============================================================================
//  SyFox — syfox.hpp
//  The System One layer: turns SI substrate field dynamics into typed
//  decisions (Choice / Score / Noul) with a Jev-compatible call shape.
//
//  Layering contract:
//    * CORE (si::Substrate)  = SI physics only. Injection, settle, resonance
//      sweep readout, Hebbian lanes, honest silence. No classifiers, no
//      transformers, no pattern-matching algorithms.
//    * THIS LAYER            = typed question plumbing + persistence +
//      a calibration TOOL that post-processes readout energies into
//      probabilities (allowed as an external tool; it never rewrites the core).
//
//  All questions are evaluated against ONE settled field, in isolation —
//  adding questions does not re-settle the substrate (Jev's flat-latency
//  property, inherited structurally).
// ============================================================================
#pragma once
#include "json.hpp"
#include "si_substrate.hpp"
#include "reason.hpp"
#include "normalize.hpp"
#include "ngram.hpp"
#include "recall.hpp"
#include "roles.hpp"
#include "distvec.hpp"
#include "calc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace syfox {

inline const char* VERSION = "3.9.0";

// ---------------------------------------------------------------------------
// v2.2 boundary injection protocol — sub-word bridges for corrupted forms.
// Words inject at the substrate's own inject_energy; the trigram bridges of
// an unknown word inject ONLY when that word passes the sub-word traction
// gate below (a corrupted form of something taught). A boundary protocol,
// not core physics: si_substrate.hpp is untouched.
// ---------------------------------------------------------------------------
inline constexpr float kBridgeEnergy = 1.0f;

// v3.3.1 — readout-silence floor for unknown_candidates (ABSOLUTE, Option B).
// Defer a choice/score question when the BEST candidate's readout energy is
// below this: the settled field carries no measurable signal on any option.
// 0.01 sits three orders below valid dim candidates (0.19 measured without
// energy-norm) and well above float dust from diffusion carryover. Documented
// in README (honest-silence section); debug visibility via SYFOX_DEBUG_READOUT=1.
inline constexpr float kUnknownCandidateFloor = 0.01f;
// v3.6.0: decide-side bigram injection dose as a fraction of the state dose
// (MRS §4b grounds bigram lanes like any other lane; the 0.5 fraction keeps
// order evidence subordinate to lexical evidence).
inline constexpr float kBigramDose = 0.5f;
// v3.7.0: decide-side TYPED pair injection dose as a fraction of the state
// dose (same grounding argument as the bigrams; role evidence is another
// subordinate channel, not the headline).
inline constexpr float kTypedDose = 0.5f;
// v3.8.0 (TASK 3): decide-side CAUSAL pair injection dose as a fraction of
// the state dose (same pattern as the bigram and typed doses).
inline constexpr float kCausalDose = 0.5f;
// v3.7.0 (Move 2): pretrain error-weighting cap. The masked-basin loss is
// in (0,1]; a lane that failed to light its basin binds at up to full
// Hebbian strength, a basin that already wins binds at the residual.
inline constexpr float kPretrainEtaCap = 1.0f;

// Sub-word traction gate: an UNKNOWN word earns trigram bridges only when a
// meaningful fraction of its trigrams already exist in the fabric — the
// signature of a CORRUPTED FORM of something taught ("refnd" keeps g3:ref
// of "refund"). A legitimate unseen word ("webhook" never taught) shares
// nothing and stays fully dark: honest silence, no manufactured evidence.
// Measured (v2.2 A/B, README): ungated bridges regress clean heldout
// (tickets 1.000 -> 0.833/0.500 depending on bridge level) because unseen
//-but-relevant words bridge into unrelated labels; with this gate, clean
// behavior returns to baseline while typo routing is retained.
inline bool subword_traction(const si::Substrate& s, const std::string& w) {
    const auto grams = si::norm::expand_ngrams({w});
    if (grams.empty()) return false;
    std::size_t known = 0;
    for (const auto& g : grams) if (s.has(g)) ++known;
    return known * 4 >= grams.size();                    // >= 25% of trigrams known
}

// ---------------------------------------------------------------------------
// v3 Milestone 2 — the distinct-experience policy (boundary layer, flag-gated).
// Measured twice (v2.1 paraphrases, v2.2 variants): re-teaching near-
// duplicate lessons REGRESSED held-out accuracy, while distinct experience
// scales (the v3 dose-response probe: tickets hidden 0.603@250 -> 0.825@8400).
// The fix lives in the TEACHING dose, not in the physics:
//   * dedup   — an exact duplicate lesson (same normalized token stream,
//               same outcome) carries ZERO new information: skip it.
//   * novelty — a lesson's Hebbian dose scales with how much of its state
//               vocabulary the fabric has never seen: eta_scale =
//               floor + (1-floor) * n_new_words / n_words. A fresh sentence
//               binds at full strength; a rephrasing of known words binds
//               at 25% — enough to track drift, not enough to smear lanes.
// ---------------------------------------------------------------------------
struct LearnPolicy {
    bool  dedup = false;
    bool  novelty = false;
    float novelty_floor = 0.25f;
    // v3.9.0 (CME): graded fingerprint engagement — a re-taught example is
    // not skipped outright (that is --dedup) but consolidated at HALF dose
    // with its hit_count bumped, so duplicates stop over-strengthening lanes
    // without pretending they were never seen. Capped table (1024),
    // deterministic eviction (lowest hit_count, then oldest first_seq).
    bool  fingerprints = false;
};

// ---------------------------------------------------------------------------
struct Answer {
    std::string qid;
    std::string type;                                    // "choice" | "score" | "noul"
    std::string choice;                                  // choice only
    float value = 0.0f;                                  // score only (weighted level)
    float probability = 0.0f;                            // noul only
    std::vector<std::pair<std::string, float>> probabilities;  // choice/score
    float confidence = 0.0f;
    bool deferred = false;
    std::string reason;                                  // honest-silence reasons
    // v3.3 readout disclosures: an EXACT top-2 probability tie (the readout
    // carries no signal between the top two; the pick is deterministic
    // criteria order), and the question tokens that gated the readout when
    // question-conditioned gating fired.
    bool tied = false;
    std::vector<std::string> gate_tokens;
};

struct Usage {
    std::size_t state_tokens = 0;
    std::size_t vocabulary = 0;
    std::size_t lanes = 0;
    int settle_passes = 0;
    float settled_energy = 0.0f;
    bool calibrated = false;
    // v3.2 retrieval-by-default: the memories that primed this decision
    // (label, resonance) pairs, strongest first; empty when retrieval off.
    std::vector<std::pair<std::string, float>> retrieved;
    // v3.4 question-context gate disclosure: true when the two-stage settle
    // composed the field with the settled question field at ctx_alpha.
    bool ctx_gate = false;
    float ctx_alpha = 0.0f;
    // v3.6.0 disclosures: bigram nodes the state actually energized; and the
    // BED §8 perturbation-contrast check (opt-in) with its two margins —
    // real structure vs adjacent-transposed bag of the same tokens.
    std::size_t bigram_tokens = 0;
    bool perturb_check = false;
    float perturb_margin_real = 0.0f;
    float perturb_margin_broken = 0.0f;
    // v3.7.0 disclosures: typed pair nodes energized (Move 3); and the
    // generalized tool scope (Move 4): when --tools runs the arithmetic
    // relevance heuristic and the state asks for a computation, the ORACLE
    // result is disclosed here while the field still answers (or defers) as
    // it measures — the register records that the refusal would be principled.
    std::size_t typed_tokens = 0;
    bool tool_checked = false;
    bool tool_used = false;
    std::string tool_name;
    std::string tool_expression;
    std::string tool_value;
    // v3.8.0 TASK 4 (EBT lineage) — energy self-verification, ALWAYS disclosed:
    // the settle's own convergence trace. motion_rate is the last pass's
    // relative energy movement (decay baseline ~0.18; structure still
    // travelling shows above it). The OPT-IN self-verify gate defers choice
    // answers measured on a still-moving field.
    bool  field_converged = false;   // true when the eps break fired
    float field_motion_rate = 0.0f;  // last-pass relative motion
    int   field_passes = 0;          // settle passes executed
    // v3.8.0 TASK 1 (Thinking-Without-Tokens lineage) — adaptive depth:
    // difficulty estimated from the state (unknown-token fraction + lane
    // isolation), settle depth k = base + round(difficulty * max_extra).
    // Both disclosed; the physics per pass is byte-identical.
    bool  adaptive_depth = false;
    float difficulty = 0.0f;
    int   k_used = 0;
    int   k_base = 0;
    // v3.8.0 TASK 2 (CSM lineage) — multi-vector spectral channels:
    bool  multi_vector_active = false;
    int   multi_vector_channels = 0;
    // v3.8.0 TASK 3 — causal pair nodes the state energized:
    std::size_t causal_tokens = 0;
    // v3.9.0 — the constructive substrate + decision-layer paper gates.
    // proof: RADE proof-pass over the typed axiom graph between the state's
    //        strongest concept and the winning anchor (premise-cited hops).
    bool  proof_checked = false;
    std::vector<std::string> proof_path;      // "subject relation object" hops
    // momentum (MCPE/CPME): fraction of candidates still GAINING energy at
    // the final settle pass (the echo-suppressed remainder peaked already).
    bool  momentum_gate = false;
    float momentum_eligible_frac = 0.0f;
    // wavefront (CMD §4): fraction of candidates the query actually reached.
    bool  wavefront_gate = false;
    float wavefront_frac = 0.0f;
    // coherence (HTR ΔR): participation-ratio of candidate energies vs the
    // whole energized field — structural rise above baseline drift.
    bool  coherence_gate = false;
    float coherence_r = 0.0f;
    float coherence_field_r = 0.0f;
    // vght (VGHT §2): fraction of walked lanes that passed the coincidence gate.
    bool  vght_gate = false;
    float vght_gated_frac = 0.0f;
    // cem (P-CMA shell): planning rounds over hash-derived injection variants.
    bool  cem_active = false;
    int   cem_rounds = 0;
    float cem_gain = 0.0f;                    // best variant margin - baseline
    // maa (MAA): unknown tokens bridged by fragment abduction this decision.
    std::vector<std::string> hyp_tokens;
    // ood-knn (fixes.md): signature distance to the stored answered states.
    bool  ood_knn = false;
    float ood_knn_dist = 0.0f;
    // rifa (SIL): accumulated contradiction tension disclosed per decision.
    bool  tension_gate = false;
    float tension = 0.0f;
};

struct Calibration {
    float choice_temperature = 1.0f;
    float score_temperature  = 1.0f;
    float noul_a = 6.0f;                                 // p = sigmoid(a*support + b)
    float noul_b = -3.0f;
    bool  fitted = false;
    std::size_t rows = 0;

    sfx::JV to_json() const {
        return sfx::JV(sfx::JVObj{
            {"choice_temperature", choice_temperature},
            {"score_temperature", score_temperature},
            {"noul_a", noul_a}, {"noul_b", noul_b},
            {"fitted", fitted}, {"rows", static_cast<double>(rows)}});
    }
    static Calibration from_json(const sfx::JV& j) {
        Calibration c;
        c.choice_temperature = static_cast<float>(j.at("choice_temperature").as_num(1.0));
        c.score_temperature  = static_cast<float>(j.at("score_temperature").as_num(1.0));
        c.noul_a = static_cast<float>(j.at("noul_a").as_num(6.0));
        c.noul_b = static_cast<float>(j.at("noul_b").as_num(-3.0));
        c.fitted = j.at("fitted").is_bool() && j.at("fitted").b;
        c.rows   = static_cast<std::size_t>(j.at("rows").as_num(0));
        return c;
    }
};

// ---------------------------------------------------------------------------
class Engine {
public:
    explicit Engine(si::SubstrateConfig cfg = {})
        : si_(cfg), adaptive_base_k_(cfg.k_settle) {
        // v3.9.0: the constructive substrate ships with SI's default rule set
        // (taxonomy flow, modus ponens, implies/causes chains, class-property
        // inheritance, contrapositive, disjunction elimination, schematic
        // generalization) — the nine primitives are ready before any axioms
        // load; the graph starts empty (reason_loaded_ stays false until a
        // model carries axioms.txt, so pre-3.9 models replay bit-identically).
        reason_.add_default_rules();
        reason_meta_.engine = &reason_;
    }

    si::Substrate& substrate() { return si_; }
    const si::Substrate& substrate() const { return si_; }
    Calibration& calibration() { return calib_; }

    // -- v3 Milestone 1: decide-side energy normalization (boundary GAIN knob) --
    // At seed scale (tens of lessons) acoustic mass is small and a state's
    // settled field sits comfortably above the silence floor. At corpus scale
    // (8,400 lessons) common words carry mass in the hundreds, the damping law
    // energy/sqrt(mass) whispers, and honest silence (absolute floor 0.05)
    // defers most states — coverage collapses to the rows that happen to
    // contain rare tokens (measured: 33/40 hidden rows deferred).
    // The fix is a measurement gain, NOT a physics change: the injection dose
    // is scaled by the mean sqrt(mass) of the state's KNOWN tokens, which
    // preserves the damping law's per-token ratios exactly (uniform mass m
    // reproduces the seed-scale total n*inject_energy bit-for-bit up to the
    // scale factor). si_substrate.hpp is untouched; the silence floor stays
    // absolute; unknown vocabulary still injects nothing and still defers.
    // Off by default: every v2.1/v2.2 number reproduces unchanged.
    void set_energy_norm(bool on) { energy_norm_ = on; }
    bool energy_norm() const { return energy_norm_; }

    // -- model persistence ----------------------------------------------------
    void save_model(const std::string& dir) {
        std::string cmd_mkdir = "mkdir -p '" + dir + "'";
        (void)std::system(cmd_mkdir.c_str());
        // v3.2: promote context signatures, then build the semantic field over
        // the final fabric. Both are deterministic; both persist into the
        // substrate v4 tail so decide-time needs no rebuild.
        si_.finalize_contexts();
        si_.build_semantics();
        // v3.7.0 (Move 1): when --distvec is on, the resonance EDGE SET is
        // re-selected from dense PPMI+SVD vectors over the lane co-occurrence
        // structure (distvec.hpp) and the vectors persist in a DSTV tail.
        // Settle physics is untouched — the edges are the same mechanism, only
        // chosen by a representation that generalizes. Rebuilt only when the
        // fabric moved since the last build (any learn/pretrain sets the dirty
        // flag; repeated saves without learning skip the SVD).
        if (distvec_on_ && distvec_dirty_) {
            si::dist::DistConfig dc;
            if (distvec_dims_ > 0) dc.dims = static_cast<int>(distvec_dims_);
            if (si_.node_count() < static_cast<std::size_t>(dc.dims))
                dc.dims = std::max<int>(1, static_cast<int>(si_.node_count()) - 1);
            si::dist::build_field(si_, dc);
            distvec_dirty_ = false;
        }
        si_.save(dir + "/substrate.bin");
        std::ofstream cf(dir + "/calibration.json");
        cf << calib_.to_json().dump();
        std::ofstream mf(dir + "/meta.json");
        mf << sfx::JV(sfx::JVObj{
            {"engine", "syfox"}, {"version", VERSION},
            {"core", "si-substrate"}, {"nodes", static_cast<double>(si_.node_count())},
            {"lanes", static_cast<double>(si_.lane_count())},
            {"evidence_records", static_cast<double>(si_.evidence_count())},
            {"teach_events", static_cast<double>(teach_seq_)},
            {"contradictions", static_cast<double>(conflicts_.size())},
            {"semantics", si_.has_semantics()},
            {"sem_edges", static_cast<double>(si_.resonance_edge_count())},
            {"lane_contexts", static_cast<double>(si_.lane_context_count())},
            {"retrieval_memories", static_cast<double>(memories_.size())},
            // v3.6.0 fabric-construction disclosures: the df table behind the
            // CMD/MIMS rarity weighting, and the lesson count it was counted
            // over. Absent (0) on fabrics built before bigram lanes landed.
            {"df_nodes", static_cast<double>(si_.df_table_size())},
            {"df_lessons", static_cast<double>(si_.df_lesson_count())},
            // v3.7.0 fabric disclosures: the dense distributional field (Move 1)
            // and the typed-lane + pretrain provenance (Moves 2-3).
            {"distvec", si_.has_distvecs()},
            {"distvec_dims", static_cast<double>(si_.dist_dims())}}).dump();
        // v3 Milestone 3: the audit trail. conflicts.jsonl = every detected
        // contradiction; lessons_index.jsonl = state-hash -> outcome index
        // (so a LATER learn invocation still detects contradictions against
        // every lesson ever taught, not just this process's).
        {
            std::ofstream of(dir + "/conflicts.jsonl");
            for (const auto& c : conflicts_) of << c.dump() << "\n";
        }
        {
            std::ofstream of(dir + "/lessons_index.jsonl");
            for (const auto& kv : taught_outcomes_)
                of << sfx::JV(sfx::JVObj{
                    {"state_hash_hex", sfx::JV(hex64(kv.first))},
                    {"seq", static_cast<double>(kv.second.seq)},
                    {"outcome", sfx::JV(kv.second.outcome)}}).dump() << "\n";
        }
    }

    // v3.3 question-conditioned readout (read-only over the settled field).
    // Ported from the original Synthetic-Intelligence repo, where EVERY token
    // of the turn — context AND question — drove the activation field
    // (si_main.cpp: "for (int id : matched) physics.drive(id, 3.0f, 6.0f, 20)").
    // SyFox's contract keeps state and question separate, so the portable
    // form gates the READOUT: question tokens that are NEW relative to the
    // state (the "who" in "Who found the radio?") carry lane mass to the
    // candidate anchors, and candidates are scaled by that mass. The settled
    // field is measured, never rewritten — same shape as the hierarchy gate.
    void set_question_gate(bool on) { question_gate_on_ = on; }
    bool question_gate_on() const { return question_gate_on_; }
    void set_question_gate_floor(float f) {
        question_gate_floor_ = std::min(1.0f, std::max(0.0f, f));
    }
    float question_gate_floor() const { return question_gate_floor_; }

    // v3.4 — honest defer for near-ties (engine-level, so CLI, C API, HTTP
    // bridge and bench share ONE behavior): when the top two probabilities
    // are closer than this margin the readout does not carry a decision and
    // the answer defers with reason "ambiguous_tie" instead of shipping a
    // coin-flip as a confident label. 0 disables. Default 0.05 (the
    // reported near-ties sit at 0.004-0.010 probability gaps).
    void set_defer_margin(float m) { defer_margin_ = std::max(0.0f, m); }
    float defer_margin() const { return defer_margin_; }

    // v3.4 — question-context gating (two-stage settle, opt-in). Stage 1
    // settles the question's own tokens into a context field; stage 2
    // re-settles the state and composes the field as
    // (1-alpha)*state + alpha*context before readout. Deterministic;
    // settle() physics untouched — this is a decision-layer composition.
    void set_ctx_gate(bool on) { ctx_gate_on_ = on; }
    bool ctx_gate_on() const { return ctx_gate_on_; }
    void set_ctx_alpha(float a) { ctx_alpha_ = std::min(1.0f, std::max(0.0f, a)); }
    float ctx_alpha() const { return ctx_alpha_; }

    // v3.6.0 — BED §8 perturbation-contrast check (opt-in; default OFF so
    // every shipped workflow is bit-identical until an operator asks for the
    // honesty check). When on, each decide() settles a structure-broken copy
    // of the state (same token multiset, adjacent pairs transposed) and
    // defers choice answers whose real margin was not earned by structure.
    void set_perturb_check(bool on) { perturb_on_ = on; }
    bool perturb_check_on() const { return perturb_on_; }
    // v3.8.0 TASK 1 — adaptive depth (default off = bit-identical replay):
    // harder states get more settle passes. k_used = k_settle(base) +
    // round(difficulty * max_extra), difficulty in [0,1] from the state's
    // unknown-token fraction and lane isolation, both measured on the fabric
    // BEFORE the settle. Deterministic per state; harvest_rows mirrors the
    // rule so the calibration fit sees the same depth as the decision.
    void set_adaptive_depth(bool on) { adaptive_depth_on_ = on; }
    bool adaptive_depth_on() const { return adaptive_depth_on_; }
    void set_adaptive_max_k(int k) { adaptive_max_k_ = std::max(0, k); }
    // v3.8.0 TASK 4 — self-verification gate (default off): when the field
    // was still MOVING above the floor on its final pass (motion_rate >
    // floor), a choice answer defers with reason "unconverged_field" — the
    // readout measured a transient, not an attractor.
    void set_self_verify(bool on) { self_verify_on_ = on; }
    bool self_verify_on() const { return self_verify_on_; }
    void set_self_verify_floor(float f) { self_verify_floor_ = std::max(0.0f, f); }
    // v3.8.0 TASK 2 — multi-vector spectral channels (default off)
    void set_multi_vector(bool on, int k) {
        multi_vector_on_ = on;
        multi_vector_k_ = std::max(1, std::min(64, k));
    }
    bool multi_vector_on() const { return multi_vector_on_; }

    // v3.7.0 (Move 1) — dense distributional field at save time (distvec.hpp).
    // Default OFF: the PPMI+SVD build is a once-per-save fabric construction
    // step and every existing fabric must replay bit-identically, so the
    // operator opts in per build (--distvec). The flag never touches settle
    // physics; it only selects which resonance edges exist.
    void set_distvec(bool on) { distvec_on_ = on; }
    bool distvec_on() const { return distvec_on_; }
    // embedding width override (--distvec-dims K; 0 = the 300 default)
    void set_distvec_dims(long k) { distvec_dims_ = k; }

    // v3.7.0 (Move 4) — generalized tool scope at decide time. When on, the
    // arithmetic relevance heuristic (calc::detect, the SAME heuristic the
    // v3.5 calc oracle ships) runs over the state before the settle; a hit
    // discloses the oracle's derivation in usage (tool_used/tool_value) while
    // the field still answers or defers exactly as it measures. The register
    // records that the refusal would be principled; the tool computes; the
    // disclosure carries the provenance. Default OFF (bench bit-identity).
    void set_tools_check(bool on) { tools_check_on_ = on; }
    bool tools_check_on() const { return tools_check_on_; }

    // -- v3.9.0 THE CONSTRUCTIVE SUBSTRATE (SI Layer 2) -----------------------
    // The engine now carries the typed axiom graph alongside the energy
    // fabric. Loading axioms interns their concepts as fabric nodes and lays
    // axiom lanes (idempotently), so the two stores share one vocabulary.
    sxr::ReasonCore reason_;
    sxr::MetaLearner reason_meta_;
    sxr::ThinkerIndex thinker_;
    bool reason_loaded_ = false;

    bool reason_loaded() const { return reason_loaded_; }
    std::size_t reason_nodes() const { return reason_.graph.nodes.size(); }
    std::size_t reason_edges() const { return reason_.graph.edges.size(); }

    // load_axioms: SI's .axioms triple format into BOTH stores. Returns the
    // number of accepted (new) triples. Graph side: teach_axiom (dedup +
    // contradiction guard). Fabric side: intern subject/object, bind an
    // axiom lane at kAxiomLaneWeight only when the lane is absent — reloading
    // the same file changes nothing (idempotent), so replays stay stable.
    static constexpr float kAxiomLaneWeight = 1.0f;
    int load_axioms(const std::string& path, FILE* log = stderr) {
        std::ifstream in(path);
        if (!in) {
            if (log) std::fprintf(log, "[axioms] cannot open %s\n", path.c_str());
            return 0;
        }
        int accepted = 0;
        std::string line;
        while (std::getline(in, line)) {
            const auto hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);
            std::istringstream iss(line);
            std::string subj, rel_tok, obj;
            if (!(iss >> subj >> rel_tok >> obj)) continue;
            const int rel = sxr::rel_from_token(rel_tok);
            if (rel < 0) {
                if (log) std::fprintf(log, "[axioms] unknown relation '%s' skipped\n",
                                      rel_tok.c_str());
                continue;
            }
            const auto rep = sxr::teach_axiom(reason_, subj, rel, obj, path);
            if (!rep.accepted || rep.duplicate) continue;
            ++accepted;
            // fabric bridge: shared vocabulary + undirected axiom lane
            const si::NodeId a = si_.intern(subj);
            const si::NodeId b = si_.intern(obj);
            if (si_.lane_weight(a, b) < kAxiomLaneWeight)
                si_.bind(a, b, kAxiomLaneWeight - si_.lane_weight(a, b));
        }
        if (accepted > 0) reason_loaded_ = true;
        if (log) std::fprintf(log, "[axioms] %s: +%d new triples\n",
                              path.c_str(), accepted);
        return accepted;
    }

    // RADE proof pass (readout-side): typed path between a state concept and
    // an anchor over the axiom graph, premise-cited. Empty = no typed path.
    std::vector<std::string> proof_path_between(
        const std::vector<std::string>& state_words,
        const std::string& anchor_label) const {
        std::vector<std::string> hops;
        if (!reason_loaded_) return hops;
        const auto lt = si::norm::normalize(anchor_label);
        int goal = -1;
        for (const auto& t : lt)
            if (reason_.graph.by_label.count(t)) { goal = reason_.graph.by_label.at(t); break; }
        if (goal < 0) return hops;
        // strongest state concept present in the axiom graph (word order =
        // arrival order; first hit wins, mirroring the injection order).
        int start = -1; std::string start_word;
        for (const auto& w : state_words) {
            if (reason_.graph.by_label.count(w)) { start = reason_.graph.by_label.at(w); start_word = w; break; }
        }
        if (start < 0 || start == goal) return hops;
        const auto path = reason_.find_path(
            reason_.graph.nodes[static_cast<std::size_t>(start)].label,
            reason_.graph.nodes[static_cast<std::size_t>(goal)].label,
            -1, 4);
        hops.reserve(path.size());
        for (int ei : path) {
            const auto& e = reason_.graph.edges[static_cast<std::size_t>(ei)];
            hops.push_back(reason_.graph.nodes[static_cast<std::size_t>(e.from)].label
                         + " " + sxr::rel_name(e.rel) + " "
                         + reason_.graph.nodes[static_cast<std::size_t>(e.to)].label);
        }
        (void)start_word;
        return hops;
    }

    // -- v3.9.0 decision-layer paper gates (all default OFF; replay-safe) ----
    void set_proof(bool disclose, bool require) { proof_on_ = disclose; require_proof_ = require; }
    bool proof_on() const { return proof_on_; }
    void set_momentum_gate(bool on) { momentum_gate_ = on; si_.set_momentum(on); }
    bool momentum_gate_on() const { return momentum_gate_; }
    void set_wavefront_gate(bool on) { wavefront_gate_ = on; }
    bool wavefront_gate_on() const { return wavefront_gate_; }
    void set_coherence_gate(bool on, float floor) {
        coherence_gate_ = on;
        coherence_floor_ = floor > 0.0f ? floor : 0.05f;
    }
    bool coherence_gate_on() const { return coherence_gate_; }
    void set_vght_gate(bool on) { vght_gate_ = on; si_.set_vght_gate(on); }
    bool vght_gate_on() const { return vght_gate_; }
    void set_cem(bool on, int rounds, int variants) {
        cem_on_ = on;
        cem_rounds_ = std::max(1, std::min(4, rounds));
        cem_variants_ = std::max(2, std::min(16, variants));
    }
    bool cem_on() const { return cem_on_; }
    void set_fragments(bool on) { fragments_on_ = on; }
    bool fragments_on() const { return fragments_on_; }
    void set_ood_knn(bool on, float tau) { ood_knn_on_ = on; ood_knn_tau_ = tau >= 0.0f ? tau : 1.0f; }
    bool ood_knn_on() const { return ood_knn_on_; }
    void set_tension_gate(bool on) { tension_on_ = on; }
    bool tension_on() const { return tension_on_; }
    void set_pro_con(bool on) { pro_con_ = on; }
    bool pro_con_on() const { return pro_con_; }
    void set_sentinels(bool on) { sentinels_ = on; }
    bool sentinels_on() const { return sentinels_; }
    float tension_acc() const { return tension_acc_; }

    // v3.2.1: returns false when the substrate file is missing (honest load
    // failure — callers surface a clear error instead of deciding on an
    // empty fabric or, worse, reporting a misleading downstream message).
    bool load_model(const std::string& dir) {
        if (!si_.load(dir + "/substrate.bin")) return false;
        std::ifstream cf(dir + "/calibration.json");
        if (cf) { std::string buf((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
                  calib_ = Calibration::from_json(sfx::JV::parse(buf)); }
        // v3 audit trail (optional files; a v2 model dir simply lacks them)
        {
            std::ifstream f(dir + "/conflicts.jsonl");
            if (f) {
                std::string line;
                while (std::getline(f, line))
                    if (!line.empty()) {
                        sfx::JV c = sfx::JV::parse(line);
                        conflicts_.push_back(c);
                        if (c.has("state_hash_hex"))
                            conflicts_by_state_[hex_to_u64(c.at("state_hash_hex").as_str())]
                                .push_back(conflicts_.size() - 1);
                    }
            }
        }
        {
            std::ifstream f(dir + "/lessons_index.jsonl");
            if (f) {
                std::string line;
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    sfx::JV r = sfx::JV::parse(line);
                    Taught t;
                    t.seq = static_cast<std::uint64_t>(r.at("seq").as_num(0));
                    t.outcome = r.at("outcome").as_str();
                    taught_outcomes_[hex_to_u64(r.at("state_hash_hex").as_str())] = t;
                    if (t.seq > teach_seq_) teach_seq_ = t.seq;
                }
            }
        }
        // v3.2: semantic hierarchy (Stage 3) ships as hierarchy.json in the
        // model dir; absent file = single-stage readout, exactly as before.
        load_hierarchy(dir + "/hierarchy.json");
        // v3.2 retrieval BY DEFAULT: a memories.jsonl shipped in the model dir
        // is loaded and fingerprinted once (settled-field cosines are then
        // free at decide time). No file => retrieval is inert, behavior of
        // every pre-3.2 model unchanged. --no-retrieval disables at runtime.
        {
            std::ifstream f(dir + "/memories.jsonl");
            if (f) {
                std::vector<recall::Memory> mems;
                std::string line;
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    try {
                        sfx::JV r = sfx::JV::parse(line);
                        recall::Memory m;
                        m.label = r.at("label").as_str();
                        m.state = si::norm::normalize(r.at("state").as_str());
                        if (!m.label.empty() && !m.state.empty()) mems.push_back(std::move(m));
                    } catch (const std::exception&) { /* skip malformed line */ }
                }
                if (!mems.empty()) set_memories(std::move(mems));
            }
        }
        // v3.9.0 CONSTRUCTIVE SUBSTRATE (opt-in per model): an axioms.txt in
        // the model dir loads the typed graph AND lays the axiom lanes.
        // Absent file = no graph, no lanes: every pre-3.9 model replays
        // bit-identically. v39.json carries the RIFA tension accumulator and
        // the OOD signature store (also absent by default).
        load_axioms(dir + "/axioms.txt", nullptr);
        {
            std::ifstream f(dir + "/v39.json");
            if (f) {
                std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                try {
                    sfx::JV r = sfx::JV::parse(buf);
                    tension_acc_ = static_cast<float>(r.at("tension_acc").as_num(0.0));
                    ood_sigs_.clear();
                    if (r.has("ood_signatures") && r.at("ood_signatures").is_arr()) {
                        for (const auto& s : r.at("ood_signatures").arr) {
                            if (!s.is_arr() || s.arr.size() < 4) continue;
                            ood_sigs_.push_back({static_cast<float>(s.arr[0].as_num(0)),
                                                 static_cast<float>(s.arr[1].as_num(0)),
                                                 static_cast<float>(s.arr[2].as_num(0)),
                                                 static_cast<float>(s.arr[3].as_num(0))});
                        }
                    }
                } catch (const std::exception&) { /* absent/torn: stay default */ }
            }
        }
        return true;
    }

    // v3.9.0: persist the v3.9 sidecar (tension + OOD signatures). Called by
    // the CLI after calibrate (which collects signatures). Never called by
    // learn/save_model, so v3.8 model dirs are byte-identical until a v3.9
    // command writes this file.
    void save_v39_state(const std::string& dir) const {
        sfx::JVArr sigs;
        for (const auto& s : ood_sigs_)
            sigs.push_back(sfx::JV(sfx::JVArr{s[0], s[1], s[2], s[3]}));
        sfx::JV out(sfx::JVObj{
            {"tension_acc", tension_acc_},
            {"ood_signatures", sfx::JV(std::move(sigs))}});
        std::ofstream f(dir + "/v39.json");
        if (f) f << out.dump() << "\n";
    }

    // v3.9.0 (fixes.md): record a decision signature for the OOD kNN gate.
    void record_ood_signature(float settled_energy, float top_energy,
                              float margin, float unk_frac) {
        if (ood_sigs_.size() >= 65536) ood_sigs_.erase(ood_sigs_.begin());
        ood_sigs_.push_back({settled_energy, top_energy, margin, unk_frac});
    }

    // -- v3.2 retrieval-by-default (associative priming) ------------------------
    // Memories are NOT a second classifier: each memory is fingerprinted by
    // its own settled field (recall.hpp Hopfield-style resonance), and the
    // top-k resonating memories inject their OUTCOME tokens at a faint dose
    // before the state settles — a physical prior from lived experience.
    // Deterministic: fingerprints are fixed at load, the query fingerprint
    // is a pure function of the state, ties break by memory order.
    void set_memories(std::vector<recall::Memory> mems) {
        memories_ = std::move(mems);
        memory_fps_.clear();
        memory_fps_.reserve(memories_.size());
        for (const auto& m : memories_)
            memory_fps_.push_back(recall::fingerprint(si_, m.state));
        retrieval_on_ = true;
    }
    void set_retrieval(bool on) { retrieval_on_ = on; }
    bool retrieval_on() const { return retrieval_on_; }
    void set_retrieval_topk(int k) { retrieval_topk_ = std::max(0, k); }
    void set_retrieval_dose(float d) { retrieval_dose_ = std::max(0.0f, d); }
    const std::vector<recall::Memory>& memories() const { return memories_; }

    // -- v3.2 Stage 3: semantic hierarchy (readout-side two-stage) --------------
    // hierarchy.json: {"intents": {"c42": "h_card"}, "categories":
    // {"h_card": {"criteria": "..."}}, "floor": 0.35}. Both levels are
    // TRAINED anchors in the same fabric; at decide, stage 1 reads category
    // energies, stage 2 scales each intent candidate by its category's
    // normalized score (floor keeps every candidate physically alive).
    bool load_hierarchy(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        try {
            sfx::JV j = sfx::JV::parse(buf);
            hier_of_.clear(); hier_cat_criteria_.clear();
            if (j.has("intents") && j.at("intents").is_obj())
                for (const auto& kv : j.at("intents").obj)
                    hier_of_[kv.first] = kv.second.as_str();
            if (j.has("categories") && j.at("categories").is_obj())
                for (const auto& kv : j.at("categories").obj)
                    hier_cat_criteria_[kv.first] = kv.second.has("criteria")
                        ? kv.second.at("criteria").as_str() : std::string();
            hier_floor_ = static_cast<float>(j.has("floor") ? j.at("floor").as_num(0.35) : 0.35);
            hier_on_ = !hier_of_.empty() && !hier_cat_criteria_.empty();
        } catch (const std::exception&) { hier_on_ = false; }
        return hier_on_;
    }
    void set_hierarchy(bool on) { hier_on_ = on; }
    bool hierarchy_on() const { return hier_on_; }

    // set the audit context tag recorded into the lane evidence ledger
    void set_context(const std::string& c) { context_ = c; }
    std::uint64_t teach_events() const { return teach_seq_; }
    const std::vector<sfx::JV>& conflicts() const { return conflicts_; }

    // -- HEBBIAN LESSON (learning = substrate rewiring) -------------------------
    // One labelled experience: state text + question + rewarded outcome.
    //  1. all concepts join the vocabulary (intern; fresh nodes are light)
    //  2. adjacent state tokens bind (co-occurrence fabric for diffusion)
    //  3. state -> outcome lanes strengthen (fire together, wire together)
    //
    // v2.2: (a) the STATE side carries character trigram lanes (ngram.hpp;
    // state_tokens()) so typos and sub-word fragments reach the fabric —
    // outcome/label side stays gram-free. (b) augment=true is the mass-
    // guarded re-teach used by paraphrase/variant lessons: concepts already
    // in the vocabulary are NOT intern()ed again, so re-exposure does not
    // raise acoustic mass (the v2.1 dose-response artifact: re-taught
    // near-duplicates re-weighted the field by 1/sqrt(mass) damping and
    // REGRESSED held-out accuracy). With the guard, a variant lesson only
    // lays/strengthens LANES (coverage), it never re-deposits mass.
    // -- v3.9.0 shared learn-path widening (mirrored in learn_example AND
    // learn_noul — the fit==decide parity lesson) ---------------------------
    // (a) CMD §2.5 sentinels: word-initial and word-final ordered pairs get
    //     boundary-marked node names so position is observable.
    // (b) RB surrogate pro/con namespaces: negated lessons bind through
    //     negation-mangled node names ("!"+token) — con-evidence structurally
    //     cannot cancel pro-evidence on shared lanes.
    void apply_v39_widening(const std::vector<std::string>& words,
                            const std::vector<std::string>& state,
                            bool augment,
                            std::vector<std::string>& state_full) {
        if (sentinels_ && si::norm::bigrams_enabled() && words.size() >= 2) {
            const std::string head = "^" + words[0] + "~" + words[1];
            const std::string tail = words[words.size() - 2] + "~"
                                   + words[words.size() - 1] + "$";
            if (!(augment && si_.has(head))) si_.intern(head);
            if (!(augment && si_.has(tail))) si_.intern(tail);
            state_full.push_back(head);
            state_full.push_back(tail);
        }
        if (pro_con_ && si::roles::verb_negated(words)) {
            for (const auto& t : state) {
                const std::string neg = "!" + t;
                if (!(augment && si_.has(neg))) si_.intern(neg);
                state_full.push_back(neg);
            }
        }
    }

    // v3.9.0 (CPME-II discrete surrogate): order evidence notes. A token in
    // the first half of the state stream is consistent (subject side); the
    // anchor always follows. Notes land AFTER the bind so the factor uses
    // accumulated history on the NEXT lesson — no same-lesson double-count.
    void note_v39_order(const std::vector<std::string>& state,
                        const std::vector<std::string>& anchors) {
        if (!si_.order_on()) return;
        for (std::size_t i = 0; i < state.size(); ++i)
            if (si_.has(state[i]))
                for (const auto& o : anchors)
                    if (si_.has(o))
                        si_.note_order(si_.find(state[i]), si_.find(o),
                                       2 * i < state.size());
    }

    void learn_example(const std::string& state_text, const std::string& instructions,
                       const std::string& outcome_text, bool augment = false,
                       const LearnPolicy& lp = LearnPolicy{}, bool* learned = nullptr) {
        const bool grams = si::norm::grams_enabled();
        const std::vector<std::string> words =
            si::norm::normalize(state_text + " " + instructions);
        std::vector<std::string> state = si::norm::state_tokens(words, grams);
        const std::size_t n_words = words.size();
        std::vector<std::string> outcome = si::norm::normalize(outcome_text);
        // Milestone 2: novelty is measured BEFORE interning (a word the fabric
        // has not seen yet is exactly the new information this lesson carries).
        float eta = 1.0f;
        if (lp.novelty && n_words > 0) {
            std::size_t fresh = 0;
            for (const auto& w : words) if (!si_.has(w)) ++fresh;
            eta = lp.novelty_floor
                + (1.0f - lp.novelty_floor) * (static_cast<float>(fresh) / static_cast<float>(n_words));
        }
        // Milestone 2: exact-duplicate lessons are skipped outright.
        if (lp.dedup) {
            std::uint64_t h = 1469598103934665603ull;
            // cast! fnv1a_hash returns float; an uncast + would promote the
            // whole chain to float (24-bit mantissa) and collapse every
            // lesson onto one hash (measured: 25111/25200 false duplicates).
            for (const auto& w : words) { h = h * 1099511628211ull + static_cast<std::uint64_t>(si::fnv1a_hash(w)); }
            for (const auto& w : outcome) { h = h * 1099511628211ull + static_cast<std::uint64_t>(si::fnv1a_hash(w)); }
            if (!lesson_hashes_.insert(h).second) {
                if (learned) *learned = false;
                return;
            }
        }
        // v3.9.0 (CME §3) GRADED FINGERPRINT ENGAGEMENT: same example again
        // consolidates at half dose (hit_count tracked, table capped). Runs
        // BEFORE the contradiction check: a re-taught identical lesson is not
        // a contradiction, it is a repetition.
        if (lp.fingerprints) {
            std::uint64_t h = 1469598103934665603ull;
            {
                std::vector<std::uint64_t> toks;
                toks.reserve(words.size() + outcome.size());
                for (const auto& w : words) toks.push_back(static_cast<std::uint64_t>(si::fnv1a_hash(w)));
                for (const auto& w : outcome) toks.push_back(static_cast<std::uint64_t>(si::fnv1a_hash(w)));
                std::sort(toks.begin(), toks.end());              // multiset fingerprint
                for (std::uint64_t t : toks) h = (h ^ t) * 0x100000001b3ull;
                h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ull;         // splitmix finalizer
                h ^= h >> 27; h *= 0x94d049bb133111ebull;
                h ^= h >> 31;
            }
            auto it = fp_table_.find(h);
            if (it != fp_table_.end()) {
                ++it->second[2];                                  // hit_count
                it->second[1] = ++teach_seq_;
                eta *= 0.5f;                                      // graded half dose
                if (learned) *learned = true;
                // fall through with the reduced eta (still consolidates)
            } else {
                if (fp_table_.size() >= 1024) {
                    auto victim = fp_table_.begin();
                    for (auto jt = fp_table_.begin(); jt != fp_table_.end(); ++jt)
                        if (jt->second[2] < victim->second[2]
                            || (jt->second[2] == victim->second[2]
                                && jt->second[0] < victim->second[0]))
                            victim = jt;
                    fp_table_.erase(victim);
                }
                fp_table_[h] = {teach_seq_ + 1, 0, 0};            // first_seq, last_seq, hits
            }
        }
        // Milestone 3: contradiction check. Same (state + question) taught
        // with a DIFFERENT outcome is a contradiction — it must surface in
        // the audit trail, never silently override (or be silently ignored).
        const std::uint64_t shash = stream_hash(words);
        const std::uint64_t seq = ++teach_seq_;
        auto prev = taught_outcomes_.find(shash);
        if (prev != taught_outcomes_.end() && prev->second.outcome != outcome_text) {
            sfx::JV c(sfx::JVObj{
                {"type", sfx::JV("contradiction")},
                {"state", sfx::JV(state_text)}, {"state_hash_hex", sfx::JV(hex64(stream_hash(si::norm::normalize(state_text))))},
                {"instructions", sfx::JV(instructions)},
                {"outcome_old", sfx::JV(prev->second.outcome)},
                {"outcome_new", sfx::JV(outcome_text)},
                {"seq_old", static_cast<double>(prev->second.seq)},
                {"seq_new", static_cast<double>(seq)},
                {"context", sfx::JV(context_)}});
            conflicts_by_state_[stream_hash(si::norm::normalize(state_text))].push_back(conflicts_.size());
            conflicts_.push_back(c);
            // v3.9.0 (RIFA SIL, sign fixed): contradictions metabolize as
            // tension. The accumulator only grows (the failed-pool paper's
            // own code DECREMENTED tension on contradiction — log(coherence)
            // <= 0 — contradicting its prose; we ship the prose). Disclosed
            // per decide; the GATE is opt-in (--tension-gate).
            tension_acc_ += 1.0f;
            // the new lesson is COUNTER-EVIDENCE for the old binding's lanes:
            // mark every old lane reachable from this state's words.
            mark_counter_lanes(words, prev->second.outcome, seq);
        }
        taught_outcomes_[shash] = Taught{seq, outcome_text};
        for (const auto& t : state)
            if (!(augment && si_.has(t))) si_.intern(t);
        for (const auto& t : outcome)
            if (!(augment && si_.has(t))) si_.intern(t);
        for (std::size_t i = 1; i < n_words; ++i)          // co-occurrence fabric
            si_.bind(si_.find(words[i - 1]), si_.find(words[i]), si_.config().learn_eta * 0.5f * eta);
        if (grams)                                         // word<->its own trigrams fabric
            for (const auto& w : words)
                for (const auto& g : si::norm::expand_ngrams({w}))
                    if (si_.has(g))
                        si_.bind(si_.find(w), si_.find(g), si_.config().learn_eta * 0.5f * eta);
        // v3.6.0 (MRS §4b / CMD §2.2 / MIMS §6): document-frequency notes for
        // this lesson, then widen the state token set with ORDERED BIGRAM
        // nodes before the Hebbian lesson. Bigram nodes bind to the outcome
        // anchors exactly like word tokens, so the fabric carries order —
        // the escape from the multiset-invariance theorem (paws 0.500,
        // xnli 0.333). With --bigrams off (or bigrams disabled globally)
        // state_full == state and the lesson reproduces the v3.5 lane set
        // EXCEPT for the IDF factor, which activates only when df notes
        // exist — see hebbian_lesson.
        si_.begin_df_lesson();
        std::vector<std::string> state_full = state;
        if (si::norm::bigrams_enabled())
            for (const auto& bg : si::norm::bigrams_of(words)) {
                if (!(augment && si_.has(bg))) si_.intern(bg);
                state_full.push_back(bg);
            }
        // v3.9.0 (CMD §2.5) SENTINEL BIGRAMS + (RB surrogate) PRO/CON
        // NAMESPACES + (CPME-II) ORDER EVIDENCE — shared widening helper,
        // mirrored exactly in learn_noul (the fit==decide lesson).
        apply_v39_widening(words, state, augment, state_full);
        // v3.7.0 (Move 3): widen with TYPED ORDERED PAIR nodes — (word, role)
        // relations from the scene grammar in roles.hpp. Interned like the
        // bigrams (mangled names never collide with normalized tokens), df
        // noted like the bigrams, bound to the outcome anchors by the SAME
        // hebbian_lesson below. Fabrics learned with typed lanes off contain
        // no typed nodes; the decide-side block is then a no-op (replay).
        if (si::roles::typed_lanes_enabled())
            for (const auto& tp : si::roles::typed_pairs_of(words)) {
                if (!(augment && si_.has(tp))) si_.intern(tp);
                state_full.push_back(tp);
            }
        // v3.8.0 (TASK 3): widen with CAUSAL pair nodes — cause?>effect and
        // means!>goal signatures from the deterministic connective scan.
        // Same intern/df/bind pipeline as the typed pairs; fabrics learned
        // with causal lanes off contain none (replay no-op).
        if (si::roles::causal_lanes_enabled())
            for (const auto& cp : si::roles::causal_pairs_of(words)) {
                if (!(augment && si_.has(cp))) si_.intern(cp);
                state_full.push_back(cp);
            }
        {
            std::unordered_set<std::string> df_seen;
            for (const auto& w : words)
                if (df_seen.insert(w).second && si_.has(w)) si_.note_df(si_.find(w));
            if (si::norm::bigrams_enabled())
                for (const auto& bg : si::norm::bigrams_of(words))
                    if (df_seen.insert(bg).second && si_.has(bg)) si_.note_df(si_.find(bg));
            if (si::roles::typed_lanes_enabled())
                for (const auto& tp : si::roles::typed_pairs_of(words))
                    if (df_seen.insert(tp).second && si_.has(tp)) si_.note_df(si_.find(tp));
            if (si::roles::causal_lanes_enabled())
                for (const auto& cp : si::roles::causal_pairs_of(words))
                    if (df_seen.insert(cp).second && si_.has(cp)) si_.note_df(si_.find(cp));
        }
        si_.hebbian_lesson(state_full, outcome, eta);
        // v3.9.0 (CPME-II §1, discrete surrogate) ORDER EVIDENCE: a state
        // token in the FIRST half of the stream is order-consistent (the
        // subject side); later tokens carry reversed evidence. Lanes that
        // keep their side across lessons consolidate at full strength
        // (Phi=1); churned lanes cancel toward the 0.4 floor. Notes land
        // AFTER the bind so the factor uses ACCUMULATED history plus this
        // lesson's evidence on the NEXT pass — no same-lesson double-count.
        note_v39_order(state, outcome);
        // v3.2 Stage 2: context-signature accumulation on the state->outcome
        // lanes. The row's own tokens (state + criteria words) are the lane's
        // context evidence; counts that survive pruning (>= 2 lessons) become
        // the lane's required set at save time, and cross-lane diffs become
        // forbidden words. Bookkeeping only — the field is untouched here.
        {
            std::vector<si::NodeId> ctx_ids;
            for (const auto& u : words) {
                if (si_.has(u)) ctx_ids.push_back(si_.find(u));
                if (ctx_ids.size() >= 16) break;
            }
            if (!ctx_ids.empty())
                for (const auto& a : state)
                    if (si_.has(a))
                        for (const auto& b : outcome)
                            if (si_.has(b))
                                for (const si::NodeId u : ctx_ids)
                                    si_.add_ctx_support(si_.find(a), si_.find(b), u);
        }
        // Milestone 3: record the support evidence for every state->outcome
        // lane this lesson just strengthened (bookkeeping; the ledger never
        // feeds back into the field).
        for (const auto& a : state)
            if (si_.has(a))
                for (const auto& b : outcome)
                    if (si_.has(b)) si_.record_support(si_.find(a), si_.find(b), seq, context_);
        if (learned) *learned = true;
    }

    // -- CALIBRATION TOOL (external post-processor, not core physics) ----------
    // Fits temperature (choice/score) and Platt (noul) on collected rows.
    // v2.1: a choice/score row carries the FULL candidate energy vector plus
    // the gold label, so the fit sees the same multi-class softmax the decide
    // path reports. The old {e_correct, worst e_wrong} pairwise view was
    // structurally blind to argmax errors on near-ties (a wrong-but-tied row
    // stays p=0.5 under ANY temperature), which let the optimizer drive T to
    // its floor and manufacture confidence on wrong answers.
    struct CalibRow {
        std::string type;
        std::string label;                                    // gold candidate key
        std::vector<std::pair<std::string, float>> energies;  // full candidate vector
        float support = 0.0f;                                 // noul: support readout
        int y = 0;                                            // noul: gold target
    };

    void fit_calibration(const std::vector<CalibRow>& rows) {
        // v2.1 adoption guard: the NLL optimum is the textbook temperature,
        // but on tiny heldout sets with an unavoidable argmax error it can
        // WORSEN ECE (sharpening raises the wrong answer's confidence too).
        // So the fit is a selection between exactly two honest candidates —
        // T = 1 and the NLL optimum — decided by measured multi-class ECE on
        // the fit rows (1 bit per question type; argmax never moves). The
        // calibrate report shows before/after either way.
        struct Cand { float T; };
        auto adopt_temp = [&](const std::string& type, float& slot) {
            float cand = slot;
            fit_temperature(rows, type, cand);
            const double e1 = calibration_ece(rows, type, 1.0f, 0, 0);
            const double e2 = calibration_ece(rows, type, cand, 0, 0);
            slot = (e2 < e1) ? cand : 1.0f;
        };
        adopt_temp("choice", calib_.choice_temperature);
        adopt_temp("score", calib_.score_temperature);
        // Platt: same 1-bit selection for the noul map
        const float a0 = calib_.noul_a, b0 = calib_.noul_b;
        const double p1 = calibration_ece(rows, "noul", 1.0f, a0, b0);
        fit_platt(rows);
        const double p2 = calibration_ece(rows, "noul", 1.0f, calib_.noul_a, calib_.noul_b);
        if (!(p2 < p1)) { calib_.noul_a = a0; calib_.noul_b = b0; }
        calib_.fitted = true;
        calib_.rows = rows.size();
    }

    // Multi-class expected calibration error over collected rows (10 equal-
    // width bins on top-probability confidence, argmax-correctness for
    // choice/score; Platt sigmoid + 0.5 threshold for noul). Same definition
    // the bench reports, so calibrate's before/after and bench's ECE agree.
    static double calibration_ece(const std::vector<CalibRow>& rows,
                                  const std::string& type, float T,
                                  float noul_a, float noul_b) {
        long bins[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        long corr[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double confs[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const float Ts = std::max(1e-6f, T);
        for (const auto& r : rows) {
            if (r.type != type) continue;
            float conf = 0.0f; bool correct = false;
            if (r.type == "noul") {
                const float p = 1.0f / (1.0f + std::exp(-(noul_a * r.support + noul_b)));
                conf = std::max(p, 1.0f - p);
                correct = (p >= 0.5f) == (r.y == 1);
            } else {
                if (r.energies.empty()) continue;
                float maxe = -1e30f;
                for (const auto& kv : r.energies) maxe = std::max(maxe, kv.second);
                float z = 0.0f, top = -1.0f; std::string amax;
                for (const auto& kv : r.energies) {
                    const float v = std::exp((kv.second - maxe) / Ts);
                    z += v;
                    if (v > top) { top = v; amax = kv.first; }
                }
                conf = top / z;
                correct = amax == r.label;
            }
            const int b = std::min(9, std::max(0, static_cast<int>(conf * 10.0f)));
            ++bins[b]; corr[b] += correct ? 1 : 0; confs[b] += conf;
        }
        long total = 0;
        for (int i = 0; i < 10; ++i) total += bins[i];
        if (total == 0) return 0.0;
        double e = 0.0;
        for (int i = 0; i < 10; ++i) {
            if (!bins[i]) continue;
            e += (static_cast<double>(bins[i]) / total)
               * std::fabs(static_cast<double>(corr[i]) / bins[i]
                           - confs[i] / bins[i]);
        }
        return e;
    }

    // -- Noul lesson (valence-aware) ------------------------------------------
    // y=true  : Hebbian bind state concepts <-> instruction concepts
    //           (supporting evidence wires the route)
    // y=false : anti-Hebbian weaken the same routes (disconfirming evidence
    //           dissolves them), so the field discriminates, not accumulates.
    void learn_noul(const std::string& state_text, const std::string& instructions, bool y,
                    bool augment = false, const LearnPolicy& lp = LearnPolicy{},
                    bool* learned = nullptr) {
        const bool grams = si::norm::grams_enabled();
        const std::vector<std::string> words = si::norm::normalize(state_text);
        std::vector<std::string> state = si::norm::state_tokens(words, grams);
        std::vector<std::string> instr = si::norm::normalize(instructions);   // gram-free
        float eta = 1.0f;
        if (lp.novelty && !words.empty()) {
            std::size_t fresh = 0;
            for (const auto& w : words) if (!si_.has(w)) ++fresh;
            eta = lp.novelty_floor
                + (1.0f - lp.novelty_floor) * (static_cast<float>(fresh) / static_cast<float>(words.size()));
        }
        if (lp.dedup) {
            std::uint64_t h = 1469598103934665603ull;
            for (const auto& w : words) { h = h * 1099511628211ull + static_cast<std::uint64_t>(si::fnv1a_hash(w)); }
            for (const auto& w : instr) { h = h * 1099511628211ull + static_cast<std::uint64_t>(si::fnv1a_hash(w)); }
            h = h * 1099511628211ull + (y ? 0x9E3779B9ull : 0x85EBCA6Bull);
            if (!lesson_hashes_.insert(h).second) {
                if (learned) *learned = false;
                return;
            }
        }
        // Milestone 3: noul contradictions — same (state + instruction) taught
        // with the opposite valence (true vs false) is a contradiction too.
        const std::string outcome_text = y ? "true" : "false";
        const std::uint64_t shash = stream_hash(words);
        const std::uint64_t seq = ++teach_seq_;
        auto prev = taught_outcomes_.find(shash);
        if (prev != taught_outcomes_.end() && prev->second.outcome != outcome_text) {
            sfx::JV c(sfx::JVObj{
                {"type", sfx::JV("contradiction")},
                {"state", sfx::JV(state_text)}, {"state_hash_hex", sfx::JV(hex64(stream_hash(si::norm::normalize(state_text))))},
                {"instructions", sfx::JV(instructions)},
                {"outcome_old", sfx::JV(prev->second.outcome)},
                {"outcome_new", sfx::JV(outcome_text)},
                {"seq_old", static_cast<double>(prev->second.seq)},
                {"seq_new", static_cast<double>(seq)},
                {"context", sfx::JV(context_)}});
            conflicts_by_state_[stream_hash(si::norm::normalize(state_text))].push_back(conflicts_.size());
            conflicts_.push_back(c);
            // v3.9.0 (RIFA SIL, sign fixed): contradictions metabolize as
            // tension. The accumulator only grows — the failed-pool paper's
            // own code DECREMENTED tension on contradiction (log(coherence)
            // <= 0 made its integrator decay on the very event its prose says
            // must charge it); we ship the prose. Disclosed per decide; the
            // GATE is opt-in (--tension-gate).
            tension_acc_ += 1.0f;
            mark_counter_lanes(words, prev->second.outcome, seq);
        }
        taught_outcomes_[shash] = Taught{seq, outcome_text};
        for (const auto& t : state)
            if (!(augment && si_.has(t))) si_.intern(t);
        for (const auto& t : instr)
            if (!(augment && si_.has(t))) si_.intern(t);
        if (grams)
            for (const auto& w : words)
                for (const auto& g : si::norm::expand_ngrams({w}))
                    if (si_.has(g))
                        si_.bind(si_.find(w), si_.find(g), si_.config().learn_eta * 0.5f * eta);
        // v3.6.0: df notes + ordered bigram widening, mirrored from
        // learn_example (noul lessons carry the same fabric construction).
        si_.begin_df_lesson();
        std::vector<std::string> state_full = state;
        if (si::norm::bigrams_enabled())
            for (const auto& bg : si::norm::bigrams_of(words)) {
                if (!(augment && si_.has(bg))) si_.intern(bg);
                state_full.push_back(bg);
            }
        // v3.9.0 shared widening: sentinels + pro/con namespaces (mirrored).
        apply_v39_widening(words, state, augment, state_full);
        // v3.7.0 (Move 3): typed pair widening, mirrored from learn_example —
        // noul lessons carry the same fabric construction.
        if (si::roles::typed_lanes_enabled())
            for (const auto& tp : si::roles::typed_pairs_of(words)) {
                if (!(augment && si_.has(tp))) si_.intern(tp);
                state_full.push_back(tp);
            }
        // v3.8.0 (TASK 3): causal pair widening, mirrored from learn_example.
        if (si::roles::causal_lanes_enabled())
            for (const auto& cp : si::roles::causal_pairs_of(words)) {
                if (!(augment && si_.has(cp))) si_.intern(cp);
                state_full.push_back(cp);
            }
        {
            std::unordered_set<std::string> df_seen;
            for (const auto& w : words)
                if (df_seen.insert(w).second && si_.has(w)) si_.note_df(si_.find(w));
            if (si::norm::bigrams_enabled())
                for (const auto& bg : si::norm::bigrams_of(words))
                    if (df_seen.insert(bg).second && si_.has(bg)) si_.note_df(si_.find(bg));
            if (si::roles::typed_lanes_enabled())
                for (const auto& tp : si::roles::typed_pairs_of(words))
                    if (df_seen.insert(tp).second && si_.has(tp)) si_.note_df(si_.find(tp));
            if (si::roles::causal_lanes_enabled())
                for (const auto& cp : si::roles::causal_pairs_of(words))
                    if (df_seen.insert(cp).second && si_.has(cp)) si_.note_df(si_.find(cp));
        }
        if (y) {
            si_.hebbian_lesson(state_full, instr, 2.0f * eta);   // supporting evidence binds hard
            for (const auto& a : state)
                if (si_.has(a))
                    for (const auto& b : instr)
                        if (si_.has(b)) si_.record_support(si_.find(a), si_.find(b), seq, context_);
        } else {
            si_.weaken(state_full, instr, eta);                  // disconfirming evidence dissolves
            for (const auto& a : state)
                if (si_.has(a))
                    for (const auto& b : instr)
                        if (si_.has(b)) si_.record_counter(si_.find(a), si_.find(b), seq, context_);
        }
        // v3.9.0 order evidence, mirrored from learn_example (noul anchors).
        note_v39_order(state, instr);
        if (learned) *learned = true;
    }

    // decide-side injection dose (Milestone-1 gain knob; see set_energy_norm)
    float state_dose(const std::vector<std::string>& words) const {
        if (!energy_norm_) return si_.config().inject_energy;
        double sum_sqrt = 0.0;
        int present = 0;
        for (const auto& w : words)
            if (si_.has(w)) {
                sum_sqrt += std::sqrt(static_cast<double>(si_.node_mass(si_.find(w))));
                ++present;
            }
        if (present == 0 || sum_sqrt <= 0.0) return si_.config().inject_energy;
        return si_.config().inject_energy
             * static_cast<float>(sum_sqrt / present);
    }

    // v3.8.0 (TASK 1) — problem difficulty from the state, BEFORE any settle.
    // Two measured proxies, equally weighted, clamped to [0,1]:
    //   unknown_frac — fraction of state tokens the fabric has never seen
    //                  (the OOD axis: dark nodes defer for a reason);
    //   isolated_frac — fraction of KNOWN tokens with zero lanes (the state
    //                  exists in the fabric but has no structure to travel:
    //                  energy cannot diffuse, more passes add nothing until
    //                  the lanes are found — the multi-hop walk pays here).
    // Deterministic per (state, fabric); the same number drives decide() and
    // harvest_rows() so the fit and the decision share one depth.
    float state_difficulty(const std::vector<std::string>& words) const {
        if (words.empty()) return 1.0f;
        int present = 0, isolated = 0;
        for (const auto& w : words) {
            if (!si_.has(w)) continue;
            ++present;
            if (si_.readout_neighbours(si_.find(w)) == 0) ++isolated;
        }
        const float unknown_frac = 1.0f - static_cast<float>(present)
                                       / static_cast<float>(words.size());
        const float isolated_frac = present > 0
            ? static_cast<float>(isolated) / static_cast<float>(present) : 1.0f;
        float d = 0.5f * (unknown_frac + isolated_frac);
        return std::min(1.0f, std::max(0.0f, d));
    }

    // v3.8.0 (TASK 1) — the ONE adaptive-depth rule shared by decide() and
    // harvest_rows() (the v3.2.1 fit==decide lesson). Returns k_used and
    // writes the difficulty. With the flag off this is the untouched base.
    int apply_adaptive_depth(const std::vector<std::string>& words,
                             float* difficulty_out) {
        // base is fixed at CONSTRUCTION: si_.set_k_settle() below mutates the
        // substrate config, and re-reading it here would ratchet the depth
        // up one decide() at a time (the second call would treat the last
        // k_used as the new base).
        const int base = adaptive_base_k_;
        si_.set_k_settle(base);
        if (!adaptive_depth_on_) {
            if (difficulty_out) *difficulty_out = state_difficulty(words);
            return base;
        }
        const float d = state_difficulty(words);
        if (difficulty_out) *difficulty_out = d;
        const int extra = static_cast<int>(std::lround(
            static_cast<double>(d) * static_cast<double>(adaptive_max_k_)));
        const int k_used = base + extra;
        si_.set_k_settle(k_used);
        return k_used;
    }

    // v3.8.0 (TASK 2) — multi-vector state: settle the SAME injection under
    // K spectral channel gates and SUPERPOSE (mean) the settled fields.
    // Used by decide() and harvest_rows() (the fit/decide parity lesson).
    // Requires a --distvec fabric (the gates are its leading spectral axes);
    // otherwise an honest no-op (disclosed in usage when available).
    bool apply_multi_vector(const std::vector<std::string>& words, Usage* usage) {
        if (!multi_vector_on_) return false;
        if (!si_.has_distvecs() || si_.dist_dims() < multi_vector_k_) return false;
        const int D = si_.dist_dims();
        const std::vector<float>& vecs = si_.distvecs();
        const std::size_t n = si_.node_count();
        std::vector<float> field_sum(n, 0.0f);
        int channels = 0;
        for (int c = 0; c < multi_vector_k_; ++c) {
            std::vector<float> coords(n);
            float mx = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                const float v = vecs[i * D + c];
                coords[i] = v;
                mx = std::max(mx, std::fabs(v));
            }
            if (mx <= 0.0f) continue;                     // dead axis: skip
            si_.set_channel_gate(std::move(coords), 0.5f * mx);
            si_.reset_field();
            si_.inject(words, state_dose(words));
            if (si::norm::bigrams_enabled() && words.size() >= 2) {
                std::vector<std::string> bgs;
                for (const auto& bg : si::norm::bigrams_of(words))
                    if (si_.has(bg)) bgs.push_back(bg);
                if (!bgs.empty()) si_.inject(bgs, state_dose(words) * kBigramDose);
            }
            if (si::roles::typed_lanes_enabled() && words.size() >= 2) {
                std::vector<std::string> tps;
                for (const auto& tp : si::roles::typed_pairs_of(words))
                    if (si_.has(tp)) tps.push_back(tp);
                if (!tps.empty()) si_.inject(tps, state_dose(words) * kTypedDose);
            }
            if (si::roles::causal_lanes_enabled() && words.size() >= 3) {
                std::vector<std::string> cps;
                for (const auto& cp : si::roles::causal_pairs_of(words))
                    if (si_.has(cp)) cps.push_back(cp);
                if (!cps.empty()) si_.inject(cps, state_dose(words) * kCausalDose);
            }
            si_.settle();
            for (std::size_t i = 0; i < n; ++i)
                field_sum[i] += si_.node_energy(static_cast<si::NodeId>(i));
            ++channels;
            si_.clear_channel_gate();
        }
        if (channels == 0) return false;
        const float inv = 1.0f / static_cast<float>(channels);
        for (std::size_t i = 0; i < n; ++i) field_sum[i] *= inv;
        si_.restore_field(field_sum);
        if (usage) {
            usage->multi_vector_active = true;
            usage->multi_vector_channels = channels;
        }
        return true;
    }

    // -- DECISION ----------------------------------------------------------------
    // v3.2.1: the retrieval-priming prologue is SHARED by decide() and
    // harvest_rows(). Bug this fixes: calibration was fitted on a field that
    // never saw the prime (harvest re-implemented injection inline and skipped
    // the block), so every fabric WITH memories was calibrated on a different
    // physics composition than the one decide() runs — measured consequence:
    // 96.6% accuracy at mean confidence 0.0016 on a strong fabric. The fit and
    // the decision must see the same field composition, bit for bit.
    // v3.9.0 (MAA §2): fracture unknown tokens, match fragments against
    // interned names (node-id ascending — deterministic), inject the matched
    // anchors at half the state dose. Returns the unknown tokens bridged.
    std::vector<std::string> fragment_abduce(const std::vector<std::string>& words,
                                             float dose) {
        std::vector<std::string> bridged;
        std::vector<std::string> anchors;
        for (const auto& w : words) {
            if (si_.has(w)) continue;                        // known: not an orphan
            if (w.size() < 3) continue;
            std::vector<std::string> frags;
            frags.push_back(w.substr(0, 3));                 // 3-char prefix
            if (w.size() >= 6) frags.push_back(w.substr(w.size() - 3, 3));  // suffix
            for (std::size_t i = 1; i + 3 <= w.size() && frags.size() < 8; ++i)
                frags.push_back(w.substr(i, 3));             // middle 3-grams
            bool hit = false;
            for (const auto& f : frags) {
                for (std::size_t id = 0; id < si_.node_count(); ++id) {
                    const std::string& lbl = si_.label_of(static_cast<si::NodeId>(id));
                    if (lbl.size() < 3) continue;
                    if (lbl.find(f) == std::string::npos) continue;
                    anchors.push_back(lbl);
                    hit = true;
                    break;                                   // first match per fragment
                }
                if (anchors.size() >= 8) break;              // bounded hypothesized set
            }
            if (hit) bridged.push_back(w);
        }
        if (!anchors.empty()) si_.inject(anchors, dose * 0.5f);   // hypothesized dose
        return bridged;
    }

    // v3.9.0 (P-CMA CEM shell, gradient-free): sample deterministic source-cap
    // perturbations of the SAME injected field, re-settle, score by the first
    // choice question's top-2 margin, keep the top-10% elites per round, refit
    // the cap draw around the elite mean, and finally RESTORE the best
    // variant's field (or the baseline when planning cannot beat it). No
    // gradients, no ML — rollout IS the settle physics, unchanged per variant.
    // Returns the best margin found; the winning field is left in the fabric.
    float cem_plan(const std::vector<std::string>& words,
                   const std::vector<std::pair<std::string, std::string>>& probes,
                   Usage& usage) {
        if (cem_presettle_.empty() || probes.empty()) return -1.0f;
        auto margin_of = [&](const std::vector<std::pair<std::string, float>>& es)
            -> float {
            if (es.size() < 2) return 0.0f;
            float p1 = -1e30f, p2 = -1e30f;
            for (const auto& e : es) {
                if (e.second > p1) { p2 = p1; p1 = e.second; }
                else if (e.second > p2) p2 = e.second;
            }
            return p1 - p2;
        };
        const float saved_cap = si_.last_source_cap();
        const float hi = std::max(1.0f, saved_cap);
        const float lo = hi >= 5.0f ? hi - 4.0f : 1.0f;
        const std::uint64_t base_hash = si_.state_hash();
        const float base_margin = margin_of(read_probes(probes));
        float best_margin = base_margin;
        float best_cap = saved_cap;
        std::vector<float> best_field;
        float mu = hi;                                 // elite-mean refit seed
        for (int round = 0; round < cem_rounds_; ++round) {
            std::vector<std::pair<float, float>> scored;   // (margin, cap)
            for (int v = 0; v < cem_variants_; ++v) {
                const std::uint64_t h = base_hash
                    ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(v + 1))
                    ^ (0xBF58476D1CE4E5B9ull * static_cast<std::uint64_t>(round + 1));
                const float cap = lo + static_cast<float>(h % 5);   // [lo, hi] — miller span
                si_.restore_field(cem_presettle_);
                si_.set_source_cap(cap);
                si_.settle();
                const float m = margin_of(read_probes(probes));
                scored.emplace_back(m, cap);
                if (m > best_margin) { best_margin = m; best_cap = cap;
                                       best_field = si_.snapshot_field(); }
            }
            // elites: top 10% (>= 1); refit the draw mean for the next round
            std::sort(scored.begin(), scored.end(),
                      [](const auto& x, const auto& y){ return x.first > y.first; });
            const std::size_t n_elite = std::max<std::size_t>(
                1, scored.size() / 10);
            float acc = 0.0f;
            for (std::size_t i = 0; i < n_elite; ++i) acc += scored[i].second;
            mu = acc / static_cast<float>(n_elite);
            (void)mu;   // span is the miller window; the refit caps the draw seed
        }
        // restore: best variant's field if it beats the baseline, else baseline
        si_.set_source_cap(saved_cap);
        if (!best_field.empty()) {
            si_.restore_field(best_field);
        } else {
            si_.restore_field(cem_presettle_);
            si_.set_source_cap(saved_cap);
            si_.settle();                              // re-run the baseline settle
        }
        usage.cem_active = true;
        usage.cem_rounds = cem_rounds_;
        usage.cem_gain = best_margin - base_margin;
        return best_margin;
    }

    // read the (label, description) probe pairs exactly as decide_choice does
    std::vector<std::pair<std::string, float>>
    read_probes(const std::vector<std::pair<std::string, std::string>>& probes) const {
        std::vector<std::pair<std::string, float>> es;
        for (const auto& p : probes)
            es.emplace_back(p.first, probe_energy(p.first, p.second));
        return es;
    }

    void prime_field(const std::vector<std::string>& words, Usage* usage) {
        si_.reset_field();
        if (retrieval_on_ && !memories_.empty() && retrieval_topk_ > 0) {
            const std::vector<float> fq = recall::fingerprint(si_, words);
            struct Hit { float r; std::size_t idx; };
            std::vector<Hit> hits;
            hits.reserve(memories_.size());
            for (std::size_t m = 0; m < memories_.size(); ++m) {
                const std::vector<float>& fm = memory_fps_[m];
                float dot = 0.0f;
                const std::size_t nn = std::min(fq.size(), fm.size());
                for (std::size_t i = 0; i < nn; ++i) dot += fq[i] * fm[i];
                hits.push_back({dot, m});
            }
            std::stable_sort(hits.begin(), hits.end(),
                             [](const Hit& x, const Hit& y) { return x.r > y.r; });
            const std::size_t kk = std::min<std::size_t>(static_cast<std::size_t>(retrieval_topk_), hits.size());
            si_.reset_field();                       // fresh field: prime + state only
            const float dose = retrieval_dose_ * si_.config().inject_energy;
            for (std::size_t i = 0; i < kk; ++i) {
                if (hits[i].r <= 0.01f) continue;    // a whisper of resonance primes nothing
                const recall::Memory& m = memories_[hits[i].idx];
                std::vector<std::string> prime;
                for (const auto& t : si::norm::normalize(m.label))
                    if (si_.has(t)) prime.push_back(t);
                if (!prime.empty()) si_.inject(prime, dose);
                if (usage) usage->retrieved.emplace_back(m.label, hits[i].r);
            }
        }
    }

    std::vector<Answer> decide(const std::string& state, const sfx::JV& questions, Usage& usage) {
        usage = Usage{};
        usage.vocabulary = si_.node_count();
        usage.lanes = si_.lane_count();

        std::vector<Answer> answers;
        const std::vector<std::string> words = si::norm::normalize(state);
        usage.state_tokens = words.size();

        // v3.8.0 (TASK 1) adaptive depth: measure the problem BEFORE any
        // settle; when the flag is on, harder states get more passes.
        // Disclosure lands in usage either way; with the flag off the base
        // k_settle is untouched (bit-identical replay).
        usage.adaptive_depth = adaptive_depth_on_;
        usage.k_base = si_.config().k_settle;
        usage.k_used = apply_adaptive_depth(words, &usage.difficulty);

        // v3.7.0 (Move 4) — generalized tool scope: the field's job stops at a
        // relevance heuristic; a detected computation is handed to the VERIFIED
        // oracle and the derivation is disclosed. The field still answers or
        // defers exactly as it measures below — this is the calc pattern made
        // first-class at decide time: field refuses/defers honestly, tool
        // computes exactly, register carries the provenance.
        if (tools_check_on_) {
            usage.tool_checked = true;
            const calc::ArithDetect ad = calc::detect(state);
            if (ad.has_arithmetic) {
                const calc::CalcResult cr = calc::evaluate(ad.expression);
                if (cr.ok) {
                    usage.tool_used = true;
                    usage.tool_name = "calc";
                    usage.tool_expression = ad.expression;
                    usage.tool_value = cr.text();
                }
            }
        }

        // v3.4 question-context gating, STAGE 1 (opt-in): settle the question's
        // own tokens (instructions + criteria descriptions, no labels) into a
        // context field. The question drives the fabric's lanes from ITS side,
        // so question-relevant entities carry energy before the state even
        // arrives. Deterministic; settle() untouched.
        std::vector<float> ctx_field;
        if (ctx_gate_on_) {
            std::vector<std::string> qtoks;
            for (const auto& qkv : questions.obj) {
                const sfx::JV& q = qkv.second;
                if (q.has("instructions"))
                    for (const auto& w : si::norm::normalize(q.at("instructions").as_str()))
                        qtoks.push_back(w);
                if (q.has("criteria")) {
                    const sfx::JV& crit = q.at("criteria");
                    if (crit.is_obj())
                        for (const auto& kv : crit.obj)
                            for (const auto& w : si::norm::normalize(kv.second.as_str()))
                                qtoks.push_back(w);
                    else if (crit.is_arr())
                        for (const auto& v : crit.arr)
                            for (const auto& w : si::norm::normalize(v.as_str()))
                                qtoks.push_back(w);
                }
            }
            if (!qtoks.empty()) {
                si_.reset_field();
                si_.inject(qtoks, state_dose(qtoks));
                si_.settle();
                ctx_field.resize(si_.node_count());
                for (std::size_t i = 0; i < ctx_field.size(); ++i)
                    ctx_field[i] = si_.node_energy(static_cast<si::NodeId>(i));
            }
        }

        // v3.2 retrieval-by-default: if the model ships memories, the query's
        // settled-field fingerprint ranks them (Hopfield-style resonance,
        // recall.hpp) and the top-k outcomes inject a faint prime dose before
        // the state settles. The prime is part of the decision composition:
        // same state + same memories => same field, bit for bit. No memories
        // (or --no-retrieval) => this block primes nothing.
        prime_field(words, &usage);
        si_.inject(words, state_dose(words));          // words at the (gained) substrate level
        // v3.6.0 (MRS §4b): inject the state's ORDERED BIGRAM nodes that the
        // fabric actually carries. Old fabrics contain no "w1~w2" nodes, so
        // `has()` filters everything and this is a no-op — bit-identical
        // replay. New fabrics get order-sensitive energy: "a~beat" reaches
        // the anchor that "beat~a" does not, which is the multiset escape.
        std::size_t bigram_hits = 0;
        if (si::norm::bigrams_enabled() && words.size() >= 2) {
            std::vector<std::string> bgs;
            for (const auto& bg : si::norm::bigrams_of(words))
                if (si_.has(bg)) bgs.push_back(bg);
            bigram_hits = bgs.size();
            if (!bgs.empty()) si_.inject(bgs, state_dose(words) * kBigramDose);
        }
        usage.bigram_tokens = bigram_hits;
        // v3.7.0 (Move 3): inject the state's TYPED pair nodes that the fabric
        // actually carries. Fabrics learned with typed lanes off contain no
        // typed nodes -> has() filters everything -> no-op, bit-identical
        // replay (unit-tested both ways, same contract as the bigrams).
        std::size_t typed_hits = 0;
        if (si::roles::typed_lanes_enabled() && words.size() >= 2) {
            std::vector<std::string> tps;
            for (const auto& tp : si::roles::typed_pairs_of(words))
                if (si_.has(tp)) tps.push_back(tp);
            typed_hits = tps.size();
            if (!tps.empty()) si_.inject(tps, state_dose(words) * kTypedDose);
        }
        usage.typed_tokens = typed_hits;
        // v3.8.0 (TASK 3): inject the state's CAUSAL pair nodes the fabric
        // actually carries. Fabrics learned with causal lanes off contain no
        // ?> / !> nodes -> has() filters everything -> no-op (replay).
        std::size_t causal_hits = 0;
        if (si::roles::causal_lanes_enabled() && words.size() >= 3) {
            std::vector<std::string> cps;
            for (const auto& cp : si::roles::causal_pairs_of(words))
                if (si_.has(cp)) cps.push_back(cp);
            causal_hits = cps.size();
            if (!cps.empty()) si_.inject(cps, state_dose(words) * kCausalDose);
        }
        usage.causal_tokens = causal_hits;
        // v3.9.0 (RB surrogate) decide-side pro/con injection: when the state
        // carries a negation cue and the fabric carries the mangled con-nodes,
        // inject them at the typed dose. Fabrics learned without --pro-con
        // contain no "!token" nodes -> has() filters everything -> no-op.
        if (pro_con_ && si::roles::verb_negated(words)) {
            std::vector<std::string> negs;
            for (const auto& w : words) {
                const std::string neg = "!" + w;
                if (si_.has(neg)) negs.push_back(neg);
            }
            if (!negs.empty()) si_.inject(negs, state_dose(words) * kTypedDose);
        }
        // v3.9.0 (MAA §2) FRAGMENT ABDUCTION (opt-in): unknown state tokens
        // are orphan nodes — degree 0, zero lanes, honest deferral. MAA
        // fractures them into sub-lexical fragments, matches the fragments
        // against interned names, and energizes the matched anchors at a
        // reduced hypothesized dose. Volatile by design: NO lane writes at
        // decide time, disclosed as hyp_tokens in the usage.
        if (fragments_on_)
            usage.hyp_tokens = fragment_abduce(words, state_dose(words));
        if (si::norm::grams_enabled()) {                 // bridges for corrupted forms, gated
            std::vector<std::string> bridges;
            for (const auto& w : words)
                if (!si_.has(w) && subword_traction(si_, w))
                    for (const auto& g : si::norm::expand_ngrams({w}))
                        if (si_.has(g)) bridges.push_back(g);
            si_.inject(bridges, kBridgeEnergy);
            usage.state_tokens += bridges.size();
        }
        // v3.9.0 (P-CMA CEM shell): keep the PRE-settle field when planning is
        // on — each planning variant restores it, draws a deterministic source
        // cap, and re-settles. (A first cut snapshotted AFTER the main settle:
        // every variant then settled an ALREADY-SETTLED field, double-decayed
        // the energies, and deferred everything — caught by the think30
        // bisection, where --cem-plan alone dropped accuracy 0.600 -> 0.267.)
        // The plan can only REPLACE the field, never mutate the physics.
        if (cem_on_) cem_presettle_ = si_.snapshot_field();
        si_.settle();
        // v3.8.0 (TASK 4): capture the MAIN settle's self-verification trace
        // before any second settle (perturb check) can overwrite it.
        {
            const auto& tr = si_.last_settle_trace();
            usage.field_passes = tr.passes;
            usage.field_converged = tr.eps_break;
            usage.field_motion_rate = tr.motion_rate();
        }
        // v3.8.0 (TASK 2): multi-vector state — settle the same injection
        // under K spectral channel gates and superpose the settled fields.
        // No-op (honestly disclosed) on fabrics without a distvec field.
        apply_multi_vector(words, &usage);
        // v3.4 question-context gating, STAGE 2: compose the settled state
        // field with the settled question field,
        // e = (1-alpha)*state + alpha*context (alpha default 0.5). The field
        // is measured afterward exactly as always; the composition is
        // disclosed as ctx_gate in the usage block.
        if (!ctx_field.empty()) {
            si_.blend_field(ctx_field, ctx_alpha_);
            usage.ctx_gate = true;
            usage.ctx_alpha = ctx_alpha_;
        }
        usage.settled_energy = si_.total_energy();
        // Honest silence: nothing settled => refuse to guess.
        bool silent = usage.settled_energy < si_.config().silence_floor;
        // v3.6.0 — BED §8 PERTURBATION-CONTRAST CHECK (opt-in,
        // set_perturb_check). The fabric is settled a SECOND time on a
        // structure-broken copy of the state: the same token multiset with
        // adjacent pairs swapped (deterministic). If the broken state's
        // winning margin matches the real state's, the readout was carried by
        // the token BAG, not by structure — BED's "the potential was trained
        // as a classifier" diagnosis, run per decision. Choice questions then
        // defer with reason "perturbation_tie" instead of shipping a
        // bag-of-words coin flip. The real field is snapshotted and restored,
        // so the readout below measures exactly what an unchecked decide()
        // measures. noul/score keep their answers (their support readouts are
        // bag statistics by design); the disclosure lands in usage.
        bool perturb_defer = false;
        if (perturb_on_ && !silent && words.size() >= 2) {
            const sfx::JV* first_choice = nullptr;
            for (const auto& qkv : questions.obj)
                if (qkv.second.has("type") && qkv.second.at("type").as_str() == "choice"
                    && qkv.second.has("criteria")) { first_choice = &qkv.second; break; }
            if (first_choice) {
                auto margins = [this](const sfx::JV& q) -> std::pair<float, float> {
                    std::vector<float> es;
                    const sfx::JV& crit = q.at("criteria");
                    if (crit.is_obj())
                        for (const auto& kv : crit.obj)
                            es.push_back(probe_energy(kv.first, kv.second.as_str()));
                    else if (crit.is_arr())
                        for (const auto& v : crit.arr)
                            es.push_back(probe_energy(v.as_str(), ""));
                    if (es.size() < 2) return {0.0f, 0.0f};
                    std::partial_sort(es.begin(), es.begin() + 2, es.end(), std::greater<float>());
                    return {es[0], es[1]};
                };
                const auto real_m = margins(*first_choice);
                const std::vector<float> saved = si_.snapshot_field();
                std::vector<std::string> pw = words;             // deterministic structure break:
                for (std::size_t i = 0; i + 1 < pw.size(); i += 2)
                    std::swap(pw[i], pw[i + 1]);                 // adjacent transposition pairs
                si_.reset_field();
                si_.inject(pw, state_dose(pw));
                if (si::norm::bigrams_enabled()) {
                    std::vector<std::string> pbgs;
                    for (const auto& bg : si::norm::bigrams_of(pw))
                        if (si_.has(bg)) pbgs.push_back(bg);
                    if (!pbgs.empty()) si_.inject(pbgs, state_dose(pw) * kBigramDose);
                }
                // v3.7.0: typed pairs of the BROKEN stream, mirrored exactly —
                // the perturbation field must see the same composition rules.
                if (si::roles::typed_lanes_enabled()) {
                    std::vector<std::string> ptps;
                    for (const auto& tp : si::roles::typed_pairs_of(pw))
                        if (si_.has(tp)) ptps.push_back(tp);
                    if (!ptps.empty()) si_.inject(ptps, state_dose(pw) * kTypedDose);
                }
                // v3.8.0 (TASK 3): causal pairs of the broken stream, mirrored.
                if (si::roles::causal_lanes_enabled()) {
                    std::vector<std::string> pcps;
                    for (const auto& cp : si::roles::causal_pairs_of(pw))
                        if (si_.has(cp)) pcps.push_back(cp);
                    if (!pcps.empty()) si_.inject(pcps, state_dose(pw) * kCausalDose);
                }
                si_.settle();
                const auto pert_m = margins(*first_choice);
                si_.restore_field(saved);                        // measure what decide() measured
                usage.perturb_check = true;
                usage.perturb_margin_real = real_m.first - real_m.second;
                usage.perturb_margin_broken = pert_m.first - pert_m.second;
                perturb_defer = pert_m.first - pert_m.second >= real_m.first - real_m.second;
            }
        }
        // v3.3.1 readout-silence visibility (opt-in): SYFOX_DEBUG_READOUT=1
        // prints the decide-level field so honest_silence vs readout-level
        // unknown_candidates deferral is measurable, not guessed.
        // v3.9.0 (P-CMA CEM shell, opt-in): plan over deterministic source-cap
        // variants of the SAME injected field before any readout. Probes come
        // from the first choice question (same first-choice convention as the
        // perturb check). The plan RESTORES the winning field — readouts below
        // measure the planned field; when no variant beats the baseline margin
        // the baseline settle is re-run, so cem_off == cem_on-with-no-gain.
        if (cem_on_ && !silent) {
            const sfx::JV* first_choice = nullptr;
            for (const auto& qkv : questions.obj)
                if (qkv.second.has("type") && qkv.second.at("type").as_str() == "choice"
                    && qkv.second.has("criteria")) { first_choice = &qkv.second; break; }
            if (first_choice) {
                std::vector<std::pair<std::string, std::string>> probes;
                const sfx::JV& crit = first_choice->at("criteria");
                if (crit.is_obj())
                    for (const auto& kv : crit.obj)
                        probes.emplace_back(kv.first, kv.second.as_str());
                else if (crit.is_arr())
                    for (const auto& v : crit.arr)
                        probes.emplace_back(v.as_str(), "");
                if (probes.size() >= 2) cem_plan(words, probes, usage);
            }
        }
        if (vght_gate_) {
            usage.vght_gate = true;
            usage.vght_gated_frac = si_.vght_gated_frac();
        }
        if (std::getenv("SYFOX_DEBUG_READOUT") != nullptr)
            std::fprintf(stderr, "DEBUG: settled_energy=%.4f silence_floor=%.3f\n",
                         usage.settled_energy, si_.config().silence_floor);

        for (const auto& qkv : questions.obj) {
            const sfx::JV& q = qkv.second;
            Answer a;
            a.qid = qkv.first;
            a.type = q.at("type").as_str();
            if (silent) {
                a.deferred = true;
                a.reason = words.empty() ? "empty_state"
                         : (usage.settled_energy <= 0.0f ? "unknown_vocabulary" : "honest_silence");
                answers.push_back(a);
                continue;
            }
            if (a.type == "choice")   decide_choice(q, words, a, &usage);
            else if (a.type == "score") decide_score(q, a);
            else if (a.type == "noul")  decide_noul(q, a);
            // v3.6.0: the perturbation-contrast verdict overrides a carried
            // choice answer only when the broken state matched or beat the
            // real margin — the readout had no order signal to stand on.
            if (a.type == "choice" && !a.deferred && perturb_defer) {
                a.deferred = true;
                a.reason = "perturbation_tie";
            }
            // v3.8.0 (TASK 4): self-verification — a choice answer measured on
            // a field that was STILL MOVING above the floor is a transient,
            // not an attractor reading. Opt-in; disclose always.
            if (a.type == "choice" && !a.deferred && self_verify_on_
                && usage.field_motion_rate > self_verify_floor_) {
                a.deferred = true;
                a.reason = "unconverged_field";
            }
            // v3.9.0 (RIFA SIL): a state with >= 2 unresolved contradictions in
            // its history is a known conflict zone — abstain earlier instead of
            // answering a contested readout. Opt-in; tension disclosed always.
            if (tension_on_) {
                usage.tension_gate = true;
                usage.tension = tension_acc_;
                const auto cit = conflicts_by_state_.find(
                    stream_hash(si::norm::normalize(state)));
                if (a.type == "choice" && !a.deferred
                    && cit != conflicts_by_state_.end() && cit->second.size() >= 2) {
                    a.deferred = true;
                    a.reason = "conflicted_state";
                }
            }
            // v3.9.0 (fixes.md) OOD DENSITY GATE: the decision signature
            // [settled, top, margin, unk] compared against every signature
            // recorded from ANSWERED states at calibrate time; far from all of
            // them => this field shape has never answered above-floor => defer.
            if (a.type == "choice" && !a.deferred && ood_knn_on_ && !ood_sigs_.empty()) {
                const std::array<float, 4> sig{usage.settled_energy,
                                               ood_last_top_, ood_last_margin_,
                                               ood_last_unk_};
                float best = -1.0f;
                for (const auto& s : ood_sigs_) {
                    float d = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        const float diff = sig[static_cast<std::size_t>(k)] - s[static_cast<std::size_t>(k)];
                        d += diff * diff;
                    }
                    d = std::sqrt(d);
                    if (best < 0.0f || d < best) best = d;
                }
                usage.ood_knn = true;
                usage.ood_knn_dist = best;
                if (best > ood_knn_tau_) {
                    a.deferred = true;
                    a.reason = "ood_density";
                }
            }
            answers.push_back(a);
        }
        return answers;
    }

    // Harvest calibration rows from labelled examples (tool side).
    std::vector<CalibRow> harvest_rows(const std::vector<sfx::JV>& examples) {
        std::vector<CalibRow> rows;
        for (const auto& ex : examples) {
            std::string state = ex.at("state").as_str();
            const sfx::JV& qs = ex.has("questions") ? ex.at("questions") : sfx::JV(sfx::JVObj{});
            if (!qs.is_obj()) continue;
            // v3.2.1: prime EXACTLY as decide() does (shared prime_field) —
            // the fit must see the decide-time field composition. For fabrics
            // without memories this is a bare reset, the pre-3.2 behavior.
            const std::vector<std::string> hw0 = si::norm::normalize(state);
            prime_field(hw0, nullptr);
            // v3.8.0 (TASK 1) fit/decide parity: the calibration fit must see
            // the SAME adaptive depth the decision will use.
            apply_adaptive_depth(hw0, nullptr);
            {
                si_.inject(hw0, state_dose(hw0));         // gained dose (Milestone-1)
            }
            // v3.6.0: bigram injection mirrored from decide() — the v3.2.1
            // lesson (the fit and the decision must see the same field
            // composition, bit for bit).
            if (si::norm::bigrams_enabled() && hw0.size() >= 2) {
                std::vector<std::string> hbgs;
                for (const auto& bg : si::norm::bigrams_of(hw0))
                    if (si_.has(bg)) hbgs.push_back(bg);
                if (!hbgs.empty()) si_.inject(hbgs, state_dose(hw0) * kBigramDose);
            }
            // v3.7.0: typed pair injection mirrored from decide() — same
            // fit/decide parity lesson.
            if (si::roles::typed_lanes_enabled() && hw0.size() >= 2) {
                std::vector<std::string> htps;
                for (const auto& tp : si::roles::typed_pairs_of(hw0))
                    if (si_.has(tp)) htps.push_back(tp);
                if (!htps.empty()) si_.inject(htps, state_dose(hw0) * kTypedDose);
            }
            // v3.8.0 (TASK 3): causal pair injection mirrored from decide().
            if (si::roles::causal_lanes_enabled() && hw0.size() >= 3) {
                std::vector<std::string> hcps;
                for (const auto& cp : si::roles::causal_pairs_of(hw0))
                    if (si_.has(cp)) hcps.push_back(cp);
                if (!hcps.empty()) si_.inject(hcps, state_dose(hw0) * kCausalDose);
            }
            // v3.9.0 (RB surrogate): pro/con decide-side injection, mirrored.
            if (pro_con_ && si::roles::verb_negated(hw0)) {
                std::vector<std::string> negs;
                for (const auto& w : hw0) {
                    const std::string neg = "!" + w;
                    if (si_.has(neg)) negs.push_back(neg);
                }
                if (!negs.empty()) si_.inject(negs, state_dose(hw0) * kTypedDose);
            }
            if (si::norm::grams_enabled()) {
                const std::vector<std::string> words2 = si::norm::normalize(state);
                std::vector<std::string> bridges;
                for (const auto& w : words2)
                    if (!si_.has(w) && subword_traction(si_, w))
                        for (const auto& g : si::norm::expand_ngrams({w}))
                            if (si_.has(g)) bridges.push_back(g);
                si_.inject(bridges, kBridgeEnergy);
            }
            si_.settle();
            // v3.8.0 (TASK 2) fit/decide parity: the fit sees the same
            // multi-vector superposition the decision sees.
            apply_multi_vector(hw0, nullptr);
            if (si_.silent()) continue;
            for (const auto& qkv : qs.obj) {
                const sfx::JV& q = qkv.second;
                std::string type = q.at("type").as_str();
                std::string label = ex.at("labels").at(qkv.first).as_str();   // correct label / level / "true"/"false"
                if (type == "choice" || type == "score") {
                    const sfx::JV& crit = q.at("criteria");
                    std::vector<std::pair<std::string, float>> energies;
                    if (crit.is_obj())
                        for (const auto& kv : crit.obj)
                            energies.emplace_back(kv.first, probe_energy(kv.first, kv.second.as_str()));
                    else if (crit.is_arr())
                        for (std::size_t i = 0; i < crit.arr.size(); ++i)
                            energies.emplace_back(crit.arr[i].as_str(), probe_energy(crit.arr[i].as_str(), ""));
                    // resolve the gold label to a candidate KEY: for array
                    // criteria the stored label may be the level INDEX ("0",
                    // "1", ...) — the old code compared it to the level text
                    // and silently dropped every score row from the fit
                    std::string gold = label;
                    if (crit.is_arr()) {
                        bool is_text = false;
                        for (const auto& v : crit.arr) if (v.as_str() == label) { is_text = true; break; }
                        if (!is_text) {
                            long idx = std::strtol(label.c_str(), nullptr, 10);
                            if (idx >= 0 && idx < static_cast<long>(crit.arr.size()))
                                gold = crit.arr[static_cast<std::size_t>(idx)].as_str();
                        }
                    }
                    CalibRow r; r.type = type; r.label = gold; r.energies = energies;
                    rows.push_back(r);
                    // v3.9.0 (fixes.md): record the ANSWERED decision signature
                    // [settled, top, margin, unk] — the in-distribution store
                    // the OOD kNN gate compares against at decide time.
                    {
                        float top = 0.0f, second = 0.0f;
                        for (const auto& e : energies) {
                            if (e.second > top) { second = top; top = e.second; }
                            else if (e.second > second) second = e.second;
                        }
                        std::size_t unk = 0;
                        for (const auto& w : hw0) if (!si_.has(w)) ++unk;
                        const float unk_frac = hw0.empty() ? 0.0f
                            : static_cast<float>(unk) / static_cast<float>(hw0.size());
                        record_ood_signature(si_.total_energy(), top, top - second, unk_frac);
                    }
                } else if (type == "noul") {
                    float support = noul_support(q.at("instructions").as_str());
                    CalibRow r; r.type = "noul";
                    r.support = support;
                    r.y = (label == "true") ? 1 : 0;
                    rows.push_back(r);
                }
            }
        }
        return rows;
    }

    // =====================================================================
    // v3.7.0 (Move 2) — ENERGY-SPACE SELF-SUPERVISION: the pretrain loop.
    // The direct analog of next-token prediction, expressed as lane physics:
    //   1. take UNLABELED raw text (one line = one stream),
    //   2. hide one token (deterministic stride schedule),
    //   3. inject the rest, settle,
    //   4. read out which BASIN got the energy — the true token's node
    //      against the strongest non-context node in the settled field,
    //   5. strengthen the context->true-token lanes in proportion to the
    //      FAILURE (error-weighted Hebbian): a basin that already wins binds
    //      at its residual; a basin that lost binds hard.
    // The loss is measurable and reported per epoch: mean of
    //   loss = 1 - e_true / (e_true + e_best_other)
    // over every masked position. No labels, no gradients, no fitted
    // parameters — the existing Hebbian rule, gated by a measured field
    // signal. Deterministic end to end: lines in file order, positions in
    // stream order, no RNG anywhere.
    //
    // This is what makes raw text into training signal, removing the
    // dependence on labeled corpora for VOCABULARY and CO-OCCURRENCE: words
    // that never appear in a labeled lesson become nodes with lanes and
    // resonance structure here, which is exactly what the distributional
    // field (Move 1) then factorizes.
    struct PretrainReport {
        std::size_t lines = 0;            // lines consumed
        std::size_t tokens = 0;           // total tokens seen
        std::size_t interned_new = 0;     // vocabulary growth (first epoch)
        std::size_t masked = 0;           // masked positions settled
        std::vector<double> epoch_loss;   // mean basin loss per epoch
        std::size_t vocab_after = 0;
        sfx::JV to_json() const {
            sfx::JVObj o{{"lines", static_cast<double>(lines)},
                         {"tokens", static_cast<double>(tokens)},
                         {"interned_new", static_cast<double>(interned_new)},
                         {"masked", static_cast<double>(masked)},
                         {"vocab_after", static_cast<double>(vocab_after)}};
            sfx::JVArr losses;
            for (double l : epoch_loss) losses.push_back(sfx::JV(l));
            o["epoch_mean_loss"] = sfx::JV(std::move(losses));
            return sfx::JV(std::move(o));
        }
    };

    PretrainReport pretrain(const std::string& corpus_path, int epochs,
                            std::size_t mask_stride, float eta_scale) {
        PretrainReport rep;
        if (epochs < 1) epochs = 1;
        if (mask_stride < 1) mask_stride = 1;
        std::ifstream in(corpus_path);
        if (!in) return rep;
        std::vector<std::vector<std::string>> lines;
        std::string raw;
        while (std::getline(in, raw)) {
            if (raw.empty()) continue;
            std::vector<std::string> w = si::norm::normalize(raw);
            if (w.size() < 3) continue;           // need context + target
            lines.push_back(std::move(w));
        }
        rep.lines = lines.size();
        for (std::size_t e = 0; e < static_cast<std::size_t>(epochs); ++e) {
            double loss_sum = 0.0;
            std::size_t loss_n = 0;
            for (const auto& words : lines) {
                // vocabulary growth happens on sight of the line (every epoch;
                // intern() is idempotent, the counter only counts the first)
                for (const auto& t : words) {
                    ++rep.tokens;
                    if (!si_.has(t)) { si_.intern(t); ++rep.interned_new; }
                }
                // df: this line is a lesson for the rarity weighting too
                si_.begin_df_lesson();
                {
                    std::unordered_set<std::string> seen;
                    for (const auto& t : words)
                        if (seen.insert(t).second && si_.has(t)) si_.note_df(si_.find(t));
                }
                for (std::size_t i = 0; i < words.size(); i += mask_stride) {
                    // context = the line minus the masked slot
                    std::vector<std::string> ctx;
                    ctx.reserve(words.size() - 1);
                    for (std::size_t j = 0; j < words.size(); ++j)
                        if (j != i) ctx.push_back(words[j]);
                    const std::string& target = words[i];
                    si_.reset_field();
                    si_.inject(ctx, si_.config().inject_energy);
                    si_.settle();
                    // basin readout: true node vs strongest non-context node
                    const si::NodeId tid = si_.find(target);
                    std::unordered_set<si::NodeId> ctx_ids;
                    for (const auto& c : ctx) ctx_ids.insert(si_.find(c));
                    float e_true = si_.node_energy(tid);
                    float e_other = 0.0f;
                    for (si::NodeId id = 0; id < si_.node_count(); ++id) {
                        if (id == tid || ctx_ids.count(id)) continue;
                        const float en = si_.node_energy(id);
                        if (en > e_other) e_other = en;
                    }
                    float loss;
                    if (e_true <= 0.0f && e_other <= 0.0f) loss = 1.0f;   // nothing lit
                    else if (e_other <= 0.0f) loss = 0.0f;                // true basin won outright
                    else loss = 1.0f - e_true / (e_true + e_other);
                    // error-weighted Hebbian: strengthen what failed, scaled
                    // by the operator's eta_scale, capped at full strength
                    const float scale = std::min(kPretrainEtaCap, std::max(0.0f, loss * eta_scale));
                    if (scale > 0.0f) si_.hebbian_lesson(ctx, {target}, scale);
                    loss_sum += loss;
                    ++loss_n;
                    ++rep.masked;
                }
            }
            rep.epoch_loss.push_back(loss_n > 0 ? loss_sum / static_cast<double>(loss_n) : 0.0);
        }
        rep.vocab_after = si_.node_count();
        return rep;
    }

private:
    // -- Milestone 3 audit internals -------------------------------------------
    struct Taught { std::uint64_t seq; std::string outcome; };

    // 64-bit hashes travel as HEX STRINGS in JSON: the house dump is %.6g and
    // doubles lose integer precision past 2^53 — a numeric hash silently
    // corrupts on every save/load round-trip (measured: reloaded conflict
    // keys no longer matched the decide-side hash).
    static std::string hex64(std::uint64_t v) {
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
        return buf;
    }
    static std::uint64_t hex_to_u64(const std::string& s) {
        return std::stoull(s, nullptr, 16);
    }

    static std::uint64_t stream_hash(const std::vector<std::string>& ws) {
        std::uint64_t h = 1469598103934665603ull;
        for (const auto& w : ws) h = h * 1099511628211ull + static_cast<std::uint64_t>(si::fnv1a_hash(w));
        return h;
    }

    // A contradicting lesson is COUNTER-EVIDENCE for the binding it disputes:
    // mark every existing lane from this state's words into the OLD outcome's
    // token set. Bookkeeping only — the physics is untouched.
    void mark_counter_lanes(const std::vector<std::string>& words,
                            const std::string& old_outcome, std::uint64_t seq) {
        std::vector<std::string> old_toks = si::norm::normalize(old_outcome);
        for (const auto& w : words) {
            if (!si_.has(w)) continue;
            const si::NodeId a = si_.find(w);
            std::vector<std::pair<si::NodeId, float>> lanes;
            si_.lanes_of(a, lanes);
            for (const auto& l : lanes)
                for (const auto& t : old_toks)
                    if (si_.has(t) && si_.find(t) == l.first)
                        si_.record_counter(a, l.first, seq, context_);
        }
    }

    // -- readout helpers --------------------------------------------------------
    float probe_energy(const std::string& label, const std::string& desc) const {
        return si_.readout(si::norm::normalize(label + " " + desc));
    }

    float noul_support(const std::string& instructions) const {
        // Noul uses the SUPPORT readout lane: breadth of corroboration.
        float e = si_.readout(si::norm::normalize(instructions), si::Substrate::ReadoutMode::Support);
        float total = si_.total_energy();
        if (total <= 0.0f) return 0.0f;
        float support = e / (total * 1.5f);                 // ratio in ~[0,1+]
        return std::min(1.5f, support);
    }

public:
    // debug/eval accessor (read-only)
    float noul_support_public(const std::string& instructions) const { return noul_support(instructions); }

public:
    // -- Milestone 3: machine-auditable evidence for a decision ------------------
    // Built AFTER decide(): per question, the winning candidate's supporting
    // lanes (weight, generation, support/counter events, provenance window,
    // context) and any contradiction records for this exact state. Read-only.
    sfx::JV evidence_json(const std::string& state, const sfx::JV& questions,
                          const std::vector<Answer>& answers) const {
        const std::vector<std::string> words = si::norm::normalize(state);
        const std::uint64_t shash = stream_hash(words);
        sfx::JVArr contr;
        auto cit = conflicts_by_state_.find(shash);
        if (cit != conflicts_by_state_.end())
            for (std::size_t idx : cit->second)
                if (idx < conflicts_.size()) contr.push_back(conflicts_[idx]);

        sfx::JVObj qout;
        std::size_t ai = 0;
        for (const auto& qkv : questions.obj) {
            const sfx::JV& q = qkv.second;
            if (ai >= answers.size()) break;
            const Answer& a = answers[ai++];
            sfx::JVObj qe;
            qe["type"] = sfx::JV(a.type);
            qe["deferred"] = sfx::JV(a.deferred);
            if (a.deferred) { qe["reason"] = sfx::JV(a.reason); qout[qkv.first] = sfx::JV(qe); continue; }

            std::string argmax;
            if (a.type == "choice") argmax = a.choice;
            else if (a.type == "score") {
                float top = -1.0f;
                for (const auto& kv : a.probabilities) if (kv.second > top) { top = kv.second; argmax = kv.first; }
            } else {
                argmax = a.probability >= 0.5f ? "true" : "false";
                qe["probability"] = std::round(a.probability * 1000.0f) / 1000.0f;
            }
            qe["argmax"] = sfx::JV(argmax);

            // the probe tokens the readout actually measured for the winner
            std::string probe_text = argmax;
            if (q.has("criteria")) {
                const sfx::JV& crit = q.at("criteria");
                if (crit.is_obj() && crit.has(argmax)) probe_text = argmax + " " + crit.at(argmax).as_str();
                else if (crit.is_arr())
                    for (const auto& v : crit.arr) if (v.as_str() == argmax) probe_text = argmax;
            }
            const std::vector<std::string> probe = si::norm::normalize(probe_text);

            struct Sup { std::string lane; float w; si::LaneEvidence e; std::uint32_t gen; };
            std::vector<Sup> sups;
            for (const auto& w : words) {
                if (!si_.has(w)) continue;
                const si::NodeId a = si_.find(w);
                if (si_.node_energy(a) <= 0.0f) continue;
                for (const auto& p : probe) {
                    if (!si_.has(p)) continue;
                    const si::NodeId b = si_.find(p);
                    const float lw = si_.lane_weight(a, b);
                    if (lw <= 0.0f) continue;
                    Sup s; s.lane = w + " -> " + p; s.w = lw;
                    s.gen = si_.generation_of(a, b);
                    si_.evidence_of(a, b, s.e);
                    sups.push_back(s);
                }
            }
            std::sort(sups.begin(), sups.end(),
                      [](const Sup& x, const Sup& y) { return x.w > y.w; });
            sfx::JVArr lanes;
            for (std::size_t i = 0; i < sups.size() && i < 8; ++i)
                lanes.push_back(sfx::JV(sfx::JVObj{
                    {"lane", sfx::JV(sups[i].lane)},
                    {"weight", std::round(sups[i].w * 10000.0f) / 10000.0f},
                    {"generation", static_cast<double>(sups[i].gen)},
                    {"support_events", static_cast<double>(sups[i].e.support_events)},
                    {"counter_events", static_cast<double>(sups[i].e.counter_events)},
                    {"first_seq", static_cast<double>(sups[i].e.first_seq)},
                    {"last_seq", static_cast<double>(sups[i].e.last_seq)},
                    {"context", sfx::JV(sups[i].e.context)}}));
            qe["supporting_lanes"] = sfx::JV(lanes);
            qe["supporting_lane_total"] = static_cast<double>(sups.size());
            qout[qkv.first] = sfx::JV(qe);
        }
        return sfx::JV(sfx::JVObj{
            {"state_hash_hex", sfx::JV(hex64(shash))},
            {"contested", sfx::JV(!contr.empty())},
            {"contradictions", sfx::JV(contr)},
            {"questions", sfx::JV(qout)},
            {"note", sfx::JV("evidence is a read-only ledger view; lane weights drive the field, the ledger never does")}});
    }

private:

    static float softmax_ps(const std::vector<float>& e, std::vector<float>& p, float T) {
        float maxe = *std::max_element(e.begin(), e.end());
        float z = 0.0f;
        p.assign(e.size(), 0.0f);
        // v2.1: honor the fitted temperature down to its search floor — the
        // old 0.005 clamp silently discarded a fitted T below it (fitted
        // confidences were computed at 0.005 no matter what calibration.json
        // said). exp((e - maxe)/T) <= 1 for any T > 0: overflow impossible.
        const float Ts = std::max(1e-6f, T);
        for (std::size_t i = 0; i < e.size(); ++i) { p[i] = std::exp((e[i] - maxe) / Ts); z += p[i]; }
        for (auto& v : p) v /= z;
        return maxe;
    }

    static float entropy_confidence(const std::vector<float>& p) {
        if (p.size() < 2) return 0.5f;
        float h = 0.0f;
        for (float v : p) if (v > 1e-9f) h -= v * std::log(v);
        float conf = 1.0f - static_cast<float>(h / std::log(static_cast<double>(p.size())));
        return std::max(0.0f, std::min(1.0f, conf));
    }

    void decide_choice(const sfx::JV& q, const std::vector<std::string>& state_words,
                       Answer& a, Usage* usage = nullptr) {
        const sfx::JV& crit = q.at("criteria");
        std::vector<std::string> labels; std::vector<float> energies;
        if (crit.is_obj())
            for (const auto& kv : crit.obj) {
                labels.push_back(kv.first);
                energies.push_back(probe_energy(kv.first, kv.second.as_str()));
            }
        else if (crit.is_arr())
            for (const auto& v : crit.arr) {
                labels.push_back(v.as_str());
                energies.push_back(probe_energy(v.as_str(), ""));
            }
        if (labels.empty()) { a.deferred = true; a.reason = "no_options"; return; }
        // v3.3 honest silence at the READOUT: if the settled field carries no
        // energy on ANY candidate, the readout has nothing to measure — the
        // old behavior answered with a uniform distribution and picked by
        // criteria order (the reported "purple elephant" OOD mode answered a
        // tie at confidence 0). Deferring here is the same honest-silence
        // contract the dark-field case already follows.
        // v3.3.1 — READOUT SILENCE THRESHOLD, explicit and documented.
        // Shipped v3.3 checked `any candidate energy > 0`, which answers on
        // dust-level diffusion carryover (0 < e < 0.01) picked by noise.
        // The floor is now a named ABSOLUTE constant: defer when the BEST
        // candidate carries less than 0.01 energy. Deliberately absolute
        // (Option B), not relative-to-field: a lit field sitting entirely on
        // non-candidate nodes must defer no matter how bright it is, and a
        // valid candidate on a dim field must answer. Above 0.01 the physics
        // measured real signal (probe-fabric candidates 0.19-0.88); below it
        // the field is carryover. Do NOT raise above 0.10 — that false-defers
        // valid dim candidates (measured 0.19 without energy-norm).
        float max_energy = 0.0f;
        for (float e : energies) max_energy = std::max(max_energy, e);
        if (std::getenv("SYFOX_DEBUG_READOUT") != nullptr) {
            std::fprintf(stderr, "DEBUG: max_candidate=%.3f threshold=%.3f\n",
                         max_energy, kUnknownCandidateFloor);
            for (std::size_t i = 0; i < labels.size(); ++i)
                std::fprintf(stderr, "DEBUG:   candidate[%s]=%.4f\n",
                             labels[i].c_str(), energies[i]);
        }
        if (max_energy < kUnknownCandidateFloor) {
            a.deferred = true;
            a.reason = "unknown_candidates";
            return;
        }
        // v3.9.0 decision-layer paper gates — each measures the settled field
        // READ-ONLY and zeroes/scales energies; none rewrites the field.
        // (a) MOMENTUM (MCPE §1 / CPME §2): a candidate whose normalized
        //     tokens all LOST energy across the final settle pass already
        //     peaked — it is the injected prompt echoing, not an emergent
        //     answer. Zero it. All candidates echo-only => honest defer.
        if (momentum_gate_ && usage) {
            std::size_t eligible = 0;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                bool gain = false;
                for (const auto& t : si::norm::normalize(labels[i]))
                    if (si_.has(t) && si_.node_momentum(si_.find(t)) > 0.0f) {
                        gain = true; break;
                    }
                if (!gain) energies[i] = 0.0f;
                else ++eligible;
            }
            usage->momentum_gate = true;
            usage->momentum_eligible_frac =
                static_cast<float>(eligible) / static_cast<float>(labels.size());
            float new_max = 0.0f;
            for (float e : energies) new_max = std::max(new_max, e);
            if (new_max < kUnknownCandidateFloor) {
                a.deferred = true;
                a.reason = "echo_only";
                return;
            }
        }
        // (b) WAVEFRONT (CMD §4): restrict readout to candidates the query
        //     actually reached — a candidate node resting at zero energy was
        //     never touched by this state's propagation; its score is lane
        //     dust, not measurement. Scale by the reached-energy share.
        if (wavefront_gate_ && usage) {
            std::vector<float> reach(labels.size(), 0.0f);
            float peak = 0.0f;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                float e = 0.0f;
                for (const auto& t : si::norm::normalize(labels[i]))
                    if (si_.has(t)) e = std::max(e, si_.node_energy(si_.find(t)));
                reach[i] = e;
                peak = std::max(peak, e);
            }
            std::size_t reached = 0;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (reach[i] <= 0.0f) energies[i] = 0.0f;
                else { energies[i] *= reach[i] / peak; ++reached; }
            }
            usage->wavefront_gate = true;
            usage->wavefront_frac =
                static_cast<float>(reached) / static_cast<float>(labels.size());
            float new_max = 0.0f;
            for (float e : energies) new_max = std::max(new_max, e);
            if (new_max < kUnknownCandidateFloor) {
                a.deferred = true;
                a.reason = "unreached_candidates";
                return;
            }
        }
        // (c) COHERENCE (HTR §2.2, no-phase surrogate): the candidate set's
        //     participation ratio must RISE above the whole field's — an
        //     answer no more concentrated than baseline drift is silence.
        if (coherence_gate_ && usage) {
            double s = 0.0, s2 = 0.0;
            std::size_t n_lit = 0;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                for (const auto& t : si::norm::normalize(labels[i])) {
                    if (!si_.has(t)) continue;
                    const float e = si_.node_energy(si_.find(t));
                    if (e <= 1e-7f) continue;
                    s += e; s2 += static_cast<double>(e) * e; ++n_lit;
                }
            }
            const float r_cand = (n_lit && s > 0.0)
                ? static_cast<float>((s * s) / (static_cast<double>(n_lit) * s2)) : 0.0f;
            const float r_field = si_.field_participation_ratio();
            usage->coherence_gate = true;
            usage->coherence_r = r_cand;
            usage->coherence_field_r = r_field;
            if (r_cand - r_field < coherence_floor_) {
                a.deferred = true;
                a.reason = "no_structural_rise";
                return;
            }
        }
        // v3.2 Stage 3 — semantic hierarchy gating (readout-side): when the
        // model ships a hierarchy and the candidates carry categories, stage-1
        // category energies scale the stage-2 intent energies. Deterministic,
        // physics-read-only: the settled field is measured, never rewritten.
        if (hier_on_ && !hier_of_.empty()) {
            std::map<std::string, float> cat_e;
            float max_cat = 0.0f;
            bool any = false;
            for (const auto& l : labels) {
                auto it = hier_of_.find(l);
                if (it == hier_of_.end()) continue;
                if (cat_e.count(it->second) == 0) {
                    auto cit = hier_cat_criteria_.find(it->second);
                    const float e = (cit != hier_cat_criteria_.end())
                        ? probe_energy(it->second, cit->second) : 0.0f;
                    cat_e[it->second] = e;
                }
                max_cat = std::max(max_cat, cat_e[it->second]);
                any = true;
            }
            if (any && max_cat > 0.0f) {
                for (std::size_t i = 0; i < labels.size(); ++i) {
                    auto it = hier_of_.find(labels[i]);
                    if (it == hier_of_.end()) continue;
                    const float gate = hier_floor_
                        + (1.0f - hier_floor_) * (cat_e[it->second] / max_cat);
                    energies[i] *= gate;
                }
            }
        }
        // v3.3 — question-conditioned readout gating. The state field stands;
        // the question's NEW tokens (present in the fabric, absent from the
        // state) distribute lane mass over the candidate anchors, and each
        // candidate is scaled by its share of that mass. Deterministic,
        // read-only: the settled field is measured, never rewritten.
        if (question_gate_on_) {
            std::vector<std::string> qtoks;
            if (q.has("instructions")) {
                for (const auto& t : si::norm::normalize(q.at("instructions").as_str())) {
                    if (!si_.has(t)) continue;                       // unknown question words gate nothing
                    if (std::find(state_words.begin(), state_words.end(), t) != state_words.end())
                        continue;                                    // the state already spoke for itself
                    if (std::find(qtoks.begin(), qtoks.end(), t) == qtoks.end())
                        qtoks.push_back(t);
                }
            }
            if (!qtoks.empty()) {
                std::unordered_map<si::NodeId, float> rel;
                std::vector<std::pair<si::NodeId, float>> ln;
                for (const auto& t : qtoks) {
                    ln.clear();
                    si_.lanes_of(si_.find(t), ln);
                    for (const auto& l : ln) rel[l.first] += l.second;
                }
                std::vector<float> r(labels.size(), 0.0f);
                float maxr = 0.0f;
                for (std::size_t i = 0; i < labels.size(); ++i) {
                    // labels are user-facing keys ("Tariq"); the fabric stores
                    // normalized concepts ("tariq") — normalize before lookup,
                    // same convention probe_energy uses for the readout.
                    const std::vector<std::string> lt = si::norm::normalize(labels[i]);
                    float ri = 0.0f;
                    bool any_tok = false;
                    for (const auto& t : lt) {
                        if (!si_.has(t)) continue;
                        any_tok = true;
                        auto it = rel.find(si_.find(t));
                        if (it != rel.end()) ri += it->second;
                    }
                    if (!any_tok) continue;                          // candidate not in the fabric: untouched
                    r[i] = ri;
                    maxr = std::max(maxr, ri);
                }
                if (maxr > 0.0f) {
                    for (std::size_t i = 0; i < labels.size(); ++i) {
                        const std::vector<std::string> lt = si::norm::normalize(labels[i]);
                        bool any_tok = false;
                        for (const auto& t : lt) if (si_.has(t)) { any_tok = true; break; }
                        if (!any_tok) continue;
                        energies[i] *= question_gate_floor_
                                     + (1.0f - question_gate_floor_) * (r[i] / maxr);
                    }
                    a.gate_tokens = qtoks;
                }
            }
        }
        std::vector<float> p;
        softmax_ps(energies, p, calib_.choice_temperature);
        std::size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        a.choice = labels[best];
        for (std::size_t i = 0; i < labels.size(); ++i)
            a.probabilities.emplace_back(labels[i], p[i]);
        a.confidence = entropy_confidence(p);
        // v3.9.0 (RADE) PROOF PASS: typed path between the state's strongest
        // axiom-graph concept and the winning anchor, premise-cited. Opt-in
        // disclosure; --require-proof defers a choice with no typed path
        // (coverage-cliff honesty: the field answered, the knowledge layer
        // cannot justify it).
        if (usage && (proof_on_ || require_proof_)) {
            usage->proof_checked = true;
            usage->proof_path = proof_path_between(state_words, labels[best]);
            if (require_proof_ && usage->proof_path.empty()) {
                a.deferred = true;
                a.reason = "no_proof_path";
            }
        }
        // v3.9.0 (fixes.md): remember the signature pieces for the OOD gate.
        if (usage) {
            float p1o = 0.0f, p2o = 0.0f;
            for (float v : p) {
                if (v > p1o) { p2o = p1o; p1o = v; }
                else if (v > p2o) p2o = v;
            }
            usage->ood_knn_dist = -1.0f;   // marker: decide() fills the distance
            ood_last_top_ = p1o;
            ood_last_margin_ = p1o - p2o;
            std::size_t unk = 0;
            for (const auto& w : state_words) if (!si_.has(w)) ++unk;
            ood_last_unk_ = state_words.empty() ? 0.0f
                : static_cast<float>(unk) / static_cast<float>(state_words.size());
        }
        // v3.3 exact-tie disclosure: when the top two probabilities are
        // EQUAL the readout carried no signal between them — the pick is
        // deterministic criteria order (std::map key order), and every reply
        // says so. NEAR-ties stay the --defer-margin knob's business.
        if (p.size() >= 2) {
            float p1 = 0.0f, p2 = 0.0f;
            for (float v : p) {
                if (v > p1) { p2 = p1; p1 = v; }
                else if (v > p2) p2 = v;
            }
            if (p1 - p2 <= 1e-9f) a.tied = true;
            // v3.4 honest defer for near-ties: a margin this thin is not a
            // decision, it is a coin-flip the readout cannot settle. Defer
            // with reason ambiguous_tie (engine-level; 0 disables).
            if (defer_margin_ > 0.0f && p1 - p2 < defer_margin_) {
                a.deferred = true;
                a.reason = "ambiguous_tie";
            }
        }
    }

    void decide_score(const sfx::JV& q, Answer& a) {
        const sfx::JV& crit = q.at("criteria");
        std::vector<std::string> labels; std::vector<float> energies;
        if (crit.is_arr())
            for (const auto& v : crit.arr) { labels.push_back(v.as_str()); energies.push_back(probe_energy(v.as_str(), "")); }
        else if (crit.is_obj())
            for (const auto& kv : crit.obj) { labels.push_back(kv.first); energies.push_back(probe_energy(kv.first, kv.second.as_str())); }
        if (labels.empty()) { a.deferred = true; a.reason = "no_levels"; return; }
        std::vector<float> p;
        softmax_ps(energies, p, calib_.score_temperature);
        float val = 0.0f;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            a.probabilities.emplace_back(labels[i], p[i]);
            val += p[i] * static_cast<float>(i);
        }
        a.value = val;
        a.confidence = entropy_confidence(p);
        // v3.4 honest defer for near-ties, score flavor: same margin over the
        // level probabilities — a near-tied scale is an ambiguous_tie too.
        if (p.size() >= 2) {
            float p1 = 0.0f, p2 = 0.0f;
            for (float v : p) {
                if (v > p1) { p2 = p1; p1 = v; }
                else if (v > p2) p2 = v;
            }
            if (defer_margin_ > 0.0f && p1 - p2 < defer_margin_) {
                a.deferred = true;
                a.reason = "ambiguous_tie";
            }
        }
    }

    void decide_noul(const sfx::JV& q, Answer& a) {
        float s = noul_support(q.at("instructions").as_str());
        a.probability = 1.0f / (1.0f + std::exp(-(calib_.noul_a * s + calib_.noul_b)));
        a.confidence = std::fabs(a.probability - 0.5f) * 2.0f;
    }

    // -- calibration fitting (tool-side numerical search, no core changes) ------
    static float nll_temperature(const std::vector<CalibRow>& rows, const std::string& type, float T) {
        // v2.1: multi-class NLL over the FULL candidate vector — the textbook
        // temperature-scaling objective (Guo et al. 2017). Every candidate
        // participates, so a wrong-but-tied gold label keeps p_gold ~ 1/nk
        // under ANY temperature and properly punishes over-sharpening.
        float nll = 0.0f; std::size_t n = 0;
        // NO floor clamp here beyond positivity: the max-subtracted softmax
        // is overflow-safe for any T > 0 (exponents are <= 0; deep negatives
        // just underflow to 0). A 0.005 floor made NLL flat below 0.005 and
        // hid the true optimum from the search.
        const float Ts = std::max(1e-6f, T);
        for (const auto& r : rows) {
            if (r.type != type || r.energies.empty()) continue;
            ++n;
            float maxe = -1e30f;
            for (const auto& kv : r.energies) maxe = std::max(maxe, kv.second);
            float z = 0.0f, pl = 0.0f;
            for (const auto& kv : r.energies) {
                const float v = std::exp((kv.second - maxe) / Ts);
                z += v;
                if (kv.first == r.label) pl = v;
            }
            nll -= std::log(std::max(pl, 1e-9f) / std::max(z, 1e-9f));
        }
        return n > 0 ? nll / static_cast<float>(n) : 0.0f;
    }

    static void fit_temperature(const std::vector<CalibRow>& rows, const std::string& type, float& T) {
        // Bounds span the substrate's OWN energy-gap scale: readout gaps on
        // unfamiliar states are O(0.001..0.05), so a temperature floor of
        // 0.005 capped achievable sharpness at exp(gap/0.005) and left
        // correct answers under-confident. 0.0005 covers gaps down to
        // ~0.003; the max-subtracted softmax is numerically stable all the
        // way down (exp(-large) simply underflows to 0). The multi-class NLL
        // guards the fit itself: with any argmax-wrong row in the fit set
        // the optimum stays off the floor (sharpening a wrong answer costs
        // NLL), which is exactly the safeguard the pairwise view lacked.
        const float lo = 0.0005f, hi = 20.0f;
        float a = lo, b = hi;
        for (int it = 0; it < 60; ++it) {                    // golden-section on NLL
            float m1 = a + (b - a) / 3.0f, m2 = b - (b - a) / 3.0f;
            if (nll_temperature(rows, type, m1) < nll_temperature(rows, type, m2)) b = m2; else a = m1;
        }
        const float cand = (a + b) / 2.0f;
        if (nll_temperature(rows, type, cand) < nll_temperature(rows, type, 1.0f))
            T = cand;
    }

    void fit_platt(const std::vector<CalibRow>& rows) {
        std::vector<std::pair<float, int>> pts;
        for (const auto& r : rows) if (r.type == "noul") pts.emplace_back(r.support, r.y);
        if (pts.size() < 4) return;
        float a = std::max(10.0f, calib_.noul_a), b = calib_.noul_b;
        float lr = 0.5f;                                     // decaying lr, long run:
        for (int it = 0; it < 4000; ++it) {                  // separable data needs a
            float ga = 0.0f, gb = 0.0f;                      // steep slope to converge
            for (const auto& pt : pts) {
                float p = 1.0f / (1.0f + std::exp(-(a * pt.first + b)));
                float err = static_cast<float>(pt.second) - p;
                ga += err * pt.first; gb += err;
            }
            ga /= static_cast<float>(pts.size()); gb /= static_cast<float>(pts.size());
            a += lr * ga; b += lr * gb;
            lr *= 0.999f;
        }
        a = std::max(0.5f, std::min(40.0f, a));
        b = std::max(-15.0f, std::min(2.0f, b));
        calib_.noul_a = a; calib_.noul_b = b;
    }

    si::Substrate si_;
    Calibration calib_;
    bool energy_norm_ = false;                          // Milestone-1 gain knob (default off)
    std::unordered_set<std::uint64_t> lesson_hashes_;   // Milestone-2 dedup memory
    // v3.2 retrieval-by-default state
    std::vector<recall::Memory> memories_;
    std::vector<std::vector<float>> memory_fps_;        // settled-field fingerprints, fixed at load
    bool retrieval_on_ = true;                          // default ON (inert until memories exist)
    int retrieval_topk_ = 5;
    float retrieval_dose_ = 0.30f;                      // prime dose as a fraction of inject_energy
    // v3.2 Stage 3 hierarchy state
    std::map<std::string, std::string> hier_of_;        // intent anchor -> category anchor
    std::map<std::string, std::string> hier_cat_criteria_; // category anchor -> criteria text
    float hier_floor_ = 0.35f;
    bool hier_on_ = false;
    // v3.3 question-conditioned readout state. Default OFF: measured on the
    // trained domains the gate is a trade-off (tickets-cal acc 0.9533 ->
    // 0.9000 with the gate on — "which team" lanes mislead), while on
    // question-driven reasoning probes it is the fix (who/latest gate rows).
    // Ships as an opt-in knob exactly like the M1 energy-norm gain: enabled
    // with --question-gate / opts.question_gate, floor tuned by measurement.
    bool question_gate_on_ = false;
    float question_gate_floor_ = 0.25f;
    float defer_margin_ = 0.05f;      // v3.4 near-tie defer margin (0 = off)
    bool ctx_gate_on_ = false;        // v3.4 question-context two-stage settle (default OFF)
    bool perturb_on_ = false;         // v3.6.0 BED §8 perturbation-contrast check (default OFF)
    float ctx_alpha_ = 0.5f;          // v3.4 context-field weight in the composed field
    bool distvec_on_ = false;         // v3.7.0 Move 1: PPMI+SVD resonance edges at save (default OFF)
    bool distvec_dirty_ = true;       // fabric moved since the last distvec build
    long distvec_dims_ = 0;           // v3.7.0: embedding width override (0 = default 300)
    // v3.8.0 TASK 1 + TASK 4: adaptive settle depth and self-verification
    bool  adaptive_depth_on_ = false;
    int   adaptive_max_k_ = 8;        // extra passes at difficulty 1.0
    int   adaptive_base_k_;           // fixed at construction (ratchet guard)
    bool  self_verify_on_ = false;
    // Default floor measured on the v3.8 bench fabrics: a SETTLED field's
    // final-pass motion_rate reads 0.189-0.212 across tickets/game/guard
    // (the ~0.18 decay baseline + residual structure). 0.25 sits ~0.04 above
    // the worst settled baseline; fields still routing energy read higher.
    float self_verify_floor_ = 0.25f;
    bool  multi_vector_on_ = false;   // v3.8.0 TASK 2 (default OFF)
    int   multi_vector_k_ = 8;        // channel count
    bool tools_check_on_ = false;     // v3.7.0 Move 4: decide-time tool disclosure (default OFF)
    // v3.9.0 decision-layer paper gates (all default OFF; replay-safe)
    bool proof_on_ = false;
    bool require_proof_ = false;
    bool momentum_gate_ = false;
    bool wavefront_gate_ = false;
    bool coherence_gate_ = false;
    float coherence_floor_ = 0.05f;
    bool vght_gate_ = false;
    bool cem_on_ = false;
    int   cem_rounds_ = 2;
    int   cem_variants_ = 8;
    bool fragments_on_ = false;
    bool ood_knn_on_ = false;
    float ood_knn_tau_ = 1.0f;
    bool tension_on_ = false;
    float tension_acc_ = 0.0f;        // RIFA SIL: grows on contradiction, never decays
    bool pro_con_ = false;
    bool sentinels_ = false;
    // fixes.md OOD signatures: [settled_energy, top_energy, margin, unk_frac]
    // collected from answered decisions at calibrate time; kNN gate at decide.
    std::vector<std::array<float, 4>> ood_sigs_;
    // CME fingerprint table: fp -> {first_seq, last_seq, hit_count}, capped.
    std::unordered_map<std::uint64_t, std::array<std::uint64_t, 3>> fp_table_;
    // v3.9.0 decide-path scratch: the pre-settle field for CEM planning and
    // the last choice question's readout signature pieces (OOD kNN gate).
    std::vector<float> cem_presettle_;
    float ood_last_top_ = 0.0f;
    float ood_last_margin_ = 0.0f;
    float ood_last_unk_ = 0.0f;
    // Milestone-3 audit state
    std::string context_ = "default";
    std::uint64_t teach_seq_ = 0;
    std::map<std::uint64_t, Taught> taught_outcomes_;               // state-stream hash -> outcome
    std::vector<sfx::JV> conflicts_;
    std::map<std::uint64_t, std::vector<std::size_t>> conflicts_by_state_;
};

} // namespace syfox
