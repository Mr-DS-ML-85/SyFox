// ============================================================================
//  SyFox core C API — thin shared-library wrapper (server bridge).
//  All decisions come from the SI substrate; this file only marshals JSON
//  and exposes the SAME boundary knobs the CLI already has (v1.0 -> v3.0):
//
//    M1  energy-norm measurement gain      -> syfox_engine_set_energy_norm
//    v1  SI salience gating + Miller window -> syfox_engine_set_source_modes
//    M5  deterministic OMP parallel settle -> syfox_engine_set_parallel_settle
//    v2.2 trigram lane policy              -> syfox_engine_set_ngrams
//    v2.2 script routing                   -> syfox_detect_script
//    M3  machine-auditable evidence        -> syfox_decide_ex(opts.evidence)
//    v3.2 semantic layer + retrieval + hierarchy -> set_semantics / set_retrieval /
//            set_retrieval_topk / set_retrieval_dose / set_hierarchy + opts keys
//
//  No transformer, no classifier, no physics changes: si_substrate.hpp is
//  untouched; every knob here exists because the CLI exposes it too.
// ============================================================================
#include "core/syfox.hpp"
#include "core/script.hpp"
#include "core/ngram.hpp"

#include <cstring>
#include <string>

using syfox::Engine;

namespace { bool truthy(const sfx::JV& v) {
    return (v.is_bool() && v.b) || (v.is_num() && v.num != 0.0);
} }

extern "C" {

struct SyFoxHandle { Engine* eng; };

SyFoxHandle* syfox_engine_create(const char* model_dir) {
    if (!model_dir) return nullptr;
    SyFoxHandle* h = new SyFoxHandle();
    h->eng = new Engine();
    // v3.2.1: a missing substrate is a load FAILURE (null handle => the
    // bridge raises "cannot load model"), not a silent empty fabric.
    if (!h->eng->load_model(model_dir)) {
        delete h->eng;
        delete h;
        return nullptr;
    }
    return h;
}

void syfox_engine_free(SyFoxHandle* h) {
    if (!h) return;
    delete h->eng;
    delete h;
}

const char* syfox_core_version(void) { return syfox::VERSION; }

// -- engine-scoped knobs (identical semantics to the CLI flags) -------------

void syfox_engine_set_energy_norm(SyFoxHandle* h, int on) {
    if (h) h->eng->set_energy_norm(on != 0);          // M1: measurement gain
}

void syfox_engine_set_source_modes(SyFoxHandle* h, int salience_gating, int miller_window) {
    if (h) h->eng->substrate().set_source_modes(salience_gating != 0, miller_window != 0);
}

void syfox_engine_set_parallel_settle(SyFoxHandle* h, int on, int threads) {
    if (!h) return;
#if defined(_OPENMP)
    if (threads > 0) omp_set_num_threads(threads);
#endif
    h->eng->substrate().set_parallel_settle(on != 0); // M5: OMP builds only
}

void syfox_engine_set_ngrams(SyFoxHandle* h, int on) {
    (void)h;                                          // process-global policy
    si::norm::grams_enabled() = (on != 0);            // v2.2 trigram lanes
}

// -- v3.2 semantic layer / retrieval-by-default / hierarchy knobs -------------
void syfox_engine_set_semantics(SyFoxHandle* h, int on) {
    if (h) h->eng->substrate().set_semantics(on != 0);
}

void syfox_engine_set_retrieval(SyFoxHandle* h, int on) {
    if (h) h->eng->set_retrieval(on != 0);
}

void syfox_engine_set_retrieval_topk(SyFoxHandle* h, int topk) {
    if (h) h->eng->set_retrieval_topk(topk);
}

void syfox_engine_set_retrieval_dose(SyFoxHandle* h, double dose) {
    if (h) h->eng->set_retrieval_dose(static_cast<float>(dose));
}

void syfox_engine_set_hierarchy(SyFoxHandle* h, int on) {
    if (h) h->eng->set_hierarchy(on != 0);
}

// -- script detection (v2.2 --lang auto) ------------------------------------
// Returns a malloc'd slug ("latin", "bengali", ...) or "unknown".
char* syfox_detect_script(const char* text) {
    if (!text) return nullptr;
    const si::script::Script sc = si::script::detect_script(text);
    const std::string slug = (sc == si::script::Script::Unknown)
        ? "unknown" : si::script::slug(sc);
    char* buf = static_cast<char*>(std::malloc(slug.size() + 1));
    std::memcpy(buf, slug.c_str(), slug.size() + 1);
    return buf;
}

// -- decide ------------------------------------------------------------------
// Shared marshaling for syfox_decide (no evidence) and syfox_decide_ex.
static std::string decide_json(SyFoxHandle* h, const char* state,
                               const char* questions_json, bool want_evidence) {
    std::string out;
    try {
        sfx::JV q = sfx::JV::parse(questions_json);
        if (!q.is_obj()) { return "{\"error\":\"questions must be an object\"}"; }
        syfox::Usage u;
        auto answers = h->eng->decide(state, q, u);
        u.calibrated = h->eng->calibration().fitted;   // decide() resets Usage
        sfx::JVObj ans;
        for (const auto& a : answers) {
            sfx::JVObj o;
            o["type"] = sfx::JV(a.type);
            if (a.type == "choice") {
                o["choice"] = sfx::JV(a.choice);
                sfx::JVObj pr;
                for (const auto& p : a.probabilities)
                    pr[p.first] = sfx::JV(std::round(p.second * 1000.0f) / 1000.0f);
                o["probabilities"] = sfx::JV(pr);
            } else if (a.type == "score") {
                o["score"] = sfx::JV(std::round(a.value * 1000.0f) / 1000.0f);
                sfx::JVObj pr;
                for (const auto& p : a.probabilities)
                    pr[p.first] = sfx::JV(std::round(p.second * 1000.0f) / 1000.0f);
                o["probabilities"] = sfx::JV(pr);
            } else if (a.type == "noul") {
                o["noul"] = sfx::JV(std::round(a.probability * 1000.0f) / 1000.0f);
            }
            o["confidence"] = sfx::JV(std::round(a.confidence * 1000.0f) / 1000.0f);
            o["deferred"] = sfx::JV(a.deferred);
            if (!a.reason.empty()) o["reason"] = sfx::JV(a.reason);
            // v3.3 readout disclosures
            if (a.tied) o["tied"] = sfx::JV(true);
            if (!a.gate_tokens.empty()) {
                sfx::JVArr gt;
                for (const auto& t : a.gate_tokens) gt.push_back(sfx::JV(t));
                o["question_gate"] = sfx::JV(gt);
            }
            ans[a.qid] = sfx::JV(o);
        }
        sfx::JVObj usage{
            {"state_tokens", static_cast<double>(u.state_tokens)},
            {"vocabulary", static_cast<double>(u.vocabulary)},
            {"lanes", static_cast<double>(u.lanes)},
            {"settled_energy", std::round(u.settled_energy * 1000.0f) / 1000.0f},
            {"calibrated", sfx::JV(u.calibrated)},
            {"engine", std::string("syfox-") + syfox::VERSION},
            {"core", "si-substrate"}};
        if (!u.retrieved.empty()) {
            sfx::JVArr ret;
            for (const auto& r : u.retrieved)
                ret.push_back(sfx::JV(sfx::JVObj{
                    {"label", sfx::JV(r.first)},
                    {"resonance", std::round(r.second * 1000.0f) / 1000.0f}}));
            usage["retrieval"] = sfx::JV(ret);        // v3.2: priming memories
        }
        sfx::JVObj root{{"answers", sfx::JV(ans)}, {"usage", sfx::JV(usage)}};
        if (want_evidence)   // M3: supporting lanes + provenance + contradictions
            root["evidence"] = h->eng->evidence_json(state, q, answers);
        out = sfx::JV(root).dump();
    } catch (const std::exception& e) {
        out = std::string("{\"error\":\"") + e.what() + "\"}";
    }
    return out;
}

// Decides and returns a malloc'd JSON string: {"answers":..., "usage":...}.
// Caller frees with syfox_string_free. On malformed questions returns
// {"error": "..."}.
char* syfox_decide(SyFoxHandle* h, const char* state, const char* questions_json) {
    if (!h || !state || !questions_json) return nullptr;
    const std::string s = decide_json(h, state, questions_json, false);
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return buf;
}

// opts_json: {"evidence":bool, "energy_norm":bool, "salience_gating":bool,
//             "miller_window":bool, "ngrams":"on"|"off",
//             "semantics":bool, "retrieval":bool, "retrieval_topk":int,
//             "retrieval_dose":num, "hierarchy":bool}
// Absent keys leave the current engine state untouched (sticky, CLI-equal).
char* syfox_decide_ex(SyFoxHandle* h, const char* state, const char* questions_json,
                      const char* opts_json) {
    if (!h || !state || !questions_json) return nullptr;
    bool evidence = false;
    if (opts_json && *opts_json) {
        try {
            sfx::JV o = sfx::JV::parse(opts_json);
            if (o.is_obj()) {
                if (o.has("evidence"))        evidence = truthy(o.at("evidence"));
                if (o.has("energy_norm"))     h->eng->set_energy_norm(truthy(o.at("energy_norm")));
                if (o.has("salience_gating") || o.has("miller_window"))
                    h->eng->substrate().set_source_modes(
                        o.has("salience_gating") && truthy(o.at("salience_gating")),
                        o.has("miller_window") && truthy(o.at("miller_window")));
                if (o.has("ngrams")) {
                    const std::string g = o.at("ngrams").as_str();
                    if (g == "on")  si::norm::grams_enabled() = true;
                    if (g == "off") si::norm::grams_enabled() = false;
                }
                if (o.has("semantics"))       h->eng->substrate().set_semantics(truthy(o.at("semantics")));
                if (o.has("retrieval"))       h->eng->set_retrieval(truthy(o.at("retrieval")));
                if (o.has("retrieval_topk"))  h->eng->set_retrieval_topk(static_cast<int>(o.at("retrieval_topk").as_num(5)));
                if (o.has("retrieval_dose"))  h->eng->set_retrieval_dose(static_cast<float>(o.at("retrieval_dose").as_num(0.30)));
                if (o.has("hierarchy"))       h->eng->set_hierarchy(truthy(o.at("hierarchy")));
                // v3.3 question-conditioned readout
                if (o.has("question_gate"))   h->eng->set_question_gate(truthy(o.at("question_gate")));
                if (o.has("question_gate_floor"))
                    h->eng->set_question_gate_floor(static_cast<float>(o.at("question_gate_floor").as_num(0.25)));
                // v3.4 honest defer + multi-hop walk + question-context gate
                if (o.has("defer_margin"))
                    h->eng->set_defer_margin(static_cast<float>(o.at("defer_margin").as_num(0.05)));
                if (o.has("no_defer") && truthy(o.at("no_defer")))
                    h->eng->set_defer_margin(0);
                if (o.has("hops"))
                    h->eng->substrate().set_hop_depth(static_cast<int>(o.at("hops").as_num(1)));
                if (o.has("ctx_gate"))        h->eng->set_ctx_gate(truthy(o.at("ctx_gate")));
                if (o.has("ctx_alpha"))
                    h->eng->set_ctx_alpha(static_cast<float>(o.at("ctx_alpha").as_num(0.5)));
            }
        } catch (...) { /* opts are optional; a bad opts object is ignored */ }
    }
    const std::string s = decide_json(h, state, questions_json, evidence);
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return buf;
}

// Model fabric summary: {"nodes":...,"lanes":...,"evidence_records":...,
// "calibrated":...,"version":...,"core":...}
char* syfox_engine_info(SyFoxHandle* h) {
    if (!h) return nullptr;
    const sfx::JVObj o{
        {"nodes", static_cast<double>(h->eng->substrate().node_count())},
        {"lanes", static_cast<double>(h->eng->substrate().lane_count())},
        {"evidence_records", static_cast<double>(h->eng->substrate().evidence_count())},
        {"calibrated", sfx::JV(h->eng->calibration().fitted)},
        {"semantics", sfx::JV(h->eng->substrate().has_semantics())},
        {"sem_edges", static_cast<double>(h->eng->substrate().resonance_edge_count())},
        {"lane_contexts", static_cast<double>(h->eng->substrate().lane_context_count())},
        {"retrieval_memories", static_cast<double>(h->eng->memories().size())},
        {"hierarchy", sfx::JV(h->eng->hierarchy_on())},
        {"version", std::string(syfox::VERSION)},
        {"core", "si-substrate"}};
    const std::string s = sfx::JV(o).dump();
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return buf;
}

void syfox_string_free(char* s) { std::free(s); }

} // extern "C"
