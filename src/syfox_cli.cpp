// ============================================================================
//  SyFox CLI — learn / calibrate / decide / demo
//  (thin tooling around the SI substrate core; no logic lives here)
// ============================================================================
#include "core/syfox.hpp"
#include "core/derive.hpp"
#include "core/bench.hpp"
#include "core/gate.hpp"
#include "core/recall.hpp"
#include "core/firewall.hpp"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

void usage_exit();

struct Args {
    std::string model = "model";
    std::string examples;
    std::string state;
    std::string questions;
    std::string domain;
    std::string concept;
    long steps = 64;                 // dream steps
    unsigned long long seed = 0x5EED5EEDull;  // dream seed (deterministic by default)
    std::string eval;                // bench eval rows (defaults to --examples)
    std::string gate;                // derivation gate rows (no-regression replay)
    std::string memories;            // recall memory store (jsonl)
    long topk = 5;                   // recall top-k
    std::string split;               // bench --split train|heldout (P1 eval split)
    bool coverage_curve = false;     // bench --coverage-curve (P2 headline metric)
    std::string synonyms;            // synonym table path (default data/synonyms.txt)
    // v2.2 multilingual + active-learning surface
    std::string lang;                // --lang auto|<slug>: route to <model>-<script>; empty = off
    int ngrams_mode = 0;             // --ngrams on|off: 1/-1 explicit; 0 = policy default
    bool augment = false;            // learn --augment: mass-guarded variant lessons
    float typos = 0;                 // bench --typos P: deterministic corruption sweep (0..100)
    std::string deferrals;           // decide --log-deferrals FILE.jsonl (active-learning loop)
    std::string out;                 // active --out FILE.jsonl (labeling worksheet)
    long min_count = 1;              // active --min-count N
    bool dedup = false;              // learn --dedup: skip exact duplicate lessons (M2)
    bool novelty = false;            // learn --novelty: per-lesson dose by novelty (M2)
    float novelty_floor = 0.25f;     // learn --novelty-floor F (A/B knob)
    bool energy_norm = false;        // decide-side energy gain for big-corpus fabrics (M1)
    float defer_margin = -1;         // v3.4: engine default ON at 0.05 (reason ambiguous_tie);
                                     // --defer-margin P overrides, --no-defer disables.
                                     // (v3.1-3.3 this was a CLI post-pass with reason low_margin,
                                     // default OFF — now the engine owns it for CLI+API+bench parity)
    bool evidence = false;           // decide --evidence: machine-auditable evidence JSON (M3)
    bool adversarial = false;        // bench --adversarial: M4 stress suite (read-only)
    std::string mix;                 // bench --mix FILE: cross-domain vocabulary source (M4)
    long threads = 0;                // --threads N: OMP settle threads (1 = sequential; 0 = default)
    long throughput = 0;             // bench --throughput N: batched multicore decisions/sec (M5)
    long epochs = 1;                 // learn --epochs N: consolidation passes (see lane_decay)
    long latency_reps = 20;          // bench --latency-reps N (timing repeats per probe)
    long replays = 2;                // bench --replays N (determinism double-run count)
    bool state_file = false, questions_file = false;
    // SI-faithful selection modes (off by default; never persisted into the model)
    bool salience_gating = false, miller_window = false;
    // v3.2 semantic layer + retrieval-by-default + two-stage router
    bool no_semantics = false;       // --no-semantics: runtime kill switch for the semantic field
    bool no_retrieval = false;       // --no-retrieval: skip associative priming
    bool no_hierarchy = false;       // --no-hierarchy: skip stage-1 category gating
    bool no_defer = false;           // --no-defer: disable v3.4 near-tie defer (engine default ON)
    long hops = -1;                  // --hops N: v3.4 multi-hop readout walk depth (engine default 1 = legacy)
    bool ctx_gate = false;           // --ctx-gate: v3.4 two-stage question-context settle
    float ctx_alpha = 0;             // --ctx-alpha F (0 = engine default 0.5)
    long retrieval_topk = -1;        // --retrieval-topk N (-1 = engine default 5)
    float retrieval_dose = 0;        // --retrieval-dose F (0 = engine default 0.30)
    std::string router;              // --router DIR: stage-1 domain fabric (router.json maps domains)
    // v3.3 question-conditioned readout (opt-in; see core/syfox.hpp note)
    bool question_gate = false;      // --question-gate: enable readout gating
    float question_gate_floor = 0;   // --question-gate-floor F (0 = engine default 0.25)
};

// v3.2.1 — BUG #1 disclosure: a kill switch on a model that lacks the
// corresponding feature is a silent no-op (pre-v3.2 fabric => no v4 semantic
// tail => --no-semantics changes nothing; no memories.jsonl => --no-retrieval
// changes nothing; no hierarchy.json => --no-hierarchy changes nothing).
// Five ablation configs then produce byte-identical output and an ablation
// study silently measures nothing. The flags still mean what they say — the
// honest fix is to SAY the switch is inert, and to name the rebuild path.
void warn_inert_switches(const syfox::Engine& eng, const Args& a,
                         const std::string& model_dir) {
    if (a.no_semantics && !eng.substrate().has_semantics())
        std::cerr << "syfox: note: --no-semantics is inert on " << model_dir
                  << " (no v3.2 semantic tail in substrate.bin; rebuild with "
                  << "./build/rebuild_sem SRC DST or retrain under v3.2 to make "
                  << "the switch measurable)\n";
    if (a.no_retrieval && eng.memories().empty())
        std::cerr << "syfox: note: --no-retrieval is inert on " << model_dir
                  << " (no memories.jsonl in the model dir; retrieval has "
                  << "nothing to prime)\n";
    if (a.no_hierarchy && !eng.hierarchy_on())
        std::cerr << "syfox: note: --no-hierarchy is inert on " << model_dir
                  << " (no hierarchy.json in the model dir)\n";
}

// v3.2.1 — BUG #3 fix, part 1: resolve a model dir across repo layouts. The
// syfox repo keeps fabrics at the root (model-<name>), the HuggingFace
// package nests them (model/<name>). `decide --router model-router16` was
// documented from the repo root and failed inside the HF package with the
// misleading "lacks router.json anchors" (the dir itself wasn't found). One
// resolution order serves both layouts; first candidate with substrate.bin
// wins, and the as-given path still wins when it exists at all (the loud
// load failure then names the real problem instead of a fake one).
std::string resolve_model_dir(const std::string& p) {
    if (p.empty()) return p;
    auto has_bin = [](const std::string& d) {
        std::ifstream f(d + "/substrate.bin");
        return static_cast<bool>(f);
    };
    if (has_bin(p)) return p;
    std::string stripped = p;
    const std::string pfx = "model-";
    if (p.rfind(pfx, 0) == 0) stripped = p.substr(pfx.size());
    const std::vector<std::string> cands = {
        "model/" + p, "model-" + stripped, "model/model-" + stripped,
        "model/" + stripped};
    for (const auto& c : cands)
        if (has_bin(c)) return c;
    return p;
}

// v3.2.1 — BUG #3 fix, part 2: honest load failure. A missing substrate used
// to load as a silent empty fabric (all honest_silence, or the fake router
// error). Now the CLI names the dir that failed.
syfox::Engine& load_or_die(syfox::Engine& eng, const std::string& dir,
                           const std::string& role) {
    if (!eng.load_model(dir)) {
        std::cerr << "syfox: " << role << " model \"" << dir
                  << "\": cannot read " << dir
                  << "/substrate.bin (no such file or directory)\n";
        // v3.3 hint: "model" is the DEFAULT --model value. Seeing it here
        // almost always means the caller's harness swallowed --model after
        // a flag it treats as value-taking (measured: a python argparse
        // add_argument('--evidence') without action='store_true' eats the
        // following --model token, and a v3.2.0 binary then decided on a
        // SILENT empty fabric: lanes 0, vocabulary 0, all unknown_vocabulary).
        if (dir == "model")
            std::cerr << "syfox: note: \"model\" is the default --model value; "
                      << "check that the preceding CLI flag did not consume "
                      << "the --model argument\n";
        std::exit(2);
    }
    return eng;
}

// Apply the CLI mode overrides after load_model(). Mode-neutral by design:
// substrate.bin stays untouched, flags live only for this process.
void apply_modes(syfox::Engine& eng, const Args& a) {
    eng.substrate().set_source_modes(a.salience_gating, a.miller_window);
    // v3.2 semantic layer knobs: the field is ON whenever the model ships it
    // (substrate v4 tail); the kill switches restore pre-3.2 behavior exactly.
    eng.substrate().set_semantics(!a.no_semantics);
    eng.set_retrieval(!a.no_retrieval);
    if (a.retrieval_topk >= 0) eng.set_retrieval_topk(static_cast<int>(a.retrieval_topk));
    if (a.retrieval_dose > 0) eng.set_retrieval_dose(a.retrieval_dose);
    if (a.no_hierarchy) eng.set_hierarchy(false);
    // v3.3 question-conditioned readout knobs (opt-in)
    if (a.question_gate) eng.set_question_gate(true);
    if (a.question_gate_floor > 0) eng.set_question_gate_floor(a.question_gate_floor);
    // v3.4 honest defer for near-ties: engine default ON at margin 0.05
    // (reason ambiguous_tie). --defer-margin P overrides; --no-defer
    // restores the v3.3 answer-always behavior.
    if (a.no_defer) eng.set_defer_margin(0);
    else if (a.defer_margin >= 0) eng.set_defer_margin(a.defer_margin);
    // v3.4 multi-hop readout walk (engine default 1 = legacy single-hop,
    // bit-identical; depths 2-8 walk lanes with 1/sqrt(hop+1) damping)
    if (a.hops >= 0) eng.substrate().set_hop_depth(static_cast<int>(a.hops));
    // v3.4 question-context two-stage settle (opt-in; alpha default 0.5)
    if (a.ctx_gate) eng.set_ctx_gate(true);
    if (a.ctx_alpha > 0) eng.set_ctx_alpha(a.ctx_alpha);
    // v3 Milestone 5: --threads N controls deterministic parallel settle on
    // OMP builds (bit-identical to sequential; test-verified). N=1 forces the
    // sequential path; N=0 leaves the default. Non-OMP builds ignore it.
    if (a.threads > 0) {
#if defined(_OPENMP)
        omp_set_num_threads(static_cast<int>(a.threads));
#endif
        eng.substrate().set_parallel_settle(a.threads != 1);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "syfox: cannot open " << path << "\n"; std::exit(2); }
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return buf;
}

std::vector<sfx::JV> load_jsonl(const std::string& path) {
    std::vector<sfx::JV> rows;
    std::ifstream f(path);
    if (!f) { std::cerr << "syfox: cannot open " << path << "\n"; std::exit(2); }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '/') continue;
        try { rows.push_back(sfx::JV::parse(line)); }
        catch (const std::exception& e) { std::cerr << "syfox: " << path << ": " << e.what() << "\n"; std::exit(2); }
    }
    return rows;
}

// The label's own description text inside criteria (for choice/score outcomes).
// For array criteria the label is the level INDEX ("0","1","2") or the level text.
std::string outcome_text(const sfx::JV& q, const std::string& label) {
    const sfx::JV& crit = q.at("criteria");
    if (crit.is_obj() && crit.has(label)) return label + " " + crit.at(label).as_str();
    if (crit.is_arr()) {
        for (const auto& v : crit.arr)
            if (v.as_str() == label) return label;                 // label is the level text
        long idx = std::strtol(label.c_str(), nullptr, 10);
        if (idx >= 0 && idx < static_cast<long>(crit.arr.size()))
            return crit.arr[static_cast<std::size_t>(idx)].as_str(); // label is the index
    }
    return label;
}

// -- multilingual routing (v2.2) -------------------------------------------
// --lang empty : v2.1 behavior — whatever --model says, no detection (off)
// --lang auto  : detect the script of the text, use <model>-<script-slug>
// --lang slug  : force a script family (latin, bengali, devanagari, ...)
// Read commands (decide/bench/calibrate/recall) fall back to the base model
// with an honest note when the routed substrate is missing; learn CREATES
// the routed substrate (that is how per-script fabrics grow).
//
// Trigram policy: when the user did not pass --ngrams explicitly, sub-word
// bridges ACTIVATE automatically whenever the routed substrate is non-Latin
// (the script barrier and tiny vocabularies make them load-bearing there);
// Latin keeps the word-level stream that reproduces the v2.1 baselines.
std::string route_model(const Args& a, const std::string& state_text,
                        std::string& note, bool for_learning) {
    note.clear();
    if (a.lang.empty()) return a.model;
    const si::script::Script sc = (a.lang == "auto")
        ? si::script::detect_script(state_text)
        : si::script::from_slug(a.lang);
    if (sc == si::script::Script::Unknown) {
        note = "script=unknown; using base model";
        return a.model;
    }
    if (a.ngrams_mode == 0)                            // policy default (explicit flag wins)
        si::norm::grams_enabled() = (sc != si::script::Script::Latin);
    const std::string suffix = std::string("-") + si::script::slug(sc);
    if (a.model.size() > suffix.size() &&
        a.model.compare(a.model.size() - suffix.size(), suffix.size(), suffix) == 0)
        return a.model;                              // already the routed substrate
    const std::string routed = a.model + suffix;
    if (!for_learning) {
        std::ifstream probe(routed + "/substrate.bin");
        if (!probe) {
            note = "no " + routed + " substrate trained; falling back to " + a.model
                 + " (honest silence still guards untaught vocabulary)";
            return a.model;
        }
    }
    note = "script=" + std::string(si::script::slug(sc)) + " -> " + routed;
    return routed;
}

void cmd_learn(const Args& a) {
    // Milestone-1 firewall: hidden/calibration splits never teach the fabric.
    if (!syfox::firewall::learn_may_read(a.examples)) {
        std::cerr << "syfox: firewall: " << a.examples << " is a "
                  << syfox::firewall::role_name(syfox::firewall::role_of_path(a.examples))
                  << " split — learn is refused (hidden rows never train, calibrate, "
                     "derive, or select models)\n";
        std::exit(2);
    }
    auto rows = load_jsonl(a.examples);
    const syfox::LearnPolicy lp{a.dedup, a.novelty, a.novelty_floor};
    long lessons = 0, skipped = 0;
    // v3: consolidation passes. The substrate's own forgetting law decays
    // every lane 0.995x per lesson, so a 25k-lesson SINGLE pass is
    // recency-truncated (early lanes are decayed away before training ends).
    // --epochs N re-teaches the same distinct lessons N times — measured on
    // game/guard hidden tests this recovers early knowledge (deterministically,
    // unlike accidental incremental accumulation). Distinctness still rules
    // per lesson: see the Milestone-2 A/B.
    const std::vector<const sfx::JV*> epoch_rows = [&]() {
        std::vector<const sfx::JV*> v;
        for (long e = 0; e < a.epochs; ++e)
            for (const auto& r : rows) v.push_back(&r);
        return v;
    }();
    if (!a.lang.empty()) {
        // v2.2 --lang: route every lesson by its script family into a
        // per-script substrate (<model>-<slug>). Latin gets its own substrate
        // like every other family — one fabric per script is the isolation the
        // routing contract promises. Group order is std::map order: deterministic.
        std::map<std::string, std::vector<const sfx::JV*>> groups;
        for (const auto& ex : rows) {
            const si::script::Script sc = (a.lang == "auto")
                ? si::script::detect_script(ex.at("state").as_str())
                : si::script::from_slug(a.lang);
            groups[si::script::slug(sc)].push_back(&ex);
        }
        sfx::JVArr routed;
        for (auto& g : groups) {
            const std::string dir = a.model + "-" + g.first;
            // trigram policy per script group (explicit --ngrams wins)
            si::norm::grams_enabled() = (a.ngrams_mode != 0)
                ? (a.ngrams_mode == 1) : (g.first != "latin");
            syfox::Engine eng;
            // first learn in a script family has no substrate yet: a failed
            // load here is a FRESH fabric (intended), not an error. Every
            // other command uses load_or_die().
            eng.load_model(resolve_model_dir(dir));
            eng.set_context(dir);                             // audit context tag (M3)
            // v3.1.3 BUGFIX: this loop MUST teach g.second (this script
            // family's rows), not epoch_rows (the whole file). The v2.2 code
            // iterated epoch_rows here, so every per-script substrate was
            // taught the ENTIRE corpus — the opposite of the isolation the
            // routing contract promises. Epochs are applied per family.
            for (long e = 0; e < a.epochs; ++e) {
                for (const auto* exp : g.second) {
                    const sfx::JV& qs = exp->at("questions");
                    const sfx::JV& labels = exp->at("labels");
                    const std::string state = exp->at("state").as_str();
                    for (const auto& qkv : qs.obj) {
                        const sfx::JV& q = qkv.second;
                        std::string type = q.at("type").as_str();
                        std::string label = labels.at(qkv.first).as_str();
                        bool learned = false;
                        if (type == "choice" || type == "score")
                            eng.learn_example(state, q.at("instructions").as_str(),
                                              outcome_text(q, label), a.augment, lp, &learned);
                        else if (type == "noul")
                            eng.learn_noul(state, q.at("instructions").as_str(),
                                           label == "true", a.augment, lp, &learned);
                        ++lessons;
                        if (!learned) ++skipped;
                    }
                }
            }
            eng.save_model(dir);
            routed.push_back(sfx::JV(sfx::JVObj{
                {"script", sfx::JV(g.first)}, {"model", sfx::JV(dir)},
                {"lessons", static_cast<double>(g.second.size())},
                {"dedup_skipped", static_cast<double>(0)},
                {"contradictions", static_cast<double>(eng.conflicts().size())},
                {"nodes", static_cast<double>(eng.substrate().node_count())},
                {"lanes", static_cast<double>(eng.substrate().lane_count())}}));
        }
        std::cout << sfx::JV(sfx::JVObj{
            {"command", sfx::JV("learn")}, {"examples", sfx::JV(a.examples)},
            {"lang", sfx::JV(a.lang)}, {"augment", sfx::JV(a.augment)},
            {"dedup", sfx::JV(a.dedup)}, {"novelty", sfx::JV(a.novelty)},
            {"epochs", static_cast<double>(a.epochs)},
            {"lessons", static_cast<double>(lessons)},
            {"dedup_skipped", static_cast<double>(skipped)},
            {"routed", sfx::JV(routed)},
            {"note", sfx::JV("lessons routed per script family: one SI substrate per script")}}).dump() << "\n";
        return;
    }
    syfox::Engine eng;
    eng.load_model(resolve_model_dir(a.model));                 // incremental if model exists (fresh = first learn)
    eng.set_context(a.model);                                   // audit context tag (M3)
    for (const auto* exp : epoch_rows) {
        std::string state = exp->at("state").as_str();
        const sfx::JV& qs = exp->at("questions");
        const sfx::JV& labels = exp->at("labels");
        for (const auto& qkv : qs.obj) {
            const sfx::JV& q = qkv.second;
            std::string type = q.at("type").as_str();
            std::string label = labels.at(qkv.first).as_str();
            bool learned = false;
            if (type == "choice" || type == "score")
                eng.learn_example(state, q.at("instructions").as_str(),
                                  outcome_text(q, label), a.augment, lp, &learned);
            else if (type == "noul")
                eng.learn_noul(state, q.at("instructions").as_str(),
                               label == "true", a.augment, lp, &learned);
            ++lessons;
            if (!learned) ++skipped;
        }
    }
    eng.save_model(a.model);
    std::cout << sfx::JV(sfx::JVObj{
        {"command", sfx::JV("learn")}, {"examples", sfx::JV(a.examples)},
        {"model", sfx::JV(a.model)}, {"augment", sfx::JV(a.augment)},
        {"dedup", sfx::JV(a.dedup)}, {"novelty", sfx::JV(a.novelty)},
        {"epochs", static_cast<double>(a.epochs)},
        {"lessons", static_cast<double>(lessons)},
        {"dedup_skipped", static_cast<double>(skipped)},
        {"nodes", static_cast<double>(eng.substrate().node_count())},
        {"lanes", static_cast<double>(eng.substrate().lane_count())},
        {"contradictions", static_cast<double>(eng.conflicts().size())},
        {"note", sfx::JV(a.dedup
            ? "exact duplicate lessons skipped (Milestone-2 distinct-experience policy)"
            : "every lesson taught (legacy behavior)")}}).dump() << "\n";
}

void cmd_calibrate(const Args& a) {
    // Milestone-1 firewall: the hidden test never sets a calibration scalar.
    if (!syfox::firewall::calibrate_may_read(a.examples)) {
        std::cerr << "syfox: firewall: " << a.examples << " is a HIDDEN test split"
                  << " — calibrate is refused (hidden rows never participate in "
                     "calibration, training, derivation, or model selection)\n";
        std::exit(2);
    }
    // v2.2 --lang: calibrate the substrate the examples route to (fitting a
    // different script's substrate would set scalars on a fabric that never
    // saw the rows — meaningless). Dominant script over the file's states.
    std::string lang_note;
    std::string model_dir = a.model;
    if (!a.lang.empty()) {
        auto probe_rows = load_jsonl(a.examples);
        std::string agg;
        for (const auto& r : probe_rows) if (r.has("state")) agg += r.at("state").as_str() + "\n";
        model_dir = route_model(a, agg, lang_note, false);
    }
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(model_dir), "calibration");
    if (a.energy_norm) eng.set_energy_norm(true);
    auto rows = load_jsonl(a.examples);
    auto calib_rows = eng.harvest_rows(rows);
    // v2.1 (P6): report calibration honestly — MULTI-CLASS ECE on the fit
    // rows BEFORE (T=1 / default Platt) and AFTER (whatever fit_calibration
    // adopted; its 1-bit ECE guard may keep T=1). Same helper the fit uses,
    // same definition bench reports.
    eng.fit_calibration(calib_rows);
    eng.save_model(model_dir);
    const auto& c = eng.calibration();
    const auto r4 = [](double v) { return std::round(v * 10000.0) / 10000.0; };
    sfx::JVObj o;
    o["command"] = sfx::JV("calibrate");
    o["model"] = sfx::JV(model_dir);
    if (!lang_note.empty()) o["lang_note"] = sfx::JV(lang_note);
    o["fit_rows"] = sfx::JV(static_cast<double>(calib_rows.size()));
    o["fit_source"] = sfx::JV(a.examples);
    o["choice_temperature"] = std::round(c.choice_temperature * 10000.0) / 10000.0;
    o["score_temperature"] = std::round(c.score_temperature * 10000.0) / 10000.0;
    o["noul_a"] = std::round(c.noul_a * 10000.0) / 10000.0;
    o["noul_b"] = std::round(c.noul_b * 10000.0) / 10000.0;
    o["multiclass_ece_before"] = sfx::JV(sfx::JVObj{    // T = 1, default Platt
        {"choice", r4(syfox::Engine::calibration_ece(calib_rows, "choice", 1.0f, 6.0f, -3.0f))},
        {"score", r4(syfox::Engine::calibration_ece(calib_rows, "score", 1.0f, 6.0f, -3.0f))},
        {"noul", r4(syfox::Engine::calibration_ece(calib_rows, "noul", 1.0f, 6.0f, -3.0f))}});
    o["multiclass_ece_after"] = sfx::JV(sfx::JVObj{     // adopted parameters
        {"choice", r4(syfox::Engine::calibration_ece(calib_rows, "choice", c.choice_temperature, c.noul_a, c.noul_b))},
        {"score", r4(syfox::Engine::calibration_ece(calib_rows, "score", c.score_temperature, c.noul_a, c.noul_b))},
        {"noul", r4(syfox::Engine::calibration_ece(calib_rows, "noul", 1.0f, c.noul_a, c.noul_b))}});
    o["note"] = sfx::JV("fit rows supply energies only; if this is the held-out "
                        "split, the fabric itself never learned from them — the "
                        "fit sets 2-3 scalars (temperature/Platt), and argmax is "
                        "unaffected (temperature is monotone)");
    std::cout << sfx::JV(o).dump() << "\n";
}

sfx::JV answers_to_json(const std::vector<syfox::Answer>& ans, const syfox::Usage& u) {
    sfx::JVObj out;
    for (const auto& a : ans) {
        sfx::JVObj o;
        if (a.type == "choice") {
            o["choice"] = sfx::JV(a.choice);
            sfx::JVObj probs;
            for (const auto& p : a.probabilities) probs[p.first] = sfx::JV(std::round(p.second * 1000.0f) / 1000.0f);
            o["probabilities"] = sfx::JV(probs);
            o["confidence"] = sfx::JV(std::round(a.confidence * 1000.0f) / 1000.0f);
        } else if (a.type == "score") {
            o["score"] = sfx::JV(std::round(a.value * 1000.0f) / 1000.0f);
            sfx::JVObj probs;
            for (const auto& p : a.probabilities) probs[p.first] = sfx::JV(std::round(p.second * 1000.0f) / 1000.0f);
            o["probabilities"] = sfx::JV(probs);
            o["confidence"] = sfx::JV(std::round(a.confidence * 1000.0f) / 1000.0f);
        } else if (a.type == "noul") {
            o["noul"] = sfx::JV(std::round(a.probability * 1000.0f) / 1000.0f);
            o["confidence"] = sfx::JV(std::round(a.confidence * 1000.0f) / 1000.0f);
        }
        o["deferred"] = sfx::JV(a.deferred);
        if (!a.reason.empty()) o["reason"] = sfx::JV(a.reason);
        // v3.3 readout disclosures
        if (a.tied) o["tied"] = sfx::JV(true);
        if (!a.gate_tokens.empty()) {
            sfx::JVArr gt;
            for (const auto& t : a.gate_tokens) gt.push_back(sfx::JV(t));
            o["question_gate"] = sfx::JV(gt);
        }
        out[a.qid] = sfx::JV(o);
    }
    sfx::JVObj usage{
        {"state_tokens", static_cast<double>(u.state_tokens)},
        {"vocabulary", static_cast<double>(u.vocabulary)},
        {"lanes", static_cast<double>(u.lanes)},
        {"settled_energy", std::round(u.settled_energy * 1000.0f) / 1000.0f},
        {"calibrated", sfx::JV(u.calibrated)},
        {"engine", std::string("syfox-") + syfox::VERSION},
        {"core", "si-substrate"},
    };
    if (!u.retrieved.empty()) {
        sfx::JVArr ret;
        for (const auto& r : u.retrieved)
            ret.push_back(sfx::JV(sfx::JVObj{
                {"label", sfx::JV(r.first)},
                {"resonance", std::round(r.second * 1000.0f) / 1000.0f}}));
        usage["retrieval"] = sfx::JV(ret);          // v3.2: memories that primed this decision
    }
    if (u.ctx_gate) {                               // v3.4: two-stage settle disclosure
        usage["ctx_gate"] = sfx::JV(true);
        usage["ctx_alpha"] = std::round(u.ctx_alpha * 1000.0) / 1000.0;
    }
    return sfx::JV(sfx::JVObj{{"answers", sfx::JV(out)}, {"usage", sfx::JV(usage)}});
}

void cmd_decide(const Args& a) {
    std::string state = a.state_file ? read_file(a.state) : a.state;
    std::string qtext = a.questions_file ? read_file(a.questions) : a.questions;
    sfx::JV questions = sfx::JV::parse(qtext);
    if (!questions.is_obj()) { std::cerr << "syfox: questions must be a JSON object\n"; std::exit(2); }
    std::string lang_note;
    std::string model_dir = route_model(a, state, lang_note, false);
    sfx::JVObj route_report;
    // -- v3.2 two-stage physics router ---------------------------------------
    // Stage 1: a SMALL dedicated router fabric (500-node class, one anchor per
    // domain) settles the state and picks the domain anchor — pure field
    // dynamics, same substrate physics, no classifier.
    // Stage 2: the domain model mapped in the router's router.json (models:
    // {anchor: model-dir}) decides the actual questions; --model stays as the
    // fallback domain layer when the mapping misses. Both stages are
    // independent SI settles; the route is disclosed in the output.
    if (!a.router.empty()) {
        const std::string router_dir = resolve_model_dir(a.router);
        syfox::Engine reng;
        load_or_die(reng, router_dir, "router");
        apply_modes(reng, a);
        if (a.energy_norm) reng.set_energy_norm(true);   // M1 gain for the router too
        sfx::JV rschema(sfx::JVObj{});
        {
            std::ifstream rf(router_dir + "/router.json");
            if (rf) {
                std::string buf((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
                rschema = sfx::JV::parse(buf);
            }
        }
        if (!rschema.has("anchors") || !rschema.at("anchors").is_obj()) {
            std::cerr << "syfox: router model " << router_dir
                      << " lacks router.json anchors (expected "
                      << router_dir << "/router.json with an \"anchors\" object; "
                      << "build one with tools/router_prepare.py)\n";
            std::exit(2);
        }
        sfx::JV rqs(sfx::JVObj{
            {"route", sfx::JV(sfx::JVObj{
                {"type", sfx::JV("choice")},
                {"instructions", sfx::JV("which domain does this state belong to")},
                {"criteria", rschema.at("anchors")}})}});
        syfox::Usage ru;
        auto rans = reng.decide(state, rqs, ru);
        ru.calibrated = reng.calibration().fitted;
        const syfox::Answer& ra = rans[0];
        sfx::JVObj rj;
        rj["anchor"] = sfx::JV(ra.deferred ? std::string() : ra.choice);
        rj["confidence"] = sfx::JV(std::round(ra.confidence * 1000.0f) / 1000.0f);
        rj["deferred"] = sfx::JV(ra.deferred);
        std::vector<std::pair<float, std::string>> ranked;
        for (const auto& p : ra.probabilities) ranked.emplace_back(p.second, p.first);
        std::sort(ranked.begin(), ranked.end(), [](const auto& x, const auto& y){ return x.first > y.first; });
        sfx::JVArr top3;
        for (std::size_t i = 0; i < ranked.size() && i < 3; ++i)
            top3.push_back(sfx::JV(sfx::JVObj{
                {"anchor", sfx::JV(ranked[i].second)},
                {"p", sfx::JV(std::round(ranked[i].first * 1000.0f) / 1000.0f)}}));
        rj["top"] = sfx::JV(top3);
        if (rschema.has("models") && rschema.at("models").is_obj()
            && !ra.deferred && rschema.at("models").has(ra.choice)) {
            // v3.2.1: the mapping target resolves across repo layouts too —
            // an HF-packaged router.json says "model-b77-sem" while the dir
            // lives at model/b77-sem.
            model_dir = resolve_model_dir(rschema.at("models").at(ra.choice).as_str());
            rj["model"] = sfx::JV(model_dir);
        }
        route_report = std::move(rj);
    }
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(model_dir), "decide");
    apply_modes(eng, a);
    if (a.energy_norm) eng.set_energy_norm(true);   // Milestone-1 gain knob
    warn_inert_switches(eng, a, model_dir);
    // --memories FILE (v3.2): explicit memory store for decide — overrides any
    // model-dir memories.jsonl for this process. Rows: {"label":..., "state":...}.
    if (!a.memories.empty()) {
        std::vector<syfox::recall::Memory> mems;
        for (const auto& r : load_jsonl(a.memories)) {
            syfox::recall::Memory m;
            m.label = r.at("label").as_str();
            m.state = si::norm::normalize(r.at("state").as_str());
            if (!m.label.empty() && !m.state.empty()) mems.push_back(std::move(m));
        }
        if (!mems.empty()) eng.set_memories(std::move(mems));
    }
    syfox::Usage u;
    auto answers = eng.decide(state, questions, u);
    u.calibrated = eng.calibration().fitted;   // decide() resets Usage; set after
    sfx::JV out = answers_to_json(answers, u);
    if (!lang_note.empty()) out.obj["lang_note"] = sfx::JV(lang_note);
    if (!route_report.empty()) out.obj["route"] = sfx::JV(route_report);
    // v3 Milestone 3: machine-auditable evidence — supporting lanes with
    // provenance, plus any contradiction records for this exact state.
    if (a.evidence)
        out.obj["evidence"] = eng.evidence_json(state, questions, answers);
    // v2.2 active-learning loop, step 1: log deferrals for human labeling.
    // Each row carries the state + the FULL question schema, so a labeled
    // row is directly teachable with `syfox learn` — no reconstruction step.
    if (!a.deferrals.empty()) {
        bool any = false;
        sfx::JVArr def;
        for (const auto& ans : answers)
            if (ans.deferred)
                def.push_back(sfx::JV(sfx::JVObj{
                    {"qid", sfx::JV(ans.qid)}, {"reason", sfx::JV(ans.reason)}}));
        if ((any = !def.empty())) {
            std::ofstream log(a.deferrals, std::ios::app);
            if (log)
                log << sfx::JV(sfx::JVObj{
                    {"state", sfx::JV(state)},
                    {"questions", questions},
                    {"deferred", sfx::JV(def)},
                    {"settled_energy", std::round(u.settled_energy * 1000.0f) / 1000.0f}}).dump() << "\n";
            out.obj["deferrals_logged"] = sfx::JV(static_cast<double>(def.size()));
        }
    }
    std::cout << out.dump() << "\n";
}

// ---------------------------------------------------------------------------
// Built-in demos. No hand-coded decision rules anywhere: every answer comes
// out of the settled field of the SI substrate.
// ---------------------------------------------------------------------------
struct Demo { std::string name, state, questions; };

const std::vector<Demo>& demos_for(const std::string& domain) {
    static const std::vector<Demo> tickets = {
        {"stripe broken, losing sales",
         "Hi, I've been trying to connect my stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
         R"({"department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems","sales":"pricing or account questions"}},"frustration":{"type":"score","instructions":"How frustrated the customer appears","criteria":["calm just stating facts","frustrated but civil","very angry strong language"]},"is_urgent":{"type":"noul","instructions":"the message conveys urgency or time sensitivity"}})"},
        {"double charge refund",
         "You charged me twice for the same invoice this month. Please refund the extra payment.",
         R"({"department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems","sales":"pricing or account questions"}},"frustration":{"type":"score","instructions":"How frustrated the customer appears","criteria":["calm just stating facts","frustrated but civil","very angry strong language"]},"is_urgent":{"type":"noul","instructions":"the message conveys urgency or time sensitivity"}})"},
        {"team plan pricing",
         "We want to upgrade to the team plan for twenty seats. Can you send the pricing?",
         R"({"department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems","sales":"pricing or account questions"}},"frustration":{"type":"score","instructions":"How frustrated the customer appears","criteria":["calm just stating facts","frustrated but civil","very angry strong language"]},"is_urgent":{"type":"noul","instructions":"the message conveys urgency or time sensitivity"}})"},
    };
    static const std::vector<Demo> game = {
        {"zombies at night",
         "Night. Zombies are spawning near the player. Health is dropping fast.",
         R"({"action":{"type":"choice","instructions":"What should the bot do next","criteria":{"flee":"run away escape avoid danger retreat safe","fight":"attack combat weapon sword strike","dig_in":"hide wait build shelter fortify safe"}}})"},
        {"calm day, build mode",
         "Daytime. No threats nearby. The player wants a safehouse and materials are available.",
         R"({"action":{"type":"choice","instructions":"What should the bot do next","criteria":{"flee":"run away escape avoid danger retreat safe","fight":"attack combat weapon sword strike","dig_in":"hide wait build shelter fortify safe"}}})"},
        {"sword vs one zombie",
         "A single zombie at close range. Full health. The player holds an iron sword.",
         R"({"action":{"type":"choice","instructions":"What should the bot do next","criteria":{"flee":"run away escape avoid danger retreat safe","fight":"attack combat weapon sword strike","dig_in":"hide wait build shelter fortify safe"}}})"},
    };
    static const std::vector<Demo> guard = {
        {"rm -rf on a coding task",
         "task: add a column to the users table. plan: run the sql migration. command: rm -rf build/",
         R"({"irreversible":{"type":"noul","instructions":"the command is irreversible or destructive"},"off_task":{"type":"noul","instructions":"the command is off task and unrelated to the goal"},"scope":{"type":"choice","instructions":"what does the command touch","criteria":{"none":"no changes at all","read":"only reads lists shows","write":"modifies project files or data","global":"system wide destructive or irreversible"}}})"},
        {"intended db reset",
         "task: reset the database. plan: restore from seed. command: make db-reset",
         R"({"irreversible":{"type":"noul","instructions":"the command is irreversible or destructive"},"off_task":{"type":"noul","instructions":"the command is off task and unrelated to the goal"},"scope":{"type":"choice","instructions":"what does the command touch","criteria":{"none":"no changes at all","read":"only reads lists shows","write":"modifies project files or data","global":"system wide destructive or irreversible"}}})"},
        {"harmless read",
         "task: list the open issues. plan: check the tracker. command: gh issue list",
         R"({"irreversible":{"type":"noul","instructions":"the command is irreversible or destructive"},"off_task":{"type":"noul","instructions":"the command is off task and unrelated to the goal"},"scope":{"type":"choice","instructions":"what does the command touch","criteria":{"none":"no changes at all","read":"only reads lists shows","write":"modifies project files or data","global":"system wide destructive or irreversible"}}})"},
    };
    if (domain == "game") return game;
    if (domain == "guard") return guard;
    return tickets;
}

void cmd_demo(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "demo");
    apply_modes(eng, a);
    std::cout << "SyFox demo — domain: " << (a.domain.empty() ? "tickets" : a.domain)
              << " | core: si-substrate (no transformer, no classifier)\n";
    for (const auto& d : demos_for(a.domain)) {
        std::cout << "\n== " << d.name << " ==\n";
        syfox::Usage u;
        auto answers = eng.decide(d.state, sfx::JV::parse(d.questions), u);
        u.calibrated = eng.calibration().fitted;   // decide() resets Usage; set after
        std::cout << answers_to_json(answers, u).dump() << "\n";
    }
}

void cmd_stats(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "stats");
    apply_modes(eng, a);   // stats reflects the modes this process would run under
    std::cout << sfx::JV(sfx::JVObj{
        {"nodes", static_cast<double>(eng.substrate().node_count())},
        {"lanes", static_cast<double>(eng.substrate().lane_count())},
        {"fabric_density", std::round(eng.substrate().fabric_density() * 1e6) / 1e6},
        {"mean_out_degree", std::round(eng.substrate().mean_out_degree() * 1e3) / 1e3},
        {"parallel_settle", sfx::JV(eng.substrate().parallel_settle_enabled())},
        {"calibrated", sfx::JV(eng.calibration().fitted)},
        {"choice_temperature", eng.calibration().choice_temperature},
        {"noul_a", eng.calibration().noul_a},
        {"noul_b", eng.calibration().noul_b},
        {"salience_gating", sfx::JV(eng.substrate().config().salience_gating)},
        {"miller_window", sfx::JV(eng.substrate().config().miller_window)},
        // v3.2 semantic layer + retrieval + hierarchy
        {"semantics", sfx::JV(eng.substrate().has_semantics() && !a.no_semantics)},
        {"sem_edges", static_cast<double>(eng.substrate().resonance_edge_count())},
        {"lane_contexts", static_cast<double>(eng.substrate().lane_context_count())},
        {"retrieval", sfx::JV(eng.retrieval_on() && !a.no_retrieval)},
        {"retrieval_memories", static_cast<double>(eng.memories().size())},
        {"hierarchy", sfx::JV(eng.hierarchy_on() && !a.no_hierarchy)}}).dump() << "\n";
}

// ---------------------------------------------------------------------------
// Jev-parity benchmark. Read-only; measures the axes the System One model
// class is judged on (accuracy, calibration, honesty, guardrail, latency,
// determinism). See core/bench.hpp for the axis-by-axis lineage.
//
// v2.1 eval-split honesty (P1): --split heldout scores the rows the fabric
// never learned from; the default train file is IN-SAMPLE and labelled as
// such in eval_source. --coverage-curve (P2) prints the coverage-vs-accuracy
// sweep — the headline metric, not top-1 accuracy.
// ---------------------------------------------------------------------------
void cmd_bench(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "bench");
    apply_modes(eng, a);                            // --threads (M5 OMP settle), SI modes
    if (a.energy_norm) eng.set_energy_norm(true);   // Milestone-1 gain knob
    std::string eval_path = !a.eval.empty() ? a.eval : a.examples;
    std::string split_note;
    if (!a.split.empty()) {
        if (!a.eval.empty()) {
            std::cerr << "syfox: pass either --eval FILE or --split NAME, not both\n";
            std::exit(2);
        }
        // domain name from the model dir: model-tickets -> data/tickets_<split>.jsonl
        std::string dom = a.model;
        const std::string pfx = "model-";
        if (dom.rfind(pfx, 0) == 0) dom = dom.substr(pfx.size());
        eval_path = "data/" + dom + "_" + a.split + ".jsonl";
        std::ifstream probe(eval_path);
        if (!probe) {
            std::cerr << "syfox: --split " << a.split << " -> expected " << eval_path
                      << " but it does not exist (run tools/split_data.py first, "
                      "or pass --eval FILE explicitly)\n";
            std::exit(2);
        }
        split_note = a.split == "heldout"
            ? "held-out 30% split; fabric never taught from these rows"
            : "train split; in-sample for the fabric";
    } else if (eval_path.empty()) {
        usage_exit();
    } else if (!a.examples.empty() && eval_path == a.examples) {
        split_note = "train split; in-sample for the fabric";
    }
    auto rows = load_jsonl(eval_path);
    if (rows.empty()) { std::cerr << "syfox: no eval rows in " << eval_path << "\n"; std::exit(2); }

    // v3 Milestone 5 --throughput N: batched multicore decisions over N worker
    // threads, each owning a PRIVATE engine copy (decisions mutate the field,
    // so workers never share a substrate). The metric is decisions/sec — a
    // PERFORMANCE axis, not an accuracy headline: a checksum proves the work
    // happened, argmaxes are not scored here. Determinism/accuracy numbers
    // come only from the default sequential (or --threads OMP) runs.
    if (a.throughput > 0) {
        const std::size_t T = static_cast<std::size_t>(a.throughput);
        auto probes = syfox::bench::eval_probes(rows);
        if (probes.empty()) { std::cerr << "syfox: no probes for throughput\n"; std::exit(2); }
        // single-thread in-process baseline (same probe order, same machine)
        long base_count = 0;
        double base_ck = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& p : probes) {
            syfox::Usage u;
            auto ans = eng.decide(p.state, p.questions, u);
            base_ck += ans.empty() ? 0.0 : static_cast<double>(ans[0].confidence);
            ++base_count;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double base_s = std::chrono::duration<double>(t1 - t0).count();
        // T workers, round-robin probe assignment, private engine copies
        std::vector<syfox::Engine> engines(T);
        for (auto& e : engines) e = eng;
        std::vector<long> counts(T, 0);
        std::vector<double> checks(T, 0.0);
        std::vector<std::thread> workers;
        const auto p0 = std::chrono::steady_clock::now();
        for (std::size_t t = 0; t < T; ++t) {
            workers.emplace_back([&engines, &probes, &counts, &checks, t]() {
                long c = 0; double ck = 0.0;
                for (std::size_t i = t; i < probes.size(); i += engines.size()) {
                    syfox::Usage u;
                    auto ans = engines[t].decide(probes[i].state, probes[i].questions, u);
                    ck += ans.empty() ? 0.0 : static_cast<double>(ans[0].confidence);
                    ++c;
                }
                counts[t] = c; checks[t] = ck;
            });
        }
        for (auto& w : workers) w.join();
        const auto p1 = std::chrono::steady_clock::now();
        const double par_s = std::chrono::duration<double>(p1 - p0).count();
        long total = 0; double ck = 0.0;
        for (std::size_t t = 0; t < T; ++t) { total += counts[t]; ck += checks[t]; }
        const double base_rate = base_s > 0 ? base_count / base_s : 0.0;
        const double par_rate  = par_s  > 0 ? total      / par_s  : 0.0;
        sfx::JVObj o;
        o["command"] = sfx::JV("bench");
        o["mode"] = sfx::JV("throughput");
        o["model"] = sfx::JV(a.model);
        o["eval_source"] = sfx::JV(eval_path);
        o["rows"] = static_cast<double>(rows.size());
        o["decisions"] = static_cast<double>(total);
        o["workers"] = static_cast<double>(T);
        o["sequential_decisions_per_sec"] = std::round(base_rate * 10.0) / 10.0;
        o["batched_decisions_per_sec"] = std::round(par_rate * 10.0) / 10.0;
        o["speedup"] = std::round((base_rate > 0 ? par_rate / base_rate : 0.0) * 1000.0) / 1000.0;
        o["work_checksum"] = std::round(ck * 1e6) / 1e6;
        o["note"] = sfx::JV("performance axis only: workers own private substrates, so "
                            "per-decision results are not comparable to the sequential "
                            "residue chain; accuracy headlines come from deterministic runs");
        if (!a.lang.empty()) o["lang_note"] = sfx::JV("throughput runs the base model; --lang routing is a read-path concern");
        std::cout << sfx::JV(o).dump() << "\n";
        return;
    }
    // v2.2 --lang: route the WHOLE eval to the substrate its dominant script
    // belongs to (per-row re-settling across engines would make the latency
    // axis meaningless). One honest note carries the routing decision.
    std::string lang_note;
    if (!a.lang.empty()) {
        std::string agg;
        for (const auto& r : rows) if (r.has("state")) agg += r.at("state").as_str() + "\n";
        const std::string model_dir = route_model(a, agg, lang_note, false);
        if (model_dir != a.model) load_or_die(eng, resolve_model_dir(model_dir), "bench (--lang routed)");
    }
    // v3 Milestone 4 --adversarial: the stress suite over the eval rows.
    // Read-only for the model under test; conflicts run on a throwaway copy.
    if (a.adversarial) {
        std::vector<sfx::JV> mix_rows;
        if (!a.mix.empty()) {
            mix_rows = load_jsonl(a.mix);
            if (mix_rows.empty()) {
                std::cerr << "syfox: no --mix rows in " << a.mix << "\n";
                std::exit(2);
            }
        }
        const std::string pool_src = a.mix.empty() ? a.model : a.mix;
        // pool provenance is recorded: builtin per-domain words, or --mix rows
        const std::vector<std::string> pool = a.mix.empty()
            ? syfox::bench::adv_builtin_pool(a.model)
            : syfox::bench::adv_pool_from_rows(mix_rows);
        auto rep = syfox::bench::adversarial_suite(eng, rows, pool);
        auto j = rep.to_json();
        j.obj["command"] = sfx::JV("bench");
        j.obj["mode"] = sfx::JV("adversarial");
        j.obj["model"] = sfx::JV(a.model);
        j.obj["eval_source"] = sfx::JV(eval_path);
        j.obj["rows"] = static_cast<double>(rows.size());
        j.obj["pool_source"] = sfx::JV(pool_src);
        j.obj["pool_words"] = static_cast<double>(pool.size());
        if (!lang_note.empty()) j.obj["lang_note"] = sfx::JV(lang_note);
        if (!split_note.empty()) j.obj["eval_split_note"] = sfx::JV(split_note);
        j.obj["firewall_note"] = sfx::JV("eval rows may be a _hidden split: bench is the "
                                         "only command allowed to read it, and it never teaches");
        std::cout << j.dump() << "\n";
        return;
    }
    // v2.2 --typos P: the measured typo-robustness claim. Same eval rows,
    // same engine, states deterministically corrupted (deletion / swap /
    // duplication chosen by word hash). Reports BOTH sides + the delta.
    if (a.typos > 0.0f) {
        std::vector<sfx::JV> corrupted = rows;
        for (auto& r : corrupted)
            if (r.is_obj() && r.has("state"))
                r.obj["state"] = sfx::JV(syfox::bench::corrupt_state(r.at("state").as_str(), a.typos));
        syfox::bench::BenchConfig bc;
        bc.latency_reps = static_cast<int>(a.latency_reps);
        bc.determinism_runs = static_cast<int>(a.replays);
        auto rep_c = syfox::bench::run(eng, rows, eval_path, bc, a.model);
        auto rep_t = syfox::bench::run(eng, corrupted, eval_path, bc, a.model);
        sfx::JVObj o;
        o["command"] = sfx::JV("bench");
        o["model"] = sfx::JV(a.model);
        o["eval_source"] = sfx::JV(eval_path);
        if (!split_note.empty()) o["eval_split_note"] = sfx::JV(split_note);
        if (!lang_note.empty()) o["lang_note"] = sfx::JV(lang_note);
        o["typo_pct"] = sfx::JV(std::round(a.typos * 10.0f) / 10.0f);
        o["choice_accuracy_clean"] = std::round(rep_c.choice_accuracy * 10000.0) / 10000.0;
        o["choice_accuracy_typos"] = std::round(rep_t.choice_accuracy * 10000.0) / 10000.0;
        o["choice_accuracy_delta"] = std::round((rep_c.choice_accuracy - rep_t.choice_accuracy) * 10000.0) / 10000.0;
        o["clean"] = rep_c.to_json();
        o["typo"] = rep_t.to_json();
        o["note"] = sfx::JV("deterministic corruption: same word is corrupted the same way on every run — the sweep replays bit-identically");
        std::cout << sfx::JV(o).dump() << "\n";
        return;
    }

    if (a.coverage_curve) {
        auto cr = syfox::bench::coverage_curve(eng, rows, 20);
        std::cout << "coverage-vs-accuracy curve — " << a.model << " on " << eval_path
                  << (split_note.empty() ? "" : " (" + split_note + ")") << "\n";
        std::cout << "  total labelled choice/score questions: " << cr.total << "\n\n";
        std::cout << "  tau    emitted  correct  coverage  acc-within\n";
        for (const auto& p : cr.points)
            std::printf("  %.2f  %7ld  %7ld  %7.1f%%  %9.1f%%\n",
                        p.threshold, p.emitted, p.correct,
                        p.coverage * 100.0, p.accuracy * 100.0);
        std::cout << "\n" << syfox::bench::coverage_plot(cr);
        auto op = [&](const char* name, const syfox::bench::OperatingPoint& o) {
            std::printf("  op %-3s cov>=%.0f%%: ", name, o.target * 100.0);
            if (o.feasible)
                std::printf("tau=%.2f coverage=%.1f%% accuracy=%.1f%%\n",
                            o.threshold, o.coverage * 100.0, o.accuracy * 100.0);
            else
                std::printf("infeasible (best: coverage=%.1f%% at tau=0)\n",
                            o.coverage * 100.0);
        };
        op("50", cr.op50); op("70", cr.op70); op("90", cr.op90);
        auto j = cr.to_json();
        j.obj["model"] = sfx::JV(a.model);
        j.obj["eval_source"] = sfx::JV(eval_path);
        if (!split_note.empty()) j.obj["eval_split_note"] = sfx::JV(split_note);
        std::cout << "\njson: " << j.dump() << "\n";
        return;
    }

    syfox::bench::BenchConfig bc;
    bc.latency_reps = static_cast<int>(a.latency_reps);
    bc.determinism_runs = static_cast<int>(a.replays);
    syfox::bench::BenchReport rep =
        syfox::bench::run(eng, rows,
                          split_note.empty() ? eval_path
                                             : eval_path + " (" + split_note + ")",
                          bc, a.model);
    auto j = rep.to_json();
    if (!a.split.empty()) j.obj["eval_split"] = sfx::JV(a.split);
    if (!lang_note.empty()) j.obj["lang_note"] = sfx::JV(lang_note);
    std::cout << j.dump() << "\n";
}

// ---------------------------------------------------------------------------
// Associative recall through field dynamics (Hopfield-style, similarity in
// settled-energy space; no token comparison, no pattern matching).
// ---------------------------------------------------------------------------
void cmd_recall(const Args& a) {
    std::string lang_note;
    const std::string model_dir = route_model(a, a.state, lang_note, false);
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(model_dir), "recall");
    if (a.state.empty() || (a.memories.empty() && a.examples.empty())) usage_exit();
    std::vector<syfox::recall::Memory> memories;
    auto memory_from_row = [](const sfx::JV& r) -> syfox::recall::Memory {
        // two accepted schemas: {"state","label"} or the examples schema
        // {"state","labels"} (label = first label value, sorted-key order)
        std::string label;
        if (r.has("label")) label = r.at("label").as_str();
        else if (r.has("labels") && r.at("labels").is_obj() && !r.at("labels").obj.empty())
            label = r.at("labels").obj.begin()->second.as_str();
        return {label, si::norm::normalize(r.at("state").as_str())};
    };
    if (!a.memories.empty()) {
        for (const auto& r : load_jsonl(a.memories)) {
            if (!r.has("state")) continue;
            memories.push_back(memory_from_row(r));
        }
    } else {
        for (const auto& r : load_jsonl(a.examples)) {
            if (!r.has("state")) continue;
            memories.push_back(memory_from_row(r));
        }
    }
    auto hits = syfox::recall::recall(eng.substrate(), a.state, memories,
                                      static_cast<int>(a.topk));
    sfx::JVArr arr;
    for (const auto& h : hits)
        arr.push_back(sfx::JV(sfx::JVObj{
            {"label", sfx::JV(h.label)},
            {"resonance", std::round(h.resonance * 10000.0f) / 10000.0f},
            {"memory_index", static_cast<double>(h.index)}}));
    std::cout << sfx::JV(sfx::JVObj{
        {"command", sfx::JV("recall")},
        {"query", sfx::JV(a.state)},
        {"memories", static_cast<double>(memories.size())},
        {"hits", sfx::JV(arr)},
        {"lang_note", sfx::JV(lang_note.empty() ? "routing off" : lang_note)},
        {"note", sfx::JV("similarity measured in the settled-energy field; no token comparison, no pattern matching")}}).dump() << "\n";
}

// ---------------------------------------------------------------------------
// v2.2 Active Learning Loop — the deployment story for a substrate that can
// only know what it was taught. SyFox's honest silence is not a failure
// mode, it is a SIGNAL: every deferral is a state the fabric could not
// route. The loop: deploy with --log-deferrals -> collect the log ->
// `syfox active` dedups/ranks it into a labeling worksheet -> a human fills
// "labels" -> `syfox learn` re-teaches. Fine-tuning built from the
// substrate's own uncertainty instead of an external drift metric.
// ---------------------------------------------------------------------------
void cmd_active(const Args& a) {
    if (a.deferrals.empty() || a.out.empty()) usage_exit();
    std::map<std::string, long> counts;                 // state -> deferral count
    std::map<std::string, sfx::JV> qs_of;               // state -> question schema
    for (const auto& r : load_jsonl(a.deferrals)) {
        if (!r.is_obj() || !r.has("state") || !r.has("questions")) continue;
        const std::string st = r.at("state").as_str();
        ++counts[st];
        if (qs_of.find(st) == qs_of.end()) qs_of[st] = r.at("questions");
    }
    // rank: deferral count desc, then state text asc — deterministic
    std::vector<std::pair<std::string, long>> ranked(counts.begin(), counts.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<std::string, long>& x, const std::pair<std::string, long>& y) {
                  if (x.second != y.second) return x.second > y.second;
                  return x.first < y.first;
              });
    std::ofstream out(a.out);
    if (!out) { std::cerr << "syfox: cannot write " << a.out << "\n"; std::exit(2); }
    long written = 0;
    for (const auto& kv : ranked) {
        if (kv.second < a.min_count) continue;
        out << sfx::JV(sfx::JVObj{
            {"state", sfx::JV(kv.first)},
            {"questions", qs_of[kv.first]},
            {"labels", sfx::JV(sfx::JVObj{})},          // <- the human fills this
            {"defer_count", static_cast<double>(kv.second)}}).dump() << "\n";
        ++written;
    }
    std::cout << sfx::JV(sfx::JVObj{
        {"command", sfx::JV("active")},
        {"deferral_log", sfx::JV(a.deferrals)},
        {"unique_deferred_states", static_cast<double>(counts.size())},
        {"min_count", static_cast<double>(a.min_count)},
        {"worksheet", sfx::JV(a.out)},
        {"rows_written", static_cast<double>(written)},
        {"note", sfx::JV("fill labels{} in the worksheet, then: syfox learn --model DIR --examples " + a.out)}}).dump() << "\n";
}

void cmd_derive(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "derive");
    // Derivation layer commands. All OFFLINE and EXPLICIT: decide() stays
    // read-only; nothing here runs implicitly. Dreaming never touches the
    // substrate — only a human-validated ledger line can become a lane.
    if (!a.gate.empty()) {
        // TRANSACTIONAL derivation: replay gate rows + close-call probes before
        // and after; any argmax flip reverts the fabric bit-for-bit.
        // v2.1 (P5): the gate rows are usually the HELD-OUT split, so the
        // report also carries gold-labelled accuracy before/after derivation.
        // Milestone-1 firewall: hidden rows never steer derivation.
        if (!syfox::firewall::derive_gate_may_read(a.gate)) {
            std::cerr << "syfox: firewall: " << a.gate << " is a HIDDEN test split"
                      << " — derive --gate is refused (hidden rows never participate "
                         "in derivation or model selection)\n";
            std::exit(2);
        }
        auto gate_rows = load_jsonl(a.gate);
        const auto acc_before = syfox::bench::labelled_accuracy(eng, gate_rows);
        std::vector<std::vector<std::string>> replay;
        const std::string replay_src = !a.examples.empty() ? a.examples : a.gate;
        for (const auto& ex : load_jsonl(replay_src))
            replay.push_back(si::norm::normalize(ex.at("state").as_str()));
        const std::string mode = !a.examples.empty() ? "harvest" : "compose";
        syfox::gate::GateConfig gc;
        // conservative derivation strength: the gate's recommended starting
        // point; anything that still flips a taught row is reverted outright
        auto rep = syfox::gate::gated_derive(eng, mode, gate_rows, replay, gc,
                                             syfox::derive::HarvestConfig::conservative(),
                                             syfox::derive::DeriveConfig::conservative());
        bool saved = false;
        const auto acc_after = rep.committed
            ? syfox::bench::labelled_accuracy(eng, gate_rows) : acc_before;
        if (rep.committed) { eng.save_model(a.model); saved = true; }
        std::string reason;
        if (!rep.committed) {
            if (rep.taught_flips > 0)
                reason = std::to_string(rep.taught_flips) + " taught argmax flip(s)";
            else if (rep.conf_inflated)
                reason = "mixed-state mean confidence rose (manufactured certainty)";
            else
                reason = "no commit condition met";
        }
        std::cout << sfx::JV(sfx::JVObj{
            {"command", sfx::JV("derive")},
            {"mode", sfx::JV(mode)},
            {"gated", sfx::JV(true)},
            {"gate", rep.to_json()},
            {"changed", static_cast<double>(rep.changed)},
            {"removed", static_cast<double>(rep.removed)},
            {"model_saved", sfx::JV(saved)},
            {"gate_rows", static_cast<double>(gate_rows.size())},
            {"gate_row_source", sfx::JV(a.gate)},
            {"harvest_source", sfx::JV(replay_src)},
            {"accuracy_before", std::round(acc_before.accuracy * 10000.0) / 10000.0},
            {"accuracy_after", std::round(acc_after.accuracy * 10000.0) / 10000.0},
            {"accuracy_n", static_cast<double>(acc_before.n)},
            {"verdict", sfx::JV(rep.committed ? "pass" : "revert")},
            {"revert_reason", sfx::JV(reason)},
            {"note", sfx::JV(rep.committed
                ? "gate passed: no replayed decision flipped, gold accuracy held; derived lanes committed"
                : "gate REVERTED the derivation: fabric restored bit-for-bit; model ships un-derived")}}).dump() << "\n";
        return;
    }
    if (!a.examples.empty()) {
        // dynamic harvest: replay states, let the field's own settle
        // dynamics nominate which pairs deserve a direct lane
        auto rows = load_jsonl(a.examples);
        std::vector<std::vector<std::string>> replay;
        for (const auto& ex : rows) replay.push_back(si::norm::normalize(ex.at("state").as_str()));
        syfox::derive::HarvestConfig hc;
        auto st = syfox::derive::harvest(eng.substrate(), replay, hc);
        eng.save_model(a.model);
        std::cout << sfx::JV(sfx::JVObj{
            {"command", sfx::JV("derive")},
            {"mode", sfx::JV("harvest")},
            {"gated", sfx::JV(false)},
            {"warning", sfx::JV("ungated derive can flip close-call decisions; pass --gate FILE.jsonl for the transactional no-regression gate")},
            {"states_replayed", static_cast<double>(replay.size())},
            {"created", static_cast<double>(st.created)},
            {"refreshed", static_cast<double>(st.refreshed)},
            {"dissolved", static_cast<double>(st.dissolved)},
            {"lanes", static_cast<double>(eng.substrate().lane_count())},
            {"note", sfx::JV("co-activation harvest: observed lanes untouched; gen-1 lanes re-verified on every run")}}).dump() << "\n";
    } else {
        // static compose: two-hop algebra over the fabric (sparse fabrics)
        syfox::derive::DeriveConfig cfg;
        syfox::derive::DeriveStats st = syfox::derive::run(eng.substrate(), cfg, 2);
        eng.save_model(a.model);
        std::cout << sfx::JV(sfx::JVObj{
            {"command", sfx::JV("derive")},
            {"mode", sfx::JV("compose")},
            {"gated", sfx::JV(false)},
            {"warning", sfx::JV("ungated derive can flip close-call decisions; pass --gate FILE.jsonl for the transactional no-regression gate")},
            {"created", static_cast<double>(st.created)},
            {"strengthened", static_cast<double>(st.strengthened)},
            {"healed", static_cast<double>(st.healed)},
            {"dissolved", static_cast<double>(st.dissolved)},
            {"lanes", static_cast<double>(eng.substrate().lane_count())},
            {"note", sfx::JV("derived lanes carry a generation; observed lanes were never weakened")}}).dump() << "\n";
    }
}

void cmd_dream(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "dream");
    syfox::derive::DreamConfig dc;
    auto cands = syfox::derive::dream(eng.substrate(), dc, a.seed, static_cast<int>(a.steps));
    const std::string ledger = a.model + "/mutations.jsonl";
    std::ofstream out(ledger, std::ios::app);
    long written = 0;
    for (const auto& c : cands) {
        if (!out) { std::cerr << "syfox: cannot write " << ledger << "\n"; break; }
        sfx::JVArr driven;
        for (si::NodeId id : c.driven) driven.push_back(sfx::JV(eng.substrate().concept_of(id)));
        out << sfx::JV(sfx::JVObj{
            {"driven", sfx::JV(driven)},
            {"emergent", sfx::JV(eng.substrate().concept_of(c.emergent))},
            {"support", std::round(c.support * 10000.0f) / 10000.0f},
            {"seed", static_cast<double>(a.seed)},
            {"validated", sfx::JV(false)}}).dump() << "\n";
        ++written;
    }
    std::cout << sfx::JV(sfx::JVObj{
        {"command", sfx::JV("dream")},
        {"candidates", static_cast<double>(written)},
        {"ledger", sfx::JV(ledger)},
        {"substrate_modified", sfx::JV(false)},
        {"note", sfx::JV("edit the ledger: set validated:true only on lines you vouch for, then run syfox promote")}}).dump() << "\n";
}

void cmd_promote(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "promote");
    const std::string ledger = a.model + "/mutations.jsonl";
    auto lines = load_jsonl(ledger);
    const float promote_gain = 6.0f;
    long applied = 0, unvalidated = 0, already = 0;
    sfx::JVArr updated;
    for (auto& line : lines) {
        if (!line.is_obj()) continue;
        const bool validated = line.has("validated") && line.at("validated").is_bool() && line.at("validated").b;
        const bool promoted  = line.has("promoted")  && line.at("promoted").is_bool()  && line.at("promoted").b;
        if (!validated) { ++unvalidated; updated.push_back(line); continue; }
        if (promoted)   { ++already;     updated.push_back(line); continue; }
        std::vector<std::string> driven;
        if (line.at("driven").is_arr())
            for (const auto& d : line.at("driven").arr) driven.push_back(d.as_str());
        const std::string emergent = line.at("emergent").as_str();
        const float support = static_cast<float>(line.at("support").as_num(0.0));
        syfox::derive::apply_promotion(eng.substrate(), driven, emergent, support, promote_gain);
        line.obj["promoted"] = sfx::JV(true);
        updated.push_back(line);
        ++applied;
    }
    if (applied > 0) {
        std::ofstream out(ledger, std::ios::trunc);
        for (const auto& l : updated) out << l.dump() << "\n";
        eng.save_model(a.model);
    }
    std::cout << sfx::JV(sfx::JVObj{
        {"command", sfx::JV("promote")},
        {"applied", static_cast<double>(applied)},
        {"unvalidated_skipped", static_cast<double>(unvalidated)},
        {"already_promoted", static_cast<double>(already)},
        {"model_saved", sfx::JV(applied > 0)}}).dump() << "\n";
}

void cmd_analogs(const Args& a) {
    syfox::Engine eng;
    load_or_die(eng, resolve_model_dir(a.model), "analogs");
    auto matches = syfox::derive::find_analogues(eng.substrate(), a.concept);
    sfx::JVArr arr;
    for (const auto& m : matches)
        arr.push_back(sfx::JV(sfx::JVObj{
            {"concept", sfx::JV(m.concept)},
            {"iso", std::round(m.iso * 1000.0f) / 1000.0f},
            {"hops", static_cast<double>(m.hops)},
            {"phi", std::round(m.phi * 1000.0f) / 1000.0f}}));
    std::cout << sfx::JV(sfx::JVObj{
        {"source", sfx::JV(a.concept)},
        {"analogs", sfx::JV(arr)},
        {"note", sfx::JV("high phi = structurally aligned AND fabric-distant: a transfer hypothesis, verify before use")}}).dump() << "\n";
}

void usage_exit() {
    std::cerr <<
        "syfox " << syfox::VERSION << " — System One decision engine (SI substrate core)\n"
        "usage:\n"
        "  syfox learn     --model DIR --examples FILE.jsonl\n"
        "  syfox calibrate --model DIR --examples FILE.jsonl\n"
        "  syfox decide    --model DIR --state '...' --questions '{...}'\n"
        "  syfox demo      --model DIR --domain tickets|game|guard\n"
        "  syfox stats     --model DIR\n"
        "  syfox derive    --model DIR [--gate FILE.jsonl] [--examples FILE.jsonl]\n"
        "  syfox dream     --model DIR [--steps N] [--seed S]\n"
        "  syfox promote   --model DIR\n"
        "  syfox analogs   --model DIR --concept WORD\n"
        "  syfox bench     --model DIR (--eval FILE.jsonl | --split train|heldout)\n"
        "                  [--coverage-curve] [--latency-reps N] [--replays N]\n"
        "                  (Jev-parity eval suite; the curve sweeps tau 0.0->1.0;\n"
        "                   --latency-reps/--replays size the timing/determinism\n"
        "                   passes — lower them for fast probes on large evals)\n"
        "  syfox recall    --model DIR --state '...' (--memories FILE.jsonl | --examples FILE.jsonl) [--topk N]\n"
        "  syfox active    --deferrals FILE.jsonl --out FILE.jsonl [--min-count N]\n"
        "  syfox version\neval-split honesty (v2.1): --split heldout scores data/<domain>_heldout.jsonl\n"
        "(rows the fabric never learned from); the default train file is in-sample.\n"
        "token normalization (v2.1): data/synonyms.txt folds synonyms + Porter-stems\n"
        "tokens before injection (--synonyms overrides the path; deterministic).\n"
        "multilingual boundary (v2.2): UTF-8 codepoint tokenization for any script;\n"
        "  --lang auto|<slug> routes each query/lesson to <model>-<script> (one SI\n"
        "  substrate per script family: latin, bengali, devanagari, cyrillic, ...);\n"
        "  non-Latin routed substrates enable the character trigram bridges\n"
        "  automatically (typo routing); --ngrams on|off overrides explicitly.\n"
        "variant lessons (v2.2): learn --augment re-teaches paraphrase/variant rows\n"
        "  WITHOUT mass re-deposition (lanes strengthen, acoustic mass unchanged).\n"
        "distinct-experience policy (v3, Milestone 2): learn --dedup skips exact\n"
        "  duplicate lessons; learn --novelty scales each lesson's Hebbian dose by\n"
        "  how much of its vocabulary is new (floor 0.25). Measured: distinct\n"
        "  experience scales, repetition does not — see the dose-response table.\n"
        "typo robustness (v2.2): bench --typos P corrupts P% of words deterministically\n"
        "  and reports clean vs corrupted accuracy.\n"
        "active learning (v2.2): decide --log-deferrals FILE records every deferral\n"
        "  with its question schema; syfox active turns the log into a labeling\n"
        "  worksheet; label it and re-learn. Deploy -> log -> label -> retrain.\n"
        "hidden-test firewall (v3): files ending _hidden.jsonl are bench-only —\n"
        "  learn, calibrate and derive --gate REFUSE them; _cal.jsonl fits\n"
        "  calibration scalars only. The 70/15/15 splits live in data/big/.\n"
        "energy normalization (v3, Milestone 1): --energy-norm scales the\n"
        "  decide-side injection dose by the mean sqrt(mass) of the state's\n"
        "  known tokens — a measurement gain for big-corpus fabrics where\n"
        "  acoustic mass would otherwise whisper below the silence floor.\n"
        "  Off by default; seed-model numbers are unchanged.\n"
        "auditable evidence (v3, Milestone 3): decide --evidence prints the\n"
        "  supporting lanes (weight, generation, support/counter events,\n"
        "  provenance window, context) and any contradiction records for the\n"
        "  exact state; learn writes conflicts.jsonl + lessons_index.jsonl.\n"
        "  Contradictory lessons NEVER silently override — they surface.\n"
        "adversarial suite (v3, Milestone 4): bench --adversarial [--mix FILE]\n"
        "  runs nine deterministic stress families (reorder, padding, typos,\n"
        "  intensifiers, self-contradiction, negation, double negation, unknown\n"
        "  concepts, near-miss / cross-domain) plus a conflicting-lessons attack\n"
        "  on a throwaway engine copy; reports accuracy, defer rate,\n"
        "  conf-when-wrong and false-confidence per family.\n"
        "multicore (v3, Milestone 5): --threads N runs the deterministic\n"
        "  parallel settle on OMP builds (per-thread scatter buffers combined\n"
        "  in fixed thread order — bit-identical to sequential, test-verified;\n"
        "  N=1 forces sequential). bench --throughput N measures batched\n"
        "  decisions/sec over N workers with private substrates — a\n"
        "  PERFORMANCE axis, not an accuracy headline. stats prints the fabric\n"
        "  density / mean out-degree that gate the GPU design (ARCHITECTURE\n"
        "  S15): no GPU claims without the density gate.\n"
        "consolidation (v3): learn --epochs N re-teaches the same distinct\n"
        "  lessons N times — the forgetting law (0.995x per lesson) makes a\n"
        "  single 25k-lesson pass recency-truncated; N passes recover early\n"
        "  knowledge deterministically.\n"
        "selection modes (SI-faithful, off by default, not saved into the model):\n"
        "  --salience-gating   rank settle sources by salience (motion history)\n"
        "                      instead of raw energy\n"
        "  --miller-window     live source cap drawn from [source_cap-4, source_cap]\n"
        "                      per decision (= [20,24] at the default cap 24;\n"
        "                      TSDA live_cap lineage, SI samples [5,9] at cap 9)\n"
        "semantic layer (v3.2, deterministic, no ML — default ON for models saved\n"
        "  by v3.2+; pre-v3.2 fabrics replay unchanged because they carry no\n"
        "  semantic tail):\n"
        "  Stage 1  omega_semantic: fixed scalar projection of each concept's\n"
        "           64-dim semantic vector (frequency encoding for resonance)\n"
        "  Stage 2  context-sensitive lanes: lanes learn required/forbidden\n"
        "           context words from the lessons that laid them; at settle a\n"
        "           mismatched lane carries less (forbidden context x0.20,\n"
        "           missing required x0.60..1.0 by match count)\n"
        "  Stage 3  semantic hierarchy: hierarchy.json in the model dir; stage 1\n"
        "           reads category anchors, stage 2 scales intent candidates\n"
        "           (--no-hierarchy disables)\n"
        "  Stage 4  semantic field: resonance edges (top-k cosine neighbours of\n"
        "           the fabric-grounded vectors) leak a small energy share to\n"
        "           semantically similar nodes at settle (conserved), and\n"
        "           readout adds a semantic-neighbour term\n"
        "  --no-semantics     runtime kill switch for the whole layer\n"
        "retrieval by default (v3.2): decide consults associative memory — a\n"
        "  memories.jsonl in the model dir (rows {\"label\":...,\"state\":...}) is\n"
        "  fingerprinted once at load; each decide ranks memories by settled-field\n"
        "  resonance and primes the field with the top-k outcomes at a faint dose\n"
        "  (0.30 x inject). Deterministic. Flags: --memories FILE (explicit store),\n"
        "  --retrieval-topk N (default 5), --retrieval-dose F, --no-retrieval.\n"
        "two-stage router (v3.2): decide --router DIR runs a small dedicated\n"
        "  router fabric (one anchor per domain; router.json holds anchors +\n"
        "  models mapping) as stage 1, then the mapped domain model decides —\n"
        "  physics-based routing, no classifier. The route is disclosed in the\n"
        "  output as route:{anchor,confidence,top,model}.\n"
        "question-conditioned readout (v3.3, opt-in): the question's NEW tokens\n"
        "  (present in the fabric, absent from the state — the \"who\" in \"Who\n"
        "  found the radio?\") gate the readout by lane mass: each candidate is\n"
        "  scaled by its share of question-to-anchor lane weight. Read-only\n"
        "  over the settled field; disclosed as question_gate:[tokens]. Flags:\n"
        "  --question-gate (enable), --question-gate-floor F (default 0.25).\n"
        "  Measured: fixes question-relevance probes (who/latest), costs 5.3\n"
        "  points on tickets-cal — hence opt-in, like the energy-norm gain.\n"
        "  Replies also disclose tied:true on exact top-2 probability ties, and\n"
        "  defer with reason unknown_candidates when NO candidate carries energy.\n"
        "honest defer for near-ties (v3.4, engine default ON): when the top two\n"
        "  probabilities are closer than the margin (default 0.05), the readout\n"
        "  does not carry a decision and the answer defers with reason\n"
        "  ambiguous_tie instead of a confident-looking coin-flip. The engine\n"
        "  owns it, so CLI, C API, HTTP bridge and bench share one behavior.\n"
        "  Flags: --defer-margin P (override), --no-defer (v3.3 answer-always).\n"
        "multi-hop readout walk (v3.4, opt-in): --hops N walks lanes BFS-style\n"
        "  from each probe anchor up to N levels with per-hop damping\n"
        "  hop_coupling/sqrt(hop+1) (hop 1 x0.707 ... hop 4 x0.447), levels\n"
        "  visited in ascending node-id order, width-capped at 64 per level —\n"
        "  deterministic. Default N=1 is the legacy single-hop readout,\n"
        "  bit-identical to every earlier version. Measured on chain probes:\n"
        "  2-hop chains already work at N=1 (diffusion); 3-4 hop chains gain\n"
        "  real margin at N=4. Readout-layer only; settle physics untouched.\n"
        "question-context gate (v3.4, opt-in): two-stage settle — stage 1\n"
        "  settles the question's own tokens (instructions + criteria\n"
        "  descriptions, no labels) into a context field; stage 2 re-settles the\n"
        "  state and the final field composes as (1-alpha)*state + alpha*context\n"
        "  (alpha default 0.5). Disclosed as usage.ctx_gate/ctx_alpha. Flags:\n"
        "  --ctx-gate (enable), --ctx-alpha F. Deterministic; settle physics\n"
        "  untouched (two settled fields composed at the decision layer).\n";
    std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage_exit();
    std::string cmd = argv[1];
    Args a;
    for (int i = 2; i < argc; ++i) {
        auto need = [&](std::string& dst, bool flag = false) {
            if (i + 1 >= argc) usage_exit();
            dst = argv[++i];
            if (flag) dst = "1";
        };
        std::string k = argv[i];
        if (k == "--model") need(a.model);
        else if (k == "--examples") need(a.examples);
        else if (k == "--state") need(a.state);
        else if (k == "--state-file") { need(a.state); a.state_file = true; }
        else if (k == "--questions") need(a.questions);
        else if (k == "--questions-file") { need(a.questions); a.questions_file = true; }
        else if (k == "--domain") need(a.domain);
        else if (k == "--concept") need(a.concept);
        else if (k == "--eval") need(a.eval);
        else if (k == "--gate") need(a.gate);
        else if (k == "--memories") need(a.memories);
        else if (k == "--split") need(a.split);
        else if (k == "--synonyms") need(a.synonyms);
        else if (k == "--lang") need(a.lang);
        else if (k == "--log-deferrals") need(a.deferrals);
        else if (k == "--deferrals") need(a.deferrals);   // active-command alias
        else if (k == "--out") need(a.out);
        else if (k == "--min-count") { if (i + 1 >= argc) usage_exit(); a.min_count = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--typos") { if (i + 1 >= argc) usage_exit(); a.typos = std::strtof(argv[++i], nullptr); }
        else if (k == "--latency-reps") { if (i + 1 >= argc) usage_exit(); a.latency_reps = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--replays") { if (i + 1 >= argc) usage_exit(); a.replays = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--augment") a.augment = true;
        else if (k == "--dedup") a.dedup = true;
        else if (k == "--novelty") a.novelty = true;
        else if (k == "--energy-norm") a.energy_norm = true;
        else if (k == "--defer-margin") { if (i + 1 >= argc) usage_exit(); a.defer_margin = std::strtof(argv[++i], nullptr); if (a.defer_margin < 0) usage_exit(); }
        else if (k == "--no-defer") a.no_defer = true;
        else if (k == "--hops") { if (i + 1 >= argc) usage_exit(); a.hops = std::strtol(argv[++i], nullptr, 10); if (a.hops < 1 || a.hops > 8) usage_exit(); }
        else if (k == "--ctx-gate") a.ctx_gate = true;
        else if (k == "--ctx-alpha") { if (i + 1 >= argc) usage_exit(); a.ctx_alpha = std::strtof(argv[++i], nullptr); if (a.ctx_alpha <= 0 || a.ctx_alpha > 1) usage_exit(); }
        else if (k == "--evidence") a.evidence = true;
        else if (k == "--adversarial") a.adversarial = true;
        else if (k == "--mix") { if (i + 1 >= argc) usage_exit(); a.mix = argv[++i]; }
        else if (k == "--threads") { if (i + 1 >= argc) usage_exit(); a.threads = std::strtol(argv[++i], nullptr, 10); if (a.threads < 0) usage_exit(); }
        else if (k == "--throughput") { if (i + 1 >= argc) usage_exit(); a.throughput = std::strtol(argv[++i], nullptr, 10); if (a.throughput < 1) usage_exit(); }
        else if (k == "--epochs") { if (i + 1 >= argc) usage_exit(); a.epochs = std::strtol(argv[++i], nullptr, 10); if (a.epochs < 1) usage_exit(); }
        else if (k == "--novelty-floor") { if (i + 1 >= argc) usage_exit(); a.novelty_floor = std::strtof(argv[++i], nullptr); }
        else if (k == "--ngrams") {
            if (i + 1 >= argc) usage_exit();
            const std::string v = argv[++i];
            if (v == "on") a.ngrams_mode = 1;
            else if (v == "off") a.ngrams_mode = -1;
            else usage_exit();
        }
        else if (k == "--coverage-curve") a.coverage_curve = true;
        else if (k == "--topk") { if (i + 1 >= argc) usage_exit(); a.topk = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--steps") { if (i + 1 >= argc) usage_exit(); a.steps = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--seed")  { if (i + 1 >= argc) usage_exit(); a.seed = std::strtoull(argv[++i], nullptr, 0); }
        else if (k == "--salience-gating") a.salience_gating = true;
        else if (k == "--miller-window") a.miller_window = true;
        // v3.2 semantic layer / retrieval / router
        else if (k == "--no-semantics") a.no_semantics = true;
        else if (k == "--no-retrieval") a.no_retrieval = true;
        else if (k == "--no-hierarchy") a.no_hierarchy = true;
        else if (k == "--retrieval-topk") { if (i + 1 >= argc) usage_exit(); a.retrieval_topk = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--retrieval-dose") { if (i + 1 >= argc) usage_exit(); a.retrieval_dose = std::strtof(argv[++i], nullptr); }
        else if (k == "--router") need(a.router);
        else if (k == "--question-gate") a.question_gate = true;
        else if (k == "--question-gate-floor") { if (i + 1 >= argc) usage_exit(); a.question_gate_floor = std::strtof(argv[++i], nullptr); }
        else usage_exit();
    }
    // v2.1 (P4): one synonym table for the whole process. --synonyms wins;
    // otherwise data/synonyms.txt when present; otherwise the embedded copy
    // (same content) inside normalize.hpp. Loaded BEFORE any command runs so
    // teach and decide always share one folding table.
    if (!a.synonyms.empty()) {
        si::norm::load_synonyms(a.synonyms);
    } else {
        std::ifstream def("data/synonyms.txt");
        if (def) si::norm::load_synonyms("data/synonyms.txt");
    }
    if (cmd == "version") { std::cout << "syfox " << syfox::VERSION << " (core: si-substrate)\n"; return 0; }
    if (cmd == "learn") { if (a.examples.empty()) usage_exit(); cmd_learn(a); return 0; }
    if (cmd == "calibrate") { if (a.examples.empty()) usage_exit(); cmd_calibrate(a); return 0; }
    if (cmd == "decide") { if (a.state.empty() || a.questions.empty()) usage_exit(); cmd_decide(a); return 0; }
    if (cmd == "demo") { cmd_demo(a); return 0; }
    if (cmd == "stats") { cmd_stats(a); return 0; }
    if (cmd == "derive") { cmd_derive(a); return 0; }
    if (cmd == "dream") { cmd_dream(a); return 0; }
    if (cmd == "promote") { cmd_promote(a); return 0; }
    if (cmd == "analogs") { if (a.concept.empty()) usage_exit(); cmd_analogs(a); return 0; }
    if (cmd == "bench") { cmd_bench(a); return 0; }
    if (cmd == "recall") { cmd_recall(a); return 0; }
    if (cmd == "active") { cmd_active(a); return 0; }
    usage_exit();
}
