// ============================================================================
//  SyFox CLI — learn / calibrate / decide / demo
//  (thin tooling around the SI substrate core; no logic lives here)
// ============================================================================
#include "core/syfox.hpp"
#include "core/derive.hpp"
#include "core/bench.hpp"
#include "core/gate.hpp"
#include "core/recall.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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
    bool state_file = false, questions_file = false;
    // SI-faithful selection modes (off by default; never persisted into the model)
    bool salience_gating = false, miller_window = false;
};

// Apply the CLI mode overrides after load_model(). Mode-neutral by design:
// substrate.bin stays untouched, flags live only for this process.
void apply_modes(syfox::Engine& eng, const Args& a) {
    eng.substrate().set_source_modes(a.salience_gating, a.miller_window);
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
    auto rows = load_jsonl(a.examples);
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
            eng.load_model(dir);                              // incremental if exists
            for (const auto* exp : g.second) {
                const sfx::JV& qs = exp->at("questions");
                const sfx::JV& labels = exp->at("labels");
                const std::string state = exp->at("state").as_str();
                for (const auto& qkv : qs.obj) {
                    const sfx::JV& q = qkv.second;
                    std::string type = q.at("type").as_str();
                    std::string label = labels.at(qkv.first).as_str();
                    if (type == "choice" || type == "score")
                        eng.learn_example(state, q.at("instructions").as_str(),
                                          outcome_text(q, label), a.augment);
                    else if (type == "noul")
                        eng.learn_noul(state, q.at("instructions").as_str(),
                                       label == "true", a.augment);
                }
            }
            eng.save_model(dir);
            routed.push_back(sfx::JV(sfx::JVObj{
                {"script", sfx::JV(g.first)}, {"model", sfx::JV(dir)},
                {"lessons", static_cast<double>(g.second.size())},
                {"nodes", static_cast<double>(eng.substrate().node_count())},
                {"lanes", static_cast<double>(eng.substrate().lane_count())}}));
        }
        std::cout << sfx::JV(sfx::JVObj{
            {"command", sfx::JV("learn")}, {"examples", sfx::JV(a.examples)},
            {"lang", sfx::JV(a.lang)}, {"augment", sfx::JV(a.augment)},
            {"routed", sfx::JV(routed)},
            {"note", sfx::JV("lessons routed per script family: one SI substrate per script")}}).dump() << "\n";
        return;
    }
    syfox::Engine eng;
    eng.load_model(a.model);                                    // incremental if model exists
    for (const auto& ex : rows) {
        std::string state = ex.at("state").as_str();
        const sfx::JV& qs = ex.at("questions");
        const sfx::JV& labels = ex.at("labels");
        for (const auto& qkv : qs.obj) {
            const sfx::JV& q = qkv.second;
            std::string type = q.at("type").as_str();
            std::string label = labels.at(qkv.first).as_str();
            if (type == "choice" || type == "score")
                eng.learn_example(state, q.at("instructions").as_str(),
                                  outcome_text(q, label), a.augment);
            else if (type == "noul")
                eng.learn_noul(state, q.at("instructions").as_str(),
                               label == "true", a.augment);
        }
    }
    eng.save_model(a.model);
    std::cout << "syfox: learned " << rows.size() << " lessons -> " << a.model
              << " (nodes=" << eng.substrate().node_count()
              << ", lanes=" << eng.substrate().lane_count() << ")\n";
}

void cmd_calibrate(const Args& a) {
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
    eng.load_model(model_dir);
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
    return sfx::JV(sfx::JVObj{{"answers", sfx::JV(out)}, {"usage", sfx::JV(usage)}});
}

void cmd_decide(const Args& a) {
    std::string state = a.state_file ? read_file(a.state) : a.state;
    std::string qtext = a.questions_file ? read_file(a.questions) : a.questions;
    sfx::JV questions = sfx::JV::parse(qtext);
    if (!questions.is_obj()) { std::cerr << "syfox: questions must be a JSON object\n"; std::exit(2); }
    std::string lang_note;
    const std::string model_dir = route_model(a, state, lang_note, false);
    syfox::Engine eng;
    eng.load_model(model_dir);
    apply_modes(eng, a);
    syfox::Usage u;
    auto answers = eng.decide(state, questions, u);
    u.calibrated = eng.calibration().fitted;   // decide() resets Usage; set after
    sfx::JV out = answers_to_json(answers, u);
    if (!lang_note.empty()) out.obj["lang_note"] = sfx::JV(lang_note);
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
    eng.load_model(a.model);
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
    eng.load_model(a.model);
    apply_modes(eng, a);   // stats reflects the modes this process would run under
    std::cout << sfx::JV(sfx::JVObj{
        {"nodes", static_cast<double>(eng.substrate().node_count())},
        {"lanes", static_cast<double>(eng.substrate().lane_count())},
        {"calibrated", sfx::JV(eng.calibration().fitted)},
        {"choice_temperature", eng.calibration().choice_temperature},
        {"noul_a", eng.calibration().noul_a},
        {"noul_b", eng.calibration().noul_b},
        {"salience_gating", sfx::JV(eng.substrate().config().salience_gating)},
        {"miller_window", sfx::JV(eng.substrate().config().miller_window)}}).dump() << "\n";
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
    eng.load_model(a.model);
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
    // v2.2 --lang: route the WHOLE eval to the substrate its dominant script
    // belongs to (per-row re-settling across engines would make the latency
    // axis meaningless). One honest note carries the routing decision.
    std::string lang_note;
    if (!a.lang.empty()) {
        std::string agg;
        for (const auto& r : rows) if (r.has("state")) agg += r.at("state").as_str() + "\n";
        const std::string model_dir = route_model(a, agg, lang_note, false);
        if (model_dir != a.model) eng.load_model(model_dir);
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
    eng.load_model(model_dir);
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
    eng.load_model(a.model);
    // Derivation layer commands. All OFFLINE and EXPLICIT: decide() stays
    // read-only; nothing here runs implicitly. Dreaming never touches the
    // substrate — only a human-validated ledger line can become a lane.
    if (!a.gate.empty()) {
        // TRANSACTIONAL derivation: replay gate rows + close-call probes before
        // and after; any argmax flip reverts the fabric bit-for-bit.
        // v2.1 (P5): the gate rows are usually the HELD-OUT split, so the
        // report also carries gold-labelled accuracy before/after derivation.
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
    eng.load_model(a.model);
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
    eng.load_model(a.model);
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
    eng.load_model(a.model);
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
        "                  [--coverage-curve]   (Jev-parity eval suite; the curve\n"
        "                                        sweeps tau 0.0->1.0 and reports\n"
        "                                        coverage %% vs accuracy-within)\n"
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
        "typo robustness (v2.2): bench --typos P corrupts P% of words deterministically\n"
        "  and reports clean vs corrupted accuracy.\n"
        "active learning (v2.2): decide --log-deferrals FILE records every deferral\n"
        "  with its question schema; syfox active turns the log into a labeling\n"
        "  worksheet; label it and re-learn. Deploy -> log -> label -> retrain.\n"
        "selection modes (SI-faithful, off by default, not saved into the model):\n"
        "  --salience-gating   rank settle sources by salience (motion history)\n"
        "                      instead of raw energy\n"
        "  --miller-window     live source cap drawn from [source_cap-4, source_cap]\n"
        "                      per decision (= [20,24] at the default cap 24;\n"
        "                      TSDA live_cap lineage, SI samples [5,9] at cap 9)\n";
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
        else if (k == "--augment") a.augment = true;
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
