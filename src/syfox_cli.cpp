// ============================================================================
//  SyFox CLI — learn / calibrate / decide / demo
//  (thin tooling around the SI substrate core; no logic lives here)
// ============================================================================
#include "core/syfox.hpp"
#include "core/derive.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model = "model";
    std::string examples;
    std::string state;
    std::string questions;
    std::string domain;
    std::string concept;
    long steps = 64;                 // dream steps
    unsigned long long seed = 0x5EED5EEDull;  // dream seed (deterministic by default)
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

void cmd_learn(const Args& a) {
    syfox::Engine eng;
    eng.load_model(a.model);                                    // incremental if model exists
    auto rows = load_jsonl(a.examples);
    for (const auto& ex : rows) {
        std::string state = ex.at("state").as_str();
        const sfx::JV& qs = ex.at("questions");
        const sfx::JV& labels = ex.at("labels");
        for (const auto& qkv : qs.obj) {
            const sfx::JV& q = qkv.second;
            std::string type = q.at("type").as_str();
            std::string label = labels.at(qkv.first).as_str();
            if (type == "choice" || type == "score")
                eng.learn_example(state, q.at("instructions").as_str(), outcome_text(q, label));
            else if (type == "noul")
                eng.learn_noul(state, q.at("instructions").as_str(), label == "true");
        }
    }
    eng.save_model(a.model);
    std::cout << "syfox: learned " << rows.size() << " lessons -> " << a.model
              << " (nodes=" << eng.substrate().node_count()
              << ", lanes=" << eng.substrate().lane_count() << ")\n";
}

void cmd_calibrate(const Args& a) {
    syfox::Engine eng;
    eng.load_model(a.model);
    auto rows = load_jsonl(a.examples);
    auto calib_rows = eng.harvest_rows(rows);
    eng.fit_calibration(calib_rows);
    eng.save_model(a.model);
    std::cout << "syfox: calibrated on " << calib_rows.size() << " readout rows -> "
              << a.model << "/calibration.json\n";
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
    syfox::Engine eng;
    eng.load_model(a.model);
    apply_modes(eng, a);
    std::string state = a.state_file ? read_file(a.state) : a.state;
    std::string qtext = a.questions_file ? read_file(a.questions) : a.questions;
    sfx::JV questions = sfx::JV::parse(qtext);
    if (!questions.is_obj()) { std::cerr << "syfox: questions must be a JSON object\n"; std::exit(2); }
    syfox::Usage u;
    auto answers = eng.decide(state, questions, u);
    u.calibrated = eng.calibration().fitted;   // decide() resets Usage; set after
    std::cout << answers_to_json(answers, u).dump() << "\n";
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
// Derivation layer commands. All OFFLINE and EXPLICIT: decide() stays
// read-only; nothing here runs implicitly. Dreaming never touches the
// substrate — only a human-validated ledger line can become a lane.
// ---------------------------------------------------------------------------
void cmd_derive(const Args& a) {
    syfox::Engine eng;
    eng.load_model(a.model);
    if (!a.examples.empty()) {
        // dynamic harvest: replay states, let the field's own settle
        // dynamics nominate which pairs deserve a direct lane
        auto rows = load_jsonl(a.examples);
        std::vector<std::vector<std::string>> replay;
        for (const auto& ex : rows) replay.push_back(si::tokenize(ex.at("state").as_str()));
        syfox::derive::HarvestConfig hc;
        auto st = syfox::derive::harvest(eng.substrate(), replay, hc);
        eng.save_model(a.model);
        std::cout << sfx::JV(sfx::JVObj{
            {"command", sfx::JV("derive")},
            {"mode", sfx::JV("harvest")},
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
        "  syfox derive    --model DIR\n"
        "  syfox dream     --model DIR [--steps N] [--seed S]\n"
        "  syfox promote   --model DIR\n"
        "  syfox analogs   --model DIR --concept WORD\n"
        "  syfox version\n"
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
        else if (k == "--steps") { if (i + 1 >= argc) usage_exit(); a.steps = std::strtol(argv[++i], nullptr, 10); }
        else if (k == "--seed")  { if (i + 1 >= argc) usage_exit(); a.seed = std::strtoull(argv[++i], nullptr, 0); }
        else if (k == "--salience-gating") a.salience_gating = true;
        else if (k == "--miller-window") a.miller_window = true;
        else usage_exit();
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
    usage_exit();
}
