// ============================================================================
//  SyFox CLI — learn / calibrate / decide / demo
//  (thin tooling around the SI substrate core; no logic lives here)
// ============================================================================
#include "core/syfox.hpp"

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
    bool state_file = false, questions_file = false;
};

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

void usage_exit() {
    std::cerr <<
        "syfox " << syfox::VERSION << " — System One decision engine (SI substrate core)\n"
        "usage:\n"
        "  syfox learn     --model DIR --examples FILE.jsonl\n"
        "  syfox calibrate --model DIR --examples FILE.jsonl\n"
        "  syfox decide    --model DIR --state '...' --questions '{...}'\n"
        "  syfox demo      --model DIR --domain tickets|game|guard\n"
        "  syfox stats     --model DIR\n"
        "  syfox version\n";
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
        else usage_exit();
    }
    if (cmd == "version") { std::cout << "syfox " << syfox::VERSION << " (core: si-substrate)\n"; return 0; }
    if (cmd == "learn") { if (a.examples.empty()) usage_exit(); cmd_learn(a); return 0; }
    if (cmd == "calibrate") { if (a.examples.empty()) usage_exit(); cmd_calibrate(a); return 0; }
    if (cmd == "decide") { if (a.state.empty() || a.questions.empty()) usage_exit(); cmd_decide(a); return 0; }
    if (cmd == "demo") { cmd_demo(a); return 0; }
    if (cmd == "stats") { cmd_stats(a); return 0; }
    usage_exit();
}
