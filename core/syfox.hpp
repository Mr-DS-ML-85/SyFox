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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace syfox {

inline const char* VERSION = "0.2.0";

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
};

struct Usage {
    std::size_t state_tokens = 0;
    std::size_t vocabulary = 0;
    std::size_t lanes = 0;
    int settle_passes = 0;
    float settled_energy = 0.0f;
    bool calibrated = false;
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
    explicit Engine(si::SubstrateConfig cfg = {}) : si_(cfg) {}

    si::Substrate& substrate() { return si_; }
    const si::Substrate& substrate() const { return si_; }
    Calibration& calibration() { return calib_; }

    // -- model persistence ----------------------------------------------------
    void save_model(const std::string& dir) const {
        std::string cmd_mkdir = "mkdir -p '" + dir + "'";
        (void)std::system(cmd_mkdir.c_str());
        si_.save(dir + "/substrate.bin");
        std::ofstream cf(dir + "/calibration.json");
        cf << calib_.to_json().dump();
        std::ofstream mf(dir + "/meta.json");
        mf << sfx::JV(sfx::JVObj{
            {"engine", "syfox"}, {"version", VERSION},
            {"core", "si-substrate"}, {"nodes", static_cast<double>(si_.node_count())},
            {"lanes", static_cast<double>(si_.lane_count())}}).dump();
    }

    void load_model(const std::string& dir) {
        si_.load(dir + "/substrate.bin");
        std::ifstream cf(dir + "/calibration.json");
        if (cf) { std::string buf((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
                  calib_ = Calibration::from_json(sfx::JV::parse(buf)); }
    }

    // -- HEBBIAN LESSON (learning = substrate rewiring) -------------------------
    // One labelled experience: state text + question + rewarded outcome.
    //  1. all concepts join the vocabulary (intern; fresh nodes are light)
    //  2. adjacent state tokens bind (co-occurrence fabric for diffusion)
    //  3. state -> outcome lanes strengthen (fire together, wire together)
    void learn_example(const std::string& state_text, const std::string& instructions,
                       const std::string& outcome_text) {
        std::vector<std::string> state = si::tokenize(state_text + " " + instructions);
        std::vector<std::string> outcome = si::tokenize(outcome_text);
        for (const auto& t : state) si_.intern(t);
        for (const auto& t : outcome) si_.intern(t);
        for (std::size_t i = 1; i < state.size(); ++i)      // co-occurrence fabric
            si_.bind(si_.find(state[i - 1]), si_.find(state[i]), si_.config().learn_eta * 0.5f);
        si_.hebbian_lesson(state, outcome);
    }

    // -- CALIBRATION TOOL (external post-processor, not core physics) ----------
    // Fits temperature (choice/score) and Platt (noul) on collected rows:
    //   row = {type, energy_of_correct, energy_of_a_wrong (choice/score) or support (noul), y}
    struct CalibRow { std::string type; float e_correct; float e_wrong; int y; };

    void fit_calibration(const std::vector<CalibRow>& rows) {
        fit_temperature(rows, "choice", calib_.choice_temperature);
        fit_temperature(rows, "score",  calib_.score_temperature);
        fit_platt(rows);
        calib_.fitted = true;
        calib_.rows = rows.size();
    }

    // -- Noul lesson (valence-aware) ------------------------------------------
    // y=true  : Hebbian bind state concepts <-> instruction concepts
    //           (supporting evidence wires the route)
    // y=false : anti-Hebbian weaken the same routes (disconfirming evidence
    //           dissolves them), so the field discriminates, not accumulates.
    void learn_noul(const std::string& state_text, const std::string& instructions, bool y) {
        std::vector<std::string> state = si::tokenize(state_text);
        std::vector<std::string> instr = si::tokenize(instructions);
        for (const auto& t : state) si_.intern(t);
        for (const auto& t : instr) si_.intern(t);
        if (y) si_.hebbian_lesson(state, instr, 2.0f);   // supporting evidence binds hard
        else   si_.weaken(state, instr);                 // disconfirming evidence dissolves
    }

    // -- DECISION ----------------------------------------------------------------
    std::vector<Answer> decide(const std::string& state, const sfx::JV& questions, Usage& usage) {
        usage = Usage{};
        usage.vocabulary = si_.node_count();
        usage.lanes = si_.lane_count();

        std::vector<Answer> answers;
        std::vector<std::string> tokens = si::tokenize(state);
        usage.state_tokens = tokens.size();

        si_.reset_field();
        si_.inject(tokens);
        si_.settle();
        usage.settled_energy = si_.total_energy();

        // Honest silence: nothing settled => refuse to guess.
        bool silent = usage.settled_energy < si_.config().silence_floor;
        for (const auto& qkv : questions.obj) {
            const sfx::JV& q = qkv.second;
            Answer a;
            a.qid = qkv.first;
            a.type = q.at("type").as_str();
            if (silent) {
                a.deferred = true;
                a.reason = tokens.empty() ? "empty_state"
                         : (usage.settled_energy <= 0.0f ? "unknown_vocabulary" : "honest_silence");
                answers.push_back(a);
                continue;
            }
            if (a.type == "choice")   decide_choice(q, a);
            else if (a.type == "score") decide_score(q, a);
            else if (a.type == "noul")  decide_noul(q, a);
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
            si_.reset_field();
            si_.inject(si::tokenize(state));
            si_.settle();
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
                    for (const auto& e : energies) {
                        CalibRow r; r.type = type;
                        r.y = (e.first == label) ? 1 : 0;
                        if (r.y == 1) r.e_correct = e.second;
                        else          r.e_wrong   = e.second;
                        // store correct energy in e_correct; pair wrong from argmax wrong
                        if (r.y == 1) {
                            float worst = -1e9f;
                            for (const auto& e2 : energies)
                                if (e2.first != label) worst = std::max(worst, e2.second);
                            r.e_wrong = worst;
                            rows.push_back(r);
                        }
                    }
                } else if (type == "noul") {
                    float support = noul_support(q.at("instructions").as_str());
                    CalibRow r; r.type = "noul";
                    r.e_correct = support;
                    r.e_wrong = 0.0f;
                    r.y = (label == "true") ? 1 : 0;
                    rows.push_back(r);
                }
            }
        }
        return rows;
    }

private:
    // -- readout helpers --------------------------------------------------------
    float probe_energy(const std::string& label, const std::string& desc) const {
        return si_.readout(si::tokenize(label + " " + desc));
    }

    float noul_support(const std::string& instructions) const {
        // Noul uses the SUPPORT readout lane: breadth of corroboration.
        float e = si_.readout(si::tokenize(instructions), si::Substrate::ReadoutMode::Support);
        float total = si_.total_energy();
        if (total <= 0.0f) return 0.0f;
        float support = e / (total * 1.5f);                 // ratio in ~[0,1+]
        return std::min(1.5f, support);
    }

public:
    // debug/eval accessor (read-only)
    float noul_support_public(const std::string& instructions) const { return noul_support(instructions); }

private:

    static float softmax_ps(const std::vector<float>& e, std::vector<float>& p, float T) {
        float maxe = *std::max_element(e.begin(), e.end());
        float z = 0.0f;
        p.assign(e.size(), 0.0f);
        for (std::size_t i = 0; i < e.size(); ++i) { p[i] = std::exp((e[i] - maxe) / std::max(0.005f, T)); z += p[i]; }
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

    void decide_choice(const sfx::JV& q, Answer& a) {
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
        std::vector<float> p;
        softmax_ps(energies, p, calib_.choice_temperature);
        std::size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        a.choice = labels[best];
        for (std::size_t i = 0; i < labels.size(); ++i)
            a.probabilities.emplace_back(labels[i], p[i]);
        a.confidence = entropy_confidence(p);
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
    }

    void decide_noul(const sfx::JV& q, Answer& a) {
        float s = noul_support(q.at("instructions").as_str());
        a.probability = 1.0f / (1.0f + std::exp(-(calib_.noul_a * s + calib_.noul_b)));
        a.confidence = std::fabs(a.probability - 0.5f) * 2.0f;
    }

    // -- calibration fitting (tool-side numerical search, no core changes) ------
    static float nll_temperature(const std::vector<CalibRow>& rows, const std::string& type, float T) {
        float nll = 0.0f; std::size_t n = 0;
        for (const auto& r : rows) {
            if (r.type != type) continue;
            ++n;
            float ec = r.e_correct / std::max(0.005f, T);
            float ew = r.e_wrong  / std::max(0.005f, T);
            float m = std::max(ec, ew);
            // log softmax of correct over wrong
            nll -= (ec - m) - std::log(std::exp(ec - m) + std::exp(ew - m));
        }
        return n > 0 ? nll / static_cast<float>(n) : 0.0f;
    }

    static void fit_temperature(const std::vector<CalibRow>& rows, const std::string& type, float& T) {
        float lo = 0.005f, hi = 20.0f;
        for (int it = 0; it < 60; ++it) {                    // golden-section on NLL
            float m1 = lo + (hi - lo) / 3.0f, m2 = hi - (hi - lo) / 3.0f;
            if (nll_temperature(rows, type, m1) < nll_temperature(rows, type, m2)) hi = m2; else lo = m1;
        }
        if (nll_temperature(rows, type, (lo + hi) / 2) < nll_temperature(rows, type, 1.0f))
            T = (lo + hi) / 2.0f;
    }

    void fit_platt(const std::vector<CalibRow>& rows) {
        std::vector<std::pair<float, int>> pts;
        for (const auto& r : rows) if (r.type == "noul") pts.emplace_back(r.e_correct, r.y);
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
};

} // namespace syfox
