// ============================================================================
//  SyFox — bench.hpp
//  The Jev-parity evaluation harness: measures SyFox on the SAME axes the
//  System One model class is judged on (researched Sept 2026):
//
//    Jev published evidence                      |  this suite measures
//    --------------------------------------------+---------------------------
//    4-workflow decision accuracy 67.8%          |  routing accuracy
//      (vs GPT-5.6 Terra 67.9% / Sol 74.1%)      |  (choice/score argmax)
//    70-500 ms one-pass latency, 193.6x faster   |  p50/p95 us per decision
//    "calibrated Bayesian confidence"            |  ECE + conf/accuracy gap
//    "never hallucinates" (schema-bound)         |  OOD defer rate
//                                                |  (honest silence)
//    pi-warden guardrail 88% hold precision      |  noul hold precision,
//                                                |  recall, confusion
//    non-autoregressive one-pass determinism     |  double-run bit identity
//    (community: 3080-task classification bench) |  close-call margin dist
//
//  Honesty rules of the suite itself:
//    * v2.1: --split heldout scores the held-out 30% (rows the fabric never
//      learned from) — the headline number. The default train file is
//      IN-SAMPLE and the eval_source label says so. Accuracy is never
//      resubstitution unless the caller asks for the train split by name.
//    * every figure here is reproducible from this binary on the same inputs;
//      no number in the output exists without a runnable check behind it.
//    * the jev_reference block carries PUBLISHED figures for side-by-side
//      reading. They are not measured here and SyFox claims no parity with
//      them beyond sharing the axes.
//
//  Read-only: bench never mutates the model (decide() and the OOD probes
//  cannot — inject() skips unknown tokens).
// ============================================================================
#pragma once
#include "syfox.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace syfox {
namespace bench {

struct BenchConfig {
    int   ood_probes       = 24;    // untaught-vocabulary honesty probes
    int   closecall_probes = 48;    // cross-domain mixed states
    int   determinism_runs = 2;     // full eval replays, compared byte-wise
    int   latency_reps     = 20;    // timed repeats per eval row
    float closecall_margin = 0.10f; // argmax margin below this = close call
};

// ---------------------------------------------------------------------------
// Deterministic typo corruption (v2.2) — the measured form of the
// typo-robustness claim. Real user input arrives misspelled; this mutates
// eval states the way keyboards do and the bench reports the accuracy drop.
//
// Determinism: corruption decisions come from FNV-1a of the WORD ITSELF,
// not of the row or position — the same word is corrupted identically
// everywhere (row-stable comparisons, replay-stable across runs).
// Works on codepoints (UTF-8), so Bengali/Hindi words corrupt correctly
// (matra drops, consonant swaps). Words < 4 codepoints are never touched
// (mirrors the n-gram lane minimum). Ops: deletion / adjacent swap /
// duplication, chosen by hash bits.
// ---------------------------------------------------------------------------
inline std::string corrupt_state(const std::string& state, float pct) {
    if (pct <= 0.0f) return state;
    auto fnv64 = [](const std::string& s) {
        std::uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    };
    // decode once: per-codepoint strings + letter flags
    std::vector<std::string> cps;
    std::vector<bool>        letter;
    for (std::size_t i = 0; i < state.size();) {
        std::uint32_t cp = 0;
        const std::size_t len = si::script::decode_utf8(state, i, cp);
        cps.push_back(state.substr(i, len));
        letter.push_back(cp != 0xFFFFFFFFu && si::script::is_letter_cp(cp));
        i += len;
    }
    const unsigned want = static_cast<unsigned>(pct);
    std::size_t i = 0;
    while (i < cps.size()) {
        if (!letter[i]) { ++i; continue; }
        std::size_t j = i;
        while (j < cps.size() && letter[j]) ++j;          // letter run [i,j)
        if (j - i >= 4) {
            std::string w;
            for (std::size_t k = i; k < j; ++k) w += cps[k];
            const std::uint64_t h = fnv64(w);
            if (static_cast<unsigned>(h % 100ull) < want) {
                const std::size_t pos  = static_cast<std::size_t>((h >> 7) % (j - i));
                const std::size_t op   = static_cast<std::size_t>((h >> 19) % 3ull);
                if (op == 0) {                                        // deletion
                    cps.erase(cps.begin() + static_cast<std::ptrdiff_t>(i + pos));
                    --j;
                } else if (op == 1 && pos + 1 < j - i) {              // adjacent swap
                    std::swap(cps[i + pos], cps[i + pos + 1]);
                } else if (op == 1) {                                 // swap impossible at edge
                    cps.erase(cps.begin() + static_cast<std::ptrdiff_t>(i + pos));
                    --j;
                } else {                                              // duplication
                    cps.insert(cps.begin() + static_cast<std::ptrdiff_t>(i + pos), cps[i + pos]);
                    ++j;
                }
            }
        }
        i = j;
    }
    std::string out;
    for (const auto& c : cps) out += c;
    return out;
}

// A replay unit: one state + the typed questions to ask about it + the
// expected labels (qid -> label). Generated probes carry no labels and are
// scored only for margins/deferral, never for accuracy.
struct Probe {
    std::string state;
    sfx::JV     questions;
    std::map<std::string, std::string> labels;
    bool        taught = true;
};

// Deterministic nonsense vocabulary for the honesty probes. Every token is
// checked against the model's vocabulary before use; a token the model
// happens to know is skipped (no fake OOD). v2.2: the check extends to
// SUB-WORD material — a word whose trigrams overlap taught g3:* lanes
// ("teapot" vs taught "team": g3:tea) carries partial evidence under the
// trigram encoding and is NOT out-of-distribution anymore; probing with it
// would fake a silence violation. Truly unentangled vocabulary must still
// defer at 100% — that is the contract this probe guards.
inline std::vector<std::string> ood_states(const si::Substrate& s, int n) {
    static const char* kNonsense[] = {
        "zorblatz", "quibblemock", "framistan", "velocirapture", "glorgnax",
        "wubblefuzz", "krakenshadow", "mimsy", "borogove", "slartibartfast",
        "flimflam", "xylophone", "recluse", "brambleditch", "thundercouch",
        "pickleweasel", "gravyboat", "odyssey", "flapdoodle", "snickersnee",
        "hornswoggle", "muffin", "teapot", "gubbins", "collywobbles",
        "discombobulate", "kerfuffle", "wabbit", "skua", "fondue"
    };
    const int K = static_cast<int>(sizeof(kNonsense) / sizeof(kNonsense[0]));
    auto subword_entangled = [&s](const std::string& w) {
        for (const auto& g : si::norm::expand_ngrams({w}))
            if (s.has(g)) return true;
        return false;
    };
    std::vector<std::string> out;
    for (int i = 0; i < n && i < K; ++i) {
        std::vector<std::string> toks;
        for (int k = 0; k < 3; ++k) {
            const std::string w = kNonsense[(i * 3 + k) % K];
            if (!s.has(w) && !subword_entangled(w)) toks.push_back(w);  // unknown at word AND sub-word level
        }
        if (toks.empty()) continue;                    // all three known: skip
        std::string st;
        for (std::size_t k = 0; k < toks.size(); ++k) { if (k) st += " "; st += toks[k]; }
        out.push_back(st);
    }
    return out;
}

// The first label of a row in sorted-key order (JVObj is a std::map, so this
// is deterministic). Used for close-call pairing.
inline std::string first_label(const sfx::JV& row) {
    if (!row.has("labels") || !row.at("labels").is_obj() || row.at("labels").obj.empty())
        return "";
    return row.at("labels").obj.begin()->second.as_str();
}

// Mixed close-call probes: pair eval rows whose first labels differ and
// concatenate their states (A's questions + labels ride along). These are
// the states a routing fabric is most likely to flip. The derivation gate
// replays the SAME generator, so bench and gate numbers stay comparable.
inline std::vector<Probe> closecall_probes(const std::vector<sfx::JV>& rows, int cap) {
    std::vector<Probe> out;
    if (rows.size() < 2) return out;
    for (std::size_t i = 0; i < rows.size() && static_cast<int>(out.size()) < cap; ++i) {
        for (std::size_t j = i + 1; j < rows.size() && static_cast<int>(out.size()) < cap; ++j) {
            const std::string la = first_label(rows[i]), lb = first_label(rows[j]);
            if (la.empty() || lb.empty() || la == lb) continue;
            if (!rows[i].has("questions") || !rows[i].at("questions").is_obj()) continue;
            if (!rows[i].has("labels")  || !rows[i].at("labels").is_obj()) continue;
            Probe p;
            p.state     = rows[i].at("state").as_str() + " " + rows[j].at("state").as_str();
            p.questions = rows[i].at("questions");
            for (const auto& kv : rows[i].at("labels").obj)
                p.labels[kv.first] = kv.second.as_str();
            p.taught    = false;
            out.push_back(p);
        }
    }
    return out;
}

// Eval probes from labelled rows: the row's own state, questions and labels.
inline std::vector<Probe> eval_probes(const std::vector<sfx::JV>& rows) {
    std::vector<Probe> out;
    for (const auto& r : rows) {
        if (!r.has("state") || !r.has("questions") || !r.has("labels")) continue;
        if (!r.at("questions").is_obj() || !r.at("labels").is_obj()) continue;
        Probe p;
        p.state     = r.at("state").as_str();
        p.questions = r.at("questions");
        for (const auto& kv : r.at("labels").obj)
            p.labels[kv.first] = kv.second.as_str();
        p.taught = true;
        out.push_back(p);
    }
    return out;
}

// Does a stored label (level text, or level INDEX for array criteria) match
// the argmax label the answer produced?
inline bool label_matches(const sfx::JV& q, const std::string& stored, const std::string& argmax) {
    if (stored == argmax) return true;
    const sfx::JV& crit = q.has("criteria") ? q.at("criteria") : sfx::JV(sfx::JVObj{});
    if (crit.is_arr()) {
        long idx = std::strtol(stored.c_str(), nullptr, 10);
        if (idx >= 0 && static_cast<std::size_t>(idx) < crit.arr.size())
            return crit.arr[static_cast<std::size_t>(idx)].as_str() == argmax;
    }
    return false;
}

struct Bin { long n = 0; long correct = 0; double conf_sum = 0.0; double ece_conf_sum = 0.0; };

// One full pass of decisions over a probe set, accumulating the Jev axes.
struct PassResult {
    struct QType {
        long n = 0, correct = 0, close_calls = 0;
        double conf_sum = 0.0, margin_sum = 0.0;
        std::map<int, Bin> ece_bins;
    };
    QType choice, score;
    long noul_tp = 0, noul_fp = 0, noul_tn = 0, noul_fn = 0;  // pred true = HOLD
    long deferred = 0, noul_n = 0;
    std::vector<double> latency_us;

    static int ece_bin(float conf) {
        int b = static_cast<int>(conf * 10.0f);
        return b < 0 ? 0 : (b > 9 ? 9 : b);
    }
    static double ece(const std::map<int, Bin>& bins) {
        long total = 0; double e = 0.0;
        for (const auto& kv : bins) total += kv.second.n;
        if (total == 0) return 0.0;
        for (const auto& kv : bins) {
            const Bin& b = kv.second;
            if (b.n == 0) continue;
            const double acc = static_cast<double>(b.correct) / static_cast<double>(b.n);
            // v2.1: ECE uses the MAX-PROBABILITY confidence (the signal
            // temperature scaling acts on), not the entropy confidence. The
            // entropy confidence stays in conf_sum for the conf-gap fields.
            const double cf  = b.ece_conf_sum / static_cast<double>(b.n);
            e += (static_cast<double>(b.n) / static_cast<double>(total)) * std::fabs(acc - cf);
        }
        return e;
    }
};

// The decision signature of one probe: qid -> (argmax, confidence), used by
// the determinism check and (in gate.hpp) by the no-regression replay.
inline std::map<std::string, std::pair<std::string, float>>
decision_signature(Engine& eng, const Probe& p) {
    std::map<std::string, std::pair<std::string, float>> sig;
    syfox::Usage u;
    auto answers = eng.decide(p.state, p.questions, u);
    std::size_t ai = 0;
    for (const auto& qkv : p.questions.obj) {
        if (ai >= answers.size()) break;
        const syfox::Answer& a = answers[ai++];
        if (a.deferred) { sig[qkv.first] = {"", -1.0f}; continue; }
        std::string amax;
        float top = -1.0f;
        for (const auto& kv : a.probabilities)
            if (kv.second > top) { top = kv.second; amax = kv.first; }
        if (a.type == "noul") amax = a.probability >= 0.5f ? "true" : "false";
        sig[qkv.first] = {amax, a.confidence};
    }
    return sig;
}

inline PassResult run_pass(Engine& eng, const std::vector<Probe>& probes, int latency_reps) {
    PassResult pr;
    for (const auto& p : probes) {
        // latency: time decide() calls for a stable p50/p95
        for (int r = 0; r < latency_reps; ++r) {
            syfox::Usage u;
            const auto t0 = std::chrono::steady_clock::now();
            auto answers = eng.decide(p.state, p.questions, u);
            const auto t1 = std::chrono::steady_clock::now();
            pr.latency_us.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
            if (r == 0)
                for (const auto& a : answers) if (a.deferred) ++pr.deferred;
        }
        // correctness pass (untimed, separate — keeps the semantics clean)
        syfox::Usage u;
        auto answers = eng.decide(p.state, p.questions, u);
        std::size_t ai = 0;
        for (const auto& qkv : p.questions.obj) {
            const sfx::JV& q = qkv.second;
            if (ai >= answers.size()) break;
            const syfox::Answer& a = answers[ai++];
            const std::string type = q.at("type").as_str();
            const std::string stored =
                p.labels.count(qkv.first) ? p.labels.at(qkv.first) : "";
            if (a.deferred) continue;
            if (type == "choice" || type == "score") {
                PassResult::QType& agg = (type == "choice") ? pr.choice : pr.score;
                if (a.probabilities.empty()) continue;
                std::string amax; float top = -1.0f, second = -1.0f;
                for (const auto& kv : a.probabilities) {
                    if (kv.second > top) { second = top; top = kv.second; amax = kv.first; }
                    else if (kv.second > second) second = kv.second;
                }
                const bool correct = p.taught && label_matches(q, stored, amax);
                ++agg.n;
                agg.correct += correct ? 1 : 0;
                agg.conf_sum += a.confidence;
                const float margin = std::max(0.0f, top - std::max(second, 0.0f));
                agg.margin_sum += margin;
                agg.close_calls += (margin < 0.10f) ? 1 : 0;
                Bin& b = agg.ece_bins[PassResult::ece_bin(top)];
                b.n += 1; b.correct += correct ? 1 : 0;
                b.conf_sum += a.confidence; b.ece_conf_sum += top;
            } else if (type == "noul") {
                ++pr.noul_n;
                const bool pred_true  = a.probability >= 0.5f;
                const bool true_label = p.taught && stored == "true";
                if (pred_true && true_label)        ++pr.noul_tp;
                else if (pred_true && !true_label)  ++pr.noul_fp;
                else if (!pred_true && !true_label) ++pr.noul_tn;
                else                                ++pr.noul_fn;
            }
        }
    }
    return pr;
}

struct BenchReport {
    std::string model;
    std::string eval_source;        // labelled with the split it came from
    long rows = 0, probes = 0, mixed_probes = 0, ood_probes = 0;

    // routing (Jev: 4-workflow accuracy 67.8%)
    double choice_accuracy = 0, score_accuracy = 0;
    long   choice_n = 0, score_n = 0;
    double choice_ece = 0, score_ece = 0;
    double conf_correct = 0, conf_wrong = 0;      // choice+score, confidence gap
    double mean_margin = 0; long close_calls = 0, margin_n = 0;

    // honesty (Jev: "never hallucinates", schema-bound)
    double ood_defer_rate = 0; long ood_n = 0;

    // guardrail (Jev: pi-warden 88% hold precision)
    long g_tp = 0, g_fp = 0, g_tn = 0, g_fn = 0;
    double hold_precision = -1.0, hold_recall = -1.0;   // -1 = undefined

    // latency (Jev: 70-500 ms)
    double lat_p50_us = 0, lat_p95_us = 0, lat_mean_us = 0;

    // determinism (Jev: non-autoregressive one-pass)
    bool deterministic = true;

    sfx::JV to_json() const {
        const auto r2 = [](double v) { return std::round(v * 1000.0) / 1000.0; };
        sfx::JVObj routing;
        routing["choice_accuracy"] = r2(choice_accuracy);
        routing["choice_n"] = static_cast<double>(choice_n);
        routing["score_accuracy"] = r2(score_accuracy);
        routing["score_n"] = static_cast<double>(score_n);
        routing["mean_margin"] = r2(mean_margin);
        routing["close_calls"] = static_cast<double>(close_calls);
        routing["margin_n"] = static_cast<double>(margin_n);

        sfx::JVObj calib;
        calib["choice_ece"] = r2(choice_ece);
        calib["score_ece"] = r2(score_ece);
        calib["mean_conf_when_correct"] = r2(conf_correct);
        calib["mean_conf_when_wrong"] = r2(conf_wrong);

        sfx::JVObj honesty;
        honesty["ood_defer_rate"] = r2(ood_defer_rate);
        honesty["ood_probes"] = static_cast<double>(ood_n);

        sfx::JVObj guard;
        guard["tp"] = static_cast<double>(g_tp);
        guard["fp"] = static_cast<double>(g_fp);
        guard["tn"] = static_cast<double>(g_tn);
        guard["fn"] = static_cast<double>(g_fn);
        if (hold_precision >= 0) guard["hold_precision"] = r2(hold_precision);
        if (hold_recall >= 0)    guard["hold_recall"]    = r2(hold_recall);

        sfx::JVObj lat;
        lat["p50"] = r2(lat_p50_us);
        lat["p95"] = r2(lat_p95_us);
        lat["mean"] = r2(lat_mean_us);

        sfx::JVObj jev;
        jev["published"] = sfx::JV(true);
        jev["workflow_accuracy"] = 0.678;
        jev["latency_ms"] = sfx::JV("[70, 500]");
        jev["guardrail_hold_precision"] = 0.88;
        jev["source"] = sfx::JV("typesafe.ai launch + press coverage, Sept 2026; axes shared, not numbers");

        sfx::JVObj o;
        o["suite"] = sfx::JV("syfox-jev-parity-v0");
        o["model"] = sfx::JV(model);
        o["eval_source"] = sfx::JV(eval_source);
        o["rows"] = static_cast<double>(rows);
        o["probes"] = static_cast<double>(probes);
        o["mixed_probes"] = static_cast<double>(mixed_probes);
        o["ood_probes"] = static_cast<double>(ood_probes);
        o["routing"] = sfx::JV(routing);
        o["calibration"] = sfx::JV(calib);
        o["honesty"] = sfx::JV(honesty);
        o["guardrail"] = sfx::JV(guard);
        o["latency_us"] = sfx::JV(lat);
        o["deterministic"] = sfx::JV(deterministic);
        o["jev_reference"] = sfx::JV(jev);
        return sfx::JV(o);
    }
};

inline BenchReport run(Engine& eng, const std::vector<sfx::JV>& eval_rows,
                       const std::string& eval_source, const BenchConfig& bc,
                       const std::string& model_name) {
    BenchReport rep;
    rep.model = model_name;
    rep.eval_source = eval_source;
    rep.rows = static_cast<long>(eval_rows.size());

    std::vector<Probe> probes = eval_probes(eval_rows);
    const std::vector<Probe> mixed = closecall_probes(eval_rows, bc.closecall_probes);
    rep.mixed_probes = static_cast<long>(mixed.size());
    probes.insert(probes.end(), mixed.begin(), mixed.end());
    rep.probes = static_cast<long>(probes.size());

    // ---- accuracy / calibration / guardrail: LABELLED rows only -------------
    // Close-call probes carry no trustworthy labels (taught=false) — counting
    // them in the accuracy/ECE denominators would report fake errors. They
    // get their own pass for the margin distribution, which is the quantity
    // they exist for.
    const std::vector<Probe> taught_probes = eval_probes(eval_rows);
    const PassResult pr  = run_pass(eng, taught_probes, bc.latency_reps);
    const PassResult pm  = mixed.empty()
        ? PassResult{} : run_pass(eng, mixed, bc.latency_reps);
    if (pr.choice.n > 0) rep.choice_accuracy = static_cast<double>(pr.choice.correct) / pr.choice.n;
    if (pr.score.n  > 0) rep.score_accuracy  = static_cast<double>(pr.score.correct)  / pr.score.n;
    rep.choice_n = pr.choice.n; rep.score_n = pr.score.n;
    rep.choice_ece = PassResult::ece(pr.choice.ece_bins);
    rep.score_ece  = PassResult::ece(pr.score.ece_bins);
    // margins/close-calls from the mixed probes when we have them (that is
    // their home turf), else from the taught rows
    {
        const long   margin_n_mixed  = pm.choice.n + pm.score.n;
        const long   close_mixed     = pm.choice.close_calls + pm.score.close_calls;
        const double sum_mixed       = pm.choice.margin_sum + pm.score.margin_sum;
        const long   margin_n_taught = pr.choice.n + pr.score.n;
        const long   close_taught    = pr.choice.close_calls + pr.score.close_calls;
        const double sum_taught      = pr.choice.margin_sum + pr.score.margin_sum;
        rep.margin_n    = margin_n_mixed > 0 ? margin_n_mixed : margin_n_taught;
        rep.close_calls = margin_n_mixed > 0 ? close_mixed    : close_taught;
        const double sum = margin_n_mixed > 0 ? sum_mixed : sum_taught;
        if (rep.margin_n > 0) rep.mean_margin = sum / rep.margin_n;
    }
    {
        // confidence gap over choice+score: mean conf on correct vs wrong.
        // bins hold SUMS, so the per-bin mean must be divided out before
        // weighting by the correct/wrong counts.
        double c_ok = 0, c_bad = 0; long n_ok = 0, n_bad = 0;
        for (const PassResult::QType* agg : {&pr.choice, &pr.score}) {
            for (const auto& kv : agg->ece_bins) {
                if (kv.second.n == 0) continue;
                const double mean_conf = kv.second.conf_sum / kv.second.n;
                c_ok  += mean_conf * kv.second.correct;
                c_bad += mean_conf * (kv.second.n - kv.second.correct);
                n_ok  += kv.second.correct;
                n_bad += kv.second.n - kv.second.correct;
            }
        }
        if (n_ok  > 0) rep.conf_correct = c_ok / n_ok;
        if (n_bad > 0) rep.conf_wrong   = c_bad / n_bad;
    }
    rep.g_tp = pr.noul_tp; rep.g_fp = pr.noul_fp;
    rep.g_tn = pr.noul_tn; rep.g_fn = pr.noul_fn;
    if (pr.noul_tp + pr.noul_fp > 0)
        rep.hold_precision = static_cast<double>(pr.noul_tp) / (pr.noul_tp + pr.noul_fp);
    if (pr.noul_tp + pr.noul_fn > 0)
        rep.hold_recall = static_cast<double>(pr.noul_tp) / (pr.noul_tp + pr.noul_fn);

    // ---- honesty: OOD probes must defer (honest silence) ---------------------
    {
        std::vector<Probe> oods;
        for (const auto& st : ood_states(eng.substrate(), bc.ood_probes)) {
            // questions template: reuse the first eval row's questions if any
            sfx::JV q = eval_rows.empty() ? sfx::JV(sfx::JVObj{})
                        : (eval_rows[0].has("questions") ? eval_rows[0].at("questions")
                                                         : sfx::JV(sfx::JVObj{}));
            Probe p; p.state = st; p.questions = q; p.taught = false;
            oods.push_back(p);
        }
        rep.ood_n = static_cast<long>(oods.size());
        long deferred = 0, scored = 0;
        for (const auto& p : oods) {
            syfox::Usage u;
            auto answers = eng.decide(p.state, p.questions, u);
            if (answers.empty()) continue;             // no questions asked: vacuous
            ++scored;
            bool all_defer = true;
            for (const auto& a : answers) if (!a.deferred) all_defer = false;
            deferred += all_defer ? 1 : 0;
        }
        rep.ood_probes = scored;
        rep.ood_defer_rate = scored ? static_cast<double>(deferred) / scored : 0.0;
    }

    // ---- determinism: full eval replay, byte-identical signatures ------------
    for (int run = 1; run < bc.determinism_runs && rep.deterministic; ++run) {
        std::vector<Probe> replay = eval_probes(eval_rows);
        replay.insert(replay.end(), mixed.begin(), mixed.end());
        if (replay.size() != probes.size()) { rep.deterministic = false; break; }
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const auto s0 = decision_signature(eng, probes[i]);
            const auto s1 = decision_signature(eng, replay[i]);
            if (s0 != s1) { rep.deterministic = false; break; }
        }
    }

    // ---- latency percentiles -------------------------------------------------
    if (!pr.latency_us.empty()) {
        std::vector<double> lat = pr.latency_us;
        std::sort(lat.begin(), lat.end());
        const auto pick = [&](double q) {
            std::size_t i = static_cast<std::size_t>(q * (lat.size() - 1));
            return lat[i];
        };
        rep.lat_p50_us = pick(0.50);
        rep.lat_p95_us = pick(0.95);
        double s = 0.0; for (double v : lat) s += v;
        rep.lat_mean_us = s / static_cast<double>(lat.size());
    }
    return rep;
}

// ---------------------------------------------------------------------------
// v2.1 — selective prediction: the coverage-vs-accuracy curve.
//
// The headline metric of an honest eval is NOT top-1 accuracy (that is the
// coverage=100% corner of this curve). We sweep a confidence threshold over
// the TOP-PROBABILITY signal and report, for each threshold, what share of
// questions the model still answers (coverage) and how often it is right
// within that share (accuracy-within-coverage). Engine-level honest silence
// (unknown vocabulary) counts as covered=false at every threshold.
//
// Choice + score questions only: these carry the routing semantics the curve
// exists for. Labelled rows only.
// ---------------------------------------------------------------------------
struct CurvePoint {
    double threshold = 0;      // emit iff top-prob >= threshold
    long   emitted   = 0;
    long   correct   = 0;
    double coverage  = 0;      // emitted / total
    double accuracy  = 0;      // correct / emitted
};

struct OperatingPoint {
    double target    = 0;      // requested minimum coverage (0.5 / 0.7 / 0.9)
    bool   feasible  = false;  // some threshold reaches the target coverage
    double threshold = 0;      // the LARGEST threshold still covering the target
    double coverage  = 0;
    double accuracy  = 0;
};

struct CoverageReport {
    long total = 0;                       // labelled choice+score questions
    std::vector<CurvePoint> points;
    OperatingPoint op50, op70, op90;      // spec-required operating points

    sfx::JV to_json() const {
        const auto r4 = [](double v) { return std::round(v * 10000.0) / 10000.0; };
        sfx::JVArr pts;
        for (const auto& p : points)
            pts.push_back(sfx::JV(sfx::JVObj{
                {"threshold", r4(p.threshold)},
                {"emitted", static_cast<double>(p.emitted)},
                {"correct", static_cast<double>(p.correct)},
                {"coverage", r4(p.coverage)},
                {"accuracy", r4(p.accuracy)}}));
        auto opj = [](const OperatingPoint& o) {
            return sfx::JV(sfx::JVObj{
                {"feasible", sfx::JV(o.feasible)},
                {"threshold", std::round(o.threshold * 10000.0) / 10000.0},
                {"coverage", std::round(o.coverage * 10000.0) / 10000.0},
                {"accuracy", std::round(o.accuracy * 10000.0) / 10000.0}});
        };
        return sfx::JV(sfx::JVObj{
            {"total", static_cast<double>(total)},
            {"points", sfx::JV(pts)},
            {"at_50_coverage", opj(op50)},
            {"at_70_coverage", opj(op70)},
            {"at_90_coverage", opj(op90)}});
    }
};

inline CoverageReport coverage_curve(Engine& eng, const std::vector<sfx::JV>& eval_rows,
                                     int steps = 20) {
    CoverageReport cr;
    struct Sample { float conf; bool correct; bool engine_deferred; };
    std::vector<Sample> samples;

    for (const auto& p : eval_probes(eval_rows)) {
        syfox::Usage u;
        auto answers = eng.decide(p.state, p.questions, u);
        std::size_t ai = 0;
        for (const auto& qkv : p.questions.obj) {
            if (ai >= answers.size()) break;
            const syfox::Answer& a = answers[ai++];
            const std::string type = qkv.second.at("type").as_str();
            if (type != "choice" && type != "score") continue;
            Sample s; s.engine_deferred = a.deferred; s.conf = 0.0f; s.correct = false;
            if (!a.deferred && !a.probabilities.empty()) {
                float top = -1.0f;
                std::string amax;
                for (const auto& kv : a.probabilities)
                    if (kv.second > top) { top = kv.second; amax = kv.first; }
                s.conf = std::max(0.0f, top);
                const std::string stored =
                    p.labels.count(qkv.first) ? p.labels.at(qkv.first) : "";
                s.correct = label_matches(qkv.second, stored, amax);
            }
            samples.push_back(s);
        }
    }
    cr.total = static_cast<long>(samples.size());
    if (cr.total == 0) return cr;

    for (int i = 0; i <= steps; ++i) {
        const double tau = static_cast<double>(i) / static_cast<double>(steps);
        CurvePoint pt; pt.threshold = tau;
        for (const auto& s : samples) {
            if (s.engine_deferred) continue;                       // honest silence: never emitted
            if (tau > 0.0 && static_cast<double>(s.conf) < tau) continue;
            ++pt.emitted; pt.correct += s.correct ? 1 : 0;
        }
        pt.coverage = static_cast<double>(pt.emitted) / static_cast<double>(cr.total);
        pt.accuracy = pt.emitted > 0
            ? static_cast<double>(pt.correct) / static_cast<double>(pt.emitted) : 0.0;
        cr.points.push_back(pt);
    }

    // Operating point for a target coverage: the LARGEST threshold whose
    // coverage still meets the target (answer at least that share, and be as
    // accurate as the curve allows at that share).
    auto pick = [&](double target) {
        OperatingPoint op; op.target = target;
        for (auto it = cr.points.rbegin(); it != cr.points.rend(); ++it) {
            if (it->coverage >= target) {
                op.feasible = true;
                op.threshold = it->threshold;
                op.coverage = it->coverage;
                op.accuracy = it->accuracy;
                return op;
            }
        }
        // infeasible: report the widest-net point (threshold 0) for context
        const CurvePoint& p0 = cr.points.front();
        op.threshold = p0.threshold; op.coverage = p0.coverage; op.accuracy = p0.accuracy;
        return op;
    };
    cr.op50 = pick(0.50); cr.op70 = pick(0.70); cr.op90 = pick(0.90);
    return cr;
}

// ASCII rendering of the coverage curve: accuracy (y, 0-100%) against
// coverage (x, 0-100%), one mark per sweep point. Thresholds fall where
// they fall; guides mark the 50/70/90 operating points.
inline std::string coverage_plot(const CoverageReport& cr, int width = 60, int height = 20) {
    std::vector<std::string> grid(height, std::string(width, ' '));
    for (const auto& p : cr.points) {
        if (p.emitted == 0 && p.threshold > 0.0) continue;
        const int x = std::min(width - 1,
            static_cast<int>(std::round(p.coverage * (width - 1))));
        const int y = std::min(height - 1,
            static_cast<int>(std::round(p.accuracy * (height - 1))));
        grid[height - 1 - y][x] = '*';       // later (higher-threshold) points overwrite
    }
    auto mark = [&](const OperatingPoint& op, char c) {
        if (!op.feasible) return;
        const int x = std::min(width - 1,
            static_cast<int>(std::round(op.coverage * (width - 1))));
        const int y = std::min(height - 1,
            static_cast<int>(std::round(op.accuracy * (height - 1))));
        grid[height - 1 - y][x] = c;         // operating points stay visible
    };
    mark(cr.op50, '5'); mark(cr.op70, '7'); mark(cr.op90, '9');

    std::ostringstream out;
    out << "  accuracy % |";
    for (int x = 0; x < width; ++x) out << '-';
    out << "| coverage %\n";
    for (int r = 0; r < height; ++r) {
        const int acc = 100 - static_cast<int>(std::lround(100.0 * r / (height - 1)));
        char label[16];
        std::snprintf(label, sizeof(label), "%4d", acc);
        out << "  " << label << "    |" << grid[r] << "|\n";
    }
    out << "  ";
    for (int i = 0; i < 11; ++i) out << '+';
    for (int i = 0; i < width - 11; ++i) out << '-';
    out << "\n  ";
    out << "0%        50%       100%   (* sweep point; 5/7/9 = coverage>=50/70/90% op)\n";
    return out.str();
}

// ---------------------------------------------------------------------------
// v2.1 — aggregate labelled accuracy over a row set (choice + score + noul).
// Used by the derive gate to report held-out accuracy before/after, and by
// any caller that wants one number over labelled rows. Deferred answers are
// neither correct nor wrong (honest silence) — they leave the denominator.
// ---------------------------------------------------------------------------
struct AccSummary {
    long n = 0, correct = 0;
    double accuracy = 0.0;
    long choice_n = 0, choice_correct = 0;
    long score_n = 0, score_correct = 0;
    long noul_n = 0, noul_correct = 0;
};

inline AccSummary labelled_accuracy(Engine& eng, const std::vector<sfx::JV>& rows) {
    const PassResult pr = run_pass(eng, eval_probes(rows), 1);
    AccSummary s;
    s.choice_n = pr.choice.n;          s.choice_correct = pr.choice.correct;
    s.score_n = pr.score.n;            s.score_correct = pr.score.correct;
    s.noul_n = pr.noul_n;              s.noul_correct = pr.noul_tp + pr.noul_tn;
    s.n = s.choice_n + s.score_n + s.noul_n;
    s.correct = s.choice_correct + s.score_correct + s.noul_correct;
    if (s.n > 0) s.accuracy = static_cast<double>(s.correct) / static_cast<double>(s.n);
    return s;
}

// ---------------------------------------------------------------------------
// v3 Milestone 4 — adversarial / OOD evaluation suite
// ---------------------------------------------------------------------------
// Nine deterministic stress families over labelled eval rows (the hidden test
// in the M1 discipline). The engine under test is NEVER mutated: all families
// run read-only decisions, except the conflicting-lessons sub-suite, which
// works on a THROWAWAY copy of the engine and reports its blast radius.
//
// Metrics per family (the four the milestone asks for, made concrete):
//   accuracy          labelled-choice accuracy under the transform
//   defer_rate        fraction of choice answers deferred (honest silence)
//   conf_when_wrong   mean confidence of WRONG non-deferred answers
//   false_conf_rate   wrong AND confidence >= 0.5 AND not deferred / answered
//
// Honesty notes baked into the report:
//   * synonym-swap paraphrases fold to the SAME token stream after
//     normalization (Porter + synonym table), so they are not adversarial
//     here — the paraphrase axis is covered by reorder + padding, which DO
//     change the token stream.
//   * score/noul questions are not scored under label-preserving transforms:
//     a transformed state may legitimately move a frustration score or flip a
//     guard valence, so scoring them would manufacture errors. Choice
//     (routing) is intent-level and label-stable; that is what is scored.
//   * every transform is a pure function of the row index: the suite replays
//     bit-identically.
// ---------------------------------------------------------------------------
struct AdvFamilyResult {
    std::string name, note;
    long n = 0, answered = 0, correct = 0, deferred = 0, high_conf_wrong = 0;
    double conf_wrong_sum = 0.0;

    double accuracy() const {
        return answered ? static_cast<double>(correct) / static_cast<double>(answered) : 0.0;
    }
    double defer_rate() const {
        return n ? static_cast<double>(deferred) / static_cast<double>(n) : 0.0;
    }
    double conf_when_wrong() const {
        long wrong = answered - correct;
        return wrong > 0 ? conf_wrong_sum / static_cast<double>(wrong) : 0.0;
    }
    double false_conf_rate() const {
        return answered ? static_cast<double>(high_conf_wrong) / static_cast<double>(answered) : 0.0;
    }
    sfx::JV to_json() const {
        return sfx::JV(sfx::JVObj{
            {"name", sfx::JV(name)},
            {"n", static_cast<double>(n)},
            {"answered", static_cast<double>(answered)},
            {"accuracy", std::round(accuracy() * 10000.0) / 10000.0},
            {"defer_rate", std::round(defer_rate() * 10000.0) / 10000.0},
            {"conf_when_wrong", std::round(conf_when_wrong() * 10000.0) / 10000.0},
            {"false_conf_rate", std::round(false_conf_rate() * 10000.0) / 10000.0},
            {"note", sfx::JV(note)}});
    }
};

// whitespace tokenizer for the state transforms (surface form; decide()
// re-normalizes internally, so transforming surface text is enough)
inline std::vector<std::string> adv_split(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string w;
    while (in >> w) out.push_back(w);
    return out;
}
inline std::string adv_join(const std::vector<std::string>& ws) {
    std::string out;
    for (std::size_t i = 0; i < ws.size(); ++i) { if (i) out += " "; out += ws[i]; }
    return out;
}

// label -> outcome text, identical to the CLI's learn-side resolution so the
// conflicting-lessons sub-suite teaches in EXACTLY the production format.
inline std::string adv_outcome_text(const sfx::JV& q, const std::string& label) {
    if (!q.has("criteria")) return label;
    const sfx::JV& crit = q.at("criteria");
    if (crit.is_obj() && crit.has(label)) return label + " " + crit.at(label).as_str();
    if (crit.is_arr()) {
        for (const auto& v : crit.arr)
            if (v.as_str() == label) return label;
        long idx = std::strtol(label.c_str(), nullptr, 10);
        if (idx >= 0 && idx < static_cast<long>(crit.arr.size()))
            return crit.arr[static_cast<std::size_t>(idx)].as_str();
    }
    return label;
}

// the first choice-type question of a row (qid + question object)
inline bool adv_first_choice(const sfx::JV& row, std::string& qid, sfx::JV& qout) {
    if (!row.has("questions") || !row.at("questions").is_obj()) return false;
    for (const auto& qkv : row.at("questions").obj) {
        if (qkv.second.has("type") && qkv.second.at("type").as_str() == "choice") {
            qid = qkv.first; qout = qkv.second; return true;
        }
    }
    return false;
}

// foreign vocabulary pools for near-miss / cross-domain families. Used only
// when the caller does not pass --mix; pools are per-domain content words.
inline std::vector<std::string> adv_builtin_pool(const std::string& model_name) {
    static const char* kGame[]   = {"zombie", "dungeon", "quest", "sword", "shelter",
                                    "respawn", "boss", "inventory", "health", "village",
                                    "attack", "defend", "resources", "night", "wall"};
    static const char* kTicket[] = {"invoice", "refund", "login", "password", "screen",
                                    "keyboard", "subscription", "shipping", "warranty",
                                    "crash", "email", "account", "payment", "update", "server"};
    static const char* kGuard[]  = {"rm", "sudo", "drop", "table", "delete", "wipe",
                                    "chmod", "root", "exec", "payload", "script",
                                    "database", "credential", "token", "deploy"};
    std::vector<std::string> out;
    auto add = [&out](const char** arr, int k) {
        for (int i = 0; i < k; ++i) out.push_back(arr[i]);
    };
    const bool has_ticket = model_name.find("ticket") != std::string::npos;
    const bool has_game   = model_name.find("game")   != std::string::npos;
    const bool has_guard  = model_name.find("guard")  != std::string::npos;
    if (has_ticket) { add(kGame, 15); add(kGuard, 15); }
    else if (has_game) { add(kTicket, 15); add(kGuard, 15); }
    else if (has_guard) { add(kTicket, 15); add(kGame, 15); }
    else { add(kGame, 15); add(kTicket, 15); }             // unknown domain: everything foreign
    return out;
}

// content words from --mix rows (the caller's own cross-domain corpus)
inline std::vector<std::string> adv_pool_from_rows(const std::vector<sfx::JV>& mix_rows) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& r : mix_rows) {
        if (!r.has("state")) continue;
        for (const auto& t : si::norm::normalize(r.at("state").as_str())) {
            if (t.size() < 4 || seen.count(t)) continue;
            seen.insert(t);
            out.push_back(t);
            if (out.size() >= 400) return out;
        }
    }
    return out;
}

struct ConflictResult {
    long taught = 0, detected = 0, contested_flagged = 0, checked = 0;
    double untouched_acc_before = 0.0, untouched_acc_control = 0.0, untouched_acc_after = 0.0;
    bool deterministic_after = true;
    sfx::JV to_json() const {
        return sfx::JV(sfx::JVObj{
            {"contradictions_taught", static_cast<double>(taught)},
            {"conflicts_detected", static_cast<double>(detected)},
            {"contested_flagged", static_cast<double>(contested_flagged)},
            {"states_checked", static_cast<double>(checked)},
            {"untouched_acc_before", std::round(untouched_acc_before * 10000.0) / 10000.0},
            {"untouched_acc_control", std::round(untouched_acc_control * 10000.0) / 10000.0},
            {"untouched_acc_after", std::round(untouched_acc_after * 10000.0) / 10000.0},
            {"deterministic_after", sfx::JV(deterministic_after)},
            {"silent_override", sfx::JV(detected != taught || contested_flagged != checked)},
            {"note", sfx::JV("control arm = same states re-taught with their GOLD outcomes "
                             "(same teach-event count, no dispute): isolates the forgetting-law "
                             "cost every teach event pays from contradiction-specific damage. "
                             "Accuracy basis: all labelled questions (choice+score+noul).")}});
    }
};

struct AdvReport {
    std::vector<AdvFamilyResult> families;
    ConflictResult conflicts;
    sfx::JV to_json() const {
        sfx::JVArr f;
        for (const auto& fam : families) f.push_back(fam.to_json());
        // worst false-confidence family = the headline honesty risk
        std::string worst; double worst_fc = -1.0;
        for (const auto& fam : families) {
            if (fam.answered < 10) continue;               // too small to judge
            if (fam.false_conf_rate() > worst_fc) { worst_fc = fam.false_conf_rate(); worst = fam.name; }
        }
        return sfx::JV(sfx::JVObj{
            {"families", sfx::JV(f)},
            {"conflicting_lessons", conflicts.to_json()},
            {"worst_false_conf_family", sfx::JV(worst)},
            {"worst_false_conf_rate", std::round(worst_fc * 10000.0) / 10000.0},
            {"note", sfx::JV("all transforms are pure functions of the row index; "
                             "the suite replays bit-identically and never mutates the "
                             "model under test (conflicts run on a throwaway copy)")}});
    }
};

// score ONE transformed probe set on choice questions only, accumulating the
// four M4 metrics. Deferred answers count toward defer_rate, leave accuracy.
inline AdvFamilyResult adv_score_family(Engine& eng, const std::vector<Probe>& probes,
                                        const std::string& name, const std::string& note) {
    AdvFamilyResult fam; fam.name = name; fam.note = note;
    for (const auto& p : probes) {
        syfox::Usage u;
        auto answers = eng.decide(p.state, p.questions, u);
        std::size_t ai = 0;
        for (const auto& qkv : p.questions.obj) {
            const sfx::JV& q = qkv.second;
            if (ai >= answers.size()) break;
            const syfox::Answer& a = answers[ai++];
            if (q.at("type").as_str() != "choice") continue;   // choice-only scoring
            ++fam.n;
            if (a.deferred) { ++fam.deferred; continue; }
            if (a.probabilities.empty()) { ++fam.deferred; continue; }
            std::string amax; float top = -1.0f;
            for (const auto& kv : a.probabilities)
                if (kv.second > top) { top = kv.second; amax = kv.first; }
            const std::string stored =
                p.labels.count(qkv.first) ? p.labels.at(qkv.first) : "";
            ++fam.answered;
            const bool correct = p.taught && label_matches(q, stored, amax);
            fam.correct += correct ? 1 : 0;
            if (!correct) {
                fam.conf_wrong_sum += a.confidence;
                if (a.confidence >= 0.5f) ++fam.high_conf_wrong;
            }
        }
    }
    return fam;
}

inline std::vector<Probe> adv_transform_rows(const std::vector<sfx::JV>& rows,
                                             std::string (*fn)(const std::string&, std::size_t)) {
    std::vector<Probe> out;
    std::size_t i = 0;
    for (const auto& r : rows) {
        if (!r.has("state") || !r.has("questions") || !r.has("labels")) { ++i; continue; }
        Probe p;
        p.state     = fn(r.at("state").as_str(), i);
        p.questions = r.at("questions");
        for (const auto& kv : r.at("labels").obj) p.labels[kv.first] = kv.second.as_str();
        p.taught    = true;
        out.push_back(p);
        ++i;
    }
    return out;
}

// -- the transform families (pure functions of state + row index) ------------
inline std::string adv_reorder(const std::string& s, std::size_t) {
    std::vector<std::string> ws = adv_split(s);
    if (ws.size() < 3) return s;
    const std::size_t h = ws.size() / 2;
    std::vector<std::string> out(ws.begin() + static_cast<long>(h), ws.end());
    out.insert(out.end(), ws.begin(), ws.begin() + static_cast<long>(h));
    return adv_join(out);
}
inline std::string adv_padding(const std::string& s, std::size_t i) {
    static const char* kPad[] = {
        "Hope you are doing well today.",
        "Also the weather is quite nice this week.",
        "This is an unrelated aside about office chairs.",
        "By the way the meeting got rescheduled again."};
    return s + " " + kPad[i % 4];
}
inline std::string adv_typo(const std::string& s, std::size_t) {
    return corrupt_state(s, 15.0f);
}
inline std::string adv_intensifiers(const std::string& s, std::size_t) {
    std::vector<std::string> ws = adv_split(s);
    std::vector<std::string> out;
    long doubled = 0;
    for (const auto& w : ws) {
        out.push_back(w);
        if (w.size() > 3 && doubled < 3) { out.push_back(w); ++doubled; }
    }
    out.push_back("URGENT"); out.push_back("URGENT");
    return adv_join(out);
}
inline std::string adv_self_contradiction(const std::string& s, std::size_t) {
    return s + " Actually never mind, everything is fine now.";
}
inline std::string adv_negation(const std::string& s, std::size_t) {
    std::vector<std::string> ws = adv_split(s);
    for (std::size_t j = 0; j < ws.size(); ++j)
        if (ws[j].size() > 3) { ws.insert(ws.begin() + static_cast<long>(j), "not"); break; }
    return adv_join(ws);
}
inline std::string adv_double_negation(const std::string& s, std::size_t) {
    return s + " It is not that this was never handled.";
}
// unknown vocabulary: rows re-generated from the OOD nonsense pool (the same
// machinery as the honesty probes, filtered against THIS model's vocabulary)
inline std::vector<Probe> adv_unknown_probes(Engine& eng, const std::vector<sfx::JV>& rows) {
    std::vector<Probe> out;
    const auto nonsense = ood_states(eng.substrate(), 24);
    if (nonsense.empty()) return out;
    for (const auto& r : rows) {
        if (!r.has("questions") || !r.has("labels")) continue;
        Probe p;
        p.state     = nonsense[out.size() % nonsense.size()];
        p.questions = r.at("questions");
        for (const auto& kv : r.at("labels").obj) p.labels[kv.first] = kv.second.as_str();
        p.taught    = false;               // nonsense has no gold answer: defer-only
        out.push_back(p);
        if (out.size() >= 48) break;
    }
    return out;
}
// near-miss: foreign words that SHARE trigrams with taught vocabulary — the
// dangerous OOD class, because the traction gate may bridge them
inline std::vector<Probe> adv_nearmiss_probes(Engine& eng, const std::vector<sfx::JV>& rows,
                                              const std::vector<std::string>& pool) {
    std::vector<Probe> out;
    if (pool.empty()) return out;
    std::vector<std::string> entangled;
    for (const auto& w : pool) {
        for (const auto& g : si::norm::expand_ngrams({w}))
            if (eng.substrate().has(g)) { entangled.push_back(w); break; }
        if (entangled.size() >= 32) break;
    }
    if (entangled.empty()) return out;
    for (const auto& r : rows) {
        if (!r.has("state") || !r.has("questions") || !r.has("labels")) continue;
        std::vector<std::string> ws = adv_split(r.at("state").as_str());
        long hit = -1;
        for (std::size_t j = ws.size(); j-- > 0;)
            if (ws[j].size() > 3) { hit = static_cast<long>(j); break; }
        if (hit >= 0) ws[static_cast<std::size_t>(hit)] = entangled[out.size() % entangled.size()];
        Probe p;
        p.state     = adv_join(ws);
        p.questions = r.at("questions");
        for (const auto& kv : r.at("labels").obj) p.labels[kv.first] = kv.second.as_str();
        p.taught    = true;
        out.push_back(p);
        if (out.size() >= 240) break;
    }
    return out;
}
// cross-domain mixture: append foreign content words to the state
inline std::vector<Probe> adv_crossdomain_probes(const std::vector<sfx::JV>& rows,
                                                 const std::vector<std::string>& pool) {
    std::vector<Probe> out;
    if (pool.empty()) return out;
    for (const auto& r : rows) {
        if (!r.has("state") || !r.has("questions") || !r.has("labels")) continue;
        std::string st = r.at("state").as_str();
        for (int j = 0; j < 4; ++j)
            st += " " + pool[(out.size() * 4 + static_cast<std::size_t>(j)) % pool.size()];
        Probe p;
        p.state     = st;
        p.questions = r.at("questions");
        for (const auto& kv : r.at("labels").obj) p.labels[kv.first] = kv.second.as_str();
        p.taught    = true;
        out.push_back(p);
        if (out.size() >= 240) break;
    }
    return out;
}

// -- the conflicting-lessons sub-suite (throwaway engine copy) ---------------
// The M3 contract under attack: contradictory lessons must (a) always be
// detected, (b) always surface as contested in the evidence, (c) keep
// decisions deterministic, and (d) cause no damage beyond what ANY teach
// event causes (the forgetting law decays every lane on every deposit, so a
// CONTROL arm — same states re-taught with their gold outcomes — isolates
// the contradiction-specific cost).
inline ConflictResult adv_conflict_suite(const Engine& eng, const std::vector<sfx::JV>& rows) {
    ConflictResult cr;
    // collect the rows we can contradict (a choice question with a criteria
    // list to pick a wrong answer from)
    struct Target { std::string state, instructions, gold_outcome, wrong_outcome; sfx::JV q; };
    std::vector<Target> targets;
    for (const auto& r : rows) {
        if (!r.has("state") || !r.has("labels")) continue;
        std::string qid; sfx::JV q;
        if (!adv_first_choice(r, qid, q)) continue;
        if (!r.at("labels").has(qid)) continue;
        const std::string gold = r.at("labels").at(qid).as_str();
        std::string wrong;
        if (q.has("criteria") && q.at("criteria").is_obj()) {
            for (const auto& kv : q.at("criteria").obj)
                if (kv.first != gold) { wrong = kv.first; break; }
        } else if (q.has("criteria") && q.at("criteria").is_arr()) {
            for (const auto& v : q.at("criteria").arr) {
                bool is_gold = v.as_str() == gold;
                long idx = std::strtol(gold.c_str(), nullptr, 10);
                if (idx >= 0 && idx < static_cast<long>(q.at("criteria").arr.size()) &&
                    q.at("criteria").arr[static_cast<std::size_t>(idx)].as_str() == v.as_str())
                    is_gold = true;
                if (!is_gold) { wrong = v.as_str(); break; }
            }
        }
        if (wrong.empty()) continue;
        targets.push_back({r.at("state").as_str(), q.at("instructions").as_str(),
                           adv_outcome_text(q, gold), adv_outcome_text(q, wrong), q});
        if (targets.size() >= 40) break;
    }
    if (targets.empty()) return cr;
    std::vector<sfx::JV> suffix(rows.begin() + static_cast<long>(targets.size()), rows.end());
    // pristine copy for the BEFORE baseline (the engine under test is never
    // touched, not even for read-only scoring)
    syfox::Engine pristine = eng;
    if (!suffix.empty())
        cr.untouched_acc_before = labelled_accuracy(pristine, suffix).accuracy;
    // ATTACK copy: establish the gold lesson, then contradict it. The
    // establish step is required — hidden rows were never taught, and a
    // wrong answer to a state the fabric never learned is NOT a dispute.
    syfox::Engine probe = eng;
    const long conflicts_before = static_cast<long>(probe.conflicts().size());
    for (const auto& t : targets) {
        probe.learn_example(t.state, t.instructions, t.gold_outcome);
        probe.learn_example(t.state, t.instructions, t.wrong_outcome);
    }
    cr.taught = static_cast<long>(targets.size());
    cr.detected = static_cast<long>(probe.conflicts().size()) - conflicts_before;
    // CONTROL copy: same teach-event count, gold outcomes only — no dispute
    syfox::Engine control = eng;
    for (const auto& t : targets) {
        control.learn_example(t.state, t.instructions, t.gold_outcome);
        control.learn_example(t.state, t.instructions, t.gold_outcome);
    }
    if (!suffix.empty())
        cr.untouched_acc_control = labelled_accuracy(control, suffix).accuracy;
    // every contradicted state must surface as contested in its evidence
    for (const auto& t : targets) {
        syfox::Usage u;
        sfx::JV qobj(sfx::JVObj{});
        qobj.obj["q"] = t.q;
        auto answers = probe.decide(t.state, qobj, u);
        sfx::JV ev = probe.evidence_json(t.state, qobj, answers);
        ++cr.checked;
        if (ev.at("contested").as_str() == "true") ++cr.contested_flagged;
        // determinism under dispute: same decision twice
        syfox::Usage u2;
        auto answers2 = probe.decide(t.state, qobj, u2);
        if (answers.size() != answers2.size()) { cr.deterministic_after = false; continue; }
        for (std::size_t k = 0; k < answers.size(); ++k)
            if (answers[k].deferred != answers2[k].deferred ||
                std::fabs(answers[k].confidence - answers2[k].confidence) > 1e-6f ||
                answers[k].choice != answers2[k].choice)
                cr.deterministic_after = false;
    }
    // blast radius: untouched suffix after the attack (compare with CONTROL,
    // not just BEFORE — every teach event pays the forgetting-law cost)
    if (!suffix.empty())
        cr.untouched_acc_after = labelled_accuracy(probe, suffix).accuracy;
    return cr;
}

// -- the full suite -----------------------------------------------------------
inline AdvReport adversarial_suite(Engine& eng, const std::vector<sfx::JV>& rows,
                                   const std::vector<std::string>& foreign_pool) {
    AdvReport rep;
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_reorder), "reorder",
        "clause halves swapped; token stream changes, gold label unchanged"));
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_padding), "padding",
        "irrelevant clause appended; co-occurrence fabric must not be hijacked"));
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_typo), "typos",
        "15% deterministic codepoint corruption (deletion/swap/duplication)"));
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_intensifiers), "intensifiers",
        "content words doubled + URGENT caps; intern-mass stress"));
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_self_contradiction), "self_contradiction",
        "opposing-sentiment clause appended; routing intent must survive"));
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_negation), "negation",
        "single 'not' inserted; scored on choice only (valence labels may flip)"));
    rep.families.push_back(adv_score_family(eng,
        adv_transform_rows(rows, adv_double_negation), "double_negation",
        "double-negation clause appended; scored on choice only"));
    // unknown vocabulary: expect defer_rate 1.0, zero answered
    {
        auto probes = adv_unknown_probes(eng, rows);
        AdvFamilyResult fam = adv_score_family(eng, probes, "unknown_concepts",
            "true OOD nonsense vocabulary; 100% defer is the contract, any answer is false confidence");
        rep.families.push_back(fam);
    }
    // near-miss: trigram-entangled foreign words
    {
        auto probes = adv_nearmiss_probes(eng, rows, foreign_pool);
        AdvFamilyResult fam = adv_score_family(eng, probes, "near_miss",
            foreign_pool.empty()
                ? "no foreign pool available (skipped)"
                : "one content word replaced by a trigram-entangled foreign word; tests the traction gate");
        rep.families.push_back(fam);
    }
    // cross-domain mixture
    {
        auto probes = adv_crossdomain_probes(rows, foreign_pool);
        AdvFamilyResult fam = adv_score_family(eng, probes, "cross_domain",
            foreign_pool.empty()
                ? "no foreign pool available (skipped)"
                : "four foreign content words appended; contamination resistance");
        rep.families.push_back(fam);
    }
    rep.conflicts = adv_conflict_suite(eng, rows);
    return rep;
}

} // namespace bench
} // namespace syfox
