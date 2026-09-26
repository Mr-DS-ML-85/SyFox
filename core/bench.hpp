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
// happens to know is skipped (no fake OOD).
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
    std::vector<std::string> out;
    for (int i = 0; i < n && i < K; ++i) {
        std::vector<std::string> toks;
        for (int k = 0; k < 3; ++k) {
            const std::string w = kNonsense[(i * 3 + k) % K];
            if (!s.has(w)) toks.push_back(w);          // unknown => stays dark
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

} // namespace bench
} // namespace syfox
