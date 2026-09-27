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
#include "normalize.hpp"
#include "ngram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace syfox {

inline const char* VERSION = "3.1.0";

// ---------------------------------------------------------------------------
// v2.2 boundary injection protocol — sub-word bridges for corrupted forms.
// Words inject at the substrate's own inject_energy; the trigram bridges of
// an unknown word inject ONLY when that word passes the sub-word traction
// gate below (a corrupted form of something taught). A boundary protocol,
// not core physics: si_substrate.hpp is untouched.
// ---------------------------------------------------------------------------
inline constexpr float kBridgeEnergy = 1.0f;

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
            {"lanes", static_cast<double>(si_.lane_count())},
            {"evidence_records", static_cast<double>(si_.evidence_count())},
            {"teach_events", static_cast<double>(teach_seq_)},
            {"contradictions", static_cast<double>(conflicts_.size())}}).dump();
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

    void load_model(const std::string& dir) {
        si_.load(dir + "/substrate.bin");
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
    }

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
        si_.hebbian_lesson(state, outcome, eta);
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
        if (y) {
            si_.hebbian_lesson(state, instr, 2.0f * eta);   // supporting evidence binds hard
            for (const auto& a : state)
                if (si_.has(a))
                    for (const auto& b : instr)
                        if (si_.has(b)) si_.record_support(si_.find(a), si_.find(b), seq, context_);
        } else {
            si_.weaken(state, instr, eta);                  // disconfirming evidence dissolves
            for (const auto& a : state)
                if (si_.has(a))
                    for (const auto& b : instr)
                        if (si_.has(b)) si_.record_counter(si_.find(a), si_.find(b), seq, context_);
        }
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

    // -- DECISION ----------------------------------------------------------------
    std::vector<Answer> decide(const std::string& state, const sfx::JV& questions, Usage& usage) {
        usage = Usage{};
        usage.vocabulary = si_.node_count();
        usage.lanes = si_.lane_count();

        std::vector<Answer> answers;
        const std::vector<std::string> words = si::norm::normalize(state);
        usage.state_tokens = words.size();

        si_.reset_field();
        si_.inject(words, state_dose(words));          // words at the (gained) substrate level
        if (si::norm::grams_enabled()) {                 // bridges for corrupted forms, gated
            std::vector<std::string> bridges;
            for (const auto& w : words)
                if (!si_.has(w) && subword_traction(si_, w))
                    for (const auto& g : si::norm::expand_ngrams({w}))
                        if (si_.has(g)) bridges.push_back(g);
            si_.inject(bridges, kBridgeEnergy);
            usage.state_tokens += bridges.size();
        }
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
                a.reason = words.empty() ? "empty_state"
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
            {
                const std::vector<std::string> hw = si::norm::normalize(state);
                si_.inject(hw, state_dose(hw));           // gained dose (Milestone-1)
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
    // Milestone-3 audit state
    std::string context_ = "default";
    std::uint64_t teach_seq_ = 0;
    std::map<std::uint64_t, Taught> taught_outcomes_;               // state-stream hash -> outcome
    std::vector<sfx::JV> conflicts_;
    std::map<std::uint64_t, std::vector<std::size_t>> conflicts_by_state_;
};

} // namespace syfox
