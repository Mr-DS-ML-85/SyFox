// ============================================================================
//  SyFox — recall.hpp
//  Associative retrieval through field dynamics.
//
//  The user constraint holds: NO transformer, NO pattern matching. Recall is
//  a Hopfield-style content-addressable memory where similarity lives in the
//  SUBSTRATE'S OWN STATE SPACE, not in symbol space:
//
//    1. settle the query state -> its settled-energy fingerprint Fq
//       (one float per concept node, L2-normalized)
//    2. settle each stored memory state -> Fm
//    3. resonance(Fq, Fm) = cosine of the two fields
//
//  There is no token comparison, no string edit distance, no n-gram overlap,
//  no embedding table: the fabric itself answers "which lived experience does
//  this state most resemble" by interference of energy distributions. Two
//  states that share no surface token can still resonate if the field routes
//  them to the same region; two states sharing tokens can resonate weakly if
//  the fabric keeps them apart.
//
//  Read-only: fingerprints are transient field states; nothing persists.
//  Deterministic: same model + same query + same memories => same ranking.
// ============================================================================
#pragma once
#include "si_substrate.hpp"
#include "normalize.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace syfox {
namespace recall {

struct Memory {
    std::string label;                 // free text (e.g. the taught outcome)
    std::vector<std::string> state;    // tokenized memory state
};

struct RecallHit {
    std::size_t index;                 // position in the input memory list
    std::string label;
    float resonance;                   // cosine in settled-energy space, [-0,1]
};

// Settled-energy fingerprint of a token list: settle, then collect the field.
// Unknown tokens contribute nothing (inject skips them — honest silence).
inline std::vector<float> fingerprint(si::Substrate& s,
                                      const std::vector<std::string>& tokens) {
    s.reset_field();
    s.inject(tokens);
    s.settle();
    std::vector<float> f(s.node_count(), 0.0f);
    double n2 = 0.0;
    for (std::size_t i = 0; i < s.node_count(); ++i) {
        f[i] = s.node_energy(static_cast<si::NodeId>(i));
        n2 += static_cast<double>(f[i]) * static_cast<double>(f[i]);
    }
    if (n2 > 0.0) {
        const float nrm = static_cast<float>(std::sqrt(n2));
        for (auto& v : f) v /= nrm;
    }
    return f;
}

inline std::vector<RecallHit> recall(si::Substrate& s,
                                     const std::string& query_state,
                                     const std::vector<Memory>& memories,
                                     int top_k = 5) {
    std::vector<RecallHit> hits;
    if (memories.empty() || s.node_count() == 0) return hits;
    // v2.2: query goes through the SAME state-side protocol as decide()
    // (words + trigram lanes when grams are on); the CLI builds memory
    // states with the same protocol, so both sides of the cosine speak the
    // same encoding. Similarity itself is still settled-field cosine — the
    // grams are fed to the physics, they are never the similarity metric.
    const std::vector<float> fq = fingerprint(s,
        si::norm::field_tokens(si::norm::normalize(query_state),
                               si::norm::grams_enabled(),
                               [&s](const std::string& w) { return s.has(w); }));
    float qn2 = 0.0f;
    for (float v : fq) qn2 += v * v;
    if (qn2 <= 0.0f) return hits;      // query resonates with nothing

    for (std::size_t m = 0; m < memories.size(); ++m) {
        const std::vector<float> fm = fingerprint(s, memories[m].state);
        float dot = 0.0f;
        for (std::size_t i = 0; i < fq.size() && i < fm.size(); ++i)
            dot += fq[i] * fm[i];
        hits.push_back({m, memories[m].label, dot});   // both normalized => cosine
    }
    std::stable_sort(hits.begin(), hits.end(),
                     [](const RecallHit& x, const RecallHit& y) {
                         return x.resonance > y.resonance;
                     });
    if (static_cast<int>(hits.size()) > top_k) hits.resize(static_cast<std::size_t>(top_k));
    return hits;
}

} // namespace recall
} // namespace syfox
