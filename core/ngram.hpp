// ============================================================================
//  SyFox — ngram.hpp
//  Multilingual boundary layer, part 2: character n-gram lanes.
//
//  Scope guard (v2.2): encoding boundary only. si_substrate.hpp is untouched.
//  A token "refund" additionally contributes sub-token lanes named
//  "g3:ref", "g3:efu", "g3:fun", "g3:und" — literal, inspectable strings,
//  NOT hashes. The physics treats them like any other concept node: they
//  are interned, co-injected with their word, and Hebbian-bound to the
//  lesson outcome. A typo ("refnd", "rfeund") still overlaps taught
//  trigrams, so energy reaches the right word lane through diffusion —
//  the field does the matching, the boundary only feeds it sub-word
//  handles. This is the fastText-style sub-word idea rebuilt as SI lane
//  fabric: no embeddings table, no neural net, no pattern matcher.
//
//  Rules (all deterministic):
//    * trigrams only (n=3), sliding window over CODEPOINTS (not bytes),
//      so Bengali/Hindi/Greek words decompose correctly;
//    * tokens shorter than 4 codepoints contribute no trigrams (their only
//      trigram would be the token itself — a redundant lane);
//    * lane name = "g3:" + the 3 codepoints, in left-to-right order;
//    * unknown query trigrams (never taught) stay dark — inject() skips
//      unknown concepts, the honest-silence contract is untouched.
// ============================================================================
#pragma once
#include "script.hpp"

#include <string>
#include <vector>

namespace si {
namespace norm {

// Process-wide trigram switch (v2.2). DEFAULT OFF — measured A/B (README
// "typo-robustness matrix"): on the small Latin seed fabrics, sub-word
// cross-talk trades clean held-out accuracy for corruption robustness at
// roughly breakeven, and the v2.1 baselines are only reproduced with the
// word-level stream. The mechanism stays one flag away and ACTIVATES
// automatically for non-Latin routed substrates (`--lang auto` policy:
// the script barrier and their tiny vocabularies make bridges load-bearing
// there — the Bengali typo-routing test proves it). `--ngrams on|off`
// overrides explicitly. Same pattern as the synonym table: one process-wide
// boundary configuration, set once before any command runs.
inline bool& grams_enabled() { static bool on = false; return on; }

// v3.6.0 — ORDERED BIGRAM LANES (MRS §4b.1-4b.2, measured 7/7; CMD §2.5;
// CPSB ordered pairs; HTR arrival-order tagging). DEFAULT ON for every NEW
// fabric: adjacent normalized token pairs join into bigram nodes ("w1~w2")
// that hebbian_lesson binds to the outcome anchors, and decide() injects the
// state's bigram nodes that exist in the fabric. This is the escape from the
// multiset-invariance theorem that pinned paws at 0.500 and xnli at 0.333
// (chance): a token bag cannot distinguish "a beats b" from "b beats a",
// but the pair multisets {"a~beat","beat~b"} and {"b~beat","beat~a"} differ.
// Old fabrics never contain "w1~w2" nodes, so the decide-side injection
// finds nothing (inject() skips unknown concepts) and stays bit-identical —
// the replay contract holds with no version bump. The '~' separator is not
// alphanumeric, so a bigram can never collide with a normalized word token.
// As with grams: one process-wide boundary configuration, set once before
// any command runs (`--bigrams on|off`).
inline bool& bigrams_enabled() { static bool on = true; return on; }

inline std::string join_bigram(const std::string& a, const std::string& b) {
    std::string g;
    g.reserve(a.size() + 1 + b.size());
    g.append(a); g.push_back('~'); g.append(b);
    return g;
}

// Adjacent ordered pairs of an already-normalized token stream, in stream
// order (deterministic; empty for streams shorter than two tokens).
inline std::vector<std::string> bigrams_of(const std::vector<std::string>& words) {
    std::vector<std::string> out;
    if (words.size() < 2) return out;
    out.reserve(words.size() - 1);
    for (std::size_t i = 1; i < words.size(); ++i)
        out.push_back(join_bigram(words[i - 1], words[i]));
    return out;
}

inline std::vector<std::string> expand_ngrams(const std::vector<std::string>& tokens) {
    std::vector<std::string> out;
    for (const auto& tok : tokens) {
        // decode codepoints once; malformed bytes were already dropped by the
        // tokenizer, but stay defensive: unknown bytes are skipped here too
        std::vector<std::uint32_t> cps;
        std::vector<std::size_t>   byte_off;   // codepoint -> byte offset
        for (std::size_t i = 0; i < tok.size();) {
            std::uint32_t cp = 0;
            const std::size_t len = script::decode_utf8(tok, i, cp);
            if (cp != 0xFFFFFFFFu) { cps.push_back(cp); byte_off.push_back(i); }
            i += len;
        }
        if (cps.size() < 4) continue;                    // sub-word handles only for real words
        out.reserve(out.size() + cps.size() - 2);
        for (std::size_t w = 0; w + 2 < cps.size(); ++w) {
            // each segment is exactly one codepoint: byte ranges come from
            // the offsets table, the last one ends at the next codepoint
            // (or end of token for the final one)
            const std::size_t b0 = byte_off[w],     e0 = byte_off[w + 1];
            const std::size_t b1 = byte_off[w + 1], e1 = byte_off[w + 2];
            const std::size_t b2 = byte_off[w + 2];
            const std::size_t e2 = (w + 3 < byte_off.size()) ? byte_off[w + 3] : tok.size();
            std::string g = "g3:";
            g.append(tok, b0, e0 - b0);
            g.append(tok, b1, e1 - b1);
            g.append(tok, b2, e2 - b2);
            out.push_back(std::move(g));
        }
    }
    return out;
}

// The ONE state-side augmentation protocol for TEACHING: word tokens first
// (preserving v0.1's adjacency fabric word->word unchanged), then each
// word's trigrams. With grams disabled this is exactly the v2.1 token
// stream, byte for byte.
inline std::vector<std::string> state_tokens(const std::vector<std::string>& words,
                                             bool with_grams) {
    if (!with_grams) return words;
    std::vector<std::string> out = words;
    const std::vector<std::string> g = expand_ngrams(words);
    out.insert(out.end(), g.begin(), g.end());
    return out;
}

// ---------------------------------------------------------------------------
// The v2.2 DECIDE-side protocol: trigram bridges for the unknown ONLY.
//
// Measured (v2.2 A/B): injecting every word's grams at decide time floods
// clean queries with sub-word energy (tickets heldout 1.000 -> 0.667, guard
// 0.750 -> 0.250) because unrelated words share trigrams and the extra
// lanes cross-talk. The fix keeps the typo robustness and drops the noise:
// a word the substrate KNOWS takes its direct lane (no grams -> clean
// behavior byte-identical to the word-only encoding); a word it does NOT
// know (a typo, an unseen term) contributes its trigrams as bridges into
// the taught gram lanes. inject() skips unknown concepts, so grams that
// were never taught stay dark on their own.
// ---------------------------------------------------------------------------
template <class HasFn>
inline std::vector<std::string> field_tokens(const std::vector<std::string>& words,
                                             bool with_grams, HasFn&& has) {
    if (!with_grams) return words;
    std::vector<std::string> out = words;
    for (const auto& w : words) {
        if (has(w)) continue;                        // known: direct lane, no bridge
        for (const auto& g : expand_ngrams({w})) out.push_back(g);
    }
    return out;
}

} // namespace norm
} // namespace si
