// ============================================================================
//  SyFox — roles.hpp  (v3.7.0, Move 3 of the AI-agent limit analysis)
//  ---------------------------------------------------------------------------
//  TYPED ORDERED LANES — the STRONG version of order, not the bigram patch.
//  The v3.6 ordered bigram lanes escape the multiset-invariance theorem for
//  ADJACENT transpositions ("a beats b" vs "b beats a") — that was measured
//  7/7. What bigrams still miss is ROLE asymmetry carried across inserted
//  material: in "alice gave the book to bob" vs "bob gave the book to alice"
//  only 2 of 5 bigram lanes differ, so the reversal margin is diluted by the
//  shared middle; with longer VPs the dilution wins. The strong version is
//  scene-grammar / symbolic parsing, pre-deep-NLP: nodes carry a ROLE and
//  lanes carry a RELATION TYPE.
//
//  This file is that grammar — compact, deterministic, inspectable, honest
//  about being a heuristic:
//
//    roles   S (subject/agent)  V (verb/action)  O (object/patient)
//            M (modifier/other content)          F (function word)
//    cues    a closed function-word list + a seed verb-cue lexicon (stemmed,
//            since normalize() runs Porter before this) + a negation list.
//            All tables are inline, printable, and overridable by
//            data/roles_lexicon.txt when present (verb cues one per line,
//            '#' comments).
//    rules   single-clause scope (lesson states are short):
//              S = nearest content token BEFORE the first verb cue,
//                  else the first content token;
//              O = nearest content token AFTER the last verb cue,
//                  else the last content token (never S itself);
//              V = the verb-cue tokens; M = every other content token;
//              F = function words (excluded from typed pairs).
//            A negation cue within 2 tokens before the first verb marks the
//            verb negated.
//
//  Typed pair NODES (mangled names, interned like the v3.6 bigrams — '~'
//  proved the pattern: a non-alphanumeric separator can never collide with
//  a normalized token):
//      S->V   "w#s>v#v"        V->O   "w#v>o#o"        S->O   "w#s>o#o"
//      negated verb: "!w#v"
//  The '#' and '>' and '!' characters are stripped by normalize(), so these
//  names live in a disjoint namespace from word tokens, exactly like "w1~w2".
//
//  Fabric construction only — hebbian_lesson binds them to outcome anchors
//  like any other node, decide() injects the ones the fabric carries, and
//  OLD fabrics contain none, so every decide-side block is a no-op on them
//  (the replay contract, unit-tested both ways like the bigrams).
//  Default OFF at the engine (a seed lexicon is not yet a general grammar);
//  --typed-lanes on|off overrides. The reversal probe measurement is part
//  of the v3.7 record.
// ============================================================================
#pragma once
#include "si_substrate.hpp"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace si {
namespace roles {

inline bool& typed_lanes_enabled() { static bool on = false; return on; }

// -- closed-class function words (normalized/stemmed forms) ------------------
inline bool is_function_word(const std::string& t) {
    static const char* const kFn[] = {
        "the", "a", "an", "of", "to", "for", "with", "on", "at", "by", "from",
        "in", "into", "about", "as", "is", "are", "was", "were", "be", "been",
        "am", "my", "your", "hi", "his", "her", "it", "our", "their", "this",
        "that", "these", "those", "he", "she", "we", "they", "i", "you",
        "do", "did", "have", "had", "will", "would", "can", "could", "should",
        "shall", "may", "might", "must", "and", "or", "but", "if", "then",
        "than", "so", "very", "realli", "just", "also", "again", "there",
        "here", "what", "which", "who", "whom", "when", "where", "whi", "how",
        "pleas", "kindli", "now", "yet", "not", nullptr};
    for (int i = 0; kFn[i]; ++i) if (t == kFn[i]) return true;
    return false;
}

// negation cues (checked BEFORE the function list: "not"/"no" are negation)
inline bool is_negation_cue(const std::string& t) {
    return t == "not" || t == "no" || t == "never" || t == "nt" || t == "cannot";
}

// -- seed verb-cue lexicon (stemmed). Compact, inspectable, extensible. -----
struct VerbLex {
    bool loaded = false;
    std::vector<std::string> cues = {
        // transfer / commerce (STEMED forms — normalize() runs Porter first:
        // pay->pai, buy->bui, say->sai, lose->los)
        "giv", "gave", "sent", "send", "receiv", "paid", "pai", "pay",
        "charg", "refund", "purchas", "bought", "bui", "buy", "order",
        "ship", "deliver", "cancel", "return", "ow", "owe", "swip",
        "billed", "bill", "sold", "offer", "offered",
        // action / change
        "fix", "repair", "break", "broke", "open", "close", "start",
        "stop", "updat", "delet", "creat", "make", "made", "move",
        // communication
        "call", "told", "tell", "said", "sai", "ask", "answer", "report",
        "notif", "inform", "confirm", "promis", "complain",
        // perception / cognition
        "found", "find", "saw", "see", "heard", "hear", "know", "knew",
        "think", "thought", "want", "need", "trie", "tri", "us", "use",
        "used",
        // conflict / evaluation (the reversal probes)
        "beat", "won", "win", "lost", "los", "lose", "defeat", "bit",
        "bite", "prais", "blame", "help", "hurt", "hit", "struck", "kick",
        "push", "love", "hate", "lik", "dislik", "trust", "doubt",
        "verifi", "approv", "deni"};
};
inline VerbLex& verb_lex() {
    static VerbLex lex;
    return lex;
}

// optional override file: one stemmed verb cue per line, '#' comments
inline void load_verb_lexicon(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    VerbLex& lex = verb_lex();
    lex.cues.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] != '#') {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty()) lex.cues.push_back(line);
        }
    }
    lex.loaded = true;
}

inline bool is_verb_cue(const std::string& t) {
    const VerbLex& lex = verb_lex();
    for (const auto& c : lex.cues) if (t == c) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Role tagging over an ALREADY-NORMALIZED token stream (what learn_example
// and decide() both hold). Deterministic; single-clause scope.
// ---------------------------------------------------------------------------
enum class Role { S, V, O, M, F };

inline std::vector<Role> tag(const std::vector<std::string>& words) {
    const std::size_t n = words.size();
    std::vector<Role> r(n, Role::M);
    if (n == 0) return r;
    for (std::size_t i = 0; i < n; ++i) {
        if (is_function_word(words[i])) r[i] = Role::F;
        else if (is_verb_cue(words[i])) r[i] = Role::V;
    }
    // first / last verb
    std::ptrdiff_t v0 = -1, vl = -1;
    for (std::size_t i = 0; i < n; ++i)
        if (r[i] == Role::V) { if (v0 < 0) v0 = static_cast<std::ptrdiff_t>(i); vl = static_cast<std::ptrdiff_t>(i); }
    if (v0 < 0) {
        // no verb cue: first content token = S, last content token (if another) = O
        std::ptrdiff_t s = -1;
        for (std::size_t i = 0; i < n; ++i)
            if (r[i] != Role::F) { s = static_cast<std::ptrdiff_t>(i); break; }
        if (s >= 0) {
            r[static_cast<std::size_t>(s)] = Role::S;
            for (std::size_t i = n; i-- > static_cast<std::size_t>(s) + 1;)
                if (r[i] != Role::F && r[i] != Role::S) { r[i] = Role::O; break; }
        }
        return r;
    }
    // S: nearest content token before the FIRST verb
    for (std::ptrdiff_t i = v0 - 1; i >= 0; --i)
        if (r[static_cast<std::size_t>(i)] != Role::F) { r[static_cast<std::size_t>(i)] = Role::S; break; }
    // O: prefer the PREPOSITIONAL PATIENT — the content token right after
    // "to"/"for" following the verb ("gave the book TO bob": bob is the
    // patient the reversal turns on; the direct object "book" is shared by
    // both readings). Without a to/for phrase: nearest content token after
    // the LAST verb (never the S slot).
    bool o_set = false;
    for (std::size_t i = static_cast<std::size_t>(vl) + 1; i + 1 < n; ++i) {
        if ((words[i] == "to" || words[i] == "for")
            && r[i + 1] != Role::F && r[i + 1] != Role::S) {
            r[i + 1] = Role::O;
            o_set = true;
            break;
        }
    }
    if (!o_set)
        for (std::size_t i = static_cast<std::size_t>(vl) + 1; i < n; ++i)
            if (r[i] != Role::F && r[i] != Role::S) { r[i] = Role::O; break; }
    return r;
}

// negated-verb detection: a negation cue within 2 tokens before the first V
inline bool verb_negated(const std::vector<std::string>& words) {
    const std::vector<Role> r = tag(words);
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (r[i] != Role::V) continue;
        for (std::size_t j = i >= 2 ? i - 2 : 0; j < i; ++j)
            if (is_negation_cue(words[j])) return true;
        return false;
    }
    return false;
}

inline std::string join_typed(const std::string& a, const char* ra,
                              const std::string& b, const char* rb) {
    std::string g;
    g.reserve(a.size() + b.size() + 6);
    g.append(a); g.push_back('#'); g.append(ra);
    g.push_back('>'); g.append(b); g.push_back('#'); g.append(rb);
    return g;
}

// ---------------------------------------------------------------------------
// Typed pair nodes of a normalized stream, in deterministic emission order:
// S->V, V->O, S->O, then the negation marker when the verb is negated.
// Empty when the tagging found no verb or no S/O counterpart.
// ---------------------------------------------------------------------------
inline std::vector<std::string> typed_pairs_of(const std::vector<std::string>& words) {
    std::vector<std::string> out;
    const std::size_t n = words.size();
    if (n < 2) return out;
    const std::vector<Role> r = tag(words);
    std::ptrdiff_t s = -1, v = -1, o = -1;
    for (std::size_t i = 0; i < n; ++i) {
        if (r[i] == Role::S) s = static_cast<std::ptrdiff_t>(i);
        else if (r[i] == Role::V) v = static_cast<std::ptrdiff_t>(i);
        else if (r[i] == Role::O) o = static_cast<std::ptrdiff_t>(i);
    }
    if (s < 0 || o < 0) return out;                 // need both parties
    if (v >= 0) {
        out.push_back(join_typed(words[static_cast<std::size_t>(s)], "s",
                                 words[static_cast<std::size_t>(v)], "v"));
        out.push_back(join_typed(words[static_cast<std::size_t>(v)], "v",
                                 words[static_cast<std::size_t>(o)], "o"));
    }
    out.push_back(join_typed(words[static_cast<std::size_t>(s)], "s",
                             words[static_cast<std::size_t>(o)], "o"));
    if (v >= 0 && verb_negated(words))
        out.push_back(std::string("!") + words[static_cast<std::size_t>(v)] + "#v");
    return out;
}

} // namespace roles
} // namespace si
