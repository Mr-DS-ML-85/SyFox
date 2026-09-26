// ============================================================================
//  SyFox — normalize.hpp
//  Input-boundary normalization: synonym folding + Porter stemming.
//
//  Scope guard (v2.1): this file lives at the ENCODING boundary, not in the
//  physics. si_substrate.hpp is byte-identical to v0.2 — energy, decay,
//  diffusion, lanes, readout untouched. What changes is only which token
//  string enters the substrate: "reimbursement" now lands on the same node
//  as "refund", "urgently" on the same node as "urgent". Teach and decide
//  share one deterministic pipeline, so the fabric stays consistent.
//
//    normalize(text) = scan alnum runs -> Porter(1980) -> synonym fold
//
//  Determinism: pure functions, fixed tables, no randomness, no neural net.
//  The synonym table is loaded from data/synonyms.txt when present
//  (--synonyms overrides the path); otherwise an embedded copy of the same
//  table applies. Table keys and values are stemmed AT LOAD TIME, so fold
//  order (porter -> synonyms) is self-consistent no matter how Porter stems
//  a given word.
// ============================================================================
#pragma once
#include "si_substrate.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace si {
namespace norm {

// ---------------------------------------------------------------------------
// Porter stemmer — faithful port of M.F. Porter's reference implementation
// (program: Porter, 1980; the classic ANSI C version). Compact, iterative,
// deterministic. Words of length <= 2 pass through untouched (same as the
// reference driver).
// ---------------------------------------------------------------------------
struct Porter {
    std::string b;
    int k = 0, k0 = 0, j = 0;          // b[0..k] is the word; j is a scratch offset

    explicit Porter(std::string w) : b(std::move(w)) {
        k = static_cast<int>(b.size()) - 1;
    }

    bool cons(int i) const {
        switch (b[i]) {
            case 'a': case 'e': case 'i': case 'o': case 'u': return false;
            case 'y': return (i == k0) ? true : !cons(i - 1);
            default: return true;
        }
    }

    // m() counts consonant-vowel sequences: <c><v> -> 0, <c>vc<v> -> 1, ...
    int m() const {
        int n = 0, i = k0;
        while (true) { if (i > j) return n; if (!cons(i)) break; ++i; }
        ++i;
        while (true) {
            while (true) { if (i > j) return n; if (cons(i)) break; ++i; }
            ++i; ++n;
            while (true) { if (i > j) return n; if (!cons(i)) break; ++i; }
            ++i;
        }
    }

    bool vowel_in_stem() const {
        for (int i = k0; i <= j; ++i) if (!cons(i)) return true;
        return false;
    }

    bool doublec(int i) const {
        if (i < k0 + 1) return false;
        if (b[i] != b[i - 1]) return false;
        return cons(i);
    }

    bool cvc(int i) const {
        if (i < k0 + 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
        const char ch = b[i];
        return !(ch == 'w' || ch == 'x' || ch == 'y');
    }

    // ends(): mirrors the reference — on success sets j = k - length.
    bool ends_j(const char* s) {
        const int length = s[0];                    // first byte = length (Pascal-style)
        if (b[k] != s[length]) return false;
        if (length > k - k0 + 1) return false;
        if (b.compare(k - length + 1, static_cast<std::size_t>(length), s + 1,
                      static_cast<std::size_t>(length)) != 0) return false;
        j = k - length;
        return true;
    }

    void setto(const char* s) {
        const int length = s[0];
        b.replace(static_cast<std::size_t>(j) + 1, static_cast<std::size_t>(length), s + 1,
                  static_cast<std::size_t>(length));
        k = j + length;
    }

    void r(const char* s) { if (m() > 0) setto(s); }

    void step1ab() {
        if (b[k] == 's') {
            if (ends_j("\04" "sses")) k -= 2;
            else if (ends_j("\03" "ies")) setto("\01" "i");
            else if (b[k - 1] != 's') k--;
        }
        if (ends_j("\03" "eed")) { if (m() > 0) k--; }
        else if ((ends_j("\02" "ed") || ends_j("\03" "ing")) && vowel_in_stem()) {
            k = j;
            if (ends_j("\02" "at"))      setto("\03" "ate");
            else if (ends_j("\02" "bl")) setto("\03" "ble");
            else if (ends_j("\02" "iz")) setto("\03" "ize");
            else if (doublec(k)) {
                k--;
                const char ch = b[k];
                if (ch == 'l' || ch == 's' || ch == 'z') k++;
            }
            else if (m() == 1 && cvc(k)) setto("\01" "e");
        }
    }

    void step1c() {
        if (ends_j("\01" "y") && vowel_in_stem()) b[k] = 'i';
    }

    void step2() {
        if (k < k0 + 1) return;
        switch (b[k - 1]) {
            case 'a': if (ends_j("\07" "ational")) { r("\03" "ate"); break; }
                      if (ends_j("\06" "tional")) { r("\04" "tion"); break; } break;
            case 'c': if (ends_j("\04" "enci")) { r("\04" "ence"); break; }
                      if (ends_j("\04" "anci")) { r("\04" "ance"); break; } break;
            case 'e': if (ends_j("\04" "izer")) { r("\04" "ize"); break; } break;
            case 'l': if (ends_j("\03" "bli")) { r("\03" "ble"); break; }
                      if (ends_j("\04" "alli")) { r("\02" "al"); break; }
                      if (ends_j("\05" "entli")) { r("\03" "ent"); break; }
                      if (ends_j("\03" "eli")) { r("\01" "e"); break; }
                      if (ends_j("\05" "ousli")) { r("\03" "ous"); break; } break;
            case 'o': if (ends_j("\07" "ization")) { r("\03" "ize"); break; }
                      if (ends_j("\05" "ation")) { r("\03" "ate"); break; }
                      if (ends_j("\04" "ator")) { r("\03" "ate"); break; } break;
            case 's': if (ends_j("\05" "alism")) { r("\02" "al"); break; }
                      if (ends_j("\07" "iveness")) { r("\03" "ive"); break; }
                      if (ends_j("\07" "fulness")) { r("\03" "ful"); break; }
                      if (ends_j("\07" "ousness")) { r("\03" "ous"); break; } break;
            case 't': if (ends_j("\05" "aliti")) { r("\02" "al"); break; }
                      if (ends_j("\05" "iviti")) { r("\03" "ive"); break; }
                      if (ends_j("\06" "biliti")) { r("\03" "ble"); break; } break;
            case 'g': if (ends_j("\04" "logi")) { r("\03" "log"); break; } break;
        }
    }

    void step3() {
        switch (b[k]) {
            case 'e': if (ends_j("\05" "icate")) { r("\02" "ic"); break; }
                      if (ends_j("\05" "ative")) { r("\00" ""); break; }
                      if (ends_j("\05" "alize")) { r("\02" "al"); break; } break;
            case 'i': if (ends_j("\05" "iciti")) { r("\02" "ic"); break; } break;
            case 'l': if (ends_j("\04" "ical")) { r("\02" "ic"); break; } break;
            case 'f': if (ends_j("\03" "ful")) { r("\00" ""); break; } break;
            case 's': if (ends_j("\04" "ness")) { r("\00" ""); break; } break;
        }
    }

    void step4() {
        static const char* const kSuffs[] = {
            "\02" "al", "\04" "ance", "\04" "ence", "\02" "er", "\02" "ic",
            "\04" "able", "\04" "ible", "\03" "ant", "\05" "ement",
            "\04" "ment", "\03" "ent", "\03" "ion", "\02" "ou", "\03" "ism",
            "\03" "ate", "\03" "iti", "\03" "ous", "\03" "ive", "\03" "ize" };
        // Faithful scan semantics: the FIRST suffix whose ending matches stops
        // the scan; "ion" (index 11) additionally requires the stem to end in
        // s/t (b[j] is the stem's last char — j was set by ends_j). If the
        // condition fails the scan CONTINUES (a later suffix may still fire).
        int i = 0;
        for (; i < 19; ++i) {
            if (ends_j(kSuffs[i])) {
                if (i != 11 || b[j] == 's' || b[j] == 't') break;
            }
        }
        if (i < 19 && m() > 1) k = j;
    }

    void step5() {
        j = k;
        if (b[k] == 'e') {
            const int a = m();
            if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
        }
        if (b[k] == 'l' && doublec(k) && m() > 1) k--;
    }

    void stem() {
        if (k <= k0 + 1) return;                 // reference driver: len <= 2 untouched
        step1ab(); step1c(); step2(); step3(); step4(); step5();
        b.resize(static_cast<std::size_t>(k) + 1);
    }
};

inline std::string porter_stem(const std::string& w) {
    if (w.size() <= 2) return w;
    Porter p(w);
    p.stem();
    return p.b;
}

// ---------------------------------------------------------------------------
// Synonym folding table: word (stemmed) -> stemmed canonical.
// ---------------------------------------------------------------------------
using SynTable = std::map<std::string, std::string>;

inline SynTable& syn_table() { static SynTable t; return t; }
inline bool& syn_loaded()    { static bool loaded = false; return loaded; }

// Embedded copy of data/synonyms.txt (kept in lockstep; the file, when present,
// is authoritative). Same conservative rules: single words, label-safe swaps.
inline const char* default_synonyms_text() {
    return
        "refund: reimbursement, reimburse\n"
        "urgent: asap, immediately, urgently, hurry\n"
        "angry: furious, mad, upset\n"
        "broken: faulty, busted\n"
        "fix: repair, mend\n"
        "problem: issue, trouble, glitch\n"
        "slow: delayed, sluggish, laggy\n"
        "crash: freeze, hang, froze\n"
        "price: cost\n"
        "scared: afraid, fearful\n"
        "money: cash\n"
        "wrong: incorrect, mistaken\n";
}

// Parse "canonical: variant, variant, ..." lines. Keys and values are stemmed
// here, so the fold is applied AFTER Porter and stays self-consistent.
inline void parse_synonyms_text(const std::string& text) {
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        auto trim = [](std::string s) {
            const auto a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return std::string();
            const auto b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        };
        auto lower = [](std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        const std::string canon_raw = lower(trim(line.substr(0, colon)));
        if (canon_raw.empty()) continue;
        const std::string canon = porter_stem(canon_raw);
        syn_table()[canon] = canon;              // canonical maps to itself
        std::string rest = line.substr(colon + 1);
        std::size_t start = 0;
        while (start <= rest.size()) {
            std::size_t comma = rest.find(',', start);
            if (comma == std::string::npos) comma = rest.size();
            const std::string var_raw = lower(trim(rest.substr(start, comma - start)));
            if (!var_raw.empty()) syn_table()[porter_stem(var_raw)] = canon;
            start = comma + 1;
        }
    }
}

inline void load_synonyms(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;                              // missing file: embedded default stays
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    parse_synonyms_text(buf);
    syn_loaded() = true;
}

inline void ensure_synonyms() {
    if (syn_loaded()) return;
    parse_synonyms_text(default_synonyms_text());
    syn_loaded() = true;
}

// ---------------------------------------------------------------------------
// The one normalization pipeline (teach, decide, recall, harvest — everyone).
// Scan rules mirror si::tokenize (alnum runs, 2..24 chars) so only the fold
// layer differs from v0.2.
// ---------------------------------------------------------------------------
inline std::vector<std::string> normalize(const std::string& text) {
    struct Once { Once() { ensure_synonyms(); } };
    static Once once;

    std::vector<std::string> out;
    std::string cur;
    auto emit = [&]() {
        if (cur.size() >= 2 && cur.size() <= 24) {
            std::string t = porter_stem(cur);
            auto it = syn_table().find(t);
            if (it != syn_table().end()) t = it->second;
            out.push_back(std::move(t));
        }
        cur.clear();
    };
    for (char raw : text) {
        unsigned char c = static_cast<unsigned char>(raw);
        if (std::isalnum(c)) cur.push_back(static_cast<char>(std::tolower(c)));
        else if (!cur.empty()) emit();
    }
    if (!cur.empty()) emit();
    return out;
}

} // namespace norm
} // namespace si
