// ============================================================================
//  SyFox — script.hpp
//  Multilingual boundary layer, part 1: Unicode script detection.
//
//  Scope guard (v2.2): this file lives at the INPUT boundary, like
//  normalize.hpp. si_substrate.hpp is untouched — energy, decay, diffusion,
//  lanes, readout all unchanged. What this adds is a deterministic answer to
//  ONE question: "which script family does this text belong to?", so the
//  operator can route queries to the right per-script substrate.
//
//  Design:
//    * UTF-8 -> codepoints (invalid bytes are separators; never a crash).
//    * every codepoint is classified into one of 35 script families via
//      Unicode block ranges (the same ranges the Unicode Charts use).
//    * detect_script(text) = majority vote over LETTER codepoints.
//      Ties break by table order (deterministic). Pure ASCII is Latin.
//    * No NLP pipeline, no neural net, no pattern matching: a range table
//      and a counter. Every decision reproducible by hand.
//
//  Coverage: 35 families. The Latin family alone carries 60+ languages;
//  the full table covers 100+ (see README "Multilingual" for the honest
//  per-family language list). A family needs its own TRAINED substrate to
//  answer questions — the router guarantees the query lands on the right
//  one; honest silence still guards untrained vocabularies.
// ============================================================================
#pragma once
#include <cstdint>
#include <string>

namespace si {
namespace script {

// ---------------------------------------------------------------------------
// Script families (Unicode block lineage). Order matters: it is the
// deterministic tie-break, and "Unknown" must stay last.
// ---------------------------------------------------------------------------
enum class Script {
    Latin, Greek, Cyrillic, Armenian, Hebrew, Arabic, Syriac, Thaana, Nko,
    Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
    Malayalam, Sinhala, Thai, Lao, Tibetan, Myanmar, Georgian, Khmer,
    Mongolian, Ethiopic, Cherokee, Coptic, Vai, Yi, Bopomofo, Han, Kana,
    Hangul, Unknown
};

inline const char* slug(Script s) {
    switch (s) {
        case Script::Latin:      return "latin";
        case Script::Greek:      return "greek";
        case Script::Cyrillic:   return "cyrillic";
        case Script::Armenian:   return "armenian";
        case Script::Hebrew:     return "hebrew";
        case Script::Arabic:     return "arabic";
        case Script::Syriac:     return "syriac";
        case Script::Thaana:     return "thaana";
        case Script::Nko:        return "nko";
        case Script::Devanagari: return "devanagari";
        case Script::Bengali:    return "bengali";
        case Script::Gurmukhi:   return "gurmukhi";
        case Script::Gujarati:   return "gujarati";
        case Script::Oriya:      return "oriya";
        case Script::Tamil:      return "tamil";
        case Script::Telugu:     return "telugu";
        case Script::Kannada:    return "kannada";
        case Script::Malayalam:  return "malayalam";
        case Script::Sinhala:    return "sinhala";
        case Script::Thai:       return "thai";
        case Script::Lao:        return "lao";
        case Script::Tibetan:    return "tibetan";
        case Script::Myanmar:    return "myanmar";
        case Script::Georgian:   return "georgian";
        case Script::Khmer:      return "khmer";
        case Script::Mongolian:  return "mongolian";
        case Script::Ethiopic:   return "ethiopic";
        case Script::Cherokee:   return "cherokee";
        case Script::Coptic:     return "coptic";
        case Script::Vai:        return "vai";
        case Script::Yi:         return "yi";
        case Script::Bopomofo:   return "bopomofo";
        case Script::Han:        return "han";
        case Script::Kana:       return "kana";
        case Script::Hangul:     return "hangul";
        default:                 return "unknown";
    }
}

inline Script from_slug(const std::string& s) {
    static const struct { const char* name; Script sc; } kTab[] = {
        {"latin", Script::Latin}, {"greek", Script::Greek},
        {"cyrillic", Script::Cyrillic}, {"armenian", Script::Armenian},
        {"hebrew", Script::Hebrew}, {"arabic", Script::Arabic},
        {"syriac", Script::Syriac}, {"thaana", Script::Thaana},
        {"nko", Script::Nko}, {"devanagari", Script::Devanagari},
        {"bengali", Script::Bengali}, {"gurmukhi", Script::Gurmukhi},
        {"gujarati", Script::Gujarati}, {"oriya", Script::Oriya},
        {"tamil", Script::Tamil}, {"telugu", Script::Telugu},
        {"kannada", Script::Kannada}, {"malayalam", Script::Malayalam},
        {"sinhala", Script::Sinhala}, {"thai", Script::Thai},
        {"lao", Script::Lao}, {"tibetan", Script::Tibetan},
        {"myanmar", Script::Myanmar}, {"georgian", Script::Georgian},
        {"khmer", Script::Khmer}, {"mongolian", Script::Mongolian},
        {"ethiopic", Script::Ethiopic}, {"cherokee", Script::Cherokee},
        {"coptic", Script::Coptic}, {"vai", Script::Vai},
        {"yi", Script::Yi}, {"bopomofo", Script::Bopomofo},
        {"han", Script::Han}, {"kana", Script::Kana},
        {"hangul", Script::Hangul},
    };
    for (const auto& e : kTab) if (s == e.name) return e.sc;
    return Script::Unknown;
}

// ---------------------------------------------------------------------------
// UTF-8 decode: one codepoint per call. Returns bytes consumed (>=1); sets
// cp to the codepoint, or to 0xFFFFFFFF for a malformed sequence (the caller
// treats it as a separator — deterministic, never a crash).
// ---------------------------------------------------------------------------
inline std::size_t decode_utf8(const std::string& s, std::size_t i,
                               std::uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { cp = c; return 1; }
    std::size_t len = 0; std::uint32_t v = 0;
    if      ((c & 0xE0) == 0xC0) { len = 2; v = c & 0x1Fu; }
    else if ((c & 0xF0) == 0xE0) { len = 3; v = c & 0x0Fu; }
    else if ((c & 0xF8) == 0xF0) { len = 4; v = c & 0x07u; }
    else { cp = 0xFFFFFFFFu; return 1; }
    if (i + len > s.size()) { cp = 0xFFFFFFFFu; return 1; }
    for (std::size_t k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) { cp = 0xFFFFFFFFu; return 1; }
        v = (v << 6) | (cc & 0x3Fu);
    }
    cp = v;
    return len;
}

// ---------------------------------------------------------------------------
// The script range table. Each entry is a Unicode block range mapped to a
// family. Blocks are scanned linearly — the table is small, calls are per
// codepoint, and a linear scan keeps the determinism story trivial.
// ---------------------------------------------------------------------------
inline Script codepoint_script(std::uint32_t cp) {
    struct R { std::uint32_t lo, hi; Script sc; };
    static const R kRanges[] = {
        // Latin (incl. Extended-A/B, Extended Additional for Vietnamese,
        // IPA extensions, phonetic extensions)
        {0x0041, 0x005A, Script::Latin}, {0x0061, 0x007A, Script::Latin},
        {0x00AA, 0x00AA, Script::Latin}, {0x00BA, 0x00BA, Script::Latin},
        {0x00C0, 0x00D6, Script::Latin}, {0x00D8, 0x00F6, Script::Latin},
        {0x00F8, 0x02B8, Script::Latin}, {0x1D00, 0x1D25, Script::Latin},
        {0x1E00, 0x1EFF, Script::Latin}, {0x2C60, 0x2C7F, Script::Latin},
        {0xA720, 0xA7FF, Script::Latin},
        {0x0370, 0x03FF, Script::Greek}, {0x1F00, 0x1FFF, Script::Greek},
        {0x0400, 0x052F, Script::Cyrillic}, {0x2DE0, 0x2DFF, Script::Cyrillic},
        {0xA640, 0xA69F, Script::Cyrillic},
        {0x0530, 0x058F, Script::Armenian},
        {0x0590, 0x05FF, Script::Hebrew},
        {0x0600, 0x06FF, Script::Arabic}, {0x0750, 0x077F, Script::Arabic},
        {0x08A0, 0x08FF, Script::Arabic}, {0xFB50, 0xFDFF, Script::Arabic},
        {0xFE70, 0xFEFF, Script::Arabic},
        {0x0700, 0x074F, Script::Syriac},
        {0x0780, 0x07BF, Script::Thaana},
        {0x07C0, 0x07FF, Script::Nko},
        {0x0900, 0x097F, Script::Devanagari},
        {0x0980, 0x09FF, Script::Bengali},
        {0x0A00, 0x0A7F, Script::Gurmukhi},
        {0x0A80, 0x0AFF, Script::Gujarati},
        {0x0B00, 0x0B7F, Script::Oriya},
        {0x0B80, 0x0BFF, Script::Tamil},
        {0x0C00, 0x0C7F, Script::Telugu},
        {0x0C80, 0x0CFF, Script::Kannada},
        {0x0D00, 0x0D7F, Script::Malayalam},
        {0x0D80, 0x0DFF, Script::Sinhala},
        {0x0E00, 0x0E7F, Script::Thai},
        {0x0E80, 0x0EFF, Script::Lao},
        {0x0F00, 0x0FFF, Script::Tibetan},
        {0x1000, 0x109F, Script::Myanmar},
        {0x10A0, 0x10FF, Script::Georgian}, {0x2D00, 0x2D2F, Script::Georgian},
        {0x1780, 0x17FF, Script::Khmer},
        {0x1800, 0x18AF, Script::Mongolian},
        {0x1200, 0x137F, Script::Ethiopic}, {0x1380, 0x139F, Script::Ethiopic},
        {0x2D80, 0x2DDF, Script::Ethiopic},
        {0x13A0, 0x13FF, Script::Cherokee},
        {0x2C80, 0x2CFF, Script::Coptic},
        {0xA500, 0xA63F, Script::Vai},
        {0xA000, 0xA48F, Script::Yi},
        {0x3100, 0x312F, Script::Bopomofo}, {0x31A0, 0x31BF, Script::Bopomofo},
        {0x2E80, 0x2EFF, Script::Han}, {0x3400, 0x4DBF, Script::Han},
        {0x4E00, 0x9FFF, Script::Han}, {0xF900, 0xFAFF, Script::Han},
        {0x3040, 0x309F, Script::Kana}, {0x30A0, 0x30FF, Script::Kana},
        {0x31F0, 0x31FF, Script::Kana},
        {0x1100, 0x11FF, Script::Hangul}, {0x3130, 0x318F, Script::Hangul},
        {0xA960, 0xA97F, Script::Hangul}, {0xAC00, 0xD7AF, Script::Hangul},
    };
    for (const auto& r : kRanges)
        if (cp >= r.lo && cp <= r.hi) return r.sc;
    return Script::Unknown;
}

inline bool is_letter_cp(std::uint32_t cp) {
    return codepoint_script(cp) != Script::Unknown;
}

inline bool is_digit_cp(std::uint32_t cp) {
    // ASCII digits + the fullwidth/decimal digit blocks of major scripts.
    // Digits are script-neutral for token scanning: they attach to the
    // current token segment like v0.1's alnum runs did.
    return (cp >= 0x0030 && cp <= 0x0039)
        || (cp >= 0x09E6 && cp <= 0x09EF)   // Bengali
        || (cp >= 0x0966 && cp <= 0x096F)   // Devanagari
        || (cp >= 0x0660 && cp <= 0x0669)   // Arabic-Indic
        || (cp >= 0x06F0 && cp <= 0x06F9)   // Extended Arabic-Indic
        || (cp >= 0xFF10 && cp <= 0xFF19);  // Fullwidth
}

// ---------------------------------------------------------------------------
// Dominant-script detection: majority vote over LETTER codepoints of the
// whole text. Digits and separators are neutral. Ties break by table order
// (earlier enum value wins) — deterministic. Empty/unknown text -> Latin
// (the ASCII default the engine has always had).
// ---------------------------------------------------------------------------
inline Script detect_script(const std::string& text) {
    long counts[35] = {0};                               // one per family
    std::uint32_t cp = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        i += decode_utf8(text, i, cp);
        if (cp == 0xFFFFFFFFu) continue;
        const Script s = codepoint_script(cp);
        if (s != Script::Unknown) ++counts[static_cast<int>(s)];
    }
    long best = 0;
    for (int k = 1; k < 35; ++k)                         // strict > : earlier wins ties
        if (counts[k] > counts[best]) best = k;
    return counts[best] == 0 ? Script::Latin : static_cast<Script>(best);
}

} // namespace script
} // namespace si
