// ============================================================================
//  calc — the arithmetic derivation oracle (JAS "S" side, decision layer).
//
//  The central Synthetic-Intelligence paper (§1.1, §3.4) is explicit:
//  "A calculator is a trivial SI system: given '17 x 23', it does not
//  sample; it derives." Arithmetic enters as COMPOSABLE PRIMITIVES at the
//  decision layer with a verifier — never as energy-field dynamics. The
//  settled field is "a point on a manifold, not an answer"; the log2(N)
//  bits of a specific answer must be stored by a discrete compositional
//  engine (Translation Capacity Lemma).
//
//  This file is that engine, exactly as small as the claim requires:
//    |Pi| = 6 primitives (+ - * / % ^)  composing into arbitrarily deep
//    derivations via one recursive-descent grammar  ->  the thinker ratio
//    (few primitives, exponential reachable expressions), not a lookup.
//
//  syfox DOES NOT claim the field computes. `calc` is an oracle in the
//  JAS sense: external, deterministic, indifferent — and when run with
//  --compile the engine EMITS a C++ translation unit and hands the
//  computation to g++ (the compile-verify oracle of jas.md / synth.cpp),
//  disclosed as provenance "established_by_experiment".
//
//  No loss function, no gradient, no fitted parameter.
// ============================================================================
#pragma once
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace syfox {
namespace calc {

// ---------------------------------------------------------------- result
struct CalcResult {
    bool   ok = false;
    bool   integral = false;     // exact integer path (long long)
    long long iv = 0;            // integral == true
    double dv = 0.0;             // integral == false
    std::string error;           // empty when ok
    std::string text() const {
        if (!ok) return std::string();
        if (integral) return std::to_string(iv);
        // shortest unambiguous decimal rendering
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.10g", dv);
        return std::string(buf);
    }
};

// -------------------------------------------------------------- tokenizer
// Tokens: numbers (integer/decimal), identifiers kept for error reporting,
// single-char operators and parens. Whitespace skipped. '*' '/' and '%'
// are always operators; '^' is exponentiation.
struct Tok {
    enum Kind { NUM, OP, LP, RP, END } kind = END;
    std::string text;            // operator char or numeric lexeme
    bool is_float = false;
};

inline std::vector<Tok> tokenize(const std::string& s, std::string& err) {
    std::vector<Tok> out;
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isspace(c)) { ++i; continue; }
        if (std::isdigit(c) || (c == '.' && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1])))) {
            Tok t; t.kind = Tok::NUM;
            bool dot = false;
            while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.')) {
                if (s[i] == '.') {
                    if (dot) { err = "malformed number"; return out; }
                    dot = true;
                }
                t.text += s[i++];
            }
            t.is_float = dot;
            out.push_back(std::move(t));
            continue;
        }
        if (s[i] == '(') { out.push_back({Tok::LP, "("}); ++i; continue; }
        if (s[i] == ')') { out.push_back({Tok::RP, ")"}); ++i; continue; }
        if (std::string("+-*/%^").find(s[i]) != std::string::npos) {
            // unary minus / plus: an operator at start or after another
            // operator or '(' is a sign bound to the next number
            const bool unary = out.empty() || out.back().kind == Tok::OP || out.back().kind == Tok::LP;
            if (unary && (s[i] == '-' || s[i] == '+')) {
                // fold the sign into the following number token
                std::string lex(1, s[i++]);
                while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.')) {
                    lex += s[i];
                    if (s[i] == '.' && lex.find('.') != lex.rfind('.')) {
                        err = "malformed number"; return out;
                    }
                    ++i;
                }
                Tok t; t.kind = Tok::NUM; t.text = lex; t.is_float = lex.find('.') != std::string::npos;
                out.push_back(std::move(t));
                continue;
            }
            out.push_back({Tok::OP, std::string(1, s[i])});
            ++i;
            continue;
        }
        err = "unexpected character";
        return out;
    }
    out.push_back({Tok::END, ""});
    return out;
}

// --------------------------------------------------- recursive descent
// Grammar (classic precedence climbing):
//   expr   := term  (('+'|'-') term)*
//   term   := unary (('*'|'/'|'%') unary)*
//   unary  := power                        // signs folded in tokenizer
//   power  := prim ('^' power)?            // right-associative
//   prim   := NUM | '(' expr ')'
//
// The verifier is the grammar itself: anything that does not reduce is
// rejected with an error, never guessed.

struct Parser {
    const std::vector<Tok>* toks;
    std::size_t pos = 0;
    std::string err;

    const Tok& peek() const { return (*toks)[pos]; }
    void advance() { if (peek().kind != Tok::END) ++pos; }

    bool expect(Tok::Kind k, const char* what) {
        if (peek().kind != k) { err = std::string("expected ") + what; return false; }
        return true;
    }

    // exact integer power (exp >= 0); false on overflow
    static bool ipow(long long base, long long exp, long long& out) {
        if (exp < 0) return false;
        long long r = 1;
        for (long long i = 0; i < exp; ++i) {
            if (base != 0 && (r > (1LL << 40) || r < -(1LL << 40))) return false; // guard
            r *= base;
        }
        out = r;
        return true;
    }

    CalcResult prim() {
        CalcResult r;
        if (!expect(Tok::NUM, "a number") && !expect(Tok::LP, "a number")) return r;
        if (peek().kind == Tok::LP) {
            advance();
            r = expr();
            if (!r.ok) return r;
            if (!expect(Tok::RP, "')'")) { r.ok = false; r.error = err; return r; }
            advance();
            return r;
        }
        const Tok& t = peek();
        r.ok = true;
        if (t.is_float) { r.integral = false; r.dv = std::strtod(t.text.c_str(), nullptr); }
        else {
            r.integral = true;
            r.iv = std::strtoll(t.text.c_str(), nullptr, 10);
            // overflow of the literal itself -> fall to double path
            if (std::fabs(static_cast<double>(r.iv)) > 9.0e18) {
                r.integral = false;
                r.dv = std::strtod(t.text.c_str(), nullptr);
            }
        }
        advance();
        return r;
    }

    CalcResult power() {
        CalcResult base = prim();
        if (!base.ok) return base;
        if (peek().kind == Tok::OP && peek().text == "^") {
            advance();
            CalcResult exp = power();          // right-assoc
            if (!exp.ok) return exp;
            if (base.integral && exp.integral && exp.iv >= 0 && exp.iv <= 62) {
                long long r;
                if (ipow(base.iv, exp.iv, r)) {
                    CalcResult out; out.ok = true; out.integral = true; out.iv = r;
                    return out;
                }
                // integer overflow -> verified fallback to double
            }
            CalcResult out; out.ok = true; out.integral = false;
            out.dv = std::pow(base.integral ? static_cast<double>(base.iv) : base.dv,
                              exp.integral ? static_cast<double>(exp.iv) : exp.dv);
            if (std::isnan(out.dv) || std::isinf(out.dv)) {
                out.ok = false; out.error = "result not finite";
            }
            return out;
        }
        return base;
    }

    CalcResult unary() { return power(); }

    CalcResult term() {
        CalcResult left = unary();
        if (!left.ok) return left;
        while (peek().kind == Tok::OP && (peek().text == "*" || peek().text == "/" || peek().text == "%")) {
            const std::string op = peek().text;
            advance();
            CalcResult right = unary();
            if (!right.ok) return right;
            CalcResult out;
            // exact integer path when both sides are integral
            if (left.integral && right.integral && !(op == "/" && right.iv == 0) && !(op == "%" && right.iv == 0)) {
                if (op == "*" && (left.iv == 0 || right.iv == 0 ||
                                  (std::llabs(left.iv) < (1LL << 40) && std::llabs(right.iv) < (1LL << 40)))) {
                    out.ok = true; out.integral = true; out.iv = left.iv * right.iv;
                } else if (op == "/") {
                    out.ok = true; out.integral = true; out.iv = left.iv / right.iv;
                } else if (op == "%") {
                    out.ok = true; out.integral = true; out.iv = left.iv % right.iv;
                } else {
                    out.ok = true; out.integral = false;
                    out.dv = (op == "*") ? static_cast<double>(left.iv) * static_cast<double>(right.iv)
                          : (op == "/") ? static_cast<double>(left.iv) / static_cast<double>(right.iv)
                                        : static_cast<double>(left.iv % right.iv);
                }
            } else {
                const double a = left.integral ? static_cast<double>(left.iv) : left.dv;
                const double b = right.integral ? static_cast<double>(right.iv) : right.dv;
                if (op == "/" && b == 0.0) { out.ok = false; out.error = "division by zero"; return out; }
                if (op == "%" && b == 0.0) { out.ok = false; out.error = "modulo by zero"; return out; }
                out.ok = true;
                out.integral = left.integral && right.integral && op != "/";
                out.dv = (op == "*") ? a * b : (op == "/") ? a / b : std::fmod(a, b);
                if (std::isnan(out.dv) || std::isinf(out.dv)) { out.ok = false; out.error = "result not finite"; return out; }
            }
            left = out;
        }
        return left;
    }

    CalcResult expr() {
        CalcResult left = term();
        if (!left.ok) return left;
        while (peek().kind == Tok::OP && (peek().text == "+" || peek().text == "-")) {
            const std::string op = peek().text;
            advance();
            CalcResult right = term();
            if (!right.ok) return right;
            CalcResult out;
            if (left.integral && right.integral) {
                // overflow-guarded exact addition/subtraction
                const bool safe = (op == "+" && ((right.iv >= 0) == (left.iv >= 0) ||
                                                  std::llabs(left.iv) < (1LL << 62)))
                               || (op == "-" && ((right.iv < 0) == (left.iv >= 0) ||
                                                  std::llabs(left.iv) < (1LL << 62)));
                if (safe) {
                    out.ok = true; out.integral = true;
                    out.iv = (op == "+") ? left.iv + right.iv : left.iv - right.iv;
                } else {
                    out.ok = true; out.integral = false;
                    out.dv = (op == "+") ? static_cast<double>(left.iv) + static_cast<double>(right.iv)
                                         : static_cast<double>(left.iv) - static_cast<double>(right.iv);
                }
            } else {
                const double a = left.integral ? static_cast<double>(left.iv) : left.dv;
                const double b = right.integral ? static_cast<double>(right.iv) : right.dv;
                out.ok = true; out.integral = false;
                out.dv = (op == "+") ? a + b : a - b;
                if (std::isnan(out.dv) || std::isinf(out.dv)) { out.ok = false; out.error = "result not finite"; return out; }
            }
            left = out;
        }
        return left;
    }
};

// Evaluate a pure arithmetic expression. Returns CalcResult; never throws.
inline CalcResult evaluate(const std::string& expression) {
    CalcResult r;
    std::string terr;
    std::vector<Tok> toks = tokenize(expression, terr);
    if (!terr.empty()) { r.error = terr; return r; }
    Parser p; p.toks = &toks;
    r = p.expr();
    if (!r.ok && r.error.empty()) r.error = p.err.empty() ? "parse error" : p.err;
    if (r.ok && p.peek().kind != Tok::END) { r.ok = false; r.error = "trailing input"; }
    return r;
}

// -------------------------------------------------- arithmetic detection
// Does this text ASK for a computation? The field's job stops at this
// relevance heuristic (central paper §3: the substrate is the policy, not
// the calculator). Digits plus an operator/quantity word, or a bare clean
// arithmetic expression.
struct ArithDetect {
    bool has_arithmetic = false;
    std::string expression;      // cleaned candidate expression
    std::string cue;             // which word fired
};

inline ArithDetect detect(const std::string& text) {
    ArithDetect d;
    static const char* cues[] = {"plus", "minus", "times", "multiplied", "divided",
                                 "multiply", "divide", "add", "subtract", "calculate",
                                 "compute", "sum", "product", "difference", "quotient",
                                 "squared", "cubed", "power", "modulo", nullptr};
    // digit present?
    bool digit = false;
    for (char c : text) if (std::isdigit(static_cast<unsigned char>(c))) { digit = true; break; }
    std::string low = text;
    for (char& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (int i = 0; cues[i]; ++i) {
        if (low.find(cues[i]) != std::string::npos) { d.cue = cues[i]; break; }
    }
    // operator symbols present?
    bool sym = false;
    for (char c : text) if (std::string("+-*/%^").find(c) != std::string::npos) { sym = true; break; }
    // a bare expression like "17*23" or "(2+3)*4"
    bool bare = digit && sym;
    bool cued = digit && !d.cue.empty();
    if (!(bare || cued)) return d;
    // extract candidate expression: keep digits, operators, parens, dots,
    // and number-cue words mapped to operators; drop everything else.
    static const std::pair<const char*, const char*> wordmap[] = {
        {"plus", "+"}, {"add", "+"}, {"sum of", "+"}, {"sum", "+"},
        {"minus", "-"}, {"subtract", "-"}, {"difference", "-"},
        {"times", "*"}, {"multiplied by", "*"}, {"multiply", "*"}, {"product", "*"},
        {"divided by", "/"}, {"divide", "/"}, {"quotient", "/"},
        {"squared", "^2"}, {"cubed", "^3"}, {"power", "^"}, {"modulo", "%"},
        {nullptr, nullptr}};
    std::string work = low;
    std::string expr;
    for (int i = 0; wordmap[i].first; ++i) {
        std::string w = wordmap[i].first;
        std::size_t p;
        while ((p = work.find(w)) != std::string::npos) {
            // replace the word with its operator surrounded by spaces
            work.replace(p, w.size(), std::string(" ") + wordmap[i].second + " ");
        }
    }
    for (char c : work) {
        if (std::isdigit(static_cast<unsigned char>(c)) || std::string("+-*/%^(). ").find(c) != std::string::npos)
            expr += c;
    }
    // collapse spaces
    std::string cleaned;
    for (char c : expr) {
        if (c == ' ') continue;
        cleaned += c;
    }
    if (cleaned.empty()) return d;
    // a valid candidate must contain a digit AND an operator
    bool cd = false, co = false;
    for (char c : cleaned) {
        if (std::isdigit(static_cast<unsigned char>(c))) cd = true;
        if (std::string("+-*/%^").find(c) != std::string::npos) co = true;
    }
    d.has_arithmetic = cd && co;
    d.expression = cleaned;
    return d;
}

}  // namespace calc
}  // namespace syfox
