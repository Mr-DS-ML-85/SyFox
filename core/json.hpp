// ============================================================================
//  SyFox — json.hpp
//  Minimal dependency-free JSON parser/serializer (a TOOL, not core physics).
//  Supports: null, bool, number, string (with escapes), array, object.
//  C++17, deterministic, no exceptions on well-formed input (throws
//  std::runtime_error with position on malformed input).
// ============================================================================
#pragma once
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sfx {

struct JV;
using JVArr = std::vector<JV>;
using JVObj = std::map<std::string, JV>;

struct JV {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };
    Type t = NUL;
    bool b = false;
    double num = 0.0;
    std::string str;
    JVArr arr;
    JVObj obj;

    JV() = default;
    JV(std::nullptr_t) : t(NUL) {}
    JV(bool v) : t(BOOL), b(v) {}
    JV(double v) : t(NUM), num(v) {}
    JV(int v) : t(NUM), num(v) {}
    JV(const char* v) : t(STR), str(v) {}
    JV(std::string v) : t(STR), str(std::move(v)) {}
    JV(JVArr v) : t(ARR), arr(std::move(v)) {}
    JV(JVObj v) : t(OBJ), obj(std::move(v)) {}

    bool is_null() const { return t == NUL; }
    bool is_bool() const { return t == BOOL; }
    bool is_num()  const { return t == NUM; }
    bool is_str()  const { return t == STR; }
    bool is_arr()  const { return t == ARR; }
    bool is_obj()  const { return t == OBJ; }

    bool has(const std::string& k) const { return t == OBJ && obj.count(k) != 0; }
    const JV& at(const std::string& k) const {
        if (t != OBJ || obj.count(k) == 0) { static const JV nil; return nil; }
        return obj.at(k);
    }
    // tolerant string reader: numbers and bools stringify too
    std::string as_str(const std::string& fallback = "") const {
        switch (t) {
            case STR: return str;
            case NUM: { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", num); return buf; }
            case BOOL: return b ? "true" : "false";
            default: return fallback;
        }
    }
    double as_num(double fallback = 0.0) const { return t == NUM ? num : fallback; }

    std::string dump() const {
        std::string out;
        dump_to(out);
        return out;
    }

    static JV parse(const std::string& text) {
        std::size_t i = 0;
        JV v = parse_value(text, i);
        skip_ws(text, i);
        if (i != text.size()) throw std::runtime_error("JSON: trailing data at " + std::to_string(i));
        return v;
    }

private:
    static void skip_ws(const std::string& s, std::size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }

    static char expect(const std::string& s, std::size_t& i, char c) {
        if (i >= s.size() || s[i] != c)
            throw std::runtime_error(std::string("JSON: expected '") + c + "' at " + std::to_string(i));
        return s[i++];
    }

    static JV parse_value(const std::string& s, std::size_t& i) {
        skip_ws(s, i);
        if (i >= s.size()) throw std::runtime_error("JSON: unexpected end");
        char c = s[i];
        if (c == '{') return parse_obj(s, i);
        if (c == '[') return parse_arr(s, i);
        if (c == '"') return JV(parse_string(s, i));
        if (c == 't') { literal(s, i, "true");  return JV(true); }
        if (c == 'f') { literal(s, i, "false"); return JV(false); }
        if (c == 'n') { literal(s, i, "null");  return JV(nullptr); }
        return parse_num(s, i);
    }

    static void literal(const std::string& s, std::size_t& i, const char* lit) {
        for (const char* p = lit; *p; ++p, ++i)
            if (i >= s.size() || s[i] != *p)
                throw std::runtime_error("JSON: bad literal at " + std::to_string(i));
    }

    static JV parse_num(const std::string& s, std::size_t& i) {
        std::size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        bool digits = false;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                                 s[i] == 'E' || s[i] == '-' || s[i] == '+')) {
            if (s[i] >= '0' && s[i] <= '9') digits = true;
            ++i;
        }
        if (!digits) throw std::runtime_error("JSON: bad number at " + std::to_string(start));
        return JV(std::stod(s.substr(start, i - start)));
    }

    static std::string parse_string(const std::string& s, std::size_t& i) {
        expect(s, i, '"');
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i >= s.size()) break;
                char e = s[i++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {                                    // \uXXXX (BMP)
                        if (i + 4 > s.size()) throw std::runtime_error("JSON: bad \\u escape");
                        unsigned cp = std::stoul(s.substr(i, 4), nullptr, 16);
                        i += 4;
                        if (cp < 0x80) out += static_cast<char>(cp);
                        else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: throw std::runtime_error("JSON: bad escape at " + std::to_string(i));
                }
            } else out += c;
        }
        throw std::runtime_error("JSON: unterminated string");
    }

    static JV parse_arr(const std::string& s, std::size_t& i) {
        expect(s, i, '[');
        JVArr arr;
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { ++i; return JV(std::move(arr)); }
        while (true) {
            arr.push_back(parse_value(s, i));
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            expect(s, i, ']');
            break;
        }
        return JV(std::move(arr));
    }

    static JV parse_obj(const std::string& s, std::size_t& i) {
        expect(s, i, '{');
        JVObj obj;
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}') { ++i; return JV(std::move(obj)); }
        while (true) {
            skip_ws(s, i);
            std::string key = parse_string(s, i);
            skip_ws(s, i);
            expect(s, i, ':');
            obj[key] = parse_value(s, i);
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            expect(s, i, '}');
            break;
        }
        return JV(std::move(obj));
    }

    static void dump_str(const std::string& v, std::string& out) {
        out += '"';
        for (unsigned char c : v) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
                    else out += static_cast<char>(c);
            }
        }
        out += '"';
    }

    void dump_to(std::string& out) const {
        switch (t) {
            case NUL: out += "null"; break;
            case BOOL: out += b ? "true" : "false"; break;
            case NUM: {
                if (std::isfinite(num)) {
                    char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", num); out += buf;
                } else out += "null";
                break;
            }
            case STR: dump_str(str, out); break;
            case ARR: {
                out += '[';
                for (std::size_t i = 0; i < arr.size(); ++i) { if (i) out += ','; arr[i].dump_to(out); }
                out += ']';
                break;
            }
            case OBJ: {
                out += '{';
                bool first = true;
                for (const auto& kv : obj) {
                    if (!first) out += ',';
                    first = false;
                    dump_str(kv.first, out);
                    out += ':';
                    kv.second.dump_to(out);
                }
                out += '}';
                break;
            }
        }
    }
};

} // namespace sfx
