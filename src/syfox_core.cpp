// ============================================================================
//  SyFox core C API — thin shared-library wrapper (server bridge).
//  All decisions come from the SI substrate; this file only marshals JSON.
// ============================================================================
#include "core/syfox.hpp"

#include <cstring>
#include <string>

using syfox::Engine;

extern "C" {

struct SyFoxHandle { Engine* eng; };

SyFoxHandle* syfox_engine_create(const char* model_dir) {
    if (!model_dir) return nullptr;
    SyFoxHandle* h = new SyFoxHandle();
    h->eng = new Engine();
    h->eng->load_model(model_dir);
    return h;
}

void syfox_engine_free(SyFoxHandle* h) {
    if (!h) return;
    delete h->eng;
    delete h;
}

// Decides and returns a malloc'd JSON string: {"answers":..., "usage":...}.
// Caller frees with syfox_string_free. On malformed questions returns
// {"error": "..."}.
char* syfox_decide(SyFoxHandle* h, const char* state, const char* questions_json) {
    if (!h || !state || !questions_json) return nullptr;
    std::string out;
    try {
        sfx::JV q = sfx::JV::parse(questions_json);
        if (!q.is_obj()) { out = "{\"error\":\"questions must be an object\"}"; }
        else {
            syfox::Usage u;
            auto answers = h->eng->decide(state, q, u);
            u.calibrated = h->eng->calibration().fitted;
            sfx::JVObj ans;
            for (const auto& a : answers) {
                sfx::JVObj o;
                o["type"] = sfx::JV(a.type);
                if (a.type == "choice") {
                    o["choice"] = sfx::JV(a.choice);
                    sfx::JVObj pr;
                    for (const auto& p : a.probabilities)
                        pr[p.first] = sfx::JV(std::round(p.second * 1000.0f) / 1000.0f);
                    o["probabilities"] = sfx::JV(pr);
                } else if (a.type == "score") {
                    o["score"] = sfx::JV(std::round(a.value * 1000.0f) / 1000.0f);
                    sfx::JVObj pr;
                    for (const auto& p : a.probabilities)
                        pr[p.first] = sfx::JV(std::round(p.second * 1000.0f) / 1000.0f);
                    o["probabilities"] = sfx::JV(pr);
                } else if (a.type == "noul") {
                    o["noul"] = sfx::JV(std::round(a.probability * 1000.0f) / 1000.0f);
                }
                o["confidence"] = sfx::JV(std::round(a.confidence * 1000.0f) / 1000.0f);
                o["deferred"] = sfx::JV(a.deferred);
                if (!a.reason.empty()) o["reason"] = sfx::JV(a.reason);
                ans[a.qid] = sfx::JV(o);
            }
            sfx::JVObj usage{
                {"state_tokens", static_cast<double>(u.state_tokens)},
                {"vocabulary", static_cast<double>(u.vocabulary)},
                {"lanes", static_cast<double>(u.lanes)},
                {"settled_energy", std::round(u.settled_energy * 1000.0f) / 1000.0f},
                {"calibrated", sfx::JV(u.calibrated)},
                {"engine", std::string("syfox-") + syfox::VERSION},
                {"core", "si-substrate"}};
            out = sfx::JV(sfx::JVObj{{"answers", sfx::JV(ans)}, {"usage", sfx::JV(usage)}}).dump();
        }
    } catch (const std::exception& e) {
        out = std::string("{\"error\":\"") + e.what() + "\"}";
    }
    char* buf = static_cast<char*>(std::malloc(out.size() + 1));
    std::memcpy(buf, out.c_str(), out.size() + 1);
    return buf;
}

void syfox_string_free(char* s) { std::free(s); }

} // extern "C"
