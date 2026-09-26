// ============================================================================
//  SyFox — gate.hpp
//  The no-regression gate: transactional derivation.
//
//  The v0.1 honest verdict (ARCHITECTURE §10) was: derive sharpens
//  well-separated domains but FLIPS close-call scenarios on small fabrics,
//  so seed models shipped un-derived and adoption waited on an eval harness.
//  This file IS that harness, promoted into the mechanism itself:
//
//    1. replay every gate row + generated close-call probes  -> baseline
//    2. snapshot the lane fabric (order-preserving, bit-exact)
//    3. run the derivation (compose or harvest)
//    4. replay the same probe set                            -> after
//    5. any argmax flip (or deferral change) => restore the snapshot
//
//  After a gate run the invariant holds in BOTH branches:
//      committed  => the fabric changed, and no replayed decision flipped
//      reverted   => the fabric is bit-identical, decide() is byte-identical
//
//  The gate is the substrate's own honest silence, lifted one level: "the
//  field does not write into stone what it cannot prove harmless."
// ============================================================================
#pragma once
#include "syfox.hpp"
#include "derive.hpp"
#include "bench.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace syfox {
namespace gate {

struct GateConfig {
    int  max_probes    = 64;   // cap on generated close-call probes
    long allowed_flips = 0;    // policy: zero tolerance by default
};

struct GateReport {
    bool ran            = false;
    bool committed      = false;
    long taught_probes  = 0;
    long mixed_probes   = 0;
    long taught_flips   = 0;
    long mixed_flips    = 0;      // informational: ambiguous states may re-resolve
    bool conf_inflated  = false;  // mixed-state mean confidence rose => false certainty
    long changed        = 0;      // lanes created/strengthened by the derivation
    long removed        = 0;      // lanes healed/dissolved by the derivation
    double mean_abs_dconf = 0.0;  // mean |delta confidence| over non-flipped (taught)
    double mean_conf_mixed_before = 0.0;
    double mean_conf_mixed_after  = 0.0;
    std::vector<std::string> flip_examples;   // up to 5, "qid: A -> B"

    sfx::JV to_json() const {
        const auto r4 = [](double v) { return std::round(v * 10000.0) / 10000.0; };
        sfx::JVArr ex;
        for (const auto& e : flip_examples) ex.push_back(sfx::JV(e));
        return sfx::JV(sfx::JVObj{
            {"ran", sfx::JV(ran)},
            {"committed", sfx::JV(committed)},
            {"taught_probes", static_cast<double>(taught_probes)},
            {"mixed_probes", static_cast<double>(mixed_probes)},
            {"taught_flips", static_cast<double>(taught_flips)},
            {"mixed_flips", static_cast<double>(mixed_flips)},
            {"conf_inflated", sfx::JV(conf_inflated)},
            {"mean_conf_mixed_before", r4(mean_conf_mixed_before)},
            {"mean_conf_mixed_after", r4(mean_conf_mixed_after)},
            {"mean_abs_dconf", r4(mean_abs_dconf)},
            {"flip_examples", sfx::JV(ex)}});
    }
};

namespace detail {

// one probe signature: qid -> (argmax or "" when deferred, confidence)
using Sig = std::map<std::string, std::pair<std::string, float>>;

inline long count_flips(const Sig& before, const Sig& after,
                        long& flips, double& dconf_sum, long& dconf_n,
                        std::vector<std::string>& examples,
                        const std::string& state_head) {
    long f = 0;
    for (const auto& kv : before) {
        auto it = after.find(kv.first);
        if (it == after.end()) continue;
        const bool was_deferred = kv.second.first.empty();
        const bool now_deferred = it->second.first.empty();
        if (was_deferred != now_deferred || kv.second.first != it->second.first) {
            ++f;
            if (static_cast<long>(examples.size()) < 5) {
                examples.push_back(state_head + " | " + kv.first + ": "
                                   + (was_deferred ? std::string("(deferred)") : kv.second.first)
                                   + " -> "
                                   + (now_deferred ? std::string("(deferred)") : it->second.first));
            }
        } else if (!was_deferred) {
            dconf_sum += std::fabs(static_cast<double>(kv.second.second) -
                                   static_cast<double>(it->second.second));
            ++dconf_n;
        }
    }
    flips += f;
    return f;
}

} // namespace detail

// Run a derivation under the gate. `mode` is "compose" or "harvest".
// `gate_rows` are labelled eval rows (taught states + close-call generator
// input). `harvest_replay` are the tokenized states the harvest mode replays
// (usually the gate rows' states). Derivation configs are caller-chosen; the
// conservative presets in derive.hpp are the recommended starting point.
// Returns the report; the CALLER saves the model only when report.committed
// is true.
inline GateReport gated_derive(Engine& eng, const std::string& mode,
                               const std::vector<sfx::JV>& gate_rows,
                               const std::vector<std::vector<std::string>>& harvest_replay,
                               const GateConfig& gc = {},
                               const derive::HarvestConfig& hc = derive::HarvestConfig{},
                               const derive::DeriveConfig&  dc = derive::DeriveConfig{}) {
    GateReport rep;
    rep.ran = true;

    // -- 1. probe plan: taught rows + generated close-call states -------------
    std::vector<bench::Probe> probes = bench::eval_probes(gate_rows);
    rep.taught_probes = static_cast<long>(probes.size());
    const std::vector<bench::Probe> mixed = bench::closecall_probes(gate_rows, gc.max_probes);
    rep.mixed_probes = static_cast<long>(mixed.size());
    probes.insert(probes.end(), mixed.begin(), mixed.end());
    if (probes.empty()) return rep;                 // nothing to protect

    // -- 2. baseline signatures ------------------------------------------------
    std::vector<detail::Sig> before;
    before.reserve(probes.size());
    for (const auto& p : probes) before.push_back(bench::decision_signature(eng, p));

    // -- 3. snapshot + derive ----------------------------------------------------
    const si::Substrate::FabricSnapshot snap = eng.substrate().snapshot_fabric();
    long changed = 0, removed = 0;
    if (mode == "harvest") {
        const derive::HarvestStats st = derive::harvest(eng.substrate(), harvest_replay, hc);
        changed = st.created + st.refreshed;
        removed = st.dissolved;
    } else {
        const derive::DeriveStats st = derive::run(eng.substrate(), dc, 2);
        changed = st.created + st.strengthened;
        removed = st.healed + st.dissolved;
    }
    rep.changed = changed;
    rep.removed = removed;

    // -- 4. after signatures + flip count ---------------------------------------
    // Policy (v0.2, refined by the strength-scan experiments):
    //   * TAUGHT rows are labelled knowledge: ANY argmax flip is a regression.
    //   * MIXED close-call probes are ambiguous BY CONSTRUCTION (two different
    //     domains concatenated) — their argmax may re-resolve either way, but
    //     derivation must not manufacture certainty about them: if the mixed
    //     mean confidence RISES, the fabric is feigning sureness => revert.
    long taught_flips = 0, mixed_flips = 0;
    double dconf_sum = 0.0; long dconf_n = 0;
    double mixed_conf_before = 0.0, mixed_conf_after = 0.0;
    long mixed_conf_n = 0;
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const bool is_mixed = i >= static_cast<std::size_t>(rep.taught_probes);
        const detail::Sig after = bench::decision_signature(eng, probes[i]);
        // mixed-state confidence accounting (non-deferred entries)
        if (is_mixed) {
            for (const auto& kv : before[i]) {
                auto it = after.find(kv.first);
                if (it == after.end()) continue;
                if (kv.second.first.empty() || it->second.first.empty()) continue;
                mixed_conf_before += kv.second.second;
                mixed_conf_after  += it->second.second;
                ++mixed_conf_n;
            }
        }
        const long f = detail::count_flips(before[i], after,
                                           is_mixed ? mixed_flips : taught_flips,
                                           dconf_sum, dconf_n, rep.flip_examples,
                                           probes[i].state.substr(0, 40));
        (void)f;
    }
    rep.taught_flips = taught_flips;
    rep.mixed_flips  = mixed_flips;
    if (mixed_conf_n > 0) {
        rep.mean_conf_mixed_before = mixed_conf_before / mixed_conf_n;
        rep.mean_conf_mixed_after  = mixed_conf_after  / mixed_conf_n;
    }
    if (dconf_n > 0) rep.mean_abs_dconf = dconf_sum / dconf_n;
    rep.conf_inflated = mixed_conf_n > 0 &&
        rep.mean_conf_mixed_after > rep.mean_conf_mixed_before + 1e-6;

    // -- 5. commit or revert -----------------------------------------------------
    if (taught_flips > gc.allowed_flips || rep.conf_inflated) {
        eng.substrate().restore_fabric(snap);
        rep.committed = false;
    } else {
        rep.committed = true;
    }
    return rep;
}

} // namespace gate
} // namespace syfox
