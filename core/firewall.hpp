// ============================================================================
//  SyFox — firewall.hpp  (Milestone 1: the hidden-test contract, in code)
// ---------------------------------------------------------------------------
//  Milestone-1 discipline: every generated corpus splits 70/15/15 into
//  train / calibration / hidden test, and the hidden test NEVER participates
//  in training, calibration, derivation, or model selection. That contract
//  is easy to state in a README and easier to break by typo, so it is
//  ENFORCED here: the split a file belongs to is decided by its filename
//  (the same <domain>_<split>.jsonl convention split_data.py and the CLI
//  already use), and the learn / calibrate / derive commands consult this
//  table before reading anything.
//
//    role        filename suffix     learn   calibrate   derive --gate   bench
//    train       _train.jsonl        YES     yes         yes             yes
//    heldout     _heldout.jsonl      yes     YES (v2.1    yes            yes
//                                            recipe)
//    cal         _cal.jsonl          no      YES         yes             yes
//    hidden      _hidden.jsonl       NO      NO          NO              YES
//    unknown     anything else       yes     yes         yes             yes
//                (worksheets, paraphrase files, deferral logs, memories)
//
//  Rationale: calibration files exist to fit 2-3 scalars (temperature /
//  Platt) on rows the fabric never learned from. Hidden files exist to be
//  SCORED, once, by bench. A hidden row that leaks into any build path is
//  not a mistake, it is a different (and worthless) number.
//  Zero dependencies. Deterministic. C++17.
// ============================================================================
#pragma once
#include <string>

namespace syfox {
namespace firewall {

enum class Role { Train, Heldout, Cal, Hidden, Unknown };

inline Role role_of_path(const std::string& path) {
    // basename only: the split lives in the filename, not the directory
    std::string base = path;
    const auto slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    auto ends_with = [&base](const char* suffix) {
        const std::size_t n = std::char_traits<char>::length(suffix);
        return base.size() >= n && base.compare(base.size() - n, n, suffix) == 0;
    };
    if (ends_with("_hidden.jsonl"))  return Role::Hidden;
    if (ends_with("_cal.jsonl"))     return Role::Cal;
    if (ends_with("_heldout.jsonl")) return Role::Heldout;
    if (ends_with("_train.jsonl"))   return Role::Train;
    return Role::Unknown;
}

inline const char* role_name(Role r) {
    switch (r) {
        case Role::Train:   return "train";
        case Role::Heldout: return "heldout";
        case Role::Cal:     return "calibration";
        case Role::Hidden:  return "hidden";
        default:            return "unknown";
    }
}

// syfox learn: train rows and legacy heldout rows and anything unclassified
// (paraphrase files, labeling worksheets). NOT calibration, NEVER hidden.
inline bool learn_may_read(const std::string& path) {
    const Role r = role_of_path(path);
    return r != Role::Hidden && r != Role::Cal;
}

// syfox calibrate: anything except the hidden test. The v2.1 recipe fits on
// the held-out split; the v3 recipe fits on the dedicated _cal split.
inline bool calibrate_may_read(const std::string& path) {
    return role_of_path(path) != Role::Hidden;
}

// derive --gate: the no-regression gate replays eval rows. Hidden rows must
// not steer derivation (model selection), so they are refused here too.
inline bool derive_gate_may_read(const std::string& path) {
    return role_of_path(path) != Role::Hidden;
}

} // namespace firewall
} // namespace syfox
