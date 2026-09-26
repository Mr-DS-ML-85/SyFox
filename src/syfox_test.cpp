// ============================================================================
//  SyFox tests — assert-based, no framework.
//  Every test names the mechanism it protects.
// ============================================================================
#include "core/syfox.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

static int failures = 0;
#define CHECK(cond, msg)                                                                  \
    do {                                                                                  \
        if (cond) { std::cout << "  ok  " << msg << "\n"; }                               \
        else { std::cout << "  FAIL " << msg << "\n"; ++failures; }                       \
    } while (0)

static void test_json() {
    std::cout << "[json]\n";
    auto v = sfx::JV::parse(R"({"a":1,"b":[true,null,"x\ny"],"c":{"d":-2.5e1}})");
    CHECK(v.is_obj(), "object parses");
    CHECK(v.at("a").as_num() == 1.0, "number");
    CHECK(v.at("b").is_arr() && v.at("b").arr.size() == 3, "array");
    CHECK(v.at("b").arr[2].str == "x\ny", "string escape");
    CHECK(std::fabs(v.at("c").at("d").as_num() + 25.0) < 1e-9, "exponent");
    std::string dumped = v.dump();
    auto v2 = sfx::JV::parse(dumped);
    CHECK(v2.dump() == dumped, "roundtrip stable");
    bool threw = false;
    try { sfx::JV::parse("{\"a\":}"); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "malformed input rejected");
}

static void test_folding() {
    std::cout << "[tokenize]\n";
    CHECK(si::tokenize("Refunds delayed!!")[0] == "refund", "plural folded");
    CHECK(si::tokenize("Charged twice.")[0] == "charg", "past folded");
    CHECK(si::tokenize("charge")[0] == "charg", "verb folded to same stem");
    CHECK(si::tokenize("hello")[0] == "hello", "plain word untouched");
    CHECK(si::tokenize("a x7").size() == 1, "single chars dropped");
}

static void test_field_physics() {
    std::cout << "[si substrate]\n";
    si::SubstrateConfig cfg;
    si::Substrate s(cfg);
    for (const auto& w : std::vector<std::string>{"alpha", "beta", "gamma"}) s.intern(w);
    CHECK(s.node_count() == 3, "vocabulary interns");
    s.bind(s.find("alpha"), s.find("beta"), 1.0f);
    s.inject({"alpha"});
    float before = s.total_energy();
    s.settle();
    float after = s.total_energy();
    CHECK(after <= before + 1e-4f, "settle is dissipative (energy never grows)");
    CHECK(s.readout({"beta"}) > 0.0f, "energy flowed along the lane to beta");
    CHECK(s.readout({"gamma"}) == 0.0f, "unconnected node stays dark");

    // lane cap: binding many partners evicts the weakest, bounded memory
    for (int i = 0; i < 400; ++i) {
        s.intern("pad" + std::to_string(i));
        s.bind(s.find("alpha"), s.find("pad" + std::to_string(i)), 0.01f);
    }
    CHECK(s.readout_neighbours(s.find("alpha")) <= s.config().lane_cap, "lane cap enforced");

    // honest silence on unknown vocabulary
    si::Substrate s2;
    for (const auto& w : std::vector<std::string>{"known", "words"}) s2.intern(w);
    s2.inject(si::tokenize("totally unknown zebra words fail"));
    s2.settle();
    CHECK(s2.total_energy() == 0.0f, "unknown tokens inject nothing (honest silence path)");

    // persistence roundtrip
    s.save("/tmp/syfox_test_substrate.bin");
    si::Substrate s3;
    s3.load("/tmp/syfox_test_substrate.bin");
    CHECK(s3.node_count() == s.node_count(), "load restores vocabulary");
    CHECK(s3.lane_count() == s.lane_count(), "load restores lane structure");
    // same injection + settle must read out identically on the restored substrate
    s.reset_field();   s.inject({"alpha"});   s.settle();
    s3.reset_field();  s3.inject({"alpha"});  s3.settle();
    CHECK(std::fabs(s3.readout({"beta"}) - s.readout({"beta"})) < 1e-5f,
          "restored substrate settles identically");
}

static syfox::Engine train_choice_engine() {
    syfox::Engine eng;
    const char* ex =
        R"({"state":"the bot sees zombies at night with low health","questions":{"action":{"type":"choice","instructions":"next move","criteria":{"flee":"run away escape avoid","fight":"attack sword combat","dig_in":"hide build shelter"}}},"labels":{"action":"flee"}})";
    auto rows = std::vector<sfx::JV>{sfx::JV::parse(ex)};
    for (const auto& r : rows) {
        eng.learn_example(r.at("state").as_str(), "next move",
                          "flee run away escape avoid");
    }
    return eng;
}

static void test_hebbian_choice() {
    std::cout << "[hebbian choice]\n";
    syfox::Engine eng = train_choice_engine();
    sfx::JV q = sfx::JV::parse(
        R"({"action":{"type":"choice","instructions":"next move","criteria":{"flee":"run away escape avoid","fight":"attack sword combat","dig_in":"hide build shelter"}}})");
    syfox::Usage u;
    auto ans = eng.decide("zombies appear at night, health is low", q, u);
    CHECK(!ans.empty() && ans[0].choice == "flee", "learned route fires on related state");
    CHECK(!ans[0].deferred, "field settled above silence floor");
    // a state sharing nothing with the trained vocabulary -> honest silence,
    // NOT a random guess (this is the whole point of the substrate)
    auto ans2 = eng.decide("completely different harvest moon farming", q, u);
    CHECK(!ans2.empty() && ans2[0].deferred, "unrelated state defers instead of guessing");
}

static void test_noul_valence() {
    std::cout << "[noul valence]\n";
    syfox::Engine eng;
    const std::string instr = "the command is irreversible or destructive";
    // true evidence: destructive vocabulary supports the statement
    eng.learn_noul("command rm -rf wipe the disk force delete", instr, true);
    eng.learn_noul("command drop table database destroy data", instr, true);
    // false evidence: harmless vocabulary weakens the route
    eng.learn_noul("command list files show status read only", instr, false);
    eng.learn_noul("command print the config view logs", instr, false);
    sfx::JV q = sfx::JV::parse(
        R"({"d":{"type":"noul","instructions":"the command is irreversible or destructive"}})");
    syfox::Usage u;
    float p_destructive = eng.decide("the command is rm -rf, it will delete everything", q, u)[0].probability;
    float p_harmless = eng.decide("the command lists open issues", q, u)[0].probability;
    CHECK(p_destructive > p_harmless, "field separates destructive from harmless");
    CHECK(p_harmless < 0.5f, "harmless command reads as false");
}

static void test_calibration_tool() {
    std::cout << "[calibration tool]\n";
    std::vector<syfox::Engine::CalibRow> rows = {
        {"choice", 2.0f, 0.2f, 1}, {"choice", 1.5f, 0.4f, 1},
        {"choice", 0.1f, 1.8f, 0}, {"choice", 0.3f, 1.2f, 0},
    };
    syfox::Engine eng;
    eng.fit_calibration(rows);
    CHECK(eng.calibration().fitted, "fitted flag set");
    CHECK(eng.calibration().choice_temperature > 0.0f, "temperature positive");
}

static void test_honest_silence_defer() {
    std::cout << "[honest silence]\n";
    syfox::Engine eng;
    sfx::JV q = sfx::JV::parse(R"({"x":{"type":"choice","instructions":"pick","criteria":{"a":"alpha","b":"beta"}}})");
    syfox::Usage u;
    auto ans = eng.decide("zzz qqq xxxttt", q, u);   // vocabulary is empty
    CHECK(ans[0].deferred && ans[0].reason == "unknown_vocabulary", "unknown vocabulary defers, never guesses");
}

int main() {
    std::cout << "SyFox test suite (core: si-substrate)\n";
    test_json();
    test_folding();
    test_field_physics();
    test_hebbian_choice();
    test_noul_valence();
    test_calibration_tool();
    test_honest_silence_defer();
    if (failures) { std::cout << failures << " FAILURES\n"; return 1; }
    std::cout << "all tests passed\n";
    return 0;
}
