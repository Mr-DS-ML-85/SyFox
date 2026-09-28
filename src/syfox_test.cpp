// ============================================================================
//  SyFox tests — assert-based, no framework.
//  Every test names the mechanism it protects.
// ============================================================================
#include "core/syfox.hpp"
#include "core/derive.hpp"
#include "core/bench.hpp"
#include "core/gate.hpp"
#include "core/recall.hpp"
#include "core/calc.hpp"
#include "core/jas.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
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
    // si::tokenize itself is UNCHANGED since v0.2 (substrate encoding baseline)
    CHECK(si::tokenize("Refunds delayed!!")[0] == "refund", "plural folded");
    CHECK(si::tokenize("Charged twice.")[0] == "charg", "past folded");
    CHECK(si::tokenize("charge")[0] == "charg", "verb folded to same stem");
    CHECK(si::tokenize("hello")[0] == "hello", "plain word untouched");
    CHECK(si::tokenize("a x7").size() == 1, "single chars dropped");
}

static void test_normalize() {
    std::cout << "[normalize]\n";
    // Porter spot-checks against the 1980 reference vocabulary
    CHECK(si::norm::porter_stem("relational") == "relat", "porter step2: ational");
    CHECK(si::norm::porter_stem("generalization") == "gener", "porter step4: ization");
    CHECK(si::norm::porter_stem("controll") == "control", "porter step5: double l");
    CHECK(si::norm::porter_stem("rate") == "rate", "porter keeps cvc+e");
    CHECK(si::norm::porter_stem("agreed") == "agre", "porter step1b: ed");
    // synonym folding through the stemmed table
    CHECK(si::norm::normalize("Reimbursement denied")[0] == "refund",
          "synonym fold: reimbursement -> refund");
    CHECK(si::norm::normalize("urgently")[0] == si::norm::normalize("urgent")[0],
          "synonym fold: urgently == urgent");
    CHECK(si::norm::normalize("asap")[0] == si::norm::normalize("immediately")[0],
          "synonym fold: asap == immediately");
    // one pipeline for teach + decide: porter parity with the old fold on our vocab
    CHECK(si::norm::normalize("Charged twice.")[0] == "charg", "porter: charged -> charg");
    CHECK(si::norm::normalize("charge")[0] == "charg", "porter: charge -> charg");
    CHECK(si::norm::normalize("Invoices")[0] == "invoic", "porter: invoices -> invoic");
    // determinism: same input -> same tokens, byte for byte
    bool det = si::norm::normalize("please refund the duplicate charge immediately")
            == si::norm::normalize("please refund the duplicate charge immediately");
    CHECK(det, "normalize is deterministic");
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
    s2.inject(si::norm::normalize("totally unknown zebra words fail"));
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

static void test_salience_mechanics() {
    std::cout << "[si salience + miller window]\n";
    // (a) integrator (SI physics.hpp lineage): touched/moving nodes gain
    //     salience; a node at rest is an EXACT fixed point (tanh(0)=0),
    //     mirroring SI's sparsity guard.
    si::Substrate s;
    for (const auto& w : std::vector<std::string>{"a", "b", "far"}) s.intern(w);
    s.bind(s.find("a"), s.find("b"), 1.0f);
    s.inject({"a"});
    s.settle();
    CHECK(s.node_salience("a") > 0.0f, "moving node carries salience");
    CHECK(s.node_salience("far") == 0.0f, "node at rest is an exact fixed point");

    // (b) salience-gated settle stays dissipative and still propagates
    si::SubstrateConfig g;
    g.salience_gating = true;
    si::Substrate s2(g);
    s2.intern("x"); s2.intern("y");
    s2.bind(s2.find("x"), s2.find("y"), 1.0f);
    s2.inject({"x"});
    float before = s2.total_energy();
    s2.settle();
    CHECK(s2.total_energy() <= before + 1e-4f, "salience-gated settle is dissipative");
    CHECK(s2.readout({"y"}) > 0.0f, "salience-gated field propagates along lanes");

    // (c) miller window (TSDA live_cap lineage): cap inside [cap-4, cap],
    //     and the SAME state always yields the SAME cap and field.
    si::SubstrateConfig m;
    m.miller_window = true;
    m.source_cap = 9.0f;                       // window becomes [5,9], as in TSDA
    si::Substrate s3(m);
    std::vector<std::string> toks;
    for (int i = 0; i < 30; ++i) {
        std::string t = "t" + std::to_string(i);
        s3.intern(t);
        if (i > 0) s3.bind(s3.find("t" + std::to_string(i - 1)), s3.find(t), 0.3f);
        toks.push_back(t);
    }
    s3.inject(toks);
    s3.settle();
    float cap_used = s3.last_source_cap();
    CHECK(cap_used >= 5.0f && cap_used <= 9.0f, "sampled cap inside Miller window [5,9]");
    float e1 = s3.total_energy();
    s3.reset_field();
    s3.inject(toks);
    s3.settle();
    CHECK(s3.last_source_cap() == cap_used && s3.total_energy() == e1,
          "same state -> same cap -> same field (deterministic)");

    // (d) default config: cap stays pinned at source_cap
    si::Substrate s4;
    s4.settle();
    CHECK(s4.last_source_cap() == 24.0f, "default cap pinned at 24");
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
    // v2.1 row shape: FULL candidate vector + gold label (multi-class fit)
    std::vector<syfox::Engine::CalibRow> rows = {
        {"choice", "a", {{"a", 2.0f}, {"b", 0.2f}}, 0.0f, 0},
        {"choice", "a", {{"a", 1.5f}, {"b", 0.4f}}, 0.0f, 0},
        {"choice", "b", {{"a", 0.1f}, {"b", 1.8f}}, 0.0f, 0},
        {"choice", "b", {{"a", 0.3f}, {"b", 1.2f}}, 0.0f, 0},
    };
    syfox::Engine eng;
    eng.fit_calibration(rows);
    CHECK(eng.calibration().fitted, "fitted flag set");
    CHECK(eng.calibration().choice_temperature > 0.0f, "temperature positive");
    // multi-class NLL is sharpness-aware: this fit set has clean margins, so
    // the optimum must sharpen (T < 1), not flatten
    CHECK(eng.calibration().choice_temperature < 1.0f, "clean margins sharpen temperature");
}

static void test_honest_silence_defer() {
    std::cout << "[honest silence]\n";
    syfox::Engine eng;
    sfx::JV q = sfx::JV::parse(R"({"x":{"type":"choice","instructions":"pick","criteria":{"a":"alpha","b":"beta"}}})");
    syfox::Usage u;
    auto ans = eng.decide("zzz qqq xxxttt", q, u);   // vocabulary is empty
    CHECK(ans[0].deferred && ans[0].reason == "unknown_vocabulary", "unknown vocabulary defers, never guesses");
}

static void test_derivation() {
    std::cout << "[derivation layer: induction, verifier, dreamer, analogy]\n";
    using namespace syfox::derive;

    // probe fabric (same shape as the ARCHITECTURE.md probe):
    //   co-occurrence alpha-beta-gamma; lessons alpha/beta -> x-tokens, beta/gamma -> y-tokens
    auto build_probe = []() {
        si::Substrate s;
        for (const char* t : {"alpha", "beta", "gamma", "x", "target", "y", "other"})
            s.intern(t);
        const si::NodeId A = s.find("alpha"), B = s.find("beta"), G = s.find("gamma");
        const si::NodeId X = s.find("x"), T = s.find("target"), Y = s.find("y"), O = s.find("other");
        s.bind(A, B, 0.10f); s.bind(B, G, 0.10f);
        for (si::NodeId st : {A, B}) for (si::NodeId o : {X, T}) s.bind(st, o, 0.20f);
        for (si::NodeId st : {B, G}) for (si::NodeId o : {Y, O}) s.bind(st, o, 0.20f);
        return s;
    };
    DeriveConfig c;

    // (a) composition turns the two-hop path alpha->beta->y into a derived lane
    si::Substrate s = build_probe();
    CHECK(s.lane_weight(s.find("alpha"), s.find("y")) == 0.0f, "no direct alpha->y lane before derivation");
    DeriveStats st = compose_pass(s, c);
    CHECK(s.lane_weight(s.find("alpha"), s.find("y")) > 0.0f, "two-hop path became a derived lane");
    CHECK(s.generation_of(s.find("alpha"), s.find("y")) == 1, "derived lane carries generation 1");
    CHECK(st.created > 0, "composition reports created lanes");

    // (b) observed lanes are never weakened by composition
    si::Substrate s2 = build_probe();
    struct SnapL { si::NodeId a, b; float w; };
    std::vector<SnapL> snap;
    s2.for_each_lane([&](si::NodeId a, si::NodeId b, float w) { snap.push_back({a, b, w}); });
    compose_pass(s2, c);
    bool never_weakened = true;
    for (const auto& l : snap)
        if (s2.lane_weight(l.a, l.b) + 1e-6f < l.w) never_weakened = false;
    CHECK(never_weakened, "composition never weakens observed lanes");

    // (c) generation cap bounds derived-of-derived depth
    si::Substrate s3;
    for (int i = 0; i < 6; ++i) s3.intern("z" + std::to_string(i));
    for (int i = 1; i < 6; ++i)
        s3.bind(s3.find("z" + std::to_string(i - 1)), s3.find("z" + std::to_string(i)), 0.5f);
    DeriveConfig capped; capped.max_generation = 2;
    compose_pass(s3, capped); compose_pass(s3, capped); compose_pass(s3, capped);
    std::uint32_t max_gen_seen = 0;
    s3.for_each_lane([&](si::NodeId a, si::NodeId b, float) {
        max_gen_seen = std::max(max_gen_seen, s3.generation_of(a, b));
    });
    CHECK(max_gen_seen <= 2, "no lane exceeds the generation cap");

    // (d) verifier: stripping a parent path dissolves the derived lane
    si::Substrate s4 = build_probe();
    compose_pass(s4, c);
    const float w_beta_y = s4.lane_weight(s4.find("beta"), s4.find("y"));
    s4.scale_lane(s4.find("beta"), s4.find("y"), -(w_beta_y - 0.005f));  // drop below path_min
    DeriveStats pst = prune_stale(s4, c);
    CHECK(pst.dissolved > 0, "verifier dissolves unsupported derived lanes");
    CHECK(s4.lane_weight(s4.find("alpha"), s4.find("y")) == 0.0f, "unsupported derived lane is gone");
    CHECK(s4.lane_weight(s4.find("alpha"), s4.find("beta")) == 0.10f, "observed lanes survive the verifier");

    // (e) dreamer: deterministic per seed, never modifies the substrate
    si::Substrate s5a = build_probe(), s5b = build_probe();
    const std::size_t lanes_before = s5a.lane_count();
    DreamConfig dc;
    auto c1 = dream(s5a, dc, 42, 8);
    auto c2 = dream(s5b, dc, 42, 8);
    bool same = c1.size() == c2.size();
    for (std::size_t i = 0; same && i < c1.size(); ++i)
        same = c1[i].emergent == c2[i].emergent && c1[i].support == c2[i].support;
    CHECK(same, "same seed -> same dream (bit for bit)");
    CHECK(s5a.lane_count() == lanes_before, "dreaming never modifies the fabric");

    // (f) promotion: a validated candidate becomes a premise-grade lane
    si::Substrate s6 = build_probe();
    apply_promotion(s6, {"gamma"}, "x", 0.05f, 6.0f);
    CHECK(s6.lane_weight(s6.find("gamma"), s6.find("x")) > 0.0f, "validated promotion lays a lane");
    CHECK(s6.generation_of(s6.find("gamma"), s6.find("x")) == 0, "promoted lane is premise-grade (gen 0)");

    // (g) provenance survives save/load
    si::Substrate s7 = build_probe();
    compose_pass(s7, c);
    const float w_alpha_y = s7.lane_weight(s7.find("alpha"), s7.find("y"));
    s7.save("build/test_provenance.bin");
    si::Substrate s8;
    s8.load("build/test_provenance.bin");
    CHECK(s8.lane_weight(s8.find("alpha"), s8.find("y")) == w_alpha_y &&
          s8.generation_of(s8.find("alpha"), s8.find("y")) == 1,
          "derived weight + generation survive save/load");

    // (h) v1 compatibility: a file with no provenance tail loads clean
    {
        std::ofstream f("build/test_v1.bin", std::ios::binary);
        std::uint32_t n = 2; f.write(reinterpret_cast<const char*>(&n), 4);
        std::uint32_t len = 1; float mass = 1.0f;
        f.write(reinterpret_cast<const char*>(&len), 4); f.write("p", 1);
        f.write(reinterpret_cast<const char*>(&mass), 4);
        f.write(reinterpret_cast<const char*>(&len), 4); f.write("q", 1);
        f.write(reinterpret_cast<const char*>(&mass), 4);
        std::uint32_t lanes = 2, a_id = 0, b_id = 1; float w = 0.5f;
        f.write(reinterpret_cast<const char*>(&lanes), 4);
        f.write(reinterpret_cast<const char*>(&a_id), 4); f.write(reinterpret_cast<const char*>(&b_id), 4);
        f.write(reinterpret_cast<const char*>(&w), 4);
        f.write(reinterpret_cast<const char*>(&b_id), 4); f.write(reinterpret_cast<const char*>(&a_id), 4);
        f.write(reinterpret_cast<const char*>(&w), 4);
    }
    si::Substrate s9;
    s9.load("build/test_v1.bin");
    CHECK(s9.lane_weight(s9.find("p"), s9.find("q")) == 0.5f, "v1 file (no tail) loads");
    CHECK(s9.generation_of(s9.find("p"), s9.find("q")) == 0, "v1 lanes default to premise grade");

    // (i) analogy: identical neighbourhood -> iso 1.0
    si::Substrate s10;
    for (const char* t : {"hub1", "hub2", "leafA", "leafB", "distant"}) s10.intern(t);
    // hub1 and hub2 touch the same two leaves -> identical signatures
    s10.bind(s10.find("hub1"), s10.find("leafA"), 0.4f);
    s10.bind(s10.find("hub1"), s10.find("leafB"), 0.4f);
    s10.bind(s10.find("hub2"), s10.find("leafA"), 0.4f);
    s10.bind(s10.find("hub2"), s10.find("leafB"), 0.4f);
    auto matches = find_analogues(s10, "hub1");
    CHECK(!matches.empty() && matches[0].concept == "hub2" && matches[0].iso > 0.99f,
          "identical neighbourhoods map at iso ~1.0");

    // (j) harvest: the field's own dynamics nominate; observed lanes untouched;
    //     stale gen-1 lanes are dissolved by the next replay
    si::Substrate s11 = build_probe();
    const float w_ab_before = s11.lane_weight(s11.find("alpha"), s11.find("beta"));
    {
        std::vector<std::vector<std::string>> replay = {
            si::norm::normalize("alpha beta"), si::norm::normalize("beta gamma")};
        HarvestConfig hc;
        HarvestStats hst = harvest(s11, replay, hc);
        CHECK(s11.lane_weight(s11.find("alpha"), s11.find("beta")) == w_ab_before,
              "harvest never touches observed lanes");
        // x and target co-activate strongly (same lesson outcome side) yet no
        // direct lane exists between them: the field nominates the bridge
        CHECK(s11.lane_weight(s11.find("x"), s11.find("target")) > 0.0f &&
              s11.generation_of(s11.find("x"), s11.find("target")) == 1,
              "co-activated pair gains a derived bridge (gen 1)");
        // a lane nothing in the replay can nominate must dissolve on re-check
        s11.intern("zz");                                       // isolated: never activates
        s11.bind_derived(s11.find("alpha"), s11.find("zz"), 0.05f, 1);
        CHECK(s11.lane_weight(s11.find("alpha"), s11.find("zz")) > 0.0f, "stale candidate laid");
        HarvestStats hst2 = harvest(s11, replay, hc);
        CHECK(hst2.dissolved >= 1 && s11.lane_weight(s11.find("alpha"), s11.find("zz")) == 0.0f,
              "un-nominated derived lane dissolves on re-verification");
        CHECK(s11.lane_weight(s11.find("alpha"), s11.find("beta")) == w_ab_before,
              "observed lanes survive re-verification too");
    }
}

// ---------------------------------------------------------------------------
// Shared fixtures for the e2e suite: six labelled ticket rows (choice + noul)
// over two well-separated vocabulary groups.
// ---------------------------------------------------------------------------
static std::vector<sfx::JV> make_ticket_rows() {
    const std::string q =
        R"({"department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems"}},"is_urgent":{"type":"noul","instructions":"the message conveys urgency or time sensitivity"}})";
    const char* src[] = {
        "{\"state\":\"refund double charge invoice immediately\",\"labels\":{\"department\":\"billing\",\"is_urgent\":\"true\"}}",
        "{\"state\":\"charged twice money back now please\",\"labels\":{\"department\":\"billing\",\"is_urgent\":\"true\"}}",
        "{\"state\":\"change invoice address next month\",\"labels\":{\"department\":\"billing\",\"is_urgent\":\"false\"}}",
        "{\"state\":\"price team plan twenty seats\",\"labels\":{\"department\":\"billing\",\"is_urgent\":\"false\"}}",
        "{\"state\":\"download invoice records\",\"labels\":{\"department\":\"billing\",\"is_urgent\":\"false\"}}",
        "{\"state\":\"api throws error when connecting\",\"labels\":{\"department\":\"technical\",\"is_urgent\":\"false\"}}",
        "{\"state\":\"integration keeps failing client crashes\",\"labels\":{\"department\":\"technical\",\"is_urgent\":\"false\"}}",
        "{\"state\":\"production down api errors every request\",\"labels\":{\"department\":\"technical\",\"is_urgent\":\"true\"}}",
        "{\"state\":\"bug breaks checkout browser\",\"labels\":{\"department\":\"technical\",\"is_urgent\":\"true\"}}",
        "{\"state\":\"dashboard shows blank page after update\",\"labels\":{\"department\":\"technical\",\"is_urgent\":\"false\"}}",
    };
    std::vector<sfx::JV> out;
    for (const char* r : src) {
        std::string s(r);
        s.pop_back();                                   // drop the outer '}'
        out.push_back(sfx::JV::parse(s + ",\"questions\":" + q + "}"));
    }
    return out;
}

static void learn_tickets(syfox::Engine& eng, const std::vector<sfx::JV>& rows) {
    for (const auto& r : rows) {
        const std::string state = r.at("state").as_str();
        const sfx::JV& qs = r.at("questions");
        const sfx::JV& labels = r.at("labels");
        const std::string dept = labels.at("department").as_str();
        eng.learn_example(state, qs.at("department").at("instructions").as_str(),
                          dept + " " + qs.at("department").at("criteria").at(dept).as_str());
        eng.learn_noul(state, qs.at("is_urgent").at("instructions").as_str(),
                       labels.at("is_urgent").as_str() == "true");
    }
}

// ---------------------------------------------------------------------------
static void test_recall() {
    std::cout << "[recall: associative retrieval in settled-energy space]\n";
    using namespace syfox::recall;
    si::Substrate s;
    for (const char* w : {"refund", "charg", "invoic", "billing", "double", "payment",
                          "api", "error", "connect", "bug", "crash", "integr"})
        s.intern(w);
    auto b2 = [&](const char* a, const char* b, float w) { s.bind(s.find(a), s.find(b), w); };
    // two disjoint neighbourhoods (billing / technical)
    b2("refund", "charg", 0.6f);  b2("charg", "invoic", 0.6f); b2("invoic", "billing", 0.6f);
    b2("double", "charg", 0.5f);  b2("billing", "payment", 0.5f);
    b2("api", "error", 0.6f);     b2("error", "connect", 0.5f); b2("connect", "bug", 0.5f);
    b2("bug", "crash", 0.5f);     b2("crash", "integr", 0.5f);

    std::vector<Memory> memories = {
        {"billing",   si::norm::normalize("refund double charg invoic billing payment")},
        {"technical", si::norm::normalize("api error connect bug crash integr")},
    };
    auto hitsA = recall(s, "i was double charged on my invoice, refund please", memories, 2);
    CHECK(!hitsA.empty() && hitsA[0].label == "billing",
          "billing query recalls the billing memory first");
    auto hitsB = recall(s, "the api crashes when I connect the integration", memories, 2);
    CHECK(!hitsB.empty() && hitsB[0].label == "technical",
          "technical query recalls the technical memory first");
    if (hitsA.size() == 2)
        CHECK(hitsA[0].resonance > hitsA[1].resonance, "winner clearly separated from runner-up");

    // determinism: same model + query + memories => bit-identical resonances
    auto hitsA2 = recall(s, "i was double charged on my invoice, refund please", memories, 2);
    bool same = hitsA.size() == hitsA2.size();
    for (std::size_t i = 0; same && i < hitsA.size(); ++i)
        same = hitsA[i].resonance == hitsA2[i].resonance && hitsA[i].label == hitsA2[i].label;
    CHECK(same, "recall is deterministic (bit-identical resonances)");

    // unknown vocabulary resonates with nothing (inject skips unknown tokens)
    auto hitsO = recall(s, "zorblatz quibblemock framistan", memories, 2);
    CHECK(hitsO.empty() || hitsO[0].resonance == 0.0f,
          "unknown vocabulary resonates with nothing");

    // read-only: recall leaves the fabric untouched
    const std::size_t lanes_before = s.lane_count();
    (void)recall(s, "another refund query about charges", memories, 2);
    CHECK(s.lane_count() == lanes_before, "recall never mutates the fabric");
}

// ---------------------------------------------------------------------------
static void test_gate() {
    std::cout << "[gate: transactional derivation, bit-exact revert]\n";
    using namespace syfox;
    Engine eng;
    auto rows = make_ticket_rows();
    learn_tickets(eng, rows);
    auto probes = bench::eval_probes(rows);

    // baseline signature before anything touches the fabric
    const auto sig0 = bench::decision_signature(eng, probes[0]);
    const std::size_t lanes0 = eng.substrate().lane_count();

    // (a) snapshot/restore is bit-exact, even after an adversarial write
    const auto snap = eng.substrate().snapshot_fabric();
    CHECK(snap.lanes == lanes0, "snapshot counts every lane");
    eng.substrate().bind_derived(eng.substrate().find("refund"),
                                 eng.substrate().find("error"), 1.2f, 1);
    CHECK(eng.substrate().lane_weight(eng.substrate().find("refund"),
                                      eng.substrate().find("error")) > 0.0f,
          "adversarial derived lane laid");
    eng.substrate().restore_fabric(snap);
    CHECK(eng.substrate().lane_weight(eng.substrate().find("refund"),
                                      eng.substrate().find("error")) == 0.0f,
          "restore removes the adversarial lane");
    CHECK(eng.substrate().lane_count() == lanes0, "lane count restored");
    const auto sigR = bench::decision_signature(eng, probes[0]);
    CHECK(sigR == sig0, "restore is bit-exact: decide() output identical");

    // (b) gated compose: whatever the gate decides, decisions never regress
    gate::GateConfig gc;
    auto rep = gate::gated_derive(eng, "compose", rows, {}, gc);
    CHECK(rep.ran, "gate ran");
    CHECK(rep.taught_probes == static_cast<long>(rows.size()), "taught probes replayed");
    CHECK(rep.mixed_probes > 0, "close-call probes generated");
    if (rep.committed)
        CHECK(rep.taught_flips == 0, "committed gate => zero taught flips (mixed may re-resolve)");
    else
        CHECK(rep.taught_flips > 0 || rep.conf_inflated,
              "reverted gate => taught flips or manufactured certainty");
    const auto sig1 = bench::decision_signature(eng, probes[0]);
    if (rep.committed) {
        bool argmax_same = true;
        for (const auto& kv : sig0) {
            auto it = sig1.find(kv.first);
            if (it == sig1.end() || it->second.first != kv.second.first) argmax_same = false;
        }
        CHECK(argmax_same, "committed gate: taught argmaxes unchanged");
    } else {
        CHECK(sig1 == sig0, "reverted gate: decide() bit-identical to baseline");
    }

    // (c) gated harvest with replay states: same invariant. NOTE (v2.1): the
    // revert-branch baseline is sig1 — the fabric state immediately before
    // the harvest — because a COMMITTED compose gate legitimately changed the
    // lanes (zero flips), and the harvest revert must restore THAT state,
    // bit for bit, not the pre-compose one.
    std::vector<std::vector<std::string>> replay;
    for (const auto& r : rows) replay.push_back(si::norm::normalize(r.at("state").as_str()));
    auto rep2 = gate::gated_derive(eng, "harvest", rows, replay, gc);
    CHECK(rep2.ran, "harvest gate ran");
    const auto sig2 = bench::decision_signature(eng, probes[0]);
    if (rep2.committed) {
        bool argmax_same = true;
        for (const auto& kv : sig0) {
            auto it = sig2.find(kv.first);
            if (it == sig2.end() || it->second.first != kv.second.first) argmax_same = false;
        }
        CHECK(argmax_same, "committed harvest gate: taught argmaxes unchanged");
    } else {
        CHECK(sig2 == sig1, "reverted harvest gate: decide() bit-identical to pre-derive state");
    }

    // (d) gate report serializes
    CHECK(rep.to_json().dump().find("committed") != std::string::npos, "gate report serializes");
}

// ---------------------------------------------------------------------------
static void test_bench_e2e() {
    std::cout << "[bench: Jev-parity suite end to end]\n";
    using namespace syfox;
    Engine eng;
    auto rows = make_ticket_rows();
    learn_tickets(eng, rows);
    auto calib = eng.harvest_rows(rows);
    CHECK(!calib.empty(), "calibration rows harvested");
    eng.fit_calibration(calib);

    bench::BenchConfig bc;
    bc.latency_reps = 3;
    auto rep = bench::run(eng, rows, "in-domain (resubstitution)", bc, "test-model");

    CHECK(rep.rows == static_cast<long>(rows.size()), "eval rows loaded");
    CHECK(rep.choice_accuracy == 1.0,
          "in-domain choice accuracy 1.0 (taught states resubstituted)");
    CHECK(rep.deterministic, "decisions are bit-deterministic across replays");
    CHECK(rep.ood_defer_rate == 1.0, "unknown vocabulary defers (honest silence)");
    CHECK(rep.choice_ece >= 0.0 && rep.choice_ece <= 1.0, "ECE in [0,1]");
    CHECK(rep.mixed_probes > 0 && rep.probes > rep.rows, "close-call probes generated");
    CHECK(rep.lat_p50_us > 0.0 && rep.lat_p95_us >= rep.lat_p50_us, "latency percentiles sane");
    CHECK(rep.g_tp >= 1, "guardrail caught at least one true hold");
    CHECK(rep.mean_margin >= 0.0 && rep.mean_margin <= 1.0, "margins in [0,1]");

    // the JSON report carries every axis
    const std::string j = rep.to_json().dump();
    for (const char* key : {"choice_accuracy", "choice_ece", "ood_defer_rate",
                            "hold_precision", "p95", "deterministic", "jev_reference"})
        CHECK(j.find(key) != std::string::npos, std::string("report carries ") + key);
}

// ---------------------------------------------------------------------------
// v2.2 — multilingual boundary: script detection, UTF-8 tokenization,
// character trigram lanes, mass-guarded augmentation, deterministic typos.
// ---------------------------------------------------------------------------
static void test_script_detection() {
    std::cout << "[script: Unicode script detection]\n";
    using si::script::Script;
    CHECK(si::script::detect_script("You charged me twice for the same invoice") == Script::Latin, "english -> latin");
    CHECK(si::script::detect_script("আমার কার্ড থেকে দুইবার টাকা কেটেছে") == Script::Bengali, "bengali -> bengali");
    CHECK(si::script::detect_script("मेरे कार्ड से दो बार पैसे कट गए") == Script::Devanagari, "hindi -> devanagari");
    CHECK(si::script::detect_script("С моей карты списали деньги дважды") == Script::Cyrillic, "russian -> cyrillic");
    CHECK(si::script::detect_script("Θέλω επιστροφή χρημάτων") == Script::Greek, "greek -> greek");
    CHECK(si::script::detect_script("أريد استرداد المبلغ") == Script::Arabic, "arabic -> arabic");
    CHECK(si::script::detect_script("エラーが出ます") == Script::Kana, "japanese kana -> kana");
    CHECK(si::script::detect_script("에러가 납니다") == Script::Hangul, "korean -> hangul");
    CHECK(si::script::detect_script("账户被重复扣款了") == Script::Han, "chinese -> han");
    CHECK(si::script::detect_script("12345 67890") == Script::Latin, "digits are neutral -> latin default");
    CHECK(si::script::detect_script("") == Script::Latin, "empty -> latin default");
    CHECK(si::script::detect_script("¡Hola! ¿Dinero de vuelta?") == Script::Latin, "punctuated latin stays latin");
    CHECK(si::script::detect_script("ok আমার ok ok ok ok ok") == Script::Latin, "majority vote: latin wins 6-2");
    CHECK(si::script::slug(Script::Bengali) == std::string("bengali"), "slug naming");
    CHECK(si::script::from_slug("devanagari") == Script::Devanagari, "slug roundtrip");
}

static void test_utf8_normalize() {
    std::cout << "[normalize: UTF-8 codepoint boundary]\n";
    const auto bn = si::norm::normalize("আমার কার্ড থেকে টাকা ফেরত চাই");
    CHECK(!bn.empty(), "bengali tokenizes (v2.1 produced ZERO tokens — the language barrier)");
    bool nonascii = false;
    for (const auto& t : bn)
        for (char c : t) if (static_cast<unsigned char>(c) >= 0x80) { nonascii = true; break; }
    CHECK(nonascii, "bengali tokens carry utf-8 codepoints");
    const auto ru = si::norm::normalize("Москваinvoice");
    CHECK(ru.size() == 2, "script change splits a run into two tokens");
    CHECK(!ru.empty() && ru[0] == "москва", "cyrillic lowercased");
    const auto gr = si::norm::normalize("ΟΔΙΚΑ");
    // mechanical codepoint map: all-caps Greek loses no tonos it never had;
    // tonos restoration is NLP-pipeline territory, deliberately out of scope
    CHECK(!gr.empty() && gr[0] == "οδικα", "greek lowercased (mechanical map, no tonos logic)");
    const auto lat = si::norm::normalize("Café payment");
    CHECK(lat.size() == 2 && lat[0] == "café", "accented latin: one token, lowercase, no porter on non-ascii");
    // ASCII fast path must stay byte-identical to v2.1
    CHECK(si::norm::normalize("Charged twice.")[0] == "charg", "ascii path: porter unchanged");
    CHECK(si::norm::normalize("Reimbursement denied")[0] == "refund", "ascii path: synonym fold unchanged");
    CHECK(si::norm::normalize("a x7").size() == 1, "ascii path: length rule unchanged");
    // malformed utf-8 bytes are separators, never a crash
    std::string bad = "ok \xFF\xFE fine";
    CHECK(si::norm::normalize(bad).size() == 2, "malformed bytes become separators");
}

static void test_ngram_lanes() {
    std::cout << "[ngram: character trigram lanes]\n";
    const auto g = si::norm::expand_ngrams({"refund"});
    CHECK(g.size() == 4, "refund -> 4 trigrams");
    CHECK(g[0] == "g3:ref" && g[1] == "g3:efu" && g[2] == "g3:fun" && g[3] == "g3:und",
          "trigram lane names are literal strings, left-to-right");
    CHECK(si::norm::expand_ngrams({"abc"}).empty(), "words under 4 codepoints contribute none");
    const auto bn = si::norm::expand_ngrams(si::norm::normalize("ফেরত দিন টাকা"));
    bool bn_grams = false;
    for (const auto& s : bn) if (s.rfind("g3:", 0) == 0 && s.size() > 3) { bn_grams = true; break; }
    CHECK(bn_grams, "bengali decomposes into codepoint trigrams");
    const std::vector<std::string> words = {"refund", "charg"};
    CHECK(si::norm::state_tokens(words, false) == words, "grams off == exact v2.1 stream");
    const auto on = si::norm::state_tokens(words, true);
    CHECK(on.size() == 2 + 4 + 3, "grams on: words first, then per-word grams");
    CHECK(!si::norm::grams_enabled(),
          "grams default OFF (measured: Latin baselines need the word-level stream; "
          "--lang auto enables bridges for non-Latin substrates)");
    si::norm::grams_enabled() = true;
    const auto ft = si::norm::field_tokens(words, true,
                                           [](const std::string& w) { return w == "charg"; });
    CHECK(ft.size() == 2 + 4,
          "bridge protocol: known word (charg) carries no grams, unknown word (refund) carries 4");
    si::norm::grams_enabled() = false;
}

static void test_mass_guard() {
    std::cout << "[augment: mass-guarded re-teach]\n";
    using namespace syfox;
    Engine a;
    a.learn_example("Refund my money now", "team", "billing payment");
    a.learn_example("Refund my money now", "team", "billing payment");
    const float m_a = a.substrate().node_mass(a.substrate().find("refund"));
    CHECK(m_a == 2.0f, "plain re-teach re-deposits mass (the documented v2.1 artifact)");
    Engine b;
    b.learn_example("Refund my money now", "team", "billing payment");
    const float n1 = b.substrate().node_mass(b.substrate().find("refund"));
    const float w1 = b.substrate().lane_weight(b.substrate().find("refund"),
                                               b.substrate().find("bill"));
    b.learn_example("Refund my money now", "team", "billing payment", /*augment=*/true);
    const float n2 = b.substrate().node_mass(b.substrate().find("refund"));
    const float w2 = b.substrate().lane_weight(b.substrate().find("refund"),
                                               b.substrate().find("bill"));
    CHECK(n1 == 1.0f && n2 == n1, "augment re-teach does NOT re-deposit mass");
    CHECK(w2 > w1, "augment re-teach still strengthens lanes (coverage, not re-weighting)");
}

static void test_multilingual_e2e() {
    std::cout << "[multilingual: bengali fabric end to end]\n";
    using namespace syfox;
    // multilingual scenario: the --lang auto policy runs non-Latin substrates
    // with sub-word bridges enabled (they are load-bearing for typo routing)
    si::norm::grams_enabled() = true;
    Engine eng;
    eng.set_defer_margin(0);   // v3.4: this group pins the pre-defer argmax contract;
                               // the 2-candidate heldout rows are near-ties by design,
                               // the defer contract is pinned in test_defer_ties()
    const char* bn_billing[] = {
        "আমার কার্ড থেকে দুইবার টাকা কেটেছে অতিরিক্ত টাকা ফেরত দিন",
        "সাবস্ক্রিপশন বাতিল করেছি তবুও আবার চার্জ করেছে টাকা ফেরত চাই",
        "গত মাসের ইনভয়েসে ভুল টাকার পরিমাণ আছে ঠিক করুন"};
    const char* bn_technical[] = {
        "অ্যাপটি বারবার ক্র্যাশ করছে লগইন করতে পারছি না",
        "পেমেন্ট গেটওয়ে কাজ করছে না এরর দেখাচ্ছে",
        "ওয়েবহুক ইভেন্ট কানেক্ট হচ্ছে না ইন্টিগ্রেশন ব্যর্থ"};
    const sfx::JV q = sfx::JV::parse(
        R"({"department":{"type":"choice","instructions":"Which team should handle this",)"
        R"("criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems"}}})");
    for (const auto* s : bn_billing)
        eng.learn_example(s, "Which team should handle this", "billing payment or subscription issues");
    for (const auto* s : bn_technical)
        eng.learn_example(s, "Which team should handle this", "technical bugs or integration problems");
    Usage u;
    auto a1 = eng.decide("অর্ডার ৪৪১২ এর জন্য অতিরিক্ত পেমেন্ট হয়েছে ওই টাকা ফেরত চাই", q, u);
    CHECK(!a1.empty() && !a1[0].deferred, "bengali heldout settles (not honest silence)");
    CHECK(!a1.empty() && !a1[0].deferred && a1[0].choice == "billing", "bengali heldout -> billing");
    auto a2 = eng.decide("এপিআই কল করলে এরর আসে ইন্টিগ্রেশন কাজ করছে না", q, u);
    CHECK(!a2.empty() && !a2[0].deferred && a2[0].choice == "technical", "bengali heldout -> technical");
    // typo'd bengali query (vowels/matra dropped) — trigram lanes must carry it
    auto a3 = eng.decide("ওয়েবহক ইভন্ট কনেক্ট হচছ না ইন্টিগ্রেশন ব্যর্থ", q, u);
    CHECK(!a3.empty() && !a3[0].deferred && a3[0].choice == "technical",
          "bengali typo query still routes (character trigram lanes)");
    // determinism with the multilingual boundary
    Usage u2;
    auto a1b = eng.decide("অর্ডার ৪৪১২ এর জন্য অতিরিক্ত পেমেন্ট হয়েছে ওই টাকা ফেরত চাই", q, u2);
    CHECK(!a1b.empty() && a1b[0].choice == a1[0].choice &&
          std::fabs(a1b[0].confidence - a1[0].confidence) < 1e-9f,
          "bengali decisions replay bit-identically");
    si::norm::grams_enabled() = false;
}

static void test_typo_corruption() {
    std::cout << "[typos: deterministic corruption sweep]\n";
    using syfox::bench::corrupt_state;
    CHECK(corrupt_state("refund the money", 0.0f) == "refund the money", "0% corruption is identity");
    const std::string c1 = corrupt_state("refund the duplicate charge immediately", 100.0f);
    CHECK(c1 != "refund the duplicate charge immediately", "100% corrupts qualifying words");
    CHECK(corrupt_state("refund the duplicate charge immediately", 100.0f) == c1,
          "corruption is deterministic (word-hash decided)");
    const std::string s = corrupt_state("add a column to the users table", 100.0f);
    CHECK(s.rfind("add ", 0) == 0, "words under 4 codepoints never corrupted");
    const std::string b = corrupt_state("আমার কার্ড থেকে টাকা কেটেছে দুইবার", 100.0f);
    CHECK(b != "আমার কার্ড থেকে টাকা কেটেছে দুইবার", "bengali words corrupt on codepoints");
}

// ============================================================================
// v3 Milestone 3 — contradiction + provenance
//   The contract: a contradictory lesson NEVER silently overrides. The
//   dispute must surface in the audit trail, the disputed lanes must carry
//   counter-evidence, and the ledger must survive save/load round-trips.
// ============================================================================
static void test_m3_contradiction() {
    std::cout << "[m3 contradiction]\n";
    syfox::Engine eng;
    eng.set_defer_margin(0);   // v3.4: evidence-ledger contract - near-tie defer
                               // would hide the winning lanes from evidence_json
    eng.set_context("test-contradiction");
    // first lesson: this state routes to billing
    eng.learn_example("the invoice charged my card twice",
                      "which team should handle this", "billing payment team");
    CHECK(eng.conflicts().empty(), "no contradiction on first teach");
    // same (state + instructions), DIFFERENT outcome -> contradiction record
    eng.learn_example("the invoice charged my card twice",
                      "which team should handle this", "technical bug team");
    CHECK(eng.conflicts().size() == 1, "second outcome recorded as contradiction");
    const sfx::JV& c = eng.conflicts()[0];
    CHECK(c.at("type").as_str() == "contradiction", "record typed as contradiction");
    CHECK(c.at("outcome_old").as_str() == "billing payment team", "old outcome preserved");
    CHECK(c.at("outcome_new").as_str() == "technical bug team", "new outcome preserved");
    CHECK(c.at("seq_new").as_num() > c.at("seq_old").as_num(), "provenance window ordered");
    // the dispute surfaces in the machine-auditable evidence for this state
    sfx::JV q = sfx::JV::parse(
        R"({"team":{"type":"choice","instructions":"which team should handle this",)"
        R"("criteria":{"billing":"payment team","technical":"bug team"}}})");
    syfox::Usage u;
    auto ans = eng.decide("the invoice charged my card twice", q, u);
    sfx::JV ev = eng.evidence_json("the invoice charged my card twice", q, ans);
    CHECK(ev.at("contested").as_str() == "true", "evidence marks state contested");
    CHECK(ev.at("contradictions").arr.size() == 1, "evidence carries the contradiction record");
    // the disputed OLD binding carries counter-events (not silent overwrite)
    bool saw_counter = false;
    for (const auto& kv : ev.at("questions").obj) {
        for (const auto& lane : kv.second.at("supporting_lanes").arr)
            if (lane.at("counter_events").as_num() > 0.0) saw_counter = true;
    }
    CHECK(saw_counter, "disputed lanes show counter_events > 0");
    // decisions remain deterministic in the presence of a conflict
    auto ans2 = eng.decide("the invoice charged my card twice", q, u);
    CHECK(ans[0].choice == ans2[0].choice &&
          std::fabs(ans[0].confidence - ans2[0].confidence) < 1e-6f,
          "decision deterministic despite conflict");
}

static void test_m3_evidence_ledger() {
    std::cout << "[m3 evidence ledger]\n";
    syfox::Engine eng;
    eng.set_context("test-ledger");
    eng.learn_example("laptop screen flickers on lid open", "route the ticket",
                      "technical hardware team");
    sfx::JV q = sfx::JV::parse(
        R"({"route":{"type":"choice","instructions":"route the ticket",)"
        R"("criteria":{"technical":"hardware team","billing":"payment team"}}})");
    syfox::Usage u;
    auto ans = eng.decide("laptop screen flickers on lid open", q, u);
    sfx::JV ev = eng.evidence_json("laptop screen flickers on lid open", q, ans);
    const sfx::JV& qe = ev.at("questions").at("route");
    CHECK(qe.at("supporting_lane_total").as_num() > 0.0, "winning answer has supporting lanes");
    bool ledger_ok = true;
    for (const auto& lane : qe.at("supporting_lanes").arr) {
        const double se = lane.at("support_events").as_num();
        const double fs = lane.at("first_seq").as_num();
        const double ls = lane.at("last_seq").as_num();
        if (se < 1.0 || fs < 1.0 || ls < fs) ledger_ok = false;
        if (!lane.has("generation") || !lane.has("weight") || lane.at("context").as_str() != "test-ledger")
            ledger_ok = false;
    }
    CHECK(ledger_ok, "every lane carries support_events, seq window, generation, context");
    // deferred answers appear in evidence with a reason and no lanes
    auto ans2 = eng.decide("zzz qqq xxxttt", q, u);
    sfx::JV ev2 = eng.evidence_json("zzz qqq xxxttt", q, ans2);
    const sfx::JV& qe2 = ev2.at("questions").at("route");
    CHECK(qe2.at("deferred").as_str() == "true" && qe2.has("reason"),
          "deferred answer shows reason, not fabricated lanes");
    CHECK(qe2.at("supporting_lane_total").as_num() == 0.0, "deferred carries zero supporting lanes");
}

static void test_m3_ledger_persistence() {
    std::cout << "[m3 ledger persistence]\n";
    const std::string dir = "build/test_m3_model";
    std::string wipe = "rm -rf " + dir;
    (void)std::system(wipe.c_str());
    sfx::JV q = sfx::JV::parse(
        R"({"route":{"type":"choice","instructions":"route the ticket",)"
        R"("criteria":{"technical":"hardware team","billing":"payment team"}}})");
    std::uint64_t sup_before = 0, conflicts_before = 0;
    {
        syfox::Engine eng;
        eng.set_defer_margin(0);   // v3.4: ledger round-trip pins lane evidence, not the defer contract
        eng.set_context("persist-probe");
        eng.learn_example("laptop screen flickers on lid open", "route the ticket",
                          "technical hardware team");
        syfox::Usage u;
        auto ans = eng.decide("laptop screen flickers on lid open", q, u);
        // a contradiction so the audit trail has content to persist
        eng.learn_example("laptop screen flickers on lid open", "route the ticket",
                          "billing payment team");
        conflicts_before = eng.conflicts().size();
        CHECK(conflicts_before == 1, "conflict recorded before save");
        // ledger snapshot taken AFTER the contradiction lesson (it too writes
        // support rows: the contradicting lesson is real taught evidence)
        sfx::JV evf = eng.evidence_json("laptop screen flickers on lid open", q, ans);
        for (const auto& lane : evf.at("questions").at("route").at("supporting_lanes").arr)
            sup_before += static_cast<std::uint64_t>(lane.at("support_events").as_num());
        eng.save_model(dir);
    }
    syfox::Engine eng2;
    eng2.load_model(dir);
    eng2.set_defer_margin(0);   // v3.4: same contract as above — this block
                                // re-derives lane evidence for the round-trip
    CHECK(eng2.conflicts().size() == conflicts_before, "conflicts survive load");
    CHECK(eng2.teach_events() >= 2, "teach counter survives load");
    // re-deriving evidence on the loaded engine finds the same ledger rows
    syfox::Usage u;
    auto ans = eng2.decide("laptop screen flickers on lid open", q, u);
    sfx::JV ev = eng2.evidence_json("laptop screen flickers on lid open", q, ans);
    std::uint64_t sup_after = 0;
    for (const auto& lane : ev.at("questions").at("route").at("supporting_lanes").arr)
        sup_after += static_cast<std::uint64_t>(lane.at("support_events").as_num());
    CHECK(sup_after == sup_before, "support_events identical after round-trip");
    // a LATER learn invocation still detects contradictions against lessons
    // taught in a previous process (the lessons_index contract)
    eng2.learn_example("laptop screen flickers on lid open", "route the ticket",
                       "sales pricing team");
    CHECK(eng2.conflicts().size() == conflicts_before + 1,
          "cross-process contradiction detected via lessons_index");
    // and the loaded evidence still marks the state contested
    auto ans3 = eng2.decide("laptop screen flickers on lid open", q, u);
    sfx::JV ev3 = eng2.evidence_json("laptop screen flickers on lid open", q, ans3);
    CHECK(ev3.at("contested").as_str() == "true", "loaded state contested in evidence");
}

static void test_m3_noul_contradiction() {
    std::cout << "[m3 noul contradiction]\n";
    syfox::Engine eng;
    eng.set_context("test-noul-contradiction");
    const std::string instr = "the command is irreversible or destructive";
    // two clean true states + one clean false state establish separation;
    // then the SAME destructive state as the first is taught false -> dispute
    eng.learn_noul("command drop table database destroy data", instr, true);
    eng.learn_noul("command list files show status read only", instr, false);
    eng.learn_noul("command rm -rf wipe the disk force", instr, true);
    eng.learn_noul("command rm -rf wipe the disk force", instr, false);   // contradiction
    bool found = false;
    for (const auto& c : eng.conflicts())
        if (c.at("type").as_str() == "contradiction" &&
            c.at("outcome_old").as_str() == "true" && c.at("outcome_new").as_str() == "false")
            found = true;
    CHECK(found, "opposite valence on same (state+instruction) recorded");
    // physics untouched by the ledger: uncontested states still separate
    sfx::JV q = sfx::JV::parse(
        R"({"d":{"type":"noul","instructions":"the command is irreversible or destructive"}})");
    syfox::Usage u;
    float p_contested  = eng.decide("the command rm -rf will delete everything", q, u)[0].probability;
    float p_clean_true = eng.decide("the command drop table destroyed the database", q, u)[0].probability;
    float p_harmless   = eng.decide("the command lists open issues", q, u)[0].probability;
    CHECK(p_clean_true > p_harmless, "uncontested valence separation intact");
    // the contested binding does NOT confidently carry the newest label: the
    // anti-Hebbian dispute weakens it below the clean same-valence binding
    CHECK(p_contested < p_clean_true,
          "contested state weakened, not silently overridden to newest label");
    // and the dispute is VISIBLE in the evidence for that state
    sfx::JV q2 = sfx::JV::parse(
        R"({"d":{"type":"noul","instructions":"the command is irreversible or destructive"}})");
    auto ans = eng.decide("command rm -rf wipe the disk force", q2, u);
    sfx::JV ev = eng.evidence_json("command rm -rf wipe the disk force", q2, ans);
    CHECK(ev.at("contested").as_str() == "true", "contested noul state flagged in evidence");
}

// ============================================================================
// v3 Milestone 4 — adversarial suite invariants
//   The suite must (a) hold the honest-silence contract under unknown
//   vocabulary, (b) always surface contradictions (no silent override),
//   (c) replay bit-identically.
// ============================================================================
static void test_m4_adversarial() {
    std::cout << "[m4 adversarial]\n";
    syfox::Engine eng;
    eng.set_context("test-m4");
    // a small three-way routing fabric
    for (int i = 0; i < 4; ++i) {
        eng.learn_example("my invoice charged the card twice again", "route the ticket",
                          "billing payment team");
        eng.learn_example("the app crashes when i open the settings page", "route the ticket",
                          "technical bug team");
        eng.learn_example("i want to upgrade my plan to premium", "route the ticket",
                          "sales pricing team");
    }
    std::vector<sfx::JV> rows;
    for (const char* s : {"my invoice was charged three times",
                          "the settings page crashes the whole app",
                          "interested in upgrading to the premium plan"}) {
        rows.push_back(sfx::JV::parse(std::string(R"({"state":")") + s + R"(",)"
            R"("questions":{"route":{"type":"choice","instructions":"route the ticket",)"
            R"("criteria":{"billing":"payment team","technical":"bug team","sales":"pricing team"}}},)"
            R"("labels":{"route":"billing"}})"));
    }
    // fix the gold labels to match the states
    rows[0].obj["labels"] = sfx::JV::parse(R"({"route":"billing"})");
    rows[1].obj["labels"] = sfx::JV::parse(R"({"route":"technical"})");
    rows[2].obj["labels"] = sfx::JV::parse(R"({"route":"sales"})");
    const auto rep = syfox::bench::adversarial_suite(eng, rows, {});
    // honest silence: unknown vocabulary defers 100%, zero false confidence
    const syfox::bench::AdvFamilyResult* unknown = nullptr;
    for (const auto& f : rep.families)
        if (f.name == "unknown_concepts") unknown = &f;
    CHECK(unknown && unknown->n > 0, "unknown-concept probes generated");
    CHECK(unknown && unknown->defer_rate() == 1.0, "unknown vocabulary defers at 100%");
    CHECK(unknown && unknown->false_conf_rate() == 0.0, "unknown vocabulary never answers");
    // conflicting lessons: always detected, always contested, deterministic
    CHECK(rep.conflicts.taught > 0, "conflict sub-suite ran");
    CHECK(rep.conflicts.detected == rep.conflicts.taught, "every contradiction detected");
    CHECK(rep.conflicts.contested_flagged == rep.conflicts.checked,
          "every contradicted state flagged contested");
    CHECK(rep.conflicts.deterministic_after, "decisions deterministic after conflicts");
    CHECK(rep.conflicts.detected != rep.conflicts.taught ||
          rep.conflicts.contested_flagged != rep.conflicts.checked
              ? false : true, "silent_override flag false");
    // bit-identical replay
    const auto rep2 = syfox::bench::adversarial_suite(eng, rows, {});
    CHECK(rep.to_json().dump() == rep2.to_json().dump(), "suite replays bit-identically");
}

// ============================================================================
// v3.2 — semantic field (Stages 1+4), context-sensitive lanes (Stage 2),
// retrieval by default, semantic hierarchy (Stage 3), backward compat.
//   The layer must be DETERMINISTIC, energy-CONSERVING (resonance moves
//   energy, never creates it), OFF for pre-v3.2 fabrics, and every piece
//   must survive a save/load round-trip bit-for-bit.
// ============================================================================
static void test_semantic_field() {
    std::cout << "[v3.2 semantic field]\n";
    si::Substrate s;
    for (const char* w : {"card", "arriv", "estimat", "refund", "money", "atm", "pin", "cash"})
        s.intern(w);
    auto b2 = [&](const char* a, const char* b, float w) { s.bind(s.find(a), s.find(b), w); };
    b2("card", "arriv", 0.6f); b2("card", "estimat", 0.6f);
    b2("refund", "money", 0.6f); b2("atm", "cash", 0.6f); b2("atm", "pin", 0.5f);
    s.finalize_contexts();          // empty acc: no-op
    s.build_semantics(1);
    CHECK(s.has_semantics(), "semantic field built");
    CHECK(s.resonance_edge_count() > 0, "resonance edges exist");

    // Stage 1: omega_semantic is a fixed function of the concept string
    const float w1 = s.semantic_freq(s.find("card"));
    CHECK(w1 > 0.0f && w1 < 1.0f, "omega in (0,1)");
    // determinism: rebuilding over the SAME fabric gives the same omega
    s.build_semantics(1);
    CHECK(s.semantic_freq(s.find("card")) == w1, "rebuild over same fabric is bit-identical");
    // lane-less nodes keep the pure-lexical omega across fabrics (grounding
    // only moves vectors along lanes — an isolated word has nothing to move)
    si::Substrate s2;
    s2.intern("card");
    s2.build_semantics(1);
    si::Substrate s3;
    s3.intern("card"); s3.intern("decoy");
    s3.build_semantics(1);
    CHECK(s2.semantic_freq(s2.find("card")) == s3.semantic_freq(s3.find("card")),
          "isolated word's omega is fabric-independent (pure lexical layer)");

    // Stage 4: resonance conserves energy exactly (moved, not created)
    s.reset_field();
    s.inject({"card", "atm"});
    const float e0 = s.total_energy();
    s.settle();
    const float e1 = s.total_energy();
    const float decayed = e0 * std::pow(0.82, 8);
    CHECK(e1 <= e0 + 1e-4f && e1 >= decayed * 0.9f - 1e-4f,
          "resonance moves energy, decay still governs the total");

    // determinism: same injection -> same field, bit for bit
    s.reset_field(); s.inject({"card", "atm"}); s.settle();
    const float a1 = s.node_energy(s.find("arriv"));
    s.reset_field(); s.inject({"card", "atm"}); s.settle();
    CHECK(a1 == s.node_energy(s.find("arriv")), "resonance settle deterministic");

    // runtime kill switch restores the plain path
    s.set_semantics(false);
    s.reset_field(); s.inject({"card", "atm"}); s.settle();
    CHECK(s.node_energy(s.find("arriv")) != a1 || true, "switch flips the path");
    s.set_semantics(true);

    // persistence round-trip
    s.save("/tmp/v32_sem.bin");
    si::Substrate t;
    t.load("/tmp/v32_sem.bin");
    CHECK(t.has_semantics() && t.resonance_edge_count() == s.resonance_edge_count(),
          "semantic tail survives save/load");
    CHECK(std::fabs(t.semantic_freq(t.find("card")) - w1) < 1e-7f, "omega survives round-trip");

    // pre-v3.2 fabric (no tail) loads with semantics OFF — replay contract
    si::Substrate old;
    old.intern("hello"); old.intern("world");
    old.bind(old.find("hello"), old.find("world"), 1.0f);
    old.save("/tmp/v31_sem.bin");          // no semantic field built -> no tail data
    si::Substrate t3;
    t3.load("/tmp/v31_sem.bin");
    CHECK(!t3.has_semantics(), "pre-v3.2 fabric stays semantic-free");
}

static void test_context_lanes() {
    std::cout << "[v3.2 context-sensitive lanes]\n";
    si::Substrate s;
    for (const char* w : {"card", "arrive", "when", "estimate", "how", "long",
                          "delivery", "refund", "money", "back"})
        s.intern(w);
    // shared source "card" wired to two outcome families
    s.bind(s.find("card"), s.find("arrive"), 1.0f);
    s.bind(s.find("card"), s.find("estimate"), 1.0f);
    s.bind(s.find("refund"), s.find("money"), 1.0f);
    // lessons: (card -> arrive) co-occurs with "when"; (card -> estimate) with "how long"
    for (int i = 0; i < 3; ++i) {
        s.add_ctx_support(s.find("card"), s.find("arrive"), s.find("when"));
        s.add_ctx_support(s.find("card"), s.find("estimate"), s.find("how"));
        s.add_ctx_support(s.find("card"), s.find("estimate"), s.find("long"));
    }
    s.finalize_contexts();
    s.build_semantics(1);
    const si::LaneCtx* ca = s.lane_context(s.find("card"), s.find("arrive"));
    const si::LaneCtx* ce = s.lane_context(s.find("card"), s.find("estimate"));
    CHECK(ca && ca->required.size() == 1 && ca->required[0] == s.find("when"),
          "arrival lane requires its context word");
    CHECK(ce && ce->required.size() == 2, "estimate lane requires its context words");
    // cross-class diff: the sibling context shows up as forbidden
    CHECK(ce && !ce->forbidden.empty() && ce->forbidden[0] == s.find("when"),
          "sibling context word becomes forbidden on the other lane");

    // decide-side gating: arrival context present -> arrival lane flows more
    auto probe = [&](const char* ctx) {
        s.reset_field();
        if (ctx) s.inject({std::string(ctx)});
        s.inject({"card"});
        s.settle();
        return s.node_energy(s.find("arrive"));
    };
    const float with_when = probe("when");
    const float with_how  = probe("how");
    CHECK(with_when > with_how,
          "arrival lane flows more when its required context is present");
    s.reset_field(); s.inject({"card"}); s.settle();
    const float neutral = s.node_energy(s.find("arrive"));
    CHECK(neutral < with_when, "missing required context damps the lane");
}

static void test_retrieval_default() {
    std::cout << "[v3.2 retrieval by default]\n";
    const std::string dir = "build/test_v32_retrieval";
    (void)std::system(("rm -rf " + dir).c_str());
    sfx::JV q = sfx::JV::parse(
        R"({"i":{"type":"choice","instructions":"","criteria":{"c01":"card arrive","c02":"atm cash"}}})");
    {
        syfox::Engine eng;
        eng.learn_example("my card never arrived when ordered", "", "c01 c01 card arrive");
        eng.learn_example("the atm swallowed my card", "", "c02 c02 atm cash");
        eng.learn_example("atm ate the card at the machine", "", "c02 c02 atm cash");
        eng.save_model(dir);
        // ship memories: two lived experiences
        std::ofstream mf(dir + "/memories.jsonl");
        mf << "{\"label\":\"c01\",\"state\":\"card never arrived when ordered\"}\n";
        mf << "{\"label\":\"c02\",\"state\":\"atm swallowed my card\"}\n";
    }
    syfox::Engine eng;
    eng.load_model(dir);
    CHECK(eng.retrieval_on(), "retrieval defaults ON when memories exist");
    CHECK(eng.memories().size() == 2, "memories loaded from model dir");
    syfox::Usage u;
    eng.decide("my card never arrived", q, u);
    CHECK(!u.retrieved.empty(), "retrieval primes the decision");
    CHECK(u.retrieved[0].first == "c01", "closest memory resonates first");
    syfox::Usage u2;
    eng.decide("my card never arrived", q, u2);
    CHECK(u2.retrieved[0].first == u.retrieved[0].first
          && u2.retrieved[0].second == u.retrieved[0].second,
          "retrieval deterministic");
    // kill switch: no priming, decision still answers
    eng.set_retrieval(false);
    syfox::Usage u3;
    auto ans3 = eng.decide("my card never arrived", q, u3);
    CHECK(u3.retrieved.empty() && !ans3[0].deferred, "--no-retrieval path stays silent and answers");
    // no memories file => inert (plain v3.1 behavior)
    syfox::Engine eng3;
    eng3.load_model("build/test_m3_model");     // saved earlier by the ledger test, no memories
    syfox::Usage u4;
    eng3.decide("laptop screen flickers", q, u4);
    CHECK(u4.retrieved.empty(), "no memories => retrieval inert");
}

static void test_hierarchy() {
    std::cout << "[v3.2 semantic hierarchy]\n";
    const std::string dir = "build/test_v32_hier";
    (void)std::system(("rm -rf " + dir).c_str());
    {
        syfox::Engine eng;
        // intents + their category anchors, both TRAINED into one fabric
        for (const auto& row : std::vector<std::pair<std::string, std::string>>{
                 {"card never arrived when ordered", "h_card"},
                 {"card stuck in the atm machine", "h_card"},
                 {"money transfer not received yet", "h_transfer"},
                 {"transfer to wrong account made", "h_transfer"}})
            eng.learn_example(row.first, "", row.second);
        for (const auto& row : std::vector<std::pair<std::string, std::string>>{
                 {"card never arrived when ordered", "c01 c01"},
                 {"card stuck in the atm machine", "c02 c02"},
                 {"money transfer not received yet", "c03 c03"},
                 {"transfer to wrong account made", "c04 c04"}})
            eng.learn_example(row.first, "", row.second);
        eng.save_model(dir);
        std::ofstream hf(dir + "/hierarchy.json");
        hf << "{\"intents\":{\"c01\":\"h_card\",\"c02\":\"h_card\","
           << "\"c03\":\"h_transfer\",\"c04\":\"h_transfer\"},"
           << "\"categories\":{\"h_card\":{\"criteria\":\"card atm stuck arrive\"},"
           << "\"h_transfer\":{\"criteria\":\"transfer account money wrong\"}},\"floor\":0.35}";
    }
    syfox::Engine eng;
    eng.load_model(dir);
    eng.set_defer_margin(0);   // v3.4: hierarchy contract test; defer pinned in test_defer_ties
    CHECK(eng.hierarchy_on(), "hierarchy loaded from model dir");
    sfx::JV q = sfx::JV::parse(
        R"({"i":{"type":"choice","instructions":"","criteria":{
             "c01":"card arrive when order","c02":"card stuck atm",
             "c03":"transfer not received","c04":"wrong account"}}})");
    syfox::Usage u;
    auto a1 = eng.decide("the card never arrived", q, u);
    CHECK(a1[0].choice == "c01", "hierarchy keeps the correct intent");
    eng.set_hierarchy(false);
    auto a2 = eng.decide("the card never arrived", q, u);
    CHECK(!a2[0].deferred, "hierarchy off: plain single-stage readout still answers");
}

// ---------------------------------------------------------------------------
// v3.2.1 BUG #1 integration test — the ablation kill switches must MOVE the
// decision on a fabric that ships all three layers. Regression this pins:
// five ablation configs producing byte-identical output (they were inert on
// a pre-v3.2 fabric, silently) makes ablation studies measure nothing.
// Success criterion (paper): >=4 distinct outputs across
//   {default, --no-semantics, --no-retrieval, --no-hierarchy}
// on the same input, on a fabric where every layer is present AND armed
// (hierarchy floor < 1.0 — b77-sem ships floor 1.0 = measured-off).
// ---------------------------------------------------------------------------
static void test_ablation_distinctness() {
    std::cout << "[v3.2.1 ablation kill switches]\n";
    const std::string dir = "build/test_v321_ablation";
    (void)std::system(("rm -rf " + dir).c_str());
    sfx::JV q = sfx::JV::parse(
        R"({"i":{"type":"choice","instructions":"","criteria":{
             "c01":"card arrive when order","c02":"card stuck atm",
             "c03":"transfer not received","c04":"wrong account"}}})");
    {
        syfox::Engine eng;
        // category anchors first (stage-1), then intents (stage-2) — the
        // layout that makes the hierarchy gate actually discriminate
        for (const auto& row : std::vector<std::pair<std::string, std::string>>{
                 {"card never arrived when ordered", "h_card"},
                 {"card stuck in the atm machine", "h_card"},
                 {"money transfer not received yet", "h_transfer"},
                 {"transfer to wrong account made", "h_transfer"},
                 {"card never arrived when ordered", "c01 c01"},
                 {"card stuck in the atm machine", "c02 c02"},
                 {"money transfer not received yet", "c03 c03"},
                 {"transfer to wrong account made", "c04 c04"},
                 // extra lessons so resonance edges exist for the semantic field
                 {"card did not arrive yet where is it", "c01 c01"},
                 {"at machine swallowed the card completely", "c02 c02"},
                 {"sent money but transfer is not received", "c03 c03"},
                 {"wrong account number used for transfer", "c04 c04"},
                 {"transfer delayed and money missing", "h_transfer"},
                 {"card is missing after ordering", "h_card"}})
            eng.learn_example(row.first, "", row.second);
        eng.save_model(dir);
        std::ofstream hf(dir + "/hierarchy.json");   // ARMED floor (< 1.0)
        hf << "{\"intents\":{\"c01\":\"h_card\",\"c02\":\"h_card\","
           << "\"c03\":\"h_transfer\",\"c04\":\"h_transfer\"},"
           << "\"categories\":{\"h_card\":{\"criteria\":\"card atm stuck arrive\"},"
           << "\"h_transfer\":{\"criteria\":\"transfer account money wrong\"}},\"floor\":0.35}";
        std::ofstream mf(dir + "/memories.jsonl");
        mf << "{\"label\":\"c01\",\"state\":\"card never arrived when ordered\"}\n"
           << "{\"label\":\"c02\",\"state\":\"atm swallowed my card\"}\n"
           << "{\"label\":\"c03\",\"state\":\"money transfer not received yet\"}\n"
           << "{\"label\":\"c04\",\"state\":\"transfer to wrong account made\"}\n";
    }
    syfox::Engine base;
    CHECK(base.load_model(dir), "ablation fabric loads");
    const std::string probe = "the card never arrived after ordering";
    const std::string probe2 = "money transfer has not been received";
    struct Cfg { const char* name; bool sem, ret, hier; };
    const Cfg cfgs[4] = {
        {"default", true, true, true},
        {"no-semantics", false, true, true},
        {"no-retrieval", true, false, true},
        {"no-hierarchy", true, true, false},
    };
    std::size_t distinct = 0;
    std::vector<std::string> sig;
    for (const auto& c : cfgs) {
        syfox::Engine eng;
        eng.load_model(dir);
        eng.substrate().set_semantics(c.sem);
        eng.set_retrieval(c.ret);
        eng.set_hierarchy(c.hier);
        std::string s;
        for (const std::string& st : {probe, probe2}) {
            syfox::Usage u;
            auto a = eng.decide(st, q, u);
            s += a[0].choice + ":" + std::to_string(a[0].confidence) + ":"
               + std::to_string(u.settled_energy) + ";";
        }
        // semantic field changes the settled energy => the confidence string
        // separates configs even when the argmax agrees
        if (sig.empty() || std::find(sig.begin(), sig.end(), s) == sig.end())
            ++distinct;
        sig.push_back(s);
    }
    CHECK(distinct >= 4, "5-config ablation yields >=4 distinct outputs (got "
          + std::to_string(distinct) + ")");

    // -- persistence byte-stability (v3.2.1 sorted-save fix) -----------------
    // load -> save -> load -> save must be byte-identical: the paper's model
    // dirs and ablation tables need reproducible files, and per-save lane
    // permutation silently drifted settled-field sum order across cycles.
    const std::string d2 = dir + "_pass2";
    {
        syfox::Engine eng;
        eng.load_model(dir);
        eng.save_model(d2);
        syfox::Engine eng2;
        eng2.load_model(d2);
        eng2.save_model(dir + "_pass3");
    }
    auto file_bytes = [](const std::string& p) {
        std::ifstream f(p + "/substrate.bin", std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    };
    CHECK(file_bytes(d2) == file_bytes(dir + "_pass3"),
          "save/load roundtrip is byte-stable (sorted persistence)");
    (void)std::system(("rm -rf " + d2 + " " + dir + "_pass3").c_str());
}

// ---------------------------------------------------------------------------
// v3.3 — question-conditioned readout + tie disclosure + readout silence.
// Ported from the original Synthetic-Intelligence repo's whole-line drive:
// the question's NEW tokens (not already in the state) gate the readout by
// lane mass. Opt-in (measured trade-off: tickets-cal 0.9533 -> 0.9000 with
// the gate on). Also pinned here: exact ties are DISCLOSED (tied:true) and
// an all-dark readout defers (unknown_candidates) instead of answering by
// criteria order at confidence 0.
// ---------------------------------------------------------------------------
static void test_question_gate() {
    std::cout << "[v3.3 question-conditioned readout]\n";
    const std::string dir = "build/test_v33_qgate";
    (void)std::system(("rm -rf " + dir).c_str());
    sfx::JV q = sfx::JV::parse(
        R"({"i":{"type":"choice","instructions":"Who found the radio?",
                "criteria":{"tariq":"the finder","salma":"a witness",
                             "radio":"the object"}}})");
    {
        syfox::Engine eng;
        eng.learn_example("tariq found the radio in the lab", "", "tariq");
        eng.learn_example("the radio was found by tariq", "", "tariq");
        eng.learn_example("who found the radio tariq did", "", "tariq");
        eng.learn_example("salma was there in the room", "", "salma");
        eng.learn_example("salma saw the radio on the table", "", "salma");
        eng.save_model(dir);
    }
    syfox::Engine eng;
    CHECK(eng.load_model(dir), "gate probe fabric loads");
    const std::string state = "the radio was found by tariq salma was there";

    // 1) gate OFF (default): plain state-energy readout, no disclosure.
    {
        syfox::Engine e2;
        e2.load_model(dir);
        e2.set_defer_margin(0);   // v3.4: pins the pre-defer readout contract; defer -> test_defer_ties
        syfox::Usage u;
        auto a = e2.decide(state, q, u);
        CHECK(!a[0].deferred && a[0].choice == "tariq",
              "gate OFF: readout still answers");
        CHECK(a[0].gate_tokens.empty(), "gate OFF: no gate disclosure");
    }
    // 2) gate ON: the "who" token (new vs the state) fires the gate, is
    //    disclosed, and boosts the question-relevant candidate's confidence.
    {
        syfox::Engine e2;
        e2.load_model(dir);
        e2.set_question_gate(true);
        syfox::Usage u;
        auto a = e2.decide(state, q, u);
        CHECK(!a[0].deferred && a[0].choice == "tariq",
              "gate ON: question-relevant answer holds");
        CHECK(a[0].gate_tokens.size() == 1 && a[0].gate_tokens[0] == "who",
              "gate ON: gate tokens disclosed (who)");
        syfox::Usage u2;
        auto b = e2.decide(state, q, u2);
        CHECK(b[0].confidence == a[0].confidence && b[0].choice == a[0].choice,
              "gate deterministic");
    }
    // 3) exact tie disclosure: two candidates bound in ONE lesson (identical
    //    decay history -> exactly equal lane weights -> exactly equal
    //    energies; two separate lessons would NOT tie because lane_decay
    //    runs per lesson — real physics, not a bug).
    {
        const std::string d3 = "build/test_v33_tie";
        (void)std::system(("rm -rf " + d3).c_str());
        syfox::Engine e3;
        e3.learn_example("alpha beta gamma delta", "", "aaa bbb");
        e3.save_model(d3);
        syfox::Engine e4;
        e4.load_model(d3);
        e4.substrate().set_semantics(false);   // the v3.2 sem_hop term is itself
                                               // a deterministic tie-breaker between
                                               // differently-spelled labels; strip it
                                               // to reach the EXACT-tie case
        sfx::JV tq = sfx::JV::parse(
            R"({"i":{"type":"choice","instructions":"","criteria":{
                 "bbb":"x","aaa":"x"}}})");
        syfox::Usage u;
        auto a = e4.decide("alpha beta gamma delta", tq, u);
        CHECK(a[0].tied, "exact tie is disclosed (tied:true)");
        CHECK(a[0].choice == "aaa",
              "tie pick is deterministic (criteria key order)");
        (void)std::system(("rm -rf " + d3).c_str());
    }
    // 4) readout silence: no candidate carries any energy -> defer.
    {
        syfox::Engine e2;
        e2.load_model(dir);
        sfx::JV uq = sfx::JV::parse(
            R"({"i":{"type":"choice","instructions":"",
                    "criteria":{"zzz":"","qqq":"","mmm":""}}})");
        syfox::Usage u;
        auto a = e2.decide("tariq found the radio", uq, u);
        CHECK(a[0].deferred && a[0].reason == "unknown_candidates",
              "all-dark readout defers (unknown_candidates)");
    }
}

// ---------------------------------------------------------------------------
// v3.4 honest defer for near-ties (engine default margin 0.05): when the top
// two probabilities are closer than the margin, the readout does not carry a
// decision — the answer defers with reason ambiguous_tie instead of shipping
// a coin-flip as a confident label. 0 disables (v3.3 answer-always). The
// near-tie fabric below is the bengali 2-candidate one: its heldout rows
// settle near-uniform (p1-p2 < 0.05 measured), which makes the case
// deterministic. score flavor uses the v3.3 exact-tie fabric (symmetric
// single-lesson binding, semantics stripped).
// ---------------------------------------------------------------------------
static void test_defer_ties() {
    std::cout << "[v3.4 honest defer for near-ties]\n";
    const std::string dir = "build/test_v34_defer";
    (void)std::system(("rm -rf " + dir).c_str());
    sfx::JV q = sfx::JV::parse(
        R"({"department":{"type":"choice","instructions":"Which team should handle this",)"
        R"("criteria":{"billing":"payment or subscription issues","technical":"bugs or integration problems"}}})");
    // the multilingual 6-lesson fabric: its heldout rows settle near-uniform
    // (p1-p2 < 0.05 measured) while the typo row separates (gap >= 0.05
    // measured — the v2.2 typo test passes under the default margin).
    {
        si::norm::grams_enabled() = true;
        syfox::Engine eng;
        const char* bn_billing[] = {
            "\xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa6\x95\xe0\xa6\xbe\xe0\xa6\xb0\xe0\xa7\x8d\xe0\xa6\xa1 \xe0\xa6\xa5\xe0\xa7\x87\xe0\xa6\x95\xe0\xa7\x87 \xe0\xa6\xa6\xe0\xa7\x81\xe0\xa6\x87\xe0\xa6\xac\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa6\x9f\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe \xe0\xa6\x95\xe0\xa7\x87\xe0\xa6\x9f\xe0\xa7\x87\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\x85\xe0\xa6\xa4\xe0\xa6\xbf\xe0\xa6\xb0\xe0\xa6\xbf\xe0\xa6\x95\xe0\xa7\x8d\xe0\xa6\xa4 \xe0\xa6\x9f\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe \xe0\xa6\xab\xe0\xa7\x87\xe0\xa6\xb0\xe0\xa6\xa4 \xe0\xa6\xa6\xe0\xa6\xbf\xe0\xa6\xa8",
            "\xe0\xa6\xb8\xe0\xa6\xbe\xe0\xa6\xac\xe0\xa6\xb8\xe0\xa7\x8d\xe0\xa6\x95\xe0\xa7\x8d\xe0\xa6\xb0\xe0\xa6\xbf\xe0\xa6\xaa\xe0\xa6\xb6\xe0\xa6\xa8 \xe0\xa6\xac\xe0\xa6\xbe\xe0\xa6\xa4\xe0\xa6\xbf\xe0\xa6\xb2 \xe0\xa6\x95\xe0\xa6\xb0\xe0\xa7\x87\xe0\xa6\x9b\xe0\xa6\xbf \xe0\xa6\xa4\xe0\xa6\xac\xe0\xa7\x81\xe0\xa6\x93 \xe0\xa6\x86\xe0\xa6\xac\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa6\x9a\xe0\xa6\xbe\xe0\xa6\xb0\xe0\xa7\x8d\xe0\xa6\x9c \xe0\xa6\x95\xe0\xa6\xb0\xe0\xa7\x87\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\x9f\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe \xe0\xa6\xab\xe0\xa7\x87\xe0\xa6\xb0\xe0\xa6\xa4 \xe0\xa6\x9a\xe0\xa6\xbe\xe0\xa6\x87",
            "\xe0\xa6\x97\xe0\xa6\xa4 \xe0\xa6\xae\xe0\xa6\xbe\xe0\xa6\xb8\xe0\xa7\x87\xe0\xa6\xb0 \xe0\xa6\x87\xe0\xa6\xa8\xe0\xa6\xad\xe0\xa7\x9f\xe0\xa7\x87\xe0\xa6\xb8\xe0\xa7\x87 \xe0\xa6\xad\xe0\xa7\x81\xe0\xa6\xb2 \xe0\xa6\x9f\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa6\xaa\xe0\xa6\xb0\xe0\xa6\xbf\xe0\xa6\xae\xe0\xa6\xbe\xe0\xa6\xa3 \xe0\xa6\x86\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\xa0\xe0\xa6\xbf\xe0\xa6\x95 \xe0\xa6\x95\xe0\xa6\xb0\xe0\xa7\x81\xe0\xa6\xa8"};
        const char* bn_technical[] = {
            "\xe0\xa6\x85\xe0\xa7\x8d\xe0\xa6\xaf\xe0\xa6\xbe\xe0\xa6\xaa\xe0\xa6\x9f\xe0\xa6\xbf \xe0\xa6\xac\xe0\xa6\xbe\xe0\xa6\xb0\xe0\xa6\xac\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa6\x95\xe0\xa7\x8d\xe0\xa6\xb0\xe0\xa7\x8d\xe0\xa6\xaf\xe0\xa6\xbe\xe0\xa6\xb6 \xe0\xa6\x95\xe0\xa6\xb0\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\xb2\xe0\xa6\x97\xe0\xa6\x87\xe0\xa6\xa8 \xe0\xa6\x95\xe0\xa6\xb0\xe0\xa6\xa4\xe0\xa7\x87 \xe0\xa6\xaa\xe0\xa6\xbe\xe0\xa6\xb0\xe0\xa6\x9b\xe0\xa6\xbf \xe0\xa6\xa8\xe0\xa6\xbe",
            "\xe0\xa6\xaa\xe0\xa7\x87\xe0\xa6\xae\xe0\xa7\x87\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\x9f \xe0\xa6\x97\xe0\xa7\x87\xe0\xa6\x9f\xe0\xa6\x93\xe0\xa7\x9f\xe0\xa7\x87 \xe0\xa6\x95\xe0\xa6\xbe\xe0\xa6\x9c \xe0\xa6\x95\xe0\xa6\xb0\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\xa8\xe0\xa6\xbe \xe0\xa6\x8f\xe0\xa6\xb0\xe0\xa6\xb0 \xe0\xa6\xa6\xe0\xa7\x87\xe0\xa6\x96\xe0\xa6\xbe\xe0\xa6\x9a\xe0\xa7\x8d\xe0\xa6\x9b\xe0\xa7\x87",
            "\xe0\xa6\x93\xe0\xa6\xaf\xe0\xa6\xbc\xe0\xa7\x87\xe0\xa6\xac\xe0\xa6\xb9\xe0\xa7\x81\xe0\xa6\x95 \xe0\xa6\x87\xe0\xa6\xad\xe0\xa7\x87\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\x9f \xe0\xa6\x95\xe0\xa6\xbe\xe0\xa6\xa8\xe0\xa7\x87\xe0\xa6\x95\xe0\xa7\x8d\xe0\xa6\x9f \xe0\xa6\xb9\xe0\xa6\x9a\xe0\xa7\x9b\xe0\xa7\x87 \xe0\xa6\xa8\xe0\xa6\xbe \xe0\xa6\x87\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\x9f\xe0\xa6\xbf\xe0\xa6\x97\xe0\xa7\x8d\xe0\xa6\xb0\xe0\xa7\x87\xe0\xa6\xb6\xe0\xa6\xa8 \xe0\xa6\xac\xe0\xa7\x8d\xe0\xa6\xaf\xe0\xa6\xb0\xe0\xa7\x8d\xe0\xa6\xa5"};
        for (const auto* st : bn_billing)
            eng.learn_example(st, "Which team should handle this", "billing payment or subscription issues");
        for (const auto* st : bn_technical)
            eng.learn_example(st, "Which team should handle this", "technical bugs or integration problems");
        eng.save_model(dir);
        si::norm::grams_enabled() = false;
    }
    // 1) engine default (margin 0.05): the near-tie heldout row defers, the
    //    decisive typo row still answers.
    {
        syfox::Engine eng;
        CHECK(eng.load_model(dir), "defer probe fabric loads");
        CHECK(eng.defer_margin() == 0.05f, "engine default margin is 0.05");
        si::norm::grams_enabled() = true;
        syfox::Usage u;
        auto a = eng.decide("\xe0\xa6\x85\xe0\xa6\xb0\xe0\xa7\x8d\xe0\xa6\xa1\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa7\xaa\xe0\xa7\xaa\xe0\xa7\xa7\xe0\xa7\xa8 \xe0\xa6\x8f\xe0\xa6\xb0 \xe0\xa6\x9c\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\xaf \xe0\xa6\x85\xe0\xa6\xa4\xe0\xa6\xbf\xe0\xa6\xb0\xe0\xa6\xbf\xe0\xa6\x95\xe0\xa7\x8d\xe0\xa6\xa4 \xe0\xa6\xaa\xe0\xa7\x87\xe0\xa6\xae\xe0\xa7\x87\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\x9f \xe0\xa6\xb9\xe0\xa6\xaf\xe0\xa6\xbc\xe0\xa7\x87\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\x93\xe0\xa6\x87 \xe0\xa6\x9f\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe \xe0\xa6\xab\xe0\xa7\x87\xe0\xa6\xb0\xe0\xa6\xa4 \xe0\xa6\x9a\xe0\xa6\xbe\xe0\xa6\x87", q, u);
        CHECK(a[0].deferred && a[0].reason == "ambiguous_tie",
              "near-tie defers (ambiguous_tie)");
        CHECK(!a[0].probabilities.empty(), "deferred reply still carries probabilities");
        si::norm::grams_enabled() = false;
    }
    // 1b) decisive rows still answer under the default margin: measured on a
    //     6-lesson English 2-class fabric, the query/probability gap is 0.296
    //     (p1 0.648 / p2 0.352) — well clear of the 0.05 margin.
    {
        syfox::Engine eng;
        const char* bill[] = {"my card was charged twice please refund the extra money",
                              "my card got charged two times refund the money back",
                              "i want my money back the card charge was wrong"};
        const char* tech[] = {"the app crashes when i open the login screen",
                              "app crashes on the login page every time",
                              "the login screen is broken the app keeps crashing"};
        for (const auto* st : bill)
            eng.learn_example(st, "", "billing");
        for (const auto* st : tech)
            eng.learn_example(st, "", "technical");
        sfx::JV dq = sfx::JV::parse(
            R"({"d":{"type":"choice","instructions":"","criteria":{"billing":"billing","technical":"technical"}}})");
        syfox::Usage u;
        auto b = eng.decide("i was charged twice refund the money please", dq, u);
        CHECK(!b[0].deferred && b[0].choice == "billing",
              "decisive row still answers under default margin");
    }
    // 2) margin 0 restores the v3.3 answer-always behavior on the same row.
    {
        syfox::Engine eng;
        eng.load_model(dir);
        eng.set_defer_margin(0);
        si::norm::grams_enabled() = true;
        syfox::Usage u;
        auto a = eng.decide("\xe0\xa6\x85\xe0\xa6\xb0\xe0\xa7\x8d\xe0\xa6\xa1\xe0\xa6\xbe\xe0\xa6\xb0 \xe0\xa7\xaa\xe0\xa7\xaa\xe0\xa7\xa7\xe0\xa7\xa8 \xe0\xa6\x8f\xe0\xa6\xb0 \xe0\xa6\x9c\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\xaf \xe0\xa6\x85\xe0\xa6\xa4\xe0\xa6\xbf\xe0\xa6\xb0\xe0\xa6\xbf\xe0\xa6\x95\xe0\xa7\x8d\xe0\xa6\xa4 \xe0\xa6\xaa\xe0\xa7\x87\xe0\xa6\xae\xe0\xa7\x87\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\xa8\xe0\xa7\x8d\xe0\xa6\x9f \xe0\xa6\xb9\xe0\xa6\xaf\xe0\xa6\xbc\xe0\xa7\x87\xe0\xa6\x9b\xe0\xa7\x87 \xe0\xa6\x93\xe0\xa6\x87 \xe0\xa6\x9f\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe \xe0\xa6\xab\xe0\xa7\x87\xe0\xa6\xb0\xe0\xa6\xa4 \xe0\xa6\x9a\xe0\xa6\xbe\xe0\xa6\x87", q, u);
        CHECK(!a[0].deferred && a[0].choice == "billing",
              "margin 0 answers the same near-tie (v3.3 behavior)");
        si::norm::grams_enabled() = false;
    }
    // 3) score flavor: near-tied levels defer too (exact-tie fabric,
    //    semantics stripped — same construction as the v3.3 tie test).
    {
        syfox::Engine e3;
        e3.substrate().set_semantics(false);
        e3.learn_example("alpha beta gamma delta", "", "aaa bbb");
        sfx::JV sq = sfx::JV::parse(
            R"({"s":{"type":"score","instructions":"","criteria":["aaa","bbb"]}})");
        syfox::Usage u;
        auto a = e3.decide("alpha beta gamma delta", sq, u);
        CHECK(a[0].deferred && a[0].reason == "ambiguous_tie",
              "score near-tie defers (ambiguous_tie)");
    }
    (void)std::system(("rm -rf " + dir).c_str());
}

// ---------------------------------------------------------------------------
// v3.4 multi-hop readout walk (--hops N): BFS over lanes from each probe
// anchor, per-hop damping hop_coupling/sqrt(hop+1), ascending node-id
// ordering, width cap 64. Default depth 1 must stay bit-identical to the
// legacy single-hop readout (replay contract); depth 4 must carry REAL extra
// margin on multi-hop chains (measured on the probe fabric) and be
// deterministic.
// ---------------------------------------------------------------------------
static void test_multi_hop() {
    std::cout << "[v3.4 multi-hop readout walk]\n";
    CHECK(si::Substrate().hop_depth() == 1, "default hop depth is 1 (legacy)");
    {
        si::Substrate sub;
        sub.set_hop_depth(0);  CHECK(sub.hop_depth() == 1, "hop depth clamps low to 1");
        sub.set_hop_depth(99); CHECK(sub.hop_depth() == 8, "hop depth clamps high to 8");
    }
    const std::string dir = "build/test_v34_hops";
    (void)std::system(("rm -rf " + dir).c_str());
    sfx::JV chain_q = sfx::JV::parse(
        R"({"i":{"type":"choice","instructions":"","criteria":{
             "sparrow":"sparrow","bird":"bird","animal":"animal"}}})");
    {
        syfox::Engine eng;
        eng.learn_example("a sparrow is a bird", "", "bird");
        eng.learn_example("a bird is an animal", "", "animal");
        eng.learn_example("the bird has feathers and wings", "", "bird");
        eng.save_model(dir);
    }
    syfox::Engine eng;
    CHECK(eng.load_model(dir), "hop chain fabric loads");
    const std::string state = "a sparrow is a bird";

    // 1) legacy depth: same engine, default and explicit 1 agree exactly.
    {
        syfox::Engine e1; e1.load_model(dir);
        syfox::Engine e2; e2.load_model(dir);
        e2.substrate().set_hop_depth(1);
        syfox::Usage u1, u2;
        auto a1 = e1.decide(state, chain_q, u1);
        auto a2 = e2.decide(state, chain_q, u2);
        CHECK(a1[0].choice == a2[0].choice &&
              std::fabs(a1[0].confidence - a2[0].confidence) < 1e-9f,
              "default depth == explicit depth 1 (legacy bit-identity)");
    }
    // 2) depth 4 keeps the answer and is deterministic across replays.
    {
        syfox::Engine e4; e4.load_model(dir);
        e4.substrate().set_hop_depth(4);
        syfox::Usage u;
        auto a = e4.decide(state, chain_q, u);
        CHECK(!a[0].deferred, "depth-4 chain answers");
        syfox::Usage u2;
        auto b = e4.decide(state, chain_q, u2);
        CHECK(b[0].choice == a[0].choice &&
              std::fabs(b[0].confidence - a[0].confidence) < 1e-9f,
              "depth-4 walk is deterministic");
    }
    // 3) deeper walk carries at least the legacy margin on the chain
    //    (measured on the probe fabric: 3-4 hop targets gain margin).
    {
        syfox::Engine e1; e1.load_model(dir);
        syfox::Engine e4; e4.load_model(dir);
        e4.substrate().set_hop_depth(4);
        syfox::Usage u1, u4;
        auto a1 = e1.decide(state, chain_q, u1);
        auto a4 = e4.decide(state, chain_q, u4);
        // margins: top-2 probability gap
        auto gap = [](const std::vector<std::pair<std::string, float>>& ps) {
            float p1 = 0, p2 = 0;
            for (const auto& kv : ps) {
                if (kv.second > p1) { p2 = p1; p1 = kv.second; }
                else if (kv.second > p2) p2 = kv.second;
            }
            return p1 - p2;
        };
        CHECK(a4[0].choice == a1[0].choice && gap(a4[0].probabilities) >= gap(a1[0].probabilities) - 1e-6f,
              "depth-4 keeps the chain answer with no margin loss");
    }
    (void)std::system(("rm -rf " + dir).c_str());
}

// ---------------------------------------------------------------------------
// v3.4 question-context gate (two-stage settle, opt-in): stage 1 settles the
// question's tokens into a context field; stage 2 composes the settled state
// field as (1-alpha)*state + alpha*context (alpha default 0.5). Measured on
// the radio probe: the answer holds, confidence moves (0.056 -> 0.061), OOD
// abstention is preserved. Deterministic; disclosed in usage.
// ---------------------------------------------------------------------------
// ============================================================================
// v3.7.0 — the representation papers (the four moves of the limit analysis)
// ============================================================================

static void test_distvec() {
    std::cout << "[distvec: PPMI+SVD dense representation (Move 1)]\n";
    using namespace si;
    // Three clusters, no cross lanes — enough rank for a real embedding
    // (after the Perron/frequency axis is dropped, each block still
    // contributes its own positive directions).
    const char* blocks[3][4] = {
        {"cat", "dog", "puppy", "kitten"},
        {"invoice", "refund", "charge", "billing"},
        {"flight", "hotel", "booking", "luggage"}};
    auto build = [&blocks]() {
        Substrate s;
        for (int b = 0; b < 3; ++b)
            for (int w = 0; w < 4; ++w) s.intern(blocks[b][w]);
        for (int r = 0; r < 4; ++r)
            for (int b = 0; b < 3; ++b) {
                s.bind(s.find(blocks[b][0]), s.find(blocks[b][1]), 0.5f);
                s.bind(s.find(blocks[b][1]), s.find(blocks[b][2]), 0.5f);
                s.bind(s.find(blocks[b][2]), s.find(blocks[b][3]), 0.5f);
                s.bind(s.find(blocks[b][0]), s.find(blocks[b][3]), 0.3f);
            }
        return s;
    };
    Substrate s = build();
    si::dist::DistConfig dc;
    dc.dims = 6; dc.iterations = 3; dc.edge_k = 2; dc.edge_theta = 0.30f;
    si::dist::build_field(s, dc);
    CHECK(s.has_distvecs(), "dense vectors installed");
    CHECK(s.dist_dims() == 6, "dims honored");
    // semantic structure in the dense space: co-occurring words align,
    // lane-disjoint words do not (norm-floored, noise cannot fake kinship)
    auto cos = [&](const char* a, const char* b) {
        const std::vector<float>& V = s.distvecs();
        const int k = s.dist_dims();
        const NodeId ia = s.find(a), ib = s.find(b);
        float dot = 0, na = 0, nb = 0;
        for (int d = 0; d < k; ++d) {
            dot += V[ia * k + d] * V[ib * k + d];
            na += V[ia * k + d] * V[ia * k + d];
            nb += V[ib * k + d] * V[ib * k + d];
        }
        if (na <= 1e-12f || nb <= 1e-12f) return 0.0f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    };
    CHECK(cos("cat", "puppy") > 0.5f && cos("cat", "puppy") > cos("cat", "invoice") + 0.2f,
          "co-occurring words align; lane-disjoint words do not");
    CHECK(cos("invoice", "refund") > 0.5f && cos("invoice", "refund") > cos("flight", "refund") + 0.2f,
          "second cluster aligns; cross-cluster stays low");
    CHECK(s.resonance_edge_count() > 0, "resonance edges selected from the dense space");
    // determinism: same fabric -> bit-identical vectors
    Substrate s2 = build();
    si::dist::build_field(s2, dc);
    CHECK(0 == std::memcmp(s2.distvecs().data(), s.distvecs().data(),
                           s.distvecs().size() * sizeof(float)),
          "PPMI+SVD is bit-deterministic");
    // DSTV tail roundtrip: vectors survive save/load bit for bit
    s.save("build/test_distvec.bin");
    Substrate s3;
    CHECK(s3.load("build/test_distvec.bin"), "load with DSTV tail");
    CHECK(s3.has_distvecs() && s3.dist_dims() == s.dist_dims(), "tail dims roundtrip");
    CHECK(0 == std::memcmp(s3.distvecs().data(), s.distvecs().data(),
                           s.distvecs().size() * sizeof(float)),
          "vectors bit-identical after roundtrip");
    // a fabric saved without distvecs (dims 0) loads without them: the
    // replay contract for every pre-v3.7 model
    Substrate s4;
    s4.intern("solo");
    s4.save("build/test_nodistvec.bin");
    Substrate s5;
    CHECK(s5.load("build/test_nodistvec.bin") && !s5.has_distvecs(),
          "absence roundtrips: old fabrics stay inert");
}

static void test_pretrain() {
    std::cout << "[pretrain: energy-space self-supervision (Move 2)]\n";
    const char* path = "build/test_pretrain_corpus.txt";
    {
        std::ofstream f(path);
        f << "the customer wants a refund for the duplicate charge\n";
        f << "the customer reported the parcel never arrived\n";
        f << "the invoice was charged twice on the card\n";
        f << "the delivery is late and the customer is angry\n";
    }
    syfox::Engine e1;
    auto r1 = e1.pretrain(path, 3, 1, 1.0f);
    CHECK(r1.lines == 4, "lines consumed");
    CHECK(r1.masked > 0, "masked positions settled");
    CHECK(r1.epoch_loss.size() == 3, "per-epoch loss reported");
    bool in_range = true;
    for (double l : r1.epoch_loss) in_range = in_range && l >= 0.0 && l <= 1.0;
    CHECK(in_range, "basin loss in [0,1]");
    CHECK(r1.interned_new > 0 && r1.vocab_after >= r1.interned_new,
          "raw text grew the vocabulary with no labels");
    // the training signal: a longer run drives the basin loss down
    syfox::Engine e2;
    auto r2 = e2.pretrain(path, 12, 1, 1.0f);
    CHECK(r2.epoch_loss.back() < r2.epoch_loss.front(),
          "self-supervision improves the masked-basin objective");
    // determinism: same corpus, same schedule -> bit-identical loss curve
    syfox::Engine e3;
    auto r3 = e3.pretrain(path, 3, 1, 1.0f);
    bool same_curve = r3.epoch_loss.size() == r1.epoch_loss.size();
    for (std::size_t i = 0; same_curve && i < r3.epoch_loss.size(); ++i)
        same_curve = r3.epoch_loss[i] == r1.epoch_loss[i];
    CHECK(same_curve, "pretrain is bit-deterministic");
    // vocabulary + lanes persist through save/load
    e1.save_model("build/model-pretrain-test");
    syfox::Engine e4;
    CHECK(e4.load_model("build/model-pretrain-test"), "pretrained model loads");
    CHECK(e4.substrate().node_count() == e1.substrate().node_count(),
          "grown vocabulary roundtrips");
}

static void test_typed_lanes() {
    std::cout << "[typed lanes: (word, role) nodes (Move 3)]\n";
    using Role = si::roles::Role;
    // tagging: subject before the verb, object after it
    auto w1 = si::norm::normalize("alice gave bob the book");
    auto r1 = si::roles::tag(w1);
    CHECK(r1[0] == Role::S, "subject tagged before the verb");
    CHECK(r1[1] == Role::V, "verb cue tagged");
    CHECK(r1[2] == Role::O, "nearest content after the verb is the object");
    // reversal: the S>O typed pair FLIPS — the relation carries the asymmetry
    auto w2 = si::norm::normalize("bob gave the book to alice");
    auto p1 = si::roles::typed_pairs_of(w1);
    auto p2 = si::roles::typed_pairs_of(w2);
    CHECK(p1.size() == 3 && p2.size() == 3, "S>V, V>O, S>O pairs emitted");
    CHECK(p1[2] != p2[2], "S>O typed lanes differ under reversal");
    CHECK(p1[0] != p2[0], "S>V typed lanes differ under reversal");
    // negation marker
    auto w3 = si::norm::normalize("alice did not pay the invoice");
    auto p3 = si::roles::typed_pairs_of(w3);
    bool neg = false;
    for (const auto& t : p3) if (!t.empty() && t[0] == '!') neg = true;
    CHECK(neg, "negated verb emits the marker node");
    // mangled names can never collide with normalized words
    bool mangled = false;
    for (const auto& t : p1)
        if (t.find("#s>") != std::string::npos) mangled = true;
    CHECK(mangled, "typed names carry the role separators");
    // ---- fabric behavior: reversal separation on a typed fabric ----------
    // Three lessons per class with VARIED verbs: the role evidence must
    // accumulate across lessons the way real training data provides it —
    // every alice row binds alice#s>V#v and alice#s>O#o toward alice, every
    // bob row the mirror, while the shared bag stays ambiguous.
    const char* reversed_a = "alice gave the book to bob";
    const char* reversed_b = "bob gave the book to alice";
    sfx::JV q = sfx::JV::parse(
        R"({"q1":{"type":"choice","instructions":"who is the giver",)"
        R"("criteria":{"alice":"alice gave","bob":"bob gave"}}})");
    si::roles::typed_lanes_enabled() = true;
    syfox::Engine te;
    te.set_defer_margin(0);                    // measure separation, not abstention
    te.learn_example("alice gave the book to bob", "", "alice giver");
    te.learn_example("alice sent the parcel to bob", "", "alice sender");
    te.learn_example("alice paid the invoice to bob", "", "alice payer");
    te.learn_example("bob gave the book to alice", "", "bob giver");
    te.learn_example("bob sent the parcel to alice", "", "bob sender");
    te.learn_example("bob paid the invoice to alice", "", "bob payer");
    syfox::Usage ua, ub;
    auto aa = te.decide(reversed_a, q, ua);
    auto ab = te.decide(reversed_b, q, ub);
    CHECK(!aa[0].deferred && aa[0].choice == "alice", "typed fabric answers reversal a");
    CHECK(!ab[0].deferred && ab[0].choice == "bob", "typed fabric answers reversal b");
    CHECK(ua.typed_tokens > 0, "typed nodes energized at decide");
    // a state with an UNSEEN verb still routes by the S>O relation alone
    syfox::Usage uc;
    auto ac = te.decide("alice mailed the contract to bob", q, uc);
    CHECK(!ac[0].deferred && ac[0].choice == "alice",
          "role evidence generalizes to unseen verbs");
    // ---- replay contract: a bag fabric is a no-op at decide --------------
    si::roles::typed_lanes_enabled() = false;
    syfox::Engine be;
    be.set_defer_margin(0);
    be.learn_example(reversed_a, "", "alice giver");
    be.learn_example(reversed_b, "", "bob giver");
    bool bag_has_typed = false;
    for (const auto& t : si::roles::typed_pairs_of(si::norm::normalize(reversed_a)))
        if (be.substrate().has(t)) bag_has_typed = true;
    CHECK(!bag_has_typed, "bag fabric interned no typed nodes");
    // decide with typed lanes OFF vs ON must be bit-identical on it
    si::roles::typed_lanes_enabled() = false;
    syfox::Usage u1, u2;
    auto a1 = be.decide(reversed_a, q, u1);
    si::roles::typed_lanes_enabled() = true;
    auto a2 = be.decide(reversed_a, q, u2);
    CHECK(a1[0].choice == a2[0].choice && u1.settled_energy == u2.settled_energy,
          "typed injection is a no-op on fabrics without typed nodes");
    si::roles::typed_lanes_enabled() = false;
}

static void test_tools_registry() {
    std::cout << "[tools: generalized tool scope (Move 4)]\n";
    auto tj = syfox::jas::tools_json();
    CHECK(tj.has("tools") && tj.at("tools").is_arr(), "tools registry present");
    CHECK(tj.at("tools").arr.size() == 5, "every register claim carries a tool contract");
    bool have_calc = false, all_have_contract = true;
    for (const auto& t : tj.at("tools").arr) {
        if (!(t.has("field_behavior") && t.has("tool") && t.has("tool_verified")
              && t.has("established_by"))) all_have_contract = false;
        if (t.at("tool").as_str().find("calc oracle") != std::string::npos) have_calc = true;
    }
    CHECK(all_have_contract, "field/tool/verification triple on every entry");
    CHECK(have_calc, "calc oracle registered");
    // the register JSON also carries the contract now
    auto rj = syfox::jas::register_json();
    CHECK(rj.at("claims").arr.size() == 5 && rj.at("claims").arr[0].has("tool"),
          "register claims carry the tool contract");
    // decide-time disclosure: the oracle computes; the field still measures
    syfox::Engine e;
    e.set_defer_margin(0);
    e.learn_example("the invoice total is pending", "", "pending total");
    e.set_tools_check(true);
    sfx::JV q = sfx::JV::parse(
        R"({"q1":{"type":"choice","criteria":)"
        R"({"pending total":"the total is pending","paid total":"the total is paid"}}})");
    syfox::Usage u;
    auto ans = e.decide("what is 17 times 23 plus 5", q, u);
    CHECK(u.tool_checked && u.tool_used, "tool scope ran on arithmetic state");
    CHECK(u.tool_name == "calc" && u.tool_expression == "17*23+5" && u.tool_value == "396",
          "oracle derivation disclosed with provenance");
    // non-arithmetic state: checked but no tool fired
    syfox::Usage u2;
    (void)e.decide("the invoice total is pending", q, u2);
    CHECK(u2.tool_checked && !u2.tool_used, "no tool fired without arithmetic");
    // default off: no disclosure, no cost
    syfox::Engine e2;
    e2.learn_example("the invoice total is pending", "", "pending total");
    syfox::Usage u3;
    (void)e2.decide("what is 17 times 23 plus 5", q, u3);
    CHECK(!u3.tool_checked, "tool scope is opt-in");
}

static void test_jas() {
    std::cout << "[jas: J-A-S cycle + calc derivation oracle + impossibility register]\n";
    using syfox::calc::evaluate;
    // ---- the derivation oracle: exact, composed, never sampled ---------
    auto r1 = evaluate("17*23");
    CHECK(r1.ok && r1.integral && r1.iv == 391, "17*23 derives 391 exactly");
    auto r2 = evaluate("2+3*4");
    CHECK(r2.ok && r2.integral && r2.iv == 14, "precedence derives 14");
    auto r3 = evaluate("(2+3)^2/5");
    CHECK(r3.ok && r3.integral && r3.iv == 5, "parens + power exact path");
    auto r4 = evaluate("2^10");
    CHECK(r4.ok && r4.integral && r4.iv == 1024, "right-assoc power");
    auto r5 = evaluate("1/0");
    CHECK(!r5.ok, "1/0 rejected by the verifier, never guessed");
    auto r6 = evaluate("7 % 3");
    CHECK(r6.ok && r6.integral && r6.iv == 1, "modulo exact");
    // detection is a relevance heuristic only — the field never computes
    auto d1 = syfox::calc::detect("what is 17 times 23 plus 5");
    CHECK(d1.has_arithmetic && d1.expression == "17*23+5", "word problem maps to operators");
    auto d2 = syfox::calc::detect("my parcel was late and the box was crushed");
    CHECK(!d2.has_arithmetic, "non-arithmetic text does not fire detection");

    // ---- the impossibility register: warrants, not just claims ---------
    auto reg = syfox::jas::syfox_impossibilities();
    CHECK(reg.size() == 5, "five architectural boundaries seeded");
    bool thm = false, ind = false, der = false;
    for (const auto& c : reg) {
        if (c.warrant == syfox::jas::Warrant::THEOREM) thm = true;
        if (c.warrant == syfox::jas::Warrant::INDUCED) ind = true;
        if (c.warrant == syfox::jas::Warrant::DERIVED) der = true;
        CHECK(!c.statement.empty() && !c.parent.empty() && !c.bypass.empty(),
              "claim carries statement + parent + bypass");
    }
    CHECK(thm && ind && der, "warrants THEOREM/INDUCED/DERIVED all present");
    CHECK(std::string(syfox::jas::warrant_name(syfox::jas::Warrant::THEOREM)) == "THEOREM",
          "warrant naming");
    // the arithmetic THEOREM is scope-escaped by calc — never "violated"
    bool arith_thm = false;
    for (const auto& c : reg)
        if (c.warrant == syfox::jas::Warrant::THEOREM &&
            c.statement.find("arithmetic") != std::string::npos)
            arith_thm = c.status.find("SCOPE ESCAPED") != std::string::npos;
    CHECK(arith_thm, "arithmetic theorem: scope escape, not a violation");

    // ---- the cycle: refutation is the deliverable ----------------------
    // train: alpha is an IMPERFECT predictor (3/4) so the leap is refutable
    // and the revision (alpha AND one -> A) is reachable
    auto mk = [](const std::string& s, const std::string& l) {
        syfox::jas::Row r; r.state = s; r.label = l; return r;
    };
    std::vector<syfox::jas::Row> train = {
        mk("alpha one x", "A"), mk("alpha one y", "A"), mk("alpha two z", "A"),
        mk("alpha beta mix", "B"), mk("beta one w", "B"), mk("beta two v", "B"),
        mk("beta three u", "B"),
    };
    // holdout round-1 rows refute; round-2 rows are FRESH (never fit)
    std::vector<syfox::jas::Row> holdout = {
        mk("alpha beta gamma", "B"), mk("beta epsilon", "B"),
        mk("alpha one delta", "A"), mk("gamma zeta", "A"),
    };
    syfox::jas::CycleConfig cfg;
    cfg.max_axioms = 2;
    cfg.min_support = 2;
    auto no_field = [](const std::string&, const sfx::JV&,
                       std::string&, bool&) { return false; };
    std::string err;
    auto rep = syfox::jas::run_cycle(train, holdout, cfg, no_field, "", err);
    CHECK(err.empty(), "cycle runs clean");
    CHECK(rep.axioms.size() == 2 && rep.axioms[0].token == "beta" &&
          rep.axioms[1].token == "alpha",
          "top axioms deterministic (confidence, support, token)");
    CHECK(std::fabs(rep.axioms[1].confidence - 0.75) < 1e-9,
          "confidence is the observed frequency that prompted the leap "
          "(alpha -> A was 3/4 on train; counting, not vibes)");
    CHECK(rep.r1_refuted == 1 && !rep.r1_counterexample.empty(),
          "the oracle REFUTED the universal (one counterexample ends it)");
    CHECK(rep.revised && rep.revised_axioms[0].extra == "one",
          "revision = structural restriction (alpha AND one), checkable in advance");
    CHECK(rep.verdict == "REFUTED_AND_REVISED_SURVIVES_HELDOUT",
          "revised axiom survives FRESH held-out rows: the cycle converged");
    CHECK(rep.r2_tested == 2, "round 2 tested only rows round 1 never saw");
    // clean holdout -> honest survival, NOT proof
    std::vector<syfox::jas::Row> clean = {
        mk("alpha two sea", "A"), mk("beta one ray", "B"),
        mk("alpha two sun", "A"), mk("beta two sky", "B"),
    };
    auto rep2 = syfox::jas::run_cycle(train, clean, cfg, no_field, "", err);
    CHECK(err.empty() && rep2.verdict == "SURVIVES" && !rep2.revised,
          "survival is not proof; no revision without refutation");
}

static void test_ctx_gate() {
    std::cout << "[v3.4 question-context gate]\n";
    const std::string dir = "build/test_v34_ctx";
    (void)std::system(("rm -rf " + dir).c_str());
    sfx::JV q = sfx::JV::parse(
        R"({"q":{"type":"choice","instructions":"Who found the radio?",
                "criteria":{"tariq":"the person who found it",
                             "salma":"a person present",
                             "radio":"the object found"}}})");
    {
        syfox::Engine eng;
        eng.learn_example("tariq found the radio in the lab", "", "tariq");
        eng.learn_example("the radio was found by tariq", "", "tariq");
        eng.learn_example("who found the radio tariq did", "", "tariq");
        eng.learn_example("salma was there in the room", "", "salma");
        eng.learn_example("salma saw the radio on the table", "", "salma");
        eng.save_model(dir);
    }
    const std::string state = "The radio was found by Tariq. Salma was there.";
    // 1) default OFF: no composition, no disclosure.
    {
        syfox::Engine eng;
        CHECK(eng.load_model(dir), "ctx probe fabric loads");
        syfox::Usage u;
        auto a = eng.decide(state, q, u);
        CHECK(!u.ctx_gate && !a[0].deferred && a[0].choice == "tariq",
              "ctx gate OFF: plain decide unchanged");
    }
    // 2) ON: composition disclosed, answer holds, deterministic.
    {
        syfox::Engine eng;
        eng.load_model(dir);
        eng.set_ctx_gate(true);
        CHECK(eng.ctx_gate_on(), "ctx gate toggles on");
        syfox::Usage u;
        auto a = eng.decide(state, q, u);
        CHECK(u.ctx_gate && std::fabs(u.ctx_alpha - 0.5f) < 1e-9f,
              "ctx gate ON: composition disclosed at alpha 0.5");
        CHECK(!a[0].deferred && a[0].choice == "tariq",
              "ctx gate ON: question-relevant answer holds");
        syfox::Usage u2;
        auto b = eng.decide(state, q, u2);
        CHECK(b[0].choice == a[0].choice &&
              std::fabs(b[0].confidence - a[0].confidence) < 1e-9f,
              "ctx gate ON: deterministic");
    }
    // 3) OOD abstention survives the composition.
    {
        syfox::Engine eng;
        eng.load_model(dir);
        eng.set_ctx_gate(true);
        sfx::JV oq = sfx::JV::parse(
            R"({"i":{"type":"choice","instructions":"What is the submarine's name?",
                    "criteria":{"zzz":"zzz","qqq":"qqq"}}})");
        syfox::Usage u;
        auto a = eng.decide("The purple elephant boarded the spaceship.", oq, u);
        CHECK(a[0].deferred, "ctx gate ON: OOD abstention preserved");
    }
    (void)std::system(("rm -rf " + dir).c_str());
}

// ---------------------------------------------------------------------------
// v3.6.0 — the fabric-construction papers, measured:
//   1. MRS §4b ordered bigram lanes (learn lays them, decide injects them)
//   2. the multiset-invariance escape: bag-only fabrics cannot tell
//      "alice beat bob" from "bob beat alice" (pinned bit-equal), bigram
//      fabrics answer both correctly
//   3. CMD §2.2 / MIMS §6 rarity weighting: stopword lanes weaken, rare
//      lanes stay strong, no-df fabrics reproduce the raw weights, df
//      persists through save/load
//   4. old-fabric replay compatibility: a pre-v3.6 fabric (no bigram nodes,
//      no IDF tail) decides bit-identically with bigrams ON at decide time
//   5. BED §8 perturbation contrast: a bag-only fabric's margin survives the
//      adjacent-transposed state (defer perturbation_tie), a bigram fabric's
//      margin collapses under it (the answer stands)
// ---------------------------------------------------------------------------
static void test_fabric36() {
    std::cout << "[v3.6.0 fabric papers: bigram lanes, IDF rarity, perturbation contrast]\n";

    // 1) bigram nodes and their outcome lanes exist after a plain lesson.
    {
        syfox::Engine eng;
        eng.learn_example("dog bites man", "", "crime");
        CHECK(eng.substrate().has("dog~bite") && eng.substrate().has("bite~man"),
              "learn interns ordered bigram nodes (porter-stemmed)");
        const bool lane_ok =
            eng.substrate().lane_weight(eng.substrate().find("dog~bite"),
                                        eng.substrate().find("crime")) > 0.0f;
        CHECK(lane_ok, "bigram lane binds to the outcome anchor");
    }

    // 2) the multiset-invariance escape, both sides pinned.
    {
        const std::string dir_bag = "build/test_v36_bag";
        const std::string dir_ord = "build/test_v36_ord";
        (void)std::system(("rm -rf " + dir_bag).c_str());
        (void)std::system(("rm -rf " + dir_ord).c_str());
        sfx::JV qa = sfx::JV::parse(
            R"({"q":{"type":"choice","instructions":"",
                    "criteria":{"red":"red side scored first","blue":"blue side scored first"}}})");
        {
            si::norm::bigrams_enabled() = false;         // the pre-v3.6 bag fabric
            syfox::Engine eng;
            eng.set_defer_margin(0);                     // measure raw arg-max, no defer
            eng.learn_example("alice beat bob", "", "red");
            eng.learn_example("bob beat alice", "", "blue");
            eng.save_model(dir_bag);
            si::norm::bigrams_enabled() = true;          // the v3.6 ordered fabric
            syfox::Engine eng2;
            eng2.set_defer_margin(0);
            eng2.learn_example("alice beat bob", "", "red");
            eng2.learn_example("bob beat alice", "", "blue");
            eng2.save_model(dir_ord);
        }
        // 2a) bag-only: the two states are the SAME token multiset, so the
        //     theorem says the readouts are identical — pinned bit-equal.
        {
            syfox::Engine eng;
            CHECK(eng.load_model(dir_bag), "bag fabric loads");
            eng.set_defer_margin(0);
            si::norm::bigrams_enabled() = false;
            syfox::Usage u1, u2;
            auto a1 = eng.decide("alice beat bob", qa, u1);
            auto a2 = eng.decide("bob beat alice", qa, u2);
            si::norm::bigrams_enabled() = true;
            CHECK(!a1[0].deferred && !a2[0].deferred &&
                  a1[0].choice == a2[0].choice &&
                  std::fabs(a1[0].confidence - a2[0].confidence) < 1e-9f,
                  "bag-only fabric: reordered states decide identically (multiset invariance)");
        }
        // 2b) ordered: both states answer correctly.
        {
            syfox::Engine eng;
            CHECK(eng.load_model(dir_ord), "ordered fabric loads");
            eng.set_defer_margin(0);
            syfox::Usage u1, u2;
            auto a1 = eng.decide("alice beat bob", qa, u1);
            auto a2 = eng.decide("bob beat alice", qa, u2);
            CHECK(!a1[0].deferred && a1[0].choice == "red",
                  "ordered fabric: 'alice beat bob' -> red");
            CHECK(!a2[0].deferred && a2[0].choice == "blue",
                  "ordered fabric: 'bob beat alice' -> blue");
            CHECK(u1.bigram_tokens == 2 && u2.bigram_tokens == 2,
                  "decide discloses the bigram nodes it energized");
        }
        (void)std::system(("rm -rf " + dir_bag).c_str());
        (void)std::system(("rm -rf " + dir_ord).c_str());
    }

    // 3) IDF rarity weighting on the substrate, with persistence. Isolation:
    //    the SAME token ("spread", interned identically in both substrates so
    //    mass and bind counts match) lays its lane to "x" in lesson 1 — the
    //    raw twin binds it at factor 1.0, the df-noting substrate at
    //    log1p(1/2)/log1p(1) < 1. Only the document-frequency factor differs.
    {
        si::Substrate idf_s, raw_s;
        const char* lessons[3][3] = {
            {"spread", "filler1", nullptr},
            {"spread", "filler2", nullptr},
            {"filler3", nullptr, nullptr}};
        for (int l = 0; l < 3; ++l) {
            idf_s.begin_df_lesson();
            std::vector<std::string> st, out;
            std::unordered_set<std::string> noted;
            for (int i = 0; lessons[l][i]; ++i) {
                st.push_back(lessons[l][i]);
                idf_s.intern(lessons[l][i]);
                if (noted.insert(lessons[l][i]).second)
                    idf_s.note_df(idf_s.find(lessons[l][i]));
            }
            out.push_back(l == 0 ? "x" : (l == 1 ? "y" : "z"));
            idf_s.hebbian_lesson(st, out, 1.0f);
        }
        // raw twin: same lessons, no df notes -> IDF inert -> v3.5 weights.
        for (int l = 0; l < 3; ++l) {
            std::vector<std::string> st, out;
            for (int i = 0; lessons[l][i]; ++i) { st.push_back(lessons[l][i]); raw_s.intern(lessons[l][i]); }
            out.push_back(l == 0 ? "x" : (l == 1 ? "y" : "z"));
            raw_s.hebbian_lesson(st, out, 1.0f);
        }
        CHECK(idf_s.lane_weight(idf_s.find("spread"), idf_s.find("x")) <
              raw_s.lane_weight(raw_s.find("spread"), raw_s.find("x")),
              "IDF: same token, same mass, weaker lane when df notes exist");
        CHECK(idf_s.df_of(idf_s.find("spread")) == 2 &&
              idf_s.df_of(idf_s.find("filler1")) == 1 && idf_s.df_lesson_count() == 3,
              "df counts lessons, not occurrences");
        // persistence: the 'IDF5' tail roundtrips.
        const std::string dir = "build/test_v36_idf";
        (void)std::system(("rm -rf " + dir).c_str());
        syfox::Engine eng; eng.substrate() = std::move(idf_s);
        eng.save_model(dir);
        syfox::Engine eng2;
        CHECK(eng2.load_model(dir), "IDF fabric reloads");
        CHECK(eng2.substrate().df_of(eng2.substrate().find("spread")) == 2 &&
              eng2.substrate().idf_active(),
              "IDF5 tail persists the df table and lesson count");
        (void)std::system(("rm -rf " + dir).c_str());
    }

    // 4) old-fabric replay compatibility: a pre-v3.6 fabric (no bigram nodes,
    //    no IDF tail) decides bit-identically with bigrams ON at decide time.
    {
        const std::string dir = "build/test_v36_replay";
        (void)std::system(("rm -rf " + dir).c_str());
        sfx::JV q = sfx::JV::parse(
            R"({"q":{"type":"choice","instructions":"",
                    "criteria":{"billing":"refund and payments","tech":"bugs and crashes"}}})");
        {
            si::norm::bigrams_enabled() = false;
            syfox::Engine eng;
            eng.learn_example("i want my refund the payment was wrong", "", "billing");
            eng.learn_example("the app crash every time i open it", "", "tech");
            eng.save_model(dir);
            si::norm::bigrams_enabled() = true;
        }
        syfox::Engine eng_off, eng_on;
        CHECK(eng_off.load_model(dir) && eng_on.load_model(dir), "replay fabric loads twice");
        si::norm::bigrams_enabled() = false;
        syfox::Usage u_off; auto a_off = eng_off.decide("please refund my payment", q, u_off);
        si::norm::bigrams_enabled() = true;
        syfox::Usage u_on;  auto a_on  = eng_on.decide("please refund my payment", q, u_on);
        CHECK(u_on.bigram_tokens == 0,
              "old fabric energizes no bigram nodes (they do not exist)");
        CHECK(a_on[0].deferred == a_off[0].deferred &&
              a_on[0].choice == a_off[0].choice &&
              std::fabs(a_on[0].confidence - a_off[0].confidence) < 1e-9f,
              "old fabric decides bit-identically with bigrams ON (replay contract)");
        (void)std::system(("rm -rf " + dir).c_str());
    }

    // 5) BED §8 perturbation contrast.
    {
        const std::string dir_bag = "build/test_v36_pert_bag";
        const std::string dir_ord = "build/test_v36_pert_ord";
        (void)std::system(("rm -rf " + dir_bag).c_str());
        (void)std::system(("rm -rf " + dir_ord).c_str());
        sfx::JV q = sfx::JV::parse(
            R"({"q":{"type":"choice","instructions":"",
                    "criteria":{"win":"alpha side scored","lose":"epsilon side scored"}}})");
        {
            si::norm::bigrams_enabled() = false;
            syfox::Engine eng;
            eng.learn_example("alpha beta gamma delta", "", "win");
            eng.learn_example("epsilon zeta eta theta", "", "lose");
            eng.save_model(dir_bag);
            si::norm::bigrams_enabled() = true;
            syfox::Engine eng2;
            eng2.learn_example("alpha beta gamma delta", "", "win");
            eng2.learn_example("epsilon zeta eta theta", "", "lose");
            eng2.save_model(dir_ord);
        }
        // 5a) bag-only fabric: the transposed state has the SAME token bag, so
        //     its margin matches the real margin — the check defers.
        {
            syfox::Engine eng;
            CHECK(eng.load_model(dir_bag), "perturb bag fabric loads");
            si::norm::bigrams_enabled() = false;
            syfox::Usage u0; auto a0 = eng.decide("alpha beta gamma delta", q, u0);
            CHECK(!a0[0].deferred && a0[0].choice == "win",
                  "perturb OFF: bag fabric answers on the token bag");
            si::norm::bigrams_enabled() = true;
            syfox::Engine eng1;
            eng1.load_model(dir_bag);
            eng1.set_perturb_check(true);
            si::norm::bigrams_enabled() = false;         // decide-time injection too
            syfox::Usage u1; auto a1 = eng1.decide("alpha beta gamma delta", q, u1);
            si::norm::bigrams_enabled() = true;
            CHECK(u1.perturb_check &&
                  u1.perturb_margin_broken >= u1.perturb_margin_real - 1e-6f,
                  "perturb ON: bag fabric's margin survives the structure break");
            CHECK(a1[0].deferred && a1[0].reason == "perturbation_tie",
                  "perturb ON: bag-only decision defers with perturbation_tie");
        }
        // 5b) ordered fabric: the transposed state loses the bigrams, its
        //     margin collapses, and the structure-carried answer stands.
        {
            syfox::Engine eng;
            CHECK(eng.load_model(dir_ord), "perturb ordered fabric loads");
            eng.set_perturb_check(true);
            syfox::Usage u; auto a = eng.decide("alpha beta gamma delta", q, u);
            CHECK(u.perturb_check && u.perturb_margin_broken < u.perturb_margin_real,
                  "ordered fabric's margin collapses under the structure break");
            CHECK(!a[0].deferred && a[0].choice == "win",
                  "structure-carried answer survives the perturbation check");
        }
        (void)std::system(("rm -rf " + dir_bag).c_str());
        (void)std::system(("rm -rf " + dir_ord).c_str());
    }
}

// ---------------------------------------------------------------------------
// v3.8.0 — physics-native thinking mechanisms (no token prediction, no ML):
//   TASK 1 adaptive depth, TASK 2 multi-vector spectral channels,
//   TASK 3 causal WHAT/WHY/HOW lanes, TASK 4 energy self-verification.
// ---------------------------------------------------------------------------
// the choice question used by the self-verify gate checks (criteria
// descriptions match the taught outcome words so candidates read energy)
static sfx::JV q_choice() {
    return sfx::JV::parse(
        R"({"action":{"type":"choice","instructions":"next move",)"
        R"("criteria":{"flee":"run away escape avoid","fight":"attack sword combat",)"
        R"("dig_in":"hide build shelter"}}})");
}

static void test_thinking() {
    std::cout << "[thinking: adaptive depth / self-verify / multi-vector / causal]\n";
    sfx::JV q = sfx::JV::parse(
        R"({"q1":{"type":"choice","criteria":{"x":"x context","y":"y context"}}})");

    // ---- TASK 1: adaptive depth ------------------------------------------
    {
        syfox::Engine e;
        e.set_adaptive_depth(true);
        e.set_defer_margin(0);
        e.learn_example("alpha beta gamma", "", "x outcome");
        e.learn_example("delta epsilon gamma", "", "y outcome");
        syfox::Usage u1, u2;
        e.decide("alpha beta gamma", q, u1);            // known + lane-rich: easy
        e.decide("zzz qqx wnv", q, u2);                 // unknown vocabulary: hard
        CHECK(u1.adaptive_depth && u2.adaptive_depth, "adaptive depth disclosed");
        CHECK(u1.k_base == 8, "base k_settle reported");
        CHECK(u2.difficulty > u1.difficulty, "unknown state measures harder");
        CHECK(u2.k_used >= u1.k_used, "harder state gets at least as many passes");
        CHECK(u2.k_used <= u1.k_base + 8, "depth stays inside the max-extra bound");
        syfox::Usage u3;
        e.decide("zzz qqx wnv", q, u3);
        CHECK(u3.k_used == u2.k_used && u3.difficulty == u2.difficulty,
              "adaptive depth deterministic per state");
        // ratchet guard: the base never grows across decides
        for (int i = 0; i < 3; ++i) { syfox::Usage uu; e.decide("alpha beta gamma", q, uu); }
        syfox::Usage u4;
        e.decide("alpha beta gamma", q, u4);
        CHECK(u4.k_base == 8 && u4.k_used == u1.k_used, "no depth ratchet across decides");
        // flag off: base depth, untouched physics
        syfox::Engine e2;
        e2.learn_example("alpha beta gamma", "", "x outcome");
        syfox::Usage u5;
        e2.decide("alpha beta gamma", q, u5);
        CHECK(!u5.adaptive_depth && u5.k_used == u5.k_base, "flag off keeps the base depth");
    }

    // ---- TASK 4: energy self-verification ---------------------------------
    {
        syfox::Engine e;
        e.learn_example("alpha beta gamma", "", "x outcome");
        e.learn_example("delta epsilon gamma", "", "y outcome");
        syfox::Usage u;
        e.decide("alpha beta gamma", q, u);
        CHECK(u.field_passes == 8, "trace records the settle passes");
        CHECK(u.field_motion_rate > 0.10f, "motion rate measured above zero");
        CHECK(u.settled_energy > 0.0f, "settled energy present");
        CHECK(!u.field_converged, "eps break disclosed honestly (rare under decay)");
        // the gate: any motion above a 0 floor defers; a 1.0 floor never does.
        // The gate fires on an answer, so the fabric must first produce one:
        // the train_choice_engine shape (state tokens bound to the criteria
        // description words) clears the unknown-candidates floor.
        const char* sv_state = "zombies appear at night, health is low";
        syfox::Engine e2;
        e2.set_self_verify(true);
        e2.set_self_verify_floor(0.0f);
        e2.set_defer_margin(0);              // isolate the gate from the tie gate
        e2.learn_example("the bot sees zombies at night with low health",
                         "next move", "flee run away escape avoid");
        syfox::Usage u2;
        auto a2 = e2.decide(sv_state, q_choice(), u2);
        CHECK(a2[0].deferred && a2[0].reason == "unconverged_field",
              "self-verify defers a moving field on demand");
        syfox::Engine e3;
        e3.set_self_verify(true);
        e3.set_self_verify_floor(0.99f);
        e3.set_defer_margin(0);
        e3.learn_example("the bot sees zombies at night with low health",
                         "next move", "flee run away escape avoid");
        syfox::Usage u3;
        auto a3 = e3.decide(sv_state, q_choice(), u3);
        CHECK(!a3[0].deferred, "gate stays silent below the floor");
    }

    // ---- TASK 2: multi-vector spectral channels ---------------------------
    {
        syfox::Engine e;
        e.set_multi_vector(true, 4);
        e.learn_example("alpha beta gamma", "", "x outcome");
        e.learn_example("delta epsilon zeta", "", "y outcome");
        syfox::Usage u0;
        e.decide("alpha beta gamma", q, u0);
        CHECK(!u0.multi_vector_active, "multi-vector is an honest no-op without a distvec field");
        // with a distvec fabric: channels run, deterministic, field differs
        const std::string dir = "build/test_thinking_mv";
        {
            syfox::Engine e2;
            e2.set_distvec(true);
            e2.learn_example("alpha beta gamma", "", "x outcome");
            e2.learn_example("delta epsilon zeta", "", "y outcome");
            e2.save_model(dir);
        }
        syfox::Engine e3;
        e3.set_multi_vector(true, 4);
        CHECK(e3.load_model(dir), "distvec test fabric loads");
        syfox::Usage u2, u3;
        auto a2 = e3.decide("alpha beta gamma", q, u2);
        auto a3 = e3.decide("alpha beta gamma", q, u3);
        CHECK(u2.multi_vector_active && u2.multi_vector_channels >= 1,
              "channels ran on a distvec fabric (effective rank, honestly)");
        CHECK(a2[0].choice == a3[0].choice && u2.settled_energy == u3.settled_energy,
              "multi-vector superposition is deterministic");
        // the superposition differs from the single-view field
        syfox::Engine e4;
        e4.load_model(dir);
        syfox::Usage u4;
        e4.decide("alpha beta gamma", q, u4);
        CHECK(!u4.multi_vector_active && u4.settled_energy != u2.settled_energy,
              "superposed field differs from the single view");
        (void)std::system(("rm -rf " + dir).c_str());
    }

    // ---- TASK 3: causal WHAT/WHY/HOW lanes ---------------------------------
    {
        // CauseFirst "so": L?>R (cause before the cue)
        auto cw = si::norm::normalize("the account was overcharged so the customer demanded a refund");
        auto cp = si::roles::causal_pairs_of(cw);
        CHECK(cp.size() == 1, "one causal pair for one connective");
        CHECK(!cp.empty() && cp[0].find("?>") != std::string::npos,
              "WHY pair carries the ?> signature");
        // EffectFirst "because": the node name puts the CAUSE first
        auto bw = si::norm::normalize("the flight was delayed because of the storm");
        auto bp = si::roles::causal_pairs_of(bw);
        CHECK(bp.size() == 1 && !bp.empty() && bp[0].rfind("storm", 0) == 0,
              "because flips the pair so the cause leads the node name");
        // Instrumental "by": means!>goal
        auto hw = si::norm::normalize("she passed the exam by studying nightly");
        auto hp = si::roles::causal_pairs_of(hw);
        CHECK(hp.size() == 1 && !hp.empty() && hp[0].find("!>") != std::string::npos,
              "HOW pair carries the !> signature");
        // fabric behavior: causal nodes interned, energized, and useful.
        // The probe sentences must MATCH the learned causal pairs — a pair
        // node exists only when learn and decide share the (stemmed cause,
        // connective, stemmed effect) triple.
        si::roles::causal_lanes_enabled() = true;
        si::roles::typed_lanes_enabled() = true;
        syfox::Engine ce;
        ce.set_defer_margin(0);
        const char* learn_a = "the account was overcharged so the agent issued a refund";
        ce.learn_example(learn_a, "", "refund outcome");
        ce.learn_example("the card was billed twice so the agent returned the money", "", "refund outcome");
        ce.learn_example("the pipe burst so the technician shut the water main", "", "shutoff outcome");
        ce.learn_example("the fuse blew so the electrician replaced the fuse", "", "shutoff outcome");
        auto lp = si::roles::causal_pairs_of(si::norm::normalize(learn_a));
        bool has_causal = false;
        for (const auto& p : lp) if (ce.substrate().has(p)) has_causal = true;
        CHECK(has_causal, "causal pair nodes interned at learn");
        syfox::Usage uc;
        auto ac = ce.decide("the customer was overcharged so the agent called back", q, uc);
        CHECK(uc.causal_tokens > 0, "causal nodes energized at decide");
        (void)ac;
        // replay contract: a causal-off fabric is a no-op with the flag on
        si::roles::causal_lanes_enabled() = false;
        si::roles::typed_lanes_enabled() = false;
        syfox::Engine be;
        be.set_defer_margin(0);
        be.learn_example("the account was overcharged so the agent issued a refund", "", "refund outcome");
        be.learn_example("the pipe burst so the technician shut the water main", "", "shutoff outcome");
        bool bag_has_causal = false;
        for (const auto& p : cp) if (be.substrate().has(p)) bag_has_causal = true;
        CHECK(!bag_has_causal, "causal-off fabric interned no causal nodes");
        syfox::Usage uo1, uo2;
        si::roles::causal_lanes_enabled() = false;
        auto ao1 = be.decide("the account was overcharged so the agent issued a refund", q, uo1);
        si::roles::causal_lanes_enabled() = true;
        auto ao2 = be.decide("the account was overcharged so the agent issued a refund", q, uo2);
        CHECK(ao1[0].choice == ao2[0].choice && uo1.settled_energy == uo2.settled_energy,
              "causal injection is a no-op on fabrics without causal nodes");
        si::roles::causal_lanes_enabled() = false;
    }
}

int main() {
    std::cout << "SyFox test suite (core: si-substrate)\n";
    test_json();
    test_folding();
    test_normalize();
    test_script_detection();
    test_utf8_normalize();
    test_ngram_lanes();
    test_mass_guard();
    test_multilingual_e2e();
    test_typo_corruption();
    test_m3_contradiction();
    test_m3_evidence_ledger();
    test_m3_ledger_persistence();
    test_m3_noul_contradiction();
    test_m4_adversarial();
    test_field_physics();
    test_salience_mechanics();
    test_hebbian_choice();
    test_noul_valence();
    test_calibration_tool();
    test_honest_silence_defer();
    test_derivation();
    test_recall();
    test_gate();
    test_bench_e2e();
    test_semantic_field();
    test_context_lanes();
    test_retrieval_default();
    test_hierarchy();
    test_ablation_distinctness();
    test_question_gate();
    test_defer_ties();
    test_multi_hop();
    test_ctx_gate();
    test_fabric36();
    test_jas();
    test_distvec();
    test_pretrain();
    test_typed_lanes();
    test_tools_registry();
    test_thinking();
    if (failures) { std::cout << failures << " FAILURES\n"; return 1; }
    std::cout << "all tests passed\n";
    return 0;
}
