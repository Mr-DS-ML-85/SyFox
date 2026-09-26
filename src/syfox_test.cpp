// ============================================================================
//  SyFox tests — assert-based, no framework.
//  Every test names the mechanism it protects.
// ============================================================================
#include "core/syfox.hpp"
#include "core/derive.hpp"
#include "core/bench.hpp"
#include "core/gate.hpp"
#include "core/recall.hpp"

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

int main() {
    std::cout << "SyFox test suite (core: si-substrate)\n";
    test_json();
    test_folding();
    test_normalize();
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
    if (failures) { std::cout << failures << " FAILURES\n"; return 1; }
    std::cout << "all tests passed\n";
    return 0;
}
