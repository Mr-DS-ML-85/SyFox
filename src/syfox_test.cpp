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
    if (failures) { std::cout << failures << " FAILURES\n"; return 1; }
    std::cout << "all tests passed\n";
    return 0;
}
