// compile probe for core/reason.hpp — builds the syllogism + insulin cases
#include "../core/reason.hpp"
#include <cassert>

int main() {
    using namespace sxr;
    ReasonCore rc;
    rc.add_default_rules();

    // 1. Socratic chain: socrates is_a man is_a mortal -> socrates is_a mortal
    rc.assert_fact("socrates", REL_IS_A, "man");
    rc.assert_fact("man", REL_IS_A, "mortal");
    auto rep = rc.reason_ext();
    assert(rc.graph.has_edge(rc.graph.by_label.at("socrates"),
                             rc.graph.by_label.at("mortal"), REL_IS_A));

    // 2. Query with proof trace: premise, premise, derived
    auto q = rc.query("socrates", REL_IS_A, "mortal");
    assert(q.holds && q.proof.size() == 3);

    // 3. Verifier accepts a fresh derivation
    assert(verify_edge(rc, q.proof_edge));

    // 4. Modus ponens: raining implies wet; raining -> wet
    ReasonCore rc2;
    rc2.add_default_rules();
    rc2.assert_fact("raining", REL_IMPLIES, "wet");
    rc2.assert_fact("raining", REL_IS_A, "weather");
    auto q2 = rc2.ask("raining", REL_IMPLIES, "wet");
    assert(q2.holds);

    // 5. Contradiction guard: whale is_a mammal; then NOT mammal refused
    ReasonCore rc3;
    rc3.add_default_rules();
    rc3.assert_fact("whale", REL_IS_A, "mammal");
    auto rep3 = teach_axiom(rc3, "whale", REL_NOT, "mammal", "test");
    assert(rep3.contradiction);

    // 6. Backward chaining proves a 3-hop goal forward rules can derive
    ReasonCore rc4;
    rc4.add_default_rules();
    rc4.assert_fact("a", REL_IS_A, "b");
    rc4.assert_fact("b", REL_IS_A, "c");
    rc4.assert_fact("c", REL_IS_A, "d");
    auto q4 = rc4.ask_backward_adaptive("a", REL_IS_A, "d");
    assert(q4.holds);

    // 7. Retraction is bounded: remove the middle premise -> derived dies
    // (find the man-is_a-mortal edge; retract; socrates derivation unsupported)
    ReasonCore rc5;
    rc5.add_default_rules();
    rc5.assert_fact("socrates", REL_IS_A, "man");
    rc5.assert_fact("man", REL_IS_A, "mortal");
    rc5.reason_ext();
    const int mid = rc5.graph.find_edge(rc5.graph.by_label.at("man"),
                                        rc5.graph.by_label.at("mortal"), REL_IS_A);
    const std::size_t killed = rc5.graph.retract(mid);
    assert(killed >= 2);   // the premise + the derived socrates edge
    assert(!rc5.graph.has_edge(rc5.graph.by_label.at("socrates"),
                               rc5.graph.by_label.at("mortal"), REL_IS_A));

    // 8. Discovery: ungrounded island found; grounded taxonomy not flagged
    ReasonCore rc6;
    rc6.add_default_rules();
    rc6.assert_fact("x1", REL_IS_A, "axiom");
    rc6.assert_fact("orphan1", REL_HAS, "orphan2");
    auto islands = find_isolated_subgraphs(rc6);
    assert(islands.size() == 1 && islands[0].nodes.size() == 2);

    // 9. Thinker index records composition
    ThinkerIndex ti;
    ti.record(q);
    assert(ti.total_queries == 1 && ti.retrieval_queries == 0);

    // 10. Axiom file round-trip (write, load, count)
    {
        FILE* f = std::fopen("build/reason_probe.axioms", "w");
        std::fprintf(f, "# probe\nlinux is_a operating_system\n"
                        "operating_system is_a software\n"
                        "linux eats whatever\n");   // unknown rel skipped
        std::fclose(f);
        ReasonCore rc7;
        rc7.add_default_rules();
        const int n = load_axiom_file(rc7, "build/reason_probe.axioms", nullptr);
        assert(n == 2);
        auto q7 = rc7.ask("linux", REL_IS_A, "software");
        assert(q7.holds);
    }

    std::printf("reason.hpp probe: ALL PASS\n");
    return 0;
}
