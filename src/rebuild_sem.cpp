// load a model dir, rebuild its semantic field, save to a NEW dir (no retrain)
#include "core/syfox.hpp"
#include <chrono>
#include <cstdio>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: rebuild_sem SRC DST\n"); return 2; }
    syfox::Engine eng;
    eng.load_model(argv[1]);
    const auto t0 = std::chrono::steady_clock::now();
    eng.save_model(argv[2]);   // finalize_contexts + build_semantics + save
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("rebuilt %s -> %s (%lld ms, nodes %zu, sem %d, edges %zu, ctx %zu)\n",
        argv[1], argv[2], (long long)ms, eng.substrate().node_count(),
        (int)eng.substrate().has_semantics(), eng.substrate().resonance_edge_count(),
        eng.substrate().lane_context_count());
    return 0;
}
