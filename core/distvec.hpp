// ============================================================================
//  SyFox — distvec.hpp  (v3.7.0, Move 1 of the AI-agent limit analysis)
//  ---------------------------------------------------------------------------
//  THE REPRESENTATION ESCAPE. The readout-side analysis of the fabric was
//  precise: the bag-of-words lane fabric is hand-written by the tokenizer,
//  and every node inherits that one choice — maximally sparse, and an unseen
//  word is a dark node. The 1990s answer (Levy & Goldberg 2014; Deerwester
//  et al. 1990 LSA), and it is NOT neural: PPMI + truncated SVD over the
//  co-occurrence structure.
//
//  What this file does, end to end, deterministic, zero dependencies:
//    1. PPMI TRANSFORM — the symmetric Hebbian lane fabric IS the
//       co-occurrence observation w(a,b) (lane weight = accumulated binding
//       mass between the two concepts). Pointwise mutual information
//       pmi = log( P(a,b) / P(a)P(b) ), clipped at 0 (PPMI), drops hub
//       pollution the raw counts carry: "the" co-occurs with everything,
//       and PPMI makes it pay through the marginal.
//    2. TRUNCATED SVD — subspace (block power) iteration on the symmetric
//       PPMI matrix W: X <- orth(W X), fixed iteration count, deterministic
//       hash initialization, modified Gram-Schmidt re-orthonormalization.
//       Rayleigh quotients give the eigenvalues (W symmetric => its
//       eigendecomposition IS its SVD); node vectors = U * Sigma^beta
//       (beta = 0.5, the Levy&Goldberg context-distributional scaling).
//       Every content node gets a dense k-dim vector — "emollient" lands
//       near "soothing" because they share contexts, with nobody labeling
//       either. This is a different OBJECT than the untyped multiset the
//       multiset-invariance theorem quantifies over — the scope escape.
//    3. RESONANCE EDGES — the semantic-resonance edges are re-selected from
//       the dense space (cos >= theta, top-k). The SETTLE physics is
//       untouched: edges are the same mechanism v3.2 shipped (sources leak
//       sem_coupling of retained energy along resonance edges, energy
//       conserved); only the EDGE SET is chosen by a representation that
//       actually generalizes. Small fabrics get the exact O(n^2) scan;
//       large fabrics get deterministic 2-hop lane candidates (pairs with
//       no lane-co-occurrence path have PPMI cosine ~0 by construction —
//       the standard inverted-index argument for embedding top-k).
//
//  Scope: representation + fabric construction, NOT dynamics. decay 0.82 /
//  diffusion 0.45 / dual-channel readout / Hebbian learning are untouched.
//  Persistence: a magic-guarded 'DSTV' tail in substrate.bin (vectors only;
//  the edges ride the existing SEM4 tail). Fabrics without the tail are
//  bit-identical to v3.6 — the replay contract.
//
//  Determinism contract: fixed iteration count (no convergence tolerance —
//  the approximation is the contract), hash-based initialization, MGS in
//  fixed column order, ties broken by (score desc, id asc). Same fabric,
//  same binary -> same vectors bit for bit.
// ============================================================================
#pragma once
#include "si_substrate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace si {
namespace dist {

struct DistConfig {
    int dims = 300;              // k: target embedding dimensionality
    int iterations = 5;          // subspace iterations (fixed, deterministic)
    float beta = 0.5f;           // vector = U * Sigma^beta
    bool drop_first = true;      // drop the Perron axis (see svd_embeddings)
    std::size_t exact_scan_max = 8000;   // n below this: exact top-k neighbour scan
    std::size_t cand_top_t = 32; // large fabrics: top-T lanes expand 2-hop candidates
    float edge_theta = 0.50f;    // min cosine for a resonance edge (= sem_theta)
    int edge_k = 6;              // resonance edges per node (= sem_neighbors)
};

// ----------------------------------------------------------------------------
// PPMI CSR over the lane fabric. Rows = nodes; entries = PPMI(a,b) > 0.
// Built from for_each_lane (public surface): each directed lane visited once,
// both directions exist (bind() is symmetric), so row sums are the marginals.
// ----------------------------------------------------------------------------
struct Ppmi {
    std::vector<std::size_t> off;    // n+1
    std::vector<NodeId>      dst;
    std::vector<float>       val;
    std::vector<float>       row_sum; // marginal of each node
    double total = 0.0;
};

inline Ppmi build_ppmi(const Substrate& s) {
    const std::size_t n = s.node_count();
    Ppmi p;
    p.off.assign(n + 1, 0);
    p.row_sum.assign(n, 0.0f);
    // pass 1: marginals (directed lanes; each undirected pair counted twice
    // overall, which cancels in the PMI ratio exactly as in Levy&Goldberg)
    s.for_each_lane([&](NodeId a, NodeId, float w) {
        p.row_sum[a] += w;
        p.total += static_cast<double>(w);
    });
    // pass 2: count surviving entries per source row, then fill. Entries of a
    // row stay in for_each_lane visit order (ascending insertion order of the
    // per-source lane vector) — deterministic.
    std::vector<std::size_t> count(n, 0);
    s.for_each_lane([&](NodeId a, NodeId b, float w) {
        if (a == b || w <= 0.0f || p.row_sum[a] <= 0.0f || p.row_sum[b] <= 0.0f) return;
        const double pmi =
            std::log((static_cast<double>(w) * p.total)
                     / (static_cast<double>(p.row_sum[a]) * static_cast<double>(p.row_sum[b])));
        if (pmi > 0.0) ++count[a];
    });
    for (std::size_t i = 0; i < n; ++i) p.off[i + 1] = p.off[i] + count[i];
    p.dst.resize(p.off[n]);
    p.val.resize(p.off[n]);
    std::vector<std::size_t> cur(p.off.begin(), p.off.end() - 1);
    s.for_each_lane([&](NodeId a, NodeId b, float w) {
        if (a == b || w <= 0.0f || p.row_sum[a] <= 0.0f || p.row_sum[b] <= 0.0f) return;
        const double pmi =
            std::log((static_cast<double>(w) * p.total)
                     / (static_cast<double>(p.row_sum[a]) * static_cast<double>(p.row_sum[b])));
        if (pmi <= 0.0) return;
        const std::size_t kk = cur[a]++;
        p.dst[kk] = b;
        p.val[kk] = static_cast<float>(pmi);
    });
    return p;
}

// ----------------------------------------------------------------------------
// Modified Gram-Schmidt, in-place, columns d = 0..k-1 of an n*k row-major
// matrix. Columns processed in ascending order (fixed); rank-deficient
// columns are zeroed (their eigenvector slot stays empty — deterministic).
// Inner row loops parallelize safely: each row entry is updated independently.
// ----------------------------------------------------------------------------
inline void mgs_orthonormalize(std::vector<float>& X, std::size_t n, std::size_t k) {
    for (std::size_t d = 0; d < k; ++d) {
        for (std::size_t pr = 0; pr < d; ++pr) {
            float dot = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:dot)
#endif
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
                dot += X[static_cast<std::size_t>(ii) * k + pr] * X[static_cast<std::size_t>(ii) * k + d];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
                X[static_cast<std::size_t>(ii) * k + d] -= dot * X[static_cast<std::size_t>(ii) * k + pr];
        }
        float n2 = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:n2)
#endif
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
            n2 += X[static_cast<std::size_t>(ii) * k + d] * X[static_cast<std::size_t>(ii) * k + d];
        if (n2 <= 1e-12f) {
            // RANK REPAIR (measured necessity): the ±1 hash basis can be
            // singular for small n — a collapsed column stays zero forever
            // (W·0 = 0) and the subspace silently loses rank. Replace it
            // deterministically: the first canonical basis vector whose
            // residual against columns 0..d-1 is well-conditioned.
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t i = 0; i < n; ++i)
                    X[i * k + d] = (i == r) ? 1.0f : 0.0f;
                for (std::size_t pr = 0; pr < d; ++pr) {
                    float dot = 0.0f;
                    for (std::size_t i = 0; i < n; ++i)
                        dot += X[i * k + pr] * X[i * k + d];
                    for (std::size_t i = 0; i < n; ++i)
                        X[i * k + d] -= dot * X[i * k + pr];
                }
                float n2r = 0.0f;
                for (std::size_t i = 0; i < n; ++i)
                    n2r += X[i * k + d] * X[i * k + d];
                if (n2r > 1e-6f) {
                    const float inv = 1.0f / std::sqrt(n2r);
                    for (std::size_t i = 0; i < n; ++i) X[i * k + d] *= inv;
                    break;
                }
            }
        } else {
            const float inv = 1.0f / std::sqrt(n2);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii)
                X[static_cast<std::size_t>(ii) * k + d] *= inv;
        }
    }
}

// ----------------------------------------------------------------------------
// Jacobi eigenvalue decomposition of a small symmetric k x k matrix
// (row-major, destroyed on exit). Deterministic: cyclic sweep order p<q,
// convergence when the off-diagonal norm drops below 1e-12 (or 60 sweeps).
// Returns eigenvalues (unsorted) and the eigenvector matrix as COLUMNS.
// ----------------------------------------------------------------------------
inline void jacobi_eigen(std::vector<float>& A, std::size_t k,
                         std::vector<float>& evals, std::vector<float>& evecs) {
    evecs.assign(k * k, 0.0f);
    for (std::size_t i = 0; i < k; ++i) evecs[i * k + i] = 1.0f;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (std::size_t p = 0; p < k; ++p)
            for (std::size_t q = p + 1; q < k; ++q)
                off += static_cast<double>(A[p * k + q]) * static_cast<double>(A[p * k + q]);
        if (off < 1e-20) break;
        for (std::size_t p = 0; p < k; ++p) {
            for (std::size_t q = p + 1; q < k; ++q) {
                const float apq = A[p * k + q];
                if (std::fabs(apq) < 1e-12f) continue;
                const double app = A[p * k + p], aqq = A[q * k + q];
                const double theta = (aqq - app) / (2.0 * static_cast<double>(apq));
                const double t = (theta >= 0.0 ? 1.0 : -1.0)
                               / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (std::size_t i = 0; i < k; ++i) {          // two-sided rotation
                    const float aip = A[i * k + p], aiq = A[i * k + q];
                    A[i * k + p] = static_cast<float>(c * aip - s * aiq);
                    A[i * k + q] = static_cast<float>(s * aip + c * aiq);
                }
                for (std::size_t j = 0; j < k; ++j) {
                    const float apj = A[p * k + j], aqj = A[q * k + j];
                    A[p * k + j] = static_cast<float>(c * apj - s * aqj);
                    A[q * k + j] = static_cast<float>(s * apj + c * aqj);
                }
                for (std::size_t i = 0; i < k; ++i) {          // eigenvector update
                    const float vip = evecs[i * k + p], viq = evecs[i * k + q];
                    evecs[i * k + p] = static_cast<float>(c * vip - s * viq);
                    evecs[i * k + q] = static_cast<float>(s * vip + c * viq);
                }
            }
        }
    }
    evals.resize(k);
    for (std::size_t d = 0; d < k; ++d) evals[d] = A[d * k + d];
}

// ----------------------------------------------------------------------------
// Subspace iteration: the PSD truncation of the symmetric PPMI matrix.
// X <- orth(W X), `iterations` times, from a deterministic ±1 hash basis,
// iterated in a slightly OVERSAMPLED subspace (k + pad — standard randomized-
// SVD practice) so degenerate ±|lambda| pairs cannot crowd a wanted direction
// out of the basis (measured: the 2x2 block [[0,a],[a,0]] has eigvals +a and
// -a of equal magnitude; without padding the iteration can return only the
// -a direction and an anti-aligned embedding pair results).
//
// RAYLEIGH-RITZ: after the walk, the k x k projection R = X^T W X is
// eigendecomposed with CYCLIC JACOBI and X is rotated by the Ritz vectors.
// Plain subspace iteration never separates a degenerate +a/-a pair (their
// |lambda| gap is zero) — the SPACE converges but individual columns keep
// mixing the pair, which zeroes their Rayleigh quotients (measured). The
// Ritz rotation resolves the space into true eigenvectors exactly.
//
// SELECTION is PSD: Ritz values rank ALGEBRAIC descending, only theta > 0
// directions are kept, scaled by theta^beta. Negative directions of a PPMI
// matrix encode mutual-avoidance (anti-correlation) — a different relation
// than the resonance the edge set needs. For a SYMMETRIC matrix the
// eigendecomposition IS the SVD; positive columns of U are exactly the
// singular vectors of the PSD part.
//
// drop_first: the largest-theta axis of a nonnegative matrix is the Perron
// vector — strictly positive, proportional to each word's total PPMI mass,
// i.e. a FREQUENCY axis (how much a word occurs, not what it means). It
// dominates every cosine at low k and collapses block structure onto the
// all-positive direction. LSA practice drops it; the engine already encodes
// the frequency axis separately (omega_semantic / semantic_freq).
//
// Vectors come back row-major (n*k_out) with k_out = min(cfg.dims, n);
// eigvals[d] publishes the theta behind each output column (0 for the
// zeroed Perron slot and any unfilled slot — the effective rank, honestly).
// ----------------------------------------------------------------------------
inline void svd_embeddings(const Substrate& s, const Ppmi& p, const DistConfig& cfg,
                           std::vector<float>& vecs, std::vector<float>& eigvals) {
    const std::size_t n = s.node_count();
    const std::size_t k_out = std::min<std::size_t>(static_cast<std::size_t>(std::max(1, cfg.dims)), n);
    vecs.assign(n * k_out, 0.0f);
    eigvals.assign(k_out, 0.0f);
    if (n == 0 || p.val.empty()) return;

    // working subspace width. Small fabrics (n <= 512) get the FULL space:
    // the Rayleigh-Ritz projection is then an EXACT eigendecomposition and
    // the iteration count is irrelevant — this is what keeps thin-spectrum
    // fabrics (few positive directions) from losing dims to under-convergence.
    // Large fabrics iterate the oversampled subspace k_out + pad; only the
    // top positive directions are selected anyway, and those converge fast.
    const std::size_t pad = std::max<std::size_t>(4, k_out / 4);
    const std::size_t k = (n <= 512) ? n : std::min(n, k_out + pad);

    // deterministic ±1 initialization, orthonormalized. The raw FNV bit is
    // NOT enough: high bits of the plain FNV-1a chain correlate for short
    // ASCII inputs (measured: every init column collapsed onto one pattern
    // and MGS zeroed the subspace), so the hash runs through the splitmix64
    // finalizer first — a fixed, dependency-free bit mixer.
    std::vector<float> X(n * k, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const std::string base = s.concept_of(static_cast<NodeId>(i));
        for (std::size_t d = 0; d < k; ++d) {
            std::uint64_t h = 1469598103934665603ull;
            const std::string seed = base + "#sv#" + std::to_string(d);
            for (unsigned char c : seed) { h ^= c; h *= 1099511628211ull; }
            h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
            h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
            h ^= h >> 33;
            X[i * k + d] = ((h >> 56) & 1u) ? 1.0f : -1.0f;
        }
    }
    mgs_orthonormalize(X, n, k);

    // fixed-iteration subspace walk with TWO buffers (no aliasing):
    // Y = W X (read basis, write scratch), then X = orth(Y).
    std::vector<float> Y(n * k, 0.0f);
    for (int it = 0; it < cfg.iterations; ++it) {
        std::fill(Y.begin(), Y.end(), 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const std::size_t b = p.off[i], e = p.off[i + 1];
            if (b == e) continue;
            float* out = &Y[i * k];
            for (std::size_t t = b; t < e; ++t) {
                const float w = p.val[t];
                const float* row = &X[static_cast<std::size_t>(p.dst[t]) * k];
                for (std::size_t d = 0; d < k; ++d) out[d] += w * row[d];
            }
        }
        X.swap(Y);
        mgs_orthonormalize(X, n, k);
    }

    // RAYLEIGH-RITZ: R = X^T W X (k x k symmetric), Jacobi-eigendecomposed,
    // and the basis rotated by the Ritz vectors. This is what separates the
    // degenerate pairs the plain walk cannot.
    std::vector<float> ritz_vals, ritz_vecs;
    {
        std::vector<float> Z(n * k, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const std::size_t b = p.off[i], e = p.off[i + 1];
            if (b == e) continue;
            float* out = &Z[i * k];
            for (std::size_t t = b; t < e; ++t) {
                const float w = p.val[t];
                const float* row = &X[static_cast<std::size_t>(p.dst[t]) * k];
                for (std::size_t d = 0; d < k; ++d) out[d] += w * row[d];
            }
        }
        std::vector<float> R(k * k, 0.0f);
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t b2 = a; b2 < k; ++b2) {
                float s = 0.0f;
                for (std::size_t i = 0; i < n; ++i)
                    s += X[i * k + a] * Z[i * k + b2];
                R[a * k + b2] = s;
                R[b2 * k + a] = s;                           // symmetric by W
            }
        jacobi_eigen(R, k, ritz_vals, ritz_vecs);
        // rotate the basis: X <- X * S  (n*k times k*k)
        std::vector<float> A(n * k, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            for (std::size_t d = 0; d < k; ++d) {
                float s = 0.0f;
                for (std::size_t j = 0; j < k; ++j)
                    s += X[i * k + j] * ritz_vecs[j * k + d];
                A[i * k + d] = s;
            }
        }
        X.swap(A);
    }

    // PSD selection: Ritz values ALGEBRAIC descending (positive directions
    // first — negative directions of a PPMI matrix encode mutual avoidance,
    // not resonance), keep theta > 0, skip the Perron/frequency axis, fill
    // k_out output columns; unfilled slots stay zero (effective rank,
    // published honestly through eigvals).
    std::vector<std::size_t> order(k);
    for (std::size_t d = 0; d < k; ++d) order[d] = d;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return ritz_vals[a] > ritz_vals[b];
    });
    std::size_t out_d = 0;
    bool perron_skipped = false;
    const bool drop_perron = cfg.drop_first;
    for (std::size_t t = 0; t < k && out_d < k_out; ++t) {
        const std::size_t src = order[t];
        if (ritz_vals[src] <= 1e-8f) break;                  // PSD part exhausted
        if (drop_perron && !perron_skipped) { perron_skipped = true; continue; }
        const float scale = std::pow(ritz_vals[src], cfg.beta);
        for (std::size_t i = 0; i < n; ++i) {
            const bool iso_row = (p.off[i + 1] == p.off[i]);
            vecs[i * k_out + out_d] =
                iso_row ? 0.0f : X[i * k + src] * scale;
        }
        eigvals[out_d] = ritz_vals[src];
        ++out_d;
    }
    // any unfilled slots stay zero: effective rank is visible in eigvals
}

// ----------------------------------------------------------------------------
// Resonance-edge selection in the dense space.
//   small fabrics (n <= exact_scan_max): exact scan over all pairs;
//   large fabrics: per node, expand the top-T lane neighbours' own top-T
//   lanes as candidates (2-hop closure), dedupe, then cosine-filter.
// Returns the CSR (offs n+1, edges (target, cos)) in the SEM4-tail format.
// Per-row independent -> OMP over rows; ties break by (cos desc, id asc).
// ----------------------------------------------------------------------------
inline void build_edges(const Substrate& s, const std::vector<float>& vecs,
                        std::size_t k, const DistConfig& cfg,
                        std::vector<std::pair<NodeId, float>>& edges,
                        std::vector<std::size_t>& offs) {
    const std::size_t n = s.node_count();
    offs.assign(n + 1, 0);
    edges.clear();
    if (n == 0 || k == 0 || vecs.size() < n * k) return;
    const std::size_t kmax = static_cast<std::size_t>(std::max(1, cfg.edge_k));
    const float theta = cfg.edge_theta;

    auto cos_at = [&](std::size_t i, std::size_t j) -> float {
        const float* a = &vecs[i * k];
        const float* b = &vecs[j * k];
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (std::size_t d = 0; d < k; ++d) {
            dot += a[d] * b[d];
            na += a[d] * a[d];
            nb += b[d] * b[d];
        }
        // norm floor: two numerically-empty vectors must not read as
        // parallel through rounding dust
        if (na <= 1e-12f || nb <= 1e-12f) return 0.0f;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    };

    auto better = [](const std::pair<float, NodeId>& x, const std::pair<float, NodeId>& y) {
        return x.first != y.first ? x.first > y.first : x.second < y.second;
    };

    // per-node candidate sets (exact scan or 2-hop closure)
    std::vector<std::vector<std::pair<float, NodeId>>> per(n);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        float na2 = 0.0f;
        for (std::size_t d = 0; d < k; ++d) na2 += vecs[i * k + d] * vecs[i * k + d];
        if (na2 <= 0.0f) continue;                       // dark node: no resonance
        std::vector<std::pair<float, NodeId>> best;
        auto push = [&](std::size_t j) {
            if (j == i) return;
            const float c = cos_at(i, j);
            if (c < theta) return;
            if (best.size() < kmax) {
                best.emplace_back(c, static_cast<NodeId>(j));
                std::sort(best.begin(), best.end(), better);
            } else if (better({c, static_cast<NodeId>(j)}, best.back())) {
                best.back() = {c, static_cast<NodeId>(j)};
                std::sort(best.begin(), best.end(), better);
            }
        };
        if (n <= cfg.exact_scan_max) {
            for (std::size_t j = 0; j < n; ++j) push(j);
        } else {
            // deterministic 2-hop closure: own lanes (top-T by weight), then
            // each neighbour's top-T lanes. Duplicates deduped via the
            // `best` bound (a repeated candidate can only re-enter if it
            // beats the tail, which keeps the result identical).
            std::vector<std::pair<NodeId, float>> mine;
            s.lanes_of(static_cast<NodeId>(i), mine);
            std::sort(mine.begin(), mine.end(), [](const auto& x, const auto& y) {
                return x.second != y.second ? x.second > y.second : x.first < y.first;
            });
            if (mine.size() > cfg.cand_top_t) mine.resize(cfg.cand_top_t);
            std::vector<std::pair<NodeId, float>> second;
            for (const auto& l1 : mine) {
                push(static_cast<std::size_t>(l1.first));
                std::vector<std::pair<NodeId, float>> nb;
                s.lanes_of(l1.first, nb);
                std::sort(nb.begin(), nb.end(), [](const auto& x, const auto& y) {
                    return x.second != y.second ? x.second > y.second : x.first < y.first;
                });
                if (nb.size() > cfg.cand_top_t) nb.resize(cfg.cand_top_t);
                for (const auto& l2 : nb) second.push_back(l2);
            }
            for (const auto& l2 : second) push(static_cast<std::size_t>(l2.first));
        }
        per[i] = std::move(best);
    }

    std::size_t running = 0;
    for (std::size_t i = 0; i < n; ++i) {
        offs[i] = running;
        running += per[i].size();
    }
    offs[n] = running;
    edges.reserve(running);
    for (std::size_t i = 0; i < n; ++i)
        for (const auto& pr : per[i]) edges.emplace_back(pr.second, pr.first);
}

// ----------------------------------------------------------------------------
// One-call build: PPMI -> SVD -> edges, installed on the substrate.
// Vectors persist via the DSTV tail; edges replace the SEM4 resonance set.
// ----------------------------------------------------------------------------
inline void build_field(Substrate& s, const DistConfig& cfg) {
    Ppmi p = build_ppmi(s);
    std::vector<float> vecs, eigvals;
    svd_embeddings(s, p, cfg, vecs, eigvals);
    const std::size_t k = eigvals.size();
    std::vector<std::pair<NodeId, float>> edges;
    std::vector<std::size_t> offs;
    build_edges(s, vecs, k, cfg, edges, offs);
    s.set_distvecs(std::move(vecs), static_cast<int>(k));
    s.set_semantic_edges(std::move(edges), std::move(offs));
}

} // namespace dist
} // namespace si
