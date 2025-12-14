#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <unordered_set>
#include <cassert>

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>

#include "octree.hpp"
#include "octree_knn.hpp"
#include "utils.hpp"  // if you have Point3D here; otherwise include where Point3D lives
#include "bruteForceKNN.cpp"


// ------------------------
// Helper: random points
// ------------------------
std::vector<Point3D> makeRandomPoints(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<Point3D> pts(n);
    for (auto &p : pts) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }
    return pts;
}

// ------------------------
// Helper: recall @ k
// ------------------------
float compute_average_recall_at_k(
    const std::vector<std::vector<int>>& exact,
    const std::vector<std::vector<int>>& approx)
{
    assert(exact.size() == approx.size());
    size_t Q = exact.size();
    if (Q == 0) return 1.0f;

    float sumRecall = 0.0f;

    for (size_t qi = 0; qi < Q; ++qi) {
        const auto& gt = exact[qi];
        const auto& ap = approx[qi];

        if (gt.empty()) {
            sumRecall += 1.0f;
            continue;
        }

        std::unordered_set<int> gtSet(gt.begin(), gt.end());
        int hit = 0;
        for (int idx : ap) {
            if (gtSet.count(idx)) hit++;
        }

        float recall = static_cast<float>(hit) /
                       static_cast<float>(gtSet.size());
        sumRecall += recall;
    }

    return sumRecall / static_cast<float>(Q);
}

// ------------------------
// Timing helpers
// ------------------------
using Clock = std::chrono::high_resolution_clock;

template <typename F>
double time_ms(F &&f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

// ------------------------
// Main benchmark
// ------------------------
int main(int argc, char** argv) {
    // Defaults; you can tweak from command line
    int N         = 100000;  // number of data points
    int Q         = 1000;    // number of queries
    int k         = 16;      // neighbors
    int leafSize  = 32;
    int maxDepth  = 10;
    int numThreads = 8;      // for TBB version

    if (argc > 1) N         = std::stoi(argv[1]);
    if (argc > 2) Q         = std::stoi(argv[2]);
    if (argc > 3) k         = std::stoi(argv[3]);
    if (argc > 4) leafSize  = std::stoi(argv[4]);
    if (argc > 5) maxDepth  = std::stoi(argv[5]);
    if (argc > 6) numThreads = std::stoi(argv[6]);

    std::cout << "Benchmark config:\n"
              << "  N          = " << N << "\n"
              << "  Q          = " << Q << "\n"
              << "  k          = " << k << "\n"
              << "  leafSize   = " << leafSize << "\n"
              << "  maxDepth   = " << maxDepth << "\n"
              << "  numThreads = " << numThreads << "\n\n";

    std::mt19937 rng(123);

    // Generate data + queries
    auto points  = makeRandomPoints(N, rng);
    auto queries = makeRandomPoints(Q, rng);

    // -----------------------------------
    // 1) Brute force (ground truth)
    // -----------------------------------
    std::vector<std::vector<int>> exact;
    double bf_ms = time_ms([&]() {
        exact = bruteForceKNN(points, queries, k);
    });
    std::cout << "[BruteForce]   query time: "
              << bf_ms << " ms  ("
              << (Q / (bf_ms / 1000.0)) << " qps)\n";

    // -----------------------------------
    // 2) Octree build
    // -----------------------------------
    Octree tree;
    double build_ms = time_ms([&]() {
        tree = buildOctree(points, leafSize, maxDepth);
    });
    std::cout << "[Octree]      build time: "
              << build_ms << " ms\n";

    // -----------------------------------
    // 3) Octree single-thread queries
    // -----------------------------------
    std::vector<std::vector<int>> octreeAns;
    double oct_ms = time_ms([&]() {
        octreeAns = octreeKNN(tree, queries, k);
    });
    float oct_recall = compute_average_recall_at_k(exact, octreeAns);

    std::cout << "[Octree ST]   query time: "
              << oct_ms << " ms  ("
              << (Q / (oct_ms / 1000.0)) << " qps), "
              << "avg recall@k = " << oct_recall << "\n";

    // -----------------------------------
    // 4) Octree parallel (TBB) scalability
    // -----------------------------------
    int maxNodesVisited = 256; // or whatever you use/tune
    std::cout << "\n[Octree TBB scalability]\n";

    for (int threads : {1, 2, 4, 8}) {
        // Limit TBB to 'threads' workers
        tbb::global_control ctrl(
            tbb::global_control::max_allowed_parallelism,
            threads
        );

        std::vector<std::vector<int>> octreeParAns;
        double oct_par_ms = time_ms([&]() {
            octreeParAns = octreeKNN_parallel_tbb(tree, queries, k, maxNodesVisited);
        });
        float oct_par_recall = compute_average_recall_at_k(exact, octreeParAns);

        std::cout << "  threads = " << threads
                << "  time = " << oct_par_ms << " ms  ("
                << (Q / (oct_par_ms / 1000.0)) << " qps), "
                << "speedup vs brute: " << (bf_ms / oct_par_ms) << "x, "
                << "recall = " << oct_par_recall << "\n";
    }

    // -----------------------------------
    // 5) Relative speedups
    // -----------------------------------
    std::cout << "\nSpeedups vs brute force (query time only):\n";
    std::cout << "  Octree ST  speedup: " << (bf_ms / oct_ms) << "x\n";
    //std::cout << "  Octree TBB speedup: " << (bf_ms / oct_par_ms) << "x\n";

    return 0;
}
