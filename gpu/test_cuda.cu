// test_cuda.cu

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_set>
#include <cassert>

#include <cuda_runtime.h>

// adjust to where Point3D lives
#include "../cpu/utils.hpp"

// forward-declare GPU function from bruteForceKNN.cu
std::vector<std::vector<int>> bruteForceKNN_GPU(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k,
    int blockSize);


// simple CPU brute-force baseline (no octree, just O(N*Q))
std::vector<std::vector<int>> bruteForceKNN_CPU(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k)
{
    int N = (int)points.size();
    int Q = (int)queries.size();
    std::vector<std::vector<int>> result(Q, std::vector<int>(k));

    auto dist2 = [](const Point3D& a, const Point3D& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return dx*dx + dy*dy + dz*dz;
    };

    const float INF = 1e30f;

    for (int qi = 0; qi < Q; ++qi) {
        const Point3D& q = queries[qi];

        std::vector<float> bestDist(k, INF);
        std::vector<int>   bestIdx (k, -1);

        for (int pi = 0; pi < N; ++pi) {
            float d2 = dist2(q, points[pi]);

            int   worstPos  = 0;
            float worstDist = bestDist[0];
            for (int j = 1; j < k; ++j) {
                if (bestDist[j] > worstDist) {
                    worstDist = bestDist[j];
                    worstPos  = j;
                }
            }

            if (d2 < worstDist) {
                bestDist[worstPos] = d2;
                bestIdx[worstPos]  = pi;
            }
        }

        result[qi] = bestIdx;
    }

    return result;
}

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

        float recall = (float)hit / (float)gtSet.size();
        sumRecall += recall;
    }

    return sumRecall / (float)Q;
}

template <typename F>
double time_ms(F&& f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}


struct ExperimentConfig {
    int N;
    int Q;
    int k;
};

// equality as sets (order-insensitive)
bool equal_sets(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    std::unordered_multiset<int> sa(a.begin(), a.end());
    std::unordered_multiset<int> sb(b.begin(), b.end());
    return sa == sb;
}

void run_experiment(const ExperimentConfig& cfg,
                    const std::vector<int>& blockSizes,
                    std::mt19937& rng)
{
    int N = cfg.N;
    int Q = cfg.Q;
    int k = cfg.k;

    std::cout << "\n=== Experiment: N=" << N
              << ", Q=" << Q
              << ", k=" << k << " ===\n";

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<Point3D> points(N), queries(Q);
    for (auto &p : points) {
        p.x = dist(rng);
        p.y = dist(rng);
        p.z = dist(rng);
    }
    for (auto &q : queries) {
        q.x = dist(rng);
        q.y = dist(rng);
        q.z = dist(rng);
    }

    std::vector<std::vector<int>> cpuRes;

    double cpu_ms = time_ms([&]() {
        cpuRes = bruteForceKNN_CPU(points, queries, k);
    });
    std::cout << "[CPU BF] time = " << cpu_ms << " ms\n";

    // For each block size, time GPU and check correctness
    for (int blockSize : blockSizes) {
        std::vector<std::vector<int>> gpuRes;
        double gpu_ms = time_ms([&]() {
            gpuRes = bruteForceKNN_GPU(points, queries, k, blockSize);
        });

        float recall = compute_average_recall_at_k(cpuRes, gpuRes);

        bool all_ok = true;
        // Bitwise/set correctness on a subset for speed (or all if you like)
        for (size_t qi = 0; qi < cpuRes.size(); ++qi) {
            if (!equal_sets(cpuRes[qi], gpuRes[qi])) {
                std::cerr << "  [bs=" << blockSize
                          << "] mismatch at query " << qi << "\n";
                all_ok = false;
                break;
            }
        }

        double speedup = cpu_ms / gpu_ms;

        std::cout << "  [GPU BF] blockSize=" << blockSize
                  << ", time = " << gpu_ms << " ms"
                  << ", speedup = " << speedup << "x"
                  << ", recall = " << recall
                  << ", " << (all_ok ? "neighbors match" : "MISMATCH")
                  << "\n";
    }
}

int main(int argc, char** argv) {
    std::mt19937 rng(123);

    // small + medium configs
    std::vector<ExperimentConfig> configs = {
        {20000, 1000,  8},   // small/medium
        {50000, 2000, 16},   // medium
    };

    // block sizes to sweep
    std::vector<int> blockSizes = {128, 256, 512};

    for (const auto& cfg : configs) {
        run_experiment(cfg, blockSizes, rng);
    }

    return 0;
}
