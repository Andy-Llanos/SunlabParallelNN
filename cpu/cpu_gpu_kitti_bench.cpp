#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cassert>
#include <unordered_set>
#include <fstream>

#include "utils.hpp"
#include "octree.hpp"
#include "octree_knn.hpp"
#include "kitti_loader.hpp"   // from above

//cpu brtueforce
//compile from project root:
///nvcc -std=c++17 -O3 \
    -Icpu -Igpu \
    cpu/cpu_gpu_kitti_bench.cpp \
    cpu/octree.cpp \
    cpu/octree_knn.cpp \
    gpu/bruteForceKNN.cu \
    -ltbb -lpthread \
    -o kitti_bench

//nvcc -std=c++17 -O3 -Icpu -Igpu cpu/cpu_gpu_kitti_bench.cpp cpu/octree.cpp cpu/octree_knn.cpp gpu/bruteForceKNN.cu -ltbb -lpthread -o kitti_bench
//run on frame:
//./kitti_bench data/kitti/velodyne/000000.bin results_kitti.csv 100000 2000 16

// GPU brute force declaration (from gpu/bruteForceKNN.cu)

// simple timing helper (you probably already have this)
template <typename F>
double time_ms(F&& f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

std::vector<std::vector<int>> bruteForceKNN_GPU(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k,
    int blockSize);

// Simple CPU brute-force baseline: exact KNN
std::vector<std::vector<int>> bruteForceKNN(
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



using Clock = std::chrono::high_resolution_clock;

template <typename F>
double time_ms(F&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

float compute_average_recall_at_k(
    const std::vector<std::vector<int>>& exact,
    const std::vector<std::vector<int>>& approx);


    
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <kitti_bin_file> <csv_output> [maxPoints] [Q] [k]\n";
        return 1;
    }
    std::string binPath = argv[1];
    std::string csvPath = argv[2];

    size_t maxPoints = (argc > 3) ? std::stoul(argv[3]) : 100000;
    int Q = (argc > 4) ? std::stoi(argv[4]) : 2000;
    int k = (argc > 5) ? std::stoi(argv[5]) : 16;

    // 1) Load KITTI
    auto points = loadKittiBin(binPath, maxPoints);
    int N = (int)points.size();
    if (N == 0) {
        std::cerr << "No points loaded from " << binPath << "\n";
        return 1;
    }
    std::cout << "Loaded " << N << " points from " << binPath << "\n";

    // 2) Generate queries as random subset of points
    if (Q > N) Q = N;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> idxDist(0, N - 1);

    std::vector<Point3D> queries(Q);
    for (int i = 0; i < Q; ++i) {
        queries[i] = points[idxDist(rng)];
    }

    // 3) Open CSV (append mode)
    std::ofstream csv(csvPath, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: cannot open CSV file: " << csvPath << "\n";
        return 1;
    }

    // optional: write header once (only if file empty; here we just always write)
    csv << "dataset,N,Q,k,leafSize,maxDepth,method,threads,blockSize,time_ms,speedup_vs_cpuBF,recall\n";
    //csv << "dataset,N,Q,k,method,time_ms,speedup_vs_cpuBF,recall\n";

    // 4) CPU brute-force (baseline)
    std::vector<std::vector<int>> bfRes;
    double bf_ms = time_ms([&]() {
        bfRes = bruteForceKNN(points, queries, k);
    });
    csv << binPath << "," << N << "," << Q << "," << k
        << "," << 0 << "," << 0 << ",cpu_bf," << 1 << "," << 0
        << "," << bf_ms << "," << 1.0 << "," << 1.0 << "\n";
    std::cout << "[CPU BF] " << bf_ms << " ms\n";

    // 5) CPU octree (sequential)
    int leafSize = 32;
    int maxDepth = 10;

    Octree tree = buildOctree(points, leafSize, maxDepth);
    std::vector<std::vector<int>> octSeq;
    double oct_seq_ms = time_ms([&]() {
        octSeq = octreeKNN(tree, queries, k);
    });
    float oct_seq_recall = compute_average_recall_at_k(bfRes, octSeq);
    double oct_seq_speedup = bf_ms / oct_seq_ms;

    csv << binPath << "," << N << "," << Q << "," << k
        << "," << leafSize << "," << maxDepth << ",cpu_octree_seq,1,0,"
        << oct_seq_ms << "," << oct_seq_speedup << "," << oct_seq_recall << "\n";
    std::cout << "[CPU Octree ST] " << oct_seq_ms << " ms, recall=" << oct_seq_recall << "\n";

    // 6) CPU octree (TBB parallel over queries)
    int maxNodesVisited = 256; // or whatever you use
    std::vector<std::vector<int>> octPar;
    double oct_par_ms = time_ms([&]() {
        octPar = octreeKNN_parallel_tbb(tree, queries, k, maxNodesVisited);
    });
    float oct_par_recall = compute_average_recall_at_k(bfRes, octPar);
    double oct_par_speedup = bf_ms / oct_par_ms;

    int threads = 8; // if you controlled via global_control earlier
    csv << binPath << "," << N << "," << Q << "," << k
        << "," << leafSize << "," << maxDepth << ",cpu_octree_tbb," << threads << ",0,"
        << oct_par_ms << "," << oct_par_speedup << "," << oct_par_recall << "\n";
    std::cout << "[CPU Octree TBB] " << oct_par_ms << " ms, recall=" << oct_par_recall << "\n";

    // 7) GPU brute-force
    int blockSize = 256;
    std::vector<std::vector<int>> gpuBF;
    double gpu_ms = time_ms([&]() {
        gpuBF = bruteForceKNN_GPU(points, queries, k, blockSize);
    });
    float gpu_recall = compute_average_recall_at_k(bfRes, gpuBF);
    double gpu_speedup = bf_ms / gpu_ms;

    csv << binPath << "," << N << "," << Q << "," << k
        << "," << 0 << "," << 0 << ",gpu_bf,0," << blockSize << ","
        << gpu_ms << "," << gpu_speedup << "," << gpu_recall << "\n";
    std::cout << "[GPU BF] " << gpu_ms << " ms, recall=" << gpu_recall << "\n";

    return 0;
}
