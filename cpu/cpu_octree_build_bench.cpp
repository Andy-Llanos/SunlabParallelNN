#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>

#include "utils.hpp"
#include "octree.hpp"
#include "kitti_loader.hpp"  // where loadKittiBin lives

using Clock = std::chrono::high_resolution_clock;

/// COMPILE IN ROOT, NOT cpu FOLDER
/// g++ -std=c++17 -O3 cpu/cpu_octree_build_bench.cpp cpu/octree.cpp -ltbb -lpthread -o cpu_octree_build_bench
/// EXECUTE :
/// ./cpu_octree_build_bench results_build.csv  (uses N=200k random)
/// ./cpu_octree_build_bench results_build.csv 100000
/// ./cpu_octree_build_bench results_build.csv file.bin

template <typename F>
double time_ms(F&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

// forward declarations (if not in header)
Octree buildOctree_seq(const std::vector<Point3D>& points,
                       int leafSize,
                       int maxDepth);

/// parallel construction version (your buildOctree == parallel8)
Octree buildOctree(const std::vector<Point3D>& points,
                   int leafSize,
                   int maxDepth);


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  ./cpu_octree_build_bench <csv_output> [N]\n"
                  << "  ./cpu_octree_build_bench <csv_output> file.bin [maxPoints]\n";
        return 1;
    }

    std::string csvPath = argv[1];
    int N = 200000;
    int leafSize = 32;
    int maxDepth = 10;
    std::string binPath;

    // Check second argument if present
    if (argc >= 3) {
        std::string arg2 = argv[2];
        if (arg2.size() > 4 && arg2.substr(arg2.size() - 4) == ".bin") {
            binPath = arg2;
        } else {
            N = std::stoi(arg2);
        }
    }

    // Open CSV output (append mode)
    std::ofstream csv(csvPath, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: cannot open CSV file: " << csvPath << "\n";
        return 1;
    }

    // Always write header once (your scripts can ignore duplicates)
    csv << "N,leafSize,maxDepth,build_seq_ms,build_par8_ms,speedup\n";

    std::vector<Point3D> points;

    // =============================
    // CASE 1: KITTI-like .bin input
    // =============================
    if (!binPath.empty()) {
        size_t maxPoints = (argc >= 4) ? std::stoul(argv[3]) : 0;

        points = loadKittiBin(binPath, maxPoints);
        N = (int)points.size();

        std::cout << "Octree build micro-benchmark (KITTI-like .bin)\n";
        std::cout << "  file = " << binPath << ", N = " << N
                  << ", leafSize = " << leafSize
                  << ", maxDepth = " << maxDepth << "\n";

    } else {
        // =============================
        // CASE 2: random [0,1]^3 points
        // =============================
        std::cout << "Octree build micro-benchmark (uniform random)\n";
        std::cout << "  N = " << N
                  << ", leafSize = " << leafSize
                  << ", maxDepth = " << maxDepth << "\n";

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        points.resize(N);
        for (auto& p : points) {
            p.x = dist(rng);
            p.y = dist(rng);
            p.z = dist(rng);
        }
    }

    // =============================
    // Warm-up to avoid first-call noise
    // =============================
    {
        auto t = buildOctree_seq(points, leafSize, maxDepth);
        (void)t;
    }

    // =============================
    // Time sequential build
    // =============================
    double seq_ms = time_ms([&]() {
        auto t = buildOctree_seq(points, leafSize, maxDepth);
        (void)t;
    });

    // =============================
    // Time parallel-8-root build (Innovation 1)
    // =============================
    double par_ms = time_ms([&]() {
        auto t = buildOctree(points, leafSize, maxDepth); // buildOctree = parallel8
        (void)t;
    });

    double speedup = seq_ms / par_ms;

    std::cout << "Sequential build:      " << seq_ms << " ms\n";
    std::cout << "Parallel-8-root build: " << par_ms << " ms\n";
    std::cout << "Innovation 1 speedup:  " << speedup << "x\n";

    // =============================
    // Write CSV row
    // =============================
    csv << N << ","
        << leafSize << ","
        << maxDepth << ","
        << seq_ms << ","
        << par_ms << ","
        << speedup << "\n";

    return 0;
}
