#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "utils.hpp"
#include "octree.hpp"
#include "kitti_loader.hpp"  // where loadKittiBin lives

using Clock = std::chrono::high_resolution_clock;
///COMPILE IN ROOT NOT cpu folder
////g++ -std=c++17 -O3 cpu/cpu_octree_build_bench.cpp cpu/octree.cpp -ltbb -lpthread -o cpu_octree_build_bench
///EXECUTE 
///./cpu_octree_build_bench
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
///parallel construction 
Octree buildOctree(const std::vector<Point3D>& points,
                             int leafSize,
                             int maxDepth);



int main(int argc, char** argv) {
    int N = 200000;
    int leafSize = 32;
    int maxDepth = 10;
    std::string binPath;

    // Usage:
    //   ./cpu_octree_build_bench               -> random N=200k points
    //   ./cpu_octree_build_bench 100000        -> random N=100k points
    //   ./cpu_octree_build_bench file.bin      -> load from .bin
    //   ./cpu_octree_build_bench file.bin 0    -> load all points from .bin
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1.size() > 4 && arg1.substr(arg1.size()-4) == ".bin") {
            binPath = arg1;
        } else {
            N = std::stoi(arg1);
        }
    }

    if (!binPath.empty()) {
        size_t maxPoints = (argc >= 3) ? std::stoul(argv[2]) : 0; // 0 = all
        auto points = loadKittiBin(binPath, maxPoints);
        N = (int)points.size();
        std::cout << "Octree build micro-benchmark (KITTI-like .bin)\n";
        std::cout << "  file = " << binPath << ", N = " << N
                  << ", leafSize = " << leafSize
                  << ", maxDepth = " << maxDepth << "\n";

        // warmup
        {
            auto t = buildOctree_seq(points, leafSize, maxDepth);
            (void)t;
        }

        double seq_ms = time_ms([&]() {
            auto t = buildOctree_seq(points, leafSize, maxDepth);
            (void)t;
        });

        double par_ms = time_ms([&]() {
            auto t = buildOctree(points, leafSize, maxDepth);
            (void)t;
        });

        std::cout << "Sequential build: " << seq_ms << " ms\n";
        std::cout << "Parallel-8-root build: " << par_ms << " ms\n";
        std::cout << "Innovation 1 speedup: " << (seq_ms / par_ms) << "x\n";
    } else {
        std::cout << "Octree build micro-benchmark (uniform random)\n";
        std::cout << "  N = " << N
                  << ", leafSize = " << leafSize
                  << ", maxDepth = " << maxDepth << "\n";

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<Point3D> points(N);
        for (auto& p : points) {
            p.x = dist(rng);
            p.y = dist(rng);
            p.z = dist(rng);
        }

        // warmup
        {
            auto t = buildOctree_seq(points, leafSize, maxDepth);
            (void)t;
        }

        double seq_ms = time_ms([&]() {
            auto t = buildOctree_seq(points, leafSize, maxDepth);
            (void)t;
        });

        double par_ms = time_ms([&]() {
            auto t = buildOctree(points, leafSize, maxDepth);
            (void)t;
        });

        std::cout << "Sequential build: " << seq_ms << " ms\n";
        std::cout << "Parallel-8-root build: " << par_ms << " ms\n";
        std::cout << "Innovation 1 speedup: " << (seq_ms / par_ms) << "x\n";
    }

    return 0;
}

