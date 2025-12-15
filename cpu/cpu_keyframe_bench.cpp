// cpu/cpu_keyframe_bench.cpp
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <cstdio>   // snprintf


#include "octree.hpp"
#include "octree_knn.hpp"
#include "kitti_loader.hpp"
#include "utils.hpp"
#include "metrics.hpp"
#include "bruteForceKNN.hpp"



using Clock = std::chrono::high_resolution_clock;
//compile from repo root:
////g++ -std=c++17 -O3 cpu/cpu_keyframe_bench.cpp cpu/octree.cpp cpu/octree_knn.cpp -ltbb -lpthread -o cpu_keyframe_bench
////g++ -std=c++17 -O3 cpu/cpu_keyframe_bench.cpp cpu/octree.cpp cpu/octree_knn.cpp -ltbb -lpthread -o cpu_keyframe_bench
//run:
//./cpu_keyframe_bench data/kitti/velodyne keyframe_results.csv 50 100000 16 2000 8
//use ths instead for tuning/debugging:
//./cpu_keyframe_bench data/kitti/velodyne keyframe_results.csv 20 30000 8 500 8


template <typename F>
double time_ms(F&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

// ---- forward decls (your repo) ----
Octree buildOctree_seq(const std::vector<Point3D>& points, int leafSize, int maxDepth);
Octree buildOctree(const std::vector<Point3D>& points, int leafSize, int maxDepth); // parallel8 build

std::vector<std::vector<int>> octreeKNN_parallel_tbb(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited);



// If compute_average_recall_at_k is NOT in a header, paste it here.
// Otherwise, remove this block and include the header where it lives.

// ---- KF helpers ----
static inline uint8_t node_code(const Octree& t, int nodeIdx) {
    const auto& n = t.nodes[nodeIdx];
    uint8_t mask = 0;
    for (int b = 0; b < 8; ++b) {
        if (n.children[b] != -1) mask |= (uint8_t)(1u << b);
    }
    return mask;
}

// Build maps: path -> code, path -> depth
static void build_path_maps(const Octree& t,
                            std::unordered_map<std::string, uint8_t>& codeByPath,
                            std::unordered_map<std::string, int>& depthByPath)
{
    codeByPath.clear();
    depthByPath.clear();
    if (t.root < 0) return;

    struct Item { int node; std::string path; int depth; };
    std::vector<Item> st;
    st.push_back({t.root, "", 0});

    while (!st.empty()) {
        Item it = std::move(st.back());
        st.pop_back();

        codeByPath[it.path] = node_code(t, it.node);
        depthByPath[it.path] = it.depth;

        const auto& n = t.nodes[it.node];
        for (int b = 0; b < 8; ++b) {
            int c = n.children[b];
            if (c == -1) continue;
            std::string childPath = it.path;
            childPath.push_back(char('0' + b));
            st.push_back({c, std::move(childPath), it.depth + 1});
        }
    }
}

static inline double pow8(int e) {
    double v = 1.0;
    for (int i = 0; i < e; ++i) v *= 8.0;
    return v;
}

static double KF_score(const Octree& keyProbe,
                       const Octree& curProbe,
                       int probeMaxDepth)
{
    std::unordered_map<std::string, uint8_t> keyCode, curCode;
    std::unordered_map<std::string, int> keyDepth, curDepth;
    build_path_maps(keyProbe, keyCode, keyDepth);
    build_path_maps(curProbe, curCode, curDepth);

    double sum = 0.0;
    std::unordered_map<std::string, bool> seen;
    seen.reserve(keyCode.size() + curCode.size());

    // key paths
    for (const auto& kv : keyCode) {
        const std::string& path = kv.first;
        uint8_t a = kv.second;
        uint8_t b = 0;
        auto itB = curCode.find(path);
        if (itB != curCode.end()) b = itB->second;

        int depth = 0;
        auto itD = keyDepth.find(path);
        if (itD != keyDepth.end()) depth = itD->second;
        else {
            auto itD2 = curDepth.find(path);
            if (itD2 != curDepth.end()) depth = itD2->second;
        }

        int dh = __builtin_popcount((unsigned)(a ^ b));
        int exp = probeMaxDepth - depth;
        if (exp < 0) exp = 0;

        sum += (double)dh * pow8(exp);
        seen[path] = true;
    }

    // cur-only paths
    for (const auto& kv : curCode) {
        const std::string& path = kv.first;
        if (seen.find(path) != seen.end()) continue;

        uint8_t a = 0;
        uint8_t b = kv.second;

        int depth = 0;
        auto itD = curDepth.find(path);
        if (itD != curDepth.end()) depth = itD->second;

        int dh = __builtin_popcount((unsigned)(a ^ b));
        int exp = probeMaxDepth - depth;
        if (exp < 0) exp = 0;

        sum += (double)dh * pow8(exp);
    }

    return sum;
}

static std::string frame_path(const std::string& dir, int idx) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d.bin", idx);
    return dir + "/" + std::string(buf);
}

int main(int argc, char** argv) {
    // Usage:
    // ./cpu_keyframe_bench <frames_dir> <csv_out> <num_frames> <maxPoints> <k> <Q> <threads>
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <frames_dir> <csv_out> <num_frames> <maxPoints> <k> <Q> <threads>\n";
        return 1;
    }

    std::string framesDir = argv[1];
    std::string csvOut = argv[2];
    int T = std::stoi(argv[3]);
    size_t maxPoints = (size_t)std::stoul(argv[4]);
    int k = std::stoi(argv[5]);
    int Q = std::stoi(argv[6]);
    //int threads = std::stoi(argv[7]);

    int leafSize = 32;
    int maxDepthFull = 10;
    int maxDepthProbe = 3;       // cheap probe tree
    int maxNodesVisited = 256;   // keep consistent with your approx settings

    // Start with these; once you see KF magnitudes, tune them.
std::vector<double> Kths = {0, 5, 10, 20, 40, 80, 160, 320 };

    std::ofstream csv(csvOut, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: cannot open " << csvOut << "\n";
        return 1;
    }
    csv << "threads,Kth,num_frames,num_keyframes,total_full_build_ms,total_probe_build_ms,total_search_ms,total_pipeline_ms,avg_recall\n";
    
    
    
        // ---------- Baseline A: SEQ build + SEQ search, rebuild every frame ----------
    double baseA_build_ms = 0.0;
    double baseA_search_ms = 0.0;
    double baseA_recall_sum = 0.0;
    int baseA_cnt = 0;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

        Octree treeFull;
        baseA_build_ms += time_ms([&]() {
            treeFull = buildOctree_seq(pts, leafSize, maxDepthFull);
        });

        std::vector<std::vector<int>> approx;
        baseA_search_ms += time_ms([&]() {
            approx = octreeKNN_seq(treeFull, queries, k, maxNodesVisited);
        });

        auto exact = bruteForceKNN(pts, queries, k);
        baseA_recall_sum += compute_average_recall_at_k(exact, approx);
        baseA_cnt++;
    }

    double baseA_total_ms = baseA_build_ms + baseA_search_ms;
    double baseA_avg_recall = baseA_cnt ? baseA_recall_sum / baseA_cnt : 0.0;

    std::cout << "\n[BASELINE A] SEQ build + SEQ search:\n"
            << "  build_ms=" << baseA_build_ms
            << " search_ms=" << baseA_search_ms
            << " total_ms=" << baseA_total_ms
            << " avg_recall=" << baseA_avg_recall << "\n";
    
    
    
    
    std::vector<int> thread_list = {1,2,4,8,16};
    for (int threads : thread_list) {
        tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);
        // run Baseline A, Baseline B, and KF sweep
        // write CSV rows including `threads`
    

    //tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);



    // ---------- baseline: build full tree every frame ----------
    double baseline_full_build_ms = 0.0;
    double baseline_search_ms = 0.0;
    double baseline_recall_sum = 0.0;
    int baseline_cnt = 0;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

        Octree treeFull;
        baseline_full_build_ms += time_ms([&]() {
            treeFull = buildOctree(pts, leafSize, maxDepthFull);
        });

        std::vector<std::vector<int>> approx;
        baseline_search_ms += time_ms([&]() {
            approx = octreeKNN_parallel_tbb(treeFull, queries, k, maxNodesVisited);
        });

        auto exact = bruteForceKNN(pts, queries, k);
        baseline_recall_sum += compute_average_recall_at_k(exact, approx);
        baseline_cnt++;
    }

    std::cout << "[BASELINE] build_every_frame: build_ms=" << baseline_full_build_ms
              << " search_ms=" << baseline_search_ms
              << " avg_recall=" << (baseline_cnt ? baseline_recall_sum / baseline_cnt : 0.0)
              << "\n";

    // ---------- keyframe policy sweep ----------
    for (double Kth : Kths) {
        double total_full_build_ms = 0.0;
        double total_probe_build_ms = 0.0;
        double total_search_ms = 0.0;
        double recall_sum = 0.0;
        int cnt = 0;
        int num_keyframes = 0;

        Octree keyFull, keyProbe;
        bool haveKey = false;

        for (int i = 0; i < T; ++i) {
            auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
            if (pts.empty()) continue;

            std::vector<Point3D> queries;
            queries.reserve(Q);
            for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

            // always build probe (cheap) for KF score
            Octree curProbe;
            total_probe_build_ms += time_ms([&]() {
                curProbe = buildOctree_seq(pts, leafSize, maxDepthProbe);
            });

            if (!haveKey) {
                // first frame is a keyframe
                total_full_build_ms += time_ms([&]() {
                    keyFull = buildOctree(pts, leafSize, maxDepthFull);
                });
                keyProbe = std::move(curProbe);
                haveKey = true;
                num_keyframes++;
            } else {
                double kf = KF_score(keyProbe, curProbe, maxDepthProbe);
                if (kf >= Kth) {
                    total_full_build_ms += time_ms([&]() {
                        keyFull = buildOctree(pts, leafSize, maxDepthFull);
                    });
                    keyProbe = std::move(curProbe);
                    num_keyframes++;
                }
            }

            // Option A: search using keyframe tree
            std::vector<std::vector<int>> approx;
            total_search_ms += time_ms([&]() {
                approx = octreeKNN_parallel_tbb(keyFull, queries, k, maxNodesVisited);
            });

            auto exact = bruteForceKNN(pts, queries, k);
            recall_sum += compute_average_recall_at_k(exact, approx);
            cnt++;
        }

        double avg_recall = cnt ? recall_sum / cnt : 0.0;
        double total_ms = total_full_build_ms + total_probe_build_ms + total_search_ms;

        std::cout << "[KF] Kth=" << Kth
                  << " keyframes=" << num_keyframes
                  << " full_build_ms=" << total_full_build_ms
                  << " probe_ms=" << total_probe_build_ms
                  << " search_ms=" << total_search_ms
                  << " total_ms=" << total_ms
                  << " avg_recall=" << avg_recall
                  << "\n";

        csv << threads << ","
        << Kth << ","
        << T << ","
        << num_keyframes << ","
        << total_full_build_ms << ","
        << total_probe_build_ms << ","
        << total_search_ms << ","
        << total_ms << ","
        << avg_recall << "\n";

    }
}

    std::cout << "Wrote CSV: " << csvOut << "\n";
    return 0;
}
