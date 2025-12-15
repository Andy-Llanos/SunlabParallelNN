// cpu/cpu_keyframe_bench.cpp
// Andy Llanos - CSE 375 Final Project
//
// This benchmark tests Algorithm 2 (keyframe reuse) vs baseline algorithms
// The idea: for video/LIDAR sequences, consecutive frames are similar, so
// we can reuse the octree from a "keyframe" instead of rebuilding every time
//
// We sweep over different Kth thresholds to see when we should rebuild:
// - Kth=0: rebuild every frame (baseline)
// - Kth=large: almost never rebuild (risky if frames change too much)
// - Kth=sweet spot: balance between build cost and accuracy

#define TBB_PREVIEW_GLOBAL_CONTROL 1  // needed to control TBB thread count
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

// compile from repo root:
// g++ -std=c++17 -O3 cpu/cpu_keyframe_bench.cpp cpu/octree.cpp cpu/octree_knn.cpp -ltbb -lpthread -o cpu_keyframe_bench
//
// run (full benchmark):
// ./cpu_keyframe_bench data/kitti/velodyne keyframe_results.csv 50 100000 16 2000 8
//
// run (quick test for tuning/debugging):
// ./cpu_keyframe_bench data/kitti/velodyne keyframe_results.csv 20 30000 8 500 8


// simple timing helper - wraps any lambda and returns milliseconds
template <typename F>
double time_ms(F&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

// ---- forward declarations for our octree functions ----
Octree buildOctree_seq(const std::vector<Point3D>& points, int leafSize, int maxDepth);
Octree buildOctree(const std::vector<Point3D>& points, int leafSize, int maxDepth); // parallel build

std::vector<std::vector<int>> octreeKNN_parallel_tbb(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited);



// ========== Keyframe Scoring Functions ==========
// These functions help us decide if we should rebuild the keyframe octree
// The KF_score measures how much the tree structure has changed between frames

// encode which children exist in a node as a bitmask (8 bits for 8 children)
// e.g., if children 0,2,5 exist -> bits 0,2,5 are set -> binary 0b00100101
// this gives us a compact way to compare tree structures
static inline uint8_t node_code(const Octree& t, int nodeIdx) {
    const auto& n = t.nodes[nodeIdx];
    uint8_t mask = 0;
    for (int b = 0; b < 8; ++b) {
        if (n.children[b] != -1) mask |= (uint8_t)(1u << b);  // set bit b if child exists
    }
    return mask;
}

// traverse the tree and build maps from path strings to node codes and depths
// path is like "0", "01", "012" - encodes which child you took at each level
// we use DFS with a stack (same pattern as in octree_knn.cpp)
static void build_path_maps(const Octree& t,
                            std::unordered_map<std::string, uint8_t>& codeByPath,
                            std::unordered_map<std::string, int>& depthByPath)
{
    codeByPath.clear();
    depthByPath.clear();
    if (t.root < 0) return;  // empty tree

    // DFS through tree, building path strings like "0", "07", "073", etc
    struct Item { int node; std::string path; int depth; };
    std::vector<Item> st;
    st.push_back({t.root, "", 0});  // start at root with empty path

    while (!st.empty()) {
        Item it = std::move(st.back());
        st.pop_back();

        // record this node's code and depth for this path
        codeByPath[it.path] = node_code(t, it.node);
        depthByPath[it.path] = it.depth;

        // push all children onto stack
        const auto& n = t.nodes[it.node];
        for (int b = 0; b < 8; ++b) {
            int c = n.children[b];
            if (c == -1) continue;  // child doesnt exist
            std::string childPath = it.path;
            childPath.push_back(char('0' + b));  // append octant number
            st.push_back({c, std::move(childPath), it.depth + 1});
        }
    }
}

// simple helper to compute 8^e (used for weighting by depth)
// nodes deeper in the tree affect fewer potential query points, so they get less weight
static inline double pow8(int e) {
    double v = 1.0;
    for (int i = 0; i < e; ++i) v *= 8.0;
    return v;
}

// compute the KF_score between keyframe's probe tree and current frame's probe tree
// Higher score = more structural difference = should rebuild keyframe
// Lower score = frames are similar = can reuse keyframe safely
//
// The score sums up weighted hamming distances between node codes:
// - For each path in the tree, compare which children exist (using bitmasks)
// - Weight by 8^(maxDepth - depth) so that differences near root matter more
// - This is because changes near root affect way more potential queries
static double KF_score(const Octree& keyProbe,
                       const Octree& curProbe,
                       int probeMaxDepth)
{
    // build the path->code and path->depth maps for both trees
    std::unordered_map<std::string, uint8_t> keyCode, curCode;
    std::unordered_map<std::string, int> keyDepth, curDepth;
    build_path_maps(keyProbe, keyCode, keyDepth);
    build_path_maps(curProbe, curCode, curDepth);

    double sum = 0.0;
    std::unordered_map<std::string, bool> seen;  // track which paths we've processed
    seen.reserve(keyCode.size() + curCode.size());

    // first pass: process all paths that exist in the keyframe
    for (const auto& kv : keyCode) {
        const std::string& path = kv.first;
        uint8_t a = kv.second;  // keyframe's node code
        uint8_t b = 0;          // current frame's node code (default to 0 if path doesn't exist)
        auto itB = curCode.find(path);
        if (itB != curCode.end()) b = itB->second;

        // get depth for weighting (prefer keyframe's depth if available)
        int depth = 0;
        auto itD = keyDepth.find(path);
        if (itD != keyDepth.end()) depth = itD->second;
        else {
            auto itD2 = curDepth.find(path);
            if (itD2 != curDepth.end()) depth = itD2->second;
        }

        // hamming distance: count how many bits differ (how many children changed)
        int dh = __builtin_popcount((unsigned)(a ^ b));  // nice builtin for counting 1s
        int exp = probeMaxDepth - depth;  // deeper nodes get less weight
        if (exp < 0) exp = 0;

        sum += (double)dh * pow8(exp);  // weighted contribution
        seen[path] = true;
    }

    // second pass: process paths that only exist in current frame (not in keyframe)
    for (const auto& kv : curCode) {
        const std::string& path = kv.first;
        if (seen.find(path) != seen.end()) continue;  // already processed above

        uint8_t a = 0;  // keyframe doesn't have this path
        uint8_t b = kv.second;  // current frame has it

        int depth = 0;
        auto itD = curDepth.find(path);
        if (itD != curDepth.end()) depth = itD->second;

        int dh = __builtin_popcount((unsigned)(a ^ b));
        int exp = probeMaxDepth - depth;
        if (exp < 0) exp = 0;

        sum += (double)dh * pow8(exp);
    }

    return sum;  // higher = more different, lower = more similar
}

// helper to get KITTI frame path like "000000.bin", "000001.bin", etc
static std::string frame_path(const std::string& dir, int idx) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d.bin", idx);  // zero-padded to 6 digits
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

    std::string framesDir = argv[1];  // e.g., "data/kitti/velodyne"
    std::string csvOut = argv[2];     // output CSV file
    int T = std::stoi(argv[3]);       // number of frames to process
    size_t maxPoints = (size_t)std::stoul(argv[4]);  // limit points per frame (for speed)
    int k = std::stoi(argv[5]);       // k neighbors to find
    int Q = std::stoi(argv[6]);       // number of query points per frame
    //int threads = std::stoi(argv[7]);  // NOTE: we loop over thread counts below

    int leafSize = 32;
    int maxDepthFull = 10;       // full tree for actual searches (deep and accurate)
    int maxDepthProbe = 3;       // probe tree for KF scoring (shallow and cheap)
    int maxNodesVisited = 256;   // early stopping for approximate KNN

    // Kth thresholds to test - sweep from 0 (rebuild every frame) to 320 (almost never rebuild)
    // After seeing initial KF_scores, you can tune these values to interesting ranges
    std::vector<double> Kths = {0, 5, 10, 20, 40, 80, 160, 320 };

    // open CSV in append mode (so we can run multiple times without overwriting)
    std::ofstream csv(csvOut, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: cannot open " << csvOut << "\n";
        return 1;
    }
    csv << "threads,Kth,num_frames,num_keyframes,total_full_build_ms,total_probe_build_ms,total_search_ms,total_pipeline_ms,avg_recall\n";



    // ========== Baseline A: SEQ build + SEQ search (rebuild every frame) ==========
    // This is the SLOWEST baseline - no parallelism at all
    // Useful to see how much parallelism helps overall
    double baseA_build_ms = 0.0;
    double baseA_search_ms = 0.0;
    double baseA_recall_sum = 0.0;
    int baseA_cnt = 0;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        // take first Q points as queries (simple approach for benchmarking)
        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

        // build tree sequentially
        Octree treeFull;
        baseA_build_ms += time_ms([&]() {
            treeFull = buildOctree_seq(pts, leafSize, maxDepthFull);
        });

        // search sequentially
        std::vector<std::vector<int>> approx;
        baseA_search_ms += time_ms([&]() {
            approx = octreeKNN_seq(treeFull, queries, k, maxNodesVisited);
        });

        // compute recall vs ground truth (brute force)
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



    // ========== Thread Scalability Sweep ==========
    // Test with different thread counts to see how well our parallel algorithms scale
    std::vector<int> thread_list = {1,2,4,8,16};
    for (int threads : thread_list) {
        // set TBB to use exactly this many threads
        tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);
        // TODO: could also run Baseline A here for each thread count if needed
        // For now we just do Baseline B (parallel) and keyframe sweep


    // ---------- Baseline B: PARALLEL build + PARALLEL search (rebuild every frame) ----------
    // This is Algorithm 1: parallel build at root + parallel queries with TBB
    // Should be faster than Baseline A due to parallelism
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

        // parallel build (8-way at root level)
        Octree treeFull;
        baseline_full_build_ms += time_ms([&]() {
            treeFull = buildOctree(pts, leafSize, maxDepthFull);
        });

        // parallel search (TBB over queries)
        std::vector<std::vector<int>> approx;
        baseline_search_ms += time_ms([&]() {
            approx = octreeKNN_parallel_tbb(treeFull, queries, k, maxNodesVisited);
        });

        auto exact = bruteForceKNN(pts, queries, k);
        baseline_recall_sum += compute_average_recall_at_k(exact, approx);
        baseline_cnt++;
    }

    std::cout << "[BASELINE B] parallel build+search (rebuild every frame): "
              << "build_ms=" << baseline_full_build_ms
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
