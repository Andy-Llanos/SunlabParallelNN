// Main benchmark comparing all three algorithms
// Sequential baseline vs Algorithm 1 vs Algorithm 2 (keyframe reuse)
// Andy Llanos - CSE 375 Final Project
//
// Usage: ./cpu_compare <kitti_dir> <output.csv> <T_frames> <maxPoints> <k> <Q_queries>
// Example: ./cpu_compare data/kitti/velodyne compare_results.csv 50 100000 16 1000

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <cstdio>   // snprintf for frame paths

#include "octree.hpp"
#include "octree_knn.hpp"
#include "kitti_loader.hpp"
#include "utils.hpp"
#include "metrics.hpp"
#include "bruteForceKNN.hpp"

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>

using Clock = std::chrono::high_resolution_clock;

// simple timing helper - runs a lambda and returns time in milliseconds
template <typename F>
double time_ms(F&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

// constructs KITTI frame path like "data/kitti/velodyne/000042.bin"
static std::string frame_path(const std::string& dir, int idx) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d.bin", idx);  // zero-padded 6 digits
    return dir + "/" + std::string(buf);
}

// ========== Keyframe Scoring Helpers (for Algorithm 2) ==========
// These functions help us decide if we should rebuild the keyframe octree
// by comparing the tree structure between frames

// encodes which children exist in a node as a bitmask (8 bits for 8 children)
// e.g. if children 0,2,5 exist -> bits 0,2,5 are set -> 0b00100101
static inline uint8_t node_code(const Octree& t, int nodeIdx) {
    const auto& n = t.nodes[nodeIdx];
    uint8_t mask = 0;
    for (int b = 0; b < 8; ++b)
        if (n.children[b] != -1)
            mask |= (uint8_t)(1u << b);
    return mask;
}

// traverses the tree and builds maps from path -> (child_code, depth)
// path is encoded as a string like "024" meaning root->child0->child2->child4
// this lets us compare tree structures even if node IDs are different
static void build_path_maps(const Octree& t,
                            std::unordered_map<std::string, uint8_t>& codeByPath,
                            std::unordered_map<std::string, int>& depthByPath)
{
    codeByPath.clear();
    depthByPath.clear();
    if (t.root < 0) return;  // empty tree

    // DFS through tree building path strings
    struct Item { int node; std::string path; int depth; };
    std::vector<Item> st;
    st.push_back({t.root, "", 0});  // root has empty path

    while (!st.empty()) {
        Item it = std::move(st.back());
        st.pop_back();

        // record this node's code and depth
        codeByPath[it.path]  = node_code(t, it.node);
        depthByPath[it.path] = it.depth;

        // add children to stack
        const auto& n = t.nodes[it.node];
        for (int b = 0; b < 8; ++b) {
            int c = n.children[b];
            if (c == -1) continue;  // child doesn't exist
            std::string childPath = it.path;
            childPath.push_back(char('0' + b));  // append child index
            st.push_back({c, std::move(childPath), it.depth + 1});
        }
    }
}

// quick 8^e calculation (used for weighting changes at different depths)
// changes near root matter more than changes deep in tree
static inline double pow8(int e) {
    double v = 1.0;
    for (int i = 0; i < e; ++i) v *= 8.0;
    return v;
}

// Keyframe scoring function - compares two probe trees
// Returns a score indicating how different the structures are
// Higher score = more different = should rebuild keyframe
// Uses XOR of child bitmasks and weights by depth (changes near root matter more)
static double KF_score(const Octree& keyProbe, const Octree& curProbe, int probeMaxDepth)
{
    // build path maps for both trees
    std::unordered_map<std::string, uint8_t> keyCode, curCode;
    std::unordered_map<std::string, int> keyDepth, curDepth;
    build_path_maps(keyProbe, keyCode, keyDepth);
    build_path_maps(curProbe, curCode, curDepth);

    double sum = 0.0;
    std::unordered_map<std::string, bool> seen;  // track which paths we've processed
    seen.reserve(keyCode.size() + curCode.size());

    // first pass: compare paths that exist in keyframe
    for (const auto& kv : keyCode) {
        const std::string& path = kv.first;
        uint8_t a = kv.second;  // keyframe's child config
        uint8_t b = 0;          // current frame's child config
        auto itB = curCode.find(path);
        if (itB != curCode.end()) b = itB->second;

        int depth = 0;
        auto itD = keyDepth.find(path);
        if (itD != keyDepth.end()) depth = itD->second;
        else {
            auto itD2 = curDepth.find(path);
            if (itD2 != curDepth.end()) depth = itD2->second;
        }

        // count how many bits differ (hamming distance)
        int dh = __builtin_popcount((unsigned)(a ^ b));  // nice builtin for counting 1s
        int exp = probeMaxDepth - depth;
        if (exp < 0) exp = 0;
        // weight by depth: changes at depth 0 get weight 8^3, depth 1 get 8^2, etc
        sum += (double)dh * pow8(exp);
        seen[path] = true;
    }

    // second pass: handle paths that only exist in current frame (new nodes)
    for (const auto& kv : curCode) {
        const std::string& path = kv.first;
        if (seen.find(path) != seen.end()) continue;  // already processed

        uint8_t a = 0;  // keyframe doesn't have this path
        uint8_t b = kv.second;

        int depth = 0;
        auto itD = curDepth.find(path);
        if (itD != curDepth.end()) depth = itD->second;

        int dh = __builtin_popcount((unsigned)(a ^ b));
        int exp = probeMaxDepth - depth; if (exp < 0) exp = 0;
        sum += (double)dh * pow8(exp);
    }

    return sum;
}

// ========== Forward Declarations ==========
Octree buildOctree_seq(const std::vector<Point3D>& points, int leafSize, int maxDepth);
Octree buildOctree(const std::vector<Point3D>& points, int leafSize, int maxDepth); // parallel version

std::vector<std::vector<int>> octreeKNN_parallel_tbb(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited);

std::vector<std::vector<int>> octreeKNN_seq(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited);

// struct to hold results from each run
struct RunResult {
    double build_ms=0, probe_ms=0, search_ms=0, total_ms=0;
    int num_keyframes=0;  // only relevant for Algorithm 2
    double avg_recall=0;
};

// ========== Algorithm Runners ==========

// SEQUENTIAL BASELINE: rebuilds octree every frame, searches sequentially
// This is what we're trying to beat with parallelism
static RunResult run_seq_baseline(const std::string& framesDir, int T, size_t maxPoints, int k, int Q,
                                 int leafSize, int maxDepthFull, int maxNodesVisited)
{
    RunResult r;
    double recall_sum = 0.0;
    int cnt = 0;

    // process T frames
    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;  // skip missing frames

        // extract queries (just use first Q points)
        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi)
            queries.push_back(pts[qi]);

        // build octree (sequential)
        Octree tree;
        r.build_ms += time_ms([&]() {
            tree = buildOctree_seq(pts, leafSize, maxDepthFull);
        });

        // search (sequential)
        std::vector<std::vector<int>> approx;
        r.search_ms += time_ms([&]() {
            approx = octreeKNN_seq(tree, queries, k, maxNodesVisited);
        });

        // validate against ground truth
        auto exact = bruteForceKNN(pts, queries, k);
        recall_sum += compute_average_recall_at_k(exact, approx);
        cnt++;
    }

    r.total_ms = r.build_ms + r.search_ms;
    r.avg_recall = cnt ? recall_sum / cnt : 0.0;
    return r;
}

// ALGORITHM 1: Parallel build + parallel search
// Rebuilds tree every frame but uses TBB for parallelism
// This is the main innovation - parallelizing octree operations
static RunResult run_alg1_build_every_frame(int threads,
                                           const std::string& framesDir, int T, size_t maxPoints, int k, int Q,
                                           int leafSize, int maxDepthFull, int maxNodesVisited)
{
    // set TBB thread count (this stays in scope for the whole function)
    tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);

    RunResult r;
    double recall_sum = 0.0;
    int cnt = 0;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi)
            queries.push_back(pts[qi]);

        // build octree (PARALLEL - 8-way at root)
        Octree tree;
        r.build_ms += time_ms([&]() {
            tree = buildOctree(pts, leafSize, maxDepthFull);
        });

        // search (PARALLEL - per-query parallelism)
        std::vector<std::vector<int>> approx;
        r.search_ms += time_ms([&]() {
            approx = octreeKNN_parallel_tbb(tree, queries, k, maxNodesVisited);
        });

        // validate
        auto exact = bruteForceKNN(pts, queries, k);
        recall_sum += compute_average_recall_at_k(exact, approx);
        cnt++;
    }

    r.total_ms = r.build_ms + r.search_ms;
    r.avg_recall = cnt ? recall_sum / cnt : 0.0;
    return r;
}

// ALGORITHM 2: Keyframe reuse with structural validation
// Builds a "keyframe" tree and reuses it across multiple frames
// Only rebuilds when structure changes significantly (measured by KF_score)
// Key insight: consecutive frames in video/LIDAR data are similar
static RunResult run_alg2_keyframe_optionA(int threads, double Kth,
                                          const std::string& framesDir, int T, size_t maxPoints, int k, int Q,
                                          int leafSize, int maxDepthFull, int maxDepthProbe, int maxNodesVisited)
{
    tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);

    RunResult r;
    double recall_sum = 0.0;
    int cnt = 0;

    // keyframe trees (persist across frames)
    Octree keyFull, keyProbe;  // keyFull is used for queries, keyProbe for validation
    bool haveKey = false;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi)
            queries.push_back(pts[qi]);

        // always build a shallow probe tree for this frame (cheap, depth=3)
        Octree curProbe;
        r.probe_ms += time_ms([&]() {
            curProbe = buildOctree_seq(pts, leafSize, maxDepthProbe);
        });

        if (!haveKey) {
            // first frame: build initial keyframe
            r.build_ms += time_ms([&]() {
                keyFull = buildOctree(pts, leafSize, maxDepthFull);  // PARALLEL build
            });
            keyProbe = std::move(curProbe);
            haveKey = true;
            r.num_keyframes++;
        } else {
            // compare structures: should we rebuild?
            double kf = KF_score(keyProbe, curProbe, maxDepthProbe);
            if (kf >= Kth) {
                // structure changed too much - rebuild keyframe
                r.build_ms += time_ms([&]() {
                    keyFull = buildOctree(pts, leafSize, maxDepthFull);  // PARALLEL
                });
                keyProbe = std::move(curProbe);
                r.num_keyframes++;
            }
            // else: reuse existing keyFull (no build cost!)
        }

        // search using keyframe tree (PARALLEL query processing)
        std::vector<std::vector<int>> approx;
        r.search_ms += time_ms([&]() {
            approx = octreeKNN_parallel_tbb(keyFull, queries, k, maxNodesVisited);
        });

        // validate
        auto exact = bruteForceKNN(pts, queries, k);
        recall_sum += compute_average_recall_at_k(exact, approx);
        cnt++;
    }

    r.total_ms = r.build_ms + r.probe_ms + r.search_ms;
    r.avg_recall = cnt ? recall_sum / cnt : 0.0;
    return r;
}

// ========== MAIN - runs all benchmarks and writes CSV ==========
int main(int argc, char** argv) {
    // Usage:
    // ./cpu_compare <frames_dir> <csv_out> <T> <maxPoints> <k> <Q>
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <frames_dir> <csv_out> <T> <maxPoints> <k> <Q>\n";
        return 1;
    }

    // parse command line args
    std::string framesDir = argv[1];   // where KITTI frames are
    std::string csvOut    = argv[2];   // output CSV file
    int T                = std::stoi(argv[3]);   // number of frames to process
    size_t maxPoints     = (size_t)std::stoul(argv[4]);  // downsample to this many points
    int k                = std::stoi(argv[5]);   // k for KNN
    int Q                = std::stoi(argv[6]);   // number of queries per frame

    // algorithm parameters (hardcoded, could make these args too)
    int leafSize = 32;          // stop splitting when node has <= 32 points
    int maxDepthFull = 10;      // full tree depth (for actual KNN queries)
    int maxDepthProbe = 3;      // shallow probe tree depth (for validation)
    int maxNodesVisited = 256;  // early stopping for octree search

    // threads to test for parallel versions
    std::vector<int> threads_list = {2,4,8,16};

    // Kth sweep for alg2 (threshold for rebuilding keyframe)
    // higher Kth = rebuild less often = faster but possibly lower recall
    std::vector<double> Kths = {0, 5, 10, 20, 40, 80};

    // open CSV output file (append mode so we can run multiple times)
    std::ofstream csv(csvOut, std::ios::app);
    if (!csv) {
        std::cerr << "ERROR: cannot open " << csvOut << "\n";
        return 1;
    }

    // write CSV header
    csv << "config,threads,Kth,num_keyframes,build_ms,probe_ms,search_ms,total_ms,avg_recall,speedup_vs_seq\n";

    // ------------- BASELINE: Sequential everything (1 thread) -------------
    std::cout << "[SEQ] running sequential baseline...\n";
    RunResult seq = run_seq_baseline(framesDir, T, maxPoints, k, Q, leafSize, maxDepthFull, maxNodesVisited);
    std::cout << "  total_ms=" << seq.total_ms << " avg_recall=" << seq.avg_recall << "\n";

    // write baseline result
    csv << "SEQ,1,-1,0,"
        << seq.build_ms << "," << 0.0 << "," << seq.search_ms << ","
        << seq.total_ms << "," << seq.avg_recall << "," << 1.0 << "\n";

    // ------------- ALGORITHM 1: Parallel build + parallel search -------------
    // Test with different thread counts to see how it scales
    for (int th : threads_list) {
        std::cout << "[ALG1] threads=" << th << " running...\n";
        RunResult a1 = run_alg1_build_every_frame(th, framesDir, T, maxPoints, k, Q,
                                                 leafSize, maxDepthFull, maxNodesVisited);
        double sp = seq.total_ms / a1.total_ms;  // speedup vs sequential
        std::cout << "  total_ms=" << a1.total_ms << " speedup_vs_seq=" << sp
                  << " avg_recall=" << a1.avg_recall << "\n";

        csv << "ALG1," << th << ",-1,0,"
            << a1.build_ms << "," << 0.0 << "," << a1.search_ms << ","
            << a1.total_ms << "," << a1.avg_recall << "," << sp << "\n";
    }

    // ------------- ALGORITHM 2: Keyframe reuse -------------
    // Test with different thread counts AND different Kth thresholds
    // This lets us explore the speed vs accuracy tradeoff
    for (int th : threads_list) {
        for (double Kth : Kths) {
            std::cout << "[ALG2] threads=" << th << " Kth=" << Kth << " running...\n";
            RunResult a2 = run_alg2_keyframe_optionA(th, Kth, framesDir, T, maxPoints, k, Q,
                                                    leafSize, maxDepthFull, maxDepthProbe, maxNodesVisited);
            double sp = seq.total_ms / a2.total_ms;  // speedup vs sequential

            std::cout << "  total_ms=" << a2.total_ms
                      << " keyframes=" << a2.num_keyframes  // how many rebuilds?
                      << " speedup_vs_seq=" << sp
                      << " avg_recall=" << a2.avg_recall << "\n";

            csv << "ALG2," << th << "," << Kth << "," << a2.num_keyframes << ","
                << a2.build_ms << "," << a2.probe_ms << "," << a2.search_ms << ","
                << a2.total_ms << "," << a2.avg_recall << "," << sp << "\n";
        }
    }

    std::cout << "\nAll done! Wrote results to: " << csvOut << "\n";
    return 0;
}
