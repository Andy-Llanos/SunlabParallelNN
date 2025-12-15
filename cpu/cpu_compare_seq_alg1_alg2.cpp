// cpu/cpu_compare_seq_alg1_alg2.cpp
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

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>

using Clock = std::chrono::high_resolution_clock;

template <typename F>
double time_ms(F&& f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    std::chrono::duration<double, std::milli> diff = t1 - t0;
    return diff.count();
}

static std::string frame_path(const std::string& dir, int idx) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d.bin", idx);
    return dir + "/" + std::string(buf);
}

// --------- KF helpers (same as your keyframe bench) ----------
static inline uint8_t node_code(const Octree& t, int nodeIdx) {
    const auto& n = t.nodes[nodeIdx];
    uint8_t mask = 0;
    for (int b = 0; b < 8; ++b) if (n.children[b] != -1) mask |= (uint8_t)(1u << b);
    return mask;
}

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

        codeByPath[it.path]  = node_code(t, it.node);
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

static double KF_score(const Octree& keyProbe, const Octree& curProbe, int probeMaxDepth)
{
    std::unordered_map<std::string, uint8_t> keyCode, curCode;
    std::unordered_map<std::string, int> keyDepth, curDepth;
    build_path_maps(keyProbe, keyCode, keyDepth);
    build_path_maps(curProbe, curCode, curDepth);

    double sum = 0.0;
    std::unordered_map<std::string, bool> seen;
    seen.reserve(keyCode.size() + curCode.size());

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
        int exp = probeMaxDepth - depth; if (exp < 0) exp = 0;
        sum += (double)dh * pow8(exp);
        seen[path] = true;
    }

    for (const auto& kv : curCode) {
        const std::string& path = kv.first;
        if (seen.find(path) != seen.end()) continue;

        uint8_t a = 0;
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

// ---- forward decls (must exist in your repo) ----
Octree buildOctree_seq(const std::vector<Point3D>& points, int leafSize, int maxDepth);
Octree buildOctree(const std::vector<Point3D>& points, int leafSize, int maxDepth); // parallel-8-root

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

struct RunResult {
    double build_ms=0, probe_ms=0, search_ms=0, total_ms=0;
    int num_keyframes=0;
    double avg_recall=0;
};

static RunResult run_seq_baseline(const std::string& framesDir, int T, size_t maxPoints, int k, int Q,
                                 int leafSize, int maxDepthFull, int maxNodesVisited)
{
    RunResult r;
    double recall_sum = 0.0;
    int cnt = 0;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

        Octree tree;
        r.build_ms += time_ms([&]() { tree = buildOctree_seq(pts, leafSize, maxDepthFull); });

        std::vector<std::vector<int>> approx;
        r.search_ms += time_ms([&]() { approx = octreeKNN_seq(tree, queries, k, maxNodesVisited); });

        auto exact = bruteForceKNN(pts, queries, k);
        recall_sum += compute_average_recall_at_k(exact, approx);
        cnt++;
    }

    r.total_ms = r.build_ms + r.search_ms;
    r.avg_recall = cnt ? recall_sum / cnt : 0.0;
    return r;
}

static RunResult run_alg1_build_every_frame(int threads,
                                           const std::string& framesDir, int T, size_t maxPoints, int k, int Q,
                                           int leafSize, int maxDepthFull, int maxNodesVisited)
{
    tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);

    RunResult r;
    double recall_sum = 0.0;
    int cnt = 0;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

        Octree tree;
        r.build_ms += time_ms([&]() { tree = buildOctree(pts, leafSize, maxDepthFull); });

        std::vector<std::vector<int>> approx;
        r.search_ms += time_ms([&]() { approx = octreeKNN_parallel_tbb(tree, queries, k, maxNodesVisited); });

        auto exact = bruteForceKNN(pts, queries, k);
        recall_sum += compute_average_recall_at_k(exact, approx);
        cnt++;
    }

    r.total_ms = r.build_ms + r.search_ms;
    r.avg_recall = cnt ? recall_sum / cnt : 0.0;
    return r;
}

static RunResult run_alg2_keyframe_optionA(int threads, double Kth,
                                          const std::string& framesDir, int T, size_t maxPoints, int k, int Q,
                                          int leafSize, int maxDepthFull, int maxDepthProbe, int maxNodesVisited)
{
    tbb::global_control ctrl(tbb::global_control::max_allowed_parallelism, threads);

    RunResult r;
    double recall_sum = 0.0;
    int cnt = 0;

    Octree keyFull, keyProbe;
    bool haveKey = false;

    for (int i = 0; i < T; ++i) {
        auto pts = loadKittiBin(frame_path(framesDir, i), maxPoints);
        if (pts.empty()) continue;

        std::vector<Point3D> queries;
        queries.reserve(Q);
        for (int qi = 0; qi < Q && qi < (int)pts.size(); ++qi) queries.push_back(pts[qi]);

        Octree curProbe;
        r.probe_ms += time_ms([&]() { curProbe = buildOctree_seq(pts, leafSize, maxDepthProbe); });

        if (!haveKey) {
            r.build_ms += time_ms([&]() { keyFull = buildOctree(pts, leafSize, maxDepthFull); });
            keyProbe = std::move(curProbe);
            haveKey = true;
            r.num_keyframes++;
        } else {
            double kf = KF_score(keyProbe, curProbe, maxDepthProbe);
            if (kf >= Kth) {
                r.build_ms += time_ms([&]() { keyFull = buildOctree(pts, leafSize, maxDepthFull); });
                keyProbe = std::move(curProbe);
                r.num_keyframes++;
            }
        }

        std::vector<std::vector<int>> approx;
        r.search_ms += time_ms([&]() { approx = octreeKNN_parallel_tbb(keyFull, queries, k, maxNodesVisited); });

        auto exact = bruteForceKNN(pts, queries, k);
        recall_sum += compute_average_recall_at_k(exact, approx);
        cnt++;
    }

    r.total_ms = r.build_ms + r.probe_ms + r.search_ms;
    r.avg_recall = cnt ? recall_sum / cnt : 0.0;
    return r;
}

int main(int argc, char** argv) {
    // Usage:
    // ./cpu_compare <frames_dir> <csv_out> <T> <maxPoints> <k> <Q>
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <frames_dir> <csv_out> <T> <maxPoints> <k> <Q>\n";
        return 1;
    }

    std::string framesDir = argv[1];
    std::string csvOut    = argv[2];
    int T                = std::stoi(argv[3]);
    size_t maxPoints     = (size_t)std::stoul(argv[4]);
    int k                = std::stoi(argv[5]);
    int Q                = std::stoi(argv[6]);

    // params
    int leafSize = 32;
    int maxDepthFull = 10;
    int maxDepthProbe = 3;
    int maxNodesVisited = 256;

    // threads to test for parallel versions
    std::vector<int> threads_list = {2,4,8,16};

    // Kth sweep for alg2
    std::vector<double> Kths = {0, 5, 10, 20, 40, 80};

    // CSV
    std::ofstream csv(csvOut, std::ios::app);
    if (!csv) { std::cerr << "ERROR: cannot open " << csvOut << "\n"; return 1; }

    csv << "config,threads,Kth,num_keyframes,build_ms,probe_ms,search_ms,total_ms,avg_recall,speedup_vs_seq\n";

    // ------------- SEQ baseline (1 thread) -------------
    std::cout << "[SEQ] running...\n";
    RunResult seq = run_seq_baseline(framesDir, T, maxPoints, k, Q, leafSize, maxDepthFull, maxNodesVisited);
    std::cout << "  total_ms=" << seq.total_ms << " avg_recall=" << seq.avg_recall << "\n";

    csv << "SEQ,1,-1,0,"
        << seq.build_ms << "," << 0.0 << "," << seq.search_ms << ","
        << seq.total_ms << "," << seq.avg_recall << "," << 1.0 << "\n";

    // ------------- Algorithm 1 (parallel build + parallel search) -------------
    for (int th : threads_list) {
        std::cout << "[ALG1] threads=" << th << " running...\n";
        RunResult a1 = run_alg1_build_every_frame(th, framesDir, T, maxPoints, k, Q,
                                                 leafSize, maxDepthFull, maxNodesVisited);
        double sp = seq.total_ms / a1.total_ms;
        std::cout << "  total_ms=" << a1.total_ms << " speedup_vs_seq=" << sp
                  << " avg_recall=" << a1.avg_recall << "\n";

        csv << "ALG1," << th << ",-1,0,"
            << a1.build_ms << "," << 0.0 << "," << a1.search_ms << ","
            << a1.total_ms << "," << a1.avg_recall << "," << sp << "\n";
    }

    // ------------- Algorithm 2 (keyframe reuse, Option A) -------------
    for (int th : threads_list) {
        for (double Kth : Kths) {
            std::cout << "[ALG2] threads=" << th << " Kth=" << Kth << " running...\n";
            RunResult a2 = run_alg2_keyframe_optionA(th, Kth, framesDir, T, maxPoints, k, Q,
                                                    leafSize, maxDepthFull, maxDepthProbe, maxNodesVisited);
            double sp = seq.total_ms / a2.total_ms;

            std::cout << "  total_ms=" << a2.total_ms
                      << " keyframes=" << a2.num_keyframes
                      << " speedup_vs_seq=" << sp
                      << " avg_recall=" << a2.avg_recall << "\n";

            csv << "ALG2," << th << "," << Kth << "," << a2.num_keyframes << ","
                << a2.build_ms << "," << a2.probe_ms << "," << a2.search_ms << ","
                << a2.total_ms << "," << a2.avg_recall << "," << sp << "\n";
        }
    }

    std::cout << "Wrote CSV: " << csvOut << "\n";
    return 0;
}
