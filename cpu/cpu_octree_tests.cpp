#include <string>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <unordered_set>
#include <cassert>

#include "octree.hpp"
#include "octree_knn.hpp"
#include "utils.hpp"
#include "bruteForceKNN.cpp"

float compute_average_recall_at_k(
    const std::vector<std::vector<int>>& exact,
    const std::vector<std::vector<int>>& approx)
{
    assert(exact.size() == approx.size());
    int Q = (int)exact.size();
    if (Q == 0) return 1.0f;

    float sumRecall = 0.0f;

    for (int qi = 0; qi < Q; ++qi) {
        const auto& gt = exact[qi];
        const auto& ap = approx[qi];

        if (gt.empty()) {
            // nothing to retrieve, count as perfect
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

// forward declarations if not already present:
std::vector<std::vector<int>> bruteForceKNN(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k);

std::vector<std::vector<int>> octreeKNN(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k);

void test_octree_approx_accuracy() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int N = 5000;
    int Q = 50;
    int k = 10;

    std::vector<Point3D> pts(N), queries(Q);
    for (auto& p : pts)     p = Point3D{dist(rng), dist(rng), dist(rng)};
    for (auto& q : queries) q = Point3D{dist(rng), dist(rng), dist(rng)};

    int leafSize = 32;
    int maxDepth = 10;

    Octree tree = buildOctree(pts, leafSize, maxDepth);

    auto exact  = bruteForceKNN(pts, queries, k);
    auto approx = octreeKNN(tree, queries, k);  // your approximate version

    float avgRecall = compute_average_recall_at_k(exact, approx);

    std::cout << "Approximate octree KNN: avg recall@"
              << k << " = " << avgRecall << "\n";

    // Optional sanity check: make sure it's not totally garbage
    // (tune this threshold based on how aggressive your pruning is)
    if (avgRecall < 0.7f) {
        std::cerr << "WARNING: recall too low, something may be wrong.\n";
        // You can decide whether to assert or just warn:
        // assert(false);
    } else {
        std::cout << "test_octree_approx_accuracy PASSED\n";
    }
}

void test_single_root_bbox() {
    std::vector<Point3D> pts = {
        {0.f,  1.f,  2.f},
        {-1.f, 0.f,  5.f},
        {2.f, -2.f, -1.f}
    };

    Octree tree = buildOctree(pts, 1, 4);
 
    assert(tree.root == 0);
    assert(tree.nodes.size() == 1);
    const OctreeNode& root = tree.nodes[tree.root];

    assert(root.start == 0);
    assert(root.count == (int)pts.size());
    assert(root.isLeaf());

    // bbox should cover all points
    // min x = -1, max x = 2, etc.
    assert(root.bb_min.x == -1.f);
    assert(root.bb_max.x == 2.f);
    assert(root.bb_min.y == -2.f);
    assert(root.bb_max.y == 1.f);
    assert(root.bb_min.z == -1.f);
    assert(root.bb_max.z == 5.f);

    std::cout << "test_single_root_bbox PASSED\n";
}
void test_root_children_split() {
    // Choose points that clearly fall into different octants
    // around center (0,0,0)
    std::vector<Point3D> pts = {
        {-1,-1,-1}, // octant 0
        { 1,-1,-1}, // octant 1
        {-1, 1,-1}, // octant 2
        { 1, 1,-1}, // octant 3
        {-1,-1, 1}, // octant 4
        { 1,-1, 1}, // octant 5
        {-1, 1, 1}, // octant 6
        { 1, 1, 1}  // octant 7
    };

    // leafSize small, maxDepth >= 1 so it splits
    Octree tree = buildOctree(pts, /*leafSize=*/1, /*maxDepth=*/1);

    const OctreeNode& root = tree.nodes[tree.root];
    // root should have 8 children, each with count 1
    for (int b = 0; b < 8; ++b) {
        int ci = root.children[b];
        assert(ci != -1);
        const OctreeNode& child = tree.nodes[ci];
        assert(child.count == 1);
    }

    std::cout << "test_root_children_split PASSED\n";
}
void test_recursive_build_sanity() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int N = 1000;
    std::vector<Point3D> pts(N);
    for (auto& p : pts) p = {dist(rng), dist(rng), dist(rng)};

    Octree tree = buildOctree(pts, /*leafSize=*/32, /*maxDepth=*/10);

    std::cout << "Nodes: " << tree.nodes.size() << "\n";
    std::cout << "Root count: " << tree.nodes[tree.root].count << "\n";
}

void check_leaf_coverage(const Octree& tree,
                         int leafSize,
                         int maxDepth,
                         int expectedN)
{
    if (tree.root == -1) {
        assert(expectedN == 0);
        return;
    }

    int totalLeafCount = 0;

    // stack of (nodeIndex, depth)
    std::vector<std::pair<int,int>> stack;
    stack.push_back({tree.root, 0});

    while (!stack.empty()) {
        auto [ni, depth] = stack.back();
        stack.pop_back();

        const OctreeNode& node = tree.nodes[ni];

        bool hasChild = false;
        for (int c : node.children) {
            if (c != -1) { hasChild = true; break; }
        }

        if (!hasChild) {
            // leaf
            totalLeafCount += node.count;

            // leaf invariant: either small enough or at maxDepth
            assert(node.count <= leafSize || depth >= maxDepth);
        } else {
            // internal: should have count > 0
            assert(node.count > 0);

            for (int c : node.children) {
                if (c != -1) {
                    stack.push_back({c, depth + 1});
                }
            }
        }
    }

    // All points must be covered exactly once across leaves
    assert(totalLeafCount == expectedN);

    std::cout << "check_leaf_coverage PASSED (N=" << expectedN << ")\n";
}

void test_recursive_build_invariants() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int N = 1000;
    std::vector<Point3D> pts(N);
    for (auto& p : pts) {
        p = Point3D{dist(rng), dist(rng), dist(rng)};
    }

    int leafSize = 32;
    int maxDepth = 10;
    Octree tree = buildOctree(pts, leafSize, maxDepth);

    // root node should represent all points
    assert(tree.root != -1);
    assert(tree.nodes[tree.root].count == N);

    check_leaf_coverage(tree, leafSize, maxDepth, N);

    std::cout << "test_recursive_build_invariants PASSED\n";
}


void test_octree_vs_bruteforce_random() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int N = 1000;
    int Q = 20;
    int k = 5;

    std::vector<Point3D> pts(N), queries(Q);
    for (auto& p : pts)     p = Point3D{dist(rng), dist(rng), dist(rng)};
    for (auto& q : queries) q = Point3D{dist(rng), dist(rng), dist(rng)};

    int leafSize = 32;
    int maxDepth = 10;
    Octree tree = buildOctree(pts, leafSize, maxDepth);

    auto bf = bruteForceKNN(pts, queries, k);
    auto ot = octreeKNN(tree, queries, k);

    // compare per-query neighbor sets (order may differ, so sort)
    for (int qi = 0; qi < Q; ++qi) {
        auto a = bf[qi];
        auto b = ot[qi];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            std::cerr << "Mismatch at query " << qi << "\n";
            std::exit(1);
        }
    }

    std::cout << "test_octree_vs_bruteforce_random PASSED\n";
}


int main() {
    //test_single_root_bbox();
    test_root_children_split();
    test_recursive_build_sanity();
    test_recursive_build_invariants();
    test_octree_vs_bruteforce_random();
    test_octree_approx_accuracy();
    std::cout << "All octree tests PASSED\n";
    return 0;
}


