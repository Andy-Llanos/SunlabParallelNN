#include <queue>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "octree_knn.hpp"


// KNN for a single query using one DFS traversal
static std::vector<int> knn_single_query(
    const Octree& tree,
    const Point3D& q,
    int k,
    int maxNodesVisited)   // NEW ARG
{
    std::priority_queue<std::pair<float,int>> heap;
    if (tree.root == -1) return {};

    std::vector<int> stack;
    stack.push_back(tree.root);

    int nodesVisited = 0;  // NEW

    while (!stack.empty()) {
        if (nodesVisited >= maxNodesVisited) break;  // <-- EARLY STOP
        int ni = stack.back();
        stack.pop_back();
        nodesVisited++;  // count this node

        const OctreeNode& node = tree.nodes[ni];

        float d_box = sqDistPointAABB(q, node.bb_min, node.bb_max);
        if (!heap.empty() && heap.size() == (size_t)k &&
            d_box > heap.top().first) {
            continue;
        }

        bool hasChild = false;
        for (int c : node.children) {
            if (c != -1) { hasChild = true; break; }
        }

        if (!hasChild) {
            for (int i = 0; i < node.count; ++i) {
                int idx = tree.indices[node.start + i];
                const Point3D& p = (*tree.points)[idx];
                float d2 = sqDist(q, p);
                if (heap.size() < (size_t)k) {
                    heap.emplace(d2, idx);
                } else if (d2 < heap.top().first) {
                    heap.pop();
                    heap.emplace(d2, idx);
                }
            }
        } else {
            for (int c : node.children) {
                if (c != -1) stack.push_back(c);
            }
        }
    }

    std::vector<int> temp;
    temp.reserve(heap.size());
    while (!heap.empty()) {
        temp.push_back(heap.top().second);
        heap.pop();
    }
    return std::vector<int>(temp.rbegin(), temp.rend());
}


std::vector<std::vector<int>> octreeKNN(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k)
{
    std::vector<std::vector<int>> neighbors;
    neighbors.reserve(queries.size());
    int maxNodesVisited = 1000; // or whatever limit you want
    for (const auto& q : queries) {
        neighbors.push_back(knn_single_query(tree, q, k, maxNodesVisited));
    }
    return neighbors;
}



std::vector<std::vector<int>> octreeKNN_parallel_tbb(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited)
{
    std::vector<std::vector<int>> neighbors(queries.size());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, (int)queries.size()),
        [&](const tbb::blocked_range<int>& r) {
            for (int qi = r.begin(); qi != r.end(); ++qi) {
                neighbors[qi] = knn_single_query(tree, queries[qi], k, maxNodesVisited);
            }
        }
    );

    return neighbors;
}
