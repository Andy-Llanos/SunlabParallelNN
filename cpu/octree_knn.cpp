// Octree KNN search implementation
// CSE 375 Final Project
#include <queue>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "octree_knn.hpp"


// does KNN for a single query point by traversing the octree
// uses DFS with a stack and pruning based on bounding boxes
static std::vector<int> knn_single_query(
    const Octree& tree,
    const Point3D& q,
    int k,
    int maxNodesVisited)   // limits how many nodes we visit for speedup
{
    // max-heap to track k closest points (stores distance and index)
    std::priority_queue<std::pair<float,int>> heap;
    if (tree.root == -1) return {};  // empty tree

    // use a stack for DFS traversal (not recursion to avoid stack overflow)
    std::vector<int> stack;
    stack.push_back(tree.root);

    int nodesVisited = 0;  // track nodes for early stopping

    // main search loop - DFS through the tree
    while (!stack.empty()) {
        // early stopping optimization - stop if we've visited enough nodes
        if (nodesVisited >= maxNodesVisited) break;

        int ni = stack.back();
        stack.pop_back();
        nodesVisited++;  // count this node

        const OctreeNode& node = tree.nodes[ni];

        // compute distance from query to this node's bounding box
        float d_box = sqDistPointAABB(q, node.bb_min, node.bb_max);

        // pruning: if bounding box is farther than our worst current neighbor, skip it
        if (!heap.empty() && heap.size() == (size_t)k &&
            d_box > heap.top().first) {
            continue;  // this whole subtree is too far away
        }

        // check if this is a leaf node (no children)
        bool hasChild = false;
        for (int c : node.children) {
            if (c != -1) { hasChild = true; break; }
        }

        if (!hasChild) {
            // LEAF NODE: check all points in this node
            for (int i = 0; i < node.count; ++i) {
                int idx = tree.indices[node.start + i];
                const Point3D& p = (*tree.points)[idx];
                float d2 = sqDist(q, p);  // squared distance

                // maintain heap of k smallest distances
                if (heap.size() < (size_t)k) {
                    heap.emplace(d2, idx);  // not full yet, just add
                } else if (d2 < heap.top().first) {
                    heap.pop();             // remove worst
                    heap.emplace(d2, idx);  // add this one
                }
            }
        } else {
            // INTERNAL NODE: add children to stack to visit later
            for (int c : node.children) {
                if (c != -1) stack.push_back(c);
            }
        }
    }

    // extract results from heap
    // note: heap is in reverse order (worst first), so we need to reverse
    std::vector<int> temp;
    temp.reserve(heap.size());
    while (!heap.empty()) {
        temp.push_back(heap.top().second);
        heap.pop();
    }
    // reverse to get nearest first
    return std::vector<int>(temp.rbegin(), temp.rend());
}


// basic octree KNN - just calls single_query for each query
// this version hardcodes maxNodesVisited
std::vector<std::vector<int>> octreeKNN(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k)
{
    std::vector<std::vector<int>> neighbors;
    neighbors.reserve(queries.size());
    int maxNodesVisited = 1000; // TODO: should probably make this configurable
    for (const auto& q : queries) {
        neighbors.push_back(knn_single_query(tree, q, k, maxNodesVisited));
    }
    return neighbors;
}

// sequential version with explicit maxNodesVisited parameter
// used for benchmarking and testing
std::vector<std::vector<int>> octreeKNN_seq(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited)
{
    std::vector<std::vector<int>> neighbors(queries.size());
    // just loop through queries sequentially
    for (int qi = 0; qi < (int)queries.size(); ++qi) {
        neighbors[qi] = knn_single_query(tree, queries[qi], k, maxNodesVisited);
    }
    return neighbors;
}



// PARALLEL version using TBB - this is the main speedup!
// parallelizes over queries since each query is independent
std::vector<std::vector<int>> octreeKNN_parallel_tbb(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited)
{
    // preallocate results array
    std::vector<std::vector<int>> neighbors(queries.size());

    // TBB automatically splits queries into ranges and runs them in parallel
    // each thread gets a range of queries to process
    tbb::parallel_for(
        tbb::blocked_range<int>(0, (int)queries.size()),
        [&](const tbb::blocked_range<int>& r) {
            // process all queries in this thread's range
            for (int qi = r.begin(); qi != r.end(); ++qi) {
                neighbors[qi] = knn_single_query(tree, queries[qi], k, maxNodesVisited);
            }
        }
    );

    return neighbors;
}
