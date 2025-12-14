#include <limits>
#include <cassert>
#include <iostream>
#include <random>
#include <mutex>          
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/parallel_for.h>

#include "octree.hpp"

static std::mutex g_nodeAllocMutex;

// ---- helper: compute bbox of points[indices[start..start+count)) ----
static void computeBoundingBox(const std::vector<Point3D>& pts,
                               const std::vector<int>& indices,
                               int start, int count,
                               Point3D& bb_min,
                               Point3D& bb_max)
{
    float inf = std::numeric_limits<float>::infinity();
    bb_min = { inf,  inf,  inf};
    bb_max = {-inf, -inf, -inf};

    for (int i = 0; i < count; ++i) {
        const Point3D& p = pts[indices[start + i]];
        if (p.x < bb_min.x) bb_min.x = p.x;
        if (p.y < bb_min.y) bb_min.y = p.y;
        if (p.z < bb_min.z) bb_min.z = p.z;

        if (p.x > bb_max.x) bb_max.x = p.x;
        if (p.y > bb_max.y) bb_max.y = p.y;
        if (p.z > bb_max.z) bb_max.z = p.z;
    }
}

// ---- helper: classify a point into one of 8 octants around (cx, cy, cz) ----
static int classifyOctant(const Point3D& p, float cx, float cy, float cz) {
    int oct = 0;
    if (p.x >= cx) oct |= 1;
    if (p.y >= cy) oct |= 2;
    if (p.z >= cz) oct |= 4;
    return oct;  // 0..7
}

// ---- internal recursive builder ----
static int buildNode(Octree& tree,
                     int start, int count,
                     int depth,
                     int leafSize,
                     int maxDepth,
                     bool parallelRoot)
{
    assert(count >= 0);

    int nodeIndex;
    {
        std::lock_guard<std::mutex> lock(g_nodeAllocMutex);
        nodeIndex = (int)tree.nodes.size();
        tree.nodes.push_back({});
    }
    OctreeNode& node = tree.nodes[nodeIndex];


    node.start = start;
    node.count = count;
    node.children.fill(-1);

    // compute bbox
    computeBoundingBox(*tree.points, tree.indices,
                       start, count,
                       node.bb_min, node.bb_max);

    // base cases: leaf if few points or too deep
    if (count <= leafSize || depth >= maxDepth) {
        return nodeIndex;
    }

    // compute center of bbox
    float cx = 0.5f * (node.bb_min.x + node.bb_max.x);
    float cy = 0.5f * (node.bb_min.y + node.bb_max.y);
    float cz = 0.5f * (node.bb_min.z + node.bb_max.z);

    // first pass: count points per octant
    int bucketCounts[8] = {0,0,0,0,0,0,0,0};

    for (int i = 0; i < count; ++i) {
        int idx = tree.indices[start + i];
        const Point3D& p = (*tree.points)[idx];
        int oct = classifyOctant(p, cx, cy, cz);
        bucketCounts[oct]++;
    }

    // if everything fell into one bucket, don't split (degenerate case)
    int nonEmptyBuckets = 0;
    for (int b = 0; b < 8; ++b)
        if (bucketCounts[b] > 0) nonEmptyBuckets++;

    if (nonEmptyBuckets <= 1) {
        // avoid infinite recursion / super-unbalanced tree
        return nodeIndex;
    }

    // second pass: prefix sums for offsets
    int bucketOffsets[8];
    int offset = 0;
    for (int b = 0; b < 8; ++b) {
        bucketOffsets[b] = offset;
        offset += bucketCounts[b];
    }
    assert(offset == count);

    // third pass: reorder indices into temp[count] by octant
    std::vector<int> temp(count);
    int cursor[8];
    for (int b = 0; b < 8; ++b)
        cursor[b] = bucketOffsets[b];

    for (int i = 0; i < count; ++i) {
        int idx = tree.indices[start + i];
        const Point3D& p = (*tree.points)[idx];
        int oct = classifyOctant(p, cx, cy, cz);
        temp[cursor[oct]++] = idx;
    }

    // copy back into tree.indices in [start, start+count)
    for (int i = 0; i < count; ++i) {
        tree.indices[start + i] = temp[i];
    }

    // recursively build children for non-empty buckets
    if (depth == 0 && parallelRoot) {
        // Innovation 1: parallelize the 8 root subtrees
        tbb::parallel_for(0, 8, [&](int b) {
            int childCount = bucketCounts[b];
            if (childCount == 0) {
                node.children[b] = -1;
                return;
            }
            int childStart = start + bucketOffsets[b];
            node.children[b] = buildNode(tree,
                                        childStart,
                                        childCount,
                                        depth + 1,
                                        leafSize,
                                        maxDepth,
                                        /*parallelRoot=*/false);
        });
    } else {
        // deeper levels or seq build: sequential children
        for (int b = 0; b < 8; ++b) {
            int childCount = bucketCounts[b];
            if (childCount == 0) {
                node.children[b] = -1;
                continue;
            }
            int childStart = start + bucketOffsets[b];
            node.children[b] = buildNode(tree,
                                        childStart,
                                        childCount,
                                        depth + 1,
                                        leafSize,
                                        maxDepth,
                                        parallelRoot);
        }
    }



    return nodeIndex;
}

    // ---- public entry point ----
    Octree buildOctree_seq(const std::vector<Point3D>& points,
                        int leafSize,
                        int maxDepth)
    {
        Octree tree;
        tree.points = &points;
        tree.indices.resize(points.size());
        for (int i = 0; i < (int)points.size(); ++i) {
            tree.indices[i] = i;
        }

        tree.nodes.clear();
        if (!points.empty()) {
            tree.nodes.reserve(points.size() * 2);
            tree.root = buildNode(tree,
                                /*start=*/0,
                                /*count=*/(int)points.size(),
                                /*depth=*/0,
                                leafSize,
                                maxDepth,
                                /*parallelRoot=*/false);
        } else {
            tree.root = -1;
        }
        return tree;
    }

    Octree buildOctree(const std::vector<Point3D>& points,
                                int leafSize,
                                int maxDepth)
    {
        Octree tree;
        tree.points = &points;
        tree.indices.resize(points.size());
        for (int i = 0; i < (int)points.size(); ++i) {
            tree.indices[i] = i;
        }

        tree.nodes.clear();
        if (!points.empty()) {
            tree.nodes.reserve(points.size() * 2);
            tree.root = buildNode(tree,
                                /*start=*/0,
                                /*count=*/(int)points.size(),
                                /*depth=*/0,
                                leafSize,
                                maxDepth,
                                /*parallelRoot=*/true);
        } else {
            tree.root = -1;
        }
        return tree;
    }


