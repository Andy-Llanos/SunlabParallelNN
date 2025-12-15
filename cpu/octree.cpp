// Octree construction implementation
// CSE 375 Final Project - ParallelNN
#include <limits>
#include <cassert>
#include <iostream>
#include <random>
#include <mutex>
#define TBB_PREVIEW_GLOBAL_CONTROL 1  // needed for TBB thread control, took me forever to figure this out
#include <tbb/parallel_for.h>

#include "octree.hpp"

// global mutex for node allocation - needed because multiple threads
// are trying to push_back to the nodes vector at the same time
static std::mutex g_nodeAllocMutex;

// helper: compute bounding box for a range of points
// basically just find min/max in each dimension
static void computeBoundingBox(const std::vector<Point3D>& pts,
                               const std::vector<int>& indices,
                               int start, int count,
                               Point3D& bb_min,
                               Point3D& bb_max)
{
    float inf = std::numeric_limits<float>::infinity();
    bb_min = { inf,  inf,  inf};   // start with infinity
    bb_max = {-inf, -inf, -inf};   // start with -infinity

    // loop through all points in this range and update min/max
    for (int i = 0; i < count; ++i) {
        const Point3D& p = pts[indices[start + i]];
        // update mins
        if (p.x < bb_min.x) bb_min.x = p.x;
        if (p.y < bb_min.y) bb_min.y = p.y;
        if (p.z < bb_min.z) bb_min.z = p.z;
        // update maxs
        if (p.x > bb_max.x) bb_max.x = p.x;
        if (p.y > bb_max.y) bb_max.y = p.y;
        if (p.z > bb_max.z) bb_max.z = p.z;
    }
}

// figure out which of the 8 octants a point belongs to
// using bit tricks: x gives bit 0, y gives bit 1, z gives bit 2
// so we get a number from 0-7 representing the octant
static int classifyOctant(const Point3D& p, float cx, float cy, float cz) {
    int oct = 0;
    if (p.x >= cx) oct |= 1;  // set bit 0
    if (p.y >= cy) oct |= 2;  // set bit 1
    if (p.z >= cz) oct |= 4;  // set bit 2
    return oct;  // returns 0..7
}

// recursive function to build octree nodes
// this is where the actual tree construction happens
static int buildNode(Octree& tree,
                     int start, int count,
                     int depth,
                     int leafSize,
                     int maxDepth,
                     bool parallelRoot)
{
    assert(count >= 0);

    // allocate a new node - MUST use mutex here because
    // multiple threads might be allocating nodes at the same time
    int nodeIndex;
    {
        std::lock_guard<std::mutex> lock(g_nodeAllocMutex);  // lock the mutex
        nodeIndex = (int)tree.nodes.size();
        tree.nodes.push_back({});
    }  // mutex unlocks here automatically
    OctreeNode& node = tree.nodes[nodeIndex];


    node.start = start;
    node.count = count;
    node.children.fill(-1);  // initially no children (-1 means empty)

    // compute bounding box for this node
    computeBoundingBox(*tree.points, tree.indices,
                       start, count,
                       node.bb_min, node.bb_max);

    // base case: stop if we have few enough points or we're too deep
    // just make this a leaf node
    if (count <= leafSize || depth >= maxDepth) {
        return nodeIndex;
    }

    // find center of bounding box - this is where we'll split
    float cx = 0.5f * (node.bb_min.x + node.bb_max.x);
    float cy = 0.5f * (node.bb_min.y + node.bb_max.y);
    float cz = 0.5f * (node.bb_min.z + node.bb_max.z);

    // first pass: count how many points go in each octant
    // need this to figure out where to put them in the reordered array
    int bucketCounts[8] = {0,0,0,0,0,0,0,0};

    for (int i = 0; i < count; ++i) {
        int idx = tree.indices[start + i];
        const Point3D& p = (*tree.points)[idx];
        int oct = classifyOctant(p, cx, cy, cz);
        bucketCounts[oct]++;
    }

    // check for degenerate case - if all points went into one bucket
    // then splitting won't help and we might infinite loop
    int nonEmptyBuckets = 0;
    for (int b = 0; b < 8; ++b)
        if (bucketCounts[b] > 0) nonEmptyBuckets++;

    if (nonEmptyBuckets <= 1) {
        // everything in one bucket - don't split, just make this a leaf
        // this happens sometimes with KITTI data where points are on a plane
        return nodeIndex;
    }

    // second pass: compute prefix sums to figure out where each bucket starts
    // this is like the parallel scan we learned about in class
    int bucketOffsets[8];
    int offset = 0;
    for (int b = 0; b < 8; ++b) {
        bucketOffsets[b] = offset;
        offset += bucketCounts[b];
    }
    assert(offset == count);  // sanity check - should add up to total

    // third pass: actually reorder the indices array by octant
    // need a temp array because we can't do it in-place
    std::vector<int> temp(count);
    int cursor[8];  // tracks where we are in each bucket
    for (int b = 0; b < 8; ++b)
        cursor[b] = bucketOffsets[b];

    for (int i = 0; i < count; ++i) {
        int idx = tree.indices[start + i];
        const Point3D& p = (*tree.points)[idx];
        int oct = classifyOctant(p, cx, cy, cz);
        temp[cursor[oct]++] = idx;  // put in right bucket and increment cursor
    }

    // copy temp back into the actual indices array
    for (int i = 0; i < count; ++i) {
        tree.indices[start + i] = temp[i];
    }

    // now recursively build children for non-empty buckets
    // this is the key parallelism innovation!
    if (depth == 0 && parallelRoot) {
        // PARALLEL VERSION: at root level, build all 8 children in parallel
        // this is Algorithm 1 from the paper
        tbb::parallel_for(0, 8, [&](int b) {
            int childCount = bucketCounts[b];
            if (childCount == 0) {
                node.children[b] = -1;  // no points in this octant
                return;
            }
            int childStart = start + bucketOffsets[b];
            // recursively build this child
            // note: parallelRoot=false so we don't parallelize deeper levels
            node.children[b] = buildNode(tree,
                                        childStart,
                                        childCount,
                                        depth + 1,
                                        leafSize,
                                        maxDepth,
                                        /*parallelRoot=*/false);
        });
    } else {
        // SEQUENTIAL VERSION: for deeper levels or sequential build
        // just loop through children one by one
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

// ---- public entry points ----

// Sequential octree build (baseline for comparison)
Octree buildOctree_seq(const std::vector<Point3D>& points,
                    int leafSize,
                    int maxDepth)
{
    Octree tree;
    tree.points = &points;  // non-owning pointer to points

    // initialize indices array to [0, 1, 2, ..., N-1]
    tree.indices.resize(points.size());
    for (int i = 0; i < (int)points.size(); ++i) {
        tree.indices[i] = i;
    }

    tree.nodes.clear();
    if (!points.empty()) {
        tree.nodes.reserve(points.size() * 2);  // preallocate roughly 2N nodes
        tree.root = buildNode(tree,
                            /*start=*/0,
                            /*count=*/(int)points.size(),
                            /*depth=*/0,
                            leafSize,
                            maxDepth,
                            /*parallelRoot=*/false);  // sequential build
    } else {
        tree.root = -1;
    }
    return tree;
}

// Parallel octree build - this is the main innovation!
// parallelizes the 8 root children
Octree buildOctree(const std::vector<Point3D>& points,
                            int leafSize,
                            int maxDepth)
{
    Octree tree;
    tree.points = &points;  // non-owning pointer

    // init indices to identity permutation
    tree.indices.resize(points.size());
    for (int i = 0; i < (int)points.size(); ++i) {
        tree.indices[i] = i;
    }

    tree.nodes.clear();
    if (!points.empty()) {
        tree.nodes.reserve(points.size() * 2);  // estimate ~2N nodes
        tree.root = buildNode(tree,
                            /*start=*/0,
                            /*count=*/(int)points.size(),
                            /*depth=*/0,
                            leafSize,
                            maxDepth,
                            /*parallelRoot=*/true);  // PARALLEL at root!
    } else {
        tree.root = -1;
    }
    return tree;
}


