#include "octree.hpp"
#include <limits>
#include <cassert>

// helper: compute bbox of points[indices[start..start+count))
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



// helper: classify a point into one of 8 octants around (cx, cy, cz)
//the octant indez is encoded in three bits :
/*
bit 0 (1) : x>= cx(positive x side)
bit 1 (2) : y>= cy(positive y side)
bit 2 (4) : z>= cz(positive z side)

returned value is in [0..7] examples:
( x < cx, y < cy, z < cz) => 0
( x >=cx, y < cy, z < cz) => 1
( x < cx, y >=cy, z < cz) => 2
( x >=cx, y >=cy, z < cz) => 3
( x < cx, y < cy, z >=cz) => 4
( x >=cx, y < cy, z >=cz) => 5
( x < cx, y >=cy, z >=cz) => 6
( x >=cx, y >=cy, z >=cz) => 7
 notes:
 Tie-breaking: points exactly on a splitting plane (p.x == cx, etc.)
//   are placed on the ">= center" side. This is deliberate and consistent
//   with how points are binned later; change to '>' or add epsilon if a
//   different tie policy is required.
// - Constant time, no side-effects.
*/
static int classifyOctant(const Point3D& p, float cx, float cy, float cz) {
    int oct = 0;
    if (p.x >= cx) oct |= 1;
    if (p.y >= cy) oct |= 2;
    if (p.z >= cz) oct |= 4;
    return oct;  // 0..7
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

    if (points.empty()) {
        tree.root = -1;
        return tree;
    }

    tree.nodes.clear();
    tree.nodes.reserve(16);
    tree.nodes.push_back({});
    tree.root = 0;

    OctreeNode& root = tree.nodes[0];
    root.start = 0;
    root.count = (int)points.size();
    root.children.fill(-1);

    computeBoundingBox(points, tree.indices,
                       root.start, root.count,
                       root.bb_min, root.bb_max);

    // if too few points or maxDepth == 0, keep as leaf
    if (root.count <= leafSize || maxDepth <= 0) {
        return tree;
    }

    // compute center of bbox
    float cx = 0.5f * (root.bb_min.x + root.bb_max.x);
    float cy = 0.5f * (root.bb_min.y + root.bb_max.y);
    float cz = 0.5f * (root.bb_min.z + root.bb_max.z);

    // first pass: count points per octant
    int bucketCounts[8] = {0,0,0,0,0,0,0,0};

    for (int i = 0; i < root.count; ++i) {
        int idx = tree.indices[root.start + i];
        const Point3D& p = points[idx];
        int oct = classifyOctant(p, cx, cy, cz);
        bucketCounts[oct]++;
    }

    // second pass: prefix sums for offsets
    int bucketOffsets[8];
    int offset = 0;
    for (int b = 0; b < 8; ++b) {
        bucketOffsets[b] = offset;
        offset += bucketCounts[b];
    }

    // third pass: reorder into temp array
    std::vector<int> temp(root.count);
    int cursor[8];
    for (int b = 0; b < 8; ++b)
        cursor[b] = bucketOffsets[b];

    for (int i = 0; i < root.count; ++i) {
        int idx = tree.indices[root.start + i];
        const Point3D& p = points[idx];
        int oct = classifyOctant(p, cx, cy, cz);
        temp[cursor[oct]++] = idx;
    }

    // copy back into tree.indices
    for (int i = 0; i < root.count; ++i) {
        tree.indices[root.start + i] = temp[i];
    }

    // create child nodes for non-empty buckets
    for (int b = 0; b < 8; ++b) {
        int childCount = bucketCounts[b];
        if (childCount == 0) {
            root.children[b] = -1;
            continue;
        }
        int childStart = root.start + bucketOffsets[b];

        int childIndex = (int)tree.nodes.size();
        tree.nodes.push_back({});
        OctreeNode& child = tree.nodes.back();
        child.start = childStart;
        child.count = childCount;
        child.children.fill(-1);

        computeBoundingBox(points, tree.indices,
                           child.start, child.count,
                           child.bb_min, child.bb_max);

        root.children[b] = childIndex;
    }

    return tree;
}




