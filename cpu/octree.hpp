#pragma once
#include <vector>
#include <array>
#include "utils.hpp"



struct OctreeNode {
    Point3D bb_min;// bounding box min corner
    Point3D bb_max;//bouinding box max corner
    std::array<int, 8> children;//indices into octree:: nodes, -1 if no child
    int start;   // index into global index array
    int count; //how many points in this node

    bool isLeaf() const {
        for (int c : children) {
            if (c != -1) return false;
        }
        return true;
    }

};
//whole octree structure over a point cloud
struct Octree {
    const std::vector<Point3D>* points = nullptr;  // non-owning reference to original point cloud
    std::vector<int> indices;                      // permutation of [0..N-1], global indez array for points
    std::vector<OctreeNode> nodes;                 // flat array of nodes
    int root = -1; 

    //protect nodes vector during parallel build!

};

/////build octree(points, leaf_size, maxdepth)
//compute boundingbox()
//parition points among children
///partition into 8 octants
///recursively call buildNode()
////SEQUENTIAL OCTREE CONSTRUCTION

Octree buildOctree_seq(const std::vector<Point3D>& points,int leafSize,int maxDepth);

///PARALLEL OCTREE CONSTURCTION
Octree buildOctree(const std::vector<Point3D>& points,int leaf_size,int maxdepth);

