#pragma once
#include <vector>
#include <queue>
#include <utility>
#include "utils.hpp"
#include "octree.hpp"

// Returns for each query: vector of k indices into 'points'
std::vector<std::vector<int>> octreeKNN(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k);


// TBB-parallel version
std::vector<std::vector<int>> octreeKNN_parallel_tbb(
    const Octree& tree,
    const std::vector<Point3D>& queries,
    int k,
    int maxNodesVisited);
//bruteforce knn cpu
std::vector<std::vector<int>> bruteForceKNN(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k);