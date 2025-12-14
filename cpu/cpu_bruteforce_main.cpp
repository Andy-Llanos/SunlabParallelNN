#include <vector>
#include <iostream>
#include <random>
#include "utils.hpp"
#include "octree_knn.hpp"
#include "octree_knn.cpp"



// forward-declare bruteForceKNN
std::vector<std::vector<int>> bruteForceKNN(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k);

int main() {
    int N = 1000;
    int Q = 10;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<Point3D> points(N), queries(Q);
    for (auto& p : points) p = Point3D{dist(rng), dist(rng), dist(rng)};
    for (auto& q : queries) q = Point3D{dist(rng), dist(rng), dist(rng)};

    auto neighbors = bruteForceKNN(points, queries, k);

    std::cout << "Brute-force KNN finished. Example neighbors for query 0:\n";
    for (int idx : neighbors[0]) {
        std::cout << idx << " ";
    }
    std::cout << "\n";
    return 0;
}
