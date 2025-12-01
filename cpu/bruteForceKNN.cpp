#include "utils.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <iostream>
//Andy llanos
//cse 375 final project

std::vector<std::vector<int>> bruteForceKNN(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k){
    std::vector<std::vector<int>> neighbors(queries.size());
    ///will iterate over each queries with qi
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        // max-heap for k smallest distances
        std::priority_queue<std::pair<float,int>> heap;
        //loop over all points with pi
        for (size_t pi = 0; pi < points.size(); ++pi) {

            float d = distance(points[pi], queries[qi]);
            //maintain heap of size k
            if (heap.size() < k) {
                heap.push({d, pi});
            } else if (d < heap.top().first) {
                heap.pop();
                heap.push({d, pi});
            }
        }

        // dump results from heap, after scanning all points we extract indices 
        std::vector<int> temp;

        while (!heap.empty()) {
            temp.push_back(heap.top().second);
            heap.pop();
        }

        //reverse to heaver nearest first order
        neighbors[qi].assign(temp.rbegin(), temp.rend());

    }

    return neighbors;
}

int main() {
    int N = 1000;
    int Q = 10;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<Point3D> points(N), queries(Q);
    for (auto& p : points) p = {dist(rng), dist(rng), dist(rng)};
    for (auto& q : queries) q = {dist(rng), dist(rng), dist(rng)};

    auto neighbors = bruteForceKNN(points, queries, k);

    std::cout << "Brute-force KNN finished. Example neighbors for query 0:\n";
    for (int idx : neighbors[0]) {
        std::cout << idx << " ";
    }
    std::cout << "\n";
    return 0;
}

