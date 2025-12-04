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

void test_distance() {
    Point3D a{0,0,0};
    Point3D b{3,4,0};
    float d2 = distance(a, b);   // or distance^2
    if (std::abs(d2 - 25.0f) > 1e-5f) {
        std::cerr << "distance test FAILED\n";
        std::exit(1);
    }
}

void test_bruteforce_small() {
    std::vector<Point3D> points = {
        {0,0,0},  // 0
        {1,0,0},  // 1
        {2,0,0},  // 2
        {3,0,0}   // 3
    };

    std::vector<Point3D> queries = {
        {0.1f,0,0},  // close to 0,1
        {2.4f,0,0}   // close to 2,3
    };

    int k = 2;
    auto neigh = bruteForceKNN(points, queries, k);

    // query 0: nearest should be 0, then 1
    if (neigh[0].size() != 2 ||
        neigh[0][0] != 0 || neigh[0][1] != 1) {
        std::cerr << "KNN small test FAILED for query 0\n";
        std::exit(1);
    }

    // query 1: nearest should be 2, then 3
    if (neigh[1].size() != 2 ||
        neigh[1][0] != 2 || neigh[1][1] != 3) {
        std::cerr << "KNN small test FAILED for query 1\n";
        std::exit(1);
    }

    std::cout << "test_bruteforce_small PASSED\n";
}


void test_bruteforce_edges() {
    std::vector<Point3D> points = {
        {0,0,0}, {0,0,0}, {1,0,0}
    };
    std::vector<Point3D> queries = {
        {0,0,0}
    };

    auto neigh_k1 = bruteForceKNN(points, queries, 1);
    auto neigh_k3 = bruteForceKNN(points, queries, 3);

    if (neigh_k1[0].size() != 1) {
        std::cerr << "k=1 size FAILED\n"; std::exit(1);
    }

    if (neigh_k3[0].size() != 3) {
        std::cerr << "k=N size FAILED\n"; std::exit(1);
    }

    std::cout << "test_bruteforce_edges PASSED\n";
}
int main() {

    test_distance();
    //
    test_bruteforce_small();
    ///
    test_bruteforce_edges();

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