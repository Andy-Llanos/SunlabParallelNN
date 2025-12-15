#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <random>

#include "utils.hpp"


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

            //float d = distance(points[pi], queries[qi]);
           float d = sqDist(points[pi], queries[qi]); // from utils.hpp

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