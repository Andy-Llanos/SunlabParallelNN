#include "utils.hpp"
#include <vector>
#include <queue>
#include <cmath>

std::vector<std::vector<int>> bruteForceKNN(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k)
{
    std::vector<std::vector<int>> neighbors(queries.size());
    
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        // max-heap for k smallest distances
        std::priority_queue<std::pair<float,int>> heap;

        for (size_t pi = 0; pi < points.size(); ++pi) {
            float d = distance(points[pi], queries[qi]);

            if (heap.size() < k) {
                heap.push({d, pi});
            } else if (d < heap.top().first) {
                heap.pop();
                heap.push({d, pi});
            }
        }

        // dump results
        while (!heap.empty()) {
            neighbors[qi].push_back(heap.top().second);
            heap.pop();
        }
    }

    return neighbors;
}
