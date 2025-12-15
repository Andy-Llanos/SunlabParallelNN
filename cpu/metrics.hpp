#pragma once
#include <vector>
#include <unordered_set>
#include <cassert>

// Average recall@k across queries, treating results as sets (order doesn't matter)
float compute_average_recall_at_k(
    const std::vector<std::vector<int>>& exact,
    const std::vector<std::vector<int>>& approx)
{
    assert(exact.size() == approx.size());
    int Q = (int)exact.size();
    if (Q == 0) return 1.0f;

    float sumRecall = 0.0f;

    for (int qi = 0; qi < Q; ++qi) {
        const auto& gt = exact[qi];
        const auto& ap = approx[qi];

        if (gt.empty()) {
            sumRecall += 1.0f;
            continue;
        }

        std::unordered_set<int> gtSet(gt.begin(), gt.end());
        int hit = 0;
        for (int idx : ap) {
            if (gtSet.count(idx)) hit++;
        }

        float recall = (float)hit / (float)gtSet.size();
        sumRecall += recall;
    }

    return sumRecall / (float)Q;
}
