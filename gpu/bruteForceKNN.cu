#include <cuda_runtime.h>
#include <cstdio>

__device__ float dist3(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

__global__ void knn_bruteforce_kernel(
    const float3* __restrict__ points, int N,
    const float3* __restrict__ queries, int Q,
    int k,
    int* out_idx)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= Q) return;

    float3 q = queries[qid];

    // simple K buffer
    float bestDist[32];
    int bestIdx[32];

    for (int i = 0; i < k; i++) {
        bestDist[i] = 1e30f;
        bestIdx[i] = -1;
    }

    // brute force
    for (int pi = 0; pi < N; pi++) {
        float d = dist3(points[pi], q);

        // insert if better than worst
        int worst = 0;
        for (int i = 1; i < k; i++)
            if (bestDist[i] > bestDist[worst]) worst = i;

        if (d < bestDist[worst]) {
            bestDist[worst] = d;
            bestIdx[worst] = pi;
        }
    }

    // write k results
    for (int i = 0; i < k; i++) {
        out_idx[qid*k + i] = bestIdx[i];
    }
}
