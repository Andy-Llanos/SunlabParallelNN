// bruteForceKNN.cu

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <unordered_set>

// adjust this include to where Point3D lives
#include "../cpu/utils.hpp"   // assumes: struct Point3D { float x, y, z; };

#define CUDA_CHECK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << std::endl;
        std::exit(code);
    }
}

constexpr int MAX_K = 64;

__device__ float dist2(const float3& a, const float3& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// 1 thread = 1 query
__global__ void knn_bruteforce_kernel(
    const float3* __restrict__ points,
    int N,
    const float3* __restrict__ queries,
    int Q,
    int k,
    int* __restrict__ outIndices  // Q * k
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= Q) return;

    const float3 q = queries[qi];

    float bestDist[MAX_K];
    int   bestIdx [MAX_K];

    // init with +inf
    for (int i = 0; i < k; ++i) {
        bestDist[i] = 1e30f;
        bestIdx[i]  = -1;
    }

    // brute-force scan
    for (int pi = 0; pi < N; ++pi) {
        float d2 = dist2(q, points[pi]);

        // find current worst (largest distance) in top-k buffer
        int   worstPos  = 0;
        float worstDist = bestDist[0];
        for (int j = 1; j < k; ++j) {
            if (bestDist[j] > worstDist) {
                worstDist = bestDist[j];
                worstPos  = j;
            }
        }

        if (d2 < worstDist) {
            bestDist[worstPos] = d2;
            bestIdx[worstPos]  = pi;
        }
    }

    // write out indices
    int base = qi * k;
    for (int j = 0; j < k; ++j) {
        outIndices[base + j] = bestIdx[j];
    }
}

// Host wrapper: same logical signature as CPU brute-force
std::vector<std::vector<int>> bruteForceKNN_GPU(
    const std::vector<Point3D>& points,
    const std::vector<Point3D>& queries,
    int k,
    int blockSize)
{
    assert(k > 0 && k <= MAX_K);

    int N = (int)points.size();
    int Q = (int)queries.size();

    // Host: convert to float3 arrays
    std::vector<float3> h_points(N);
    std::vector<float3> h_queries(Q);

    for (int i = 0; i < N; ++i) {
        h_points[i] = make_float3(points[i].x, points[i].y, points[i].z);
    }
    for (int i = 0; i < Q; ++i) {
        h_queries[i] = make_float3(queries[i].x, queries[i].y, queries[i].z);
    }

    // Device memory
    float3* d_points  = nullptr;
    float3* d_queries = nullptr;
    int*    d_indices = nullptr;

    size_t pointsBytes  = N * sizeof(float3);
    size_t queriesBytes = Q * sizeof(float3);
    size_t indicesBytes = Q * k * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_points,  pointsBytes));
    CUDA_CHECK(cudaMalloc(&d_queries, queriesBytes));
    CUDA_CHECK(cudaMalloc(&d_indices, indicesBytes));

    CUDA_CHECK(cudaMemcpy(d_points,  h_points.data(),
                          pointsBytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(),
                          queriesBytes, cudaMemcpyHostToDevice));

    // Launch config: 1 thread = 1 query
    int gridSize = (Q + blockSize - 1) / blockSize;

    knn_bruteforce_kernel<<<gridSize, blockSize>>>(
        d_points, N,
        d_queries, Q,
        k,
        d_indices
    );

    // synchronize + error check
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<int> h_indices(Q * k);
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices,
                          indicesBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_indices));

    // reshape into [Q][k]
    std::vector<std::vector<int>> result(Q, std::vector<int>(k));
    for (int qi = 0; qi < Q; ++qi) {
        for (int j = 0; j < k; ++j) {
            result[qi][j] = h_indices[qi * k + j];
        }
    }

    return result;
}
