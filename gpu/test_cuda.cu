#include <cstdio>
#include <cuda_runtime.h>

__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 16;
    float h_a[N], h_b[N], h_c[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t size = N * sizeof(float);

    cudaError_t err;

    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess) { printf("cudaMalloc d_a failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess) { printf("cudaMalloc d_b failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_c, size);
    if (err != cudaSuccess) { printf("cudaMalloc d_c failed: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < N; ++i) {
        printf("%2d: %f + %f = %f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}



