cmake_minimum_required(VERSION 3.10)
project(ParallelNN LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

add_subdirectory(cpu)
add_subdirectory(gpu)

file(GLOB GPU_SRC
    gpu_knn_bruteforce.cu
    gpu_knn_octree.cu
    test_gpu_kernels.cpp
)

add_executable(test_gpu_kernels ${GPU_SRC})
set_target_properties(test_gpu_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
