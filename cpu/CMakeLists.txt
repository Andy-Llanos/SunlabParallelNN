cmake_minimum_required(VERSION 3.10)
project(ParallelNN LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

add_subdirectory(cpu)
add_subdirectory(gpu)

file(GLOB CPU_SRC
    brute_force_knn.cpp
    octree.cpp
    octree_knn.cpp
    parallel_knn.cpp
    test_cpu_kernels.cpp
)

add_executable(test_cpu_kernels ${CPU_SRC})
target_include_directories(test_cpu_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
