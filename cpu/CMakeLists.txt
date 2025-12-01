

file(GLOB CPU_SRC
    bruteForceKNN.cpp
    octree.cpp
    octree_knn.cpp
    parallel_knn.cpp
    test_cpu_kernels.cpp
)

add_executable(cpu_tests ${CPU_SRC})
target_include_directories(cpu_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

