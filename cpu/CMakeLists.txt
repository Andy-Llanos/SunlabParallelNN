# Manual TBB linking for systems without CMake TBB support
find_library(TBB_LIB tbb REQUIRED)

# Tests executable (uses bruteForceKNN, octree, octree_knn)
add_executable(cpu_octree_tests
    cpu_octree_tests.cpp
    bruteForceKNN.cpp
    octree.cpp
    octree_knn.cpp
)
target_include_directories(cpu_octree_tests PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Standalone brute-force demo executable
add_executable(cpu_bruteforce_knn
    cpu_bruteforce_main.cpp
    bruteForceKNN.cpp
)
target_include_directories(cpu_bruteforce_knn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cpu_octree_tests PUBLIC ${TBB_LIB})
