file(GLOB CPU_SRC
    bruteForceKNN.cpp
    octree.cpp
    octree.cpp
    utils.hpp
)

add_executable(cpu_tests ${CPU_SRC})
target_include_directories(cpu_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(cpu_bruteforce_knn bruteForceKNN.cpp)
target_include_directories(cpu_bruteforce_knn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

