
file(GLOB GPU_SRC
    bruteForceKNN.cu
    octree.cu
    test_gpu_kernels.cpp
)

add_executable(gpu_tests ${GPU_SRC})
set_target_properties(gpu_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
