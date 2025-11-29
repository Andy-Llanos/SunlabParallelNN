cmake_minimum_required(VERSION 3.10)
project(ParallelNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

add_executable(test_cuda gpu/test_cuda.cu)
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
