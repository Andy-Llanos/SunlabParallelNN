cmake_minimum_required(VERSION 3.10)
project(ParallelNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(cpu)
add_subdirectory(gpu)
