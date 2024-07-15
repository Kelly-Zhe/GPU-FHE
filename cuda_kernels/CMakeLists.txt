cmake_minimum_required(VERSION 3.28)
project(GPU_FHE_CUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(pybind11 REQUIRED)

add_executable(moddown_core_cuda moddown_kernel.cpp
        kernels.h)
add_executable(modup_core_cuda modup_kernel.cpp
        kernels.h)

pybind11_add_module(fhe_cuda binding.cpp)