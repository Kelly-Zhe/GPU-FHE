cmake_minimum_required(VERSION 3.28)
project(GPU_FHE_CUDA LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit REQUIRED)
add_executable(moddown_core_cuda moddown_kernel.cu)
target_include_directories(moddown_core_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(modup_core_cuda modup_kernel.cu)
target_include_directories(modup_core_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})