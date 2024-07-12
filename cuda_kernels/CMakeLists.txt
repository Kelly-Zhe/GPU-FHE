cmake_minimum_required(VERSION 3.28)
project(GPU_FHE)

set(CMAKE_CXX_STANDARD 14)

add_executable(moddown_gpu moddown_op.cu)
add_executable(modup_gpu modup_op.cu)