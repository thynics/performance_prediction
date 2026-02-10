cmake_minimum_required(VERSION 3.18)
project(deft_xgb_h20 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

add_executable(microbench
    src/microbench/main.cu
    src/microbench/microbench.cu
)

add_executable(tasks_runner
    src/tasks/main.cu
    src/tasks/gemm.cu
    src/tasks/reduction.cu
    src/tasks/softmax.cu
    src/tasks/layernorm.cu
    src/tasks/embedding_gather.cu
    src/tasks/attention_like.cu
)

set_target_properties(microbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tasks_runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
