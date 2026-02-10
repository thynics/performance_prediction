#include "tasks.h"
#include "common.h"

__global__ void kernel_reduction(const float* in, float* out, int n) {
    __shared__ float s[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float v = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        v += in[i];
    }
    s[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out, s[0]);
    }
}

float run_task_reduction(const TaskParams& p) {
    int n = p.n;
    float* in = nullptr;
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 1, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));

    int blocks = 256;
    float ms = time_kernel([&]() { kernel_reduction<<<blocks, 256>>>(in, out, n); });

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return ms;
}
