#include "tasks.h"
#include "common.h"

#include <math.h>

__global__ void kernel_layernorm(const float* in, float* out, int n) {
    __shared__ float smean;
    __shared__ float svar;
    int row = blockIdx.x;
    const float* row_in = in + row * n;
    float* row_out = out + row * n;

    __shared__ float buf[256];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += row_in[i];
    }
    buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) smean = buf[0] / n;
    __syncthreads();

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = row_in[i] - smean;
        local_var += d * d;
    }
    buf[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) svar = buf[0] / n;
    __syncthreads();

    float inv_std = rsqrtf(svar + 1e-5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] = (row_in[i] - smean) * inv_std;
    }
}

float run_task_layernorm(const TaskParams& p) {
    int n = p.n;
    int batch = p.batch;
    size_t bytes = (size_t)n * batch * sizeof(float);
    float* in = nullptr;
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&in, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMemset(in, 1, bytes));

    dim3 block(256);
    dim3 grid(batch);
    float ms = time_kernel([&]() { kernel_layernorm<<<grid, block>>>(in, out, n); });

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return ms;
}
