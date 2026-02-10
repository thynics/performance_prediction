#include "tasks.h"
#include "common.h"

#include <math.h>

__global__ void kernel_softmax(const float* in, float* out, int n) {
    __shared__ float smax;
    __shared__ float ssum;
    int row = blockIdx.x;
    const float* row_in = in + row * n;
    float* row_out = out + row * n;

    // compute max
    float local_max = -1e20f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    // reduce max
    __shared__ float buf[256];
    buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (buf[threadIdx.x + stride] > buf[threadIdx.x]) {
                buf[threadIdx.x] = buf[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) smax = buf[0];
    __syncthreads();

    // compute sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += expf(row_in[i] - smax);
    }
    buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ssum = buf[0];
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - smax) / ssum;
    }
}

float run_task_softmax(const TaskParams& p) {
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
    float ms = time_kernel([&]() { kernel_softmax<<<grid, block>>>(in, out, n); });

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return ms;
}
