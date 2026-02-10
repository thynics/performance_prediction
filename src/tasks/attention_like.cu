#include "tasks.h"
#include "common.h"

__global__ void kernel_attention_like(const float* Q, const float* K, const float* V, float* out,
                                      int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float acc = 0.0f;
        int base = idx * dim;
        for (int d = 0; d < dim; ++d) {
            acc += Q[base + d] * K[base + d];
        }
        out[idx] = acc * V[base];
    }
}

float run_task_attention_like(const TaskParams& p) {
    int total = p.b * p.h * p.seqlen;
    int dim = p.dim;

    size_t bytes = (size_t)total * dim * sizeof(float);
    float* Q = nullptr;
    float* K = nullptr;
    float* V = nullptr;
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&Q, bytes));
    CUDA_CHECK(cudaMalloc(&K, bytes));
    CUDA_CHECK(cudaMalloc(&V, bytes));
    CUDA_CHECK(cudaMalloc(&out, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(Q, 1, bytes));
    CUDA_CHECK(cudaMemset(K, 1, bytes));
    CUDA_CHECK(cudaMemset(V, 1, bytes));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float ms = time_kernel([&]() { kernel_attention_like<<<blocks, threads>>>(Q, K, V, out, total, dim); });

    CUDA_CHECK(cudaFree(Q));
    CUDA_CHECK(cudaFree(K));
    CUDA_CHECK(cudaFree(V));
    CUDA_CHECK(cudaFree(out));
    return ms;
}
