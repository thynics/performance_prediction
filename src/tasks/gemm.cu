#include "tasks.h"
#include "common.h"

__global__ void kernel_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

float run_task_gemm(const TaskParams& p) {
    int M = p.m, N = p.n, K = p.k;
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, a_bytes));
    CUDA_CHECK(cudaMalloc(&B, b_bytes));
    CUDA_CHECK(cudaMalloc(&C, c_bytes));
    CUDA_CHECK(cudaMemset(A, 1, a_bytes));
    CUDA_CHECK(cudaMemset(B, 1, b_bytes));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    float ms = time_kernel([&]() { kernel_gemm<<<grid, block>>>(A, B, C, M, N, K); });

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return ms;
}
