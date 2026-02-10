#include "microbench.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (0)

__global__ void kernel_fma_fp32(float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = 1.0f;
        float y = 1.0001f;
        #pragma unroll 4
        for (int i = 0; i < iters; ++i) {
            x = fmaf(x, y, 1.0f);
        }
        out[idx] = x;
    }
}

__global__ void kernel_fma_fp64(double* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = 1.0;
        double y = 1.0001;
        for (int i = 0; i < iters; ++i) {
            x = fma(x, y, 1.0);
        }
        out[idx] = x;
    }
}

__global__ void kernel_int32(int* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int x = 1;
        int y = 3;
        for (int i = 0; i < iters; ++i) {
            x = x * y + 1;
        }
        out[idx] = x;
    }
}

__global__ void kernel_dram_copy(const float* in, float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = 0.0f;
        for (int i = 0; i < iters; ++i) {
            v = in[idx];
            out[idx] = v;
        }
    }
}

__global__ void kernel_shared_bw(const float* in, float* out, int n, int iters) {
    extern __shared__ float s[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < n) {
        s[tid] = in[idx];
        __syncthreads();
        float v = 0.0f;
        for (int i = 0; i < iters; ++i) {
            v += s[tid];
        }
        out[idx] = v;
    }
}

__global__ void kernel_pointer_chase(const int* idxs, int* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cur = idx;
        for (int i = 0; i < iters; ++i) {
            cur = idxs[cur];
        }
        out[idx] = cur;
    }
}

__global__ void kernel_mixed_ai(const float* in, float* out, int n, int iters, int compute) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        for (int i = 0; i < iters; ++i) {
            #pragma unroll 4
            for (int c = 0; c < compute; ++c) {
                v = fmaf(v, 1.0001f, 0.9999f);
            }
        }
        out[idx] = v;
    }
}

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

__global__ void kernel_divergence(float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = 0.0f;
        if (idx % 2 == 0) {
            for (int i = 0; i < iters; ++i) v += 1.0f;
        } else {
            for (int i = 0; i < iters / 4; ++i) v += 1.0f;
        }
        out[idx] = v;
    }
}

__global__ void kernel_atomic(int* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < iters; ++i) {
            atomicAdd(out, 1);
        }
    }
}

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

BenchParams parse_params(const ParamMap& params) {
    BenchParams p;
    if (params.count("size")) p.size = (int)params.at("size");
    if (params.count("iters")) p.iters = (int)params.at("iters");
    if (params.count("compute")) p.compute = (int)params.at("compute");
    if (params.count("m")) p.m = (int)params.at("m");
    if (params.count("n")) p.n = (int)params.at("n");
    if (params.count("k")) p.k = (int)params.at("k");
    return p;
}

static float time_kernel(std::function<void()> launcher) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launcher();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

float run_bench(const std::string& name, const BenchParams& p) {
    const int threads = 256;
    const int blocks = (p.size + threads - 1) / threads;

    if (name == "fma_fp32") {
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(float)));
        float ms = time_kernel([&]() { kernel_fma_fp32<<<blocks, threads>>>(out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "fma_fp64") {
        double* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(double)));
        float ms = time_kernel([&]() { kernel_fma_fp64<<<blocks, threads>>>(out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "int32") {
        int* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(int)));
        float ms = time_kernel([&]() { kernel_int32<<<blocks, threads>>>(out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "dram_copy" || name == "l2_copy") {
        float* in = nullptr;
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, p.size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(float)));
        CUDA_CHECK(cudaMemset(in, 1, p.size * sizeof(float)));
        float ms = time_kernel([&]() { kernel_dram_copy<<<blocks, threads>>>(in, out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "shared_bw") {
        float* in = nullptr;
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, p.size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(float)));
        CUDA_CHECK(cudaMemset(in, 1, p.size * sizeof(float)));
        size_t shmem = threads * sizeof(float);
        float ms = time_kernel([&]() { kernel_shared_bw<<<blocks, threads, shmem>>>(in, out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "pointer_chase") {
        std::vector<int> host(p.size);
        for (int i = 0; i < p.size; ++i) host[i] = (i + 1) % p.size;
        int* idxs = nullptr;
        int* out = nullptr;
        CUDA_CHECK(cudaMalloc(&idxs, p.size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(idxs, host.data(), p.size * sizeof(int), cudaMemcpyHostToDevice));
        float ms = time_kernel([&]() { kernel_pointer_chase<<<blocks, threads>>>(idxs, out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(idxs));
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "mixed_ai") {
        float* in = nullptr;
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, p.size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(float)));
        CUDA_CHECK(cudaMemset(in, 1, p.size * sizeof(float)));
        float ms = time_kernel([&]() { kernel_mixed_ai<<<blocks, threads>>>(in, out, p.size, p.iters, p.compute); });
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "reduction") {
        float* in = nullptr;
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, p.size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
        CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
        int red_blocks = 256;
        float ms = time_kernel([&]() { kernel_reduction<<<red_blocks, 256>>>(in, out, p.size); });
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "divergence") {
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, p.size * sizeof(float)));
        float ms = time_kernel([&]() { kernel_divergence<<<blocks, threads>>>(out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "atomic") {
        int* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, sizeof(int)));
        CUDA_CHECK(cudaMemset(out, 0, sizeof(int)));
        float ms = time_kernel([&]() { kernel_atomic<<<blocks, threads>>>(out, p.size, p.iters); });
        CUDA_CHECK(cudaFree(out));
        return ms;
    }
    if (name == "tensor_gemm") {
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
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        float ms = time_kernel([&]() { kernel_gemm<<<grid, block>>>(A, B, C, M, N, K); });
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B));
        CUDA_CHECK(cudaFree(C));
        return ms;
    }

    throw std::runtime_error("Unknown bench name: " + name);
}
