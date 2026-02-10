#include "tasks.h"
#include "common.h"

#include <vector>

__global__ void kernel_embedding_gather(const float* emb, const int* idx, float* out, int batch, int dim, int vocab) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;
    if (tid < total) {
        int row = tid / dim;
        int col = tid % dim;
        int id = idx[row] % vocab;
        out[tid] = emb[id * dim + col];
    }
}

float run_task_embedding_gather(const TaskParams& p) {
    int batch = p.batch;
    int dim = p.dim;
    int vocab = p.vocab;

    size_t emb_bytes = (size_t)vocab * dim * sizeof(float);
    size_t out_bytes = (size_t)batch * dim * sizeof(float);

    float* emb = nullptr;
    int* idx = nullptr;
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&emb, emb_bytes));
    CUDA_CHECK(cudaMalloc(&idx, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out, out_bytes));

    CUDA_CHECK(cudaMemset(emb, 1, emb_bytes));

    std::vector<int> host_idx(batch);
    for (int i = 0; i < batch; ++i) host_idx[i] = i % vocab;
    CUDA_CHECK(cudaMemcpy(idx, host_idx.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (batch * dim + threads - 1) / threads;
    float ms = time_kernel([&]() { kernel_embedding_gather<<<blocks, threads>>>(emb, idx, out, batch, dim, vocab); });

    CUDA_CHECK(cudaFree(emb));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaFree(out));
    return ms;
}
