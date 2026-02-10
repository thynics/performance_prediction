#pragma once

#include <string>
#include <unordered_map>

struct TaskParams {
    int m = 256;
    int n = 256;
    int k = 256;
    int batch = 256;
    int vocab = 65536;
    int dim = 128;
    int b = 8;
    int h = 8;
    int seqlen = 128;
};

using ParamMap = std::unordered_map<std::string, double>;

TaskParams parse_task_params(const ParamMap& params);
float run_task(const std::string& name, const TaskParams& p);

float run_task_gemm(const TaskParams& p);
float run_task_reduction(const TaskParams& p);
float run_task_softmax(const TaskParams& p);
float run_task_layernorm(const TaskParams& p);
float run_task_embedding_gather(const TaskParams& p);
float run_task_attention_like(const TaskParams& p);
