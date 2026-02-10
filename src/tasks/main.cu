#include "tasks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

static ParamMap parse_params_str(const std::string& s) {
    ParamMap out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto pos = item.find('=');
        if (pos == std::string::npos) continue;
        std::string key = item.substr(0, pos);
        std::string val = item.substr(pos + 1);
        out[key] = std::atof(val.c_str());
    }
    return out;
}

TaskParams parse_task_params(const ParamMap& params) {
    TaskParams p;
    if (params.count("m")) p.m = (int)params.at("m");
    if (params.count("n")) p.n = (int)params.at("n");
    if (params.count("k")) p.k = (int)params.at("k");
    if (params.count("batch")) p.batch = (int)params.at("batch");
    if (params.count("vocab")) p.vocab = (int)params.at("vocab");
    if (params.count("dim")) p.dim = (int)params.at("dim");
    if (params.count("b")) p.b = (int)params.at("b");
    if (params.count("h")) p.h = (int)params.at("h");
    if (params.count("seqlen")) p.seqlen = (int)params.at("seqlen");
    return p;
}

float run_task(const std::string& name, const TaskParams& p) {
    if (name == "gemm") return run_task_gemm(p);
    if (name == "reduction") return run_task_reduction(p);
    if (name == "softmax") return run_task_softmax(p);
    if (name == "layernorm") return run_task_layernorm(p);
    if (name == "embedding_gather") return run_task_embedding_gather(p);
    if (name == "attention_like") return run_task_attention_like(p);
    throw std::runtime_error("Unknown task: " + name);
}

int main(int argc, char** argv) {
    std::string task;
    std::string params_str;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            task = argv[++i];
        } else if (std::strcmp(argv[i], "--params") == 0 && i + 1 < argc) {
            params_str = argv[++i];
        }
    }

    if (task.empty()) {
        std::cerr << "Usage: tasks_runner --task <name> --params key=val,..." << std::endl;
        return 1;
    }

    ParamMap params = parse_params_str(params_str);
    TaskParams p = parse_task_params(params);

    try {
        float ms = run_task(task, p);
        std::cout << "{\"time_ms\":" << ms << "}" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
