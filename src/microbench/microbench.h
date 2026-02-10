#pragma once

#include <string>
#include <unordered_map>

struct BenchParams {
    int size = 1 << 20;
    int iters = 256;
    int compute = 8;
    int m = 256;
    int n = 256;
    int k = 256;
};

using ParamMap = std::unordered_map<std::string, double>;

BenchParams parse_params(const ParamMap& params);
float run_bench(const std::string& name, const BenchParams& p);
