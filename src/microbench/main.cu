#include "microbench.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

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

int main(int argc, char** argv) {
    std::string bench;
    std::string params_str;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            bench = argv[++i];
        } else if (std::strcmp(argv[i], "--params") == 0 && i + 1 < argc) {
            params_str = argv[++i];
        }
    }

    if (bench.empty()) {
        std::cerr << "Usage: microbench --bench <name> --params key=val,..." << std::endl;
        return 1;
    }

    ParamMap params = parse_params_str(params_str);
    BenchParams p = parse_params(params);

    try {
        float ms = run_bench(bench, p);
        std::cout << "{\"time_ms\":" << ms << "}" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
