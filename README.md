# DEFT XGBoost on H20 (Performance Model)

This repo implements the DEFT-style **performance prediction model** for H20 using XGBoost and Nsight Compute counters at the **baseline frequency**.

## Quick Start

### 1) Python environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build CUDA binaries
```
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

You will get:
- `build/microbench`
- `build/tasks_runner`

### 3) Phase 0: Verify frequency grid
```
python tools/dvfs_nvml.py --config config/h20.yaml --print-grid
```

### 4) Phase 1: Collect training data
Run microbenches across the grid:
```
python tools/runner.py bench --exe build/microbench --job-config config/train_benches.yaml
```
Profile microbenches at baseline for U vectors:
```
python tools/ncu_profile.py bench --exe build/microbench --job-config config/train_benches.yaml
python tools/features.py --mode bench
```
Build datasets:
```
python tools/dataset_build.py
```

### 5) Phase 2: Train XGBoost
```
python tools/train_xgb.py
```

### 6) Phase 3: Unseen task evaluation
Run tasks across the grid:
```
python tools/runner.py task --exe build/tasks_runner --job-config config/test_tasks.yaml
```
Profile tasks at baseline:
```
python tools/ncu_profile.py task --exe build/tasks_runner --job-config config/test_tasks.yaml
python tools/features.py --mode task
```
Predict + evaluate:
```
python tools/predict_eval.py
python tools/report.py
```

### 7) Phase 4: U-drift probe (optional)
```
python tools/drift_probe.py --exe build/tasks_runner
```

## Notes
- `runner.py` sets DVFS via NVML/nvidia-smi; if locks fail it will warn and continue.
- Nsight Compute is only used at baseline by default (per DEFT).
- Output artifacts are under `data/`.

## Outputs
- `data/datasets/train.parquet`, `val.parquet`, `test_tasks.parquet`
- `data/models/xgb_perf.json`, `meta.json`
- `data/reports/` (summary + plots)
