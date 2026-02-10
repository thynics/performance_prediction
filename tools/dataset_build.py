import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils import load_yaml


def _load_runs(raw_dir: str, mode: str) -> List[Dict[str, Any]]:
    records = []
    for path in glob.glob(os.path.join(raw_dir, f"runs_{mode}_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def _load_features(feature_dir: str, mode: str) -> Dict[str, Dict[str, Any]]:
    feats = {}
    for path in glob.glob(os.path.join(feature_dir, f"{mode}_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        feats[obj["bench_instance_id"]] = obj
    return feats


def build_dataset(raw_dir: str, feature_dir: str, mode: str) -> pd.DataFrame:
    runs = _load_runs(raw_dir, mode)
    feats = _load_features(feature_dir, mode)
    print(f"[dataset] mode={mode} runs={len(runs)} features={len(feats)}")

    # Baseline lookup
    baseline_time = {}
    baseline_freq = {}
    for r in runs:
        bid = r["bench_instance_id"]
        base = r.get("baseline")
        if not base:
            continue
        if r["freq_target"]["mem"] == base["mem"] and r["freq_target"]["gpu"] == base["gpu"]:
            baseline_time[bid] = r["time_stat"]["median"]
            baseline_freq[bid] = base

    rows = []
    for r in runs:
        bid = r["bench_instance_id"]
        if bid not in feats or bid not in baseline_time:
            continue
        u = feats[bid]
        u_dims = u["u_dims"]
        u_vals = u["u_values"]
        base = baseline_freq[bid]
        t_default = baseline_time[bid]
        t_f = r["time_stat"]["median"]
        if t_f <= 0:
            continue
        s = t_default / t_f
        r_gpu = r["freq_target"]["gpu"] / base["gpu"]
        r_mem = r["freq_target"]["mem"] / base["mem"]
        row = {
            "bench_instance_id": bid,
            "name": r["name"],
            "run_type": r["run_type"],
            "R_gpu": r_gpu,
            "R_mem": r_mem,
            "T_default": t_default,
            "T_f": t_f,
            "S": s,
            "actual_gpu_clk": r["freq_actual"]["gpu"],
            "actual_mem_clk": r["freq_actual"]["mem"],
        }
        for i, dim in enumerate(u_dims):
            row[dim] = u_vals[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datasets from raw runs and features")
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--feature-dir", default="data/ncu/features")
    parser.add_argument("--out-dir", default="data/datasets")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = cfg["dataset"]["out_dir"] if cfg.get("dataset") else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Bench dataset
    df_bench = build_dataset(args.raw_dir, args.feature_dir, "bench")
    bench_path = os.path.join(out_dir, "bench.parquet")
    df_bench.to_parquet(bench_path, index=False)
    print(f"[dataset] bench rows={len(df_bench)} -> {bench_path}")

    if not df_bench.empty:
        # Group split
        val_ratio = cfg["train"]["split"]["val_ratio"]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=cfg["train"]["random_seed"])
        train_idx, val_idx = next(gss.split(df_bench, groups=df_bench["bench_instance_id"]))
        df_train = df_bench.iloc[train_idx]
        df_val = df_bench.iloc[val_idx]
        train_path = os.path.join(out_dir, "train.parquet")
        val_path = os.path.join(out_dir, "val.parquet")
        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)
        print(f"[dataset] train rows={len(df_train)} -> {train_path}")
        print(f"[dataset] val rows={len(df_val)} -> {val_path}")

    # Task dataset
    df_task = build_dataset(args.raw_dir, args.feature_dir, "task")
    test_path = os.path.join(out_dir, "test_tasks.parquet")
    df_task.to_parquet(test_path, index=False)
    print(f"[dataset] test_tasks rows={len(df_task)} -> {test_path}")

    print(f"[dataset] done")


if __name__ == "__main__":
    main()
