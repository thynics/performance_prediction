import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from dvfs_nvml import DvfsController, build_grid_from_config
from features import compute_u_vector
from utils import ensure_dir, load_yaml, stable_id


def _expand_params(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    from itertools import product

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    out = []
    for combo in product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _jobs(cfg_path: str) -> List[Dict[str, Any]]:
    cfg = load_yaml(cfg_path)
    jobs = []
    for item in cfg.get("tasks", []):
        name = item["name"]
        param_grid = item.get("params", {})
        for params in _expand_params(param_grid):
            jobs.append({"name": name, "params": params})
    return jobs


def _format_params(params: Dict[str, Any]) -> str:
    return ",".join([f"{k}={v}" for k, v in sorted(params.items())])


def _small_grid(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mems = sorted(set(m for m, _ in pairs))
    sel_mems = sorted({mems[0], mems[len(mems) // 2], mems[-1]})
    grid = []
    for mem in sel_mems:
        gpus = sorted([g for m, g in pairs if m == mem])
        sel_gpus = sorted({gpus[0], gpus[len(gpus) // 2], gpus[-1]})
        for gpu in sel_gpus:
            grid.append((mem, gpu))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe U drift across frequencies")
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--task-config", default="config/test_tasks.yaml")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out-dir", default="data/reports")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ncu_cfg = load_yaml(cfg["ncu"]["metrics_config"])
    metrics = ncu_cfg["collection"]["metrics"]
    metrics_arg = ",".join(metrics)
    extra_args = cfg.get("ncu", {}).get("extra_args", [])

    ctrl = DvfsController(cfg["device"]["gpu_index"])
    if cfg.get("device", {}).get("persistence_mode"):
        ctrl.set_persistence_mode(True)
    pairs = ctrl.query_supported_clocks()
    grid = _small_grid(pairs)
    baseline = build_grid_from_config(cfg, pairs)["baseline"]
    override = cfg.get("baseline", {}).get("override_mhz", {})
    if override.get("mem") and override.get("gpu"):
        baseline = (override["mem"], override["gpu"])

    ensure_dir(args.out_dir)
    rows = []

    # avoid duplicate baseline in grid
    grid = [g for g in grid if g != baseline]

    jobs = _jobs(args.task_config)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["device"]["gpu_index"])
    for job in jobs:
        job_id = stable_id(job["name"], job["params"])

        # baseline U
        baseline_u = None
        for mem, gpu in [baseline] + grid:
            if mem == baseline[0] and gpu == baseline[1]:
                tag = "baseline"
            else:
                tag = f"mem{mem}_gpu{gpu}"

            cmd = ["ncu", "--metrics", metrics_arg, "--csv"]
            if extra_args:
                cmd.extend(extra_args)
            cmd += [
                args.exe,
                "--task",
                job["name"],
                "--params",
                _format_params(job["params"]),
            ]

            if args.dry_run:
                print(" ".join(cmd))
                continue

            ctrl.set_app_clocks(mem, gpu)
            ctrl.wait_stable(
                mem,
                gpu,
                cfg["dvfs"]["stable_check"]["tol_mhz"],
                cfg["dvfs"]["stable_check"]["consecutive"],
                cfg["dvfs"]["stable_check"]["max_wait_s"],
                cfg["dvfs"]["stable_check"]["poll_ms"],
            )

            import subprocess

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
            if proc.returncode != 0:
                raise RuntimeError(f"NCU failed: {proc.stderr}")
            kernel_metrics = _load_kernel_metrics_from_stdout(proc.stdout)
            uvec = compute_u_vector(ncu_cfg, kernel_metrics)

            if tag == "baseline":
                baseline_u = np.array(uvec["values"], dtype=float)
                rows.append({
                    "bench_instance_id": job_id,
                    "tag": tag,
                    "drift_l1": 0.0,
                    "drift_l2": 0.0,
                })
            else:
                vec = np.array(uvec["values"], dtype=float)
                drift_l1 = float(np.sum(np.abs(vec - baseline_u)))
                drift_l2 = float(np.linalg.norm(vec - baseline_u))
                rows.append({
                    "bench_instance_id": job_id,
                    "tag": tag,
                    "drift_l1": drift_l1,
                    "drift_l2": drift_l2,
                })

    if not args.dry_run:
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.out_dir, "drift_probe.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")


def _load_kernel_metrics_from_stdout(stdout: str) -> Dict[str, Dict[str, float]]:
    # reuse csv parser from features but avoid file IO
    import pandas as pd
    from io import StringIO
    from features import _parse_metric_value

    lines = [ln for ln in stdout.splitlines() if ln.strip() and not ln.startswith("==")]
    df = pd.read_csv(StringIO("\n".join(lines)))
    metric_col = "Metric Name"
    value_col = "Metric Value"
    kernel_col = "Kernel Name"
    metrics: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        kernel = str(row[kernel_col])
        metric = str(row[metric_col])
        value = _parse_metric_value(row[value_col])
        metrics.setdefault(kernel, {})[metric] = value
    return metrics


if __name__ == "__main__":
    main()
