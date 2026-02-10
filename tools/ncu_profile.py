import argparse
import os
import subprocess
import time
from typing import Any, Dict, List

from dvfs_nvml import DvfsController, build_grid_from_config
from utils import ensure_dir, load_yaml, stable_id


def _expand_params(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    from itertools import product

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    out = []
    for combo in product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _jobs(cfg_path: str, mode: str) -> List[Dict[str, Any]]:
    cfg = load_yaml(cfg_path)
    key = "benches" if mode == "bench" else "tasks"
    jobs = []
    for item in cfg.get(key, []):
        name = item["name"]
        param_grid = item.get("params", {})
        for params in _expand_params(param_grid):
            jobs.append({"name": name, "params": params})
    return jobs


def _format_params(params: Dict[str, Any]) -> str:
    return ",".join([f"{k}={v}" for k, v in sorted(params.items())])


def run_ncu(cmd: List[str], env: Dict[str, str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"NCU failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile with Nsight Compute at baseline")
    parser.add_argument("mode", choices=["bench", "task"])
    parser.add_argument("--exe", required=True)
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--out-dir", default="data/ncu")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ncu_cfg = load_yaml(cfg["ncu"]["metrics_config"])
    metrics = ncu_cfg["collection"]["metrics"]
    metrics_arg = ",".join(metrics)

    ctrl = DvfsController(cfg["device"]["gpu_index"])
    if cfg.get("device", {}).get("persistence_mode"):
        ctrl.set_persistence_mode(True)
    pairs = ctrl.query_supported_clocks()
    grid_info = build_grid_from_config(cfg, pairs)
    baseline = grid_info["baseline"]
    override = cfg.get("baseline", {}).get("override_mhz", {})
    if override.get("mem") and override.get("gpu"):
        baseline = (override["mem"], override["gpu"])

    # set baseline clocks
    ctrl.set_app_clocks(baseline[0], baseline[1])
    ctrl.wait_stable(
        baseline[0],
        baseline[1],
        cfg["dvfs"]["stable_check"]["tol_mhz"],
        cfg["dvfs"]["stable_check"]["consecutive"],
        cfg["dvfs"]["stable_check"]["max_wait_s"],
        cfg["dvfs"]["stable_check"]["poll_ms"],
    )

    jobs = _jobs(args.job_config, args.mode)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["device"]["gpu_index"])
    ensure_dir(args.out_dir)

    for job in jobs:
        job_id = stable_id(job["name"], job["params"])
        out_csv = os.path.join(args.out_dir, f"{args.mode}_{job_id}.csv")
        cmd = ["ncu", "--metrics", metrics_arg, "--csv"]
        extra_args = cfg.get("ncu", {}).get("extra_args", [])
        if extra_args:
            cmd.extend(extra_args)
        kernel_regex = cfg.get("ncu", {}).get("kernel_regex")
        if kernel_regex and kernel_regex != ".*":
            cmd.extend(["--kernel-regex", kernel_regex])
        launch_count = cfg.get("ncu", {}).get("launch_count")
        if launch_count is not None:
            cmd.extend(["--launch-count", str(launch_count)])
        cmd += [
            args.exe,
            f"--{args.mode}",
            job["name"],
            "--params",
            _format_params(job["params"]),
        ]
        stdout = run_ncu(cmd, env)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(stdout)
        print(f"[NCU] {out_csv}")


if __name__ == "__main__":
    main()
