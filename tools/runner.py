import argparse
import glob
import json
import os
import subprocess
import time
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

from dvfs_nvml import DvfsController, build_grid_from_config
from utils import ensure_dir, load_yaml, parse_time_ms_from_stdout, stable_id, percentiles


def _expand_params(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    out = []
    for combo in product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _bench_jobs(cfg_path: str) -> List[Dict[str, Any]]:
    cfg = load_yaml(cfg_path)
    jobs = []
    for bench in cfg.get("benches", []):
        name = bench["name"]
        param_grid = bench.get("params", {})
        for params in _expand_params(param_grid):
            jobs.append({"name": name, "params": params})
    return jobs


def _task_jobs(cfg_path: str) -> List[Dict[str, Any]]:
    cfg = load_yaml(cfg_path)
    jobs = []
    for task in cfg.get("tasks", []):
        name = task["name"]
        param_grid = task.get("params", {})
        for params in _expand_params(param_grid):
            jobs.append({"name": name, "params": params})
    return jobs


def _format_params(params: Dict[str, Any]) -> str:
    return ",".join([f"{k}={v}" for k, v in sorted(params.items())])


def _load_existing(out_dir: str, mode: str):
    existing = set()
    for path in glob.glob(os.path.join(out_dir, f"runs_{mode}_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    bid = r.get("bench_instance_id")
                    mem = r.get("freq_target", {}).get("mem")
                    gpu = r.get("freq_target", {}).get("gpu")
                    if bid is not None and mem is not None and gpu is not None:
                        existing.add((bid, mem, gpu))
                except Exception:
                    continue
    return existing


def run_jobs(mode: str, exe: str, job_cfg: str, config: str, out_dir: str, resume: bool) -> None:
    cfg = load_yaml(config)
    ctrl = DvfsController(cfg["device"]["gpu_index"])
    if cfg.get("device", {}).get("persistence_mode"):
        ctrl.set_persistence_mode(True)
    pairs = ctrl.query_supported_clocks()
    grid_info = build_grid_from_config(cfg, pairs)
    baseline = grid_info["baseline"]
    override = cfg.get("baseline", {}).get("override_mhz", {})
    if override.get("mem") and override.get("gpu"):
        baseline = (override["mem"], override["gpu"])

    grid = grid_info["grid"]
    if override.get("mem") and override.get("gpu"):
        if (override["mem"], override["gpu"]) not in grid:
            grid = grid + [(override["mem"], override["gpu"])]
    jobs = _bench_jobs(job_cfg) if mode == "bench" else _task_jobs(job_cfg)

    ensure_dir(out_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"runs_{mode}_{timestamp}.jsonl")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["device"]["gpu_index"])

    existing = _load_existing(out_dir, mode) if resume else set()

    for job in jobs:
        job_id = stable_id(job["name"], job["params"])
        for mem_mhz, gpu_mhz in grid:
            freq_id = f"mem{mem_mhz}_gpu{gpu_mhz}"
            record_id = f"{job_id}__{freq_id}"
            if resume and (job_id, mem_mhz, gpu_mhz) in existing:
                continue

            # set clocks
            method_ok = False
            method_used = None
            for method in cfg["dvfs"]["lock_method_preference"]:
                if method == "app_clocks":
                    method_ok = ctrl.set_app_clocks(mem_mhz, gpu_mhz)
                elif method == "lock_clocks":
                    method_ok = ctrl.set_lock_clocks(mem_mhz, gpu_mhz)
                elif method == "power_limit_only":
                    # no change, just proceed
                    method_ok = True
                if method_ok:
                    method_used = method
                    break

            if not method_ok:
                print(f"[WARN] failed to set clocks for {record_id}")

            stable = True
            if method_used in ("app_clocks", "lock_clocks"):
                stable = ctrl.wait_stable(
                    mem_mhz,
                    gpu_mhz,
                    cfg["dvfs"]["stable_check"]["tol_mhz"],
                    cfg["dvfs"]["stable_check"]["consecutive"],
                    cfg["dvfs"]["stable_check"]["max_wait_s"],
                    cfg["dvfs"]["stable_check"]["poll_ms"],
                )
                if not stable:
                    print(f"[WARN] clocks not stable for {record_id}")

            probe = cfg.get("dvfs", {}).get("probe_kernel", {})
            if probe.get("enabled"):
                time.sleep(probe.get("duration_ms", 0) / 1000.0)

            actual_mem, actual_gpu = ctrl.get_current_clocks()
            try:
                app_mem, app_gpu = ctrl.get_app_clocks()
            except Exception:
                app_mem, app_gpu = None, None

            warmup = cfg["timing"]["warmup_runs"]
            repeats = cfg["timing"]["repeat_runs"]
            times = []
            cmd = [exe, f"--{mode}", job["name"], "--params", _format_params(job["params"])]

            for _ in range(warmup):
                _run_cmd(cmd, env)

            for _ in range(repeats):
                rc, stdout, stderr = _run_cmd(cmd, env)
                if rc != 0:
                    raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
                t_ms = parse_time_ms_from_stdout(stdout)
                times.append(t_ms)

            p50, p95 = percentiles(times, [50, 95])
            record = {
                "run_type": mode,
                "name": job["name"],
                "params": job["params"],
                "bench_instance_id": job_id,
                "freq_target": {"mem": mem_mhz, "gpu": gpu_mhz},
                "freq_actual": {"mem": actual_mem, "gpu": actual_gpu},
                "freq_app": {"mem": app_mem, "gpu": app_gpu},
                "dvfs_method": method_used,
                "times_ms": times,
                "time_stat": {"median": p50, "p95": p95},
                "baseline": {"mem": baseline[0], "gpu": baseline[1]},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cmd": " ".join(cmd),
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benches or tasks across frequency grid")
    parser.add_argument("mode", choices=["bench", "task"])
    parser.add_argument("--exe", required=True)
    parser.add_argument("--job-config", required=True)
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_jobs(args.mode, args.exe, args.job_config, args.config, args.out_dir, args.resume)


if __name__ == "__main__":
    main()
