import argparse
import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from utils import load_yaml


class DvfsController:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.nvml = None
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self.nvml = None
            self.handle = None

    def _run(self, cmd: List[str]) -> str:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
        return proc.stdout

    def get_current_clocks(self) -> Tuple[int, int]:
        if self.nvml is not None:
            mem = self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_MEM)
            gfx = self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_GRAPHICS)
            return int(mem), int(gfx)
        out = self._run([
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=clocks.mem,clocks.gr",
            "--format=csv,noheader,nounits",
        ])
        parts = out.strip().split(",")
        mem = int(parts[0].strip())
        gfx = int(parts[1].strip())
        return mem, gfx

    def query_supported_clocks(self) -> List[Tuple[int, int]]:
        if self.nvml is not None:
            mem_clocks = self.nvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
            pairs = []
            for mem in mem_clocks:
                gfxs = self.nvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, mem)
                for gfx in gfxs:
                    pairs.append((int(mem), int(gfx)))
            return pairs
        # Try query-supported-clocks first (newer nvidia-smi format)
        pairs: List[Tuple[int, int]] = []
        try:
            out_query = self._run([
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "--query-supported-clocks=mem,gr",
                "--format=csv,noheader,nounits",
            ])
            for line in out_query.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    pairs.append((int(parts[0]), int(parts[1])))
        except Exception:
            out_query = ""

        if pairs:
            return pairs

        # Fallback to SUPPORTED_CLOCKS section
        out = self._run([
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "-q",
            "-d",
            "SUPPORTED_CLOCKS",
        ])
        mem = None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Memory") and "MHz" in line:
                m = re.search(r"(\\d+)\\s*MHz", line)
                mem = int(m.group(1)) if m else None
            elif line.startswith("Graphics") and "MHz" in line and mem is not None:
                m = re.search(r"(\\d+)\\s*MHz", line)
                if m:
                    gfx = int(m.group(1))
                    pairs.append((mem, gfx))
        if not pairs:
            snippet = out[:800] if out else out_query[:800]
            raise RuntimeError(f"Unable to parse supported clocks from nvidia-smi output. Snippet:\\n{snippet}")
        return pairs

    def set_app_clocks(self, mem_mhz: int, gpu_mhz: int) -> bool:
        if self.nvml is not None:
            try:
                self.nvml.nvmlDeviceSetApplicationsClocks(self.handle, mem_mhz, gpu_mhz)
                return True
            except Exception:
                return False
        try:
            self._run([
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "-ac",
                f"{mem_mhz},{gpu_mhz}",
            ])
            return True
        except Exception:
            return False

    def set_lock_clocks(self, mem_mhz: Optional[int], gpu_mhz: Optional[int]) -> bool:
        ok = True
        try:
            if gpu_mhz is not None:
                self._run([
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "-lgc",
                    f"{gpu_mhz},{gpu_mhz}",
                ])
            if mem_mhz is not None:
                self._run([
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "-lmc",
                    f"{mem_mhz},{mem_mhz}",
                ])
        except Exception:
            ok = False
        return ok

    def set_power_limit(self, watts: int) -> bool:
        try:
            self._run([
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "-pl",
                str(watts),
            ])
            return True
        except Exception:
            return False

    def set_persistence_mode(self, enabled: bool) -> bool:
        try:
            self._run([
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "-pm",
                "1" if enabled else "0",
            ])
            return True
        except Exception:
            return False

    def wait_stable(self, target_mem: int, target_gpu: int, tol_mhz: int, consecutive: int, max_wait_s: int, poll_ms: int = 50) -> bool:
        hit = 0
        start = time.time()
        while time.time() - start < max_wait_s:
            mem, gpu = self.get_current_clocks()
            if abs(mem - target_mem) <= tol_mhz and abs(gpu - target_gpu) <= tol_mhz:
                hit += 1
                if hit >= consecutive:
                    return True
            else:
                hit = 0
            time.sleep(poll_ms / 1000.0)
        return False


def _select_baseline(pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
    pairs_sorted = sorted(pairs, key=lambda p: (p[0], p[1]))
    max_mem = max(p[0] for p in pairs_sorted)
    max_gpu_at_mem = max(p[1] for p in pairs_sorted if p[0] == max_mem)
    return max_mem, max_gpu_at_mem


def _choose_levels(values: List[int], percents: List[float]) -> List[int]:
    values_sorted = sorted(set(values))
    vmax = max(values_sorted)
    out = []
    for pct in percents:
        target = vmax * pct
        nearest = min(values_sorted, key=lambda v: abs(v - target))
        out.append(nearest)
    return sorted(set(out), reverse=True)


def _uniform_select(values: List[int], count: int) -> List[int]:
    values_sorted = sorted(set(values))
    if count <= 1:
        return [values_sorted[len(values_sorted) // 2]]
    if count >= len(values_sorted):
        return values_sorted
    idxs = [int(round(i * (len(values_sorted) - 1) / (count - 1))) for i in range(count)]
    sel = [values_sorted[i] for i in idxs]
    return sorted(set(sel))


def build_grid_from_config(cfg: Dict[str, object], pairs: List[Tuple[int, int]]) -> Dict[str, object]:
    grid_cfg = cfg["freq_grid"]
    mem_levels_cfg = grid_cfg["mem_levels"]
    gpu_levels_cfg = grid_cfg["gpu_levels_per_mem"]

    mem_values = [p[0] for p in pairs]
    gpu_map = {}
    for mem, gpu in pairs:
        gpu_map.setdefault(mem, []).append(gpu)

    if mem_levels_cfg["policy"] == "percent_of_max":
        mem_levels = _choose_levels(mem_values, mem_levels_cfg["percents"])
    else:
        mem_levels = sorted(set(mem_values), reverse=True)

    grid = []
    for mem in mem_levels:
        gpus = gpu_map.get(mem, [])
        if not gpus:
            continue
        if gpu_levels_cfg["policy"] == "uniform_between_min_max":
            gpu_sel = _uniform_select(gpus, gpu_levels_cfg["count"])
        else:
            gpu_sel = sorted(set(gpus))
        for gpu in gpu_sel:
            grid.append((mem, gpu))

    if grid_cfg.get("include_baseline", True):
        baseline = _select_baseline(pairs)
        if baseline not in grid:
            grid.append(baseline)

    max_pairs = grid_cfg.get("max_pairs")
    if max_pairs and len(grid) > max_pairs:
        grid = grid[:max_pairs]

    return {"grid": sorted(grid), "baseline": _select_baseline(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="DVFS utilities")
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--print-grid", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ctrl = DvfsController(cfg["device"]["gpu_index"])
    pairs = ctrl.query_supported_clocks()
    grid_info = build_grid_from_config(cfg, pairs)

    baseline_override = cfg.get("baseline", {}).get("override_mhz", {})
    if baseline_override and baseline_override.get("mem") and baseline_override.get("gpu"):
        grid_info["baseline"] = (baseline_override["mem"], baseline_override["gpu"])

    if args.print_grid:
        print(json.dumps({
            "baseline": {"mem": grid_info["baseline"][0], "gpu": grid_info["baseline"][1]},
            "grid": [{"mem": m, "gpu": g} for m, g in grid_info["grid"]],
        }, indent=2))


if __name__ == "__main__":
    main()
