import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd
from io import StringIO

from utils import ensure_dir, load_yaml


def _parse_metric_value(val: Any) -> float:
    if val is None:
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ("", "nan", "N/A", "n/a"):
        return float("nan")
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _load_kernel_metrics(csv_path: str) -> Dict[str, Dict[str, float]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip() and not ln.startswith("==")]
    df = pd.read_csv(StringIO("".join(lines)))
    # Try to locate columns
    metric_col = None
    value_col = None
    kernel_col = None
    for col in df.columns:
        if col.lower() in ("metric name", "metric", "name"):
            metric_col = col
        if col.lower() in ("metric value", "value"):
            value_col = col
        if col.lower() in ("kernel name", "kernel"):
            kernel_col = col
    if metric_col is None or value_col is None or kernel_col is None:
        # Fallback to known NCU csv headers
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


def _apply_transform(val: float, transform: Dict[str, Any]) -> float:
    if transform is None:
        return val
    if transform.get("type") == "scale":
        val = val * float(transform.get("scale", 1.0))
    if "clamp" in transform:
        lo, hi = transform["clamp"]
        if val < lo:
            val = lo
        if val > hi:
            val = hi
    return val


def compute_u_vector(cfg: Dict[str, Any], kernel_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    u_dims = cfg["u_vector"]["dims"]
    transforms = cfg.get("transforms", {})
    weight_source = cfg["aggregation"]["weight_source"]
    missing_policy = cfg.get("missing_metric_policy", "drop_or_zero")

    # Build per-kernel vectors
    per_kernel = []
    for kernel, metrics in kernel_metrics.items():
        weights = metrics.get(weight_source, 1.0)
        vec = []
        ok = True
        for dim in u_dims:
            if "source" in dim:
                val = metrics.get(dim["source"])
            else:
                vals = [metrics.get(s) for s in dim.get("sources", [])]
                vals = [v for v in vals if v is not None]
                if not vals:
                    val = None
                elif dim.get("reduce") == "max":
                    val = max(vals)
                else:
                    val = vals[0]
            if val is None:
                if missing_policy in ("drop", "drop_or_zero"):
                    ok = False
                val = 0.0
            transform = transforms.get(dim.get("transform"))
            val = _apply_transform(float(val), transform)
            vec.append(val)
        if ok or missing_policy == "drop_or_zero":
            per_kernel.append((weights, vec))

    if not per_kernel:
        raise RuntimeError("No kernel metrics found to build U vector")

    # time-weighted mean
    total_w = sum(w for w, _ in per_kernel)
    if total_w == 0:
        total_w = 1.0
    agg = [0.0 for _ in range(len(u_dims))]
    for w, vec in per_kernel:
        for i, v in enumerate(vec):
            agg[i] += (w * v)
    agg = [v / total_w for v in agg]
    return {"dims": [d["name"] for d in u_dims], "values": agg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build U vectors from NCU CSV")
    parser.add_argument("--mode", choices=["bench", "task"], required=True)
    parser.add_argument("--ncu-dir", default="data/ncu")
    parser.add_argument("--out-dir", default="data/ncu/features")
    parser.add_argument("--metrics-config", default="config/ncu_metrics.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.metrics_config)
    ensure_dir(args.out_dir)

    pattern = os.path.join(args.ncu_dir, f"{args.mode}_*.csv")
    for csv_path in glob.glob(pattern):
        base = os.path.basename(csv_path)
        job_id = base.replace(f"{args.mode}_", "").replace(".csv", "")
        kernel_metrics = _load_kernel_metrics(csv_path)
        uvec = compute_u_vector(cfg, kernel_metrics)
        out_path = os.path.join(args.out_dir, f"{args.mode}_{job_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "bench_instance_id": job_id,
                "u_dims": uvec["dims"],
                "u_values": uvec["values"],
            }, f, indent=2)
        print(f"[U] {out_path}")


if __name__ == "__main__":
    main()
