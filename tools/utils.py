import json
import os
import re
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required. Please pip install -r requirements.txt")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_id(name: str, params: Dict[str, Any]) -> str:
    items = sorted(params.items(), key=lambda kv: kv[0])
    parts = [name]
    for k, v in items:
        sval = str(v).replace("/", "-")
        parts.append(f"{k}={sval}")
    return "__".join(parts)


def parse_params_str(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    params = {}
    for seg in s.split(","):
        if not seg:
            continue
        if "=" not in seg:
            raise ValueError(f"Bad param segment: {seg}")
        k, v = seg.split("=", 1)
        k = k.strip()
        v = v.strip()
        if re.fullmatch(r"-?\d+", v):
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                pass
        params[k] = v
    return params


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_time_ms_from_stdout(stdout: str) -> float:
    stdout = stdout.strip()
    if not stdout:
        raise ValueError("Empty stdout from benchmark")
    # Accept JSON like {"time_ms": 1.23}
    if stdout.startswith("{"):
        obj = json.loads(stdout)
        if "time_ms" in obj:
            return float(obj["time_ms"])
    # Accept lines like TIME_MS=1.23
    for line in stdout.splitlines():
        if line.startswith("TIME_MS="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"Unable to parse time_ms from stdout: {stdout[:200]}")


def percentiles(values: List[float], p: List[float]) -> List[float]:
    try:
        import numpy as np
    except ImportError:
        values_sorted = sorted(values)
        out = []
        for pct in p:
            if not values_sorted:
                out.append(float("nan"))
                continue
            k = (len(values_sorted) - 1) * (pct / 100.0)
            f = int(k)
            c = min(f + 1, len(values_sorted) - 1)
            if f == c:
                out.append(values_sorted[int(k)])
            else:
                d0 = values_sorted[f] * (c - k)
                d1 = values_sorted[c] * (k - f)
                out.append(d0 + d1)
        return out
    arr = np.array(values, dtype=float)
    return [float(np.percentile(arr, pct)) for pct in p]
