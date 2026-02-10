import argparse
import json
import os
from typing import Dict

import pandas as pd


def _bucket(row: pd.Series) -> str:
    u_sm = row.get("u_sm_sol", 0.0)
    u_dram = row.get("u_dram_sol", 0.0)
    if u_sm > 0.5 and u_sm > u_dram:
        return "compute"
    if u_dram > 0.5 and u_dram > u_sm:
        return "memory"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary report")
    parser.add_argument("--rows", default="data/reports/predict_eval_rows.csv")
    parser.add_argument("--out-dir", default="data/reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.rows)
    if df.empty:
        print("No rows to report.")
        return

    df["freq_distance"] = (df["R_gpu"] - 1.0).abs() + (df["R_mem"] - 1.0).abs()
    df["bucket"] = df.apply(_bucket, axis=1)

    summary = {
        "overall_time_mape": float(df["time_mape"].mean()),
        "overall_speedup_mape": float(df["speedup_mape"].mean()),
        "by_bucket": df.groupby("bucket").agg({"time_mape": "mean", "speedup_mape": "mean"}).to_dict("index"),
        "by_task": df.groupby("name").agg({"time_mape": "mean", "speedup_mape": "mean"}).to_dict("index"),
    }

    with open(os.path.join(args.out_dir, "report_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional plots
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        df.plot.scatter(x="freq_distance", y="time_mape")
        plt.title("Time MAPE vs Frequency Distance")
        plt.savefig(os.path.join(args.out_dir, "mape_vs_freq_distance.png"))
        plt.close()

        plt.figure()
        df.boxplot(column="time_mape", by="bucket")
        plt.title("Time MAPE by Bucket")
        plt.suptitle("")
        plt.savefig(os.path.join(args.out_dir, "mape_by_bucket.png"))
        plt.close()
    except Exception:
        pass

    print("Report written to data/reports/")


if __name__ == "__main__":
    main()
