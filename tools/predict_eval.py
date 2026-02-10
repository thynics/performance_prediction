import argparse
import json
import os
from typing import Dict

import pandas as pd
import xgboost as xgb

from utils import load_yaml


def mape(y_true, y_pred):
    return (abs(y_true - y_pred) / y_true).replace([float("inf"), float("nan")], 0.0).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on unseen tasks")
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--model", default="data/models/xgb_perf.json")
    parser.add_argument("--meta", default="data/models/meta.json")
    parser.add_argument("--data", default="data/datasets/test_tasks.parquet")
    parser.add_argument("--out-dir", default="data/reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.data)
    if df.empty:
        print("No test data found.")
        return
    print(f"[predict] rows={len(df)}")
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_cols = meta["feature_columns"]
    X = df[feat_cols]

    booster = xgb.Booster()
    booster.load_model(args.model)
    pred_s = booster.predict(xgb.DMatrix(X))

    df = df.copy()
    df["S_hat"] = pred_s
    df["S_hat"] = df["S_hat"].clip(lower=1e-6)
    df["T_hat"] = df["T_default"] / df["S_hat"]

    df["speedup_mape"] = (df["S"] - df["S_hat"]).abs() / df["S"]
    df["time_mape"] = (df["T_f"] - df["T_hat"]).abs() / df["T_f"]

    overall = {
        "speedup_mape": float(df["speedup_mape"].mean()),
        "time_mape": float(df["time_mape"].mean()),
    }

    per_task = df.groupby("name").agg({"speedup_mape": "mean", "time_mape": "mean"}).to_dict("index")

    out_rows = os.path.join(args.out_dir, "predict_eval_rows.csv")
    df.to_csv(out_rows, index=False)

    summary = {
        "overall": overall,
        "per_task": per_task,
    }
    with open(os.path.join(args.out_dir, "predict_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[predict] overall time_mape={overall['time_mape']:.4f} speedup_mape={overall['speedup_mape']:.4f}")
    print(f"[predict] wrote {out_rows}")


if __name__ == "__main__":
    main()
