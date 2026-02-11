import argparse
import json
import os
from typing import List

import pandas as pd
import xgboost as xgb

from utils import load_yaml


def _feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "bench_instance_id",
        "name",
        "run_type",
        "R_gpu",
        "R_mem",
        "T_default",
        "T_f",
        "S",
        "actual_gpu_clk",
        "actual_mem_clk",
    }
    # include U dims + R_gpu/R_mem
    cols = [c for c in df.columns if c not in exclude]
    # append ratios
    cols = cols + ["R_gpu", "R_mem"]
    # Ensure unique
    return list(dict.fromkeys(cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost performance model")
    parser.add_argument("--config", default="config/h20.yaml")
    parser.add_argument("--train", default="data/datasets/train.parquet")
    parser.add_argument("--val", default="data/datasets/val.parquet")
    parser.add_argument("--out-dir", default="data/models")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    if df_train.empty or df_val.empty:
        print("Training or validation dataset is empty.")
        return

    feat_cols = _feature_columns(df_train)
    print(f"[train] train rows={len(df_train)} val rows={len(df_val)} features={len(feat_cols)}")
    X_train = df_train[feat_cols]
    y_train = df_train["S"]
    X_val = df_val[feat_cols]
    y_val = df_val["S"]

    params = cfg["xgboost"]["params"]
    model_path = os.path.join(args.out_dir, "xgb_perf.json")
    print("[train] training...")
    try:
        model = xgb.XGBRegressor(
            objective=cfg["xgboost"]["objective"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            min_child_weight=params["min_child_weight"],
            eval_metric=cfg["xgboost"]["eval_metric"],
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
        )
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            print(f"[train] best_iteration={best_iter}")
        model.save_model(model_path)
        model_type = "sklearn"
    except TypeError:
        # Fallback for older xgboost sklearn API without early_stopping_rounds
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        train_params = {
            "objective": cfg["xgboost"]["objective"],
            "max_depth": params["max_depth"],
            "eta": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "lambda": params["reg_lambda"],
            "min_child_weight": params["min_child_weight"],
            "eval_metric": cfg["xgboost"]["eval_metric"],
        }
        booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
            verbose_eval=False,
        )
        best_iter = booster.best_iteration if hasattr(booster, "best_iteration") else None
        if best_iter is not None:
            print(f"[train] best_iteration={best_iter}")
        booster.save_model(model_path)
        model_type = "booster"

    meta = {
        "feature_columns": feat_cols,
        "xgboost_params": params,
        "objective": cfg["xgboost"]["objective"],
        "eval_metric": cfg["xgboost"]["eval_metric"],
        "model_type": model_type,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
