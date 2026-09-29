#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_DATASETS = [
    "B0_å‡ºç”ŸåŸºæº–",
    "D0_70æ—¥é½¡åŸºæº–",
    "D1_70æ—¥é½¡æ ¸å¿ƒæ°£è±¡",
    "D2_70æ—¥é½¡å®Œæ•´æ°£è±¡",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-zip", type=Path, default=Path("tabiclv2_inputs.zip"))
    p.add_argument("--work-dir", type=Path, default=Path("tabiclv2_inputs"))
    p.add_argument("--output-dir", type=Path, default=Path("tabiclv2_results"))
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--tasks", nargs="+", default=["classification", "regression"])
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-chronological", action="store_true")
    return p.parse_args()


def ensure_inputs(args):
    if not args.work_dir.exists():
        args.work_dir.mkdir(parents=True)
    manifest_path = args.work_dir / "manifest.json"
    data_path = args.work_dir / "data.npz"
    if not manifest_path.exists() or not data_path.exists():
        with zipfile.ZipFile(args.input_zip) as z:
            z.extractall(args.work_dir)
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(args, manifest, dataset):
    info = manifest["datasets"][dataset]
    features = info["features"]
    categorical = info["categorical"]
    full_num = manifest["full_numeric"]
    full_cat = manifest["full_categorical"]
    num_index = {c: i for i, c in enumerate(full_num)}
    cat_index = {c: i for i, c in enumerate(full_cat)}
    with np.load(args.work_dir / "data.npz", allow_pickle=False) as z:
        xnum = z["Xnum"]
        xcat = z["Xcat"]
        data = {}
        for c in features:
            if c in categorical:
                vals = manifest["category_values"][c]
                codes = xcat[:, cat_index[c]].astype(int)
                data[c] = pd.Series([vals[i] for i in codes], dtype="string")
            else:
                data[c] = xnum[:, num_index[c]].astype(np.float32)
        X = pd.DataFrame(data, columns=features)
        y_cls = pd.Series(z["y_cls"].astype(int))
        y_reg_arr = z["y_reg"].astype(float)
        if manifest.get("anonymized") and "y_reg_mean" in manifest:
            y_reg_arr = y_reg_arr * float(manifest["y_reg_std"]) + float(manifest["y_reg_mean"])
        y_reg = pd.Series(y_reg_arr)
        groups = pd.Series(z["birth_group"].astype(str))
        ids = pd.Series(z["pig_id"].astype(str))
    return X, y_cls, y_reg, groups, ids

def topk_precision(y_true, score):
    k = int(np.sum(y_true == 1))
    if k <= 0:
        return float("nan")
    idx = np.argsort(-score)[:k]
    return float(np.mean(y_true[idx] == 1))


def cls_metrics(y_true, score, pred):
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "ap": float(average_precision_score(y_true, score)),
        "topk_precision": topk_precision(y_true, score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def reg_metrics(y_true, pred):
    rmse = float(math.sqrt(mean_squared_error(y_true, pred)))
    denom = float(np.mean(np.abs(y_true)))
    rho = float(spearmanr(y_true, pred, nan_policy="omit").statistic)
    return {
        "r2": float(r2_score(y_true, pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, pred)),
        "rrmse": float(rmse / denom) if denom else float("nan"),
        "spearman": rho,
    }


def new_classifier(args):
    from tabicl import TabICLClassifier
    return TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        device=args.device,
        use_amp=False,
        use_fa3=False,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        allow_auto_download=True,
        verbose=True,
    )


def new_regressor(args):
    from tabicl import TabICLRegressor
    return TabICLRegressor(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        device=args.device,
        use_amp=False,
        use_fa3=False,
        checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
        allow_auto_download=True,
        verbose=True,
    )


def positive_score(model, X):
    p = np.asarray(model.prdict_proba(X))
    classes = np.asarray(model.classes_)
    pos = int(np.flatnonzero(classes == 1)[0])
    return p[:, pos]


def chrono_split(groups, frac=0.7):
    # birth groups are order-preserving pseudonyms (G0000, G0001, ...).
    g = groups.astype(str)
    unique = np.array(sorted(pd.Series(g.unique()).tolist()))
    cut = max(1, min(len(unique)-1, int(math.floor(len(unique)*frac))))
    train_groups = set(unique[:cut])
    tr = np.flatnonzero(g.isin(train_groups).to_numpy())
    te = np.flatnonzero(~g.isin(train_groups).to_numpy())
    return tr, te


def summary(df, group_cols):
    if df.empty:
        return df
    nums = [c for c in df.select_dtypes(include=[np.number]).columns if c not in group_cols]
    g = df.groupby(group_cols, dropna=False)[nums]
    return g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_inputs(args)
    cls_rows, reg_rows, chrono_cls, chrono_reg = [], [], [], []
    oof_master = None

    for ds in args.datasets:
        print(f"\n===== DATASET {ds} =====", flush=True)
        X, y_cls, y_reg, groups, ids = load_dataset(args, manifest, ds)
        splitter = StratifiedGroupKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
        )
        splits = list(splitter.split(X, y_cls, groups))
        oof = pd.DataFrame({
            "pig_id": ids,
            "birth_group": groups,
            "actual_hot_high25": y_cls,
            "test_adg_calc": y_reg,
        })
        cls_col = f"CLS|{ds}|TabICLv2"
        reg_col = f"REG|{ds}|TabICLv2"
        oof[cls_col] = np.nan
        oof[reg_col] = np.nan

        for i, (tr, te) in enumerate(splits, 1):
            fold = f"fold_{i:02d}"
            print(f"--- {ds} {fold}: train={len(tr)} test={len(te)}", flush=True)
            if "classification" in args.tasks:
                t0 = time.perf_counter()
                m = new_classifier(args)
                m.fit(X.iloc[tr], y_cls.iloc[tr])
                score = positive_score(m, X.iloc[te])
                pred = np.asarray(m.predict(X.iloc[te])).astype(int)
                row = {
                    "dataset": ds, "model": "TabICLv2", "fold": fold,
                    "n_train": len(tr), "n_test": len(te),
                    "positives_test": int(y_cls.iloc[te].sum()),
                    **cls_metrics(y_cls.iloc[te].to_numpy(), score, pred),
                    "seconds": time.perf_counter() - t0,
                }
                cls_rows.append(row)
                oof.loc[te, cls_col] = score
                del m
                gc.collect()
            if "regression" in args.tasks:
                t0 = time.perf_counter()
                m = new_regressor(args)
                m.fit(X.iloc[tr], y_reg.iloc[tr])
                pred = np.asarray(m.predict(X.iloc[te])), dtype=float).reshape(-1)
                row = {
                    "dataset": ds, "model": "TabICLv2", "fold": fold,
                    "n_train": len(tr), "n_test": len(te),
                    **reg_metrics(y_reg.iloc[te].to_numpy(), pred),
                    "seconds": time.perf_counter() - t0,
                }
                reg_rows.append(row)
                oof.loc[te, reg_col] = pred
                del m
                gc.collect()

        if not args.skip_chronological:
            tr, te = chrono_split(groups)
            print(f"--- {ds} chronological: train={len(tr)} test={len(te)}", flush=True)
            if "classification" in args.tasks:
                t0 = time.perf_counter()
                m = new_classifier(args)
                m.fit(X.iloc[tr], y_cls.iloc[tr])
                score = positive_score(m, X.iloc[te])
                pred = np.asarray(m.predict(X.iloc[te])).astype(int)
                chrono_cls.append({
                    "dataset": ds, "model": "TabICLv2", "split": "chronological_70_30",
                    "n_train": len(tr), "n_test": len(te),
                    "positives_test": int(y_cls.iloc[te].sum()),
                    **cls_metrics(y_cls.iloc[te].to_numpy(), score, pred),
                    "seconds": time.perf_counter() - t0,
                })
                del m
                gc.collect()
            if "regression" in args.tasks:
                t0 = time.perf_counter()
                m = new_regressor(args)
                m.fit(X.iloc[tr], y_reg.iloc[tr])
                pred = np.asarray(m.predict(X.iloc[te]), dtype=float).reshape(-1)
                chrono_reg.append({
                    "dataset": ds, "model": "TabICLv2", "split": "chronological_70_30",
                    "n_train": len(tr), "n_test": len(te),
                    **reg_metrics(y_reg.iloc[te].to_numpy(), pred),
                    "seconds": time.perf_counter() - t0,
                })
                del m
                gc.collect()

        if oof_master is None:
            oof_master = oof
        else:
            oof_master = oof_master.merge(
                oof[["pig_id", cls_col, reg_col]], on="pig_id", how="outer"
            )

    cls_df = pd.DataFrame(cls_rows)
    reg_df = pd.DataFrame(reg_rows)
    cls_df.to_csv(args.output_dir / "tabiclv2_classification_fold_results.csv", index=False)
    reg_df.to_csv(args.output_dir / "tabiclv2_regression_fold_results.csv", index=False)
    summary(cls_df, ["dataset", "model"]).to_csv(
        args.output_dir / "tabiclv2_classification_summary.csv", index=False
    )
    summary(reg_df, ["dataset", "model"]).to_csv(
        args.output_dir / "tabiclv2_regression_summary.csv", index=False
    )
    pd.DataFrame(chrono_cls).to_csv(
        args.output_dir / "tabiclv2_chronological_classification.csv", index=False
    )
    pd.DataFrame(chrono_reg).to_csv(
        args.output_dir / "tabiclv2_chronological_regression.csv", index=False
    )
    if oof_master is not None:
        oof_master.to_csv(args.output_dir / "tabiclv2_oof_predictions.csv", index=False)

    import tabicl
    meta = {
        "tabicl_runtime_version": getattr(tabicl, "__version__", "unknown"),
        "classifier_checkpoint": "tabicl-classifier-v2-20260212.ckpt",
        "regressor_checkpoint": "tabicl-regressor-v2-20260212.ckpt",
        "datasets": args.datasets,
        "tasks": args.tasks,
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "n_jobs": args.n_jobs,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "n_common": manifest["n_common"],
    }
    with open(args.output_dir"ò'F&–6Çc%÷'VåöÖWFFFæ§6öâ"Â'r"ÂVæ6öF–æsÒ'WFbÓ‚"’2c ¢§6öâæGV×†ÖWFÂbÂVç7W&Uö66–“ÔfÇ6RÂ–æFVçCÓ"¢&–çB‚$4ôÕÄUDTB"Â&w2æ÷WGWEöF—"ç&W6öÇfR‚’ÂfÇW6ƒÕG'VR ¦–bõöæÖUõòÓÒ%õöÖ–åõò# ¢Ö–â‚ 