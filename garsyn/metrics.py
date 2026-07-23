import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


METRICS = ["mse", "rmse", "r2", "pearson", "spearman"]


def _safe_corr(fn, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    try:
        return float(fn(y_true, y_pred)[0])
    except Exception:
        return 0.0


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": _safe_corr(pearsonr, y_true, y_pred),
        "spearman": _safe_corr(spearmanr, y_true, y_pred),
    }


def evaluate_prediction_frame(df, labels):
    rows = []
    for label in labels:
        pred_col = f"pred_{label}"
        if label in df.columns and pred_col in df.columns:
            rows.append({"label": label, **regression_metrics(df[label], df[pred_col])})
    return pd.DataFrame(rows)


def summarize(metric_df):
    return metric_df.groupby(["method", "label"])[METRICS].agg(["mean", "std"]).reset_index()
