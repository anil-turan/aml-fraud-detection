"""
Transaction score drift monitoring using Evidently AI.

Monitors two types of drift in production:
  1. Feature drift — transaction characteristics shifting over time
     (e.g., new fraud pattern causing amount distribution to change)
  2. Score drift — model output distribution shifting
     (indicates concept drift; may require model retraining)

In production, this runs nightly comparing yesterday's transactions
against the reference distribution from the model training window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    score_col: str = "anomaly_score",
    feature_cols: Optional[list] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compute statistical drift between reference and current windows.
    Uses KS test for continuous features, chi-squared for categoricals.
    Returns a summary dict — use Evidently HTML report for full detail.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently import ColumnMapping

        cols = feature_cols or [c for c in reference_df.columns
                                if c not in ("transaction_id", "is_fraud", score_col)]
        ref = reference_df[cols + ([score_col] if score_col in reference_df.columns else [])]
        cur = current_df[cols  + ([score_col] if score_col in current_df.columns else [])]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        if save_path:
            report.save_html(save_path)

        result = report.as_dict()
        drift_metrics = result["metrics"][0]["result"]
        return {
            "dataset_drift":    drift_metrics.get("dataset_drift", False),
            "drifted_features": drift_metrics.get("number_of_drifted_columns", 0),
            "share_drifted":    drift_metrics.get("share_of_drifted_columns", 0.0),
        }

    except ImportError:
        return _fallback_ks_drift(reference_df, current_df, feature_cols)


def _fallback_ks_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> dict:
    """KS-test drift detection when Evidently is not installed."""
    from scipy import stats

    cols = feature_cols or [
        c for c in reference_df.columns
        if c not in ("transaction_id", "is_fraud") and reference_df[c].dtype in [np.float64, np.int64]
    ]
    drifted = []
    for col in cols:
        if col not in current_df.columns:
            continue
        stat, p = stats.ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
        if p < 0.05:
            drifted.append(col)

    return {
        "dataset_drift":    len(drifted) > len(cols) * 0.3,
        "drifted_features": len(drifted),
        "share_drifted":    len(drifted) / max(len(cols), 1),
        "drifted_columns":  drifted,
    }


def score_drift_alert(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """KS test on model score distributions to detect concept drift."""
    from scipy import stats

    stat, p = stats.ks_2samp(reference_scores, current_scores)
    mean_shift = float(current_scores.mean() - reference_scores.mean())

    return {
        "ks_statistic": round(float(stat), 4),
        "p_value":      round(float(p), 4),
        "drift_detected": p < alpha,
        "mean_score_shift": round(mean_shift, 4),
        "alert": abs(mean_shift) > 0.05,  # >5% absolute shift triggers retraining review
    }
