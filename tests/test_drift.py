"""Tests for score/feature drift monitoring -- checks the KS-test fallback
path (Evidently is optional; these tests don't depend on it being
installed) correctly stays quiet on matched distributions and fires on a
genuine shift."""

import numpy as np
import pandas as pd

from src.monitoring.drift import _fallback_ks_drift, run_drift_report, score_drift_alert


def test_score_drift_alert_quiet_for_identical_distributions():
    rng = np.random.default_rng(0)
    scores = rng.beta(2, 20, 2000)
    result = score_drift_alert(scores, scores)
    assert result["drift_detected"] == False  # noqa: E712 (numpy.bool_, not python bool)
    assert result["alert"] == False  # noqa: E712
    assert result["ks_statistic"] == 0.0


def test_score_drift_alert_fires_on_a_real_mean_shift():
    rng = np.random.default_rng(1)
    reference = rng.beta(2, 20, 2000)  # mean ~ 0.09
    current = rng.beta(2, 8, 2000)  # mean ~ 0.20, clearly shifted higher
    result = score_drift_alert(reference, current)
    assert result["drift_detected"] == True  # noqa: E712 (numpy.bool_, not python bool)
    assert result["alert"] == True  # noqa: E712
    assert result["mean_score_shift"] > 0.05


def test_score_drift_alert_no_alert_for_small_shift_below_threshold():
    rng = np.random.default_rng(2)
    reference = rng.normal(0.5, 0.01, 2000)
    current = rng.normal(0.505, 0.01, 2000)  # tiny shift, well under 5%
    result = score_drift_alert(reference, current)
    assert result["alert"] is False


def _feature_frame(rng, n, amount_scale=1.0):
    return pd.DataFrame({
        "transaction_id": np.arange(n),
        "is_fraud": rng.integers(0, 2, n),
        "amount": rng.lognormal(4.5, 1.0, n) * amount_scale,
        "hour": rng.integers(0, 24, n),
    })


def test_fallback_ks_drift_finds_nothing_between_matched_samples():
    rng = np.random.default_rng(3)
    reference = _feature_frame(rng, 3000)
    current = _feature_frame(rng, 3000)
    result = _fallback_ks_drift(reference, current, feature_cols=["amount", "hour"])
    assert result["drifted_features"] == 0
    assert result["dataset_drift"] is False


def test_fallback_ks_drift_detects_a_genuine_amount_shift():
    rng = np.random.default_rng(4)
    reference = _feature_frame(rng, 3000, amount_scale=1.0)
    current = _feature_frame(rng, 3000, amount_scale=3.0)  # amounts 3x larger
    result = _fallback_ks_drift(reference, current, feature_cols=["amount", "hour"])
    assert "amount" in result["drifted_columns"]
    assert result["drifted_features"] >= 1


def test_run_drift_report_returns_expected_keys():
    """Uses whichever path is available (Evidently if installed, else the
    KS fallback) -- both must return the same summary shape."""
    rng = np.random.default_rng(5)
    reference = _feature_frame(rng, 2000)
    current = _feature_frame(rng, 2000, amount_scale=2.0)
    result = run_drift_report(reference, current, score_col="is_fraud", feature_cols=["amount", "hour"])
    assert "dataset_drift" in result
    assert "share_drifted" in result
