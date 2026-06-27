"""Tests for SAR workflow — regulatory alert triage."""
import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_transactions
from src.sar.workflow import (
    AlertPriority,
    generate_alerts,
    generate_sar_records,
    sar_summary,
)


@pytest.fixture
def sample_df():
    return generate_transactions(n=500, fraud_rate=0.10, seed=7)


@pytest.fixture
def high_scores():
    return np.ones(500) * 0.9


@pytest.fixture
def low_scores():
    return np.zeros(500) * 0.1


def test_generate_alerts_returns_list(sample_df, high_scores):
    alerts = generate_alerts(sample_df, high_scores, threshold=0.30)
    assert isinstance(alerts, list)
    assert len(alerts) > 0


def test_high_scores_create_high_priority_alerts(sample_df, high_scores):
    alerts = generate_alerts(sample_df, high_scores, threshold=0.30)
    for alert in alerts:
        assert alert.priority == AlertPriority.HIGH


def test_low_scores_create_no_alerts(sample_df, low_scores):
    alerts = generate_alerts(sample_df, low_scores, threshold=0.30)
    assert len(alerts) == 0


def test_alert_has_required_fields(sample_df, high_scores):
    alerts = generate_alerts(sample_df, high_scores, threshold=0.30)
    alert = alerts[0]
    assert alert.alert_id.startswith("ALT-")
    assert 0.0 <= alert.anomaly_score <= 1.0
    assert len(alert.reasons) > 0


def test_sar_records_generated_for_high_priority(sample_df, high_scores):
    alerts = generate_alerts(sample_df, high_scores, threshold=0.30)
    records = generate_sar_records(alerts, sample_df)
    assert len(records) > 0
    for record in records:
        assert record.sar_id.startswith("SAR-")
        assert record.priority == "HIGH"


def test_sar_summary_keys(sample_df, high_scores):
    alerts = generate_alerts(sample_df, high_scores, threshold=0.30)
    summary = sar_summary(alerts)
    for key in ("total_alerts", "high_priority", "medium_priority", "sar_required"):
        assert key in summary


def test_mixed_scores_triage(sample_df):
    rng = np.random.default_rng(42)
    scores = rng.uniform(0.0, 1.0, size=len(sample_df))
    alerts = generate_alerts(sample_df, scores, threshold=0.30)
    summary = sar_summary(alerts)
    assert summary["total_alerts"] == len(alerts)
    assert summary["high_priority"] + summary["medium_priority"] == summary["sar_required"]
