"""Tests for alert-budget precision@k and cost-sensitive threshold selection."""

import numpy as np

from src.evaluation.alert_budget import (
    alert_budget_threshold,
    cost_curve,
    expected_cost,
    optimal_cost_threshold,
    precision_at_k,
    precision_recall_at_k_curve,
    recall_at_k,
)


def _perfectly_ranked_world(n=1000, fraud_rate=0.05, seed=0):
    """Scores rank every fraud case above every normal case — the easy case
    where precision@k = 1.0 for any k <= n_fraud."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    y_true = np.zeros(n, dtype=int)
    y_true[:n_fraud] = 1
    rng.shuffle(y_true)
    # fraud gets a score in (0.5, 1.0), normal gets a score in (0.0, 0.5)
    scores = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    return y_true, scores


def test_precision_at_k_is_perfect_when_ranking_is_perfect():
    y_true, scores = _perfectly_ranked_world(n=1000, fraud_rate=0.05)
    n_fraud = y_true.sum()
    assert precision_at_k(y_true, scores, k=n_fraud) == 1.0
    assert precision_at_k(y_true, scores, k=n_fraud // 2) == 1.0


def test_precision_at_k_matches_base_rate_for_uninformative_scores():
    rng = np.random.default_rng(1)
    n, fraud_rate = 20_000, 0.05
    y_true = (rng.random(n) < fraud_rate).astype(int)
    scores = rng.random(n)  # uninformative: independent of y_true
    p = precision_at_k(y_true, scores, k=2000)
    assert abs(p - fraud_rate) < 0.02


def test_recall_at_k_reaches_one_when_k_covers_all_fraud():
    y_true, scores = _perfectly_ranked_world(n=500, fraud_rate=0.1)
    assert recall_at_k(y_true, scores, k=len(y_true)) == 1.0
    assert recall_at_k(y_true, scores, k=0) == 0.0


def test_precision_recall_at_k_curve_shapes():
    y_true, scores = _perfectly_ranked_world()
    k_values = np.array([10, 50, 100])
    precisions, recalls = precision_recall_at_k_curve(y_true, scores, k_values)
    assert precisions.shape == recalls.shape == k_values.shape
    # recall is monotonically non-decreasing in k
    assert np.all(np.diff(recalls) >= 0)


def test_alert_budget_threshold_produces_requested_alert_count():
    rng = np.random.default_rng(2)
    scores = rng.random(5000)
    budget = 100
    threshold = alert_budget_threshold(scores, budget_k=budget)
    n_alerts = int(np.sum(scores >= threshold))
    assert n_alerts >= budget
    assert n_alerts < budget + 5  # ties aside, should be tight


def test_alert_budget_threshold_zero_budget_yields_no_alerts():
    scores = np.array([0.1, 0.5, 0.9])
    threshold = alert_budget_threshold(scores, budget_k=0)
    assert np.sum(scores >= threshold) == 0


def test_expected_cost_zero_for_perfect_classifier():
    y_true, scores = _perfectly_ranked_world(n=500, fraud_rate=0.1)
    cost = expected_cost(y_true, scores, threshold=0.5, cost_fp=100.0, cost_fn=8_500.0)
    assert cost == 0.0


def test_cost_curve_matches_expected_cost_pointwise():
    y_true, scores = _perfectly_ranked_world(n=300, fraud_rate=0.1, seed=3)
    thresholds = np.array([0.2, 0.5, 0.8])
    grid_thresholds, costs = cost_curve(y_true, scores, cost_fp=50.0, cost_fn=8_500.0,
                                        thresholds=thresholds)
    for t, c in zip(grid_thresholds, costs):
        assert c == expected_cost(y_true, scores, t, cost_fp=50.0, cost_fn=8_500.0)


def test_optimal_threshold_shifts_lower_when_missed_fraud_is_costlier():
    """When cost_fn >> cost_fp, the cost-minimising threshold should be lower
    (catch more fraud, tolerate more false alarms) than when the two costs
    are comparable."""
    rng = np.random.default_rng(4)
    n = 5000
    y_true = (rng.random(n) < 0.05).astype(int)
    # noisy but informative scores
    scores = np.clip(y_true * 0.5 + rng.normal(0.3, 0.25, n), 0, 1)

    thresh_symmetric, _ = optimal_cost_threshold(y_true, scores, cost_fp=100.0, cost_fn=100.0)
    thresh_asymmetric, _ = optimal_cost_threshold(y_true, scores, cost_fp=50.0, cost_fn=8_500.0)
    assert thresh_asymmetric <= thresh_symmetric


def test_optimal_cost_threshold_beats_extreme_thresholds():
    rng = np.random.default_rng(5)
    n = 5000
    y_true = (rng.random(n) < 0.05).astype(int)
    scores = np.clip(y_true * 0.5 + rng.normal(0.3, 0.25, n), 0, 1)
    cost_fp, cost_fn = 50.0, 8_500.0

    best_thresh, best_cost = optimal_cost_threshold(y_true, scores, cost_fp, cost_fn)
    cost_alert_none = expected_cost(y_true, scores, threshold=1.01, cost_fp=cost_fp, cost_fn=cost_fn)
    cost_alert_all = expected_cost(y_true, scores, threshold=-0.01, cost_fp=cost_fp, cost_fn=cost_fn)
    assert best_cost <= cost_alert_none
    assert best_cost <= cost_alert_all
