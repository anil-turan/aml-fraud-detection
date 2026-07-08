"""
Alert-budget precision@k and cost-sensitive threshold selection.

Two questions AUC-PR/F1 don't answer for a real AML team:

  1. Analysts have fixed capacity. If only the top-k highest-scored alerts
     can actually be reviewed today, what fraction of *those* are real fraud?
     That is precision@k — the number a fraud-ops manager staffing a shift
     actually cares about, not the threshold-free AUC-PR.

  2. False positives and false negatives are not equally costly: a false
     alarm wastes analyst time; a missed fraud is a financial loss (and,
     under POCA 2002, a potential regulatory failure). The F1-optimal
     threshold (`ensemble.find_best_threshold`) implicitly weighs FP and FN
     equally — the correct threshold under an asymmetric cost matrix is
     usually different from the F1-optimal one, sometimes considerably so.

Cost model, following Elkan (2001)'s cost-sensitive framework: at a given
threshold, `expected_cost = FP * cost_fp + FN * cost_fn`. True positives and
true negatives are treated as costless baseline outcomes — the choice being
optimised is purely how to trade off the two error types.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-scored alerts — the fraction of an
    analyst's fixed daily review capacity that is actually fraud."""
    if k <= 0:
        return 0.0
    top_k = np.argsort(-scores)[:k]
    return float(y_true[top_k].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of *all* fraud in the period caught within the top-k alerts."""
    total_fraud = y_true.sum()
    if total_fraud == 0 or k <= 0:
        return 0.0
    top_k = np.argsort(-scores)[:k]
    return float(y_true[top_k].sum() / total_fraud)


def precision_recall_at_k_curve(
    y_true: np.ndarray, scores: np.ndarray, k_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep precision@k and recall@k across a range of analyst-capacity
    budgets, e.g. k_values = round(len(y_true) * np.array([0.005, 0.01, ...]))."""
    precisions = np.array([precision_at_k(y_true, scores, k) for k in k_values])
    recalls = np.array([recall_at_k(y_true, scores, k) for k in k_values])
    return precisions, recalls


def alert_budget_threshold(scores: np.ndarray, budget_k: int) -> float:
    """The score threshold implied by a fixed alert budget: the score of the
    budget_k-th highest-scored transaction. Thresholding at this value
    produces (up to score ties) exactly `budget_k` alerts."""
    if budget_k <= 0:
        return float(scores.max()) + 1e-9  # strictly above every score: 0 alerts
    sorted_desc = np.sort(scores)[::-1]
    idx = min(budget_k, len(sorted_desc)) - 1
    return float(sorted_desc[idx])


def expected_cost(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, cost_fp: float, cost_fn: float
) -> float:
    """Total expected cost of operating at `threshold` under the asymmetric
    FP/FN cost matrix. TP and TN are costless (Elkan, 2001)."""
    y_pred = (scores >= threshold).astype(int)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return float(fp * cost_fp + fn * cost_fn)


def cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost at every threshold in `thresholds` (default: a 201-point
    grid over the observed score range)."""
    if thresholds is None:
        thresholds = np.linspace(scores.min(), scores.max(), 201)
    costs = np.array([expected_cost(y_true, scores, t, cost_fp, cost_fn) for t in thresholds])
    return thresholds, costs


def optimal_cost_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid-search the threshold that minimises expected cost. Returns
    (best_threshold, best_expected_cost)."""
    thresholds, costs = cost_curve(y_true, scores, cost_fp, cost_fn, thresholds)
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
