"""
Isolation Forest baseline for unsupervised anomaly detection.

Isolation Forest isolates anomalies by randomly partitioning the feature space.
Anomalies require fewer splits to isolate → shorter path length → lower score.

Why it works for transactions:
  - No labels required — trains on all transactions
  - Handles high-dimensional, mixed-scale data well
  - Decision function gives a continuous anomaly score for ranking
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.02,
    n_estimators: int = 200,
    seed: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return normalised anomaly scores in [0, 1] — higher = more anomalous."""
    raw = model.decision_function(X)
    # decision_function returns negative scores for anomalies; invert and normalise
    scores = -raw
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        return (scores - min_s) / (max_s - min_s)
    return scores


def train_lof(
    X_train: np.ndarray,
    contamination: float = 0.02,
    n_neighbors: int = 20,
) -> LocalOutlierFactor:
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores_lof(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    raw = model.decision_function(X)
    scores = -raw
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        return (scores - min_s) / (max_s - min_s)
    return scores
