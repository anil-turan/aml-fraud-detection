"""
Ensemble: Supervised (XGBoost) + Unsupervised (Autoencoder) score fusion.

Strategy:
  1. Train XGBoost on labelled data → probability score P(fraud)
  2. Train Autoencoder on all data (unsupervised) → reconstruction error score
  3. Fuse with weighted average: ensemble_score = α·xgb_score + (1-α)·ae_score
  4. Final decision at threshold τ chosen to maximise F1 on validation set

Why ensemble > single model:
  - XGBoost captures known fraud patterns in labelled data
  - Autoencoder catches novel/zero-day fraud patterns with no labels
  - Fusion reduces false negatives from each model's blind spots
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float | None = None,
    seed: int = 42,
) -> xgb.XGBClassifier:
    if scale_pos_weight is None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def normalise(scores: np.ndarray) -> np.ndarray:
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        return (scores - min_s) / (max_s - min_s)
    return scores


def ensemble_scores(
    xgb_proba: np.ndarray,
    ae_errors: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Weighted fusion: alpha weight on supervised XGBoost score."""
    xgb_norm = normalise(xgb_proba)
    ae_norm   = normalise(ae_errors)
    return alpha * xgb_norm + (1 - alpha) * ae_norm


def find_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Grid-search threshold that maximises F1 on validation set."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh
