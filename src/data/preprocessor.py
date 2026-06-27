"""
Feature preprocessing for anomaly detection models.

Provides a sklearn-compatible pipeline that:
  - Scales numeric features (StandardScaler)
  - Handles any remaining missing values (median imputation)
  - Returns numpy arrays suitable for both sklearn and PyTorch models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.generator import FEATURE_COLS


def build_preprocessor() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    pipe = build_preprocessor()
    X_train_scaled = pipe.fit_transform(X_train[FEATURE_COLS])
    X_test_scaled  = pipe.transform(X_test[FEATURE_COLS])
    return X_train_scaled, X_test_scaled, pipe
