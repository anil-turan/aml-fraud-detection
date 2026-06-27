"""Tests for anomaly detection models."""
import numpy as np
import pytest

from src.data.generator import generate_transactions, get_train_test_split
from src.data.preprocessor import preprocess
from src.models.isolation_forest import anomaly_scores_if, train_isolation_forest
from src.models.ensemble import ensemble_scores, find_best_threshold, normalise


@pytest.fixture(scope="module")
def dataset():
    df = generate_transactions(n=2000, fraud_rate=0.05, seed=0)
    X_train, X_test, y_train, y_test = get_train_test_split(df, seed=0)
    X_train_s, X_test_s, pipe = preprocess(X_train, X_test)
    return X_train_s, X_test_s, y_train.values, y_test.values


def test_isolation_forest_trains(dataset):
    X_train, *_ = dataset
    model = train_isolation_forest(X_train, contamination=0.05)
    assert hasattr(model, "decision_function")


def test_anomaly_scores_range(dataset):
    X_train, X_test, *_ = dataset
    model = train_isolation_forest(X_train, contamination=0.05)
    scores = anomaly_scores_if(model, X_test)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert len(scores) == len(X_test)


def test_normalise_bounds():
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = normalise(raw)
    assert abs(normed.min() - 0.0) < 1e-6
    assert abs(normed.max() - 1.0) < 1e-6


def test_ensemble_scores_shape(dataset):
    X_train, X_test, y_train, y_test = dataset
    model = train_isolation_forest(X_train, contamination=0.05)
    ae_scores = anomaly_scores_if(model, X_test)
    xgb_scores = np.random.RandomState(0).rand(len(X_test))
    fused = ensemble_scores(xgb_scores, ae_scores, alpha=0.6)
    assert fused.shape == (len(X_test),)
    assert fused.min() >= 0.0
    assert fused.max() <= 1.0


def test_find_best_threshold(dataset):
    _, X_test, _, y_test = dataset
    scores = np.random.RandomState(1).rand(len(y_test))
    threshold = find_best_threshold(scores, y_test)
    assert 0.1 <= threshold <= 0.9
