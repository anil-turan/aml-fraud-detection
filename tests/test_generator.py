"""Tests for synthetic transaction data generator."""
import numpy as np
import pandas as pd
import pytest

from src.data.generator import FEATURE_COLS, generate_transactions, get_train_test_split


def test_generate_basic_shape():
    df = generate_transactions(n=1000, fraud_rate=0.02)
    assert len(df) == 1000
    assert "is_fraud" in df.columns
    assert "transaction_id" in df.columns


def test_fraud_rate():
    df = generate_transactions(n=10_000, fraud_rate=0.02)
    actual_rate = df["is_fraud"].mean()
    assert abs(actual_rate - 0.02) < 0.005


def test_feature_cols_present():
    df = generate_transactions(n=500)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"


def test_no_nulls():
    df = generate_transactions(n=1000)
    assert df[FEATURE_COLS].isnull().sum().sum() == 0


def test_log_amount_positive():
    df = generate_transactions(n=500)
    assert (df["log_amount"] >= 0).all()


def test_train_test_split_stratified():
    df = generate_transactions(n=5000, fraud_rate=0.02)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    train_rate = y_train.mean()
    test_rate  = y_test.mean()
    assert abs(train_rate - test_rate) < 0.01


def test_unique_transaction_ids():
    df = generate_transactions(n=1000)
    assert df["transaction_id"].nunique() == len(df)
