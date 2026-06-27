"""
Synthetic transaction data generator.

Generates realistic bank transaction patterns with injected anomalies
covering the main fraud typologies seen in AML/KYC scenarios:
  - Large cash deposits (structuring / smurfing)
  - Round-amount transactions (ML indicator)
  - Rapid sequential transactions (velocity abuse)
  - Geographic anomalies (country-risk mismatches)
  - Off-hours high-value transfers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


FRAUD_RATE = 0.02   # 2% fraud — realistic for retail banking
N_DEFAULT  = 50_000


def generate_transactions(
    n: int = N_DEFAULT,
    fraud_rate: float = FRAUD_RATE,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud  = int(n * fraud_rate)
    n_normal = n - n_fraud

    # ── Normal transactions ──────────────────────────────────────────────────
    normal = pd.DataFrame({
        "amount":           rng.lognormal(mean=4.5, sigma=1.8, size=n_normal).clip(1, 50_000),
        "hour":             rng.integers(7, 22, size=n_normal),
        "day_of_week":      rng.integers(0, 7,  size=n_normal),
        "n_txn_last_1h":    rng.poisson(1.2, size=n_normal).clip(0, 20),
        "n_txn_last_24h":   rng.poisson(4.0, size=n_normal).clip(0, 50),
        "country_risk":     rng.choice([0, 1, 2], p=[0.80, 0.15, 0.05], size=n_normal),
        "is_new_payee":     rng.choice([0, 1],    p=[0.85, 0.15],        size=n_normal),
        "channel":          rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15],  size=n_normal),
        "balance_before":   rng.lognormal(8.5, 1.2, size=n_normal).clip(100, 500_000),
        "is_fraud":         np.zeros(n_normal, dtype=int),
    })
    normal["amount_to_balance"] = normal["amount"] / (normal["balance_before"] + 1)
    normal["is_round_amount"]   = (normal["amount"] % 1000 == 0).astype(int)

    # ── Fraudulent transactions (5 typologies) ───────────────────────────────
    typology = rng.choice(
        ["structuring", "round_amount", "velocity", "geo_risk", "off_hours"],
        size=n_fraud,
    )

    fraud = pd.DataFrame({
        "amount":           np.where(
                                typology == "structuring",
                                rng.uniform(8_000, 9_999, n_fraud),   # just below £10K CTR
                                np.where(
                                    typology == "round_amount",
                                    rng.choice([5000, 10000, 25000, 50000], size=n_fraud),
                                    rng.lognormal(7.0, 1.2, n_fraud).clip(500, 200_000),
                                ),
                            ),
        "hour":             np.where(
                                typology == "off_hours",
                                rng.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud),
                                rng.integers(0, 24, n_fraud),
                            ),
        "day_of_week":      rng.integers(0, 7, n_fraud),
        "n_txn_last_1h":    np.where(
                                typology == "velocity",
                                rng.integers(8, 30, n_fraud),
                                rng.poisson(1.5, n_fraud).clip(0, 10),
                            ),
        "n_txn_last_24h":   np.where(
                                typology == "velocity",
                                rng.integers(25, 80, n_fraud),
                                rng.poisson(5, n_fraud).clip(0, 30),
                            ),
        "country_risk":     np.where(
                                typology == "geo_risk",
                                2,
                                rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15], size=n_fraud),
                            ),
        "is_new_payee":     rng.choice([0, 1], p=[0.4, 0.6], size=n_fraud),
        "channel":          rng.choice([0, 1, 2], p=[0.30, 0.45, 0.25], size=n_fraud),
        "balance_before":   rng.lognormal(7.5, 1.5, n_fraud).clip(100, 200_000),
        "is_fraud":         np.ones(n_fraud, dtype=int),
    })
    fraud["amount_to_balance"] = fraud["amount"] / (fraud["balance_before"] + 1)
    fraud["is_round_amount"]   = (fraud["amount"] % 1000 == 0).astype(int)

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=seed)

    # Derived features
    df["log_amount"]         = np.log1p(df["amount"])
    df["is_off_hours"]       = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["velocity_ratio"]     = df["n_txn_last_1h"] / (df["n_txn_last_24h"] + 1)
    df["high_risk_new_payee"]= (df["country_risk"] == 2) & (df["is_new_payee"] == 1)
    df["high_risk_new_payee"]= df["high_risk_new_payee"].astype(int)

    df = df.reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(len(df))])

    return df


FEATURE_COLS = [
    "log_amount",
    "amount_to_balance",
    "hour",
    "day_of_week",
    "n_txn_last_1h",
    "n_txn_last_24h",
    "velocity_ratio",
    "country_risk",
    "is_new_payee",
    "is_round_amount",
    "is_off_hours",
    "high_risk_new_payee",
    "channel",
]


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLS]
    y = df["is_fraud"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
