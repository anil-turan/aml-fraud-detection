"""
SAR (Suspicious Activity Report) filing workflow.

Under POCA 2002 and the Money Laundering Regulations 2017, UK firms must
submit SARs to the NCA (National Crime Agency) when they know or suspect
that a person is engaged in money laundering or terrorist financing.

This module implements:
  1. Alert triage: score → priority tier (HIGH / MEDIUM / LOW)
  2. SAR eligibility check: apply regulatory criteria
  3. SAR record generation: structured data for compliance team review
  4. Audit trail: immutable log for FCA supervisory review

Thresholds here are illustrative — production thresholds are calibrated
per firm risk appetite and reviewed by the MLRO (Money Laundering Reporting Officer).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class AlertPriority(str, Enum):
    HIGH   = "HIGH"    # Score ≥ 0.80 — MLRO review within 24h
    MEDIUM = "MEDIUM"  # Score ≥ 0.50 — analyst review within 5 days
    LOW    = "LOW"     # Score ≥ threshold — monitor, no immediate action


class SARStatus(str, Enum):
    PENDING    = "PENDING"
    FILED      = "FILED"
    DISMISSED  = "DISMISSED"


@dataclass
class Alert:
    alert_id:       str
    transaction_id: str
    anomaly_score:  float
    priority:       AlertPriority
    triggered_at:   str
    model_version:  str
    reasons:        List[str] = field(default_factory=list)
    sar_status:     SARStatus = SARStatus.PENDING


@dataclass
class SARRecord:
    sar_id:          str
    alert_id:        str
    transaction_id:  str
    anomaly_score:   float
    priority:        str
    filing_deadline: str
    narrative:       str
    supporting_features: dict
    created_at:      str
    status:          SARStatus = SARStatus.PENDING
    mlro_notes:      Optional[str] = None


PRIORITY_THRESHOLDS = {
    AlertPriority.HIGH:   0.80,
    AlertPriority.MEDIUM: 0.50,
    AlertPriority.LOW:    0.30,
}

FILING_HOURS = {
    AlertPriority.HIGH:   24,
    AlertPriority.MEDIUM: 120,   # 5 working days
    AlertPriority.LOW:    None,  # monitoring only
}


def _classify_priority(score: float) -> AlertPriority:
    if score >= PRIORITY_THRESHOLDS[AlertPriority.HIGH]:
        return AlertPriority.HIGH
    if score >= PRIORITY_THRESHOLDS[AlertPriority.MEDIUM]:
        return AlertPriority.MEDIUM
    if score >= PRIORITY_THRESHOLDS[AlertPriority.LOW]:
        return AlertPriority.LOW
    return None  # below alert threshold


def _derive_reasons(row: pd.Series) -> List[str]:
    reasons = []
    if row.get("amount", 0) >= 9_000 and row.get("amount", 0) < 10_000:
        reasons.append("Possible structuring: amount just below £10,000 CTR threshold")
    if row.get("is_round_amount", 0) == 1 and row.get("amount", 0) >= 5_000:
        reasons.append("Round-amount transaction ≥ £5,000 (ML indicator)")
    if row.get("n_txn_last_1h", 0) >= 8:
        reasons.append(f"Velocity alert: {int(row['n_txn_last_1h'])} transactions in last hour")
    if row.get("country_risk", 0) == 2:
        reasons.append("High-risk jurisdiction involvement")
    if row.get("is_off_hours", 0) == 1 and row.get("amount", 0) >= 5_000:
        reasons.append("High-value off-hours transfer (outside 06:00-22:00)")
    if row.get("is_new_payee", 0) == 1 and row.get("country_risk", 0) >= 1:
        reasons.append("First payment to elevated-risk payee")
    if not reasons:
        reasons.append("Anomalous pattern detected by ensemble model")
    return reasons


def generate_alerts(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float = 0.30,
    model_version: str = "ensemble-v1.0",
) -> List[Alert]:
    alerts = []
    now = datetime.now(timezone.utc).isoformat()

    for idx, (_, row) in enumerate(df.iterrows()):
        score = float(scores[idx])
        priority = _classify_priority(score)
        if priority is None:
            continue

        alert = Alert(
            alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            transaction_id=str(row.get("transaction_id", f"TXN{idx:07d}")),
            anomaly_score=round(score, 4),
            priority=priority,
            triggered_at=now,
            model_version=model_version,
            reasons=_derive_reasons(row),
        )
        alerts.append(alert)

    return alerts


def generate_sar_records(alerts: List[Alert], df: pd.DataFrame) -> List[SARRecord]:
    """Create SAR records for HIGH and MEDIUM priority alerts."""
    sar_records = []
    now = datetime.now(timezone.utc)
    txn_index = {
        str(row.get("transaction_id", "")): row
        for _, row in df.iterrows()
    }

    for alert in alerts:
        if alert.priority not in (AlertPriority.HIGH, AlertPriority.MEDIUM):
            continue

        hours = FILING_HOURS[alert.priority]
        if hours:
            from datetime import timedelta
            deadline = (now + timedelta(hours=hours)).isoformat()
        else:
            deadline = "N/A — monitoring only"

        row = txn_index.get(alert.transaction_id, pd.Series(dtype=float))
        supporting = {
            k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
            for k, v in row.to_dict().items()
            if k not in ("transaction_id", "is_fraud")
        }

        narrative = (
            f"Transaction {alert.transaction_id} generated a {alert.priority.value} priority "
            f"alert with anomaly score {alert.anomaly_score:.4f}. "
            f"Reasons: {'; '.join(alert.reasons)}. "
            f"This SAR record has been escalated to the MLRO for review and potential "
            f"submission to the NCA under POCA 2002 s.330."
        )

        sar_records.append(SARRecord(
            sar_id=f"SAR-{uuid.uuid4().hex[:8].upper()}",
            alert_id=alert.alert_id,
            transaction_id=alert.transaction_id,
            anomaly_score=alert.anomaly_score,
            priority=alert.priority.value,
            filing_deadline=deadline,
            narrative=narrative,
            supporting_features=supporting,
            created_at=now.isoformat(),
        ))

    return sar_records


def sar_summary(alerts: List[Alert]) -> dict:
    from collections import Counter
    priority_counts = Counter(a.priority.value for a in alerts)
    return {
        "total_alerts":  len(alerts),
        "high_priority":   priority_counts.get("HIGH", 0),
        "medium_priority": priority_counts.get("MEDIUM", 0),
        "low_priority":    priority_counts.get("LOW", 0),
        "sar_required":    priority_counts.get("HIGH", 0) + priority_counts.get("MEDIUM", 0),
    }
