"""
Evaluation metrics for anomaly detection in financial transactions.

AUC-ROC alone is misleading for imbalanced fraud datasets (2% fraud rate).
Primary metrics here: AUC-PR, F1, precision-recall at threshold, KS statistic.

Business metric: Alert Precision — of all flagged transactions, what fraction
are actually fraudulent? High precision reduces analyst alert fatigue.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model":         model_name,
        "roc_auc":       roc_auc_score(y_true, scores),
        "avg_precision": average_precision_score(y_true, scores),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "precision":     tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall":        tp / (tp + fn) if (tp + fn) > 0 else 0,
        "fpr":           fp / (fp + tn) if (fp + tn) > 0 else 0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold":     threshold,
    }


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """KS = max separation between fraud and non-fraud score CDFs."""
    fraud_scores  = scores[y_true == 1]
    normal_scores = scores[y_true == 0]
    thresholds = np.linspace(0, 1, 200)
    ks = max(
        abs(
            np.mean(fraud_scores  <= t) -
            np.mean(normal_scores <= t)
        )
        for t in thresholds
    )
    return ks


def plot_pr_curves(
    results: list[Dict],
    y_true: np.ndarray,
    score_dict: Dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')

    colours = ['#0f766e', '#7c3aed', '#dc2626', '#b45309', '#0369a1']

    for ax in axes:
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.tick_params(colors='#94a3b8')

    # PR curves
    for i, (name, scores) in enumerate(score_dict.items()):
        p, r, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        axes[0].plot(r, p, color=colours[i % len(colours)], lw=2,
                     label=f"{name} (AP={ap:.3f})")

    fraud_rate = y_true.mean()
    axes[0].axhline(fraud_rate, color='#475569', ls='--', lw=1.2,
                    label=f"Random ({fraud_rate:.3f})")
    axes[0].set_xlabel("Recall", color='#94a3b8')
    axes[0].set_ylabel("Precision", color='#94a3b8')
    axes[0].set_title("Precision-Recall Curves", color='#f1f5f9', fontweight='bold')
    axes[0].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9', fontsize=9)

    # ROC curves
    for i, (name, scores) in enumerate(score_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        axes[1].plot(fpr, tpr, color=colours[i % len(colours)], lw=2,
                     label=f"{name} (AUC={auc:.3f})")

    axes[1].plot([0, 1], [0, 1], color='#475569', ls='--', lw=1.2, label="Random")
    axes[1].set_xlabel("False Positive Rate", color='#94a3b8')
    axes[1].set_ylabel("True Positive Rate",  color='#94a3b8')
    axes[1].set_title("ROC Curves", color='#f1f5f9', fontweight='bold')
    axes[1].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_score_distributions(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#94a3b8')

    ax.hist(scores[y_true == 0], bins=60, alpha=0.6,
            color='#0f766e', label='Normal', density=True)
    ax.hist(scores[y_true == 1], bins=60, alpha=0.7,
            color='#dc2626', label='Fraud', density=True)
    ax.axvline(threshold, color='#f59e0b', ls='--', lw=2,
               label=f'Threshold ({threshold:.2f})')

    ax.set_xlabel("Anomaly Score", color='#94a3b8')
    ax.set_ylabel("Density", color='#94a3b8')
    ax.set_title(f"{model_name} — Score Distribution", color='#f1f5f9', fontweight='bold')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()
