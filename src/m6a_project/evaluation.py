"""All primary metrics are site-level development metrics."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def metrics(frame):
    y, p = frame.label.to_numpy(), frame.score.to_numpy()
    if set(y) != {0, 1} or not np.isfinite(p).all():
        raise ValueError("Metrics require both binary classes and finite predictions")
    precision, recall, _ = precision_recall_curve(y, p)
    corr = spearmanr(frame.n_reads, p).statistic if len(np.unique(p)) > 1 else np.nan
    return {
        "sites": len(y),
        "positive_sites": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "fraction_score_ge_099": float(np.mean(p >= 0.99)),
        "fraction_score_eq_1": float(np.mean(p == 1)),
        "unique_scores": len(np.unique(p)),
        "depth_score_spearman": float(corr) if np.isfinite(corr) else None,
        "mean_sampling_std": float(frame.sampling_std.mean()),
    }


def save_diagnostics(frame, output, partition_label="Validation"):
    output = Path(output)
    y, p = frame.label, frame.score
    precision, recall, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
        output / "pr_curve.csv", index=False
    )
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(output / "roc_curve.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(recall, precision)
    axes[0, 0].axhline(y.mean(), color="gray", linestyle=":")
    axes[0, 0].set(xlabel="Recall", ylabel="Precision", title=f"{partition_label} precision-recall")
    axes[0, 1].plot(fpr, tpr)
    axes[0, 1].set(
        xlabel="False positive rate", ylabel="True positive rate", title=f"{partition_label} ROC"
    )
    for label in (0, 1):
        axes[1, 0].hist(
            p[y == label],
            bins=np.linspace(0, 1, 41),
            alpha=0.5,
            density=True,
            label=f"Label {label}",
        )
    axes[1, 0].legend()
    axes[1, 0].set(xlabel="Site score", ylabel="Density", title="Score distribution")
    axes[1, 1].hexbin(frame.n_reads, p, gridsize=35, mincnt=1, bins="log")
    axes[1, 1].set(xlabel="Read depth", ylabel="Site score", title="Coverage diagnostic")
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=140)
    plt.close(fig)
    motifs = []
    for motif, group in frame.assign(motif=frame.sequence.str[1:6]).groupby("motif"):
        row = {
            "motif": motif,
            "sites": len(group),
            "prevalence": group.label.mean(),
            "mean_score": group.score.mean(),
        }
        if group.label.nunique() == 2:
            row.update(
                average_precision=average_precision_score(group.label, group.score),
                roc_auc=roc_auc_score(group.label, group.score),
            )
        motifs.append(row)
    pd.DataFrame(motifs).to_csv(output / "motif_metrics.csv", index=False)
    edges = np.linspace(0, 1, 11)
    calibrated = frame.assign(score_bin=pd.cut(frame.score, edges, include_lowest=True))
    calibrated.groupby("score_bin", observed=True).agg(
        sites=("label", "size"), mean_score=("score", "mean"), observed_rate=("label", "mean")
    ).to_csv(output / "calibration.csv")
