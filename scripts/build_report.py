"""Summarize logged runs without training or selecting a new split."""

import argparse
import json
from pathlib import Path

import pandas as pd

from m6a_project.evaluation import plt
from m6a_project.tracking import rebuild_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="data0_baselines_v2")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output or f"docs/results/{args.set}")
    output.mkdir(parents=True, exist_ok=True)
    index = rebuild_index()
    completed = index[(index.experiment_set == args.set) & (index.status == "completed")].copy()
    if completed.empty:
        raise ValueError("No completed runs")
    completed = completed.sort_values("started_utc").drop_duplicates(
        ["approach", "model"], keep="last"
    )
    columns = [
        "approach",
        "model",
        "average_precision",
        "roc_auc",
        "pr_auc_trapezoid",
        "brier_score",
        "log_loss",
        "fit_seconds",
        "prediction_seconds",
        "fraction_score_ge_099",
        "run_path",
    ]
    completed[columns].to_csv(output / "comparison.csv", index=False)
    main_approaches = ["site_mean9", "read9_noisy_or", "read15_noisy_or"]
    primary = completed[completed.approach.isin(main_approaches) & (completed.model != "dummy")]
    comparison = primary.pivot(
        index="model", columns="approach", values="average_precision"
    ).reindex(columns=main_approaches)
    differences = pd.DataFrame(
        {
            "A2_minus_A1_AP": comparison.read9_noisy_or - comparison.site_mean9,
            "A3_minus_A2_AP": comparison.read15_noisy_or - comparison.read9_noisy_or,
        }
    )
    differences.to_csv(output / "approach_differences.csv")
    ax = comparison.plot.bar(figsize=(10, 5), rot=0)
    ax.set(ylabel="Validation average precision", title="Data0: same gene split across approaches")
    ax.legend(title="Approach", fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "comparison.png", dpi=150)
    plt.close(ax.figure)
    # Save paired prediction keys and check all comparisons use precisely the same labels.
    expected = None
    frames = {}
    for row in primary.itertuples():
        frame = pd.read_parquet(Path(row.run_path) / "validation_predictions.parquet")
        keys = frame[["gene_id", "transcript_id", "transcript_position", "label"]]
        if expected is None:
            expected = keys
        else:
            pd.testing.assert_frame_equal(
                expected.reset_index(drop=True), keys.reset_index(drop=True)
            )
        frames[(row.model, row.approach)] = frame
    summary = {
        "experiment_set": args.set,
        "runs": len(completed),
        "same_validation_keys_verified": True,
        "best_validation_run": primary.sort_values("average_precision", ascending=False)
        .iloc[0]
        .run_path,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(completed[columns].to_string(index=False))
    print("\nPaired AP differences:\n", differences.to_string())


if __name__ == "__main__":
    main()
