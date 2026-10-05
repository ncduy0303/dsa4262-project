"""Summarize a completed full training run and its validation-loss stopping decision."""

import json
from pathlib import Path

import pandas as pd

from m6a_project.data import file_hash
from m6a_project.evaluation import plt


def main():
    summary_path = Path("docs/results/m6anet_full_training.json")
    summary = json.loads(summary_path.read_text())
    run = Path(summary["run_path"])
    timing, metrics = summary["timing"], summary["metrics"]
    config = json.loads((run / "config.json").read_text())
    history = pd.read_csv(run / "official_output/history.csv")
    assert len(history) == timing["epochs_completed"]
    assert int(history.loc[history.val_loss.idxmin(), "epoch"]) == timing["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_loss, label="Training: balanced sampled stream")
    axes[0].plot(history.epoch, history.val_loss, label="Validation: natural prevalence")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross entropy", title="Training and validation loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(history.epoch, history.val_roc_auc, label="Validation ROC AUC")
    axes[1].plot(history.epoch, history.val_pr_auc, label="Validation trapezoidal PR AUC")
    axes[1].set(xlabel="Epoch", ylabel="AUC", title="Validation discrimination")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(timing["best_epoch"], linestyle=":", color="black", label="Selected epoch")
    fig.tight_layout()
    fig.savefig(run / "learning_curves.png", dpi=160)
    fig.savefig("docs/results/m6anet_learning_curves.png", dpi=160)
    plt.close(fig)
    history.to_csv("docs/results/m6anet_training_history.csv", index=False)
    artifacts = {
        str(p.relative_to(run)): file_hash(p)
        for p in [
            run / "split.csv",
            run / "official_output/model_states.pt",
            run / "official_output/checkpoint.pt",
            run / "official_output/last_checkpoint.pt",
            run / "official_output/train_norm_dict.joblib",
            run / "official_output/history.csv",
            run / "validation_predictions.parquet",
            run / "learning_curves.png",
        ]
    }
    (run / "artifact_hashes.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    selected = history.loc[history.epoch == timing["best_epoch"]].iloc[0]
    tail = history.tail(config["patience"] + 1)
    rows = "\n".join(
        f"| {int(r.epoch)} | {r.train_loss:.6f} | {r.val_loss:.6f} | {int(r.best_epoch)} |"
        for r in tail.itertuples()
    )
    report = f"""# Full m6Anet training on data0

Selected **epoch {timing["best_epoch"]}**, the checkpoint with the lowest validation
binary cross entropy, **{timing["best_val_loss"]:.6f}**. Training completed
**{timing["epochs_completed"]} epochs**. Stop reason: `{timing["stop_reason"]}`.

## Epoch selection

- Prespecified ceiling: {config["epochs"]} epochs; minimum: {config["min_epochs"]} epochs.
- Stop after {config["patience"]} epochs without an improvement greater than
  {config["min_delta"]} in validation binary cross entropy relative to the last
  meaningful improvement. The checkpoint with the absolute lowest loss is retained,
  even if its improvement is smaller than this patience threshold.
- Each validation epoch averages {config["validation_iterations"]} passes using the
  same seed ({config["validation_seed"]}) and worker allocation. Validation RNG is
  isolated from training RNG to keep the comparison stable.
- Selected checkpoint training-stream loss: {selected.train_loss:.6f}.
- Training loss is measured on the balanced oversampled stream during optimization.
  Validation loss is measured after the epoch at natural class prevalence. Their
  absolute difference is not a directly comparable train/validation generalization gap.
- Test data never participates in stopping or checkpoint selection.

Last recorded epochs:

| Epoch | Training loss | Validation loss | Best epoch so far |
|---|---:|---:|---:|
{rows}

![Learning curves](m6anet_learning_curves.png)

## Selected model validation metrics

| Metric | Value |
|---|---:|
| Average precision | {metrics["average_precision"]:.6f} |
| ROC AUC | {metrics["roc_auc"]:.6f} |
| Trapezoidal PR AUC | {metrics["pr_auc_trapezoid"]:.6f} |
| Binary cross entropy | {metrics["log_loss"]:.6f} |
| Brier score | {metrics["brier_score"]:.6f} |

These are development validation results, not test estimates. The checkpoint's saved
predictions were reproduced exactly after reloading its weights. No pretrained weights
or pretrained normalization were used.

## Data and implementation

- Frozen v2 split: {summary["split"]["train"]["sites"]:,} training,
  {summary["split"]["val"]["sites"]:,} validation, {summary["split"]["test"]["sites"]:,} test sites.
- All partitions are disjoint by gene and transcript. The test fold was part of
  historical v1 training, so old v1 models cannot be used for independent test claims.
- Training-only official motif normalization, official m6Anet architecture and loss,
  learned sequence embedding, 20-read Noisy-OR pooling, official balanced oversampling.
- Adam learning rate {config["learning_rate"]}, weight decay {config["weight_decay"]},
  batch size {config["batch_size"]}. CPU Docker, Linux amd64 emulation on arm64 macOS,
  {config["threads"]} PyTorch threads and {config["workers"]} DataLoader workers.
- Normalized reads are cached in RAM. Cached sample features, motif indices, and labels
  were compared exactly against the official dataset under matching RNG states.
- The main uv Python environment remains separate from the pinned legacy container.
- Data1/data2 remain unused. Test predictions were not generated.

## Measured runtime

| Stage | Seconds |
|---|---:|
| Input adaptation | {metrics["adapter_seconds"]:.1f} |
| Training-only normalization | {timing["normalization_seconds"]:.1f} |
| Build normalized read cache | {timing["cache_seconds"]:.1f} |
| All training epochs | {timing["train_seconds"]:.1f} |
| Per-epoch validation passes | {timing["validation_seconds"]:.1f} |
| Full run through metric export | {metrics["total_seconds"]:.1f} |

Full recorded runtime: {metrics["total_seconds"] / 60:.2f} minutes. This includes the final
checkpoint prediction verification. Main-process peak RSS was
{timing["main_process_peak_rss_mib"]:.1f} MiB; the largest child peak was
{timing["largest_child_peak_rss_mib"]:.1f} MiB. These are separate process peaks, not a
sum of simultaneous RAM usage or a Docker VM total.

## Reproduce and reuse

```bash
uv run python scripts/train_m6anet.py --config configs/experiments/m6anet_full.toml
uv run python scripts/report_m6anet_training.py
```

Run directory: `{run}`.

- `official_output/model_states.pt`: selected model weights, suitable for the official
  m6Anet model architecture saved in `official_output/model_config.toml`.
- `official_output/train_norm_dict.joblib`: matching training-only normalization.
- `official_output/checkpoint.pt`: selected weights plus optimizer, RNG, history,
  and early-stopping state.
- `official_output/last_checkpoint.pt`: final-epoch state for future recovery work.
  The current CLI starts a fresh logged run; it does not yet expose a resume command.
- `official_output/history.csv`, `learning_curves.png`, `training.log`, and
  `validation_predictions.parquet`: recorded learning trajectory and selected predictions.
- Source snapshots, image ID, environment and input hashes, and `artifact_hashes.json`
  preserve provenance. `experiments/index.csv` indexes the completed experiment.

Historical smoke, classical, and pretrained reports are retained. Classical v1 models
used a larger training partition, and the pretrained model has unknown training overlap.
Their scores are contextual references, not matched v2 training comparisons.
"""
    Path("docs/results/m6anet_full_training.md").write_text(report)
    print(report.split("## Epoch selection")[0])


if __name__ == "__main__":
    main()
