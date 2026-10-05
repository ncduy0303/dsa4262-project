# Data0 v2 split and m6Anet CPU smoke test

Completed one full training epoch from random weights. CPU training is feasible in the
current local Docker setup: **115.7 seconds for training** and
**109.4 seconds for five validation passes**. This is a timing
and integration check, not evidence that one epoch is sufficient for final accuracy.

## Split

| Partition | Sites | Genes | Positive sites | Prevalence |
|---|---:|---:|---:|---:|
| test | 24,694 | 770 | 1,107 | 4.48% |
| train | 71,794 | 2,307 | 3,265 | 4.55% |
| val | 25,350 | 775 | 1,103 | 4.35% |

- Five-fold `StratifiedGroupKFold`, seed 42: fold 0 validation, fold 1 test, others train.
- All pairs of partitions are disjoint by gene and transcript. Both classes occur in each.
- Classical v2 and m6Anet configurations produce byte-identical split manifests.
- Split SHA256: `90843c676442a0e7b6e47dca10c697ed03d42bec2c0f3730d794fb1f34a95f95`.
- Validation membership is unchanged from v1. The new test fold was part of historical
  v1 training. Old models must not be used to claim independent test performance.
- No test scoring was performed. No test reads are present in the m6Anet adapted input.
  Data1 and data2 were not used.

## Measured timings

| Stage | Seconds |
|---|---:|
| Adapt data0 training/validation into official input format | 39.8 |
| Fit training-only motif normalization | 64.9 |
| One full training epoch | 115.7 |
| Validation, five read-sampling passes | 109.4 |
| Training plus validation | 225.1 |
| Full run through prediction/metric export, including startup | 335.1 |

The full recorded run took 5.59 minutes. Diagnostic plot rendering
and subsequent report/notebook checks are outside this timer. Input adaptation and
normalization are setup costs that need not be repeated for every future epoch, provided
inputs and the training split remain unchanged. Longer-run time has not been measured.

- CPU-only Docker, Linux amd64 emulation on an arm64 macOS host.
- Container allowance: 4 CPUs, 4 PyTorch threads,
  2 DataLoader workers, batch size 256.
- Main-process peak RSS: 390.6 MiB.
- Largest completed child peak RSS: 285.7 MiB.
  These are separate process peaks, not total simultaneous RAM or Docker VM memory.
- Official m6Anet 2.1.0, Python 3.8, CPU PyTorch 1.6.0 in the existing isolated image.
  The project's uv Python 3.12.8 environment is unchanged.

## Training behavior and validation

- Unmodified official `MILModel`: 9 signals plus learned sequence embeddings, Noisy-OR
  pooling over 20 sampled reads, 7,697 trainable parameters.
- Official `ImbalanceOverSampler`: 137,058 site presentations
  across 536 batches. Each majority-class site is included once,
  while minority sites are sampled with replacement to balance the epoch.
- Adam, learning rate 0.0004, weight decay 0, site-level binary cross entropy.
- Normalization fitted only to the 71,794 training sites; all
  66 motifs covered. No pretrained normalization or weights loaded.
- Official `train_one_epoch` and `validate` functions are used. The CLI's automatic
  test-evaluation stage is skipped. Worker RNGs use recorded seeds instead of upstream
  wall-clock seeding.
- Model-state hashes differ before/after training; loss is finite. Validation predictions
  match every frozen validation key and label exactly.

| Validation metric after one epoch | Value |
|---|---:|
| Average precision | 0.367761 |
| ROC AUC | 0.865749 |
| Trapezoidal PR AUC | 0.367181 |
| Binary cross entropy | 0.449716 |
| Brier score | 0.144862 |

Training-set metrics use the balanced oversampled stream and are not directly comparable
with natural-prevalence validation AP. Prior classical and pretrained reports remain
historical v1 references. Rerun classical baselines on v2 for a matched training-data
comparison; do not use this single epoch to conclude model superiority or convergence.

## Reproduce and inspect

```bash
uv run python scripts/train_m6anet.py --config configs/experiments/m6anet_train.toml
uv run python scripts/execute_notebooks.py notebooks/05_m6anet_training_smoke.ipynb
```

- Configuration: `configs/experiments/m6anet_train.toml`.
- Run: `experiments/data0_m6anet_scratch_v2/20261005T103446_official_m6anet_m6anet_scratch_119c482f`.
- `config.json`, `split.csv`, `data_provenance.json`, `official_environment.json`,
  `source/`, and `source_hashes.json` preserve settings and provenance.
- `official_output/timing.json` contains full timing and training metrics.
- `official_output/train_norm_dict.joblib`, `model_states.pt`, and `checkpoint.pt`
  retain normalization, weights, and optimizer/RNG state.
- `official_output/executed_training_driver.py` preserves the driver used in this run.
- `validation_predictions.parquet`, diagnostic curves/plots, and raw sampling passes
  retain outputs. `artifact_hashes.json` checksums the key artifacts.
- `experiments/index.csv` records this completed run alongside historical experiments.

The default classical configuration is now v2. Its training loop evaluates validation
only. The explicit `scripts/evaluate_test.py` command checks the frozen model's split,
input hashes, and preprocessing before test evaluation. It rejects historical two-way
models. This command was tested only with synthetic fixtures; the biological test fold
remains unevaluated. All 18 automated tests passed, including three-way leakage checks,
train/validation-only adaptation, the full classical runner, and explicit test evaluation.
