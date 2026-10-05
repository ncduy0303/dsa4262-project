# Classical m6A baselines

Reproducible, data0-only experiments for DSA4262 Task 1. The approved scope leaves data1 and data2 untouched. Core transformations and training live in `src/m6a_project`; notebooks explore the same logged results.

## Setup

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/) first, then:

```bash
uv sync --locked --group dev --group notebook
uv run pytest -q
```

Python is pinned to 3.12.8. `uv.lock` pins the scientific and notebook dependencies. CPU execution is the default. The classical baselines do not require PyTorch, conda, or a GPU. The optional official m6Anet benchmark uses its own Docker/conda environment. In a restricted environment, point `UV_CACHE_DIR` and `XDG_CACHE_HOME` to writable locations.

## Experiments

```bash
uv run python scripts/run_experiments.py
# Skip exact completed configurations with matching inputs and implementation:
uv run python scripts/run_experiments.py --resume
# A single logged experiment:
uv run python scripts/train.py --approach read15_noisy_or --model logistic
uv run python scripts/build_report.py
```

- Configuration: `configs/experiments/baselines.toml`.
- Current v2 split: five stratified gene-grouped folds, seed 42. Fold 0 is validation, fold 1 is test, and the other folds are training (approximately 60/20/20). Genes and transcripts are disjoint across all three. Test scores are excluded from experiment selection.
- A1 `site_mean9`: nine feature means across all reads, one site-level classifier.
- A2 `read9_noisy_or`: nine features per sampled read, inherited site labels, Noisy-OR pooling.
- A3 `read15_noisy_or`: A2 plus three two-dimensional fixed sequence PCA vectors.
- `site_mean9_matched`: secondary control using the same read samples as A2/A3.
- Estimators: logistic regression, histogram gradient boosting, and random forest. A prevalence dummy is also logged.
- Read training samples 20 reads per site and uses weight `1/20` per read so total weight per site is one. Validation averages five independently sampled 20-read site predictions. Seeds are derived from the site key, phase, and repetition, so input order and batching do not alter the samples.
- Fixed feature/model settings are shared across approaches. The random forest uses a quarter-sized bootstrap sample per tree, maximum depth 12, and minimum leaf size 20; this is recorded in the configuration. Automatic random validation for boosting and read-level out-of-bag evaluation are disabled.

The script creates a new immutable run directory for each fit. The main matrix, dummy, and matched-sample controls all run through the same tracker. To change a setting, copy the TOML, choose a descriptive `experiment_set`, and run with `--config path/to/config.toml`.

## Read the results

- `docs/results/report.md`: historical v1 interpretation and verification.
- `docs/results/data0_baselines_v2/`: reports for future v2 classical runs.
- `docs/results/m6anet_training_smoke.md`: current three-way split and CPU training timing.
- `docs/results/comparison.csv` and `comparison.png`: model comparisons.
- `experiments/index.csv`: all completed and failed runs.
- `experiments/<set>/<run>/`: configuration, source snapshot, environment, split, metrics, warnings/logs, model, per-site predictions, and diagnostics.
- `artifacts/cache/`: data0-only versioned feature/read arrays, sampled indices, split, and input hashes. These are derived and can be regenerated.

AP, ROC AUC, and trapezoidal PR AUC are reported separately. These are validation estimates used for development, not independent test estimates. For a constant dummy score, trapezoidal PR AUC has a coarse-curve endpoint artifact; use its AP/prevalence as the useful no-skill reference.

## Notebooks

```bash
uv run python -m ipykernel install --sys-prefix --name m6a-project --display-name 'm6A project (uv)'
uv run jupyter lab
```

Notebooks 01-04 preserve the historical v1 results and configuration. Notebook 05 reads the new v2 split and training smoke results; notebook 06 reads full-training learning curves. Open the notebooks and select **m6A project (uv)**. Saved notebooks contain outputs. Re-execution reads caches and logged runs. In notebook 02, `RUN_NEW_EXPERIMENT = False` prevents accidental retraining; enabling it invokes the same tracked `run_experiment` function. Run the baseline matrix and build the report before replaying notebooks 02 and 03.

For automated execution, use `uv run python scripts/execute_notebooks.py`. It creates a project-local kernel specification rather than changing a global kernel.

## Label-free prediction and CSV export

Only load joblib models created by you or a trusted source. The bundle includes its scaler, sequence mapping, estimator, feature definition, sampling policy, class mapping, and provenance.

```bash
uv run python scripts/predict.py \
  --model experiments/SET/RUN/model.joblib \
  --input PATH_TO_JSON_OR_JSON_GZ \
  --output artifacts/predictions/new_predictions.csv
```

Replace the model and input placeholders. Output must not already exist. No label file is required. The command validates the exact three-column header, input-key coverage/order, uniqueness, finite scores, and probability bounds, then writes a checksum manifest. The generic predictor supports future datasets, but the implemented experiment runner permits only data0.

Read-based models fail explicitly on sites with fewer than 20 reads. Unsupported sequence contexts fail for the embedding model. Original per-read identifiers are absent, so read indices are local to a site and cannot be used for tracking molecules across sites.

A generated synthetic prediction-only example is provided in `tests/fixtures/prediction_example.json`. It is not biological training data. Use it with a locally trained model to smoke-test the command. The data0 validation prediction export is under `artifacts/validation/`; no leaderboard upload has been performed.

## Interpretation and extensions

- Inherited site labels are noisy read targets. These read classifiers do not optimize the m6Anet site-level MIL objective, and their outputs are not validated molecule-level modification probabilities.
- Noisy-OR pooling does not guarantee calibrated site scores, even with a fixed read count. Calibration and saturation diagnostics are saved.
- The PCA sequence embedding is fixed and non-neural, unlike m6Anet's learned embedding. It can collapse different motifs into the same vector. Its variance and collision statistics are recorded.
- Increasing performance from A1 to A3 is a hypothesis. Negative results remain in the report.
- Data1/data2 evaluation, calibration, tuning, and Task 2 analysis remain future work. Official m6Anet training and the pretrained benchmark are documented below.

Extend `features.py` to add representations, `models.py` to register classifiers, and `pooling.py` to add aggregators. Preserve the saved split and paired read samples when comparing changes. Prediction requires the complete versioned bundle, not just the sklearn estimator.

## Provenance and references

The implementation follows the approved `docs/data0-classical-baselines-plan.md` and local course references. Scientific motivation: Hendra et al., *Detection of m6A from direct RNA sequencing using a multiple instance learning framework*, Nature Methods (2022), [DOI](https://doi.org/10.1038/s41592-022-01666-1), supplied as `reference/m6Anet.pdf`. This code is a classical adaptation, not a reproduction of the neural method. AI assistance and corrections are documented in `docs/ai-use.md`.

Large datasets, caches, and experiment artifacts are ignored for new Git additions. Some data files were already tracked before this implementation; `.gitignore` does not remove existing tracked files. Review redistribution permissions and repository contents before making the course repository public.

## Official m6Anet benchmark

See `environments/m6anet/README.md` for the isolated legacy environment and `docs/results/m6anet_comparison.md` for the measured comparison on the same frozen validation sites. Run `uv run python scripts/benchmark_m6anet.py` after building the container. Pretrained HCT116 training overlap is unknown, so treat this as a pretrained reference rather than an independent test.

## Three-way split and local m6Anet training

The default `baselines.toml` now writes `data0_baselines_v2` runs. Training and validation
use 71,794 and 25,350 sites respectively; 24,694 sites are reserved for test. The old
`baselines_v1.toml`, immutable runs, and reports remain available. The validation fold
is unchanged, but v2 training excludes the new test fold. Earlier v1 models trained on
those test sites, so do not evaluate those old models on the new test set or interpret
it as historically untouched. Rerun the classical matrix for matched v2 comparisons.

```bash
# Requires the existing pinned m6Anet Docker image and a running Docker engine:
uv run python scripts/train_m6anet.py --config configs/experiments/m6anet_train.toml
# Retrain classical baselines on the same v2 split:
uv run python scripts/run_experiments.py
uv run python scripts/build_report.py
# Only after selecting and freezing a v2 classical model using validation:
uv run python scripts/evaluate_test.py --model experiments/SET/RUN/model.joblib
```

The m6Anet smoke runner executes exactly one complete CPU training epoch from random
initialization, followed by five validation sampling passes. It uses the official
architecture, learned embedding, Noisy-OR objective, Adam, and class oversampler.
Normalization is fitted on training reads only. No test data is exported to its input,
and no test dataset is instantiated. The runner calls official training functions
because the upstream CLI automatically evaluates test data after training. Worker
seeds are deterministic instead of the upstream wall-clock-based initialization.

Every run records the split, raw-data hashes, source snapshot, Docker image ID,
installed package provenance, normalization, trained weights, optimizer/RNG checkpoint,
validation probabilities, and separate preparation/training/validation timings.
`configs/experiments/m6anet_train.toml` controls the smoke experiment. The main uv
Python environment remains independent of legacy m6Anet dependencies.

Test evaluation is an explicit command, never part of the experiment sweep. It rejects
old two-way models and mismatched split, inputs, or preprocessing. Reserve it until
model selection is complete; repeated test-driven tuning would invalidate the holdout.


## Full m6Anet training with validation-loss selection

```bash
uv run python scripts/train_m6anet.py --config configs/experiments/m6anet_full.toml
uv run python scripts/report_m6anet_training.py
```

The default training command now uses `m6anet_full.toml`. It starts from random
weights, trains for at most 50 epochs, and stops after five epochs without a
validation-loss improvement greater than 0.0005, after a minimum of ten epochs.
The checkpoint with the absolute lowest validation loss is retained. Validation
uses fixed seeded read samples across epochs and never consumes the test fold.
Normalized reads are cached in memory with equivalence checks against the official
dataset. The official m6Anet model, oversampler, loss, and training function remain
in use. Training-stream loss reflects balanced oversampling; validation uses the
natural class prevalence, so their absolute values are not directly comparable.

`docs/results/m6anet_full_training.md` and notebook 06 show the completed learning
curves and selected model. Each run retains both selected and last checkpoints,
optimizer/RNG state, normalization, source/environment hashes, and validation
predictions. The entry point creates a fresh run; resume is not yet exposed as a CLI
option. Use the explicit smoke configuration above to repeat the one-epoch timing
experiment without changing the full-training report.
