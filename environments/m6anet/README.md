# Official pretrained m6Anet environment

The main Python 3.12 uv environment remains separate. This benchmark installs the official Bioconda `m6anet=2.1.0` package in a Linux amd64 conda environment using micromamba. The recipe retains Python 3.8 and PyTorch 1.6.0 as required upstream.

```bash
docker build --platform linux/amd64 -t dsa4262-m6anet:2.1.0 environments/m6anet
uv run python scripts/benchmark_m6anet.py
```

Docker Desktop must be running on macOS. Docker is needed only for this benchmark. The Linux amd64 container can also run on an x86_64 Linux Docker host.

- The CPU build of PyTorch is selected explicitly; `cpuonly` alone did not prevent the solver selecting a CUDA build from another channel.
- MKL is pinned to 2023.2 because the initial environment resolved a newer MKL that failed when importing legacy PyTorch with an `iJIT_IsProfilingActive` undefined-symbol error. This is an environment compatibility pin, not a modification of the m6Anet algorithm. Related upstream report: https://github.com/pytorch/pytorch/issues/123097.
- NumPy stays below 1.24 and pandas below 2 for the legacy stack. Exact resolved versions, package URLs, checkpoint hashes, normalization hashes, and the Docker image ID are saved with every benchmark run.
- The base image is pinned by digest. The generated `conda-explicit-linux-64.txt`, when present, records the exact successful solve and can be used with `Dockerfile.locked` for reconstruction.
- Inference has networking disabled. Only the benchmark script directory and this run's artifacts are mounted. The original datasets are not mounted in the inference container.

The adapter selects only the existing data0 validation keys, appends synthetic per-site read indices after the nine original signals, and rebuilds byte offsets. It checks exact signal preservation and seeks every generated index. The original inputs are never edited.

The primary benchmark uses `HCT116_RNA002`, seed 42, CPU, one inference worker, 20 reads per sample, and 1,000 sampling iterations. The official release samples with replacement. Our classical models used five repetitions without replacement, so this is a comparison of complete methods, not an isolated architecture ablation.

`save_per_batch` is set larger than the total number of batches. In the inspected official release, the write condition is `(it + 1) % save_per_batch`, and the function has no final leftover flush. This setting makes that condition true on every batch for this dataset, preventing dropped predictions without editing upstream code. Exact output-key coverage is independently enforced.

The checkpoint was trained on HCT116. Its overlap with the course data0 training/validation sites and the precise course sequencing chemistry are not established. Report this as a pretrained reference with possible training overlap, not an independent generalization estimate or a matched-training comparison.

Upstream sources: https://github.com/GoekeLab/m6anet and https://github.com/bioconda/bioconda-recipes/blob/master/recipes/m6anet/meta.yaml. The paper is available locally as `reference/m6Anet.pdf`.

To build from the exact successful conda package export, use `docker build --platform linux/amd64 -f environments/m6anet/Dockerfile.locked -t dsa4262-m6anet:2.1.0 environments/m6anet`. The per-run image ID and checkpoint hashes identify the measured run.

## Local training smoke test

From the repository root, run `uv run python scripts/train_m6anet.py --config configs/experiments/m6anet_train.toml`.
The configuration is `configs/experiments/m6anet_train.toml`. This uses the same
pinned CPU image and trains official m6Anet from scratch for exactly one epoch.
The main Python/uv environment does not install legacy PyTorch.

- Uses only the v2 data0 training and validation folds; test reads are omitted.
- Fits the official 5-mer normalization on training reads and reuses it for validation.
- Uses official `MILModel`, `ImbalanceOverSampler`, binary cross entropy,
  `train_one_epoch`, and `validate`, with Adam and the supplied architecture.
- Uses seeded DataLoader workers; the upstream builder uses wall-clock seeds.
- Skips the CLI's automatic test evaluation. Test scoring is reserved for a frozen model.
- Records separate adaptation, normalization, training, and validation times, plus
  main-process and largest-child peak RSS. These RSS values are separate process
  peaks, not a simultaneous total-memory estimate.
- Uses Linux amd64 emulation on the current arm64 host. Timings describe this
  Docker setup, not native ARM PyTorch performance.

`model_states.pt` is the official model state dictionary. `train_norm_dict.joblib`
is the training-only normalization. `checkpoint.pt` additionally saves optimizer
and RNG state for future resume support. The current entry point remains a
one-epoch smoke runner. See `docs/results/m6anet_training_smoke.md` for results.


## Full training

`uv run python scripts/train_m6anet.py` now defaults to `m6anet_full.toml` and the
`train_full.py` container driver. It applies validation-loss early stopping and saves
selected and last checkpoints separately. `training_control.py` is independent of
PyTorch and tested in the main uv environment. Five fixed-seed validation passes
provide comparable per-epoch losses without influencing training RNG state.

`MemoryReads` caches official normalized signals as float32 and preserves the
original 20-read sampling without replacement. Before training, it verifies feature,
motif, and label equality to official `NanopolishDS` samples under matching RNG.
The selected checkpoint's predictions are reproduced after reloading its weights.
The source directory mounted into the container is the immutable run snapshot.
