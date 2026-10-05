"""Run in the pinned Python 3.8 container using official m6Anet components."""

import hashlib
import inspect
import json
import random
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import toml
import torch
from m6anet.model.model import MILModel
from m6anet.utils import data_utils, sampler_utils, training_utils
from m6anet.utils.constants import DEFAULT_MODEL_CONFIG
from m6anet.utils.loss_functions.loss_functions import binary_cross_entropy_loss
from torch.utils.data import DataLoader


def seed_worker(worker_id):
    # Upstream builder seeds workers from wall-clock seconds. Use recorded RNG seeds instead.
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    root = Path("/result")
    cfg = json.loads((root / "config.json").read_text())
    if cfg["epochs"] != 1:
        raise ValueError("This smoke runner executes exactly one epoch")
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    start = time.perf_counter()
    output = root / "official_output"
    output.mkdir()
    for name, module in [
        ("training", training_utils),
        ("sampling", sampler_utils),
        ("data", data_utils),
    ]:
        (output / (name + "_source.py")).write_text(inspect.getsource(module))
    (output / "model_config.toml").write_text(Path(DEFAULT_MODEL_CONFIG).read_text())
    print("Computing normalization from training sites only", flush=True)
    norm_start = time.perf_counter()
    train_ds = data_utils.NanopolishDS(
        str(root / "adapted_input"), mode="Train", n_processes=cfg["workers"], min_reads=20
    )
    norm_seconds = time.perf_counter() - norm_start
    for mean, std in train_ds.norm_dict.values():
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid training normalization")
    norm_path = output / "train_norm_dict.joblib"
    joblib.dump(train_ds.norm_dict, norm_path)
    val_ds = data_utils.NanopolishDS(
        str(root / "adapted_input"), mode="Val", norm_path=str(norm_path), min_reads=20
    )
    for sequence in val_ds.data_info.kmer:
        if any(sequence[i : i + 5] not in train_ds.norm_dict for i in range(3)):
            raise ValueError("Validation motif absent from training normalization")
    sampler = sampler_utils.ImbalanceOverSampler(train_ds)
    options = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["workers"],
        "collate_fn": data_utils.train_collate,
        "worker_init_fn": seed_worker,
    }
    train_dl = DataLoader(train_ds, sampler=sampler, **options)
    val_dl = DataLoader(val_ds, shuffle=False, **options)
    model = MILModel(toml.load(DEFAULT_MODEL_CONFIG)).to("cpu")
    initial_hash = hashlib.sha256(
        b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())
    ).hexdigest()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    setup = {
        "training_sites": len(train_ds),
        "validation_sites": len(val_ds),
        "sampled_sites_per_epoch": len(sampler),
        "training_batches": len(train_dl),
        "validation_batches_per_pass": len(val_dl),
        "normalization_seconds": norm_seconds,
        "normalization_motifs": len(train_ds.norm_dict),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial_hash,
        "pretrained_weights_loaded": False,
        "test_evaluated": False,
        "normalization_partition": "train",
        "worker_seeding": "torch.initial_seed modulo 2**32; numpy and random",
        "threads": torch.get_num_threads(),
        "workers": cfg["workers"],
    }
    write(output / "setup.json", setup)
    print(json.dumps(setup, indent=2), flush=True)
    print("Starting one complete oversampled training epoch", flush=True)
    train_start = time.perf_counter()
    train_result = training_utils.train_one_epoch(
        model, train_dl, "cpu", optimizer, binary_cross_entropy_loss
    )
    train_seconds = time.perf_counter() - train_start
    if not np.isfinite(train_result["avg_loss"]):
        raise ValueError("Non-finite training loss")
    final_hash = hashlib.sha256(
        b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())
    ).hexdigest()
    if final_hash == initial_hash:
        raise ValueError("Model weights did not change")
    write(output / "train_metrics.json", train_result)
    print(f"Training completed in {train_seconds:.3f} seconds; validating", flush=True)
    val_start = time.perf_counter()
    val_result = training_utils.validate(
        model, val_dl, "cpu", binary_cross_entropy_loss, n_iterations=cfg["validation_iterations"]
    )
    val_seconds = time.perf_counter() - val_start
    predictions = np.asarray(val_result.pop("y_pred"))
    truth = val_result.pop("y_true")
    frame = val_ds.data_info.copy()
    np.testing.assert_array_equal(frame.modification_status.to_numpy(), truth)
    frame["score"] = predictions.mean(axis=0)
    frame["sampling_std"] = predictions.std(axis=0)
    frame.rename(columns={"modification_status": "label", "kmer": "sequence"}).to_csv(
        output / "validation_predictions.csv", index=False
    )
    np.save(output / "validation_repetitions.npy", predictions)
    torch.save(model.state_dict(), output / "model_states.pt")
    torch.save(
        {
            "epoch": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        output / "checkpoint.pt",
    )
    result = dict(
        setup,
        train=train_result,
        validation=val_result,
        train_seconds=train_seconds,
        validation_seconds=val_seconds,
        container_work_seconds=time.perf_counter() - start,
        final_state_sha256=final_hash,
        main_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        largest_child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    )
    write(output / "timing.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
