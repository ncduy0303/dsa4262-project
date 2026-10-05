"""CPU training with official m6Anet MIL components and validation-loss selection."""

import hashlib
import inspect
import json
import random
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import toml
import torch
from m6anet.model.model import MILModel
from m6anet.utils import data_utils, sampler_utils, training_utils
from m6anet.utils.constants import DEFAULT_MODEL_CONFIG
from m6anet.utils.loss_functions.loss_functions import binary_cross_entropy_loss
from torch.utils.data import DataLoader, Dataset
from train_smoke import seed_worker, write
from training_control import EarlyStopping


def state_hash(model):
    return hashlib.sha256(
        b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())
    ).hexdigest()


def rng_state():
    return {
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


class MemoryReads(Dataset):
    """Cache official normalized features; preserve official read sampling exactly."""

    def __init__(self, source):
        self.data_info = source.data_info
        self.labels = source.labels
        self.min_reads = source.min_reads
        self.features, self.kmers = [], []
        for index in range(len(source)):
            _, _, _, features, sequence = source.load_data(index)
            motifs = [sequence[i : i + 5] for i in range(3)]
            mean, std = source.get_norm_factor(motifs)
            self.features.append(np.asarray((features - mean) / std, dtype=np.float32))
            self.kmers.append(
                torch.LongTensor(
                    np.repeat(
                        np.array([source.kmer_to_int[k] for k in motifs]).reshape(1, 3),
                        self.min_reads,
                        axis=0,
                    )
                )
            )
        # Verify the complete returned tuple against the official Dataset, with identical RNG.
        saved = rng_state()
        for index in sorted({0, len(source) // 2, len(source) - 1}):
            state = rng_state()
            expected = source[index]
            restore_rng(state)
            actual = self[index]
            if not torch.equal(expected[0], actual[0]) or not torch.equal(expected[1], actual[1]):
                raise ValueError("Memory cache differs from official dataset")
            if expected[2] != actual[2]:
                raise ValueError("Cache label differs")
        restore_rng(saved)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        values = self.features[index]
        indices = np.random.choice(len(values), self.min_reads, replace=False)
        return torch.from_numpy(values[indices]), self.kmers[index], self.labels[index]


def fixed_validation(model, loader, cfg):
    saved = rng_state()
    try:
        np.random.seed(cfg["validation_seed"])
        random.seed(cfg["validation_seed"])
        torch.manual_seed(cfg["validation_seed"])
        return training_utils.validate(
            model,
            loader,
            "cpu",
            binary_cross_entropy_loss,
            n_iterations=cfg["validation_iterations"],
        )
    finally:
        restore_rng(saved)


def main():
    root = Path("/result")
    cfg = json.loads((root / "config.json").read_text())
    output = root / "official_output"
    output.mkdir()
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    start = time.perf_counter()
    (output / "model_config.toml").write_text(Path(DEFAULT_MODEL_CONFIG).read_text())
    (output / "executed_training_driver.py").write_text(Path(__file__).read_text())
    for name, module in [
        ("training", training_utils),
        ("data", data_utils),
        ("sampler", sampler_utils),
    ]:
        (output / (name + "_source.py")).write_text(inspect.getsource(module))
    print("Fitting official normalization on training sites only", flush=True)
    norm_start = time.perf_counter()
    train_ds = data_utils.NanopolishDS(
        str(root / "adapted_input"), mode="Train", n_processes=cfg["workers"], min_reads=20
    )
    for mean, std in train_ds.norm_dict.values():
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid normalization")
    norm_path = output / "train_norm_dict.joblib"
    joblib.dump(train_ds.norm_dict, norm_path)
    norm_seconds = time.perf_counter() - norm_start
    val_ds = data_utils.NanopolishDS(
        str(root / "adapted_input"), mode="Val", norm_path=str(norm_path), min_reads=20
    )
    if set(train_ds.data_info.set_type) != {"Train"} or set(val_ds.data_info.set_type) != {"Val"}:
        raise ValueError("Unexpected dataset partitions")
    cache_start = time.perf_counter()
    if cfg["cache_reads_in_memory"]:
        print("Caching reads and checking equivalence to official dataset", flush=True)
        train_ds, val_ds = MemoryReads(train_ds), MemoryReads(val_ds)
    cache_seconds = time.perf_counter() - cache_start
    options = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["workers"],
        "collate_fn": data_utils.train_collate,
        "worker_init_fn": seed_worker,
    }
    sampler = sampler_utils.ImbalanceOverSampler(train_ds)
    train_dl = DataLoader(train_ds, sampler=sampler, **options)
    val_dl = DataLoader(val_ds, shuffle=False, **options)
    model = MILModel(toml.load(DEFAULT_MODEL_CONFIG)).to("cpu")
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    stopper = EarlyStopping(cfg["patience"], cfg["min_delta"], cfg["min_epochs"])
    setup = {
        "training_sites": len(train_ds),
        "validation_sites": len(val_ds),
        "sampled_sites_per_epoch": len(sampler),
        "training_batches": len(train_dl),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial_hash,
        "normalization_seconds": norm_seconds,
        "cache_seconds": cache_seconds,
        "cache_equivalence_verified": cfg["cache_reads_in_memory"],
        "normalization_partition": "train",
        "pretrained_weights_loaded": False,
        "test_evaluated": False,
        "fixed_validation_seed": cfg["validation_seed"],
        "threads": cfg["threads"],
        "workers": cfg["workers"],
    }
    write(output / "setup.json", setup)
    print(json.dumps(setup, indent=2), flush=True)
    history = []
    stop_reason = "maximum_epochs"
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        tr = training_utils.train_one_epoch(
            model, train_dl, "cpu", optimizer, binary_cross_entropy_loss
        )
        train_seconds = time.perf_counter() - epoch_start
        if not np.isfinite(tr["avg_loss"]):
            raise ValueError("Non-finite training loss")
        val_start = time.perf_counter()
        va = fixed_validation(model, val_dl, cfg)
        validation_seconds = time.perf_counter() - val_start
        predictions, truth = np.asarray(va.pop("y_pred")), va.pop("y_true")
        improved, stop = stopper.step(epoch, va["avg_loss"])
        row = {
            "epoch": epoch,
            "train_loss": tr["avg_loss"],
            "val_loss": va["avg_loss"],
            "train_roc_auc": tr["roc_auc"],
            "val_roc_auc": va["roc_auc"],
            "val_pr_auc": va["pr_auc"],
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "best_epoch": stopper.best_epoch,
            "bad_epochs": stopper.bad_epochs,
            "checkpoint_improved": improved,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        checkpoint = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            early_stopping=vars(stopper),
            history=history,
            **rng_state(),
        )
        torch.save(checkpoint, output / "last_checkpoint.pt")
        if improved:
            torch.save(checkpoint, output / "checkpoint.pt")
            torch.save(model.state_dict(), output / "model_states.pt")
            frame = val_ds.data_info.copy()
            np.testing.assert_array_equal(frame.modification_status.to_numpy(), truth)
            frame["score"], frame["sampling_std"] = (
                predictions.mean(axis=0),
                predictions.std(axis=0),
            )
            frame.rename(columns={"modification_status": "label", "kmer": "sequence"}).to_csv(
                output / "validation_predictions.csv", index=False
            )
            np.save(output / "validation_repetitions.npy", predictions)
        write(output / "progress.json", dict(row, elapsed_seconds=time.perf_counter() - start))
        print("EPOCH " + json.dumps(row), flush=True)
        if stop:
            stop_reason = "validation_loss_early_stopping"
            break
    # Reload and independently reproduce selected checkpoint predictions with fixed validation samples.
    model.load_state_dict(torch.load(output / "model_states.pt", map_location="cpu"))
    verification = fixed_validation(model, val_dl, cfg)
    np.testing.assert_allclose(
        verification["y_pred"], np.load(output / "validation_repetitions.npy"), rtol=0, atol=0
    )
    if state_hash(model) == initial_hash:
        raise ValueError("Model did not train")
    result = dict(
        setup,
        epochs_completed=len(history),
        best_epoch=stopper.best_epoch,
        best_val_loss=stopper.best_loss,
        stop_reason=stop_reason,
        checkpoint_predictions_reproduced=True,
        train_seconds=sum(r["train_seconds"] for r in history),
        validation_seconds=sum(r["validation_seconds"] for r in history),
        container_work_seconds=time.perf_counter() - start,
        final_state_sha256=state_hash(model),
        main_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        largest_child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    )
    write(output / "timing.json", result)
    print("COMPLETED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
