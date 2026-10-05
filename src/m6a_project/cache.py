"""Versioned derived arrays. Never use validation to fit a transformation."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import enforce_training_scope, file_hash, iter_sites, load_labels
from .sampling import sample_indices
from .splitting import make_split, split_summary

CACHE_VERSION = 2


def prepare_data(config):
    enforce_training_scope(config)
    identity = {k: config[k] for k in ("seed", "n_splits", "fold", "n_reads", "repetitions")}
    identity["test_fold"] = config.get("test_fold")
    identity.update(
        version=CACHE_VERSION,
        input_hash=file_hash(config["input"]),
        labels_hash=file_hash(config["labels"]),
    )
    identity["source_hashes"] = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ("cache.py", "data.py", "sampling.py", "splitting.py")
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(config["cache_dir"]) / key
    if (path / "complete.json").exists():
        return load_cache(path)
    path.mkdir(parents=True, exist_ok=True)
    # No completed cache is overwritten. Incomplete construction is recoverable.
    split = make_split(
        load_labels(config["labels"]),
        config["seed"],
        config["n_splits"],
        config["fold"],
        config.get("test_fold"),
    )
    manifest = split.copy()
    manifest["sequence"] = ""
    manifest["n_reads"] = 0
    manifest["json_start"] = 0
    mapping = {(r.transcript_id, r.transcript_position): i for i, r in manifest.iterrows()}
    train_ids = np.flatnonzero(manifest.partition.to_numpy() == "train")
    val_ids = np.flatnonzero(manifest.partition.to_numpy() == "val")
    train_map = {idx: i for i, idx in enumerate(train_ids)}
    val_map = {idx: i for i, idx in enumerate(val_ids)}
    k, repeats = config["n_reads"], config["repetitions"]

    def array(name, shape, dtype="float64"):
        return np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    means = array("means", (len(manifest), 9))
    train = array("train_reads", (len(train_ids), k, 9))
    val = array("val_reads", (len(val_ids), repeats, k, 9))
    train_indices = array("train_indices", (len(train_ids), k), "int32")
    val_indices = array("val_indices", (len(val_ids), repeats, k), "int32")
    seen = set()
    for site in iter_sites(config["input"]):
        if site.key not in mapping:
            raise ValueError(f"JSON site has no label: {site.key}")
        idx = mapping[site.key]
        seen.add(idx)
        manifest.at[idx, "sequence"] = site.sequence
        manifest.at[idx, "n_reads"] = len(site.reads)
        means[idx] = site.reads.mean(axis=0)
        if idx in train_map:
            j = train_map[idx]
            ix = sample_indices(site.key, len(site.reads), k, config["seed"], "train")
            train[j], train_indices[j] = site.reads[ix], ix
        elif idx in val_map:
            j = val_map[idx]
            for rep in range(repeats):
                ix = sample_indices(site.key, len(site.reads), k, config["seed"], "predict", rep)
                val[j, rep], val_indices[j, rep] = site.reads[ix], ix
        if len(seen) % 20000 == 0:
            print(f"Cached {len(seen)} sites", flush=True)
    if len(seen) != len(manifest):
        raise ValueError("Some labelled sites are missing from JSON")
    manifest = manifest.drop(columns="json_start")
    manifest.to_parquet(path / "manifest.parquet", index=False)
    manifest[["gene_id", "transcript_id", "transcript_position", "label", "partition"]].to_csv(
        path / "split.csv", index=False
    )
    for value in (means, train, val, train_indices, val_indices):
        value.flush()
    identity["split_hash"] = file_hash(path / "split.csv")
    identity["summary"] = split_summary(manifest)
    identity["array_hashes"] = {p.name: file_hash(p) for p in path.glob("*.npy")}
    (path / "complete.json").write_text(json.dumps(identity, indent=2))
    return load_cache(path)


def load_cache(path):
    path = Path(path)
    result = {
        "path": path,
        "manifest": pd.read_parquet(path / "manifest.parquet"),
        "identity": json.loads((path / "complete.json").read_text()),
    }
    for name in ("means", "train_reads", "val_reads"):
        result[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    return result


def export_partition_json(config, cache, output, partition="val"):
    if partition not in {"val", "test"}:
        raise ValueError("Export requires val or test partition")
    enforce_training_scope(config)
    import gzip

    table = cache["manifest"]
    table = table[table.partition == partition]
    if table.empty:
        raise ValueError(f"No {partition} partition in this split")
    keys = set(zip(table.transcript_id, table.transcript_position, strict=True))
    opener = gzip.open if str(config["input"]).endswith(".gz") else open
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with opener(config["input"], "rt") as source, output.open("w") as target:
        for line in source:
            obj = json.loads(line)
            tx = next(iter(obj))
            pos = next(iter(obj[tx]))
            if (tx, int(pos)) in keys:
                target.write(line)
    shutil.copy2(cache["path"] / "split.csv", output.parent / "split.csv")


def export_validation_json(config, cache, output):
    """Backward-compatible validation export for existing notebooks."""
    return export_partition_json(config, cache, output, "val")
