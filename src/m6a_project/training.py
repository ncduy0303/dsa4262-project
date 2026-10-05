"""One runner used by CLI and notebooks. The read objective is explicitly weak supervision."""

import json
import time
import tomllib
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .cache import prepare_data
from .data import enforce_training_scope, file_hash
from .embeddings import SequenceEmbedding
from .evaluation import metrics, save_diagnostics
from .features import feature_names, read_features
from .models import make_model
from .prediction import ModelBundle, predict_cached
from .tracking import Run, write_json


def load_config(path="configs/experiments/baselines.toml"):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def resolve_config(config, approach=None, model=None):
    config = json.loads(json.dumps(config))
    if approach is not None:
        config["approach"] = approach
    if model is not None:
        config["model"] = model
    feature_names(config["approach"])
    config["estimator_params"] = config.get("model_params", {}).get(config["model"], {})
    config["training_objective"] = (
        "inherited_site_label" if config["approach"].startswith("read") else "site_label"
    )
    config["class_weight_policy"] = "natural_prevalence_equal_total_site_weight"
    config["sampling_policy"] = "sha256_site_seed_without_replacement_v1"
    config["implementation_hashes"] = {
        p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    return config


def run_experiment(config, approach=None, model=None, cache=None):
    config = resolve_config(config, approach, model)
    enforce_training_scope(config)
    with Run(config) as run, threadpool_limits(limits=config["threads"]):
        total_start = time.perf_counter()
        cache = prepare_data(config) if cache is None else cache
        # Supplying a cache is an optimization, not a way to change the experiment's split.
        for name in ("seed", "n_splits", "fold", "n_reads", "repetitions"):
            if cache["identity"][name] != config[name]:
                raise ValueError(f"Cache/config mismatch: {name}")
        if cache["identity"].get("test_fold") != config.get("test_fold"):
            raise ValueError("Cache/config mismatch: test_fold")
        if cache["identity"]["input_hash"] != file_hash(config["input"]) or cache["identity"][
            "labels_hash"
        ] != file_hash(config["labels"]):
            raise ValueError("Cache/config input mismatch")
        write_json(
            run.path / "data_provenance.json",
            {**cache["identity"], "cache_path": str(cache["path"])},
        )
        table = cache["manifest"]
        table[["gene_id", "transcript_id", "transcript_position", "label", "partition"]].to_csv(
            run.path / "split.csv", index=False
        )
        train_ids = np.flatnonzero(table.partition.to_numpy() == "train")
        labels = table.iloc[train_ids].label.to_numpy()
        seq = table.iloc[train_ids].sequence.to_list()
        embedding = SequenceEmbedding() if config["approach"] == "read15_noisy_or" else None
        if embedding:
            write_json(
                run.path / "embedding.json",
                {
                    **embedding.summary(),
                    "lookup": {k: v.tolist() for k, v in embedding.lookup.items()},
                },
            )
        if config["approach"] == "site_mean9":
            x = np.asarray(cache["means"][train_ids])
            y = labels
            weight = np.ones(len(y))
        elif config["approach"] == "site_mean9_matched":
            x = cache["train_reads"].mean(axis=1)
            y = labels
            weight = np.ones(len(y))
        else:
            x = read_features(cache["train_reads"], seq, embedding)
            y = np.repeat(labels, config["n_reads"])
            weight = np.full(len(y), 1.0 / config["n_reads"])
        estimator = make_model(
            config["model"], config["estimator_params"], config["seed"], config["threads"]
        )
        write_json(
            run.path / "feature_schema.json",
            {
                "names": feature_names(config["approach"]),
                "training_rows": len(y),
                "training_sites": len(labels),
                "site_weight": 1,
                "estimator": estimator.get_params(deep=True),
            },
        )
        start = time.perf_counter()
        estimator.fit(x, y, classifier__sample_weight=weight)
        fit_seconds = time.perf_counter() - start
        del x, y, weight
        bundle = ModelBundle(
            estimator,
            config["approach"],
            embedding,
            config,
            {"run_id": run.id, "split_hash": cache["identity"]["split_hash"]},
        )
        start = time.perf_counter()
        predictions = predict_cached(bundle, cache)
        prediction_seconds = time.perf_counter() - start
        predictions.to_parquet(run.path / "validation_predictions.parquet", index=False)
        result = metrics(predictions)
        result.update(fit_seconds=fit_seconds, prediction_seconds=prediction_seconds)
        joblib.dump(bundle, run.path / "model.joblib", compress=3)
        loaded = joblib.load(run.path / "model.joblib")
        reloaded = predict_cached(loaded, cache)
        np.testing.assert_allclose(predictions.score, reloaded.score, rtol=1e-12, atol=1e-12)
        result["save_load_max_abs_difference"] = float(
            np.abs(predictions.score - reloaded.score).max()
        )
        save_diagnostics(predictions, run.path)
        result["total_seconds"] = time.perf_counter() - total_start
        write_json(run.path / "metrics.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return run.path
