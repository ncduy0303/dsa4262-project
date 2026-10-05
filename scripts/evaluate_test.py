"""Explicit held-out evaluation of a frozen v2 classical model, logged separately."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from m6a_project.cache import export_partition_json, prepare_data
from m6a_project.evaluation import metrics, save_diagnostics
from m6a_project.prediction import predict_json
from m6a_project.tracking import Run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted, selected v2 model.joblib")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if bundle.config.get("test_fold") is None:
        raise ValueError("Historical two-way models cannot evaluate the new test fold")
    cache = prepare_data(bundle.config)
    if bundle.provenance["split_hash"] != cache["identity"]["split_hash"]:
        raise ValueError("Model and held-out split do not match")
    if any(
        bundle.config["implementation_hashes"].get(name) != value
        for name, value in cache["identity"]["source_hashes"].items()
    ):
        raise ValueError("Preprocessing changed since training; use the saved source snapshot")
    training_provenance = json.loads(Path(args.model).with_name("data_provenance.json").read_text())
    for key in ("input_hash", "labels_hash", "split_hash"):
        if cache["identity"][key] != training_provenance[key]:
            raise ValueError(f"Training/evaluation data changed: {key}")
    config = dict(
        bundle.config,
        experiment_set=bundle.config["experiment_set"] + "_test",
        selected_model=str(Path(args.model).resolve()),
        evaluation_partition="test",
    )
    with Run(config) as run:
        (run.path / "rerun.txt").write_text(
            f"uv run python scripts/evaluate_test.py --model {Path(args.model).resolve()}\n"
        )
        export_partition_json(config, cache, run.path / "test.json", "test")
        predictions = predict_json(
            args.model, run.path / "test.json", run.path / "test_predictions.csv"
        )
        expected = cache["manifest"].query("partition == 'test'")
        frame = expected.merge(
            predictions,
            on=["transcript_id", "transcript_position"],
            validate="one_to_one",
            how="left",
        )
        if len(predictions) != len(expected) or not np.isfinite(frame.score).all():
            raise ValueError("Incomplete held-out predictions")
        # Generic predictor averages repetitions but does not return their dispersion.
        frame["sampling_std"] = 0.0
        result = metrics(frame)
        result["mean_sampling_std"] = None
        result["evaluation_partition"] = "test"
        result["split_hash"] = cache["identity"]["split_hash"]
        write_json(run.path / "data_provenance.json", cache["identity"])
        write_json(run.path / "metrics.json", result)
        save_diagnostics(frame, run.path, partition_label="Test")
        frame.drop(columns="sampling_std").to_parquet(
            run.path / "test_predictions.parquet", index=False
        )


if __name__ == "__main__":
    main()
