"""Verify standalone JSON predictions against logged data0 validation predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from m6a_project.cache import export_validation_json, prepare_data
from m6a_project.prediction import predict_json
from m6a_project.training import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/baselines.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    cache = prepare_data(config)
    directory = Path("artifacts/validation")
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "data0_validation.json"
    export_validation_json(config, cache, source)
    index = pd.read_csv("experiments/index.csv")
    completed = index[
        (index.status == "completed") & (index.experiment_set == config["experiment_set"])
    ]
    records = []
    # Cover each requested approach through its best measured estimator.
    for approach in config["approaches"]:
        selected = (
            completed[completed.approach == approach].sort_values("average_precision").iloc[-1]
        )
        run = Path(selected.run_path)
        output = directory / f"{approach}_predictions.csv"
        with threadpool_limits(limits=config["threads"]):
            if output.exists():
                raise FileExistsError(f"Verification output exists: {output}")
            predicted = predict_json(run / "model.joblib", source, output)
        expected = pd.read_parquet(run / "validation_predictions.parquet")
        joined = predicted.merge(
            expected[["transcript_id", "transcript_position", "score"]],
            on=["transcript_id", "transcript_position"],
            validate="one_to_one",
            suffixes=("_json", "_cached"),
        )
        assert len(joined) == len(expected) == len(predicted)
        np.testing.assert_allclose(joined.score_json, joined.score_cached, rtol=1e-12, atol=1e-12)
        records.append(
            {
                "approach": approach,
                "model": selected.model,
                "sites": len(predicted),
                "max_abs_difference": float(np.abs(joined.score_json - joined.score_cached).max()),
                "output": str(output),
                "model_path": str(run / "model.joblib"),
            }
        )
        print(records[-1], flush=True)
    Path("docs/results/prediction_verification.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
