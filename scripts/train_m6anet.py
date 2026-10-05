"""Logged CPU m6Anet training with a protected data0 test fold."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from m6a_project.data import enforce_training_scope, file_hash, load_labels
from m6a_project.evaluation import metrics, save_diagnostics
from m6a_project.m6anet_benchmark import prepare_input
from m6a_project.splitting import make_split, split_summary
from m6a_project.tracking import Run, write_json
from m6a_project.training import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/m6anet_full.toml")
    parser.add_argument("--run-config")
    args = parser.parse_args()
    config = (
        json.loads(Path(args.run_config).read_text())
        if args.run_config
        else load_config(args.config)
    )
    enforce_training_scope(config)
    if config.get("test_fold") is None or config["epochs"] < 1:
        raise ValueError("Training requires a test fold and positive epoch count")
    full_training = config["epochs"] > 1
    if full_training and not 1 <= config["min_epochs"] <= config["epochs"]:
        raise ValueError("Minimum epochs must be within the training budget")
    config["docker_image_id"] = subprocess.check_output(
        [
            "docker",
            "image",
            "inspect",
            config.get("docker_image_id", config["image"]),
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    split = make_split(
        load_labels(config["labels"]),
        config["seed"],
        config["n_splits"],
        config["fold"],
        config["test_fold"],
    )
    with Run(config) as run:
        start = time.perf_counter()
        print(f"Run directory: {run.path}", flush=True)
        (run.path / "rerun.txt").write_text(
            f"uv run python scripts/train_m6anet.py --run-config {run.path / 'config.json'}\n"
        )
        shutil.copytree("environments/m6anet", run.path / "source/environments/m6anet")
        write_json(
            run.path / "source_hashes.json",
            {
                str(p.relative_to(run.path / "source")): file_hash(p)
                for p in (run.path / "source").rglob("*")
                if p.is_file()
            },
        )
        split.to_csv(run.path / "split.csv", index=False)
        write_json(run.path / "split_summary.json", split_summary(split))
        adapter_start = time.perf_counter()
        _, provenance = prepare_input(
            config["input"],
            run.path / "split.csv",
            run.path / "adapted_input",
            partitions=("train", "val"),
        )
        provenance.update(
            labels_hash=file_hash(config["labels"]),
            adapter_seconds=time.perf_counter() - adapter_start,
        )
        write_json(run.path / "data_provenance.json", provenance)
        docker = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--cpus",
            str(config["threads"]),
            "--shm-size",
            "1g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"OMP_NUM_THREADS={config['threads']}",
            "-e",
            f"MKL_NUM_THREADS={config['threads']}",
            "-v",
            f"{(run.path / 'source/environments/m6anet').resolve()}:/benchmark:ro",
            "-v",
            f"{run.path.resolve()}:/result",
            config["docker_image_id"],
        ]
        with (run.path / "official_environment.json").open("w") as handle:
            subprocess.run(
                docker + ["python", "/benchmark/provenance.py"], stdout=handle, check=True
            )
        driver = "train_full.py" if full_training else "train_smoke.py"
        command = docker + ["python", "-u", f"/benchmark/{driver}"]
        write_json(run.path / "command.json", command)
        process_start = time.perf_counter()
        with (run.path / "training.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"Training container failed; see {run.path / 'training.log'}")
        container_seconds = time.perf_counter() - process_start
        predictions = pd.read_csv(run.path / "official_output/validation_predictions.csv")
        expected = split.query("partition == 'val'")
        keys = ["transcript_id", "transcript_position"]
        if predictions.duplicated(keys).any() or set(
            map(tuple, predictions[keys].to_numpy())
        ) != set(map(tuple, expected[keys].to_numpy())):
            raise ValueError("Validation predictions do not match frozen split")
        checked = predictions.merge(
            expected[keys + ["label"]], on=keys, validate="one_to_one", suffixes=("", "_expected")
        )
        if not checked.label.equals(checked.label_expected):
            raise ValueError("Validation label mismatch")
        predictions.to_parquet(run.path / "validation_predictions.parquet", index=False)
        timing = json.loads((run.path / "official_output/timing.json").read_text())
        result = metrics(predictions)
        result.update(
            fit_seconds=timing["train_seconds"],
            prediction_seconds=timing["validation_seconds"],
            normalization_seconds=timing["normalization_seconds"],
            container_seconds=container_seconds,
            adapter_seconds=provenance["adapter_seconds"],
            total_seconds=time.perf_counter() - start,
            epochs=timing.get("epochs_completed", 1),
            best_epoch=timing.get("best_epoch", 1),
            test_evaluated=False,
        )
        write_json(run.path / "metrics.json", result)
        save_diagnostics(predictions, run.path)
        write_json(
            config.get("summary_path", "docs/results/m6anet_training_smoke.json"),
            {
                "run_path": str(run.path),
                "split": split_summary(split),
                "metrics": result,
                "timing": timing,
                "test_caveat": "Newly held out from v2 training; used in historical v1 training.",
            },
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
