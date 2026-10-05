"""Run the official pretrained model in its isolated conda container and compare site scores."""

import argparse
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

import pandas as pd

from m6a_project.data import file_hash
from m6a_project.evaluation import save_diagnostics
from m6a_project.m6anet_benchmark import evaluate_output, prepare_input
from m6a_project.tracking import Run, write_json


def run_command(command, output, stderr):
    with Path(output).open("w") as out, Path(stderr).open("w") as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/m6anet.toml")
    parser.add_argument("--run-config")
    args = parser.parse_args()
    if args.run_config:
        config = json.loads(Path(args.run_config).read_text())
    else:
        with open(args.config, "rb") as handle:
            config = tomllib.load(handle)
    root = Path.cwd().resolve()
    input_path = Path(config["input"]).resolve()
    if input_path.parent != root / "data/data0":
        raise ValueError("This benchmark is authorized for data0 only")
    summary = json.loads(Path(config["baseline_summary"]).read_text())
    baseline = Path(config.get("baseline_run", summary["best_validation_run"]))
    if (
        config.get("frozen_split_sha256")
        and file_hash(baseline / "split.csv") != config["frozen_split_sha256"]
    ):
        raise ValueError("Saved benchmark split has changed")
    baseline_provenance = json.loads((baseline / "data_provenance.json").read_text())
    if file_hash(input_path) != baseline_provenance["input_hash"]:
        raise ValueError("Data0 changed since the frozen baseline experiment")
    config["baseline_run"] = str(baseline)
    config["frozen_split_sha256"] = file_hash(baseline / "split.csv")
    with Run(config) as run:
        start = time.perf_counter()
        (run.path / "rerun.txt").write_text(
            f"uv run python scripts/benchmark_m6anet.py --run-config {run.path / 'config.json'}\n"
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
        shutil.copy2(baseline / "split.csv", run.path / "split.csv")
        docker_id = subprocess.check_output(
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
        config["docker_image_id"] = docker_id
        write_json(run.path / "config.json", config)
        adapted = run.path / "adapted_input"
        _, adapter_info = prepare_input(input_path, baseline / "split.csv", adapted)
        write_json(
            run.path / "data_provenance.json", {**adapter_info, "baseline_run": str(baseline)}
        )
        docker_base = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--cpus",
            str(config["threads"]),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"OMP_NUM_THREADS={config['threads']}",
            "-e",
            f"MKL_NUM_THREADS={config['threads']}",
            "-v",
            f"{(root / 'environments/m6anet').resolve()}:/benchmark:ro",
            "-v",
            f"{run.path.resolve()}:/result",
            docker_id,
        ]
        # Only the derived validation data and this run directory are mounted, not the other datasets.
        run_command(
            docker_base + ["python", "/benchmark/provenance.py"],
            run.path / "official_environment.json",
            run.path / "provenance_stderr.log",
        )
        provenance = json.loads((run.path / "official_environment.json").read_text())
        (run.path / "conda-explicit.txt").write_text(provenance["conda_explicit"])
        output = run.path / "official_output"
        command = docker_base + [
            "m6anet",
            "inference",
            "--input_dir",
            "/result/adapted_input",
            "--out_dir",
            "/result/official_output",
            "--pretrained_model",
            config["pretrained_model"],
            "--device",
            "cpu",
            "--seed",
            str(config["seed"]),
            "--n_processes",
            str(config["n_processes"]),
            "--batch_size",
            str(config["batch_size"]),
            "--save_per_batch",
            str(config["save_per_batch"]),
            "--num_iterations",
            str(config["num_iterations"]),
        ]
        write_json(run.path / "command.json", command)
        print(f"Running official m6Anet on {adapter_info['sites']} validation sites", flush=True)
        inference_start = time.perf_counter()
        run_command(command, run.path / "inference_stdout.log", run.path / "inference_stderr.log")
        inference_seconds = time.perf_counter() - inference_start
        predictions, result = evaluate_output(
            output / "data.site_proba.csv", baseline / "validation_predictions.parquet"
        )
        predictions.to_parquet(run.path / "validation_predictions.parquet", index=False)
        predictions[["transcript_id", "transcript_position", "score"]].to_csv(
            run.path / "validation_predictions.csv", index=False
        )
        result.update(
            prediction_seconds=inference_seconds,
            total_seconds=time.perf_counter() - start,
            sampling_iterations=config["num_iterations"],
            independent_test_claim=False,
        )
        write_json(run.path / "metrics.json", result)
        save_diagnostics(predictions.assign(sampling_std=0.0), run.path)
        baselines = pd.read_csv("docs/results/comparison.csv")
        benchmark_row = {
            **result,
            "approach": config["approach"],
            "model": config["model"],
            "run_path": str(run.path),
        }
        combined = pd.concat([baselines, pd.DataFrame([benchmark_row])], ignore_index=True)
        combined.to_csv("docs/results/comparison_with_m6anet.csv", index=False)
        write_json(
            "docs/results/m6anet_summary.json",
            {
                "run_path": str(run.path),
                "metrics": result,
                "checkpoint": config["pretrained_model"],
                "independence": config["training_overlap"],
            },
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
