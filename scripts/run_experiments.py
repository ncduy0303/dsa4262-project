"""Run the fixed comparison; failures remain logged and are returned as a nonzero exit."""

import argparse
import json
from pathlib import Path

from m6a_project.cache import prepare_data
from m6a_project.data import file_hash
from m6a_project.tracking import rebuild_index
from m6a_project.training import load_config, resolve_config, run_experiment


def already_completed(config):
    root = Path(config["output_dir"]) / config["experiment_set"]
    for path in root.glob("*/config.json"):
        if json.loads(path.read_text()) == config:
            status = json.loads((path.parent / "status.json").read_text())
            if status["status"] == "completed":
                provenance = json.loads((path.parent / "data_provenance.json").read_text())
                if provenance["input_hash"] == file_hash(config["input"]) and provenance[
                    "labels_hash"
                ] == file_hash(config["labels"]):
                    return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/baselines.toml")
    parser.add_argument("--resume", action="store_true", help="Skip completed exact configurations")
    args = parser.parse_args()
    config = load_config(args.config)
    jobs = [("site_mean9", "dummy")] if config.get("include_dummy") else []
    jobs += [(approach, model) for model in config["models"] for approach in config["approaches"]]
    if config.get("include_matched_control"):
        jobs += [("site_mean9_matched", model) for model in config["models"]]
    cache = None
    errors = []
    for approach, model in jobs:
        resolved = resolve_config(config, approach, model)
        if args.resume and already_completed(resolved):
            print(f"Already completed {approach}/{model}", flush=True)
            continue
        try:
            run_experiment(resolved, cache=cache)
            if cache is None:
                cache = prepare_data(config)
        except Exception as exc:  # noqa: BLE001 - log each failed experiment and continue the matrix
            errors.append(f"{approach}/{model}: {exc}")
            print(errors[-1], flush=True)
    rebuild_index(config["output_dir"])
    if errors:
        raise SystemExit("Failed runs:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
