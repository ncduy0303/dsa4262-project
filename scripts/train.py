"""Train one immutable, logged experiment."""

import argparse
import json

from m6a_project.training import load_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/baselines.toml")
    parser.add_argument("--run-config", help="Resolved JSON from an existing run")
    parser.add_argument("--approach", default="site_mean9")
    parser.add_argument("--model", default="logistic")
    args = parser.parse_args()
    if args.run_config:
        with open(args.run_config) as handle:
            config = json.load(handle)
        result = run_experiment(config)
    else:
        result = run_experiment(load_config(args.config), args.approach, args.model)
    print(result)


if __name__ == "__main__":
    main()
