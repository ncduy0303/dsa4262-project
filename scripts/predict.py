"""Predict from JSON and a trusted local model bundle; no labels required."""

import argparse

from threadpoolctl import threadpool_limits

from m6a_project.prediction import predict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        frame = predict_json(args.model, args.input, args.output, args.batch_size)
    print(f"Validated {len(frame)} site predictions: {args.output}")


if __name__ == "__main__":
    main()
