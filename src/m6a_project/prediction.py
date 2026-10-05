"""Complete model bundle and label-free JSON prediction."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import file_hash, iter_sites
from .features import read_features
from .models import positive_probability
from .pooling import noisy_or
from .sampling import sample_indices


@dataclass
class ModelBundle:
    model: object
    approach: str
    embedding: object
    config: dict
    provenance: dict
    version: int = 1

    def predict_arrays(self, means, sampled_reads, sequences):
        if self.approach == "site_mean9":
            scores = positive_probability(self.model, means)
            return scores, np.zeros(len(scores))
        n_sites, repetitions, n_reads, _ = sampled_reads.shape
        outputs = []
        for rep in range(repetitions):
            reads = sampled_reads[:, rep]
            if self.approach == "site_mean9_matched":
                score = positive_probability(self.model, reads.mean(axis=1))
            else:
                features = read_features(reads, sequences, self.embedding)
                p = positive_probability(self.model, features).reshape(n_sites, n_reads)
                score = noisy_or(p)
            outputs.append(score)
        outputs = np.stack(outputs, axis=1)
        return outputs.mean(axis=1), outputs.std(axis=1)


def predict_cached(bundle, cache, batch_size=512):
    manifest = cache["manifest"]
    ids = np.flatnonzero(manifest.partition.to_numpy() == "val")
    scores, variation = [], []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        score, std = bundle.predict_arrays(
            cache["means"][batch],
            cache["val_reads"][start : start + batch_size],
            manifest.iloc[batch].sequence.to_list(),
        )
        scores.extend(score)
        variation.extend(std)
    result = manifest.iloc[ids].copy()
    result["score"], result["sampling_std"] = scores, variation
    return result


def validate_submission(frame, expected_keys):
    columns = ["transcript_id", "transcript_position", "score"]
    if list(frame.columns) != columns:
        raise ValueError(f"Submission columns must be exactly {columns}")
    if frame.isna().any().any() or frame.duplicated(columns[:2]).any():
        raise ValueError("Missing values or duplicate submission keys")
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
        raise ValueError("Invalid probability")
    if (
        not np.isfinite(frame.transcript_position).all()
        or (frame.transcript_position % 1 != 0).any()
    ):
        raise ValueError("Invalid position")
    keys = list(zip(frame.transcript_id, frame.transcript_position.astype(int), strict=True))
    if keys != list(expected_keys):
        raise ValueError("Submission keys/order do not match input")


def predict_json(model_path, input_path, output_path, batch_size=512):
    """Use only trusted local joblib bundles. No metadata or labels are required."""
    bundle = joblib.load(model_path)
    if bundle.version != 1:
        raise ValueError("Unsupported model bundle version")
    config = bundle.config
    rows, expected, batch = [], [], []

    def flush(sites):
        means = np.stack([s.reads.mean(axis=0) for s in sites])
        if bundle.approach == "site_mean9":
            sampled = None
        else:
            sampled = np.stack(
                [
                    np.stack(
                        [
                            s.reads[
                                sample_indices(
                                    s.key,
                                    len(s.reads),
                                    config["n_reads"],
                                    config["seed"],
                                    "predict",
                                    rep,
                                )
                            ]
                            for rep in range(config["repetitions"])
                        ]
                    )
                    for s in sites
                ]
            )
        scores, _ = bundle.predict_arrays(means, sampled, [s.sequence for s in sites])
        rows.extend(
            (s.transcript_id, s.transcript_position, float(score))
            for s, score in zip(sites, scores, strict=True)
        )

    for site in iter_sites(input_path):
        expected.append(site.key)
        batch.append(site)
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    if not rows:
        raise ValueError("Empty prediction input")
    frame = pd.DataFrame(rows, columns=["transcript_id", "transcript_position", "score"])
    validate_submission(frame, expected)
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    import json

    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "model_sha256": file_hash(model_path),
                "input_sha256": file_hash(input_path),
                "csv_sha256": file_hash(path),
                "sites": len(frame),
                "config": config,
            },
            indent=2,
        )
    )
    return frame
