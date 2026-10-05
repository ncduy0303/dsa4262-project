"""Feature construction is independent of model choice and labels."""

import numpy as np

from .data import SIGNAL_NAMES

APPROACHES = ("site_mean9", "read9_noisy_or", "read15_noisy_or", "site_mean9_matched")


def feature_names(approach):
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach {approach}")
    return SIGNAL_NAMES + (
        [f"{pos}_embedding_{dim}" for pos in ("minus1", "center", "plus1") for dim in (0, 1)]
        if approach == "read15_noisy_or"
        else []
    )


def mean_features(reads):
    return np.asarray(reads).mean(axis=-2)


def read_features(reads, sequences, embedding=None):
    reads = np.asarray(reads)
    if reads.ndim != 3 or reads.shape[-1] != 9:
        raise ValueError("Expected sites by sampled reads by nine features")
    if embedding is None:
        return reads.reshape(-1, 9)
    seq = embedding.transform(sequences)
    tiled = np.broadcast_to(seq[:, None, :], (*reads.shape[:2], 6))
    return np.concatenate([reads, tiled], axis=-1).reshape(-1, 15)
