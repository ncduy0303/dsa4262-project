"""Sampling is stable across input order, chunk sizes, and Python hash randomization."""

import hashlib

import numpy as np


def sample_indices(key, n_available, n_reads, seed, phase, repetition=0):
    if n_available < n_reads:
        raise ValueError(f"Site {key} has {n_available} reads; requires {n_reads}")
    token = f"{seed}|{phase}|{key[0]}|{key[1]}|{repetition}".encode()
    local_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(local_seed).choice(n_available, n_reads, replace=False)
