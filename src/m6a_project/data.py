"""Strict site records; only the first nine columns are accepted as signals."""

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_NAMES = [
    f"{pos}_{kind}"
    for pos in ("minus1", "center", "plus1")
    for kind in ("dwell", "signal_std", "current_mean")
]


@dataclass
class Site:
    transcript_id: str
    transcript_position: int
    sequence: str
    reads: np.ndarray

    @property
    def key(self):
        return self.transcript_id, self.transcript_position


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_sites(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    seen = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if len(record) != 1:
                raise ValueError(f"Line {line_number}: expected exactly one transcript")
            tx, positions = next(iter(record.items()))
            if len(positions) != 1:
                raise ValueError(f"Line {line_number}: expected exactly one site")
            pos, contexts = next(iter(positions.items()))
            if len(contexts) != 1:
                raise ValueError(f"Line {line_number}: expected exactly one sequence")
            sequence, values = next(iter(contexts.items()))
            sequence = sequence.upper().replace("U", "T")
            if len(sequence) != 7 or set(sequence) - set("ACGT"):
                raise ValueError(f"Line {line_number}: invalid seven-base context")
            reads = np.asarray(values, dtype=np.float64)
            if reads.ndim != 2 or reads.shape[1] != 9 or not len(reads):
                raise ValueError(
                    f"Line {line_number}: expected nonempty read matrix with 9 columns"
                )
            if not np.isfinite(reads).all():
                raise ValueError(f"Line {line_number}: nonfinite signals")
            position = int(pos)
            if str(position) != str(pos) or position < 0:
                raise ValueError(f"Noncanonical transcript position: {pos}")
            site = Site(str(tx), position, sequence, reads)
            if site.key in seen:
                raise ValueError(f"Duplicate site: {site.key}")
            seen.add(site.key)
            yield site


def load_labels(path):
    table = pd.read_csv(path, dtype={"transcript_id": str, "gene_id": str})
    columns = ["gene_id", "transcript_id", "transcript_position", "label"]
    if not set(columns) <= set(table):
        raise ValueError(f"Missing required label columns: {columns}")
    table = table[columns].copy()
    if table.isna().any().any() or not table.label.isin([0, 1]).all():
        raise ValueError("Labels must be complete binary site labels")
    if (table.transcript_position % 1 != 0).any() or (table.transcript_position < 0).any():
        raise ValueError("Positions must be nonnegative integers")
    table["transcript_position"] = table.transcript_position.astype(int)
    table["label"] = table.label.astype(int)
    if table.duplicated(["transcript_id", "transcript_position"]).any():
        raise ValueError("Duplicate labelled site")
    if table.groupby("transcript_id").gene_id.nunique().max() != 1:
        raise ValueError("A transcript maps to multiple genes")
    return table


def enforce_training_scope(config):
    """Guard the current authorized data0-only experiment runner, not generic prediction."""
    for field in ("input", "labels"):
        path = Path(config[field]).resolve()
        if path.parent.name != "data0" or path.parent.parent.name != "data":
            raise ValueError(f"Current experiment scope permits data0 only: {field}")
