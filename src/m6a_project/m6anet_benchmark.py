"""Official pretrained m6Anet comparison on an existing validation manifest."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_hash, iter_sites
from .evaluation import metrics


def prepare_input(input_path, split_path, output_dir, partitions=("val",)):
    if not partitions or not set(partitions) <= {"train", "val"}:
        raise ValueError("Benchmark adapter permits only train and validation")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    split = pd.read_csv(split_path)
    required = {"gene_id", "transcript_id", "transcript_position", "label", "partition"}
    if not required <= set(split):
        raise ValueError("Expected a saved labelled train/validation manifest")
    validation = split[split.partition.isin(partitions)].copy()
    if validation.empty or validation.duplicated(["transcript_id", "transcript_position"]).any():
        raise ValueError("Invalid validation manifest")
    keys = set(zip(validation.transcript_id, validation.transcript_position, strict=True))
    seen = set()
    rows = []
    with (output / "data.json").open("wb") as handle:
        for site in iter_sites(input_path):
            if site.key not in keys:
                continue
            if len(site.reads) < 20:
                raise ValueError(f"Site {site.key} does not meet m6Anet minimum coverage")
            values = [list(read) + [idx] for idx, read in enumerate(site.reads.tolist())]
            start = handle.tell()
            raw = (
                json.dumps(
                    {site.transcript_id: {str(site.transcript_position): {site.sequence: values}}},
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            handle.write(raw)
            # Verify serialization preserves signal values before recording the adapter success.
            restored = np.asarray(
                json.loads(raw)[site.transcript_id][str(site.transcript_position)][site.sequence]
            )
            np.testing.assert_array_equal(restored[:, :9], site.reads)
            np.testing.assert_array_equal(restored[:, 9], np.arange(len(site.reads)))
            rows.append(
                {
                    "transcript_id": site.transcript_id,
                    "transcript_position": site.transcript_position,
                    "n_reads": len(site.reads),
                    "start": start,
                    "end": handle.tell(),
                    "sequence": site.sequence,
                }
            )
            seen.add(site.key)
    if seen != keys:
        raise ValueError(f"Missing {len(keys - seen)} selected sites in original input")
    info = pd.DataFrame(rows)
    info.drop(columns="sequence").to_csv(output / "data.info", index=False)
    labelled = info.merge(
        validation, on=["transcript_id", "transcript_position"], validate="one_to_one", sort=False
    )
    labelled["set_type"] = labelled.partition.map({"train": "Train", "val": "Val"})
    labelled.rename(columns={"label": "modification_status", "sequence": "kmer"}).to_csv(
        output / "data.info.labelled", index=False
    )
    # Independently seek every offset in the generated file, checking both coordinates and widths.
    with (output / "data.json").open("rb") as handle:
        for row in info.itertuples():
            handle.seek(row.start)
            obj = json.loads(handle.read(row.end - row.start))
            reads = obj[row.transcript_id][str(row.transcript_position)][row.sequence]
            if len(reads) != row.n_reads or any(len(read) != 10 for read in reads):
                raise ValueError("Adapter index round-trip mismatch")
    manifest = {
        "source_sha256": file_hash(input_path),
        "split_sha256": file_hash(split_path),
        "data_json_sha256": file_hash(output / "data.json"),
        "data_info_sha256": file_hash(output / "data.info"),
        "sites": len(info),
        "partitions": list(partitions),
        "labelled_info_sha256": file_hash(output / "data.info.labelled"),
        "reads": int(info.n_reads.sum()),
        "synthetic_read_ids": "zero-based, local to each site",
        "signal_round_trip": "exact equality for every signal value",
        "schema": "9 original signal columns followed by synthetic read index",
    }
    (output / "adapter.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return info, manifest


def evaluate_output(raw_path, baseline_path):
    """Require full, one-to-one coverage. Never silently evaluate a smaller subset."""
    raw = pd.read_csv(raw_path)
    keys = ["transcript_id", "transcript_position"]
    required = {*keys, "n_reads", "probability_modified", "kmer"}
    if not required <= set(raw):
        raise ValueError(f"Missing official output columns: {required - set(raw)}")
    if raw.duplicated(keys).any():
        raise ValueError("Duplicate m6Anet site predictions")
    expected = pd.read_parquet(baseline_path)
    expected_keys = set(map(tuple, expected[keys].to_numpy()))
    actual_keys = set(map(tuple, raw[keys].to_numpy()))
    if actual_keys != expected_keys:
        raise ValueError(
            f"m6Anet coverage mismatch: missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}"
        )
    merged = expected.drop(columns=["score", "sampling_std"]).merge(
        raw[keys + ["n_reads", "probability_modified", "kmer"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_m6anet"),
        sort=False,
    )
    if not (merged.n_reads == merged.n_reads_m6anet).all():
        raise ValueError("Read count mismatch")
    if not (merged.sequence.str[1:6] == merged.kmer).all():
        raise ValueError("Central motif mismatch")
    merged["score"] = merged.probability_modified
    if not np.isfinite(merged.score).all() or not merged.score.between(0, 1).all():
        raise ValueError("Invalid m6Anet probability")
    # Official inference returns the average, not the individual sampling repetitions.
    merged["sampling_std"] = np.nan
    result = metrics(merged.assign(sampling_std=0.0))
    result["mean_sampling_std"] = None
    result["prediction_coverage"] = len(merged) / len(expected)
    return merged, result
