import json

import numpy as np
import pandas as pd
import pytest

from m6a_project.m6anet_benchmark import evaluate_output, prepare_input


def test_adapter_only_validation_and_exact_signals(tmp_path):
    original = tmp_path / "original.json"
    signal = np.arange(180, dtype=float).reshape(20, 9) / 100
    records = [
        {"train_tx": {"1": {"AAGACCA": signal.tolist()}}},
        {"val_tx": {"2": {"AAGACCA": signal.tolist()}}},
    ]
    original.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    split = pd.DataFrame(
        [("g0", "train_tx", 1, 0, "train"), ("g1", "val_tx", 2, 1, "val")],
        columns=["gene_id", "transcript_id", "transcript_position", "label", "partition"],
    )
    split.to_csv(tmp_path / "split.csv", index=False)
    before = original.read_bytes()
    info, manifest = prepare_input(original, tmp_path / "split.csv", tmp_path / "adapted")
    assert len(info) == 1 and manifest["sites"] == 1
    assert original.read_bytes() == before
    values = json.loads((tmp_path / "adapted/data.json").read_text())["val_tx"]["2"]["AAGACCA"]
    np.testing.assert_array_equal(np.array(values)[:, :9], signal)
    np.testing.assert_array_equal(np.array(values)[:, 9], np.arange(20))
    assert "label" not in pd.read_csv(tmp_path / "adapted/data.info")
    with pytest.raises(FileExistsError):
        prepare_input(original, tmp_path / "split.csv", tmp_path / "adapted")


def test_benchmark_uses_site_probability_and_exact_coverage(tmp_path):
    baseline = pd.DataFrame(
        {
            "gene_id": ["g0", "g1"],
            "transcript_id": ["tx0", "tx1"],
            "transcript_position": [1, 2],
            "label": [0, 1],
            "partition": ["val", "val"],
            "sequence": ["AAGACCA", "AAGACCA"],
            "n_reads": [20, 30],
            "score": [0.1, 0.8],
            "sampling_std": [0.0, 0.0],
        }
    )
    baseline.to_parquet(tmp_path / "baseline.parquet", index=False)
    raw = pd.DataFrame(
        {
            "transcript_id": ["tx1", "tx0"],
            "transcript_position": [2, 1],
            "n_reads": [30, 20],
            "probability_modified": [0.9, 0.2],
            "kmer": ["AGACC", "AGACC"],
            "mod_ratio": [0.01, 0.99],
        }
    )
    raw.to_csv(tmp_path / "official.csv", index=False)
    output, result = evaluate_output(tmp_path / "official.csv", tmp_path / "baseline.parquet")
    np.testing.assert_array_equal(output.score, [0.2, 0.9])
    assert result["prediction_coverage"] == 1
    assert result["average_precision"] == 1
    assert result["mean_sampling_std"] is None
    raw.iloc[:1].to_csv(tmp_path / "partial.csv", index=False)
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_output(tmp_path / "partial.csv", tmp_path / "baseline.parquet")
    pd.concat([raw, raw]).to_csv(tmp_path / "duplicate.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_output(tmp_path / "duplicate.csv", tmp_path / "baseline.parquet")


def test_training_adapter_never_exports_test(tmp_path):
    signal = np.arange(180, dtype=float).reshape(20, 9).tolist()
    records = [{name: {"1": {"AAGACCA": signal}}} for name in ("train", "val", "test")]
    original = tmp_path / "original.json"
    original.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    split = pd.DataFrame(
        [(name, name, 1, i % 2, name) for i, name in enumerate(("train", "val", "test"))],
        columns=["gene_id", "transcript_id", "transcript_position", "label", "partition"],
    )
    split.to_csv(tmp_path / "split.csv", index=False)
    info, _ = prepare_input(
        original, tmp_path / "split.csv", tmp_path / "adapted", partitions=("train", "val")
    )
    assert set(info.transcript_id) == {"train", "val"}
    labelled = pd.read_csv(tmp_path / "adapted/data.info.labelled")
    assert set(labelled.set_type) == {"Train", "Val"}
    assert set(labelled.kmer) == {"AAGACCA"}
    assert '"test"' not in (tmp_path / "adapted/data.json").read_text()
    with pytest.raises(ValueError, match="only train and validation"):
        prepare_input(
            original, tmp_path / "split.csv", tmp_path / "forbidden", partitions=("test",)
        )
