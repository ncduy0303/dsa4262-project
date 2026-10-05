import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from m6a_project.data import enforce_training_scope, iter_sites
from m6a_project.embeddings import SequenceEmbedding
from m6a_project.features import mean_features, read_features
from m6a_project.models import make_model
from m6a_project.pooling import noisy_or
from m6a_project.prediction import ModelBundle, predict_json, validate_submission
from m6a_project.sampling import sample_indices
from m6a_project.splitting import assert_disjoint, make_split


def test_parser_plain_and_gzip(tmp_path):
    record = {
        "tx": {"1": {"AAGACCA": [[1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9, 10, 11]]}}
    }
    text = json.dumps(record) + "\n"
    for suffix in (".json", ".json.gz"):
        path = tmp_path / ("input" + suffix)
        if suffix.endswith("gz"):
            with gzip.open(path, "wt") as handle:
                handle.write(text)
        else:
            path.write_text(text)
        site = next(iter_sites(path))
        assert site.key == ("tx", 1)
        np.testing.assert_array_equal(mean_features(site.reads), np.arange(2, 11))
    path = tmp_path / "duplicate.json"
    path.write_text(text + text)
    with pytest.raises(ValueError, match="Duplicate"):
        list(iter_sites(path))


def test_gene_split_and_invalid_grouping():
    table = pd.DataFrame(
        [(f"g{i}", f"tx{i}_{j}", j, j % 2) for i in range(30) for j in range(4)],
        columns=["gene_id", "transcript_id", "transcript_position", "label"],
    )
    split = make_split(table)
    assert_disjoint(split)
    pd.testing.assert_frame_equal(split, make_split(table))
    leaked = split.copy()
    gene = leaked.loc[0, "gene_id"]
    rows = leaked.index[leaked.gene_id == gene]
    leaked.loc[rows[0], "partition"] = "val"
    leaked.loc[rows[1], "partition"] = "train"
    with pytest.raises(ValueError, match="leaks"):
        assert_disjoint(leaked)


def test_sampling_is_stable_and_without_replacement():
    x = sample_indices(("t", 1), 100, 20, 42, "predict", 0)
    np.testing.assert_array_equal(x, sample_indices(("t", 1), 100, 20, 42, "predict", 0))
    assert len(set(x)) == 20
    assert not np.array_equal(x, sample_indices(("t", 1), 100, 20, 42, "predict", 1))
    with pytest.raises(ValueError, match="requires"):
        sample_indices(("t", 1), 19, 20, 42, "predict")


def test_pooling_endpoints_and_stability():
    assert noisy_or([0, 0]) == 0
    assert noisy_or([0, 1]) == 1
    assert noisy_or([0.2, 0.3]) == pytest.approx(0.44)
    assert noisy_or(np.full(1000, 1e-15)) > 0
    with pytest.raises(ValueError):
        noisy_or([np.nan])
    with pytest.raises(ValueError):
        noisy_or([])


def test_embedding_and_signal_alignment():
    embedding = SequenceEmbedding()
    assert len(embedding.motifs) == 66
    reads = np.arange(2 * 20 * 9).reshape(2, 20, 9)
    seq = ["AAGACCA", "AAGACCA"]
    output = read_features(reads, seq, embedding)
    assert output.shape == (40, 15)
    np.testing.assert_array_equal(output[:, :9], reads.reshape(-1, 9))
    np.testing.assert_allclose(
        output[0, 9:], np.concatenate([embedding.lookup[k] for k in ["AAGAC", "AGACC", "GACCA"]])
    )
    np.testing.assert_allclose(output[0, 9:], output[-1, 9:])
    with pytest.raises(ValueError):
        embedding.transform(["CCCCCCC"])


def test_model_scaler_fitted_only_on_training():
    train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    model = make_model("logistic", {"C": 1.0})
    model.fit(train, [0, 0, 1, 1])
    before = model.named_steps["scale"].mean_.copy()
    model.predict_proba(np.array([[1000.0, 1000.0]]))
    np.testing.assert_array_equal(before, train.mean(axis=0))
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, before)


@pytest.mark.parametrize(
    "approach", ["site_mean9", "read9_noisy_or", "read15_noisy_or", "site_mean9_matched"]
)
def test_bundle_json_roundtrip(tmp_path, approach):
    rng = np.random.default_rng(1)
    dim = 15 if approach == "read15_noisy_or" else 9
    model = make_model("logistic", {"C": 1.0})
    model.fit(rng.normal(size=(60, dim)), np.arange(60) % 2)
    embedding = SequenceEmbedding() if dim == 15 else None
    bundle = ModelBundle(
        model, approach, embedding, {"n_reads": 20, "seed": 42, "repetitions": 5}, {}
    )
    joblib.dump(bundle, tmp_path / "model.joblib")
    reads = rng.uniform(size=(30, 9))
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"t": {"1": {"AAGACCA": reads.tolist()}}}) + "\n")
    frame = predict_json(tmp_path / "model.joblib", path, tmp_path / "output.csv", batch_size=1)
    sampled = np.array(
        [[reads[sample_indices(("t", 1), 30, 20, 42, "predict", r)] for r in range(5)]]
    )
    expected, _ = bundle.predict_arrays(reads.mean(axis=0)[None], sampled, ["AAGACCA"])
    np.testing.assert_allclose(frame.score, expected, rtol=1e-12)
    with pytest.raises(FileExistsError):
        predict_json(tmp_path / "model.joblib", path, tmp_path / "output.csv")


def test_submission_contract():
    frame = pd.DataFrame({"transcript_id": ["t"], "transcript_position": [1], "score": [0.3]})
    validate_submission(frame, [("t", 1)])
    with pytest.raises(ValueError):
        validate_submission(frame, [("wrong", 1)])
    with pytest.raises(ValueError):
        validate_submission(frame.assign(score=np.nan), [("t", 1)])


def test_scope_rejects_other_data():
    with pytest.raises(ValueError):
        enforce_training_scope(
            {"input": "data/data1/dataset1.json", "labels": "data/data0/data.info.labelled"}
        )


@pytest.mark.parametrize("test_fold", [None, 1])
def test_full_runner_cache_and_failure_tracking(tmp_path, monkeypatch, test_fold):
    from m6a_project.cache import prepare_data
    from m6a_project.tracking import rebuild_index
    from m6a_project.training import run_experiment

    root = tmp_path / "data" / "data0"
    root.mkdir(parents=True)
    input_path = root / "dataset0.json"
    label_path = root / "data.info.labelled"
    records, rows = [], []
    for gene in range(20):
        for label in (0, 1):
            tx, pos = f"t{gene}", label
            reads = (np.arange(180).reshape(20, 9) / 100 + label).tolist()
            records.append(json.dumps({tx: {str(pos): {"AAGACCA": reads}}}))
            rows.append((f"g{gene}", tx, pos, label))
    input_path.write_text("\n".join(records) + "\n")
    pd.DataFrame(rows, columns=["gene_id", "transcript_id", "transcript_position", "label"]).to_csv(
        label_path, index=False
    )
    config = {
        "input": str(input_path),
        "labels": str(label_path),
        "seed": 42,
        "n_splits": 5,
        "fold": 0,
        "test_fold": test_fold,
        "n_reads": 20,
        "repetitions": 5,
        "threads": 1,
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "runs"),
        "experiment_set": "integration",
        "model_params": {"logistic": {"C": 1.0}},
    }
    # Keep provenance snapshots scoped to this synthetic project in this test.
    monkeypatch.chdir(tmp_path)
    run_path = run_experiment(config, "read15_noisy_or", "logistic")
    assert json.loads((run_path / "status.json").read_text())["status"] == "completed"
    cache = prepare_data(config)
    manifest = cache["manifest"]
    assert_disjoint(manifest)
    if test_fold is not None:
        assert set(manifest.partition) == {"train", "val", "test"}
        predictions = pd.read_parquet(run_path / "validation_predictions.parquet")
        assert set(predictions.partition) == {"val"}
        assert not (cache["path"] / "test_reads.npy").exists()
    expected_train = manifest.query("partition == 'train'").label.to_numpy()
    # Explicit test evaluation is separate from fitting and is exercised only on synthetic data.
    import runpy

    monkeypatch.setattr("sys.argv", ["evaluate_test.py", "--model", str(run_path / "model.joblib")])
    evaluator = Path(__file__).resolve().parents[1] / "scripts/evaluate_test.py"
    if test_fold is None:
        with pytest.raises(ValueError, match="Historical two-way"):
            runpy.run_path(str(evaluator), run_name="__main__")
    else:
        runpy.run_path(str(evaluator), run_name="__main__")
        result_path = next((tmp_path / "runs/integration_test").glob("*/metrics.json"))
        result = json.loads(result_path.read_text())
        assert result["sites"] == int((manifest.partition == "test").sum())
        assert result["evaluation_partition"] == "test"
    # The sampled first feature equals the source row value plus the label, irrespective of split.
    assert np.all(cache["train_reads"].min(axis=(1, 2)) == expected_train)
    assert (run_path / "source_hashes.json").exists()
    with pytest.raises(ValueError, match="Unknown classifier"):
        run_experiment(config, "site_mean9", "does_not_exist", cache=cache)
    index = rebuild_index(config["output_dir"])
    assert set(index.status) == {"completed", "failed"}
    failed = Path(index[index.status == "failed"].iloc[0].run_path)
    assert "Unknown classifier" in (failed / "error.txt").read_text()


def test_three_way_split_preserves_validation_and_checks_every_pair():
    rows = [(f"g{i}", f"t{i}", label, label) for i in range(30) for label in (0, 1)]
    labels = pd.DataFrame(
        rows, columns=["gene_id", "transcript_id", "transcript_position", "label"]
    )
    old = make_split(labels)
    split = make_split(labels, test_fold=1)
    pd.testing.assert_frame_equal(split, make_split(labels, test_fold=1))
    assert set(split.partition) == {"train", "val", "test"}
    assert set(old.index[old.partition == "val"]) == set(split.index[split.partition == "val"])
    leaked = split.copy()
    test_row = leaked.index[leaked.partition == "test"][0]
    leaked.loc[test_row, "gene_id"] = leaked.loc[leaked.partition == "val", "gene_id"].iloc[0]
    with pytest.raises(ValueError, match="leaks"):
        assert_disjoint(leaked)
    with pytest.raises(ValueError, match="test fold"):
        make_split(labels, test_fold=0)
