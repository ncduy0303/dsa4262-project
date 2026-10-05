"""Predetermined stratified gene folds, shared by all estimators."""

from itertools import combinations

from sklearn.model_selection import StratifiedGroupKFold


def assert_disjoint(table):
    partitions = set(table.partition)
    if not {"train", "val"} <= partitions or not partitions <= {"train", "val", "test"}:
        raise ValueError("Invalid split partitions")
    subsets = [table[table.partition == name] for name in sorted(partitions)]
    for left, right in combinations(subsets, 2):
        for col in ("gene_id", "transcript_id"):
            if set(left[col]) & set(right[col]):
                raise ValueError(f"Split leaks {col}")
    if table.duplicated(["transcript_id", "transcript_position"]).any():
        raise ValueError("Duplicate sites across split")
    for subset in subsets:
        if set(subset.label) != {0, 1}:
            raise ValueError("Each partition must contain both classes")


def make_split(labels, seed=42, n_splits=5, fold=0, test_fold=None):
    if not 0 <= fold < n_splits:
        raise ValueError("Invalid fold")
    if test_fold is not None and (
        n_splits < 3 or not 0 <= test_fold < n_splits or test_fold == fold
    ):
        raise ValueError("Invalid test fold")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = labels.copy()
    result["partition"] = "train"
    for index, (_, heldout) in enumerate(
        splitter.split(labels, labels.label, groups=labels.gene_id)
    ):
        if index == fold:
            result.loc[result.index[heldout], "partition"] = "val"
        elif index == test_fold:
            result.loc[result.index[heldout], "partition"] = "test"
    assert_disjoint(result)
    return result


def split_summary(table):
    return {
        name: {
            "sites": len(part),
            "genes": part.gene_id.nunique(),
            "positive_sites": int(part.label.sum()),
            "prevalence": float(part.label.mean()),
        }
        for name, part in table.groupby("partition")
    }
