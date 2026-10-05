"""Fixed sequence PCA, not the task-learned neural embedding used by m6Anet."""

from itertools import product

import numpy as np
from sklearn.decomposition import PCA


def vocabulary():
    contexts = (
        a + d + r + "AC" + h + z for a, d, r, h, z in product("ACGT", "AGT", "AG", "ACT", "ACGT")
    )
    return sorted({s[offset : offset + 5] for s in contexts for offset in range(3)})


class SequenceEmbedding:
    def __init__(self):
        self.motifs = vocabulary()
        alphabet = "ACGT"
        descriptors = np.array(
            [
                [float(base == letter) for base in motif for letter in alphabet]
                for motif in self.motifs
            ]
        )
        self.pca = PCA(n_components=2, svd_solver="full")
        self.values = self.pca.fit_transform(descriptors)
        self.lookup = dict(zip(self.motifs, self.values, strict=True))

    def transform(self, sequences):
        result = []
        for sequence in sequences:
            if len(sequence) != 7:
                raise ValueError("Expected a seven-base sequence")
            try:
                result.append(np.concatenate([self.lookup[sequence[o : o + 5]] for o in range(3)]))
            except KeyError as exc:
                raise ValueError(f"Unsupported 5-mer in {sequence}") from exc
        return np.asarray(result).reshape(-1, 6)

    def summary(self):
        return {
            "method": "positional_nucleotide_onehot_then_PCA",
            "vocabulary_size": len(self.motifs),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "distinct_vectors_at_10_decimals": len(np.unique(self.values.round(10), axis=0)),
        }
