import numpy as np


def noisy_or(probabilities, axis=-1):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape[axis] == 0:
        raise ValueError("Cannot pool an empty bag")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Read scores must be finite and in [0, 1]")
    with np.errstate(divide="ignore"):
        return -np.expm1(np.log1p(-probabilities).sum(axis=axis))
