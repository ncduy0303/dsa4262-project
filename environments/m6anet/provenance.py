"""Executed inside the unmodified official conda environment (Python 3.8)."""
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import m6anet
import torch
from m6anet.utils import constants, inference_utils


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


weights, threshold, norm = constants.PRETRAINED_CONFIGS["HCT116_RNA002"]
result = {
    "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
    "m6anet": importlib.metadata.version("m6anet"), "model": "HCT116_RNA002",
    "weights_sha256": sha(weights), "norm_sha256": sha(norm),
    "model_config_sha256": sha(constants.DEFAULT_MODEL_CONFIG),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "conda_explicit": subprocess.check_output(["micromamba", "list", "-n", "base", "--explicit"], text=True),
    "official_inference_source": inspect.getsource(inference_utils),
    "source_sha256": {},
}
root = os.path.dirname(m6anet.__file__)
for directory, _, filenames in os.walk(root):
    for name in filenames:
        if name.endswith(".py"):
            path = os.path.join(directory, name)
            result["source_sha256"][os.path.relpath(path, root)] = sha(path)
print(json.dumps(result, indent=2))
