"""Execute notebooks against the uv interpreter with a project-local kernelspec."""

import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Optional notebook paths; default all")
    args = parser.parse_args()
    root = Path.cwd()
    kernel_root = root / "artifacts/jupyter/share/jupyter"
    spec = kernel_root / "kernels/m6a-project"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "m6A project (uv)",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(root / "artifacts/jupyter/runtime")
    os.environ["IPYTHONDIR"] = str(root / "artifacts/ipython")
    paths = (
        [Path(p) for p in args.notebooks]
        if args.notebooks
        else sorted((root / "notebooks").glob("*.ipynb"))
    )
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        NotebookClient(
            nb, timeout=300, kernel_name="m6a-project", resources={"metadata": {"path": str(root)}}
        ).execute()
        nbformat.write(nb, path)
        print(f"Saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
