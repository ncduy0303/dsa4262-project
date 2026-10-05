"""Immutable local runs with source snapshots, errors, and a rebuildable index."""

import json
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from .data import file_hash


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def git_output(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else "unavailable"


def source_snapshot(path):
    target = path / "source"
    target.mkdir()
    for name in ("src", "scripts", "configs", "notebooks", "tests"):
        if Path(name).exists():
            shutil.copytree(
                name,
                target / name,
                ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
            )
    for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md"):
        if Path(name).exists():
            shutil.copy2(name, target / name)
    write_json(
        path / "source_hashes.json",
        {str(p.relative_to(target)): file_hash(p) for p in target.rglob("*") if p.is_file()},
    )
    (path / "source.diff").write_text(
        git_output(
            "diff",
            "HEAD",
            "--",
            "src",
            "scripts",
            "configs",
            "notebooks",
            "tests",
            "pyproject.toml",
            "uv.lock",
        )
    )


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, text):
        self.logfile.write(text)
        self.logfile.flush()
        return self.stream.write(text)

    def flush(self):
        self.logfile.flush()
        self.stream.flush()


class Run:
    def __init__(self, config):
        self.config = config
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.id = f"{stamp}_{config['approach']}_{config['model']}_{uuid.uuid4().hex[:8]}"
        self.root = Path(config["output_dir"])
        self.path = self.root / config["experiment_set"] / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.status = {
            "run_id": self.id,
            "status": "running",
            "started_utc": datetime.now(UTC).isoformat(),
        }
        write_json(self.path / "config.json", config)
        write_json(self.path / "status.json", self.status)
        write_json(
            self.path / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "packages": {d.metadata["Name"]: d.version for d in distributions()},
                "git_commit": git_output("rev-parse", "HEAD").strip(),
                "git_source_status": git_output(
                    "status", "--short", "--", "src", "scripts", "configs", "notebooks", "tests"
                ),
                "uv_lock_sha256": file_hash("uv.lock") if Path("uv.lock").exists() else None,
            },
        )
        source_snapshot(self.path)
        (self.path / "rerun.txt").write_text(
            f"uv run python scripts/train.py --run-config {self.path / 'config.json'}\n"
        )

    def __enter__(self):
        self.log = (self.path / "run.log").open("w")
        self.stdout = redirect_stdout(Tee(sys.stdout, self.log))
        self.stderr = redirect_stderr(Tee(sys.stderr, self.log))
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.warn = warnings.catch_warnings(record=True)
        self.captured_warnings = self.warn.__enter__()
        warnings.simplefilter("always")
        print(f"Starting {self.id}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.warn.__exit__(exc_type, exc, tb)
        write_json(self.path / "warnings.json", [str(w.message) for w in self.captured_warnings])
        self.status.update(
            status="completed" if exc_type is None else "failed",
            finished_utc=datetime.now(UTC).isoformat(),
        )
        if exc_type:
            (self.path / "error.txt").write_text(
                "".join(traceback.format_exception(exc_type, exc, tb))
            )
        write_json(self.path / "status.json", self.status)
        print(f"{self.status['status']}: {self.id}", flush=True)
        self.stderr.__exit__(exc_type, exc, tb)
        self.stdout.__exit__(exc_type, exc, tb)
        self.log.close()
        rebuild_index(self.root)
        return False


def rebuild_index(root="experiments"):
    root = Path(root)
    rows = []
    for status_path in sorted(root.glob("*/*/status.json")):
        path = status_path.parent
        config = json.loads((path / "config.json").read_text())
        row = json.loads(status_path.read_text())
        row.update({k: config[k] for k in ("experiment_set", "approach", "model", "seed")})
        row["run_path"] = str(path)
        if (path / "metrics.json").exists():
            row.update(json.loads((path / "metrics.json").read_text()))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        temporary = root / "index.tmp.csv"
        result.to_csv(temporary, index=False)
        temporary.replace(root / "index.csv")
    return result
