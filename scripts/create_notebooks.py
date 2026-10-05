"""Generate thin, executable notebooks around the shared experiment API."""

from pathlib import Path

import nbformat as nbf

BOOT = """from pathlib import Path
import os
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
import json
import pandas as pd
from IPython.display import display, Image
from m6a_project.training import load_config, run_experiment
from m6a_project.tracking import rebuild_index
from m6a_project.cache import prepare_data
CONFIG = load_config()
"""


def write(name, title, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "name": "m6a-project",
        "display_name": "m6A project (uv)",
        "language": "python",
    }
    nb.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(BOOT)]
    for kind, text in cells:
        nb.cells.append(
            nbf.v4.new_code_cell(text) if kind == "code" else nbf.v4.new_markdown_cell(text)
        )
    nbf.write(nb, Path("notebooks") / name)


write(
    "01_data0_and_split.ipynb",
    "# Data0 and the frozen gene split\nOnly data0 is accessed. All reported scores are validation estimates.",
    [
        (
            "code",
            'cache = prepare_data(CONFIG)\nmanifest = cache["manifest"]\ndisplay(pd.DataFrame(cache["identity"]["summary"]).T)\nprint("Split SHA-256:", cache["identity"]["split_hash"])',
        ),
        (
            "code",
            'from m6a_project.splitting import assert_disjoint\nassert_disjoint(manifest)\ndisplay(manifest.groupby("partition").n_reads.describe())',
        ),
        (
            "code",
            'from m6a_project.embeddings import SequenceEmbedding\nembedding = SequenceEmbedding()\ndisplay(embedding.summary())\ndisplay(pd.DataFrame(embedding.values, index=embedding.motifs, columns=["PC1", "PC2"]).head(12))',
        ),
        (
            "markdown",
            "The PCA vocabulary comes from motif grammar, not validation frequencies or labels. This fixed representation differs from m6Anet's learned neural embedding. Read models inherit site labels, which are noisy read targets.",
        ),
    ],
)
write(
    "02_classical_comparison.ipynb",
    "# Classical baseline experiments\nThe same runner handles CLI and notebook experiments. Every fitting call creates a new logged run.",
    [
        (
            "code",
            """# Enable deliberately to create a new experiment; normal replay only reads saved results.
RUN_NEW_EXPERIMENT = False
if RUN_NEW_EXPERIMENT:
    config = dict(CONFIG)
    config["experiment_set"] = "notebook_exploration"
    config["notebook_source"] = "notebooks/02_classical_comparison.ipynb"
    config["purpose"] = "Explore a documented variant using the frozen data0 split"
    run_experiment(config, approach="read9_noisy_or", model="logistic")""",
        ),
        (
            "code",
            'index = rebuild_index()\ncomplete = index.query("experiment_set == \'data0_baselines_v1\' and status == \'completed\'")\ndisplay(complete[["approach", "model", "average_precision", "roc_auc", "pr_auc_trapezoid", "brier_score", "fit_seconds"]])',
        ),
        (
            "code",
            'display(Image(filename="docs/results/comparison.png"))\ndisplay(pd.read_csv("docs/results/approach_differences.csv"))',
        ),
        (
            "markdown",
            "Compare approaches within each classifier. Better performance from A1 to A3 is a hypothesis, not a requirement. The matched-mean control uses the same read samples as A2/A3. Validation has been used for model selection and is not an independent test.",
        ),
    ],
)
write(
    "03_diagnostics.ipynb",
    "# Pooling and model diagnostics\nInspect calibration, saturation, coverage, and sequence representation before interpreting performance.",
    [
        (
            "code",
            'table = pd.read_csv("docs/results/comparison.csv")\ndisplay(table[["approach", "model", "log_loss", "brier_score", "fraction_score_ge_099"]])',
        ),
        (
            "code",
            """for model in ["logistic", "hist_boosting", "random_forest"]:
    selected = table[(table.model == model) & (table.approach == "read15_noisy_or")].iloc[-1]
    print(model, selected.run_path)
    display(Image(filename=str(Path(selected.run_path) / "diagnostics.png")))""",
        ),
        (
            "code",
            """selected = table[(table.model == "logistic") & (table.approach == "read15_noisy_or")].iloc[-1]
run_dir = Path(selected.run_path)
display(pd.read_csv(run_dir / "calibration.csv"))
display(pd.read_csv(run_dir / "motif_metrics.csv"))
print(json.loads((run_dir / "embedding.json").read_text())["explained_variance_ratio"])""",
        ),
        (
            "markdown",
            "Noisy-OR scores from inherited-label training are not established read modification probabilities. Saturation and calibration errors must be reported. No stoichiometry estimate or claim of superiority to official m6Anet is made.",
        ),
    ],
)
