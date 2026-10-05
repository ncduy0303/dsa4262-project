"""Render the measured pretrained comparison and its interpretation limits."""

import json
import shutil
from pathlib import Path

import nbformat as nbf
import pandas as pd

from m6a_project.evaluation import plt


def main():
    output = Path("docs/results")
    summary = json.loads((output / "m6anet_summary.json").read_text())
    run = Path(summary["run_path"])
    environment = json.loads((run / "official_environment.json").read_text())
    config = json.loads((run / "config.json").read_text())
    shutil.copy2(run / "conda-explicit.txt", "environments/m6anet/conda-explicit-linux-64.txt")
    table = pd.read_csv(output / "comparison_with_m6anet.csv")
    primary = table[
        (
            table.approach.isin(
                ["site_mean9", "read9_noisy_or", "read15_noisy_or", "official_m6anet"]
            )
        )
        & (table.model != "dummy")
    ].copy()
    primary["display"] = primary.model + " / " + primary.approach
    primary.loc[primary.approach == "official_m6anet", "display"] = (
        "m6Anet pretrained HCT116 (overlap unknown)"
    )
    primary = primary.sort_values("average_precision")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#c75532" if a == "official_m6anet" else "#3575a2" for a in primary.approach]
    ax.barh(primary.display, primary.average_precision, color=colors)
    ax.set(
        xlabel="Validation average precision",
        title="Same data0 validation sites; pretrained overlap is unknown",
    )
    fig.tight_layout()
    fig.savefig(output / "comparison_with_m6anet.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Official pretrained m6Anet comparison",
        "",
        "The official m6Anet HCT116 RNA002 checkpoint was run on the unchanged data0 validation set. This is a pretrained reference; its overlap with the course data is unknown. It is not an independent generalization estimate or a matched-training comparison.",
        "",
        "| Model | Approach | AP | ROC AUC | Trapezoidal PR AUC | Brier score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.sort_values("average_precision", ascending=False).itertuples():
        lines.append(
            f"| {row.model} | {row.approach} | {row.average_precision:.4f} | {row.roc_auc:.4f} | {row.pr_auc_trapezoid:.4f} | {row.brier_score:.4f} |"
        )
    result = summary["metrics"]
    best = (
        table[(table.model != "dummy") & (table.approach != "official_m6anet")]
        .sort_values("average_precision")
        .iloc[-1]
    )
    lines += [
        "",
        "![Validation comparison](comparison_with_m6anet.png)",
        "",
        "**Measured comparison**",
        "",
        f"- m6Anet returned predictions for all {result['sites']:,} validation sites, with {result['positive_sites']:,} positive labels. Key sets, read counts, central motifs, and finite probability bounds were verified.",
        f"- The AP difference relative to the strongest classical run ({best.model}, {best.approach}) is {result['average_precision'] - best.average_precision:+.4f}. This is a descriptive difference, not a significance or generalization claim.",
        f"- Official inference took {result['prediction_seconds']:.1f} seconds in the Linux amd64 container on this Apple Silicon host. Classical models ran natively, so runtimes are not a hardware-matched speed comparison.",
        "",
        "**Environment and input integrity**",
        "",
        f"- Official Bioconda m6Anet {environment['m6anet']}, Python {environment['python'].split()[0]}, PyTorch {environment['torch']}. The upstream inference implementation and checkpoint were not edited.",
        "- Synthetic site-local read indices were appended after the original nine signal features. All signal values were checked for exact preservation, and all byte offsets were independently checked. Only validation sites were supplied to the model.",
        f"- Frozen split SHA-256: `{config['frozen_split_sha256']}`.",
        f"- Checkpoint SHA-256: `{environment['weights_sha256']}`.",
        f"- Normalization SHA-256: `{environment['norm_sha256']}`.",
        "- Per-run artifacts include raw official site/read outputs, joined validation scores, metrics, curves, logs, model/config hashes, source snapshots, container identity, and the exact conda package export.",
        "",
        "**Interpretation limits**",
        "",
        "- Pretraining used HCT116, and we do not know which course data0 genes/sites it included. Our validation split was withheld from the classical models, but cannot be assumed to have been withheld from the pretrained checkpoint.",
        "- HCT116_RNA002 is the official default human model. The supplied course context does not establish sequencing chemistry independently, so checkpoint chemistry compatibility remains an assumption to confirm.",
        f"- m6Anet uses its own normalization and learned neural embedding, 20 reads sampled with replacement, and {config['num_iterations']:,} iterations. Classical read baselines use fixed PCA embeddings and five 20-read repetitions without replacement. These differences are intentionally retained and documented.",
        "- Compare `probability_modified` from the site output. `mod_ratio` is not used as a site probability. Official inference does not return per-iteration variance, so that diagnostic is recorded as unavailable.",
        "- A fair training-controlled comparison would retrain m6Anet on exactly the training genes and estimate normalization from those genes. That is a separate experiment.",
        "",
        "**Reproduce**",
        "",
        "```bash",
        "docker build --platform linux/amd64 -t dsa4262-m6anet:2.1.0 environments/m6anet",
        "uv run python scripts/benchmark_m6anet.py",
        "uv run python scripts/report_m6anet.py",
        "```",
        "",
        f"Benchmark run: `{run}`. See `environments/m6anet/README.md` for the legacy dependency pins and the official batching workaround. The primary classical comparison remains unchanged in `comparison.csv`.",
        "",
    ]
    (output / "m6anet_comparison.md").write_text("\n".join(lines))
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "name": "m6a-project",
        "display_name": "m6A project (uv)",
        "language": "python",
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Official pretrained m6Anet benchmark\nSame data0 validation sites. Potential pretrained training overlap prevents treating this as an independent test."
        ),
        nbf.v4.new_code_cell(
            'from pathlib import Path\nimport os, json\nimport pandas as pd\nfrom IPython.display import display, Image\nif not Path("pyproject.toml").exists():\n    os.chdir(Path.cwd().parent)\nsummary = json.loads(Path("docs/results/m6anet_summary.json").read_text())\nrun = Path(summary["run_path"])'
        ),
        nbf.v4.new_code_cell(
            'comparison = pd.read_csv("docs/results/comparison_with_m6anet.csv")\ndisplay(comparison[["model", "approach", "average_precision", "roc_auc", "pr_auc_trapezoid", "brier_score"]])'
        ),
        nbf.v4.new_code_cell(
            'display(Image(filename="docs/results/comparison_with_m6anet.png"))\ndisplay(summary["metrics"])'
        ),
        nbf.v4.new_code_cell(
            'display(Image(filename=str(run / "diagnostics.png")))\ndisplay(pd.read_csv(run / "calibration.csv"))'
        ),
        nbf.v4.new_markdown_cell(
            "The checkpoint was trained on HCT116. Overlap with the course sites is unknown. m6Anet uses a learned neural embedding, pretrained normalization, and 1,000 with-replacement sampling iterations; the classical read models use fixed PCA and five without-replacement repetitions. Treat these results as a comparison of complete methods, not an isolated architecture effect."
        ),
    ]
    nbf.write(nb, "notebooks/04_m6anet_benchmark.ipynb")
    print(primary[["model", "approach", "average_precision", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
