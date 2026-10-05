# Initial attempt for Task 1: plan for review

> Superseded for current experimental scope by [the data0-only classical baseline plan](data0-classical-baselines-plan.md), drafted for review on 5 October 2026. Retained as historical context.

Prepared 5 October 2026; rechecked against the Canvas redownload on 5 October 2026. Status: references reviewed and data audited; model training, environment installation, and leaderboard submission have not been performed.

**Recommendation:** retain Python, uv, and Jupyter. Put reusable processing, training, evaluation, and prediction in a Python package with command-line entry points. Use notebooks for exploration and presentation. Start with logistic regression and histogram gradient boosting on site-level summaries, and benchmark against m6Anet in a separate environment.

1. **Requirements and deadline**
   - Intermediate predictions are due 7 October at 23:59. The handout says questions stop being answered from 5 October at 21:00 through 7 October. Confirm unresolved dataset details in class today.
   - Task 1 requires separate training and prediction scripts, a saved model, comparison with m6Anet, and probabilities in CSV columns `transcript_id,transcript_position,score`.
   - Evaluation uses ROC AUC and precision-recall AUC. The slides explicitly require splitting by gene ID. Labels belong to sites, not individual reads.
   - The final repository must be public, documented, executable on AWS Ubuntu, and include a small prediction example. Task 2 requires predictions across all available SG-NEx direct RNA-Seq datasets, cross-cell-line analysis, and an interactive visualization.
   - The report must document AI use and verification, including actual erroneous or unverified outputs. Keep a contemporaneous log rather than inventing examples later.
   - Sources: [local handout](../reference/handout_project2_RNAModifications.html) and [project slides](../reference/05_NUS_Genomics_TeamProject_2026.pdf), especially slides 42-65. The refreshed handout corrects the individual teamwork-summary deadline to 29 October at 23:59. The intermediate and final team deadlines remain unchanged.

2. **What the supplied files actually contain**

   | Folder  |   Sites | Labels                                           | Proposed role                                                                |
   | ------- | ------: | ------------------------------------------------ | ---------------------------------------------------------------------------- |
   | `data0` | 121,838 | 5,475 positive; 4.49%                            | Primary labelled training and evaluation source; includes gene IDs           |
   | `data1` |  90,810 | 6,593 positive; 7.26%                            | Authorized additional training source; needs gene mapping                    |
   | `data2` |   1,323 | 0, 0.25, 0.5, 0.7, 0.75, 0.95, 1; 189 sites each | Untouched external prediction/evaluation set with unresolved label semantics |
   - Redownload verification: all three newly downloaded `.json.gz` archives decompress to JSON files with exactly the same SHA-256 hashes as the original audit. All three metadata files are byte-identical to the Git baseline, which was unchanged during the original audit. The dataset content, labels, overlap findings, and evaluation recommendations therefore remain unchanged. See [redownload comparison](data-redownload-comparison.json).
   - The archives were retained, and decompressed JSON files were recreated beside them. Future ingestion should accept JSON or gzip-compressed JSON for sequential reading; m6Anet byte-offset access must use the decompressed JSON, since the metadata offsets describe that content.
   - These counts were calculated from local files, not inferred from filenames. Full JSON scans confirmed unique site keys, exact metadata key coverage, finite numeric values, and nine values per read in all folders. Existing offsets and read counts in data1/data2 match their JSON files. See [audit results](data-audit.json).
   - Reads per site range from 20 to 991 in data0, 20 to 994 in data1, and 26 to 3,438 in data2. Stream the JSON one site at a time; do not load every read into a single dataframe.
   - data0 and data1 share 67,320 `(transcript_id, transcript_position)` keys. Labels disagree at 3,872 shared keys. Preserve source identity and labels; do not overwrite or majority-vote them.
   - data1 lacks gene IDs. Mapping its transcript IDs through data0 leaves 15,137 sites unmapped. Do not substitute transcript grouping and describe it as gene-independent evaluation.
   - User-provided clarification: the professor announced data1 and data2 as additional labelled datasets that are free to use for training. Their use is authorized. Reserving data2 as an external test is our experimental choice, not an official dataset designation.
   - I do not know the cell line, replicate identity, chemistry, annotation release, or official leaderboard role of each folder. The fractional values and synthetic-looking transcript IDs in data2 suggest a controlled experiment, but that interpretation is [UNVERIFIED]. Neither supplied reference establishes it.

3. **Leakage-safe splitting and evaluation**
   - First establish a fully valid baseline on data0: fixed gene-disjoint train, validation, and internal test partitions, targeting 70%/15%/15% of genes. Record the actual site counts and class balance. Adjust grouping only for feasibility, never to improve reported scores.
   - Save one split manifest with a fixed seed and input hashes. All sites, isoforms, and observations belonging to a gene must stay together. Fit encoders, scalers, imputation, feature selection, and any normalization on training data only. IDs and offsets are join keys, not predictive features.
   - Train/validate across the authorized data0 and data1 sources using the same gene assignment across sources. Obtain the matching transcript-to-gene annotation if available. Until then, include only verified mapped data1 records in the primary combined experiment and report the exclusions. Keep unmapped records out of strict evaluation.
   - Maintain both a data0-only and combined-source experiment. Preserve distinct observations at shared sites and consider per-site sample weighting so duplicated sites do not dominate. Report validation metrics separately by source, plus pooled metrics with explicit weighting.
   - Include a domain-transfer diagnostic: train on data0 training genes and evaluate on data1 held-out genes. A whole-folder data0-to-data1 comparison may be reported only with shared genes excluded, or explicitly labelled as an overlapping-site diagnostic.
   - Use validation to select the model and settings. Evaluate the internal test once after selection. Keep data2 out of all fitting, tuning, threshold selection, and calibration.
   - Metrics on binary labels: ROC AUC, average precision (AP), and trapezoidal PR AUC, with names clearly distinguished. Use AP as the proposed development criterion, report both course metrics, and confirm the leaderboard's PR integration convention. Include class prevalence, site counts, PR/ROC curves, and runtime. AP and trapezoidal PR area are not interchangeable; see [scikit-learn's AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
   - Add confusion matrices, precision, recall, and F1 at a threshold selected on validation only. Do not optimize accuracy for these imbalanced datasets. Gene-level bootstrap intervals and repeated grouped validation are useful follow-up checks after the first complete run.
   - For data2, always produce valid predictions. Do not round fractional labels or run binary AUC on them. If documentation confirms modification fractions, assess score ordering and distributions by fraction; evaluate stoichiometry MAE/RMSE only for outputs that actually estimate a fraction. Site modification probability and fraction of modified reads are different targets. Binary metrics on the 0/1 subset require confirmation that those endpoints encode the relevant binary truth.
   - A leaderboard CSV can be generated without labels. I do not know whether data2 is the official 7 October input; verify the release instructions before uploading.

4. **Small, interpretable baseline ladder**

   | Model                       | Features                                       | Purpose                                                      |
   | --------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
   | Training-prevalence dummy   | None                                           | Sanity check and metric reference                            |
   | Logistic regression         | Mean of each signal feature, sequence encoding | Transparent first learned baseline                           |
   | Logistic regression         | Richer summaries, same sequence encoding       | Isolate the value of read variability                        |
   | Histogram gradient boosting | Same richer features                           | Simple nonlinear contender without a new boosting dependency |
   | m6Anet                      | Read-level features and sequence               | Required external method comparison                          |
   - Basic features: the per-site means of the nine read features. Richer features: mean, standard deviation, median, lower/upper quartiles for each signal feature, plus position-wise sequence one-hot encoding. For logistic regression, standardize numeric summaries inside the saved pipeline. Handle unseen categories explicitly.
   - Test `log1p(n_reads)` as an ablation, not an unquestioned default: depth can carry dataset-specific information. Compare sequence-only, signal-only, and combined features if time permits. Do not infer the relative benefits before measuring them.
   - Begin with default regularization and a small preset configuration grid. Compare unweighted and balanced training on the same validation set. Avoid oversampling before splitting and avoid treating every read as if it had its site's label.
   - Do not assume class-weighted outputs are calibrated probabilities. Report calibration diagnostics; introduce calibration using training-only grouped out-of-fold predictions if needed. Preserve an untouched test set.
   - Defer a custom neural network, attention MIL, large hyperparameter searches, and ensembles until the first end-to-end artifact is reproducible. Keep raw read bags accessible so they can be added later without redesigning ingestion.

5. **Python and environment decision**
   - Main project: Python 3.12, managed by uv. Use `requires-python = ">=3.12,<3.13"` initially and pin an exact supported patch in `.python-version` when setting up. Commit `pyproject.toml` and `uv.lock`; reproduce with `uv sync --locked`.
   - Rationale: Python 3.12 is already installed locally and is a conservative common choice for the scientific stack and future modern PyTorch work. Its published support horizon extends to October 2028. See [Python support status](https://devguide.python.org/versions/), [scikit-learn installation](https://scikit-learn.org/stable/install.html), and [PyTorch installation](https://pytorch.org/get-started/locally/). Exact package compatibility still requires a lock resolution and macOS/Linux smoke tests; it has not been tested here.
   - Core dependencies: NumPy, pandas, SciPy, scikit-learn, PyArrow, joblib, matplotlib, and a small CLI implemented with argparse. Optional groups: JupyterLab/ipykernel for notebooks; pytest/ruff for development; PyTorch for future MIL; Plotly/Streamlit for Task 2 exploration.
   - m6Anet: separate conda environment with Python 3.8, following the handout. Do not force its legacy dependencies into the main uv project. The [upstream setup file](https://raw.githubusercontent.com/GoekeLab/m6anet/master/setup.py) specifies Python `>=3.7,<3.9` and `torch==1.6.0`, despite broader wording in its installation guide.
   - Prefer a reproducible Linux x86_64 conda setup for this legacy benchmark if native Apple Silicon installation is problematic. Export the resolved environment and record the m6Anet version. Verify installation before making runtime promises. Do not silently upgrade its PyTorch dependency or call a modified implementation the official baseline.
   - Use CPU for the first baseline. No AWS provisioning is needed for the plan; if legacy benchmarking needs AWS, select an environment and budget explicitly during implementation.
   - uv's default cache was not writable in this agent sandbox; pointing `UV_CACHE_DIR` at a writable temporary directory allowed interpreter discovery. This is not evidence that uv itself is broken.

6. **m6Anet benchmark and data adapter**
   - Run the published pretrained human model first as an off-the-shelf reference, after confirming chemistry compatibility. Its default checkpoint was trained on HCT116, so independence from the course data is unverified. Flag potential training overlap rather than calling this a fair from-scratch comparison. See the [official repository](https://github.com/GoekeLab/m6anet).
   - For a controlled comparison, train m6Anet on the same gene partitions, recompute normalization from training genes only, and use the same validation/model-selection rule. Schedule this after the first baseline if it cannot finish before the meeting. The [training guide](https://m6anet.readthedocs.io/en/latest/training.html) describes the required split metadata.
   - Build adapters in a derived directory, preserving originals. Name adapted signals `data.json`; build byte offsets and read counts; generate training fields `modification_status` and `set_type` where needed. The provided data0 `data.info.labelled` is not the full m6Anet training-index schema despite its name.
   - Version-check the loader. The [current source](https://raw.githubusercontent.com/GoekeLab/m6anet/master/m6anet/utils/data_utils.py) treats the final read-array column as a read ID. Our files have exactly nine signal columns. Append documented synthetic read IDs if required, retain all signal columns unchanged, and rebuild offsets after serialization. These identifiers cannot recover original molecules or support cross-site read tracking.
   - Smoke-test adapted records against originals and confirm signal columns, sequence, coordinates, and read counts survive conversion. Test on a small subset before full inference.
   - Compare `data.site_proba.csv` site probabilities with our predictions on identical labelled keys. Report prediction coverage and any missing sites. Do not substitute `mod_ratio` for site probability. If fraction labels are confirmed, treat the latter as a separate stoichiometry assessment.
   - Record checkpoint, normalization, seed, read sampling settings, iterations, runtime, and eligible-site count. Do not hide unsupported sites by shrinking only one model's evaluation set.

7. **Extensible repository and commands**

   Proposed files below do not exist yet, apart from this plan and the audit JSON.

   ```text
   pyproject.toml / uv.lock / .python-version
   src/m6a_project/
     io.py              # streamed site records and validation
     features.py        # versioned site summaries
     splits.py          # gene-level manifests
     models.py          # model factories and saved pipelines
     evaluation.py      # metrics and matched-site comparisons
     submission.py      # strict CSV contract
     cli.py
   scripts/train.py
   scripts/predict.py
   scripts/benchmark_m6anet.py
   configs/             # datasets, feature sets, model settings
   notebooks/           # audit, baseline results, benchmark report
   tests/fixtures/      # small documented prediction example
   environments/        # legacy m6Anet conda specification
   docs/                # setup, provenance, AI log, decisions
   artifacts/           # models, metrics, plots, manifests
   submissions/         # validated CSVs and manifests
   ```

   - Canonical data record: source/sample ID, transcript ID, position, sequence, read features, optional gene ID and label. Labels must be optional at inference. Join on source plus site keys for multi-sample work.
   - Cache deterministic per-site features as Parquet, keyed by raw-file hash and feature configuration. Keep preprocessing that learns from data inside the fitted model pipeline. Save feature schema, package versions, config, split hash, and seed with each model.
   - Proposed commands: `uv run python scripts/train.py --config configs/baseline.toml` and `uv run python scripts/predict.py --model artifacts/best/model.joblib --input data/data2/dataset2.json --output submissions/data2.csv`. These are interface targets, not currently runnable commands.
   - Prediction must require only JSON and a model, never the labels or training notebook state. A notebook should call the same package functions and execute from top to bottom.
   - Validate exact header/order, unique keys, exact input-key coverage, finite scores in [0,1], and no accidental dataframe index. Preserve input order for traceability. Record a CSV checksum and model provenance. Keep the final upload separate from file generation.
   - Meaningful tests: signal order and aggregation on hand-checkable records, zero group overlap across sources, preprocessing fit isolation, save/load prediction agreement, adapter round-trip, and submission contract. Run a clean Ubuntu prediction smoke test before delivery.
   - Keep large data, caches, and model runs out of Git; retain manifests and small permitted fixtures. Make public only the materials required and allowed for final code evaluation.

8. **Task 2 extension and framework choices**
   - Add a sample manifest with cell line, replicate, chemistry, reference version, input URI, and processing status. Reuse the prediction CLI for batch samples and write partitioned Parquet with sample/site identity, score, read depth, and model version.
   - Compare cell lines using shared callable sites and depth/replicate-aware summaries. Report missing coverage distinctly from low probability. Do not equate model score differences with demonstrated biological differential methylation.
   - Keep a future bag-model interface alongside the tabular interface, enabling PyTorch MIL without changing output contracts. Do not claim site-summary baselines estimate read-level methylation fractions.
   - Use Plotly with Streamlit for a first researcher-facing explorer, with sample, gene, motif, coverage, and score filters. This is a proposed design, not a required framework. The handout allows other hosting platforms.
   - Jupyter plus scripts plus uv is appropriate here. A workflow tool such as Snakemake can be evaluated when Task 2 spans many samples; a hosted experiment tracker, workflow service, or alternative language adds unnecessary scope to this initial attempt. Simple configuration files and local run manifests are enough initially.
   - SG-NEx access and sample provenance should follow the [official project repository](https://github.com/GoekeLab/sg-nex-data).

9. **Execution order after plan review**
   - First milestone, for tonight's meeting: initialize uv and package layout; preserve audit/provenance; create gene splits; build feature cache; run dummy, logistic regression, and histogram boosting; save curves and metric table; generate and validate predictions for data2. Report its test metrics only where label meaning supports them.
   - In the same work period, establish the legacy m6Anet environment and a verified adapter smoke test. Run pretrained inference if installation succeeds. A full controlled m6Anet retraining benchmark is the next priority, not a prerequisite for producing our first valid CSV. Actual timing is unknown until tested.
   - On 6 October: resolve dataset provenance and unmapped genes; complete m6Anet comparison; inspect source/motif/depth errors; make a small number of validation-driven improvements; execute notebooks and clean installation checks.
   - Before 7 October 23:59: freeze the selected settings; evaluate internal test once; refit on data0/data1 development data after recording evaluation, retaining data2 as the chosen external holdout; predict the confirmed leaderboard input; validate and archive CSV/model/config; upload using the official instructions and record receipt.
   - Completion criteria: documented install, independently executable train/predict scripts, saved reproducible model, comparable metrics with coverage, m6Anet comparison or an explicit evidenced blocker, a valid submission file, and a meeting-ready notebook/report. Do not represent any of these implementation deliverables as completed by this planning pass.

10. **Details to resolve from the release announcement or teaching team**
    - Which file is the intermediate leaderboard input? What are the sample identities and chemistry of data0 and data1? Authorization to use the additional datasets has been confirmed by the professor's announcement relayed by the user.
    - What precisely does data2's label encode? Are there paired mixtures or shared source molecules that affect uncertainty estimates?
    - Which transcript annotation provides data1's missing gene IDs?
    - Which m6Anet version/checkpoint is expected, and how should its training overlap be discussed?
    - What PR AUC convention, upload location, naming rule, and submission limits apply?

These uncertainties do not prevent implementing the data0 baseline, reusable prediction code, or CSV validation. They do prevent an honest claim of an independent binary test score on all three folders or of a confirmed leaderboard-ready upload target.
