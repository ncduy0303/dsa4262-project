# Data0-only classical baselines: plan for approval

Approved by the user and implemented on 5 October 2026. This replaces the experimental scope in `initial-attempt-plan.md`. See [the results report](results/report.md) for measured outcomes and verification. The existing audits remain useful provenance.

**Proposed outcome:** reproducible experiments comparing the three requested feature/pooling approaches using interchangeable sklearn classifiers, one shared gene-grouped data0 train/validation split, Python core modules, exploratory notebooks, and complete local experiment records.

1. **Scope and experimental question**

   - Use only `data/data0/dataset0.json` (or its gzip equivalent) and `data/data0/data.info.labelled`. Do not load, modify, evaluate, predict, or tune against data1 or data2 during this phase. Prior observations about those datasets will not determine modelling choices.
   - Compare: site-mean signals; individual-read signals with Noisy-OR; individual-read signals plus a two-dimensional embedding for each neighboring 5-mer with Noisy-OR.
   - The hypothesis is that preserving read variation and adding sequence information improves validation performance. Improvement is not guaranteed. Record negative results and compare approaches within the same classifier family.
   - No neural networks, PyTorch, pretrained neural embeddings, or m6Anet inference in this phase. The paper informs the experimental design. A measured comparison with official m6Anet remains a later course requirement, not an output of this classical-only phase.
   - Retain a generic saved-model prediction script and submission CSV validator for later use. Demonstrate them on data0 validation records only. There is no independent test score in this phase, and validation performance must not be described as test performance.

2. **What we take from the supplied m6Anet paper**

   - Source: Hendra et al., *Detection of m6A from direct RNA sequencing using a multiple instance learning framework*, Nature Methods (2022), supplied as `reference/m6Anet.pdf`. See Figure 1a and Methods, PDF pages 10-11, particularly equations 4-7 and the model-evaluation protocol.
   - A site is a bag of reads. The measured label belongs to the site; true per-read modification labels are unavailable. The paper learns read scores and embeddings jointly by optimizing a site-level loss through Noisy-OR.
   - Each read has nine signal inputs from three neighboring positions. Three 5-mer embeddings contribute six additional values. The embedding vocabulary has 66 5-mers across the neighboring windows, not 66 possible central DRACH motifs.
   - The paper samples 20 reads per site and averages site predictions over repeated sampling at evaluation. Its described evaluation uses five repetitions. Use these as initial sampling settings, not the iteration defaults of a later software release.
   - The paper splits by gene and describes k-mer-specific signal normalization. Our primary baseline comparison uses ordinary model-specific scaling so sequence information enters only method 3. K-mer-specific normalization is an optional, separately named later ablation because it would also introduce sequence information into methods 1 and 2.
   - The exact feature order in our files is dwell time, signal standard deviation, and mean current at each of the three positions. Preserve that order explicitly rather than copying an ordering from a schematic.

3. **One frozen train/validation split**

   - Use `gene_id` as the group. Propose one saved split targeting 80% training and 20% validation with seed 42. Use one predetermined fold of shuffled `StratifiedGroupKFold(n_splits=5)` to approximately balance site labels while preserving groups; do not execute cross-validation or choose a fold by model performance in this first phase.
   - Report actual gene/site counts and prevalence. Group sizes mean the requested proportion is approximate. Assert no shared genes, transcripts, or site keys. All reads from a site inherit its partition.
   - Save the manifest and hashes of both data0 input files. Do not regenerate the split for each experiment. Keep class prevalence natural in validation.
   - Fit signal scaling and any data-derived transformations on training records only. Never provide gene ID, transcript ID, site coordinates, label, or source-file offsets as predictive features.
   - Validation is used for development and model selection. As experiments accumulate, its scores become optimistic selection estimates. A later independent evaluation requires a separately approved dataset or protocol.

4. **The three approaches**

   | ID | Representation supplied to estimator | Estimator training unit | Final site score |
   |---|---|---|---|
   | A1: `site_mean9` | Nine column-wise means across all reads at a site | Site | Classifier positive-class probability |
   | A2: `read9_noisy_or` | Original nine signal features per sampled read | Read with inherited site label | Noisy-OR on sampled read probabilities, averaged across repetitions |
   | A3: `read15_noisy_or` | Same nine signals plus three two-dimensional 5-mer encodings | Read with inherited site label | Exactly the same pooling and sampling as A2 |

   - A1 retains nine separate means; it does not collapse the different signal units into one scalar. Do not add sequence, coverage, quantiles, or extra summary statistics to the primary A1 baseline.
   - A2/A3 use identical sampled read indices for a given site and seed. A3 repeats the same sequence encoding for every read at that site because the provided sequence is site-level context.
   - Features and estimators are independent. The same model factory should accept either nine or fifteen columns without knowing how they were constructed.

5. **Explicit approximation for sklearn read training**

   - Recommended initial policy: assign each sampled training read its parent site's label and fit a standard classifier. Log this as `training_objective = inherited_site_label`; call the resulting read scores weakly supervised scores.
   - A positive site may contain unmodified reads. This label inheritance therefore creates noisy targets and does not reproduce m6Anet's joint bag-level learning. Do not claim biologically validated read probabilities or estimate modification fractions from these scores.
   - Sample a fixed 20 reads without replacement from each training site using a persisted seed. Give every site equal total training weight. This bounds the read-table size and prevents high-coverage sites from dominating. Use the same sampled training set for A2 and A3 across models.
   - Start with unweighted site classes. Test balanced site-class weights only as a logged follow-up, applying the corresponding weight consistently to all sampled reads. Do not oversample before splitting or silently combine class-weight schemes.
   - If sampled reads are later increased or resampled, treat that as a new experiment, not a model-specific convenience. Do not claim that this baseline trains on every available read.
   - An optional future, still non-neural alternative is a logistic read scorer optimized directly against site labels through Noisy-OR using SciPy. It needs a custom estimator and objective, and cannot be reproduced merely by replacing sklearn LogisticRegression with a random forest. Keep it out of the initial interchangeable-model comparison unless separately approved.
   - This is the main methodological decision for approval: the initial A2/A3 experiments use noisy inherited labels to preserve compatibility with ordinary sklearn estimators. If exact bag-level training is required instead, revise this part before implementation.

6. **Non-neural 66-to-two-dimensional sequence embedding**

   - Recommended initial alternative: construct the theoretical DRACH-compatible 5-mer vocabulary, encode each motif by nucleotide identity at each position, and reduce those sequence descriptors to two dimensions with sklearn PCA. This yields a saved lookup table from each of the 66 motifs to a vector in R² without using labels or a neural model.
   - Use sorted vocabulary, a fixed alphabet, and full-SVD PCA; persist the exact fitted transform and lookup table. Fit this transform once on the fixed vocabulary, not on validation records or their frequencies. A vocabulary derived solely from the predefined motif grammar is not a learned statistic from validation.
   - Do not use PCA on arbitrary integer motif IDs: their numeric ordering has no biological meaning. Position-wise nucleotide one-hot descriptors preserve an explicit sequence representation before compression.
   - For seven-base context `s`, use the windows `s[0:5]`, `s[1:6]`, `s[2:7]`. Concatenate their embeddings in that order after the nine signals. Validate the vocabulary against training sequences; reject unsupported contexts with a clear error rather than silently mapping them to an arbitrary vector.
   - This is a fixed unsupervised sequence embedding, not m6Anet's task-learned embedding. Two-dimensional compression may discard useful motif distinctions. Record explained variance and inspect motif collisions; do not promise that A3 will outperform A2.
   - A separately logged position-wise one-hot sequence control can later test whether poor A3 performance is caused by the compression. It changes input dimensionality and must not replace the requested 15-feature experiment without being identified.
   - Loading the paper's pretrained embedding would import neural training and potentially overlapping data. Do not do that in this phase.

7. **Noisy-OR implementation and coverage control**

   - For read scores `p_j`, use `P_site = 1 - product(1 - p_j)`. Implement using `-expm1(sum(log1p(-p)))` for numerical stability, with explicit handling of endpoints and invalid probabilities. Save the calculation policy.
   - Primary validation: sample 20 reads without replacement per site, pool within each sample, repeat five times, and average the five site probabilities. Predict in batches and preserve the mapping between each read and its site. Independent repetitions may overlap in their selected reads.
   - The previous audit found at least 20 reads for every data0 site. If a future input has fewer reads, fail with an explicit coverage report under the primary policy; do not silently change the sampling rule or omit CSV rows.
   - Pooling all reads would confound the score with coverage. For illustration, Python calculation gives a pooled score of approximately 0.182 for 20 identical read scores of 0.01, but approximately 0.993 for 500 such scores. Stable arithmetic does not remove this statistical effect.
   - Monitor score saturation near one, ties, score-versus-depth, and variation across sampling repetitions. Noisy-OR of weak-label classifier outputs is not automatically calibrated as site probability.
   - A1 uses all reads as requested, whereas A2/A3 use fixed read samples. Include a secondary A1 control that averages the same sampled reads, predicts each sample, and averages predictions across repetitions. This separates some sampling effects from representation effects while leaving the requested A1 intact.
   - Raw all-read Noisy-OR and alternative mean/max pooling can be optional diagnostics, clearly separated from the primary comparison. Do not select pooling choices using data1/data2.

8. **Classifiers and experiment sequence**

   - Initial estimator families: `LogisticRegression` with numeric standardization; `HistGradientBoostingClassifier`; and `RandomForestClassifier` as a bagged-tree ensemble. They expose probability predictions and allow sample weighting. Consult their official sklearn documentation when pinning versions and constructing estimators.
   - First implement a training-prevalence dummy and all approaches with logistic regression, verifying the pipeline. Then run all three approaches with boosting and random forest. The main comparison contains nine approach/model combinations.
   - Start with one documented configuration per estimator family, held fixed across approaches. Suggested starting points: regularized logistic regression with C=1; histogram boosting with 100 iterations and automatic early stopping disabled; random forest with 100 trees, bounded depth, and a documented minimum leaf size. Final resource settings are recorded before launching the comparison.
   - Disable sklearn's automatic random read-level validation for early stopping. For the initial boosting run, use a fixed iteration count. Any later early-stopping set must be grouped within the training genes.
   - Do not use random-forest out-of-bag read scores as an independent evaluation: other reads from the same site can be in training. Explicit `BaggingClassifier` support may be added to the factory, with the same evaluation rule.
   - A limited tuning round follows only after the complete matrix and diagnostics exist. Give approaches comparable tuning budgets and log all unsuccessful runs. Runtime is unknown until measured; reduce cost through shared sampling/caches and model settings, not by quietly evaluating different sites.

9. **Metrics and interpretation**

   - Calculate all primary metrics at site level on the same validation keys: average precision (AP), ROC AUC, and trapezoidal PR AUC. Keep AP and trapezoidal PR AUC separately named. Use AP as the initial model-selection criterion and retain both course-specified AUC views.
   - Save precision-recall and ROC curves, label prevalence, coverage, fit/prediction time, and prediction distributions. Include log loss and Brier score as calibration diagnostics without implying that read-level scores have known ground truth.
   - If reporting precision, recall, or F1, identify the threshold and whether validation selected it. Such threshold-dependent results remain development estimates.
   - Report A2-minus-A1 and A3-minus-A2 within each classifier, not only the best model from each approach. Paired bootstrap intervals resampling validation genes can assess uncertainty in those differences; they do not correct repeated model-selection bias.
   - Compare score saturation, depth dependence, and motif-specific results to explain failures. Calibration, if later added, must use grouped out-of-fold site predictions inside training and a final evaluation on the unchanged validation partition.
   - A decreasing or non-monotonic performance pattern is a valid result. Possible explanations to investigate include noisy read labels, pooling saturation, low-dimensional sequence compression, and estimator limitations. These are hypotheses until supported by diagnostics.

10. **Modular Python package and exploratory notebooks**

    Proposed structure, not implemented yet:

    ```text
    pyproject.toml / uv.lock / .python-version
    configs/experiments/               # resolved approach/model settings
    src/m6a_project/
      data.py                         # streamed JSON/gzip and site records
      splitting.py                    # saved gene-grouped manifest
      features.py                     # mean9, read9, read15 transforms
      embeddings.py                   # vocabulary and frozen PCA lookup
      sampling.py                     # reproducible per-site read selection
      models.py                       # sklearn estimator factories
      training.py                     # site or inherited-label read training
      pooling.py                      # independent probability aggregators
      evaluation.py                   # site metrics and diagnostics
      tracking.py                     # run lifecycle and artifact logging
      prediction.py                   # load complete bundle and predict sites
    scripts/train.py
    scripts/predict.py
    scripts/run_experiments.py
    notebooks/01_data0_and_split.ipynb
    notebooks/02_classical_comparison.ipynb
    notebooks/03_diagnostics.ipynb
    tests/
    experiments/<experiment_set_id>/<run_id>/
    ```

    - Keep Python 3.12 and uv. Core dependencies: NumPy, pandas, SciPy, scikit-learn, joblib, matplotlib, and PyArrow for cached tables. Notebook and development dependencies live in optional groups. No legacy m6Anet environment is needed for this phase.
    - Feature builder: `fit(training_sites)` and `transform(sites)` return numeric features and site membership. Model factory returns a classifier with `fit(..., sample_weight=...)` and `predict_proba(...)`. Pooler consumes read probabilities plus membership and sampling metadata. The training strategy bridges labels and estimator inputs.
    - Store the signal scaler, embedding transform, estimator, input schema, class mapping, pooling/sampling policy, and training provenance in one versioned model bundle. Loading the classifier alone is insufficient to reproduce a prediction.
    - Notebooks call core functions and the same tracked experiment runner as scripts. They may propose configurations and visualize saved results, but must not contain the only implementation of a transformation or training method.
    - Stream raw reads and cache sampled numeric arrays with offsets or site IDs. Do not create a huge Python-object dataframe of all reads. Cache keys include input hashes, split, feature definition, embedding version, and sampling seed.
    - The later Task 2 extension adds sample manifests and batch prediction around these interfaces. No Task 2 data is accessed now.

11. **Log every experiment and make it reproducible**

    - Use a local file-backed tracker first; no service or external account is required. Every notebook/script experiment goes through `run_experiment(config)` and receives an immutable run ID and an experiment-set ID.
    - Before fitting, save the resolved configuration, purpose/hypothesis, timestamp, input hashes, exact split and read-sampling policy, all seeds, feature names, embedding identity, model parameters, Python/package versions, and uv-lock hash.
    - Record code provenance including Git commit plus dirty diff and any relevant untracked source/config/notebook snapshots. A commit hash alone is insufficient when experimenting with uncommitted code.
    - On completion, save the full model bundle, site-level validation predictions with keys and labels, metrics JSON, curves, diagnostics, fit/prediction timing, and the exact command or configuration needed to rerun. Record the notebook path/source when applicable.
    - Failed or interrupted runs retain status, logs, and errors. Never overwrite a prior result. Maintain an index table across runs for filtering by approach, model, seed, and experiment set.
    - Log cache provenance and sampling settings even when a run reuses cached data. Keep large artifacts out of Git, but document their location and reproducible regeneration. Keep small configs, code, and the experiment index under version control.
    - Verify saved-model predictions agree with original predictions under the recorded sampling policy. Record environment/hardware details and numerical tolerances rather than promising bitwise identity across platforms.
    - MLflow can be added later if the team wants a tracking UI; it is not necessary to satisfy complete local logging in the initial implementation.

12. **Implementation stages and approval boundary**

    - Stage 1: initialize the uv project, data0-only input configuration, frozen split, tracker, and meaningful tests for parsing, group isolation, and deterministic sampling.
    - Stage 2: implement the independent feature builders, fixed sequence embedding, model factory, and stable pooler. Test feature shape/order, pooling edge cases, train-only fitting, and full-bundle serialization.
    - Stage 3: run dummy plus the logistic-regression comparison; inspect leakage checks and Noisy-OR saturation before spending time on larger models.
    - Stage 4: complete boosting and random-forest comparisons, the matched-read A1 control, and the result/diagnostic notebooks. Preserve failures and deviations in the log.
    - Stage 5: verify a clean notebook execution and saved-model prediction on data0 validation JSON. Validate CSV header `transcript_id,transcript_position,score`, exact site coverage, unique keys, finite values, and bounds. No automatic refit on validation, prediction on held-aside folders, or upload is included.
    - Acceptance: the same configuration reruns from CLI or notebook; approaches and estimators can be swapped independently; all primary runs use the same split; every run has complete provenance; and the comparison states its weak-supervision and embedding limitations.
    - Approval of this draft would approve the proposed inherited-label read training, fixed PCA sequence embedding, and repeated 20-read Noisy-OR evaluation. These choices deliberately adapt the paper to the requested classical-only scope. Review them before implementation.

**Public implementation references:** [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Scientific design is grounded in the supplied paper; the non-neural adaptations above are proposals, not claims from that paper.
