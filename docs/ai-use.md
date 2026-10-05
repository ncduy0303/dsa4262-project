# AI use and verification log

| Tool | Prompt/output | Use and verification |
|---|---|---|
| Codex | Inspect references and data; propose an extensible baseline plan | Read local handout/slides/paper, audited the original inputs, and checked upstream documentation. The user reviewed and revised the plan. |
| Codex | Implement approved data0-only classical experiments with logged runs | Created Python modules, tests, notebooks, configurations, and documentation. Verified grouped partitions, deterministic sampling, saved-model predictions, and submission structure. Model performance is measured from local runs, not inferred from the paper. |

Concrete corrections:

- During the initial audit, filename-based reasoning suggested only data0 contained labels. Reading the actual metadata showed data1 has binary labels and data2 has fractional labels. The audit and plan were corrected. The subsequently approved implementation intentionally uses data0 alone.
- During implementation, an automatically removed unused import became necessary after code-provenance hashing was added. Linting and the CLI startup exposed `NameError: Path is not defined`. The import was restored before model training began. The test suite and full runner then executed successfully.
- An initially available scientific document runtime did not contain BeautifulSoup. The attempted import failed, so the first handout extraction used Python's standard-library HTML parser. No project conclusion was based on the failed extraction.

Verified methodological limits:

- Site-label inheritance is not true read ground truth and is not m6Anet's MIL training objective.
- PCA embeddings are an approved non-neural substitute, not embeddings learned by the paper's neural model.
- Improved performance from approach 1 through approach 3 was a hypothesis. Actual contrary results are retained.

No external code was copied wholesale into the project. The paper supplies the pooling formula and scientific motivation; scikit-learn supplies the classifier, PCA, metrics, and grouped-split implementations. Generated code remains subject to the recorded tests and stated evaluation limitations.

Additional benchmark implementation: the first conda solve chose a CUDA PyTorch package despite a `cpuonly` request, so the official CPU build was selected explicitly. A smoke test then revealed an MKL/PyTorch binary incompatibility. Pinning MKL 2023.2 resolved it without modifying upstream m6Anet. The official output was checked for complete validation coverage, matching read counts and motifs, and probability bounds. Pretraining overlap remains unknown and is stated in the comparison.


## Data0 v2 split and CPU training smoke test

At the user's request, added a three-way stratified gene split with validation fold 0
and test fold 1. Historical v1 runs are preserved; the new test fold was previously
used for v1 training. Added test-safe cache construction, an explicit frozen-model
test evaluator, and a logged one-epoch official m6Anet CPU runner. The runner uses
training-only normalization, random initialization, official MIL components, and
seeded workers. It skips the upstream CLI's automatic test evaluation. Data1 and
data2 remain outside the task. See the smoke report for measured outcomes.


## Full m6Anet training

At the user's request, implemented full CPU training with a prespecified 50-epoch
ceiling, minimum ten epochs, patience five, and validation-loss improvement threshold
0.0005. The absolute lowest validation-loss checkpoint is selected. Added fixed
validation sampling with isolated RNG state, an in-memory read cache verified against
the official dataset, per-epoch histories, selected/latest checkpoints, and reload
prediction verification. These are runner adaptations around official m6Anet model,
loss, oversampling, and training/validation functions. Test remains excluded from
training and selection. Final results are recorded in the full-training report.
