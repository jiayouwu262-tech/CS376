# Intel Ling Cross-Domain Evaluation

This directory evaluates the Linguistically Informed (Ling) classifiers alongside the existing `intel_bow/` work. It reuses the group's saved Ling pipelines and exactly the same Intel Mapping A official test set, without downloading, cleaning, splitting, or fitting Intel data again.

The four-row BoW/Ling comparison and changes within each training source are in [docs/RESULTS.md](docs/RESULTS.md). Machine-readable results are in [outputs/metrics_summary.csv](outputs/metrics_summary.csv) and [outputs/intel_bow_ling_comparison.csv](outputs/intel_bow_ling_comparison.csv).

## Inputs and Models

- `../intel_bow/data/intel_cleaned/intel_test_mapping_a.csv`: 5,099 examples, with 2,567 polite examples (1) and 2,532 impolite examples (0).
- The BoW sample registry and prediction files: used to verify sample IDs, row order, text, and true labels.
- `../results/model_Ling_wiki.joblib` and `../results/model_Ling_se.joblib`: existing group-trained pipelines, reused without retraining.
- The Wikipedia and Stack Exchange pipelines use 566 and 719 fixed unigram features, respectively, plus 21 politeness_api features. Strategy names and order exactly match `../results/strategy_columns.txt`.
- The original training script is `stanford_replicate.py` at the repository root. It uses `CountVectorizer(min_df=10)` and `LinearSVC(C=1.0, max_iter=20000, random_state=42)`.

## Feature Extraction

Extraction follows the group's default `TextParser(verbosity=1000)` and `PolitenessStrategies()` calls with `markers=False`. Parsing uses `en_core_web_sm` in parse mode with NER disabled, retaining POS tags and dependencies. Only text and identifying sample IDs are supplied to the extractor; labels, source, and reasoning are not features.

ConvoKit's individual-utterance interfaces allow failures to be recorded by sample ID. They use the same core processing as the group's corpus transform calls. An equivalence check on 32 real Intel examples confirmed identical features. A full run first checks 32 examples before processing all 5,099.

For a valid parse, strategies that are not present are explicitly returned as 0 by the original extractor. Missing parses, missing strategy keys, nonbinary values, and exceptions are recorded as failures; their rows are retained and evaluation is blocked. Extraction failures are never replaced with zero vectors. Successfully parsed examples with no detected strategies are recorded separately.

All 5,099 examples were parsed and processed successfully, with **zero failures**. Three valid examples had no detected strategies and retain all-zero strategy vectors. The 32-example API equivalence check passed for all 21 features.

## Files

| Path | Contents |
|---|---|
| `configs/intel_ling_models.json` | Shared BoW inputs, model paths, strategy columns, and extraction settings |
| `features/intel_test_mapping_a_ling_features.csv` | sample_id and 21 binary strategy features |
| `features/smoke_features.csv` | Features for the first 32-example check |
| `features/parse_audit.csv` | Per-sample status, sentence count, token count, and nonzero strategy count |
| `features/parse_failures.csv` | Failed sample IDs, stages, and reasons; header only when there are no failures |
| `features/strategy_prevalence.csv` | Counts and proportions for each strategy |
| `outputs/metrics_summary.csv` | Metrics, sample counts, and confusion matrices for both Ling models |
| `outputs/wikipedia_intel_mapping_a_predictions.csv` | Per-sample Wikipedia-trained Ling predictions |
| `outputs/stack_exchange_intel_mapping_a_predictions.csv` | Per-sample Stack Exchange-trained Ling predictions |
| `outputs/intel_bow_ling_comparison.csv` | Two existing BoW rows and two measured Ling rows |
| `docs/RESULTS.md` | Four-row comparison and changes relative to BoW |
| `provenance/` | Model/input hashes, versions, extraction checks, and result verification |

Prediction files include `sample_id`, `true_label`, `predicted_label`, and `polite_score`, plus the original row index and the strategy component of the linear score. Polite=1 is the positive class. ROC-AUC uses the continuous margin oriented toward that class. Labels come from the original pipeline's `predict()` and were checked against the group's `score > 0` rule.

## Reproduce the Evaluation

Run from `intel_ling/`:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/run_intel_ling.py
```

This extracts features, evaluates both models, verifies predictions, and prints and saves the four-row comparison. On macOS/Linux, replace the Python path with `.venv/bin/python`.

To reuse exported features whose input hashes and runtime versions still match:

```powershell
.venv/Scripts/python.exe scripts/run_intel_ling.py --reuse-features
```

Individual steps and the small-sample check are also available:

```powershell
.venv/Scripts/python.exe scripts/extract_features.py --smoke-only
.venv/Scripts/python.exe scripts/extract_features.py
.venv/Scripts/python.exe scripts/evaluate_intel_ling.py
.venv/Scripts/python.exe scripts/verify_results.py
.venv/Scripts/python.exe scripts/check_corpus_equivalence.py
.venv/Scripts/python.exe scripts/build_comparison.py
```

## Environment and Scope

This run used **ConvoKit 5.0.0**, **spaCy 3.8.18**, **en_core_web_sm 3.8.0**, and **scikit-learn 1.9.1**. Other versions are recorded in `requirements.txt` and `provenance/feature_extraction.json`.

The group repository did not lock the ConvoKit, spaCy, or language-model versions used during Stanford training. This evaluation preserves the original calls and feature definitions and records its actual runtime versions; exact dependency-version equality with training cannot be established.

This work does not rerun Stanford cross-validation or cross-domain scoring, train new models, or use Intel for vocabulary fitting, tuning, or threshold selection. The original paper did not report Intel experiments, so the comparison omits paper_accuracy.
