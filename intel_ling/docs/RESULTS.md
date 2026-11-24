# Intel BoW and Ling Cross-Domain Results

All four rows use the same Intel Mapping A official test set: 5,099 examples, including 2,567 polite examples (1) and 2,532 impolite examples (0). Sample IDs, row order, and true labels were checked for exact alignment.

Accuracy, precision, recall, and F1 are shown as percentages; CSV files retain values on a 0-1 scale. ROC-AUC uses continuous SVM margins oriented toward polite=1 and is displayed to three decimal places.

| model | train | test | eval | n | accuracy | precision_polite | recall_polite | f1_polite | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BoW | Wikipedia | Intel | cross-domain | 5099 | 77.60% | 78.93% | 75.73% | 77.30% | 0.859 |
| Ling | Wikipedia | Intel | cross-domain | 5099 | 83.51% | 83.40% | 83.95% | 83.67% | 0.918 |
| BoW | Stack Exchange | Intel | cross-domain | 5099 | 69.39% | 67.06% | 77.02% | 71.70% | 0.772 |
| Ling | Stack Exchange | Intel | cross-domain | 5099 | 74.41% | 71.70% | 81.22% | 76.16% | 0.827 |

## Changes from BoW Within Each Training Source

- Wikipedia: accuracy +5.90 percentage points; F1 +6.38 percentage points; ROC-AUC +0.0588.
- Stack Exchange: accuracy +5.02 percentage points; F1 +4.47 percentage points; ROC-AUC +0.0554.

## Implementation and Environment

Ling reuses the repository's existing `results/model_Ling_wiki.joblib` and `results/model_Ling_se.joblib` without retraining. Each uses its fixed unigram vocabulary and 21 politeness_api features. Numeric strategy columns retain the fitted model's original names and order, with integer 0/1 values.

Features are extracted from Intel text using the original default `TextParser(verbosity=1000)` settings: parse mode, en_core_web_sm, and NER disabled. `PolitenessStrategies()` uses politeness_api with markers=False. Individual-utterance calls record sample-specific failures and use the same core processing functions as the group's corpus transform calls.

Versions: python 3.13.7, convokit 5.0.0, spacy 3.8.18, en-core-web-sm 3.8.0, scikit-learn 1.9.1, numpy 2.3.2, scipy 1.16.1, pandas 2.3.2, joblib 1.5.2, nltk 3.10.4.

The group repository did not lock the ConvoKit, spaCy, or language-model versions used in training. This evaluation preserves its call settings and pins the actual runtime versions listed above.

Parsing or extraction failures are not replaced with zero features or silently dropped. Failures are recorded in `features/parse_failures.csv`. Valid parses with no detected strategies retain all-zero strategy vectors and are marked separately in `features/parse_audit.csv`. Verification is recorded in `provenance/result_verification.json`.

The two BoW rows are read directly from `../intel_bow/outputs/intel_bow/metrics_summary.csv`; BoW files are not modified or rescored by this comparison. The original paper did not report Intel experiments, so this table does not include paper_accuracy.
