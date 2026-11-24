# Intel Mapping A Results

Mapping A excludes somewhat polite and neutral: class 1=polite; class 0=impolite.

Official test: 5,099 samples; class 1=2,567; class 0=2,532.

All four existing Stanford-trained pipelines are reused without training, vocabulary fitting, tuning, or threshold changes. All rows in this table use the same IDs and true labels.

| model | train | test | mapping | eval | n | accuracy | precision_polite | recall_polite | f1_polite | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BoW | Wikipedia | Intel | A | cross-domain | 5099 | 77.60% | 78.93% | 75.73% | 77.30% | 0.859 |
| Ling | Wikipedia | Intel | A | cross-domain | 5099 | 83.51% | 83.40% | 83.95% | 83.67% | 0.918 |
| BoW | Stack Exchange | Intel | A | cross-domain | 5099 | 69.39% | 67.06% | 77.02% | 71.70% | 0.772 |
| Ling | Stack Exchange | Intel | A | cross-domain | 5099 | 74.41% | 71.70% | 81.22% | 76.16% | 0.827 |

Precision/recall/F1 columns retain the group's `*_polite` names, but always evaluate class 1 as defined by the selected mapping. For B, class 1 includes somewhat polite. ROC-AUC uses continuous margins oriented toward class 1. CSV metrics use a 0-1 scale.

## Ling versus BoW

- Wikipedia: accuracy +5.90 percentage points; F1 +6.38 percentage points; AUC +0.0588.
- Stack Exchange: accuracy +5.02 percentage points; F1 +4.47 percentage points; AUC +0.0554.

These results have been independently verified from saved predictions, including rank-sum AUC checks. Shared Mapping A sample IDs retain their original predictions and margins. See `provenance/result_verification.json`.
