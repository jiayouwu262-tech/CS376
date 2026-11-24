# Intel Mapping B Results

Mapping B keeps all four original categories: class 1=polite or somewhat polite; class 0=impolite or neutral.

Official test: 10,200 samples; class 1=5,115; class 0=5,085.

All four existing Stanford-trained pipelines are reused without training, vocabulary fitting, tuning, or threshold changes. All rows in this table use the same IDs and true labels.

| model | train | test | mapping | eval | n | accuracy | precision_polite | recall_polite | f1_polite | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BoW | Wikipedia | Intel | B | cross-domain | 10200 | 73.53% | 70.84% | 80.25% | 75.25% | 0.809 |
| Ling | Wikipedia | Intel | B | cross-domain | 10200 | 79.21% | 75.31% | 87.08% | 80.77% | 0.868 |
| BoW | Stack Exchange | Intel | B | cross-domain | 10200 | 63.68% | 60.04% | 82.44% | 69.48% | 0.698 |
| Ling | Stack Exchange | Intel | B | cross-domain | 10200 | 67.18% | 62.68% | 85.40% | 72.29% | 0.741 |

Precision/recall/F1 columns retain the group's `*_polite` names, but always evaluate class 1 as defined by the selected mapping. For B, class 1 includes somewhat polite. ROC-AUC uses continuous margins oriented toward class 1. CSV metrics use a 0-1 scale.

## Ling versus BoW

- Wikipedia: accuracy +5.68 percentage points; F1 +5.52 percentage points; AUC +0.0583.
- Stack Exchange: accuracy +3.50 percentage points; F1 +2.82 percentage points; AUC +0.0423.

These results have been independently verified from saved predictions, including rank-sum AUC checks. Shared Mapping A sample IDs retain their original predictions and margins. See `provenance/result_verification.json`.
