# Intel Mapping A and Mapping B Evaluation

All eight evaluations are complete: four existing Stanford-trained models, each evaluated under both binary mappings. No models were retrained. The original intel_bow/ and intel_ling/ files are preserved.

## Keep the mappings separate

| Folder | Test examples | Class 1 | Class 0 |
|---|---:|---|---|
| [intel_mapping_a](intel_mapping_a/README.md) | 5,099 | polite: 2,567 | impolite: 2,532 |
| [intel_mapping_b](intel_mapping_b/README.md) | 10,200 | polite + somewhat polite: 5,115 | impolite + neutral: 5,085 |

Mapping B uses the entire official test split; Mapping A excludes the two middle labels. B therefore changes both the test population and the binary task definition.

Each folder has its own data, Ling features, four-row metric summary, four prediction CSVs, verification records, English results documentation and reproducible runner. Both reference the existing ../results/ models.

## All measured results

| model | train | mapping | n | accuracy | precision_polite | recall_polite | f1_polite | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BoW | Wikipedia | A | 5099 | 77.60% | 78.93% | 75.73% | 77.30% | 0.859 |
| Ling | Wikipedia | A | 5099 | 83.51% | 83.40% | 83.95% | 83.67% | 0.918 |
| BoW | Stack Exchange | A | 5099 | 69.39% | 67.06% | 77.02% | 71.70% | 0.772 |
| Ling | Stack Exchange | A | 5099 | 74.41% | 71.70% | 81.22% | 76.16% | 0.827 |
| BoW | Wikipedia | B | 10200 | 73.53% | 70.84% | 80.25% | 75.25% | 0.809 |
| Ling | Wikipedia | B | 10200 | 79.21% | 75.31% | 87.08% | 80.77% | 0.868 |
| BoW | Stack Exchange | B | 10200 | 63.68% | 60.04% | 82.44% | 69.48% | 0.698 |
| Ling | Stack Exchange | B | 10200 | 67.18% | 62.68% | 85.40% | 72.29% | 0.741 |

Precision/recall/F1 evaluate class 1 under each mapping. ROC-AUC uses continuous SVM margins oriented toward class 1. Results were independently recomputed from saved per-sample predictions.

Mapping B had zero extraction failures. Its shared 5,099 Mapping A IDs retain exactly the same predicted labels and numerically matching margins under each model.
