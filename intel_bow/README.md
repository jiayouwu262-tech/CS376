# Intel BoW Cross-Domain Evaluation

This directory adds **Intel data preparation and external BoW evaluation** to the group's Stanford replication. It loads the existing `../results/model_BoW_wiki.joblib` and `../results/model_BoW_se.joblib` pipelines directly. The models are reused without retraining.

## Results

Both models are evaluated on the same official Intel Mapping A test set: **5,099 examples**, including **2,567 polite examples (1)** and **2,532 impolite examples (0)**.

| Training source | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Wikipedia | 77.60% | 0.7893 | 0.7573 | 0.7730 | 0.8587 |
| Stack Exchange | 69.39% | 0.6706 | 0.7702 | 0.7170 | 0.7717 |

Precision, recall, and F1 use polite=1 as the positive class. ROC-AUC uses continuous SVM margins oriented toward polite=1. All metrics in the CSV files are stored as values between 0 and 1.

The models retain the group's `CountVectorizer(min_df=10)` unigram counts and `LinearSVC(C=1.0, max_iter=20000, random_state=42)`. The corresponding Stanford reference accuracies are 59.72% for Wiki-to-SE and 71.07% for SE-to-Wiki. Those Stanford experiments were not rerun here. Only `text` is passed to the models; Intel data is not used for vocabulary fitting, training, or tuning.

## Files

- `data/intel_cleaned/intel_test_mapping_a.csv`: the test set used by both models.
- `outputs/intel_bow/dataset_counts.csv`: row counts, binary class counts, and SHA256 hashes for all 14 prepared CSVs. Only the test CSV required for the main evaluation is included here; the preparation script can regenerate the other files.
- `outputs/intel_bow/metrics_summary.csv`: complete metrics and confusion matrices for both models.
- `outputs/intel_bow/*_predictions.csv`: sample IDs, true labels, predicted labels, and continuous scores.
- `outputs/intel_bow/intel_mapping_a_sample_registry.csv`: IDs, original row indices, text, and text hashes for pairing with Ling predictions.
- `scripts/evaluate_intel_bow.py`: evaluates the fitted pipelines using transform, predict, and decision_function.
- `scripts/prepare_intel.py`: an enhanced copy of the repository's original `clean_intel.py`, adding stable IDs and output-path arguments while preserving its cleaning, mapping, and sampling rules. The original root-level script is retained.
- `configs/intel_bow_models.json`: references the existing models in the repository's `results/` directory. Keep this directory structure when moving or uploading files.

## Reproduce the Evaluation

Run from the repository's `intel_bow/` directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/evaluate_intel_bow.py --config configs/intel_bow_models.json
```

The models were saved with scikit-learn 1.9.1; use the same version to load them. Each pipeline already contains its fitted vectorizer. ConvoKit and spaCy are not required for this BoW evaluation.

## Cleaning and Sample IDs

The data comes from [Intel/polite-guard](https://huggingface.co/datasets/Intel/polite-guard), revision `7b34586171914fa7ab75fd275aca8469e1464a64`. Cleaning strips leading and trailing whitespace, collapses repeated whitespace, removes duplicate texts within each split, and removes train/test overlap from training only. Capitalization and punctuation are preserved.

The dataset declares the CDLA-Permissive-2.0 license. Its full text is included in `licenses/CDLA-Permissive-2.0.txt`.

The cleaned train/validation/test sizes are 79,997/10,000/10,200. Mapping A retains only polite=1 and impolite=0, yielding 5,099 test examples. Mapping B and the four balanced training samples are preparation outputs; they are not used for training or the main evaluation in this directory.

The upstream data has no sample IDs. Before filtering, the script creates IDs in the form `intel_{original_split}_{zero_based_original_row:06d}`. Prediction files retain these IDs and their row order. Pair BoW and Ling results by these shared IDs rather than independently renumbered rows.

To regenerate all prepared CSVs from the pinned dataset revision:

```powershell
.venv/Scripts/python.exe scripts/prepare_intel.py --revision 7b34586171914fa7ab75fd275aca8469e1464a64 --output-dir data/intel_cleaned --cache-dir data/hf_cache
```

This preparation command requires internet access. Repeating the evaluation only requires the included test CSV and the models already in the repository.
