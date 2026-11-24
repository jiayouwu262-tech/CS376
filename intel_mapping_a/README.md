# Intel Mapping A Evaluation

Mapping A excludes somewhat polite and neutral: class 1=polite; class 0=impolite.

This directory contains only this mapping's data, features, four-model results, per-sample predictions, scripts and provenance. The other mapping is in `../intel_mapping_b/`. The original `intel_bow/` and `intel_ling/` work is preserved.

## Results

See [docs/RESULTS.md](docs/RESULTS.md) and [outputs/metrics_summary.csv](outputs/metrics_summary.csv).

## Directory guide

- `data/`: the exact mapped official test and ID/text registry.
- `features/`: Ling feature matrix, extraction audit, failures and prevalence.
- `outputs/`: four-row metric summary and four per-sample prediction CSVs.
- `configs/evaluation.json`: mapping, expected class counts and existing model paths.
- `provenance/`: model hashes, feature-extraction versions and independent verification.

## Reproduce

Run from this directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/run_evaluation.py
```

The runner reuses verified local features by default. Add `--reextract` to regenerate them, or `--reextract --no-source-cache` to parse every text again. Cached Mapping A features are reused only after checking IDs, exact text, versions, columns and hashes; labels are never inputs to extraction. Missing optional caches cause fresh parsing, not missing rows.

Models are loaded from the repository's existing `../results/` files. BoW uses the frozen unigram vectorizer; Ling uses that source's frozen unigrams plus the 21 politeness_api columns. Predictions retain the original `predict()` decisions. No Intel fitting occurs.

Parsing failures are saved and block evaluation rather than being dropped or replaced with zero vectors. Valid no-strategy rows retain zeros. The runtime uses ConvoKit 5.0.0, spaCy 3.8.18, en_core_web_sm 3.8.0 and scikit-learn 1.9.1. Group training parser versions were not locked; original call settings are retained.

A and B change both the test population and the definition of class 1; report them separately. Mapping B is an additional evaluation of the same four Stanford-trained models, not a new training run.

Intel source: https://huggingface.co/datasets/Intel/polite-guard, revision `7b34586171914fa7ab75fd275aca8469e1464a64`. Cleaning and mapping are inherited from the existing prepared CSVs; no new split or cleaning was performed. The declared dataset license is included in `licenses/CDLA-Permissive-2.0.txt`.

## Original Labels in This Test Set

| Original label | Samples | Binary class |
|---|---:|---:|
| polite | 2,567 | 1 |
| impolite | 2,532 | 0 |

## Extraction Audit

All 5,099 test rows have Ling features. Failed rows: 0. Valid all-zero strategy rows: 3.

Reused 5,099 verified text-identical Mapping A feature rows and parsed 0 additional texts. All four classifiers were evaluated on the complete test set for this mapping. Feature reuse does not reuse ground-truth labels or fit any model.
