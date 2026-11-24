"""Run four frozen-model Intel evaluations for one explicitly configured mapping.

All code, output messages and documentation are English. No classifier or
vectorizer is fitted. The same runner is copied into separate A/B directories.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]


def path(name):
    return (ROOT / name).resolve()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, value):
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def environment():
    names = ["convokit", "spacy", "en-core-web-sm", "scikit-learn", "numpy", "scipy", "pandas", "joblib", "nltk"]
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in names}}


def load_inputs(cfg):
    test = pd.read_csv(path(cfg["test_csv"]))
    assert sha(path(cfg["test_csv"])) == cfg["test_sha256"]
    assert len(test) == cfg["expected"]["n"]
    assert test.sample_id.is_unique and test.sample_id.notna().all()
    assert test.text.notna().all() and test.text.str.len().gt(0).all()
    assert test.label.map(cfg["label_mapping"]).equals(test.binary_label)
    assert test.binary_label.value_counts().to_dict() == {1: cfg["expected"]["positive"], 0: cfg["expected"]["negative"]}
    overlap = None
    if path(cfg["reference_a_test"]).exists():
        ref = pd.read_csv(path(cfg["reference_a_test"])).set_index("sample_id")
        common = test.set_index("sample_id").loc[ref.index]
        assert common.text.equals(ref.text) and common.binary_label.equals(ref.binary_label)
        overlap = len(ref)
    registry = test[["sample_id", "source_row_index", "text", "label", "binary_label"]].copy()
    registry["text_sha256"] = registry.text.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    registry.to_csv(path("data/sample_registry.csv"), index=False)
    dump("provenance/dataset.json", {"mapping": cfg["mapping"], "n": len(test),
         "original_label_counts": test.label.value_counts().to_dict(),
         "binary_class_counts": {str(key): int(value) for key, value in test.binary_label.value_counts().items()},
         "label_mapping": cfg["label_mapping"], "test_sha256": cfg["test_sha256"],
         "official_split": "test", "new_cleaning_or_splitting": False,
         "mapping_a_text_and_label_overlap_checked": overlap})
    return test


def load_models(cfg):
    columns = path(cfg["strategy_columns_file"]).read_text(encoding="utf-8").splitlines()
    assert len(columns) == 21 and len(set(columns)) == 21
    models, info = {}, []
    for spec in cfg["models"]:
        p = path(spec["path"])
        assert sha(p) == spec["sha256"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(p)
        f, svm = model.named_steps["features"], model.named_steps["svm"]
        transforms = {name: (transformer, selected) for name, transformer, selected in f.transformers_ if name != "remainder"}
        assert transforms["unigrams"][1] == "text" and f.remainder == "drop"
        vec = f.named_transformers_["unigrams"]
        assert vec.min_df == 10 and vec.ngram_range == (1, 1) and not vec.binary
        assert svm.C == 1.0 and svm.max_iter == 20000 and svm.random_state == 42
        assert model.classes_.tolist() == [0, 1]
        if spec["model"] == "Ling":
            assert set(transforms) == {"unigrams", "strategies"}
            assert list(transforms["strategies"][1]) == columns
            assert svm.coef_.shape[1] == len(vec.vocabulary_) + len(columns)
        else:
            assert set(transforms) == {"unigrams"}
            assert svm.coef_.shape[1] == len(vec.vocabulary_)
        models[(spec["model"], spec["train"])] = model
        info.append({**spec, "vocabulary_size": len(vec.vocabulary_), "n_strategies": len(columns) if spec["model"] == "Ling" else 0,
                     "n_features": svm.coef_.shape[1], "svm_params": svm.get_params()})
    dump("provenance/models.json", {"model_revision": cfg["model_revision"], "models": info})
    return models, columns


def extract_features(cfg, test, columns, reuse=True, source_cache=True):
    feature_path = path("features/ling_features.csv")
    manifest_path = path("provenance/feature_extraction.json")
    versions = environment()
    if reuse and feature_path.exists() and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert old["test_sha256"] == sha(path(cfg["test_csv"]))
        assert old["feature_sha256"] == sha(feature_path)
        assert old["environment"] == versions and old["strategy_columns"] == columns and old["failures"] == 0
        features = pd.read_csv(feature_path)
        assert features.sample_id.equals(test.sample_id)
        assert features[columns].isin([0, 1]).all().all()
        print(f"Mapping {cfg['mapping']}: reused {len(features)} verified Ling feature rows.", flush=True)
        return features
    cache, cache_text, cache_note = None, None, "disabled or unavailable"
    if source_cache:
        try:
            spec = cfg["source_feature_cache"]
            manifest = json.loads(path(spec["manifest"]).read_text(encoding="utf-8"))
            assert manifest["environment"] == versions and manifest["strategy_columns"] == columns and manifest["failures"] == 0
            assert manifest["test_sha256"] == sha(path(spec["test"]))
            assert manifest["feature_sha256"] == sha(path(spec["features"]))
            source = pd.read_csv(path(spec["test"]))
            source_features = pd.read_csv(path(spec["features"]))
            assert source.sample_id.equals(source_features.sample_id)
            assert source_features[columns].isin([0, 1]).all().all()
            cache = source_features.set_index("sample_id")
            cache_text = source.set_index("sample_id").text
            cache_note = "verified text-identical Mapping A feature cache"
        except (FileNotFoundError, AssertionError, KeyError, ValueError) as exc:
            cache_note = f"Source cache not used: {type(exc).__name__}: {exc}"
    parser, strategies, speaker, utterance_class = None, None, None, None
    records, audit, errors = [], [], []
    started = time.monotonic()

    def one(row):
        nonlocal parser, strategies, speaker, utterance_class
        stage = "parse"
        try:
            if cache is not None and row.sample_id in cache.index:
                assert cache_text.loc[row.sample_id] == row.text, "Cached ID has different text"
                values = cache.loc[row.sample_id, columns].astype(int).tolist()
                status, origin, n_sentences, n_tokens = "ok", "verified_mapping_a_cache", None, None
            else:
                if parser is None:
                    from convokit.model import Speaker, Utterance
                    from convokit.text_processing.textParser import TextParser
                    from convokit.politenessStrategies.politenessStrategies import PolitenessStrategies
                    parser, strategies = TextParser(verbosity=1000), PolitenessStrategies()
                    assert parser.mode == "parse" and strategies.strategy_collection == "politeness_api"
                    nlp = parser.aux_input["spacy_nlp"]
                    assert nlp.meta["name"] == "core_web_sm" and "parser" in nlp.pipe_names and "ner" not in nlp.pipe_names
                    speaker, utterance_class = Speaker(id="intel-evaluation"), Utterance
                utterance = utterance_class(id=row.sample_id, text=row.text, speaker=speaker)
                utterance = parser.transform_utterance(utterance)
                parsed = utterance.meta.get("parsed")
                if not isinstance(parsed, list) or not parsed:
                    raise ValueError("Missing dependency parse")
                tokens = [token for sent in parsed for token in sent.get("toks", [])]
                if not tokens or any(not all(key in token for key in ["tok", "tag", "dep", "dn"]) for token in tokens):
                    raise ValueError("Incomplete parse tokens")
                if any(not token["tag"] or not token["dep"] for token in tokens):
                    raise ValueError("Blank POS/dependency value")
                stage = "strategies"
                utterance = strategies.transform_utterance(utterance, markers=False)
                values_dict = utterance.meta.get("politeness_strategies")
                if not isinstance(values_dict, dict) or set(columns) - set(values_dict):
                    raise ValueError("Missing expected strategy dictionary/keys")
                values = [values_dict[column] for column in columns]
                if any(value not in [0, 1, False, True] for value in values):
                    raise ValueError("Invalid strategy values")
                values = list(map(int, values))
                status, origin, n_sentences, n_tokens = "ok", "parsed_this_run", len(parsed), len(tokens)
            records.append({"sample_id": row.sample_id, **dict(zip(columns, values))})
            audit.append({"sample_id": row.sample_id, "status": status, "origin": origin, "sentences": n_sentences,
                          "tokens": n_tokens, "n_nonzero_strategies": sum(values), "all_zero_valid": sum(values) == 0})
        except Exception as exc:
            errors.append({"sample_id": row.sample_id, "stage": stage, "error_type": type(exc).__name__, "reason": str(exc)})
            records.append({"sample_id": row.sample_id, **dict.fromkeys(columns, np.nan)})
            audit.append({"sample_id": row.sample_id, "status": "failed", "origin": "failed", "sentences": None,
                          "tokens": None, "n_nonzero_strategies": None, "all_zero_valid": False})

    for i, row in enumerate(test.itertuples(), start=1):
        one(row)
        if i == 32:
            smoke = pd.DataFrame(records)
            smoke.to_csv(path("features/smoke_features.csv"), index=False)
            n_nonzero = int(smoke[columns].sum().sum())
            dump("provenance/smoke_check.json", {"n": i, "failures": len(errors), "nonzero_feature_values": n_nonzero})
            print(f"Mapping {cfg['mapping']} smoke check: 32 rows, {len(errors)} failures, {n_nonzero} nonzero strategy values.", flush=True)
            if errors or not n_nonzero:
                pd.DataFrame(errors).to_csv(path("features/parse_failures.csv"), index=False)
                raise RuntimeError("Smoke check failed; evaluation blocked")
        if i % 1000 == 0:
            print(f"Mapping {cfg['mapping']}: extracted {i}/{len(test)} feature rows; failures={len(errors)}.", flush=True)
    features = pd.DataFrame(records).reindex(columns=["sample_id"] + columns)
    assert features.sample_id.equals(test.sample_id)
    features.to_csv(feature_path, index=False)
    pd.DataFrame(audit).to_csv(path("features/parse_audit.csv"), index=False)
    pd.DataFrame(errors, columns=["sample_id", "stage", "error_type", "reason"]).to_csv(path("features/parse_failures.csv"), index=False)
    present = features[columns].sum()
    pd.DataFrame({"strategy": columns, "n_present": present.to_numpy(), "prevalence": present.to_numpy()/len(test)}).to_csv(path("features/strategy_prevalence.csv"), index=False)
    cached_n = sum(item["origin"] == "verified_mapping_a_cache" for item in audit)
    dump("provenance/feature_extraction.json", {"mapping": cfg["mapping"], "n_samples": len(test),
         "test_sha256": sha(path(cfg["test_csv"])), "feature_sha256": sha(feature_path), "environment": versions,
         "strategy_columns": columns, "failures": len(errors), "cached_mapping_a_rows": cached_n,
         "newly_parsed_rows": sum(item["origin"] == "parsed_this_run" for item in audit),
         "source_cache_note": cache_note, "successful_all_zero_rows": sum(item["all_zero_valid"] for item in audit),
         "settings": "TextParser(verbosity=1000), parse mode, en_core_web_sm, NER disabled; PolitenessStrategies(), politeness_api, markers=False",
         "training_environment_note": "Group training parser versions were not locked; original calls are retained and actual runtime versions are pinned.",
         "elapsed_seconds": time.monotonic()-started})
    if errors or not present.sum():
        raise RuntimeError(f"Feature extraction failed for {len(errors)} rows; no rows dropped or zero-imputed")
    return features


def forbid_fit(*args, **kwargs):
    raise AssertionError("Intel must not be used to fit classifiers or vectorizers")


def evaluate(cfg, test, models, features, columns):
    y = test.binary_label.to_numpy()
    results, checks, runs = [], [], []
    for spec in cfg["models"]:
        kind, source = spec["model"], spec["train"]
        model = models[(kind, source)]
        X = test[["text"]].copy()
        if kind == "Ling":
            X = pd.concat([X, features[columns].astype("int64")], axis=1)
        vec = model.named_steps["features"].named_transformers_["unigrams"]
        vocabulary, weights = dict(vec.vocabulary_), model.named_steps["svm"].coef_.copy()
        with patch.object(Pipeline, "fit", forbid_fit), patch.object(CountVectorizer, "fit", forbid_fit), \
                patch.object(CountVectorizer, "fit_transform", forbid_fit), patch.object(LinearSVC, "fit", forbid_fit):
            prediction = model.predict(X)
            score = model.decision_function(X)
            if kind == "Ling":
                matrix = model.named_steps["features"].transform(X)
                block = matrix[:, -len(columns):]
                block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
                np.testing.assert_array_equal(block, features[columns].to_numpy())
        assert np.isfinite(score).all() and score.shape == (len(test),)
        np.testing.assert_array_equal(prediction, (score > 0).astype(int))
        assert vocabulary == vec.vocabulary_
        np.testing.assert_array_equal(weights, model.named_steps["svm"].coef_)
        assert sha(path(spec["path"])) == spec["sha256"]
        name = f"outputs/{kind.lower()}_{source.lower().replace(' ', '_')}_mapping_{cfg['mapping'].lower()}_predictions.csv"
        output = test[["sample_id", "source_row_index", "binary_label"]].rename(columns={"binary_label": "true_label"}).copy()
        output["predicted_label"], output["polite_score"] = prediction, score
        output["model"], output["train"], output["mapping"], output["status"] = kind, source, cfg["mapping"], "completed"
        output.to_csv(path(name), index=False)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        row = {"model": kind, "train": source, "test": "Intel", "mapping": cfg["mapping"], "eval": "cross-domain",
               "n": len(test), "n_positive_1": int((y == 1).sum()), "n_negative_0": int((y == 0).sum()),
               "accuracy": accuracy_score(y, prediction), "precision_polite": precision_score(y, prediction),
               "recall_polite": recall_score(y, prediction), "f1_polite": f1_score(y, prediction), "roc_auc": roc_auc_score(y, score),
               "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "prediction_csv": name, "status": "completed"}
        results.append(row)
        # Independently recompute every metric from the exported prediction rows.
        saved = pd.read_csv(path(name))
        assert saved.sample_id.equals(test.sample_id) and saved.true_label.equals(test.binary_label.rename("true_label"))
        p, s = saved.predicted_label.to_numpy(), saved.polite_score.to_numpy()
        tp2, tn2 = int(np.sum((y == 1) & (p == 1))), int(np.sum((y == 0) & (p == 0)))
        fp2, fn2 = int(np.sum((y == 0) & (p == 1))), int(np.sum((y == 1) & (p == 0)))
        npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
        independent = {"accuracy": (tp2+tn2)/len(y), "precision_polite": tp2/(tp2+fp2), "recall_polite": tp2/(tp2+fn2),
                       "f1_polite": 2*tp2/(2*tp2+fp2+fn2), "roc_auc": (rankdata(s)[y == 1].sum()-npos*(npos+1)/2)/(npos*nneg)}
        for metric, value in independent.items():
            assert abs(value-row[metric]) < 1e-12, (kind, source, metric)
        reference = path(cfg["reference_a_predictions"][f"{kind}|{source}"])
        overlap = 0
        if reference.exists():
            old = pd.read_csv(reference).set_index("sample_id")
            current = saved.set_index("sample_id").loc[old.index]
            np.testing.assert_array_equal(old.true_label, current.true_label)
            np.testing.assert_array_equal(old.predicted_label, current.predicted_label)
            np.testing.assert_allclose(old.polite_score, current.polite_score, rtol=1e-12, atol=1e-12)
            overlap = len(old)
        checks.append({"model": kind, "train": source, "status": "PASS", "n": len(saved),
                       "auc_independently_checked_by_rank_sum": True, "fit_forbidden": True,
                       "mapping_a_reference_predictions_unchanged_on_shared_ids": overlap, **independent})
        runs.append({**spec, "prediction_csv": name, "prediction_sha256": sha(path(name))})
    summary = pd.DataFrame(results)
    summary.to_csv(path("outputs/metrics_summary.csv"), index=False)
    dump("provenance/result_verification.json", {"status": "PASS", "mapping": cfg["mapping"], "checks": checks})
    dump("provenance/evaluation_manifest.json", {"mapping": cfg["mapping"], "environment": environment(),
         "test_sha256": sha(path(cfg["test_csv"])), "config_sha256": sha(path("configs/evaluation.json")), "runs": runs})
    return summary


def report(cfg, summary):
    shown = summary[["model", "train", "test", "mapping", "eval", "n", "accuracy", "precision_polite", "recall_polite", "f1_polite", "roc_auc"]].copy()
    for column in ["accuracy", "precision_polite", "recall_polite", "f1_polite"]:
        shown[column] = shown[column].map(lambda value: f"{value*100:.2f}%")
    shown["roc_auc"] = shown.roc_auc.map(lambda value: f"{value:.3f}")
    print(shown.to_string(index=False), flush=True)
    explanation = ("Mapping A excludes somewhat polite and neutral: class 1=polite; class 0=impolite."
                   if cfg["mapping"] == "A" else "Mapping B keeps all four original categories: class 1=polite or somewhat polite; class 0=impolite or neutral.")
    lines = [f"# Intel Mapping {cfg['mapping']} Results", "", explanation, "",
             f"Official test: {cfg['expected']['n']:,} samples; class 1={cfg['expected']['positive']:,}; class 0={cfg['expected']['negative']:,}.", "",
             "All four existing Stanford-trained pipelines are reused without training, vocabulary fitting, tuning, or threshold changes. All rows in this table use the same IDs and true labels.", "",
             "| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"]*len(shown.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    lines += ["", "Precision/recall/F1 columns retain the group's `*_polite` names, but always evaluate class 1 as defined by the selected mapping. For B, class 1 includes somewhat polite. ROC-AUC uses continuous margins oriented toward class 1. CSV metrics use a 0-1 scale.", "", "## Ling versus BoW", ""]
    for source in ["Wikipedia", "Stack Exchange"]:
        b = summary[(summary.model == "BoW") & (summary.train == source)].iloc[0]
        l = summary[(summary.model == "Ling") & (summary.train == source)].iloc[0]
        lines.append(f"- {source}: accuracy {(l.accuracy-b.accuracy)*100:+.2f} percentage points; F1 {(l.f1_polite-b.f1_polite)*100:+.2f} percentage points; AUC {l.roc_auc-b.roc_auc:+.4f}.")
    lines += ["", "These results have been independently verified from saved predictions, including rank-sum AUC checks. Shared Mapping A sample IDs retain their original predictions and margins. See `provenance/result_verification.json`.", ""]
    path("docs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    other = "b" if cfg["mapping"] == "A" else "a"
    readme = [f"# Intel Mapping {cfg['mapping']} Evaluation", "", explanation, "",
              "This directory contains only this mapping's data, features, four-model results, per-sample predictions, scripts and provenance. The other mapping is in `../intel_mapping_"+other+"/`. The original `intel_bow/` and `intel_ling/` work is preserved.", "",
              "## Results", "", "See [docs/RESULTS.md](docs/RESULTS.md) and [outputs/metrics_summary.csv](outputs/metrics_summary.csv).", "",
              "## Directory guide", "", "- `data/`: the exact mapped official test and ID/text registry.",
              "- `features/`: Ling feature matrix, extraction audit, failures and prevalence.",
              "- `outputs/`: four-row metric summary and four per-sample prediction CSVs.",
              "- `configs/evaluation.json`: mapping, expected class counts and existing model paths.",
              "- `provenance/`: model hashes, feature-extraction versions and independent verification.", "",
              "## Reproduce", "", "Run from this directory:", "", "```powershell", "python -m venv .venv", ".venv/Scripts/python.exe -m pip install -r requirements.txt", ".venv/Scripts/python.exe scripts/run_evaluation.py", "```", "",
              "The runner reuses verified local features by default. Add `--reextract` to regenerate them, or `--reextract --no-source-cache` to parse every text again. Cached Mapping A features are reused only after checking IDs, exact text, versions, columns and hashes; labels are never inputs to extraction. Missing optional caches cause fresh parsing, not missing rows.", "",
              "Models are loaded from the repository's existing `../results/` files. BoW uses the frozen unigram vectorizer; Ling uses that source's frozen unigrams plus the 21 politeness_api columns. Predictions retain the original `predict()` decisions. No Intel fitting occurs.", "",
              "Parsing failures are saved and block evaluation rather than being dropped or replaced with zero vectors. Valid no-strategy rows retain zeros. The runtime uses ConvoKit 5.0.0, spaCy 3.8.18, en_core_web_sm 3.8.0 and scikit-learn 1.9.1. Group training parser versions were not locked; original call settings are retained.", "",
              "A and B change both the test population and the definition of class 1; report them separately. Mapping B is an additional evaluation of the same four Stanford-trained models, not a new training run.", "",
              f"Intel source: https://huggingface.co/datasets/Intel/polite-guard, revision `{cfg['intel_revision']}`. Cleaning and mapping are inherited from the existing prepared CSVs; no new split or cleaning was performed. The declared dataset license is included in `licenses/CDLA-Permissive-2.0.txt`.", ""]
    counts = pd.read_csv(path(cfg["test_csv"])).label.value_counts().to_dict()
    readme += ["## Original Labels in This Test Set", "", "| Original label | Samples | Binary class |", "|---|---:|---:|"]
    for label in ["polite", "somewhat polite", "neutral", "impolite"]:
        if label in counts:
            readme.append(f"| {label} | {counts[label]:,} | {cfg['label_mapping'][label]} |")
    readme += [""]
    extraction = json.loads(path("provenance/feature_extraction.json").read_text(encoding="utf-8"))
    readme += ["## Extraction Audit", "",
               f"All {len(pd.read_csv(path(cfg['test_csv']))):,} test rows have Ling features. Failed rows: {extraction['failures']}. Valid all-zero strategy rows: {extraction['successful_all_zero_rows']}.", "",
               f"Reused {extraction['cached_mapping_a_rows']:,} verified text-identical Mapping A feature rows and parsed {extraction['newly_parsed_rows']:,} additional texts. All four classifiers were evaluated on the complete test set for this mapping. Feature reuse does not reuse ground-truth labels or fit any model.", ""]
    path("README.md").write_text("\n".join(readme), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--no-source-cache", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(path("configs/evaluation.json").read_text(encoding="utf-8"))
    test = load_inputs(cfg)
    models, columns = load_models(cfg)
    features = extract_features(cfg, test, columns, reuse=not args.reextract, source_cache=not args.no_source_cache)
    summary = evaluate(cfg, test, models, features, columns)
    report(cfg, summary)


if __name__ == "__main__":
    main()
