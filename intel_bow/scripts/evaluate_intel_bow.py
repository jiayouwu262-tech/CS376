"""Evaluate fitted BoW pipelines on Intel; never fit/tune on Intel test.

All artifact paths in the JSON config are relative to the project root.
Use only trusted teammate joblib/pickle artifacts. Custom tokenizer modules
can be made importable with python_paths. No estimator parameters are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_fitted(estimator, texts, polite_class, vectorizer=None, input_format="text_list",
                   require_text_only_unigrams=False):
    """Use predict() for original decisions; orient decision_function to polite."""
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or np.sum(classes == polite_class) != 1:
        raise ValueError(f"Expected binary classes containing polite_class={polite_class!r}; got {classes!r}")
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if type(final).__name__ not in {"SVC", "LinearSVC", "NuSVC"}:
        raise ValueError(f"Expected an SVM classifier, got {type(final).__name__}")
    components = list(getattr(estimator, "named_steps", {}).values()) + ([vectorizer] if vectorizer is not None else [])
    if require_text_only_unigrams:
        features = estimator.named_steps["features"]
        active = [(name, transform, columns) for name, transform, columns in features.transformers_
                  if not (isinstance(transform, str) and transform == "drop")]
        if len(active) != 1 or active[0][0] != "unigrams" or active[0][2] != "text":
            raise ValueError("Expected only the group's fitted unigram transformer on text")
        components.append(active[0][1])
    if vectorizer is None and not hasattr(estimator, "steps"):
        raise ValueError("A bare SVM requires its original fitted vectorizer or a complete pipeline")
    for component in components:
        if hasattr(component, "ngram_range") and component.ngram_range != (1, 1):
            raise ValueError(f"Not unigram-only: ngram_range={component.ngram_range}")
    if input_format == "text_list":
        inputs = texts
    elif input_format == "dataframe_text":
        inputs = pd.DataFrame({"text": texts})
    else:
        raise ValueError(f"Unknown input_format: {input_format}")
    X = vectorizer.transform(inputs) if vectorizer is not None else inputs
    original_predictions = np.asarray(estimator.predict(X))
    if original_predictions.shape != (len(texts),) or not np.isin(original_predictions, classes).all():
        raise ValueError("Unexpected predicted labels/shape")
    scores = np.asarray(estimator.decision_function(X))
    if scores.shape != (len(texts),) or not np.isfinite(scores).all():
        raise ValueError("Expected finite 1-D continuous binary SVM decision scores")
    # sklearn binary SVM margins point toward classes_[1].
    polite_scores = scores if classes[1] == polite_class else -scores
    predictions = (original_predictions == polite_class).astype(int)
    return predictions, polite_scores, classes.tolist()


def metrics_for(y, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, pos_label=1, zero_division=0),
            "recall": recall_score(y, predictions, pos_label=1, zero_division=0),
            "f1": f1_score(y, predictions, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(y, scores),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/intel_bow_models.json")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    test_path = resolve(config["test_csv"])
    out = resolve(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(test_path)
    required = {"text", "binary_label", "sample_id"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if not df.binary_label.isin([0, 1]).all() or df.binary_label.nunique() != 2:
        raise ValueError("Invalid binary evaluation labels")
    if df.text.isna().any() or not df.sample_id.is_unique or df.sample_id.isna().any():
        raise ValueError("Null input or invalid sample IDs")
    if "label" not in df or not df.label.map({"polite": 1, "impolite": 0}).equals(df.binary_label):
        raise ValueError("Test file is not strict Mapping A")
    rows, runs = [], []
    for spec in config["models"]:
        source = spec["training_source"]
        slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        pred_path = out / f"{slug}_intel_mapping_a_predictions.csv"
        pred = df[[c for c in ["sample_id", "source_row_index", "binary_label"] if c in df]].rename(columns={"binary_label": "true_label"}).copy()
        pred["predicted_label"] = pd.array([pd.NA] * len(df), dtype="Int64")
        pred["polite_score"] = np.nan
        pred["training_source"] = source
        pred["mapping"] = "A"
        row = {"training_source": source, "test_set": test_path.relative_to(ROOT).as_posix() if test_path.is_relative_to(ROOT) else str(test_path),
               "mapping": "A", "n_samples": len(df), "n_impolite_0": int(df.binary_label.eq(0).sum()),
               "n_polite_1": int(df.binary_label.eq(1).sum()), **dict.fromkeys(METRICS, np.nan),
               "status": "pending", "reason": ""}
        run = {"training_source": source, "spec": spec, "test_sha256": fingerprint(test_path)}
        paths = [resolve(spec["model_path"])]
        if spec.get("vectorizer_path"):
            paths.append(resolve(spec["vectorizer_path"]))
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            row.update(status="blocked_missing_model", reason="Missing: " + "; ".join(missing) + ". Supply a complete fitted pipeline OR SVM + original fitted vectorizer, custom tokenizer code, label convention and environment versions.")
        else:
            try:
                for p in spec.get("python_paths", []):
                    sys.path.insert(0, str(resolve(p)))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    estimator = joblib.load(paths[0])
                    vectorizer = joblib.load(paths[1]) if len(paths) > 1 else None
                predictions, scores, classes = predict_fitted(
                    estimator, df.text.tolist(), spec["polite_class"], vectorizer,
                    input_format=spec.get("input_format", "text_list"),
                    require_text_only_unigrams=spec.get("require_text_only_unigrams", False))
                row.update(metrics_for(df.binary_label.to_numpy(), predictions, scores), status="completed")
                pred["predicted_label"] = predictions
                pred["polite_score"] = scores
                run.update(classes=classes, artifact_hashes={p.relative_to(ROOT).as_posix(): fingerprint(p) for p in paths},
                           decision_policy="original predict(); no threshold tuning", score_policy="decision_function oriented to polite_class")
            except Exception as exc:
                row.update(status="blocked_model_error", reason=f"{type(exc).__name__}: {exc}")
        pred["status"] = row["status"]
        pred.to_csv(pred_path, index=False)
        row["prediction_csv"] = pred_path.relative_to(ROOT).as_posix() if pred_path.is_relative_to(ROOT) else str(pred_path)
        run.update(status=row["status"], reason=row["reason"])
        rows.append(row)
        runs.append(run)
    pd.DataFrame(rows).to_csv(out / "metrics_summary.csv", index=False)
    import platform
    import scipy
    provenance = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                  "pandas": pd.__version__, "numpy": np.__version__,
                  "scipy": scipy.__version__, "joblib": joblib.__version__,
                  "source_repository": config.get("source_repository"),
                  "config_path": config_path.relative_to(ROOT).as_posix(), "config_sha256": fingerprint(config_path), "runs": runs}
    (out / "evaluation_manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(rows).drop(columns=["reason"]).to_string(index=False))
    for row in rows:
        if row["reason"]:
            print(row["training_source"] + ": " + row["reason"])
    return 0 if all(row["status"] == "completed" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
