"""Recompute metrics from prediction CSVs and verify alignment and frozen inference."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ling_common import config, dump, inspect_models, load_test, path, sha


def forbidden_fit(*args, **kwargs):
    raise AssertionError("No model fitting is allowed during Intel evaluation")


def main():
    cfg = config()
    test = load_test(cfg)
    models, columns, _ = inspect_models(cfg)
    feature = pd.read_csv(path(cfg["features_csv"]))
    assert feature.sample_id.equals(test.sample_id)
    X = pd.concat([test[["text"]], feature[columns].astype("int64")], axis=1)
    summary = pd.read_csv(path(cfg["output_dir"]) / "metrics_summary.csv")
    assert len(summary) == 2 and summary.status.eq("completed").all()
    failures = pd.read_csv(path("features/parse_failures.csv"))
    assert failures.empty
    checks = []
    for row in summary.itertuples():
        prediction = pd.read_csv(path(row.prediction_csv))
        assert prediction.sample_id.equals(test.sample_id) and prediction.sample_id.is_unique
        np.testing.assert_array_equal(prediction.true_label, test.binary_label)
        assert len(prediction) == 5099 and prediction.polite_score.notna().all()
        y, p, s = prediction.true_label.to_numpy(), prediction.predicted_label.to_numpy(), prediction.polite_score.to_numpy()
        tp = int(np.sum((y == 1) & (p == 1)))
        tn = int(np.sum((y == 0) & (p == 0)))
        fp = int(np.sum((y == 0) & (p == 1)))
        fn = int(np.sum((y == 1) & (p == 0)))
        npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
        values = {"accuracy": (tp+tn)/len(y), "precision_polite": tp/(tp+fp), "recall_polite": tp/(tp+fn),
                  "f1_polite": 2*tp/(2*tp+fp+fn),
                  "roc_auc": (rankdata(s)[y == 1].sum() - npos*(npos+1)/2)/(npos*nneg)}
        for key, value in values.items():
            assert abs(value - getattr(row, key)) < 1e-12, (key, value, getattr(row, key))
        model = models[row.training_source]
        vectorizer = model.named_steps["features"].named_transformers_["unigrams"]
        vocabulary = dict(vectorizer.vocabulary_)
        weights = model.named_steps["svm"].coef_.copy()
        with patch.object(Pipeline, "fit", forbidden_fit), patch.object(CountVectorizer, "fit", forbidden_fit), \
             patch.object(CountVectorizer, "fit_transform", forbidden_fit), patch.object(LinearSVC, "fit", forbidden_fit):
            actual = model.predict(X)
            actual_scores = model.decision_function(X)
        np.testing.assert_array_equal(actual, p)
        np.testing.assert_allclose(actual_scores, s, rtol=1e-13, atol=1e-13)
        assert vocabulary == vectorizer.vocabulary_
        np.testing.assert_array_equal(weights, model.named_steps["svm"].coef_)
        bow = pd.read_csv(path(cfg["bow_predictions"][row.training_source]))
        assert bow.sample_id.equals(prediction.sample_id)
        np.testing.assert_array_equal(bow.true_label, prediction.true_label)
        checks.append({"training_source": row.training_source, "n": len(y), **values, "status": "PASS", "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                       "ids_and_labels_match_bow": True, "fit_methods_forbidden": True, "frozen_model_unchanged": True})
    preservation_path = path("provenance/bow_preservation.json")
    preserved_count = 0
    if preservation_path.exists():
        preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
        for entry in preservation:
            assert sha(path(entry["path"])) == entry["sha256"], f"Changed BoW file: {entry['path']}"
        preserved_count = len(preservation)
    dump("provenance/result_verification.json", {"status": "PASS", "parse_failures": 0,
         "bow_preserved_file_count": preserved_count, "metrics_independently_recomputed": True,
         "auc_recomputed_via_rank_sum": True, "models": checks})
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
