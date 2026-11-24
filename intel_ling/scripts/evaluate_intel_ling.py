"""Evaluate the two frozen group Ling pipelines on the aligned Intel features."""
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from ling_common import config, dump, environment, inspect_models, load_test, path, sha


def main():
    cfg = config()
    test = load_test(cfg)
    models, columns, model_info = inspect_models(cfg)
    features_path = path(cfg["features_csv"])
    manifest = json.loads(path("provenance/feature_extraction.json").read_text(encoding="utf-8"))
    assert manifest["test_sha256"] == sha(path(cfg["test_csv"]))
    assert manifest["feature_sha256"] == sha(features_path)
    assert manifest["failures"] == 0 and manifest["strategy_columns"] == columns
    features = pd.read_csv(features_path)
    assert features.sample_id.equals(test.sample_id) and len(features) == 5099
    assert features[columns].notna().all().all() and features[columns].isin([0, 1]).all().all()
    assert features[columns].to_numpy().sum() > 0
    X = pd.concat([test[["text"]].reset_index(drop=True), features[columns].astype("int64").reset_index(drop=True)], axis=1)
    assert X.columns.tolist() == ["text"] + columns
    y = test.binary_label.to_numpy()
    out = path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    metrics, runs = [], []
    for spec in cfg["models"]:
        source = spec["training_source"]
        model = models[source]
        original = sha(path(spec["model_path"]))
        score = np.asarray(model.decision_function(X))
        prediction = np.asarray(model.predict(X))
        assert model.classes_.tolist() == [0, 1]
        assert score.shape == (len(y),) and np.isfinite(score).all()
        np.testing.assert_array_equal(prediction, (score > 0).astype(int))
        matrix = model.named_steps["features"].transform(X)
        # Confirm that the fitted pipeline actually consumes the 21 numeric strategy columns.
        block = matrix[:, -len(columns):]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        np.testing.assert_array_equal(block, features[columns].to_numpy())
        strategy_contribution = block @ model.named_steps["svm"].coef_[0, -len(columns):]
        assert np.count_nonzero(strategy_contribution) > 0
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        name = f"{slug}_intel_mapping_a_predictions.csv"
        result = test[["sample_id", "source_row_index", "binary_label"]].rename(columns={"binary_label": "true_label"}).copy()
        result["predicted_label"] = prediction
        result["polite_score"] = score
        result["strategy_score_component"] = strategy_contribution
        result["training_source"] = source
        result["model"] = "Ling"
        result["mapping"] = "A"
        result["status"] = "completed"
        result.to_csv(out / name, index=False)
        metrics.append({"model": "Ling", "training_source": source, "test_set": "Intel official test",
                        "mapping": "A", "eval": "cross-domain", "n_samples": len(y),
                        "n_polite_1": int((y == 1).sum()), "n_impolite_0": int((y == 0).sum()),
                        "accuracy": accuracy_score(y, prediction), "precision_polite": precision_score(y, prediction, pos_label=1),
                        "recall_polite": recall_score(y, prediction, pos_label=1), "f1_polite": f1_score(y, prediction, pos_label=1),
                        "roc_auc": roc_auc_score(y, score), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                        "prediction_csv": f"{cfg['output_dir']}/{name}", "status": "completed"})
        assert original == sha(path(spec["model_path"]))
        runs.append({**spec, "model_sha256": original, "matrix_shape": list(matrix.shape),
                     "strategy_block_equals_saved_features": True,
                     "rows_with_nonzero_strategy_score": int(np.count_nonzero(strategy_contribution)),
                     "prediction_sha256": sha(out / name), "decision_policy": "original pipeline.predict; matches score>0",
                     "score_policy": "continuous decision_function points to polite=1"})
    summary = pd.DataFrame(metrics)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    dump("provenance/evaluation_manifest.json", {"source_repository": cfg["source_repository"], "environment": environment(),
         "test_sha256": sha(path(cfg["test_csv"])), "feature_sha256": sha(features_path),
         "config_sha256": sha(path("configs/intel_ling_models.json")), "runs": runs})
    print(summary.drop(columns=["prediction_csv"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
