"""Shared input, model and provenance checks for the Intel Ling evaluation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parents[1]


def path(relative):
    return (ROOT / relative).resolve()


def relative(p):
    return Path(os.path.relpath(p, ROOT)).as_posix()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, value):
    target = path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config():
    return json.loads(path("configs/intel_ling_models.json").read_text(encoding="utf-8"))


def environment():
    names = ["convokit", "spacy", "en-core-web-sm", "scikit-learn", "numpy", "scipy", "pandas", "joblib", "nltk"]
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in names}}


def load_test(cfg):
    df = pd.read_csv(path(cfg["test_csv"]))
    if len(df) != 5099 or df.binary_label.value_counts().to_dict() != {1: 2567, 0: 2532}:
        raise ValueError("Intel Mapping A row/class counts do not match the shared experiment")
    assert df.sample_id.is_unique and df.sample_id.notna().all()
    assert df.text.notna().all() and df.text.str.len().gt(0).all()
    assert df.label.map({"polite": 1, "impolite": 0}).equals(df.binary_label)
    registry = pd.read_csv(path(cfg["bow_registry"]))
    pd.testing.assert_series_equal(df.sample_id, registry.sample_id)
    pd.testing.assert_series_equal(df.binary_label, registry.binary_label)
    pd.testing.assert_series_equal(df.text, registry.text)
    if "text_sha256" in registry:
        assert df.text.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()).equals(registry.text_sha256)
    for source, name in cfg["bow_predictions"].items():
        prediction = pd.read_csv(path(name))
        pd.testing.assert_series_equal(df.sample_id, prediction.sample_id)
        np.testing.assert_array_equal(df.binary_label, prediction.true_label)
        assert len(prediction) == len(df) and prediction.status.eq("completed").all()
    return df


def inspect_models(cfg):
    expected = path(cfg["strategy_columns_file"]).read_text(encoding="utf-8").splitlines()
    models, info = {}, []
    for spec in cfg["models"]:
        artifact = path(spec["model_path"])
        assert sha(artifact) == spec["sha256"], f"Changed model artifact: {artifact}"
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(artifact)
        features, svm = model.named_steps["features"], model.named_steps["svm"]
        transformers = {name: (transformer, columns) for name, transformer, columns in features.transformers_}
        assert set(name for name in transformers if name != "remainder") == {"unigrams", "strategies"}
        vectorizer, text_column = transformers["unigrams"]
        _, strategy_columns = transformers["strategies"]
        assert text_column == "text" and list(strategy_columns) == expected
        assert features.remainder == "drop" and vectorizer.ngram_range == (1, 1)
        assert model.classes_.tolist() == [0, 1] and spec["polite_class"] == 1
        assert type(svm).__name__ == "LinearSVC"
        assert svm.coef_.shape == (1, len(vectorizer.vocabulary_) + len(expected))
        weights = svm.coef_[0, -len(expected):]
        assert np.count_nonzero(weights) > 0, "Ling strategy weights are all zero"
        models[spec["training_source"]] = model
        info.append({**spec, "vocabulary_size": len(vectorizer.vocabulary_), "strategy_columns": expected,
                     "n_strategy_columns": len(expected), "svm_feature_count": svm.coef_.shape[1],
                     "strategy_weights": dict(zip(expected, map(float, weights))),
                     "strategy_weights_nonzero": int(np.count_nonzero(weights)),
                     "classes": model.classes_.tolist(), "svm_params": svm.get_params(),
                     "vectorizer_params": {k: str(v) if k == "dtype" else v for k, v in vectorizer.get_params().items()}})
    return models, expected, info
