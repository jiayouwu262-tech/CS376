"""
Replication of Danescu-Niculescu-Mizil et al. (2013), Table 4.

Trains two SVMs on the Stanford Politeness Corpus (via ConvoKit):
  - BoW:  unigrams only
  - Ling: unigrams + politeness-strategy features
and evaluates them in-domain (cross-validation) and cross-domain
(Wikipedia <-> Stack Exchange). Also saves the fitted models so they can be
applied to Intel Polite Guard without retraining.

Setup (once):
    pip install convokit scikit-learn pandas joblib
    python -m spacy download en_core_web_sm

Run:
    python replicate_stanford.py            # 10-fold CV (team plan)
    python replicate_stanford.py --loo      # leave-one-out, as in the paper (slower)

Outputs (in ./results):
    replication_results.csv   our numbers next to the paper's
    top_features_<model>_<domain>.csv   largest positive/negative SVM weights
    model_<model>_<domain>.joblib       fitted pipelines for the Intel step
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from convokit import Corpus, PolitenessStrategies, TextParser, download
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
OUT_DIR = "results"

# Accuracies reported in Table 4 of the original paper.
PAPER = {
    ("BoW", "wiki", "wiki"): 79.84, ("BoW", "se", "se"): 74.47,
    ("BoW", "wiki", "se"): 64.23, ("BoW", "se", "wiki"): 72.17,
    ("Ling", "wiki", "wiki"): 83.79, ("Ling", "se", "se"): 78.19,
    ("Ling", "wiki", "se"): 67.53, ("Ling", "se", "wiki"): 75.43,
}
CORPORA = {"wiki": "wikipedia-politeness-corpus", "se": "stack-exchange-politeness-corpus"}
EXPECTED_PER_CLASS = {"wiki": 1089, "se": 1651}


# Data

def load_domain(name):
    """Return a DataFrame with text, label (1 polite / 0 impolite) and strategy features."""
    corpus = Corpus(filename=download(CORPORA[name]))

    # Politeness strategies need dependency parses. Some ConvoKit releases ship
    # them with the corpus; parse only if they are missing.
    first = next(corpus.iter_utterances())
    if "parsed" not in first.meta:
        corpus = TextParser(verbosity=1000).transform(corpus)
    corpus = PolitenessStrategies().transform(corpus, markers=False)

    rows = []
    for utt in corpus.iter_utterances():
        row = {"id": utt.id, "text": utt.text,
               "score": utt.meta.get("Normalized Score"),
               "binary": utt.meta.get("Binary")}
        row.update(utt.meta["politeness_strategies"])
        rows.append(row)
    df = pd.DataFrame(rows)

    # Polite = top quartile, impolite = bottom quartile (paper, Sec. 4).
    # Use ConvoKit's Binary label if it reproduces the paper's class sizes;
    # otherwise rebuild the quartiles from the normalized score.
    kept = df[df["binary"].isin([1, -1])]
    counts = kept["binary"].value_counts().to_dict()
    if counts.get(1) == counts.get(-1) == EXPECTED_PER_CLASS[name]:
        df = kept.copy()
        df["label"] = (df["binary"] == 1).astype(int)
        source = "ConvoKit Binary label"
    else:
        lo, hi = df["score"].quantile([0.25, 0.75])
        df = df[(df["score"] <= lo) | (df["score"] >= hi)].copy()
        df["label"] = (df["score"] >= hi).astype(int)
        source = "quartiles of Normalized Score"

    n_pol, n_imp = int(df["label"].sum()), int((1 - df["label"]).sum())
    print(f"[{name}] {n_pol} polite / {n_imp} impolite (from {source}); "
          f"paper: {EXPECTED_PER_CLASS[name]} per class")
    return df.reset_index(drop=True)


# Models

def make_model(kind, strategy_cols):
    # Paper drops unigrams seen < 10 times; min_df=10 (document frequency) approximates this.
    unigrams = ("unigrams", CountVectorizer(min_df=10), "text")
    if kind == "BoW":
        features = ColumnTransformer([unigrams])
    else:
        features = ColumnTransformer([unigrams, ("strategies", "passthrough", strategy_cols)])
    return Pipeline([("features", features),
                     ("svm", LinearSVC(C=1.0, max_iter=20000, random_state=SEED))])


def score(y, pred, decision):
    return {"accuracy": 100 * accuracy_score(y, pred),
            "precision_polite": precision_score(y, pred),
            "recall_polite": recall_score(y, pred),
            "f1_polite": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, decision)}


def top_features(model, n=20):
    names = model.named_steps["features"].get_feature_names_out()
    weights = model.named_steps["svm"].coef_.ravel()
    order = np.argsort(weights)
    rows = [(names[i], weights[i], "impolite") for i in order[:n]]
    rows += [(names[i], weights[i], "polite") for i in order[::-1][:n]]
    return pd.DataFrame(rows, columns=["feature", "weight", "direction"])


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo", action="store_true",
                        help="leave-one-out CV in-domain, as in the paper")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    data = {name: load_domain(name) for name in CORPORA}
    # Use the strategy features present in both domains, in a fixed order.
    strategy_cols = sorted(c for c in data["wiki"].columns
                           if c.startswith("feature_politeness") and c in data["se"].columns)
    print(f"{len(strategy_cols)} politeness-strategy features")

    cv = LeaveOneOut() if args.loo else StratifiedKFold(10, shuffle=True, random_state=SEED)
    cv_name = "leave-one-out" if args.loo else "10-fold"
    results = []

    for kind in ["BoW", "Ling"]:
        for train in CORPORA:
            df = data[train]
            X, y = df[["text"] + strategy_cols], df["label"].values

            # In-domain: cross-validation. The vectorizer is refit inside each
            # fold by the Pipeline, so no test vocabulary leaks into training.
            model = make_model(kind, strategy_cols)
            dec = cross_val_predict(model, X, y, cv=cv, method="decision_function", n_jobs=-1)
            results.append({"model": kind, "train": train, "test": train,
                            "eval": cv_name, **score(y, (dec > 0).astype(int), dec)})

            # Fit on the full source domain: used for cross-domain tests,
            # feature analysis, and later for Intel Polite Guard.
            model.fit(X, y)
            joblib.dump(model, os.path.join(OUT_DIR, f"model_{kind}_{train}.joblib"))
            top_features(model).to_csv(
                os.path.join(OUT_DIR, f"top_features_{kind}_{train}.csv"), index=False)

            for test in CORPORA:
                if test == train:
                    continue
                t = data[test]
                dec = model.decision_function(t[["text"] + strategy_cols])
                results.append({"model": kind, "train": train, "test": test,
                                "eval": "cross-domain",
                                **score(t["label"].values, (dec > 0).astype(int), dec)})

    out = pd.DataFrame(results)
    out["paper_accuracy"] = [PAPER[(r.model, r.train, r.test)] for r in out.itertuples()]
    out["difference"] = out["accuracy"] - out["paper_accuracy"]
    out.to_csv(os.path.join(OUT_DIR, "replication_results.csv"), index=False)

    pd.set_option("display.width", 140)
    print(out[["model", "train", "test", "eval", "accuracy", "paper_accuracy",
               "difference", "f1_polite", "roc_auc"]].round(2).to_string(index=False))
    with open(os.path.join(OUT_DIR, "strategy_columns.txt"), "w") as f:
        f.write("\n".join(strategy_cols))


if __name__ == "__main__":
    main()
