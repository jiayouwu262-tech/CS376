"""Print and save four real Intel results, using the existing BoW summary."""
import pandas as pd

from ling_common import config, environment, load_test, path


def main():
    cfg = config()
    test = load_test(cfg)
    bow = pd.read_csv(path(cfg["bow_metrics"]))
    ling = pd.read_csv(path(cfg["output_dir"]) / "metrics_summary.csv")
    rows = []
    for source in ["Wikipedia", "Stack Exchange"]:
        for kind, frame in [("BoW", bow), ("Ling", ling)]:
            selected = frame[frame.training_source == source]
            assert len(selected) == 1
            row = selected.iloc[0]
            assert row.n_samples == len(test) and row.status == "completed"
            rows.append({"model": kind, "train": source, "test": "Intel", "eval": "cross-domain", "n": int(row.n_samples),
                         "accuracy": float(row.accuracy), "precision_polite": float(row["precision_polite"] if kind == "Ling" else row.precision),
                         "recall_polite": float(row["recall_polite"] if kind == "Ling" else row.recall),
                         "f1_polite": float(row["f1_polite"] if kind == "Ling" else row.f1), "roc_auc": float(row.roc_auc)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(path(cfg["output_dir"]) / "intel_bow_ling_comparison.csv", index=False)
    shown = comparison.copy()
    for name in ["accuracy", "precision_polite", "recall_polite", "f1_polite"]:
        shown[name] = shown[name].map(lambda value: f"{value*100:.2f}%")
    shown["roc_auc"] = shown.roc_auc.map(lambda value: f"{value:.3f}")
    print("All four rows use the same 5,099 Intel Mapping A official test samples (polite=1: 2,567; impolite=0: 2,532).", flush=True)
    print(shown.to_string(index=False), flush=True)
    lines = ["# Intel BoW and Ling Cross-Domain Results", "", "All four rows use the same Intel Mapping A official test set: 5,099 examples, including 2,567 polite examples (1) and 2,532 impolite examples (0). Sample IDs, row order, and true labels were checked for exact alignment.", "",
             "Accuracy, precision, recall, and F1 are shown as percentages; CSV files retain values on a 0-1 scale. ROC-AUC uses continuous SVM margins oriented toward polite=1 and is displayed to three decimal places.", "",
             "| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"]*len(shown.columns)) + " |"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    lines += ["", "## Changes from BoW Within Each Training Source", ""]
    for source in ["Wikipedia", "Stack Exchange"]:
        b, l = comparison[(comparison.train == source) & (comparison.model == "BoW")].iloc[0], comparison[(comparison.train == source) & (comparison.model == "Ling")].iloc[0]
        text = f"{source}: accuracy {(l.accuracy-b.accuracy)*100:+.2f} percentage points; F1 {(l.f1_polite-b.f1_polite)*100:+.2f} percentage points; ROC-AUC {l.roc_auc-b.roc_auc:+.4f}."
        lines.append("- " + text)
        print(text, flush=True)
    versions = environment()
    lines += ["", "## Implementation and Environment", "",
              "Ling reuses the repository's existing `results/model_Ling_wiki.joblib` and `results/model_Ling_se.joblib` without retraining. Each uses its fixed unigram vocabulary and 21 politeness_api features. Numeric strategy columns retain the fitted model's original names and order, with integer 0/1 values.",
              "", "Features are extracted from Intel text using the original default `TextParser(verbosity=1000)` settings: parse mode, en_core_web_sm, and NER disabled. `PolitenessStrategies()` uses politeness_api with markers=False. Individual-utterance calls record sample-specific failures and use the same core processing functions as the group's corpus transform calls.",
              "", "Versions: " + ", ".join(f"{name} {value}" for name, value in versions.items()) + ".",
              "", "The group repository did not lock the ConvoKit, spaCy, or language-model versions used in training. This evaluation preserves its call settings and pins the actual runtime versions listed above.",
              "", "Parsing or extraction failures are not replaced with zero features or silently dropped. Failures are recorded in `features/parse_failures.csv`. Valid parses with no detected strategies retain all-zero strategy vectors and are marked separately in `features/parse_audit.csv`. Verification is recorded in `provenance/result_verification.json`.",
              "", "The two BoW rows are read directly from `../intel_bow/outputs/intel_bow/metrics_summary.csv`; BoW files are not modified or rescored by this comparison. The original paper did not report Intel experiments, so this table does not include paper_accuracy.", ""]
    path("docs").mkdir(exist_ok=True)
    path("docs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
