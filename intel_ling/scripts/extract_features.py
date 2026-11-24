"""Extract original ConvoKit politeness_api features from exactly the shared text.

The individual-utterance methods call the same processing functions as the
group's Corpus.transform calls, while allowing per-sample failure recording.
"""
import argparse
import inspect
import json
import time

import numpy as np
import pandas as pd

from ling_common import config, dump, environment, inspect_models, load_test, path, relative, sha


def main():
    parser_args = argparse.ArgumentParser(description=__doc__)
    parser_args.add_argument("--smoke-only", action="store_true")
    parser_args.add_argument("--reuse", action="store_true", help="Reuse verified full features with identical provenance")
    args = parser_args.parse_args()
    cfg = config()
    test = load_test(cfg)
    _, columns, model_info = inspect_models(cfg)
    dump("provenance/models.json", {"source_repository": cfg["source_repository"], "models": model_info})
    versions = environment()
    input_hash = sha(path(cfg["test_csv"]))
    feature_path = path(cfg["features_csv"])
    if args.reuse and feature_path.exists() and path("provenance/feature_extraction.json").exists():
        manifest = json.loads(path("provenance/feature_extraction.json").read_text(encoding="utf-8"))
        assert manifest["test_sha256"] == input_hash
        assert manifest["environment"] == versions
        assert manifest["strategy_columns"] == columns
        assert manifest["feature_sha256"] == sha(feature_path)
        assert manifest["failures"] == 0
        cached = pd.read_csv(feature_path)
        assert cached.sample_id.equals(test.sample_id) and cached[columns].notna().all().all()
        print(f"Reused verified features: {len(cached)} rows, {len(columns)} strategies", flush=True)
        return

    import spacy
    from convokit.model import Speaker, Utterance
    from convokit.text_processing.textParser import TextParser
    from convokit.politenessStrategies.politenessStrategies import PolitenessStrategies

    if not spacy.util.is_package(cfg["extraction"]["spacy_model"]):
        raise RuntimeError("Missing spaCy model; run python -m spacy download en_core_web_sm")
    # These are exactly the constructor defaults used by stanford_replicate.py.
    parser = TextParser(verbosity=1000)
    strategies = PolitenessStrategies()
    nlp = parser.aux_input["spacy_nlp"]
    assert parser.mode == cfg["extraction"]["mode"] == "parse"
    assert strategies.strategy_collection == cfg["extraction"]["strategy_collection"] == "politeness_api"
    assert "parser" in nlp.pipe_names and "tagger" in nlp.pipe_names and "ner" not in nlp.pipe_names
    assert nlp.meta["name"] == "core_web_sm"
    speaker = Speaker(id="intel-evaluation")
    errors, records, audit = [], [], []
    start = time.monotonic()

    def extract(row):
        stage = "parse"
        try:
            # Labels, source, reasoning and conversation context are not inputs.
            utt = Utterance(id=row.sample_id, text=row.text, speaker=speaker)
            utt = parser.transform_utterance(utt)
            parsed = utt.meta.get("parsed")
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("Missing or empty dependency parse")
            tokens = [token for sent in parsed for token in sent.get("toks", [])]
            if not tokens or any(not all(key in token for key in ["tok", "tag", "dep", "dn"]) for token in tokens):
                raise ValueError("Incomplete parse tokens/POS/dependencies")
            if any(not token["dep"] or not token["tag"] for token in tokens):
                raise ValueError("Blank POS tag/dependency")
            stage = "strategies"
            utt = strategies.transform_utterance(utt, markers=False)
            features = utt.meta.get("politeness_strategies")
            if not isinstance(features, dict):
                raise ValueError("Missing strategy feature dictionary")
            missing = set(columns) - set(features)
            if missing:
                raise ValueError(f"Missing model-required feature keys: {sorted(missing)}")
            values = [features[column] for column in columns]
            if any(value not in [0, 1, False, True] for value in values):
                raise ValueError("Non-binary or missing strategy value")
            values = [int(value) for value in values]
            records.append({"sample_id": row.sample_id, **dict(zip(columns, values))})
            audit.append({"sample_id": row.sample_id, "status": "ok", "sentences": len(parsed),
                          "tokens": len(tokens), "n_nonzero_strategies": sum(values),
                          "all_zero_valid": sum(values) == 0})
        except Exception as exc:
            errors.append({"sample_id": row.sample_id, "stage": stage, "error_type": type(exc).__name__, "reason": str(exc)})
            records.append({"sample_id": row.sample_id, **dict.fromkeys(columns, np.nan)})
            audit.append({"sample_id": row.sample_id, "status": "failed", "sentences": None,
                          "tokens": None, "n_nonzero_strategies": None, "all_zero_valid": False})

    def save_errors():
        target = path("features/parse_failures.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors, columns=["sample_id", "stage", "error_type", "reason"]).to_csv(target, index=False)

    smoke_n = min(cfg["extraction"]["smoke_n"], len(test))
    for row in test.iloc[:smoke_n].itertuples():
        extract(row)
    save_errors()
    smoke = pd.DataFrame(records)
    smoke.to_csv(path("features/smoke_features.csv"), index=False)
    smoke_counts = smoke[columns].sum()
    dump("provenance/smoke_check.json", {"n": smoke_n, "failures": len(errors),
         "nonzero_feature_values": int(smoke_counts.sum()),
         "strategies_observed": smoke_counts[smoke_counts > 0].astype(int).to_dict(), "sample_ids": smoke.sample_id.tolist()})
    print(f"Smoke check: {smoke_n} rows; failures={len(errors)}; nonzero feature values={int(smoke_counts.sum())}", flush=True)
    if errors or smoke_counts.sum() == 0:
        raise RuntimeError("Smoke check failed or entire strategy block is zero; see features/parse_failures.csv")
    if args.smoke_only:
        return
    for i, row in enumerate(test.iloc[smoke_n:].itertuples(), start=smoke_n + 1):
        extract(row)
        if i % 500 == 0:
            print(f"Extracted {i}/{len(test)} rows; failures={len(errors)}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    save_errors()
    result = pd.DataFrame(records).reindex(columns=["sample_id"] + columns)
    assert result.sample_id.equals(test.sample_id) and len(result) == len(test)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(feature_path, index=False)
    pd.DataFrame(audit).to_csv(path("features/parse_audit.csv"), index=False)
    counts = result[columns].sum()
    pd.DataFrame({"strategy": columns, "n_present": counts.to_numpy(), "prevalence": counts.to_numpy()/len(test)}).to_csv(
        path("features/strategy_prevalence.csv"), index=False)
    valid_zero_ids = [entry["sample_id"] for entry in audit if entry["status"] == "ok" and entry["all_zero_valid"]]
    dump("provenance/feature_extraction.json", {
        "test_csv": cfg["test_csv"], "test_sha256": input_hash, "n_samples": len(test),
        "feature_csv": cfg["features_csv"], "feature_sha256": sha(feature_path), "strategy_columns": columns,
        "strategy_dtype": "integer 0/1; missing extraction is NaN and blocks evaluation",
        "environment": versions, "failures": len(errors), "successful_all_zero_strategy_rows": len(valid_zero_ids),
        "successful_all_zero_sample_ids": valid_zero_ids, "nonzero_feature_values": int(counts.sum()),
        "settings": cfg["extraction"], "spacy_pipeline": nlp.pipe_names, "spacy_disabled": nlp.disabled,
        "spacy_model_meta": {key: nlp.meta.get(key) for key in ["lang", "name", "version", "spacy_version"]},
        "implementation": "ConvoKit TextParser.transform_utterance then PolitenessStrategies.transform_utterance(markers=False); same core functions as group's corpus transform",
        "source_file_hashes": {cls.__module__: sha(inspect.getfile(cls)) for cls in [TextParser, PolitenessStrategies]},
        "elapsed_seconds": time.monotonic()-start,
        "training_environment_note": "Group repository did not lock ConvoKit/spaCy/language-model versions; current run pins actual versions and preserves original call settings."
    })
    if errors:
        raise RuntimeError(f"{len(errors)} samples failed extraction; rows retained, evaluation blocked rather than zero-imputed")
    assert result[columns].notna().all().all() and result[columns].isin([0, 1]).all().all()
    if counts.sum() == 0:
        raise RuntimeError("Entire strategy block is zero")
    print(f"Completed {len(result)} rows; failures=0; valid all-zero rows={len(valid_zero_ids)}", flush=True)


if __name__ == "__main__":
    main()
