"""Confirm the row-wise extractor matches the group's Corpus.transform API."""
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from convokit.model import Corpus, Utterance, Speaker
from convokit.convokitConfig import ConvoKitConfig
from convokit.text_processing.textParser import TextParser
from convokit.politenessStrategies.politenessStrategies import PolitenessStrategies

from ling_common import config, dump, inspect_models, load_test, path


def main():
    cfg = config()
    test = load_test(cfg).iloc[:cfg["extraction"]["smoke_n"]]
    _, columns, _ = inspect_models(cfg)
    expected = pd.read_csv(path(cfg["features_csv"])).iloc[:len(test)]
    speaker = Speaker(id="equivalence-test")
    utterances = [Utterance(id=row.sample_id, conversation_id=row.sample_id, speaker=speaker, text=row.text)
                  for row in test.itertuples()]
    with tempfile.TemporaryDirectory(dir=path("provenance")) as temp:
        # Only the config-file location is redirected; computation stays on the default memory backend.
        config_file = path(temp) / "convokit.yml"
        config_file.write_text("default_backend: mem\n", encoding="utf-8")
        with patch("convokit.model.corpus.ConvoKitConfig", lambda: ConvoKitConfig(filename=str(config_file))):
            corpus = Corpus(utterances=utterances, backend="mem")
        corpus = TextParser(verbosity=1000).transform(corpus)
        corpus = PolitenessStrategies().transform(corpus, markers=False)
        for row in expected.itertuples(index=False, name=None):
            sample_id = row[0]
            observed = corpus.get_utterance(sample_id).meta["politeness_strategies"]
            np.testing.assert_array_equal([observed[column] for column in columns], row[1:])
    result = {"status": "PASS", "n": len(test), "n_strategies": len(columns),
              "corpus_transform_equals_rowwise_features": True,
              "constructor_settings": "TextParser(verbosity=1000); PolitenessStrategies().transform(corpus, markers=False)"}
    dump("provenance/corpus_api_equivalence.json", result)
    print(result)


if __name__ == "__main__":
    main()
