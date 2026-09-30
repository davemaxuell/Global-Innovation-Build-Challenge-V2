import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np

from scglm.data import SourceStream
from scglm.prepare_data import ExclusionIndex, choose_split, fingerprint, tokenizer_train, tokenize_split


class DataTests(unittest.TestCase):
    def test_stream_shift_shards_wrap_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a.bin", Path(directory) / "b.bin"]
            np.arange(5, dtype="<u2").tofile(paths[0])
            np.arange(5, 11, dtype="<u2").tofile(paths[1])
            stream = SourceStream(paths, seq_len=4)
            x, y = stream.next_batch(2)
            np.testing.assert_array_equal(x, np.arange(8).reshape(2, 4))
            np.testing.assert_array_equal(y, np.arange(1, 9).reshape(2, 4))
            state = stream.state_dict()
            expected = stream.next_batch(2)
            self.assertEqual(stream.cycles, 1)
            restored = SourceStream(paths, seq_len=4)
            restored.load_state_dict(state)
            actual = restored.next_batch(2)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(restored.state_dict(), stream.state_dict())

    def test_contamination_and_partition_normalization(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
        index = ExclusionIndex()
        index.add(text)
        self.assertTrue(index.matches("Unrelated lead. " + text.upper() + " End."))
        self.assertFalse(index.matches("one two three four five six seven eight nine ten eleven twelve thirteen fourteen"))
        index.titles.add("held out article")
        self.assertTrue(index.matches("different contents", "Held_out_article"))
        self.assertEqual(fingerprint("HELLO  world\n"), fingerprint("hello world"))
        self.assertEqual(choose_split(fingerprint("HELLO world")), choose_split(fingerprint("hello world")))

    def test_tokenizer_and_document_export_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "raw").mkdir()
            train_text = ("A careful language model predicts words from earlier context. "
                          "Scientists compare experiments and record observations. " * 30)
            config = {"sources": {"web": {"revision": "fixture", "target_train_tokens": 256}},
                      "tokenizer_training_chars_per_source": 1000, "vocab_size": 300,
                      "document_shuffle_buffer": 2, "seed": 42, "development_max_tokens": 64}
            with (output / "raw" / "web.train.jsonl").open("w") as handle:
                for index in range(3):
                    handle.write(json.dumps({"id": str(index), "source": "web", "text": train_text}) + "\n")
            with (output / "raw" / "web.final.jsonl").open("w") as handle:
                handle.write(json.dumps({"id": "heldout", "source": "web", "text": "Unseen evaluation document. " * 60}) + "\n")
            tokenizer = tokenizer_train(config, output)
            self.assertEqual(tokenizer.get_vocab_size(), 300)
            self.assertEqual(tokenizer.token_to_id("<|eos|>"), 2)
            train = tokenize_split(config, output, tokenizer, "web", "train")
            self.assertEqual(train["tokens"], 256)
            self.assertEqual(Path(train["path"]).stat().st_size, 512)
            evaluation = tokenize_split(config, output, tokenizer, "web", "final")
            record = json.loads(Path(evaluation["jsonl_path"]).read_text())
            self.assertEqual(record["tokens"][-1], 2)
            self.assertLessEqual(len(record["tokens"]), 64)
            self.assertEqual(record["id"], "heldout")
            self.assertFalse((output / "web" / "train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
