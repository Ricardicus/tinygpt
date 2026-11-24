from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bpe import BPE
from corpus import discover_files, load_config, split_validation_files
from datareader import TokenizedDataReader
from other.count_tokens import main


def report(*args):
    output = io.StringIO()
    with redirect_stdout(output):
        main([str(arg) for arg in args])
    return output.getvalue()


class CountTokensTests(unittest.TestCase):
    def test_saved_and_fresh_counts_match_training_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            tokenizer = BPE(vocab_size=32, lower_case=False)
            tokenizer.add_corpus("Hello world HELLO")
            tokenizer.train()
            tok_path = root / "tokenizer.json"
            tokenizer.save(tok_path)
            config = root / "training_data.json"
            config.write_text(json.dumps({"suffixes": [".txt"], "exclude_files": ["skip.txt"]}))
            (corpus / "skip.txt").write_text("excluded characters")
            for i in range(4):
                (corpus / f"{i}.txt").write_text("HELLO world\n" * (i + 1))
            paths = discover_files(corpus, load_config(config))
            train, _ = split_validation_files(paths, corpus, 0.25, 17)
            cache = root / "corpus_train_tokens.bin"
            reader = TokenizedDataReader(corpus, tokenizer, 4, lower_case=False, files=train,
                                         cache=True, cache_path=cache)
            count = len(reader._tokens)
            with patch.object(BPE, "encode", side_effect=AssertionError("cache scan")):
                result = report(root, "--context-length", 4)
                self.assertIn(f"Total tokens: {count:,}\n", result)
                self.assertIn(f"Target tokens per epoch: {len(reader) * 4:,}\n", result)
            cache.unlink()
            result = report(corpus, "--tokenizer", tok_path, "--lower-case", "false",
                            "--validation-fraction", 0.25, "--validation-seed", 17)
            self.assertIn(f"Total tokens: {count:,}\n", result)
            self.assertFalse(cache.exists())

    def test_ambiguous_or_damaged_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            corpus.mkdir()
            full = corpus.with_name("corpus_tokens.bin")
            full.write_bytes(bytes(28))
            with redirect_stderr(io.StringIO()):
                self.assertIn("Total tokens: 7\n", report(corpus))
                corpus.with_name("corpus_train_tokens.bin").touch()
                with self.assertRaises(SystemExit):
                    report(corpus)
                self.assertIn("Total tokens: 7\n", report(full))
                Path(str(full) + ".json").write_text('{"bytes": 32}')
                with self.assertRaises(SystemExit):
                    report(full)
                full.write_bytes(bytes(31))
                with self.assertRaises(SystemExit):
                    report(full)
