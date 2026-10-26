import json
import io
import itertools
from array import array
from contextlib import redirect_stdout
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bpe import BPE
from corpus import discover_files, load_config, text_chunks
from datareader import TextCorpusReader, TokenizedDataReader


class CorpusTests(unittest.TestCase):
    def test_external_exclusions_are_config_relative_and_exact_corpus_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            for group in ("books", "wiki"):
                (corpus / group).mkdir(parents=True)
                (corpus / group / "same.txt").write_text("hello world " * 10)
            rules = root / "reports/rules"
            rules.mkdir(parents=True)
            (rules / "excluded.txt").write_text("# Reviewed exclusions\nbooks/same.txt\n")
            config_path = rules / "training_data.json"
            config_path.write_text(json.dumps({"suffixes": [".txt"], "exclude_dirs": [],
                                                "exclude_files_from": ["excluded.txt"]}))
            config = load_config(config_path)
            files = discover_files(corpus, config)
            self.assertEqual([p.relative_to(corpus).as_posix() for p in files], ["wiki/same.txt"])
            bpe = BPE(vocab_size=32)
            bpe.add_corpus("hello world")
            bpe.train()
            # Preselected validation/training lists cannot bypass file exclusions.
            reader = TokenizedDataReader(corpus, bpe, 2, files=sorted(corpus.rglob("*.txt")), corpus_config=config)
            self.assertEqual(reader.files, files)

    def test_invalid_exclusion_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for exclusions in (["../outside.txt"], ["/absolute.txt"], ["books\\bad.txt"], [None], "not a list"):
                path.write_text(json.dumps({"suffixes": [".txt"], "exclude_files": exclusions}))
                with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                    load_config(path)
            path.write_text(json.dumps({"suffixes": [".txt"], "exclude_files_from": ["missing.txt"]}))
            with self.assertRaises(FileNotFoundError):
                load_config(path)

    def test_changed_exclusions_invalidate_token_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            root.mkdir()
            (root / "a.txt").write_text("hello world " * 10)
            (root / "b.txt").write_text("hello world " * 10)
            bpe = BPE(vocab_size=32)
            bpe.add_corpus("hello world")
            bpe.train()
            first = TokenizedDataReader(root, bpe, 2, cache=True)
            config = {"suffixes": [".txt"], "exclude_dirs": [], "exclude_files": ["b.txt"]}
            second = TokenizedDataReader(root, bpe, 2, cache=True, corpus_config=config)
            self.assertNotEqual(first._fingerprint, second._fingerprint)
            self.assertLess(len(second._tokens), len(first._tokens))

    def test_large_cache_roundtrip_reports_read_and_write_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = TokenizedDataReader.__new__(TokenizedDataReader)
            reader.verbose = True
            reader._cache_path = Path(directory) / "tokens.bin"
            reader._meta_path = Path(directory) / "tokens.bin.json"
            reader._fingerprint = "test-fingerprint"
            expected = array("I", [1, 256, 65536]) * (1_400_000)
            reader._tokens = expected
            with patch("datareader.time.monotonic", side_effect=itertools.count(step=3).__next__), \
                    redirect_stdout(io.StringIO()) as output:
                reader._save_cache()
                reader._tokens = array("I")
                reader._load_cache()
            self.assertEqual(reader._tokens, expected)
            self.assertEqual(reader._cache_path.stat().st_size, len(expected) * 4)
            self.assertIn("Cache saving:", output.getvalue())
            self.assertIn("Cache loading:", output.getvalue())

    def test_initialization_discovers_files_without_reading_corpus_contents(self):
        from main import getData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "book.txt").write_text("whole book\n" * 100)
            with patch("datareader.text_chunks", side_effect=AssertionError("Unnecessary corpus read")), \
                    redirect_stdout(io.StringIO()) as output:
                reader = getData(rawdata_folder=root)
            self.assertEqual(len(reader.files), 1)
            self.assertIsNone(reader._length)
            self.assertIn("DataReader ready: 1 corpus files", output.getvalue())

    def test_chunk_count_reports_progress_and_caches_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "book.txt").write_text("whole book\n" * 1000)
            reader = TextCorpusReader(root, shuffle=False, verbose=True)
            expected = sum(1 for _ in reader)
            with patch("datareader.time.monotonic", side_effect=itertools.count(step=3).__next__), \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(len(reader), expected)
            self.assertIn("Counting input: file 1 / 1", output.getvalue())
            self.assertIn("Input count complete:", output.getvalue())
            with patch("datareader.text_chunks", side_effect=AssertionError("Repeated count")):
                self.assertEqual(len(reader), expected)

    def test_recursive_selection_and_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'src').mkdir()
            (root / '.git').mkdir()
            (root / 'src' / 'main.CPP').write_text('hello world\n' * 200)
            (root / '.git' / 'hidden.txt').write_text('secret')
            (root / 'ignore.bin').write_text('ignored')
            (root / 'link.txt').symlink_to(root / 'src' / 'main.CPP')
            files = discover_files(root)
            self.assertEqual([p.name for p in files], ['main.CPP'])
            chunks = list(text_chunks(files[0], chunk_size=7))
            self.assertEqual(' '.join(chunks).split(), ['hello', 'world'] * 200)
            reader = TextCorpusReader(root, shuffle=False)
            self.assertEqual(len(reader), len(list(reader)))
            self.assertEqual(' '.join(reader).split(), ['hello', 'world'] * 200)

    def test_long_words_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.txt'
            path.write_text('before ' + 'x' * 10000 + ' after')
            self.assertEqual(' '.join(text_chunks(path, 17, 100)).split(), ['before', 'after'])
            path.write_bytes(b'hello\x00world')
            with self.assertRaises(ValueError):
                list(text_chunks(path))

    def test_bpe_limits_and_roundtrip(self):
        bpe = BPE(vocab_size=30, max_unique_words=2)
        bpe.add_corpus('apple banana cherry apple')
        self.assertEqual(len(bpe.word_freq), 2)
        self.assertEqual(bpe.skipped_words, 1)
        bpe.train()
        self.assertFalse(bpe.word_freq)
        self.assertTrue(bpe.encode('cherry'))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'bpe.json')
            bpe.lower_case = False
            bpe.save(path)
            restored = BPE()
            restored.load(path)
            self.assertFalse(restored.lower_case)
            self.assertEqual(restored.encode('apple'), bpe.encode('apple'))

    def test_bpe_rejects_alphabet_that_leaves_no_room_for_merges(self):
        bpe = BPE(vocab_size=16)
        bpe.add_corpus("abcdefghijklmnopqrstuvwxyz")
        with self.assertRaisesRegex(ValueError, "increase vocab_size"):
            bpe.train()
        self.assertTrue(bpe.word_freq)
        self.assertFalse(bpe.vocab)

    def test_new_tokenizer_expands_for_unicode_training_and_validation(self):
        from main import getBPE
        text = "hello world " + "".join(chr(0x4e00 + index) for index in range(100))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held_out = root / "validation.txt"
            held_out.write_text("Held-out characters: café Ω", encoding="utf-8")
            tokenizer = root / "tokenizer.json"
            with redirect_stdout(io.StringIO()) as output:
                bpe = getBPE([text], vocab_size=32, verbose=False,
                             tokenizer_path=tokenizer, lower_case=False,
                             bpe_part=1, character_sources=[held_out], bpe_alphabet_policy="expand")
            self.assertIn("Expanding vocabulary", output.getvalue())
            self.assertGreater(bpe.vocab_size, 32)
            self.assertTrue(bpe.merges)
            self.assertEqual(bpe.decode(bpe.encode(text)), text)
            self.assertEqual(bpe.decode(bpe.encode("café Ω")), "café Ω")
            self.assertLess(max(bpe.vocab.values()), bpe.vocab_size)
            saved = json.loads(tokenizer.read_text())
            self.assertEqual(saved["vocab_size"], bpe.vocab_size)
            with patch.object(BPE, "train", side_effect=AssertionError("Existing tokenizer must be reused")):
                restored = getBPE([], vocab_size=32, verbose=False, tokenizer_path=tokenizer)
            self.assertEqual(restored.vocab_size, bpe.vocab_size)
            self.assertEqual(restored.encode(text), bpe.encode(text))

    def test_default_policy_caps_unicode_alphabet_and_persists_unknown_fallback(self):
        from main import getBPE
        rare = "".join(chr(0x4e00 + index) for index in range(1000))
        sample = "hello world " + rare
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Path(directory) / "tokenizer.json"
            with redirect_stdout(io.StringIO()) as output:
                bpe = getBPE([sample, "Ω " * 1000], vocab_size=256, bpe_part=0.5,
                             tokenizer_path=tokenizer, lower_case=False, verbose=False)
            self.assertEqual(bpe.vocab_size, 256)
            self.assertLessEqual(len(bpe.vocab), 256)
            self.assertIn("Limiting BPE base alphabet", output.getvalue())
            self.assertIn("Ω", bpe.vocab)  # Counts include text outside the word sample.
            self.assertEqual(bpe.decode(bpe.encode("hello Ω")), "hello Ω")
            self.assertEqual(bpe.decode(bpe.encode("\U0001f984")), "<unk>")
            self.assertIn("<unk>", bpe.decode(bpe.encode(sample)))
            restored = BPE(model_file=tokenizer)
            restored.load()
            self.assertEqual(restored.unknown_token, "<unk>")
            self.assertEqual(restored.encode(sample), bpe.encode(sample))
            self.assertEqual(restored.decode(restored.encode("hello\u2003world")), "hello<unk>world")

    def test_explicit_alphabet_cap_and_error_policy(self):
        from main import getBPE
        text = "hello world " + "".join(chr(0x4e00 + index) for index in range(1000))
        with redirect_stdout(io.StringIO()):
            limited = getBPE([text], vocab_size=2048, bpe_part=1,
                             bpe_max_chars=128, verbose=False)
        # The full alphabet would fit 2048, but an explicit cap still applies.
        self.assertEqual(limited.vocab_size, 2048)
        self.assertEqual(limited.unknown_token, "<unk>")
        with self.assertRaisesRegex(ValueError, "increase vocab_size"):
            getBPE([text], vocab_size=256, bpe_alphabet_policy="error", verbose=False)
        for cap in (1, 256):
            with self.assertRaises(ValueError):
                getBPE([], vocab_size=256, bpe_max_chars=cap, verbose=False)

    def test_cache_invalidates_for_corpus_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            root.mkdir()
            source = root / 'input.txt'
            source.write_text('hello world ' * 20)
            bpe = BPE(vocab_size=30)
            bpe.add_corpus('hello world')
            bpe.train()
            cache = Path(tmp) / 'tokens.bin'
            def read():
                return TokenizedDataReader(root, bpe, 2, shuffle=False,
                                           cache=True, cache_path=cache, verbose=True)
            with redirect_stdout(io.StringIO()) as output:
                first = read()
                cached = read()
            self.assertIn("windows of size 3", output.getvalue())
            self.assertIn("cache loaded", output.getvalue())
            self.assertEqual(list(first), list(cached))
            self.assertEqual(list(first), list(read()))
            self.assertEqual(first[-1], list(first)[-1])
            with self.assertRaises(IndexError):
                first[len(first)]
            source.write_text('hello world ' * 40)
            second = read()
            self.assertGreater(len(second), len(first))
            old = second._fingerprint
            bpe.vocab = {token: len(bpe.vocab) - 1 - index for token, index in bpe.vocab.items()}
            self.assertNotEqual(read()._fingerprint, old)

    def test_pretraining_shuffle_spans_files_and_covers_every_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'nested').mkdir()
            (root / 'a.txt').write_text('alpha ' * 100)
            (root / 'nested' / 'b.cpp').write_text('beta ' * 100)
            bpe = BPE(vocab_size=40)
            bpe.add_corpus('alpha beta')
            bpe.train()
            reader = TokenizedDataReader(root, bpe, 1, shuffle=False)
            ordered = list(reader)
            reader.shuffle = True
            state = random.getstate()
            try:
                random.seed(42)
                first = list(reader)
                second = list(reader)
            finally:
                random.setstate(state)
            self.assertCountEqual(first, ordered)
            self.assertCountEqual(second, ordered)
            self.assertNotEqual(first, ordered)
            self.assertNotEqual(first, second)
            # The first batch mixes windows from the two distant source files.
            self.assertGreater(len({tuple(w) for w in first[:16]}), 1)

    def test_whitespace_roundtrip_through_readers_and_cache(self):
        text = " \tdef Main():\r\n    return 1\r\n\r\n\t# café </w>\n  "
        bpe = BPE(vocab_size=90, lower_case=False)
        bpe.add_corpus(text)
        bpe.train()
        self.assertEqual(bpe.decode(bpe.encode(text)), text)
        self.assertEqual(bpe.decode(bpe.encode("\t\n  ")), "\t\n  ")
        self.assertEqual(bpe.decode(bpe.encode("")), "")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            root.mkdir()
            path = root / 'code.cpp'
            path.write_bytes(text.encode())
            for size in (1, 2, 7, 65536):
                chunks = list(text_chunks(path, chunk_size=size))
                self.assertEqual("".join(chunks), text)
                ids = [i for chunk in chunks for i in bpe.encode(chunk)]
                self.assertEqual(ids, bpe.encode(text))
            self.assertEqual("".join(TextCorpusReader(root, lower_case=False)), text)
            model_path = str(Path(tmp) / 'bpe.json')
            bpe.save(model_path)
            restored = BPE()
            restored.load(model_path)
            self.assertTrue(restored.preserve_whitespace)
            self.assertEqual(restored.decode(restored.encode(text)), text)
            for _ in range(2):
                reader = TokenizedDataReader(root, restored, 2, lower_case=False, cache=True)
                self.assertEqual(restored.decode(reader._tokens), text)

    def test_legacy_tokenizer_keeps_old_encoding(self):
        bpe = BPE()
        bpe.load(str(Path(__file__).resolve().parents[1] / 'tokenizer.json'))
        self.assertFalse(bpe.preserve_whitespace)
        self.assertEqual(bpe.encode('hello world'), bpe.encode('  hello\n\tworld  '))

    def test_fast_encoding_matches_original_ordered_merge_pass(self):
        rng = random.Random(17)
        words = [''.join(rng.choices('abcde', k=rng.randint(1, 30)))
                 for _ in range(300)] + ['aaaaaaa', 'abababab', 'abcabcabc']
        for preserve in (True, False):
            bpe = BPE(vocab_size=100, lower_case=False)
            bpe.preserve_whitespace = preserve
            if not preserve:
                bpe.alphabet.add(bpe.end_of_word_mark)
            bpe.add_corpus(' '.join(words))
            bpe.train()
            for word in words:
                symbols = list(word) + ([] if preserve else [bpe.end_of_word_mark])
                for a, b in bpe.merges:
                    i = 0
                    while i < len(symbols) - 1:
                        if symbols[i:i + 2] == [a, b]:
                            symbols[i:i + 2] = [a + b]
                        else:
                            i += 1
                expected = [bpe.vocab[symbol] for symbol in symbols]
                actual = bpe.encode_word(word)
                self.assertEqual(actual, expected)
                actual.clear()
                self.assertEqual(bpe.encode_word(word), expected)

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps({'suffixes': ['txt']}))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == '__main__':
    unittest.main()
