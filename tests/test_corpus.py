import json
import io
from contextlib import redirect_stdout
import random
import tempfile
import unittest
from pathlib import Path

from bpe import BPE
from corpus import discover_files, load_config, text_chunks
from datareader import TextCorpusReader, TokenizedDataReader


class CorpusTests(unittest.TestCase):
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
