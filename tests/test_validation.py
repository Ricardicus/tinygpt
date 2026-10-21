import contextlib
import csv
import io
import math
from pathlib import Path
import random
import tempfile
import unittest

import torch

from bpe import BPE
from datareader import TokenizedDataReader
import main as training
from tinygpt import TinyGPT


class ValidationTests(unittest.TestCase):
    def test_file_split_is_repeatable_disjoint_and_does_not_change_rng(self):
        root = Path('/tmp/validation-corpus')
        files = [root / f'{i}.txt' for i in range(20)]
        state = random.getstate()
        train, valid = training.split_validation_files(files, root, .2, 42)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(valid), 4)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(set(train + valid), set(files))
        again = training.split_validation_files(list(reversed(files)), root, .2, 42)
        self.assertEqual(set(valid), set(again[1]))
        with self.assertRaisesRegex(ValueError, 'at least two'):
            training.split_validation_files(files[:1], root, .2, 42)

    def test_evaluation_matches_token_weighted_loss_and_preserves_training_state(self):
        torch.manual_seed(7)
        model = TinyGPT(12, 3, 12, 3, 1)
        model.train()
        windows = [[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 1]]
        before = {name: p.clone() for name, p in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        loss, perplexity, tokens = training.evaluate(model, windows, 2, 10, 'cpu')
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(tokens, 9)
        self.assertAlmostEqual(perplexity, math.exp(loss))
        self.assertTrue(all(torch.equal(before[k], p) for k, p in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            batch = torch.tensor(windows)
            logits = model(batch[:, :-1])
            expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), batch[:, 1:].reshape(-1)).item()
        self.assertAlmostEqual(loss, expected, places=5)
        second = training.evaluate(model, windows, 2, 10, 'cpu')
        self.assertFalse(model.training)
        self.assertEqual((loss, perplexity, tokens), second)
        with self.assertRaisesRegex(ValueError, 'no complete'):
            training.evaluate(model, [], 2, 1, 'cpu')

    def test_training_logs_validation_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'corpus'
            corpus.mkdir()
            for i in range(4):
                (corpus / f'{i}.txt').write_text('hello world ' * 3)
            bpe = BPE(vocab_size=30)
            bpe.add_corpus('hello world ' * 3)
            bpe.train()
            tokenizer = root / 'tokenizer.json'
            bpe.save(str(tokenizer))
            model = root / 'model.pt'
            log = root / 'progress.csv'
            args = ['--rawdata', str(corpus), '--tokenizer', str(tokenizer),
                    '--model', str(model), '--log-progress', str(log),
                    '--device', 'cpu', '--d-model', '12', '--n-heads', '3',
                    '--num-layers', '1', '--context-length', '4', '--batch-size', '2',
                    '--validation-fraction', '.25', '--validation-every-steps', '2',
                    '--validation-batches', '2', '--verbose', 'false']
            with contextlib.redirect_stdout(io.StringIO()):
                training.main(args + ['--epochs', '1'])
            checkpoint = torch.load(model, map_location='cpu')
            with open(str(log) + '.validation.csv') as source:
                rows = list(csv.DictReader(source))
            steps = [int(row['global_step']) for row in rows]
            self.assertEqual(steps[0], 0)
            self.assertIn(2, steps)
            self.assertEqual(steps[-1], checkpoint['global_step'])
            self.assertEqual(len(steps), len(set(steps)))
            self.assertTrue(all(math.isfinite(float(row['validation_loss'])) for row in rows))
            with log.open() as source:
                progress = list(csv.DictReader(source))
            self.assertEqual(len(progress), checkpoint['global_step'])
            for step, row in enumerate(progress, 1):
                latest = [r for r in rows if int(r['global_step']) <= step][-1]
                self.assertEqual(float(row['validation_loss']), float(latest['validation_loss']))
            self.assertTrue((root / 'corpus_train_tokens.bin').exists())
            self.assertTrue((root / 'corpus_validation_tokens.bin').exists())
            train_files, valid_files = training.split_validation_files(sorted(corpus.glob('*.txt')), corpus, .25, 42)
            for files in (train_files, valid_files):
                reader = TokenizedDataReader(corpus, bpe, 4, files=files)
                self.assertEqual(set(reader.files), set(files))
                self.assertEqual(len(reader._tokens), len(bpe.encode('hello world ' * 3)) * len(files))
            with contextlib.redirect_stdout(io.StringIO()):
                training.main(args + ['--epochs', '2'])
            self.assertEqual(torch.load(model, map_location='cpu')['epoch'], 2)
            with log.open() as source:
                resumed = list(csv.DictReader(source))
            self.assertEqual(resumed[:len(progress)], progress)
            self.assertTrue(all(math.isfinite(float(row['validation_loss'])) for row in resumed))

    def test_legacy_progress_log_is_upgraded_without_inventing_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.csv'
            path.write_text('epoch,step,loss,mean_loss\n3,14575,2.049,2.176\n')
            training.prepare_progress_log(path)
            with path.open() as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, training.PROGRESS_COLUMNS)
                self.assertEqual(list(reader), [dict(epoch='3', step='14575', loss='2.049',
                                                    mean_loss='2.176', validation_loss='')])
            before = path.read_bytes()
            training.prepare_progress_log(path)
            self.assertEqual(path.read_bytes(), before)
            # A malformed old log is not replaced with a partially upgraded one.
            path.write_text('epoch,step,loss,mean_loss\n1,1,2\n')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Malformed'):
                training.prepare_progress_log(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_invalid_options_are_rejected(self):
        for options in (['--validation-fraction', '1'], ['--validation-fraction', '-.1'],
                        ['--validation-batches', '0'], ['--validation-every-steps', '-1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(options)


if __name__ == '__main__':
    unittest.main()
