from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from bpe import BPE
from main import generate, save_checkpoint
from tinygpt import TinyGPT


class InferenceTests(unittest.TestCase):
    def test_script_loads_checkpoint_and_generates_requested_tokens(self):
        script = Path(__file__).resolve().parents[1] / 'other/inference.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bpe = BPE(vocab_size=32, lower_case=False)
            bpe.add_corpus('Hello world Hello world')
            bpe.train()
            bpe.vocab_size = len(bpe.vocab)
            bpe.save(str(root / 'tokenizer.json'))
            model = TinyGPT(bpe.vocab_size, 4, 12, 3, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(root / 'model.pt', model, optimizer, 1, 1.0, 1)
            torch.manual_seed(42)
            expected = generate(model, bpe, 'Hello', 7, 'cpu')
            command = [sys.executable, str(script), '--model', str(root / 'model.pt'),
                       '--prompt', 'Hello', '--max-new-tokens', '7', '--device', 'cpu', '--seed', '42']
            result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, expected + '\n')
            result = subprocess.run(command + ['--prompt', '☃'], cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unknown symbol', result.stderr)


if __name__ == '__main__':
    unittest.main()
