import tempfile
import unittest
from pathlib import Path

import torch

from bpe import BPE
from main import finetune_examples, main, RESPONSE_MARKER
from tinygpt import TinyGPT


class FineTuneTests(unittest.TestCase):
    def test_masks_prompt_and_covers_long_response(self):
        text = '### Instruction\nSay hello' + RESPONSE_MARKER + ('hello world! ' * 20) + '\n\n'
        tok = BPE(vocab_size=80, lower_case=False)
        tok.add_corpus(text)
        tok.train()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.txt'
            path.write_text(text)
            windows = list(finetune_examples(path, tok, 16))
            actual = [t for _, y in windows for t in y if t != -100]
            self.assertEqual(actual, tok.encode(text.split(RESPONSE_MARKER, 1)[1]))
            self.assertTrue(any(-100 in y for _, y in windows))
            self.assertTrue(all(len(x) <= 16 and len(x) == len(y) for x, y in windows))

    def test_training_produces_separate_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'corpus'
            data.mkdir()
            text = '### Instruction\nSay hello' + RESPONSE_MARKER + 'hello world\n\n'
            for i in range(3):
                (data / f'{i}.txt').write_text(text)
            tok = BPE(vocab_size=64, lower_case=False)
            tok.add_corpus(text)
            tok.train()
            tok.save(str(root / 'tokenizer.json'))
            model = TinyGPT(vocab_size=64, context_length=16, d_model=8, n_heads=2, num_layers=1)
            base = root / 'base.pt'
            cfg = dict(vocab_size=64, context_length=16, d_model=8, n_heads=2, num_layers=1)
            torch.save({'config': cfg, 'model_state_dict': model.state_dict()}, base)
            main(['finetune', '--data', str(data), '--base-model', str(base), '--tokenizer', str(root / 'tokenizer.json'),
                  '--output', str(root / 'sft.pt'), '--epochs', '1', '--device', 'cpu'])
            output = torch.load(root / 'sft.pt', weights_only=True)
            self.assertEqual(output['epoch'], 1)
            self.assertTrue(base.exists())
            self.assertFalse(torch.equal(output['model_state_dict']['E'], model.state_dict()['E']))


if __name__ == '__main__':
    unittest.main()
