import csv
from contextlib import redirect_stdout
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from bpe import BPE
from main import finetune_examples, main, RESPONSE_MARKER, save_checkpoint
from tinygpt import TinyGPT


class FineTuneTests(unittest.TestCase):
    def test_old_and_new_context_order_use_same_prompt(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(text.encode())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.txt'
            for text in ('### Instruction\nQuestion\n\n### Context\nBackground',
                         '### Context\nBackground\n\n### Instruction\nQuestion'):
                path.write_text(text + RESPONSE_MARKER + 'Answer')
                tokenizer.reset_mock()
                list(finetune_examples(path, tokenizer, 128, '<end>'))
                self.assertEqual(tokenizer.encode.call_args_list[0].args[0],
                                 '### Context\nBackground\n\n### Instruction\nQuestion' + RESPONSE_MARKER)

    def test_masks_prompt_and_covers_long_response(self):
        text = '### Instruction\nSay hello' + RESPONSE_MARKER + ('hello world! ' * 20) + '\n\n'
        tok = BPE(vocab_size=80, lower_case=False)
        tok.add_corpus(text)
        tok.train()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.txt'
            path.write_text(text)
            windows = list(finetune_examples(path, tok, 16, 'hello'))
            actual = [t for _, y in windows for t in y if t != -100]
            self.assertEqual(actual, tok.encode(text.split(RESPONSE_MARKER, 1)[1]) + tok.encode('hello'))
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
            progress_path = root / 'sft.pt.progress.csv'
            progress_path.write_text('epoch,step,loss,mean_loss,validation_loss,lr,grad_norm\n1,99,1,1,,0.01,1\n')
            validation_path = root / 'sft.pt.progress.csv.validation.csv'
            validation_path.write_text('epoch,global_step,validation_loss,perplexity,tokens\n1,99,0,1,5')
            console = io.StringIO()
            with redirect_stdout(console), patch.object(
                TinyGPT, 'compile', autospec=True,
                side_effect=lambda model: torch.nn.Module.compile(model, backend='aot_eager'),
            ) as compile_model, patch('main.save_checkpoint', wraps=save_checkpoint) as save:
                main(['finetune', '--data', str(data), '--base-model', str(base), '--tokenizer', str(root / 'tokenizer.json'),
                      '--output', str(root / 'sft.pt'), '--epochs', '2', '--device', 'cpu',
                      '--response-end-marker', 'hello', '--compile-model',
                      '--batch-size', '1', '--validation-every-steps', '1', '--save-every-steps', '1'])
            compile_model.assert_called_once()
            summary = console.getvalue().split('Counting training windows', 1)[0]
            self.assertIn(f'Base checkpoint: {base}', summary)
            self.assertIn('Context length: 16', summary)
            self.assertIn('QK normalization: False', summary)
            self.assertIn(f'Total trainable parameters: {sum(parameter.numel() for parameter in model.parameters()):,}', summary)
            output = torch.load(root / 'sft.pt', weights_only=True)
            self.assertEqual(output['epoch'], 2)
            self.assertEqual(output['response_end_marker'], 'hello')
            self.assertTrue(base.exists())
            self.assertFalse(torch.equal(output['model_state_dict']['E'], model.state_dict()['E']))
            model.load_state_dict(output['model_state_dict'])
            with progress_path.open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]['step'], '99')
            self.assertEqual(rows[0]['tokens_per_second'], '')
            rows = rows[1:]
            self.assertTrue(rows)
            self.assertGreater(float(rows[-1]['grad_norm']), 0)
            self.assertTrue(all(float(row['tokens_per_second']) > 0 for row in rows))
            self.assertIn('tokens/s', console.getvalue())
            with validation_path.open(newline='') as source:
                validation = list(csv.DictReader(source))
            self.assertEqual(validation[0]['global_step'], '99')
            self.assertGreater(len(rows), 2)
            self.assertEqual([call.args[5] for call in save.call_args_list], [int(row['step']) for row in rows])
            self.assertEqual([row['global_step'] for row in validation[1:]], [row['step'] for row in rows])
            for epoch in (1, 2):
                last_step = max(int(row['step']) for row in rows if int(row['epoch']) == epoch)
                self.assertIn(f"Epoch {epoch}/2 | Step {last_step:6d} (100.00%)", console.getvalue())
            self.assertEqual(output['loss'], float(validation[-1]['validation_loss']))
            self.assertAlmostEqual(float(validation[-1]['perplexity']),
                                   math.exp(float(validation[-1]['validation_loss'])))


if __name__ == '__main__':
    unittest.main()
