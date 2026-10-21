from pathlib import Path
from contextlib import redirect_stdout
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from bpe import BPE
from main import generate, main, parse_args, save_checkpoint
from tinygpt import TinyGPT


class InferenceTests(unittest.TestCase):
    def test_sampling_filters_unlikely_tokens_and_supports_greedy_generation(self):
        class FixedModel(torch.nn.Module):
            context_length = 2

            def forward(self, x):
                return torch.tensor([0., 1., 2., 3.]).expand(1, x.shape[1], 4)

        bpe = Mock()
        bpe.encode.return_value = [0]
        bpe.decode.side_effect = lambda ids: ids
        with patch('main.torch.multinomial', return_value=torch.tensor([[3]])) as sampler:
            ids = generate(FixedModel(), bpe, 'prompt', 3, temperature=.5, top_k=2)
        self.assertEqual(ids, [0, 3, 3, 3])
        self.assertEqual(sampler.call_count, 3)
        for call in sampler.call_args_list:
            probabilities = call.args[0]
            self.assertTrue(torch.equal(probabilities[:, :2], torch.zeros(1, 2)))
            self.assertTrue(torch.allclose(probabilities[:, 2:], torch.softmax(torch.tensor([[4., 6.]]), dim=-1)))
        with patch('main.torch.multinomial') as sampler:
            ids = generate(FixedModel(), bpe, 'prompt', 3, temperature=0)
            sampler.assert_not_called()
        self.assertEqual(ids, [0, 3, 3, 3])
        # K larger than the vocabulary should still work.
        self.assertEqual(len(generate(FixedModel(), bpe, 'prompt', 1, top_k=100)), 2)

    def test_invalid_sampling_settings_are_rejected(self):
        for options in (['--temperature', '-1'], ['--temperature', 'nan'],
                        ['--temperature', 'inf'], ['--top-k', '-1']):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parse_args(options)

    def test_main_honors_new_token_count_for_saved_and_legacy_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bpe = BPE(vocab_size=32, lower_case=False)
            bpe.add_corpus('Hello world Hello world')
            bpe.train()
            bpe.vocab_size = len(bpe.vocab)
            bpe.save(str(root / 'tokenizer.json'))
            model = TinyGPT(bpe.vocab_size, 4, 12, 3, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(root / 'saved.pt', model, optimizer, 1, 1.0, 1)
            torch.save(model.state_dict(), root / 'legacy.pt')
            for checkpoint in ('saved.pt', 'legacy.pt'):
                for count in (0, 1, 7):
                    with self.subTest(checkpoint=checkpoint, max_new_tokens=count):
                        # Deterministic sampling independent of random numbers
                        # consumed while main constructs the checkpoint's model.
                        sample = lambda probs, num_samples: probs.argmax(dim=-1, keepdim=True)
                        with patch('main.torch.multinomial', side_effect=sample):
                            expected = generate(model, bpe, 'Hello', count, 'cpu')
                        output = io.StringIO()
                        with redirect_stdout(output), patch('main.torch.multinomial', side_effect=sample) as sampler:
                            main(['--model', str(root / checkpoint),
                                  '--tokenizer', str(root / 'tokenizer.json'),
                                  '--prompt', 'Hello',
                                  '--max-new-tokens', str(count), '--device', 'cpu',
                                  '--verbose', 'false', '--context-length', '4',
                                  '--d-model', '12', '--n-heads', '3', '--num-layers', '1'])
                        self.assertEqual(sampler.call_count, count)
                        actual = output.getvalue().split('--- Generated Text ---\n', 1)[1]
                        self.assertEqual(actual, expected + '\n')

    def test_main_rejects_negative_new_token_count(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--max-new-tokens', '-1'])

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
