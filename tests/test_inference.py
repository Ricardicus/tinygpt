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
from tinygpt import TinyGPT, generate_token_ids


class InferenceTests(unittest.TestCase):
    def test_no_kv_cache_flag_recomputes_each_window(self):
        self.assertTrue(parse_args([]).use_kv_cache)
        options = parse_args(['--no-kv-cache'])
        self.assertFalse(options.use_kv_cache)
        model = TinyGPT(20, 4, 12, 3, 1).eval()
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.decode.side_effect = list
        expected = generate(model, tokenizer, 'prompt', 5, temperature=0)
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            actual = generate(model, tokenizer, 'prompt', 5, temperature=0, use_kv_cache=options.use_kv_cache)
        self.assertEqual(actual, expected)
        self.assertEqual([call.args[0].size(1) for call in forward.call_args_list], [2, 3, 4, 4, 4])
        self.assertTrue(all(call.kwargs['kv_cache'] is None for call in forward.call_args_list))

    @torch.no_grad()
    def test_cached_decoding_matches_full_window_logits_and_rebuilds_on_overflow(self):
        for options in ({}, {'qk_norm': True, 'moe': True}):
            torch.manual_seed(42)
            model = TinyGPT(20, 5, 12, 3, 2, **options).eval()
            for prompt in ([2, 3], [1, 2, 3, 4, 5, 6]):
                with self.subTest(options=options, prompt=prompt):
                    history = list(prompt)
                    observed = []
                    def capture(module, inputs, output):
                        observed.append((inputs[0].size(1), output[:, -1].clone()))

                    hook = model.register_forward_hook(capture)
                    try:
                        tokens = list(generate_token_ids(model, prompt, 6, 'cpu', 0, 0))
                    finally:
                        hook.remove()
                    lengths = [length for length, _ in observed]
                    self.assertEqual(lengths, [2, 1, 1, 1, 5, 5] if len(prompt) == 2 else [5] * 6)
                    for token, (_, logits) in zip(tokens, observed):
                        reference = model(torch.tensor([history[-5:]]))[:, -1]
                        torch.testing.assert_close(logits, reference)
                        self.assertEqual(token, reference.argmax().item())
                        history.append(token)

    def test_show_end_marker_only_when_generated(self):
        class FixedModel(torch.nn.Module):
            context_length = 4

            def forward(self, inputs):
                return torch.tensor([0., 1.]).expand(1, inputs.shape[1], 2)

        tokenizer = Mock()
        tokenizer.encode.return_value = [0]
        tokenizer.decode.side_effect = lambda ids: ''.join(['prompt', 'answer<end>extra'][token] for token in ids)
        model = FixedModel()
        self.assertEqual(generate(model, tokenizer, 'prompt', 2, temperature=0, response_end_marker='<end>'), 'promptanswer')
        self.assertEqual(generate(model, tokenizer, 'prompt', 2, temperature=0, response_end_marker='<end>',
                                  return_token_count=True), ('promptanswer', 1))
        self.assertEqual(generate(model, tokenizer, 'prompt', 2, temperature=0, response_end_marker='<end>', show_end_marker=True), 'promptanswer<end>')
        self.assertEqual(generate(model, tokenizer, 'prompt', 0, temperature=0, response_end_marker='<end>', show_end_marker=True), 'prompt')

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
            self.assertNotIn('tokens/s', result.stderr)
            timed = subprocess.run(command + ['--tokens-per-second'], cwd=root, text=True, capture_output=True, check=True)
            self.assertEqual(timed.stdout, result.stdout)
            self.assertRegex(timed.stderr, r'Generated 7 tokens in [\d.]+ s \([\d.]+ tokens/s\)')
            result = subprocess.run(command + ['--prompt', '☃'], cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unknown symbol', result.stderr)


if __name__ == '__main__':
    unittest.main()
