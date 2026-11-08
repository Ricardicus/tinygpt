import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import main as training
from tinygpt import TinyGPT, QKRMSNorm


class QKNormTests(unittest.TestCase):
    def test_compiled_norm_matches_forward_backward_and_state(self):
        norm = QKRMSNorm(8)
        x = torch.randn(2, 3, 8, requires_grad=True)
        upstream = torch.randn_like(x)
        reference = norm(x)
        gradients = torch.autograd.grad(reference, (x, norm.weight), upstream)
        state = {k: v.clone() for k, v in norm.state_dict().items()}
        # Exercise compiled autograd without requiring a CUDA toolchain in tests.
        norm.compile(backend='aot_eager', fullgraph=True)
        actual = norm(x)
        actual_gradients = torch.autograd.grad(actual, (x, norm.weight), upstream)
        torch.testing.assert_close(actual, reference)
        for a, b in zip(actual_gradients, gradients):
            torch.testing.assert_close(a, b)
        self.assertEqual(set(state), set(norm.state_dict()))
        for key, value in state.items():
            torch.testing.assert_close(norm.state_dict()[key], value)

    def test_compile_requires_qk_and_visits_both_norms(self):
        with self.assertRaises(ValueError):
            TinyGPT(20, 4, 12, 3, 1).compile_qk_norms()
        model = TinyGPT(20, 4, 12, 3, 2, qk_norm=True)
        keys = set(model.state_dict())
        with patch.object(QKRMSNorm, 'compile') as compile_norm:
            model.compile_qk_norms()
            self.assertEqual(compile_norm.call_count, 4)
        self.assertEqual(set(model.state_dict()), keys)

    def test_normalizes_each_head_and_preserves_dtype(self):
        norm = QKRMSNorm(4)
        x = torch.randn(2, 3, 5, 4) * 100
        y = norm(x)
        torch.testing.assert_close(y.square().mean(-1), torch.ones(2, 3, 5))
        self.assertEqual(norm(x.bfloat16()).dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(norm(torch.zeros_like(x))).all())
        y.sum().backward()
        self.assertTrue(torch.isfinite(norm.weight.grad).all())

    def test_model_backward_and_causality(self):
        model = TinyGPT(20, 4, 12, 3, 2, qk_norm=True)
        x = torch.tensor([[1, 2, 3, 4]])
        y = model(x)
        altered = model(torch.tensor([[1, 2, 8, 9]]))
        torch.testing.assert_close(y[:, :2], altered[:, :2])
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(model.layers[0].attn.q_norm.weight.grad).all())

    def test_legacy_and_qk_checkpoint_resume(self):
        for enabled in (False, True):
            with self.subTest(qk_norm=enabled), tempfile.TemporaryDirectory() as directory:
                model = TinyGPT(20, 4, 12, 3, 1, qk_norm=enabled)
                optimizer = torch.optim.AdamW(model.parameters())
                x = torch.tensor([[1, 2, 3, 4]])
                model(x).square().mean().backward()
                optimizer.step()
                path = Path(directory) / 'model.pt'
                training.save_checkpoint(path, model, optimizer, 0, 1.0, 1)
                saved = torch.load(path, weights_only=True)
                if not enabled:
                    # Simulate an actual old checkpoint without the new config key.
                    del saved['config']['qk_norm']
                    torch.save(saved, path)
                # Deliberately opposite CLI value: checkpoint architecture must win.
                flags = ['--no-qk-norm'] if enabled else ['--qk-norm']
                with patch.object(training, 'getData', return_value=[]), \
                        patch.object(training, 'getBPE', return_value=SimpleNamespace(vocab_size=20)), \
                        patch.object(training, 'TokenizedDataReader', return_value=[]), \
                        patch.object(training, 'stream_batches', return_value=iter([])), \
                        contextlib.redirect_stdout(io.StringIO()):
                    training.main(['--device', 'cpu', '--model', str(path), '--epochs', '1', '--verbose', 'false'] + flags)
                restored = torch.load(path, weights_only=True)
                self.assertEqual(restored['config']['qk_norm'], enabled)
                loaded = TinyGPT(**restored['config'])
                loaded.load_state_dict(restored['model_state_dict'], strict=True)
                torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
                for key, state in saved['optimizer_state_dict']['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(restored['optimizer_state_dict']['state'][key][name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
