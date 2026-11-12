import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import main as training
from tinygpt import TinyGPT, MixtureOfExperts, MOE_DEFAULTS


class MoETests(unittest.TestCase):
    def test_compiled_surrounding_model_and_bf16(self):
        model = TinyGPT(20, 4, 8, 2, 1, moe=True)
        model.compile(backend='aot_eager')
        # Some CPU oneDNN builds lack BF16 backward; use portable CPU kernels.
        with torch.backends.mkldnn.flags(enabled=False):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                logits, aux = model(torch.tensor([[1, 2, 3, 4]]), return_aux=True)
                loss = logits.float().square().mean() + .01 * aux
            loss.backward()
        self.assertTrue(torch.isfinite(model.layers[0].mlp.router.weight.grad).all())

    def test_sparse_dispatch_matches_reference_and_router_gradients(self):
        for k in (1, 2):
            block = MixtureOfExperts(8, 3, k)
            x = torch.randn(2, 4, 8, requires_grad=True)
            actual, aux = block(x)
            flat = x.reshape(-1, 8)
            probs = block.router(flat).softmax(-1)
            weights, indices = probs.topk(k, dim=-1)
            if k > 1:
                weights = weights / weights.sum(-1, keepdim=True)
            expected = torch.zeros_like(flat)
            for i, expert in enumerate(block.experts):
                gates = (weights * (indices == i)).sum(-1, keepdim=True)
                expected = expected + expert(flat) * gates
            torch.testing.assert_close(actual, expected.reshape_as(x))
            (actual.square().mean() + .01 * aux).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(block.router.weight.grad.abs().sum().item(), 0)

    def test_causality_and_empty_experts(self):
        model = TinyGPT(20, 4, 8, 2, 2, moe=True)
        x = torch.tensor([[1, 2, 3, 4]])
        y, aux = model(x, return_aux=True)
        torch.testing.assert_close(y[:, :2], model(torch.tensor([[1, 2, 8, 9]]))[:, :2])
        self.assertTrue(torch.isfinite(aux))
        block = MixtureOfExperts(8, 4, 1)
        with torch.no_grad():
            block.router.weight.zero_()
            block.router.weight[0].fill_(1)
        result, _ = block(torch.ones(1, 1, 8))
        result.sum().backward()
        self.assertIsNone(block.experts[1].fc1.weight.grad)

    def test_training_and_legacy_resume(self):
        for enabled in (False, True):
            with self.subTest(moe=enabled), tempfile.TemporaryDirectory() as directory:
                model = TinyGPT(20, 4, 8, 2, 1, moe=enabled)
                optimizer = torch.optim.AdamW(model.parameters())
                path = Path(directory) / 'model.pt'
                training.save_checkpoint(path, model, optimizer, 0, 1., 0)
                saved = torch.load(path, weights_only=True)
                if not enabled:
                    for key in MOE_DEFAULTS:
                        del saved['config'][key]
                    torch.save(saved, path)
                x = torch.tensor([[1, 2, 3, 4]])
                with patch.object(training, 'getData', return_value=[]), \
                     patch.object(training, 'getBPE', return_value=SimpleNamespace(vocab_size=20)), \
                     patch.object(training, 'TokenizedDataReader', return_value=[]), \
                     patch.object(training, 'stream_batches', return_value=iter([(x, x)])), \
                     contextlib.redirect_stdout(io.StringIO()):
                    training.main(['--device', 'cpu', '--model', str(path), '--epochs', '1',
                                   '--verbose', 'false', '--no-moe' if enabled else '--moe'])
                checkpoint = torch.load(path, weights_only=True)
                self.assertEqual(checkpoint['config']['moe'], enabled)
                restored = TinyGPT(**checkpoint['config'])
                restored.load_state_dict(checkpoint['model_state_dict'], strict=True)
                optimizer = torch.optim.AdamW(restored.parameters())
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.assertTrue(torch.isfinite(restored(x)).all())


if __name__ == '__main__':
    unittest.main()
