import contextlib
import copy
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import main as training
from tinygpt import TinyGPT


class CompilationTests(unittest.TestCase):
    def test_pretraining_optimization_defaults_and_opt_outs(self):
        for flags, fused, compiled in (
            ([], True, True), (['--no-fused-adamw'], False, True),
            (['--no-compile-loss', '--compile-model'], True, False),
        ):
            args = training.parse_args(flags)
            self.assertEqual((args.fused_adamw, args.compile_loss), (fused, compiled))

    def test_joint_loss_and_gradients_match_eager(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compiled_loss = torch.compile(training.pretraining_loss)
        x = torch.tensor([[1, 4, 3, 2]], device=device)
        y = torch.tensor([[4, 3, 2, 5]], device=device)
        for moe in (False, True):
            with self.subTest(moe=moe):
                torch.manual_seed(42)
                eager = TinyGPT(20, 4, 8, 2, 1, qk_norm=True, moe=moe).to(device)
                compiled = copy.deepcopy(eager)
                loss, objective = training.pretraining_loss(eager, x, y)
                actual_loss, actual_objective = compiled_loss(compiled, x, y)
                torch.testing.assert_close(actual_loss, loss)
                torch.testing.assert_close(actual_objective, objective)
                objective.backward()
                actual_objective.backward()
                for actual, expected in zip(compiled.parameters(), eager.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad)

    def test_switching_adamw_backend_resumes_the_same_update(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for start_fused in (False, True):
            with self.subTest(start_fused=start_fused):
                model = torch.nn.Linear(4, 2).to(device)
                reference = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.8, 0.95),
                                              fused=start_fused, weight_decay=0.03)
                x = torch.arange(4, dtype=torch.float32, device=device)
                model(x).sum().backward()
                reference.step()
                restored = copy.deepcopy(model)
                optimizer = training.pretraining_optimizer(restored, 0.2, not start_fused,
                                                           copy.deepcopy(reference.state_dict()))
                group = optimizer.param_groups[0]
                self.assertEqual(bool(group['fused']), not start_fused)
                self.assertEqual((group['lr'], group['betas'], group['weight_decay']), (0.01, (0.8, 0.95), 0.03))
                for original, param in zip(model.parameters(), restored.parameters()):
                    state = optimizer.state[param]
                    self.assertEqual(state['step'].device.type, 'cpu' if start_fused else device)
                    for key, value in reference.state[original].items():
                        torch.testing.assert_close(value.cpu(), state[key].cpu(), rtol=0, atol=0)
                for net, opt in ((model, reference), (restored, optimizer)):
                    opt.zero_grad(set_to_none=True)
                    net(x).square().sum().backward()
                    opt.step()
                for original, param in zip(model.parameters(), restored.parameters()):
                    torch.testing.assert_close(param, original)

    def test_progress_reports_heartbeat_and_failure(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with training.compilation_progress('test', interval=0.005):
                threading.Event().wait(0.03)
            with self.assertRaises(ValueError):
                with training.compilation_progress('failure'):
                    raise ValueError('test')
        text = output.getvalue()
        self.assertIn('starting test', text)
        self.assertIn('still running', text)
        self.assertIn('finished test', text)
        self.assertIn('failed or was interrupted', text)
        self.assertNotIn('finished failure', text)

    def test_whole_model_training_and_resume_preserve_checkpoint_format(self):
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 5]])

        def compile_for_test(model):
            # Compile forward and autograd on CPU without needing CUDA/Inductor.
            torch.nn.Module.compile(model, backend='aot_eager')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            flags = ['--device', 'cpu', '--model', str(path), '--verbose', 'false',
                     '--d-model', '12', '--n-heads', '3', '--num-layers', '1',
                     '--context-length', '4', '--qk-norm', '--compile-model', '--compile-qk-norm']
            with patch.object(training, 'getData', return_value=[]), \
                    patch.object(training, 'getBPE', return_value=SimpleNamespace(vocab_size=20)), \
                    patch.object(training, 'TokenizedDataReader', return_value=[]), \
                    patch.object(training, 'stream_batches', side_effect=lambda *a: iter([(x, y)])), \
                    patch.object(TinyGPT, 'compile', autospec=True, side_effect=compile_for_test) as compile_model, \
                    patch.object(TinyGPT, 'compile_qk_norms') as compile_qk, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                training.main(flags + ['--epochs', '1'])
                training.main(flags + ['--epochs', '2', '--no-compile-loss', '--no-fused-adamw'])
            self.assertEqual(compile_model.call_count, 2)
            compile_qk.assert_not_called()
            self.assertIn('finished first training backward', output.getvalue())
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint['global_step'], 2)
            restored = TinyGPT(**checkpoint['config'])
            restored.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer = torch.optim.AdamW(restored.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.assertTrue(torch.isfinite(restored(x)).all())


if __name__ == '__main__':
    unittest.main()
