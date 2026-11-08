import contextlib
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
                training.main(flags + ['--epochs', '2'])
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
