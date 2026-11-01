import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import main as training
from tinygpt import TinyGPT


class CheckpointTests(unittest.TestCase):
    def test_lr_default_and_explicit_argument_tracking(self):
        args = training.parse_args([])
        self.assertEqual(args.lr, 0.0003)
        self.assertFalse(args.lr_explicit)
        for flags in (["--lr", "0.00003"], ["--lr=0.00003"], ["--lr", "0.0003"]):
            self.assertTrue(training.parse_args(flags).lr_explicit)
        for value in ("-1", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(["--lr", value])

    def test_resume_lr_override_preserves_optimizer_moments_and_other_settings(self):
        model = TinyGPT(20, 4, 12, 3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, betas=(0.8, 0.95), weight_decay=0.2)
        model(torch.tensor([[1, 2, 3, 4]])).square().mean().backward()
        optimizer.step()
        saved_optimizer = optimizer.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for flags, expected_lr in (([], 0.005), (["--lr", "0.00003"], 0.00003),
                                       (["--lr=0.00003"], 0.00003), (["--lr", "0.0003"], 0.0003)):
                with self.subTest(flags=flags):
                    training.save_checkpoint(path, model, optimizer, 0, 1.5, 5)
                    # Zero batches isolate restoration from subsequent optimizer updates.
                    with patch.object(training, "getData", return_value=[]), \
                            patch.object(training, "getBPE", return_value=SimpleNamespace(vocab_size=20)), \
                            patch.object(training, "TokenizedDataReader", return_value=[]), \
                            patch.object(training, "stream_batches", return_value=iter([])), \
                            contextlib.redirect_stdout(io.StringIO()) as output:
                        training.main(["--device", "cpu", "--model", str(path), "--epochs", "1", "--verbose", "false"] + flags)
                    result = torch.load(path, map_location="cpu")
                    restored = result["optimizer_state_dict"]
                    self.assertEqual(restored["param_groups"][0]["lr"], expected_lr)
                    self.assertEqual(restored["param_groups"][0]["betas"], (0.8, 0.95))
                    self.assertEqual(restored["param_groups"][0]["weight_decay"], 0.2)
                    self.assertEqual(result["global_step"], 5)
                    for key, state in saved_optimizer["state"].items():
                        for name, tensor in state.items():
                            torch.testing.assert_close(restored["state"][key][name], tensor, rtol=0, atol=0)
                    for name, tensor in model.state_dict().items():
                        torch.testing.assert_close(result["model_state_dict"][name], tensor, rtol=0, atol=0)
                    self.assertIn("--lr overrides checkpoint" if flags else "Learning rate (checkpoint)", output.getvalue())

    def test_failed_save_preserves_previous_checkpoint(self):
        model = TinyGPT(20, 4, 12, 3, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"previous checkpoint")
            with patch.object(training.torch, "save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    training.save_checkpoint(path, model, optimizer, 0, 1.0, 2, 2)
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_interval_saves_epoch_end_and_resume(self):
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 5]])
        snapshots = []
        real_save = training.save_checkpoint

        def capture(*args, **kwargs):
            real_save(*args, **kwargs)
            snapshots.append(torch.load(args[0], map_location="cpu"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            log = Path(directory) / "progress.csv"
            args = ["--device", "cpu", "--model", str(path), "--log-progress", str(log), "--d-model", "12", "--n-heads", "3", "--num-layers", "1", "--context-length", "4", "--save-every-steps", "2", "--verbose", "false"]
            with patch.object(training, "getData", return_value=[]), patch.object(training, "getBPE", return_value=SimpleNamespace(vocab_size=20)), patch.object(training, "TokenizedDataReader", return_value=[]), patch.object(training, "stream_batches", side_effect=lambda *args: iter([(x, y)] * 3)), patch.object(training, "save_checkpoint", side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
                training.main(args + ["--epochs", "1"])
                self.assertEqual([(s["epoch"], s["step"], s["global_step"]) for s in snapshots], [(0, 2, 2), (1, 0, 3)])
                periodic = snapshots[0]
                self.assertTrue(periodic["optimizer_state_dict"]["state"])
                torch.save(periodic, path)
                snapshots.clear()
                training.main(args + ["--epochs", "1"])
                self.assertEqual([(s["epoch"], s["step"], s["global_step"]) for s in snapshots], [(0, 2, 4), (1, 0, 5)])
                for name, tensor in periodic["model_state_dict"].items():
                    self.assertTrue(torch.isfinite(snapshots[-1]["model_state_dict"][name]).all())
                with log.open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 6)
                self.assertTrue(all(row['validation_loss'] == '' for row in rows))

    def test_negative_interval_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                training.parse_args(["--save-every-steps", "-1"])


if __name__ == "__main__":
    unittest.main()
