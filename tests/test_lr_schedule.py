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
from lr_schedule import WarmupCosine, resolve_schedule


class ScheduleTests(unittest.TestCase):
    def test_warmup_cosine_boundaries_floor_and_no_warmup(self):
        schedule = WarmupCosine(1.0, 2, 6, 0.1)
        rates = [schedule.rate(step) for step in range(8)]
        self.assertEqual(rates[:2], [0.5, 1.0])
        self.assertAlmostEqual(rates[3], 0.55)
        self.assertEqual(rates[5:], [0.1, 0.1, 0.1])
        self.assertTrue(all(a >= b for a, b in zip(rates[1:], rates[2:])))
        self.assertEqual(WarmupCosine(1, 0, 3, 0).rate(0), 1)
        self.assertEqual(WarmupCosine(1, 0, 3, 0).rate(2), 0)
        self.assertEqual(WarmupCosine(1, 0, 1, 0.1).rate(0), 0.1)

    def test_new_schedule_on_legacy_checkpoint_starts_at_current_update(self):
        args = training.parse_args(['--lr-schedule', 'cosine', '--warmup-steps', '2'])
        schedule = resolve_schedule(args, None, 0.001, 900000, 100)
        self.assertEqual(schedule.start_step, 900000)
        self.assertEqual(schedule.decay_steps, 100)
        self.assertEqual(schedule.rate(900000), 0.0005)
        self.assertEqual(schedule.rate(900001), 0.001)
        self.assertAlmostEqual(schedule.rate(900099), 0.0001)

    def test_resume_retains_horizon_and_peak_unless_explicitly_overridden(self):
        saved = WarmupCosine(0.001, 10, 100).state_dict()
        schedule = resolve_schedule(training.parse_args([]), saved, 0.0002, 90, 500)
        self.assertEqual(schedule.state_dict(), saved)
        args = training.parse_args(['--lr', '0.0003', '--lr-decay-steps', '200'])
        changed = resolve_schedule(args, saved, 0.0002, 90, 500)
        self.assertEqual(changed.peak_lr, 0.0003)
        self.assertEqual(changed.decay_steps, 200)
        self.assertEqual(changed.start_step, 0)
        self.assertEqual(changed.warmup_steps, 10)
        self.assertIsNone(resolve_schedule(training.parse_args(['--lr-schedule', 'constant']),
                                           saved, 0.0002, 90, 500))
        with self.assertRaisesRegex(ValueError, 'require --lr-schedule cosine'):
            resolve_schedule(training.parse_args(['--warmup-steps', '2']), None, 0.001, 0, 100)
        with self.assertRaisesRegex(ValueError, 'requires 0 <='):
            resolve_schedule(training.parse_args(['--lr-schedule', 'cosine', '--warmup-steps', '100']),
                             None, 0.001, 0, 100)

    def test_invalid_options_fail_before_training(self):
        for flags in (['--warmup-steps', '-1'], ['--lr-decay-steps', '0'],
                      ['--min-lr-ratio', '-0.1'], ['--min-lr-ratio', '1.1'],
                      ['--min-lr-ratio', 'nan'], ['--min-lr-ratio', 'inf'],
                      ['--lr-schedule', 'constant', '--warmup-steps', '2'],
                      ['--warmup-steps', '10', '--lr-decay-steps', '10']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(flags)

    def test_optimizer_rates_and_mid_epoch_resume_match_uninterrupted_training(self):
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 5]])
        snapshots = []
        save = training.save_checkpoint

        def capture(*args, **kwargs):
            save(*args, **kwargs)
            snapshots.append(torch.load(args[0], map_location='cpu'))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, log = root / 'model.pt', root / 'progress.csv'
            flags = ['--device', 'cpu', '--model', str(path), '--log-progress', str(log),
                     '--d-model', '12', '--n-heads', '3', '--num-layers', '1',
                     '--context-length', '4', '--batch-size', '2', '--save-every-steps', '2',
                     '--verbose', 'false']
            # Eight windows / batch size two = four optimizer updates per epoch.
            with patch.object(training, 'getData', return_value=[]), \
                    patch.object(training, 'getBPE', return_value=SimpleNamespace(vocab_size=20)), \
                    patch.object(training, 'TokenizedDataReader', return_value=[None] * 8), \
                    patch.object(training, 'stream_batches', side_effect=lambda *a: iter([(x, y)] * 4)), \
                    patch.object(training, 'save_checkpoint', side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()):
                training.main(flags + ['--epochs', '2', '--lr-schedule', 'cosine',
                                       '--warmup-steps', '2', '--lr', '0.001'])
                uninterrupted = next(s for s in snapshots if s['global_step'] == 6)
                periodic = snapshots[0]
                self.assertEqual(periodic['global_step'], 2)
                self.assertEqual(periodic['lr_schedule']['decay_steps'], 8)
                expected = WarmupCosine(**periodic['lr_schedule'])
                with log.open() as source:
                    rows = list(csv.DictReader(source))
                for step, row in enumerate(rows):
                    self.assertAlmostEqual(float(row['lr']), expected.rate(step))
                self.assertEqual(float(rows[-1]['lr']), 0.0001)
                torch.save(periodic, path)
                snapshots.clear()
                # Omit all scheduler flags: restore peak, horizon, and global progress.
                training.main(flags + ['--epochs', '1'])
                resumed = snapshots[-1]
                self.assertEqual(resumed['global_step'], 6)
                self.assertEqual(resumed['lr_schedule'], periodic['lr_schedule'])
                self.assertEqual(resumed['optimizer_state_dict']['param_groups'][0]['lr'],
                                 expected.rate(5))
                for name, tensor in uninterrupted['model_state_dict'].items():
                    torch.testing.assert_close(resumed['model_state_dict'][name], tensor, rtol=0, atol=0)

    def test_five_column_log_upgrade_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.csv'
            path.write_text('epoch,step,loss,mean_loss,validation_loss\n1,1,3,3,4\n')
            training.prepare_progress_log(path)
            with path.open() as source:
                self.assertEqual(list(csv.DictReader(source)),
                                 [dict(epoch='1', step='1', loss='3', mean_loss='3', validation_loss='4', lr='')])


if __name__ == '__main__':
    unittest.main()
