"""Offline checks for the mixed Wikipedia/FineWeb corpus builder."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from corpus import discover_files, load_config
from other import prepare_wikipedia_fineweb_dataset as builder
from tinygpt import TinyGPT


PROSE = (
    "The natural world contains many different species of plants and animals. "
    "Scientists study these organisms to understand their relationships and the "
    "environments in which they live. "
)


def make_wikipedia(folder):
    counts = builder.wikipedia.prepare(
        [{"text": PROSE, "url": "https://en.wikipedia.org/wiki/Nature"},
         {"text": PROSE.replace("Scientists", "Researchers"), "url": "https://en.wikipedia.org/wiki/Science"}],
        folder, 2**20, 0, 100)
    summary = {**counts, "source": {"dataset": "wikimedia/wikipedia", "config": "20231101.en",
                                   "revision": builder.wikipedia.REVISION}}
    builder.wikipedia.write_json(folder / "metadata/summary.json", summary)
    return summary


def fineweb_record(text, **kwargs):
    return {"text": text, "language": "en", "language_score": .99,
            "url": "https://example.org/article", **kwargs}


class MixedDatasetTests(unittest.TestCase):
    def test_status_reports_even_when_stream_has_not_yielded_more_records(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = builder.CorpusWriter(Path(directory), io.StringIO())
            observed = threading.Event()
            messages = []

            def capture(message, **kwargs):
                messages.append(message)
                observed.set()

            with patch('builtins.print', side_effect=capture):
                with builder.report_progress(writer, 1024 * 2**20, interval=.01):
                    self.assertTrue(observed.wait(timeout=2))
                # Leaving the context joins the reporter; no background output remains.
                count = len(messages)
            self.assertEqual(len(messages), count)
            self.assertIn('1024.0 MiB remaining', messages[0])
            self.assertIn('network wait', messages[0])

    def test_reuses_wikipedia_and_streams_only_the_budget_then_writes_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wiki = root / "wiki"
            old_summary = make_wikipedia(wiki)
            original = (wiki / "metadata/files.jsonl").read_bytes()
            records = (fineweb_record(f"Document number {i}. " + PROSE * 20, id=str(i)) for i in range(1000))
            output = root / "mixed"
            with patch.object(builder.wikipedia, "main") as download_wiki, \
                 patch.object(builder, "download_fineweb", return_value=(records, "test-revision")) as download_web, \
                 contextlib.redirect_stdout(io.StringIO()):
                builder.main(["--output", str(output), "--wikipedia-dir", str(wiki),
                              "--fineweb-mib", "1", "--min-chars", "100"])
            download_wiki.assert_not_called()
            download_web.assert_called_once_with(42, revision=None)
            summary = json.loads((output / "metadata/summary.json").read_text())
            self.assertEqual(summary["counts"]["wikipedia"]["bytes"], old_summary["bytes"])
            self.assertLessEqual(summary["counts"]["fineweb-edu"]["bytes"], 2**20)
            self.assertGreater(summary["counts"]["fineweb-edu"]["bytes"], 2**20 - 10000)
            self.assertLess(summary["counts"]["fineweb-edu"]["examined"], 1000)
            self.assertEqual(summary["sources"]["fineweb-edu"]["revision"], "test-revision")
            manifest = [json.loads(line) for line in (output / "metadata/files.jsonl").read_text().splitlines()]
            files = discover_files(output / "corpus", load_config(output / "training_data.json"))
            self.assertEqual(set(files), {output / r["path"] for r in manifest})
            self.assertEqual(sum(r["bytes"] for r in manifest), summary["bytes"])
            for record in manifest:
                data = (output / record["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                self.assertEqual(len(data), record["bytes"])
                self.assertTrue(data.endswith(b'\n\n'))
                self.assertTrue(record["url"])
            self.assertEqual((wiki / "metadata/files.jsonl").read_bytes(), original)
            launcher = output / "train.sh"
            subprocess.run(["bash", "-n", str(launcher)], check=True)
            # Execute with a harmless Python stand-in to verify argument quoting,
            # defaults, fresh artifact paths and caller overrides.
            capture = root / "capture args"
            capture.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            capture.chmod(0o755)
            result = subprocess.run(["bash", str(launcher), "--epochs", "1"],
                                    env={**os.environ, "PYTHON": str(capture)}, check=True,
                                    capture_output=True, text=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[args.index('--d-model') + 1], '512')
            self.assertEqual(args[args.index('--num-layers') + 1], '8')
            self.assertEqual(args[args.index('--n-heads') + 1], '8')
            self.assertEqual(args[args.index('--batch-size') + 1], '4')
            self.assertEqual(args[-2:], ['--epochs', '1'])
            self.assertEqual(args[args.index('--model') + 1], str(output / 'artifacts/model.pt'))
            with torch.device('meta'):
                model = TinyGPT(8192, 512, 512, 8, 8)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 29676544)
            with patch.object(builder, "download_fineweb") as download, self.assertRaises(SystemExit):
                builder.main(["--output", str(output), "--wikipedia-dir", str(wiki)])
            download.assert_not_called()

    def test_downloads_missing_or_empty_wikipedia_cache(self):
        for empty in (False, True):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / 'wiki'
                if empty:
                    folder.mkdir()
                with patch.object(builder.wikipedia, 'main', side_effect=lambda argv: make_wikipedia(folder)) as download:
                    summary = builder.wikipedia_source(folder, 256, 100, 42)
                download.assert_called_once_with(['--output', str(folder), '--max-mib', '256',
                                                  '--min-chars', '100', '--seed', '42'])
                self.assertEqual(summary['source']['config'], '20231101.en')

    def test_filters_and_deduplicates_before_measuring_utf8_budget(self):
        text = PROSE.replace('natural world', 'world of café owners')
        data = builder.wikipedia.clean_text(text).encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            writer = builder.CorpusWriter(Path(directory), io.StringIO())
            writer.add('wikipedia', builder.wikipedia.clean_text(PROSE).encode(), {})
            records = [fineweb_record(PROSE), fineweb_record(text, language='fr'),
                       fineweb_record(text, language_score=.5), fineweb_record('def main(): return 0;'),
                       fineweb_record(text + '\x00'), fineweb_record(text),
                       fineweb_record('This must never be consumed')]
            builder.prepare_fineweb(records, writer, len(data), 100)
            counts = writer.counts['fineweb-edu']
            self.assertEqual(counts['files'], 1)
            self.assertEqual(counts['bytes'], len(data))
            self.assertEqual(counts['duplicates'], 1)
            self.assertEqual(counts['language_filtered'], 2)
            self.assertEqual(counts['too_short_after_filtering'], 1)
            self.assertEqual(counts['invalid_text'], 1)
            self.assertEqual(counts['examined'], 6)

    def test_rejects_corrupt_or_unverified_wikipedia(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'wiki'
            summary = make_wikipedia(folder)
            first = next((folder / 'corpus').rglob('*.txt'))
            first.write_text('changed')
            writer = builder.CorpusWriter(Path(directory) / 'mixed', io.StringIO())
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                builder.copy_wikipedia(folder, summary, writer)
            summary['source'] = {'local_source': 'fullEnglish'}
            builder.wikipedia.write_json(folder / 'metadata/summary.json', summary)
            with patch.object(builder.wikipedia, 'main') as download, self.assertRaisesRegex(ValueError, 'Expected downloaded English'):
                builder.wikipedia_source(folder, 256, 100, 42)
            download.assert_not_called()

    def test_exhausted_stream_is_not_reported_as_a_completed_gigabyte(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = builder.CorpusWriter(Path(directory), io.StringIO())
            with self.assertRaisesRegex(ValueError, 'stream ended'):
                builder.prepare_fineweb([fineweb_record(PROSE)], writer, 2**20, 100)

    def test_download_caches_one_shard_at_a_time_and_reads_without_threads(self):
        hub, parquet = Mock(), MagicMock()
        arrow = ModuleType('pyarrow')
        arrow.parquet = parquet
        hub.HfApi.return_value.dataset_info.return_value.sha = 'resolved-revision'
        hub.HfApi.return_value.list_repo_files.return_value = [
            'README.md', 'sample/10BT/000.parquet', 'sample/10BT/001.parquet',
            'data/unwanted.parquet']
        hub.hf_hub_download.side_effect = [OSError('connection closed'), '/tmp/cached.parquet']
        source = parquet.ParquetFile.return_value.__enter__.return_value
        batch = Mock()
        batch.to_pylist.return_value = [fineweb_record(PROSE)]
        source.iter_batches.return_value = [batch]
        with patch.dict('sys.modules', {'pyarrow': arrow, 'pyarrow.parquet': parquet, 'huggingface_hub': hub}), \
             patch.object(builder.time, 'sleep'):
            records, revision = builder.download_fineweb(42)
            hub.hf_hub_download.assert_not_called()
            self.assertEqual(next(records)['text'], PROSE)
            self.assertEqual(hub.hf_hub_download.call_count, 2)
            self.assertEqual(hub.hf_hub_download.call_args_list[0], hub.hf_hub_download.call_args_list[1])
            records.close()
        self.assertEqual(revision, 'resolved-revision')
        source.iter_batches.assert_called_once_with(batch_size=64, use_threads=False)
        parquet.ParquetFile.return_value.__exit__.assert_called_once()
        self.assertTrue(hub.hf_hub_download.call_args.args[1].startswith('sample/10BT/'))
        self.assertEqual(hub.hf_hub_download.call_args.kwargs['revision'], 'resolved-revision')

    def test_finalize_old_interrupted_output_without_any_network_and_preserve_orphans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'mixed'
            wiki = Path(directory) / 'wiki'
            summary = make_wikipedia(wiki)
            (root / 'metadata').mkdir(parents=True)
            manifest_path = root / 'metadata/files.jsonl'
            with manifest_path.open('w') as manifest:
                writer = builder.CorpusWriter(root, manifest)
                builder.copy_wikipedia(wiki, summary, writer)
                writer.add('fineweb-edu', builder.wikipedia.clean_text(PROSE.replace('Scientists', 'Biologists')).encode(), {})
            original_records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            original_data = {r['path']: (root / r['path']).read_bytes() for r in original_records}
            with manifest_path.open('ab') as output:
                output.write(b'{"path":')
            orphan = root / 'corpus/fineweb-edu/00000/00000001.txt'
            orphan.write_text('text written before its manifest row')
            with patch.object(builder, 'download_fineweb') as download, \
                 patch.object(builder.wikipedia, 'main') as wiki_download, \
                 contextlib.redirect_stdout(io.StringIO()):
                builder.main(['--output', str(root), '--wikipedia-dir', str(Path(directory) / 'absent'),
                              '--finalize-existing'])
            download.assert_not_called()
            wiki_download.assert_not_called()
            summary = json.loads((root / 'metadata/summary.json').read_text())
            self.assertTrue(summary['finalized_existing'])
            self.assertEqual(summary['files'], 3)
            self.assertIsNone(summary['sources']['fineweb-edu']['revision'])
            self.assertEqual((root / 'metadata/recovered-orphans/corpus/fineweb-edu/00000/00000001.txt').read_text(),
                             'text written before its manifest row')
            self.assertEqual(next((root / 'metadata').glob('interrupted-manifest-tail-*.bin')).read_bytes(), b'{"path":')
            for path, data in original_data.items():
                self.assertEqual((root / path).read_bytes(), data)
            self.assertEqual(len(discover_files(root / 'corpus', load_config(root / 'training_data.json'))), 3)
            self.assertTrue((root / 'train.sh').is_file())

    def test_resume_preserves_documents_and_recovers_after_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'mixed'
            wiki = Path(directory) / 'wiki'
            make_wikipedia(wiki)
            first = fineweb_record('The first example. ' + PROSE * 20)
            closed = []

            def interrupted():
                try:
                    yield first
                    raise KeyboardInterrupt
                finally:
                    closed.append(True)

            args = ['--output', str(root), '--wikipedia-dir', str(wiki), '--fineweb-mib', '1', '--min-chars', '100']
            with patch.object(builder, 'download_fineweb', return_value=(interrupted(), 'original-revision')), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                builder.main(args)
            self.assertEqual(error.exception.code, 130)
            self.assertEqual(closed, [True])
            self.assertFalse((root / 'metadata/summary.json').exists())
            manifest = root / 'metadata/files.jsonl'
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            before = {r['path']: (root / r['path']).stat().st_mtime_ns for r in records}
            remaining = iter([first] + [fineweb_record(f'Example {i}. ' + PROSE * 20) for i in range(1000)])
            with patch.object(builder, 'download_fineweb', return_value=(remaining, 'original-revision')) as download, \
                 contextlib.redirect_stdout(io.StringIO()):
                builder.main(['--output', str(root), '--wikipedia-dir', str(wiki), '--resume'])
            download.assert_called_once_with(42, revision='original-revision')
            summary = json.loads((root / 'metadata/summary.json').read_text())
            self.assertEqual(summary['limits']['fineweb_bytes'], 2**20)
            self.assertLessEqual(summary['counts']['fineweb-edu']['bytes'], 2**20)
            self.assertGreater(summary['counts']['fineweb-edu']['bytes'], 2**20 - 10000)
            for path, modified in before.items():
                self.assertEqual((root / path).stat().st_mtime_ns, modified)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len({r['sha256'] for r in records}), len(records))


if __name__ == '__main__':
    unittest.main()
