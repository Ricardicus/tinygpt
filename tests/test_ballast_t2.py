import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from other import download_datasets as b


class BallastTests(unittest.TestCase):
    def test_streams_passages_with_pinned_revision(self):
        datasets, hub = Mock(), Mock()
        datasets.load_dataset.return_value = [{'text': 'Short definition.'}]
        hub.HfApi.return_value.dataset_info.return_value.sha = 'commit123'
        with patch.dict('sys.modules', {'datasets': datasets, 'huggingface_hub': hub}):
            metadata = {}
            self.assertEqual(list(b.source_records('ballast-t2', metadata)), [{'text': 'Short definition.'}])
        datasets.load_dataset.assert_called_once_with('OpenBallast/ballast-t2', name='passages', split='train', revision='commit123', streaming=True)
        self.assertEqual(metadata['revision'], 'commit123')

    def test_preserves_short_passages_and_provenance(self):
        record = {'text': 'x² = 4\r\n\r\nExample: x = 2', 'chunk_id': 'abc', 'qid': 'X:book:1',
                  'source_id': 'openstax-book', 'source_url': 'https://example.org/source',
                  'revision': 'gitsha', 'license_id': 'CC-BY-4.0', 'license_evidence': 'LICENSE@gitsha', 'section': 'definition'}
        self.assertEqual(b.record_text('ballast-t2', record), 'x² = 4\n\nExample: x = 2\n\n')
        with tempfile.TemporaryDirectory() as directory, patch.object(b, 'source_records', return_value=iter([record, record])):
            root = Path(directory) / 'out'
            result = b.prepare({'ballast-t2': None}, root)
            row = json.loads((root / 'metadata/files.jsonl').read_text())
            self.assertEqual(row['source_url'], record['source_url'])
            self.assertEqual(row['license_id'], 'CC-BY-4.0')
            self.assertEqual(row['revision'], 'gitsha')
            self.assertEqual(row['sha256'], hashlib.sha256((root / row['path']).read_bytes()).hexdigest())
            self.assertEqual(result['counts']['ballast-t2']['duplicates'], 1)

    def test_cap_and_cli(self):
        data = [{'text': 'One.'}, {'text': 'Two.'}]
        with tempfile.TemporaryDirectory() as directory, patch.object(b, 'source_records', return_value=iter(data)):
            result = b.prepare({'ballast-t2': 7}, Path(directory) / 'out')
            self.assertEqual(result['bytes'], 6)
            self.assertEqual(result['counts']['ballast-t2']['stopped_at_byte_limit'], 1)
        with patch.object(b, 'prepare') as prepare:
            b.main(['--ballast-t2', 'full', '--output', '/tmp/ballast'])
            prepare.assert_called_once_with({'ballast-t2': None}, Path('/tmp/ballast'))


if __name__ == '__main__':
    unittest.main()
