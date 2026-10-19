"""Offline checks for Wikipedia filtering, provenance and corpus integration."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from corpus import discover_files, load_config
from other import prepare_wikipedia_dataset as builder


PROSE = (
    "The natural world contains many different species of plants and animals. "
    "Scientists study these organisms to understand their relationships and the "
    "environments in which they live. "
)


class WikipediaPreparationTests(unittest.TestCase):
    def test_filters_code_and_markup_while_preserving_prose(self):
        text = PROSE + "\n\n```python\n\ndef example():\n    return 1\n\n```\n\n"
        text += "<pre>int main() { return 0; }</pre>\n\n"
        text += "def another():\n    return 2\n\n{{template}}\n\nShort heading\n\n"
        text += PROSE.replace("natural world", "world of café owners")
        cleaned = builder.clean_text(text)
        self.assertIn(PROSE.strip(), cleaned)
        self.assertIn("café", cleaned)
        for unwanted in ("return", "template", "Short heading", "```", "<pre>"):
            self.assertNotIn(unwanted, cleaned)
        self.assertTrue(cleaned.endswith("\n\n"))
        with self.assertRaisesRegex(ValueError, "NUL"):
            builder.clean_text(PROSE + "\x00")

    def test_downloaded_corpus_records_article_provenance_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            articles = [
                {"id": "1", "title": "Nature", "url": "https://en.wikipedia.org/wiki/Nature", "text": PROSE},
                {"id": "2", "title": "Duplicate", "url": "https://en.wikipedia.org/wiki/Duplicate", "text": PROSE},
                {"id": "3", "title": "Science", "url": "https://en.wikipedia.org/wiki/Science", "text": PROSE.replace("Scientists", "Researchers")},
            ]
            root = Path(directory) / "output"
            with patch.object(builder, "download_records", return_value=articles) as download:
                builder.main(["--output", str(root), "--min-chars", "100"])
            download.assert_called_once_with(42)
            files = discover_files(root / "corpus", load_config(root / "training_data.json"))
            self.assertEqual(len(files), 2)
            summary = json.loads((root / "metadata/summary.json").read_text())
            self.assertEqual(summary["duplicates"], 1)
            self.assertEqual(summary["source"]["dataset"], "wikimedia/wikipedia")
            self.assertEqual(summary["source"]["config"], "20231101.en")
            self.assertNotIn("local_source", summary["source"])
            records = [json.loads(line) for line in (root / "metadata/files.jsonl").read_text().splitlines()]
            self.assertEqual({root / r["path"] for r in records}, set(files))
            for record in records:
                data = (root / record["path"]).read_bytes()
                self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(record["bytes"], len(data))
                self.assertTrue(record["url"].startswith("https://en.wikipedia.org/wiki/"))
                self.assertIn(record["id"], {"1", "3"})
            self.assertIn("--validation-fraction 0.05", (root / "train.sh").read_text())
            # A second preparation cannot destroy an existing corpus or artifacts.
            before = files[0].read_bytes()
            with self.assertRaises(SystemExit):
                builder.main(["--output", str(root)])
            self.assertEqual(files[0].read_bytes(), before)

    def test_output_budget_keeps_whole_files(self):
        records = [{"text": PROSE}, {"text": PROSE.replace("Scientists", "Researchers")},
                   {"text": PROSE.replace("Scientists", "Biologists")}]
        budget = sum(len(builder.clean_text(r["text"]).encode()) for r in records[:2])
        with tempfile.TemporaryDirectory() as directory:
            counts = builder.prepare(records, Path(directory) / "output", budget, 0, 100)
            self.assertEqual(counts["files"], 2)
            self.assertEqual(counts["bytes"], budget)
            self.assertEqual(counts["over_byte_budget"], 1)

    def test_local_source_option_is_rejected(self):
        with patch.object(builder, "download_records") as download:
            with self.assertRaises(SystemExit):
                builder.main(["--local-source", "fullEnglish"])
            download.assert_not_called()

    def test_download_uses_only_pinned_english_snapshot_and_bounded_shuffle(self):
        datasets = Mock()
        with patch.dict("sys.modules", {"datasets": datasets}):
            records = builder.download_records(42)
        datasets.load_dataset.assert_called_once_with(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True,
            revision=builder.REVISION)
        datasets.load_dataset.return_value.shuffle.assert_called_once_with(seed=42, buffer_size=1000)
        self.assertIs(records, datasets.load_dataset.return_value.shuffle.return_value)


if __name__ == "__main__":
    unittest.main()
