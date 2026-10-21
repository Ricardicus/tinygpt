"""Offline checks for the interactive corpus downloader."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from other import download_datasets as builder

PROSE = (
    "The natural world contains many different species of plants and animals. "
    "Scientists study these organisms to understand their relationships and the "
    "environments in which they live. "
)


class DownloadDatasetsTests(unittest.TestCase):
    def test_limits_are_decimal_gb_or_full(self):
        self.assertEqual(builder.parse_limit("0.125"), 125_000_000)
        self.assertEqual(builder.parse_limit("0.000000001"), 1)
        self.assertIsNone(builder.parse_limit(" FULL "))
        for invalid in ("0", "-1", "nan", "inf", "", "many", "0.0000000001"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                builder.parse_limit(invalid)

    def test_default_folder_describes_sources_and_limits(self):
        self.assertEqual(builder.default_folder({"wikipedia": 250_000_000, "dolly": None}),
                         "datasets/wikipedia-0.25gb_dolly-full")

    def test_dolly_preserves_short_responses_and_context(self):
        text = builder.record_text("dolly", {"instruction": "Add 1 and 1", "context": "Use integers", "response": "2"})
        self.assertEqual(text, "### Instruction\nAdd 1 and 1\n\n### Context\nUse integers\n\n### Response\n2\n\n")
        self.assertEqual(builder.record_text("dolly", {"instruction": "Incomplete"}), "")

    def test_gutenberg_keeps_notices_and_normalizes_boundaries(self):
        self.assertEqual(builder.record_text("gutenberg", {"text": "License\r\n\r\nBook\r\n"}),
                         "License\n\nBook\n\n")

    def test_gutenberg_uses_seven_day_chart_and_utf8_text_endpoint(self):
        html = '<h2 id="books-last7">Top 100</h2><ol>'
        html += ''.join(f'<li><a href="/ebooks/{i}">Book {i} &amp; author</a></li>' for i in range(100))
        html += '</ol><h2>Other chart</h2><ol><li><a href="/ebooks/999">Ignore</a></li></ol>'
        chart_response, book_response = Mock(), Mock()
        chart_response.read.return_value = html.encode()
        book_response.read.return_value = b'\xef\xbb\xbf*** START OF THE PROJECT GUTENBERG EBOOK ***\nBook\nLicense'
        book_response.url = "https://www.gutenberg.org/cache/epub/0/pg0.txt"
        responses = [Mock(), Mock()]
        responses[0].__enter__ = Mock(return_value=chart_response)
        responses[1].__enter__ = Mock(return_value=book_response)
        for response in responses:
            response.__exit__ = Mock(return_value=False)
        with patch.object(builder, "urlopen", side_effect=responses) as get, patch.object(builder.time, "sleep"):
            records = builder.gutenberg_records()
            first = next(records)
            records.close()
        self.assertEqual(first["id"], "0")
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["title"], "Book 0 & author")
        self.assertFalse(first["text"].startswith("\ufeff"))
        self.assertEqual(get.call_args_list[1].args[0].full_url, "https://www.gutenberg.org/ebooks/0.txt.utf-8")

    def test_mixed_download_full_exhaustion_hashes_and_training_files(self):
        sources = {"wikipedia": [{"id": "1", "url": "https://example.org/wiki", "text": PROSE},
                                 {"id": "2", "text": PROSE}],
                   "dolly": [{"instruction": "Question?", "response": "Yes", "category": "open_qa"}],
                   "fineweb-edu": [{"text": PROSE}, {"text": PROSE.replace("Scientists", "Researchers")}],
                   "gutenberg": [{"id": "84", "rank": 1, "text": "Book\n\nLicense notice"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mixed"
            with patch.object(builder, "source_records", side_effect=lambda name, meta: iter(sources[name])):
                result = builder.prepare(dict.fromkeys(sources), root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["files"], 4)
            self.assertEqual(result["counts"]["wikipedia"]["duplicates"], 1)
            self.assertEqual(result["counts"]["fineweb-edu"]["duplicates"], 1)
            rows = [json.loads(line) for line in (root / "metadata/files.jsonl").read_text().splitlines()]
            for row in rows:
                data = (root / row["path"]).read_bytes()
                self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(row["bytes"], len(data))
                self.assertTrue(data.endswith(b"\n\n"))
            self.assertEqual(sum(row["bytes"] for row in rows), result["bytes"])
            self.assertIn("--validation-fraction 0.05", (root / "train.sh").read_text())
            self.assertEqual(json.loads((root / "training_data.json").read_text())["suffixes"], [".txt"])
            self.assertIn("Dolly", (root / "README.md").read_text())

    def test_budget_stops_at_whole_document_and_closes_iterator(self):
        doc = {"instruction": "A question", "response": "A response"}
        size = len(builder.record_text("dolly", doc).encode())
        iterator = Mock()
        iterator.__iter__ = Mock(return_value=iter([doc, {**doc, "response": "Another answer"}]))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(builder, "source_records", return_value=iterator):
                result = builder.prepare({"dolly": size + 1}, Path(directory) / "limited")
        self.assertEqual(result["bytes"], size)
        self.assertEqual(result["counts"]["dolly"]["stopped_at_byte_limit"], 1)
        iterator.close.assert_called_once()

    def test_rejects_existing_output_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "precious.txt").write_text("keep")
            with patch.object(builder, "source_records") as download:
                with self.assertRaisesRegex(ValueError, "never overwritten"):
                    builder.prepare({"dolly": None}, root)
                download.assert_not_called()
            self.assertEqual((root / "precious.txt").read_text(), "keep")

    def test_network_error_preserves_partial_manifest_and_status(self):
        def interrupted():
            yield {"instruction": "Hello", "response": "Hi"}
            raise OSError("offline")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "partial"
            with patch.object(builder, "source_records", return_value=interrupted()):
                with self.assertRaisesRegex(OSError, "offline"):
                    builder.prepare({"dolly": None}, root)
            summary = json.loads((root / "metadata/summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["files"], 1)
            self.assertEqual(len((root / "metadata/files.jsonl").read_text().splitlines()), 1)
            self.assertFalse((root / "train.sh").exists())

    def test_zero_usable_documents_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(builder, "source_records", return_value=iter([{"instruction": "Hello", "response": "Hi"}])):
                with self.assertRaisesRegex(ValueError, "larger limit"):
                    builder.prepare({"dolly": 1}, Path(directory) / "tiny")

    def test_hub_stream_pins_revision_and_uses_complete_fineweb(self):
        datasets, hub = Mock(), Mock()
        hub.HfApi.return_value.dataset_info.return_value.sha = "resolved-sha"
        datasets.load_dataset.return_value = [{"text": "test"}]
        with patch.dict("sys.modules", {"datasets": datasets, "huggingface_hub": hub}):
            provenance = {}
            self.assertEqual(list(builder.source_records("fineweb-edu", provenance)), [{"text": "test"}])
        datasets.load_dataset.assert_called_once_with("HuggingFaceFW/fineweb-edu", name="default",
                                                     split="train", revision="resolved-sha", streaming=True)
        self.assertEqual(provenance["revision"], "resolved-sha")

    def test_plain_wizard_reprompts_invalid_values_and_accepts_custom_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "custom"
            with patch("builtins.input", side_effect=["5", "2,4", "nan", "0.5", "full", str(root), "y"]):
                selected, actual_root = builder.plain_plan()
        self.assertEqual(selected, {"dolly": 500_000_000, "gutenberg": None})
        self.assertEqual(actual_root, root)

    def test_tui_multiselection_limit_edit_and_folder(self):
        import curses
        screen = Mock()
        screen.getmaxyx.return_value = (24, 110)
        # Start with all four, uncheck Wikipedia, edit Dolly, keep remaining sources.
        screen.get_wch.side_effect = [" ", curses.KEY_DOWN, "e", "\x15", "0", ".", "5", "\n",
                                     "o", "/", "t", "m", "p", "/", "t", "u", "i", "-", "x", "\n", "\n", "\n"]
        with patch.object(curses, "wrapper", side_effect=lambda f: f(screen)), \
                patch.object(curses, "curs_set"), patch.object(builder, "validate_output", side_effect=Path):
            selected, root = builder.tui_plan()
        self.assertEqual(selected, {"dolly": 500_000_000, "fineweb-edu": 1_000_000_000, "gutenberg": None})
        self.assertEqual(root, Path("/tmp/tui-x"))


if __name__ == "__main__":
    unittest.main()
