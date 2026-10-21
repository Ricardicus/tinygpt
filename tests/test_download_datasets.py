"""Offline checks for the interactive corpus downloader."""
import csv
import gzip
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

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

    def test_catalog_reads_languages_and_filters_non_text(self):
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(["Text#", "Type", "Title", "Language", "Authors"])
        writer.writerows([["1", "Text", "English, book", "en", "Author"],
                          ["2", "Text", "Bilingual", "en; fr", "Author 2"],
                          ["3", "Sound", "Audio", "de", "Speaker"]])
        data = gzip.compress(text.getvalue().encode("utf-8-sig"))
        builder.gutenberg_catalog.cache_clear()
        try:
            with patch.object(builder, "fetch_gutenberg", return_value=(data, "catalog-url")) as fetch:
                books, metadata = builder.gutenberg_catalog()
                self.assertEqual(builder.gutenberg_languages(), [("en", 2), ("fr", 1)])
                fetch.assert_called_once()
            self.assertEqual(len(books), 2)
            self.assertEqual(books[1]["languages"], ["en", "fr"])
            self.assertEqual(metadata["catalog_sha256"], hashlib.sha256(data).hexdigest())
        finally:
            builder.gutenberg_catalog.cache_clear()

    def test_catalog_downloader_exceeds_100_and_skips_missing_text(self):
        books = [{"id": str(i), "languages": ["en"], "title": f"Book {i}"} for i in range(1, 104)]
        books += [{"id": "104", "languages": ["fr"], "title": "French"}]
        data = b"\xef\xbb\xbf*** START OF THE PROJECT GUTENBERG EBOOK ***\nBook\nLicense"
        replies = [HTTPError("url", 404, "No text", None, None)] + [(data, "text-url")] * 102
        provenance = {}
        with patch.object(builder, "gutenberg_catalog", return_value=(books, {"catalog_sha256": "hash"})), \
                patch.object(builder, "fetch_gutenberg", side_effect=replies) as fetch, \
                patch.object(builder.time, "sleep"):
            records = list(builder.gutenberg_records(("en",), provenance))
        self.assertEqual(len(records), 102)
        self.assertEqual(fetch.call_count, 103)
        self.assertEqual(records[-1]["id"], "103")
        self.assertEqual(provenance["skipped_unavailable"], 1)
        self.assertEqual(provenance["languages"], ["en"])
        self.assertFalse(records[0]["text"].startswith("\ufeff"))

    def test_language_union_includes_bilingual_books_once(self):
        books = [{"id": "1", "languages": ["en", "fr"], "title": "Bilingual"},
                 {"id": "2", "languages": ["de"], "title": "German"}]
        data = b"*** START OF THE PROJECT GUTENBERG EBOOK ***\nBook"
        with patch.object(builder, "gutenberg_catalog", return_value=(books, {})), \
                patch.object(builder, "fetch_gutenberg", return_value=(data, "url")), \
                patch.object(builder.time, "sleep"):
            records = list(builder.gutenberg_records(("en", "fr")))
        self.assertEqual([record["id"] for record in records], ["1"])

    def test_plain_language_selection_defaults_english_and_accepts_multiple(self):
        with patch.object(builder, "gutenberg_languages", return_value=[("fr", 200), ("en", 100)]):
            with patch("builtins.input", return_value=""):
                self.assertEqual(builder.plain_languages(), ("en",))
            with patch("builtins.input", side_effect=["0", "1,2"]):
                self.assertEqual(builder.plain_languages(), ("fr", "en"))

    def test_language_screen_defaults_english_and_can_select_multiple(self):
        import curses
        screen = Mock()
        screen.getmaxyx.return_value = (24, 110)
        screen.get_wch.side_effect = [curses.KEY_DOWN, " ", "\n"]
        languages = builder.language_screen(screen, [("en", 10), ("fr", 5)], ("en",))
        self.assertEqual(languages, ("en", "fr"))

    def test_gutenberg_gb_cap_and_languages_reach_output_metadata(self):
        books = [{"id": "1", "languages": ["fr"], "title": "French", "authors": "An author"},
                 {"id": "2", "languages": ["fr"], "title": "French 2", "authors": "Another author"}]
        data = b"*** START OF THE PROJECT GUTENBERG EBOOK ***\nA book\nLicense"
        second = data + b" more text"
        limit = len(builder.record_text("gutenberg", {"text": data.decode()}).encode()) + 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "french"
            with patch.object(builder, "gutenberg_catalog", return_value=(books, {"catalog_sha256": "catalog-hash"})), \
                    patch.object(builder, "fetch_gutenberg", side_effect=[(data, "url-1"), (second, "url-2")]), \
                    patch.object(builder.time, "sleep"):
                summary = builder.prepare({"gutenberg": limit}, root, ("fr",))
            self.assertGreater(summary["bytes"], limit)
            self.assertEqual(summary["files"], 2)
            self.assertEqual(summary["limit_modes"]["gutenberg"], "soft_whole_book")
            self.assertEqual(summary["counts"]["gutenberg"]["stopped_at_byte_limit"], 1)
            self.assertEqual(summary["sources"]["gutenberg"]["languages"], ["fr"])
            rows = [json.loads(line) for line in (root / "metadata/files.jsonl").read_text().splitlines()]
            row = rows[0]
            self.assertEqual(row["languages"], ["fr"])
            self.assertEqual(row["authors"], "An author")
            self.assertEqual((root / rows[1]["path"]).read_bytes(),
                             builder.record_text("gutenberg", {"text": second.decode()}).encode())
        self.assertEqual(builder.default_folder({"gutenberg": 1_000_000_000}, ("en", "fr")),
                         "datasets/gutenberg-1gb_en-fr")

    def test_soft_limit_saves_first_oversized_book_and_does_not_fetch_next(self):
        book = {"id": "1", "text": "Complete book including ending and license notice"}
        records = Mock()
        records.__iter__ = Mock(return_value=iter([book, {"id": "2", "text": "Another book"}]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "whole-book"
            with patch.object(builder, "source_records", return_value=records):
                summary = builder.prepare({"gutenberg": 1}, root)
            self.assertEqual(summary["files"], 1)
            self.assertEqual(summary["counts"]["gutenberg"]["examined"], 1)
            row = json.loads((root / "metadata/files.jsonl").read_text())
            self.assertEqual((root / row["path"]).read_bytes(),
                             builder.record_text("gutenberg", book).encode())
            records.close.assert_called_once()

    def test_top_language_options_are_limited_to_ten_by_count(self):
        books = [{"languages": [f"lang{i}"]} for i in range(12)] + [{"languages": ["en"]}] * 20
        with patch.object(builder, "gutenberg_catalog", return_value=(books, {})):
            choices = builder.gutenberg_languages()
        self.assertEqual(len(choices), 10)
        self.assertEqual(choices[0], ("en", 20))

    def test_mixed_download_full_exhaustion_hashes_and_training_files(self):
        sources = {"wikipedia": [{"id": "1", "url": "https://example.org/wiki", "text": PROSE},
                                 {"id": "2", "text": PROSE}],
                   "dolly": [{"instruction": "Question?", "response": "Yes", "category": "open_qa"}],
                   "fineweb-edu": [{"text": PROSE}, {"text": PROSE.replace("Scientists", "Researchers")}],
                   "gutenberg": [{"id": "84", "rank": 1, "text": "Book\n\nLicense notice"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mixed"
            with patch.object(builder, "source_records", side_effect=lambda name, meta, *args: iter(sources[name])):
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
            with patch("builtins.input", side_effect=["5", "2,4", "nan", "0.5", "full", str(root), "y"]), \
                    patch.object(builder, "plain_languages", return_value=("en", "fr")):
                selected, actual_root, languages = builder.plain_plan()
        self.assertEqual(selected, {"dolly": 500_000_000, "gutenberg": None})
        self.assertEqual(actual_root, root)
        self.assertEqual(languages, ("en", "fr"))

    def test_tui_multiselection_limit_edit_and_folder(self):
        import curses
        screen = Mock()
        screen.getmaxyx.return_value = (24, 110)
        # Start with all four, uncheck Wikipedia, edit Dolly, keep remaining sources.
        screen.get_wch.side_effect = [" ", curses.KEY_DOWN, "e", "\x15", "0", ".", "5", "\n",
                                     "o", "/", "t", "m", "p", "/", "t", "u", "i", "-", "x", "\n", "\n", "\n"]
        with patch.object(curses, "wrapper", side_effect=lambda f: f(screen)), \
                patch.object(curses, "curs_set"), patch.object(builder, "validate_output", side_effect=Path):
            selected, root, languages = builder.tui_plan()
        self.assertEqual(selected, {"dolly": 500_000_000, "fineweb-edu": 1_000_000_000, "gutenberg": 250_000_000})
        self.assertEqual(root, Path("/tmp/tui-x"))
        self.assertEqual(languages, ("en",))


if __name__ == "__main__":
    unittest.main()
