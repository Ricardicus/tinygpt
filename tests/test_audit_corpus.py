"""Offline checks for corpus character auditing and whole-file loss estimates."""
import argparse
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import itertools

from other import audit_corpus as auditor
from corpus import discover_files, load_config


class CorpusAuditTests(unittest.TestCase):
    def test_export_existing_audit_applies_exclusions_without_rescanning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            corpus = root / "corpus"
            corpus.mkdir(parents=True)
            for name, text in {"a.txt": "English prose", "b.txt": "hello אב", "old.txt": "already excluded"}.items():
                (corpus / name).write_text(text, encoding="utf-8")
            original_config = {"suffixes": [".txt"], "exclude_dirs": [], "exclude_files": ["old.txt"]}
            config_path = root / "training_data.json"
            config_path.write_text(json.dumps(original_config))
            report = root / "report"
            with redirect_stdout(io.StringIO()):
                auditor.audit(root, report)
            # Export after the audit must read CSV/summary only, not corpus text.
            with patch.object(auditor, "scan_file", side_effect=AssertionError("Rescan")), \
                    patch.object(auditor, "discover_files", side_effect=AssertionError("Rediscovery")), \
                    redirect_stdout(io.StringIO()):
                selection = auditor.exclusion_config(report, 1)
            result_config = load_config(selection["config"])
            self.assertEqual(result_config["exclude_files"], ["b.txt", "old.txt"])
            self.assertEqual([p.name for p in discover_files(corpus, result_config)], ["a.txt"])
            self.assertEqual(selection["bytes_removed"], len("hello אב".encode()))
            self.assertEqual(json.loads(config_path.read_text()), original_config)
            self.assertEqual((corpus / "b.txt").read_text(), "hello אב")

    def test_export_cli_new_audit_and_later_threshold_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "corpus").mkdir(parents=True)
            (root / "corpus/a.txt").write_text("hello")
            (root / "corpus/b.txt").write_text("אבגד", encoding="utf-8")
            report = root / "report"
            with redirect_stdout(io.StringIO()):
                auditor.main([str(root), "--output", str(report), "--exclude-above", "1"])
                auditor.main(["--from-report", str(report / "summary.json"), "--exclude-above", "100"])
            limited = load_config(report / "exclusions-gt-1/training_data.json")
            keep_all = load_config(report / "exclusions-gt-100/training_data.json")
            self.assertEqual(len(discover_files(root / "corpus", limited)), 1)
            self.assertEqual(len(discover_files(root / "corpus", keep_all)), 2)
            with self.assertRaisesRegex(ValueError, "new or empty"), redirect_stdout(io.StringIO()):
                auditor.exclusion_config(report, 1)

    def test_generated_config_filters_tokenizer_training_and_validation(self):
        import main as training
        from datareader import TokenizedDataReader
        reader_files = []
        class RecordingReader(TokenizedDataReader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                reader_files.extend(self.files)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            for name in ("one.txt", "two.txt"):
                (corpus / name).write_text("hello world " * 8)
            (corpus / "excluded.txt").write_text("אבגד" * 10, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                auditor.audit(root, root / "report")
                selection = auditor.exclusion_config(root / "report", 1)
                with patch.object(training, "TokenizedDataReader", RecordingReader):
                    training.main(["--rawdata", str(corpus), "--data-config", selection["config"],
                                   "--tokenizer", str(root / "tokenizer.json"), "--model", str(root / "model.pt"),
                                   "--device", "cpu", "--verbose", "false", "--vocab_size", "32",
                                   "--d-model", "12", "--n-heads", "3", "--num-layers", "1",
                                   "--context-length", "4", "--batch-size", "2", "--epochs", "1",
                                   "--validation-fraction", "0.5", "--validation-batches", "1"])
            chosen = {Path(path).name for path in reader_files}
            self.assertEqual(chosen, {"one.txt", "two.txt"})
            tokenizer = json.loads((root / "tokenizer.json").read_text())
            self.assertFalse(any("א" in token for token in tokenizer["vocab"]))
            self.assertTrue((root / "model.pt").exists())

    def test_policy_distinguishes_english_typography_foreign_scripts_and_ascii(self):
        for char in "Hello café—‘quoted’…\n":
            self.assertTrue(auditor.character_info(char, "english")[0], repr(char))
        for char in "אב漢😀\x00\u200b":
            self.assertFalse(auditor.character_info(char, "english")[0], repr(char))
        for char in "é—א":
            self.assertFalse(auditor.character_info(char, "ascii")[0])
        self.assertEqual(auditor.character_info("א", "english")[1], "non_latin_letter")

    def test_per_file_percentages_bytes_and_locations_across_small_read_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.txt"
            text = "Hello\nabcאב\n"
            path.write_text(text, encoding="utf-8")
            record, characters = auditor.scan_file(path, Path("books/book.txt"), "english", 3, block_size=4)
        self.assertEqual(record["characters"], len(text))
        self.assertEqual(record["bytes"], len(text.encode()))
        self.assertEqual(record["outside_policy"], 2)
        self.assertEqual(record["outside_policy_bytes"], 4)
        self.assertAlmostEqual(record["outside_policy_percent"], 200 / len(text))
        self.assertEqual(record["non_latin_letters"], 2)
        self.assertTrue(all(example["line"] == 2 for example in record["examples"]))
        self.assertEqual(record["examples"][0]["character_offset"], text.index("א"))
        self.assertEqual(characters["א"], 1)

    def test_whole_file_removal_estimates_source_totals_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            data = {"wikipedia/a.txt": "hello\n", "fineweb-edu/b.txt": "abcאב\n", "gutenberg/c.txt": "אבגד"}
            for relative, text in data.items():
                path = root / "corpus" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            before = {path: path.read_bytes() for path in (root / "corpus").rglob("*.txt")}
            output = root / "reports"
            with redirect_stdout(io.StringIO()), patch.object(auditor.time, "monotonic", side_effect=itertools.count(step=3).__next__):
                summary = auditor.audit(root, output, thresholds=(0, 40, 100))
            self.assertEqual(summary["totals"]["bytes"], 22)
            self.assertEqual(summary["totals"]["flagged_file_bytes"], 16)
            self.assertEqual(summary["removal_scenarios"]["0"]["bytes_removed"], 16)
            self.assertEqual(summary["removal_scenarios"]["40"]["bytes_removed"], 8)
            self.assertEqual(summary["removal_scenarios"]["100"]["bytes_removed"], 0)
            self.assertEqual(summary["removal_scenarios"]["40"]["sources"]["gutenberg"]["source_percent_removed"], 100)
            with (output / "files.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            flagged = [json.loads(line) for line in (output / "flagged-files.jsonl").read_text().splitlines()]
            self.assertEqual(len(flagged), 2)
            self.assertEqual(json.loads((output / "summary.json").read_text())["totals"]["bytes"], 22)
            with (output / "characters.csv").open() as source:
                chars = {row["codepoint"]: row for row in csv.DictReader(source)}
            self.assertEqual(chars["U+05D0"]["name"], "HEBREW LETTER ALEF")
            self.assertEqual(chars["U+05D0"]["files"], "2")
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_invalid_utf8_and_control_bytes_are_reported_without_losing_byte_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            data = b"hello\x00\xffworld"
            path.write_bytes(data)
            record, _ = auditor.scan_file(path, Path("bad.txt"), "ascii", 3)
        self.assertEqual(record["bytes"], len(data))
        self.assertEqual(record["outside_policy"], 2)
        self.assertEqual(record["reasons"]["invalid_utf8_byte"], 1)
        self.assertEqual(record["reasons"]["control_or_format"], 1)
        json.dumps(record).encode("utf-8")

    def test_threshold_validation(self):
        self.assertEqual(auditor.parse_thresholds("5,0,1,1"), [0, 1, 5])
        for value in ("-1", "101", "nan", "inf", "", "abc"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                auditor.parse_thresholds(value)

    def test_cli_ascii_policy_and_cost_of_removing_an_entire_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            source = root / "corpus" / "fineweb-edu"
            source.mkdir(parents=True)
            (source / "clean.txt").write_text("A clean English file.")
            accented = "café—"
            (source / "accented.txt").write_text(accented, encoding="utf-8")
            output = root / "ascii-report"
            with redirect_stdout(io.StringIO()):
                auditor.main([str(root), "--policy", "ascii", "--thresholds", "0,5", "--examples", "1", "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            scenario = summary["removal_scenarios"]["0.0"]
            self.assertEqual(scenario["bytes_removed"], len(accented.encode()))
            self.assertEqual(scenario["if_entire_affected_sources_removed"]["bytes_removed"], summary["totals"]["bytes"])
            self.assertGreater(scenario["if_entire_affected_sources_removed"]["bytes_removed"], scenario["bytes_removed"])
            details = json.loads((output / "flagged-files.jsonl").read_text())
            self.assertEqual(len(details["examples"]), 1)
            self.assertEqual(details["outside_policy_percent"], 40)
            english, _ = auditor.scan_file(source / "accented.txt", Path("accented.txt"), "english", 1)
            self.assertEqual(english["outside_policy"], 0)

    def test_reports_never_overwrite_existing_data_or_enter_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "corpus").mkdir()
            (root / "corpus/a.txt").write_text("hello")
            with self.assertRaisesRegex(ValueError, "outside"):
                auditor.audit(root, root / "corpus/reports")
            output = root / "reports"
            output.mkdir()
            (output / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                auditor.audit(root, output)
            self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_accepts_plain_text_folder_and_default_reports_are_outside_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rawdata"
            root.mkdir()
            (root / "book.txt").write_text("English only.")
            with redirect_stdout(io.StringIO()):
                result = auditor.audit(root)
            self.assertEqual(result["totals"]["flagged_files"], 0)
            self.assertTrue(list(Path(directory).glob("audits/*/summary.json")))


if __name__ == "__main__":
    unittest.main()
