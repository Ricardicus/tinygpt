"""Dataset builder checks; these run without network access or PyTorch."""
from pathlib import Path
import json
import runpy
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from other import prepare_example_dataset as builder


class DatasetPreparationTests(unittest.TestCase):
    def test_resume_rebuilds_partial_corpus_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "corpus/code/old.h"
            partial.parent.mkdir(parents=True)
            partial.write_text("partial output")
            source = root / "sources/repos/cache.txt"
            source.parent.mkdir(parents=True)
            source.write_text("cached source")
            notices = root / "metadata/licenses/NOTICE"
            notices.parent.mkdir(parents=True)
            notices.write_text("notice")
            books = root / "local-books"
            books.mkdir()
            (books / "book.txt").write_text("book")
            with patch.object(builder, "fetch_repositories") as fetch, \
                 patch.object(builder, "books") as prepare_books, \
                 patch.object(builder, "code", return_value=[]), \
                 patch.object(builder, "write_helpers"), patch.object(builder.subprocess, "run"):
                builder.main(["--output", directory, "--resume", "--books-dir", str(books)])
            fetch.assert_called_once_with((root / "sources/repos").resolve(), set())
            prepare_books.assert_called_once()
            self.assertEqual(source.read_text(), "cached source")
            backup, = (root / "metadata").glob("interrupted-build-*")
            self.assertEqual((backup / "corpus/code/old.h").read_text(), "partial output")
            self.assertEqual((backup / "licenses/NOTICE").read_text(), "notice")
            self.assertFalse(partial.exists())
            self.assertTrue((root / "metadata/summary.json").exists())

    def test_resume_rejects_completed_dataset_without_moving_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "metadata/summary.json").write_text('{}')
            (root / "corpus").mkdir()
            with patch.object(builder, "fetch_repositories") as fetch:
                with self.assertRaises(SystemExit):
                    builder.main(["--output", directory, "--resume"])
            fetch.assert_not_called()
            self.assertTrue((root / "corpus").exists())
            self.assertFalse(list((root / "metadata").glob("interrupted-build-*")))

    def test_refresh_helpers_does_not_fetch_or_rebuild_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            summary = {"files": 1, "bytes": 100, "books": {"files": 1}, "repositories": 0}
            (root / "metadata/summary.json").write_text(json.dumps(summary))
            with patch.object(builder, "ROOT", root), patch.object(builder, "write_helpers") as helpers, patch.object(builder, "fetch_repositories") as fetch:
                builder.main(["--output", directory, "--refresh-helpers"])
                helpers.assert_called_once_with(summary)
                fetch.assert_not_called()

    def test_tokenizer_helper_rebuilds_oversized_alphabet_without_losing_characters(self):
        from bpe import BPE

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "artifacts").mkdir()
            sample = root / "sample.txt"
            sample.write_text("hello hello hello world world")
            alphabet = [chr(0x4e00 + i) for i in range(5002)]
            (root / "metadata/alphabet.json").write_text(json.dumps(alphabet))
            (root / "metadata/files.jsonl").write_text(json.dumps({"path": "sample.txt"}) + "\n")
            output = root / "artifacts/tokenizer.json"
            output.write_text(json.dumps({"vocab_size": 4096, "vocab": {char: i for i, char in enumerate(alphabet)}, "merges": []}))
            helper = root / "prepare_tokenizer.py"
            helper.write_text(builder.SUPPLEMENTAL_FILES["prepare_tokenizer.py"])
            namespace = runpy.run_path(str(helper))
            namespace["main"]()
            tokenizer = BPE()
            tokenizer.load(str(output))
            self.assertEqual(tokenizer.vocab_size, 8192)
            self.assertTrue(tokenizer.merges)
            self.assertLess(max(tokenizer.vocab.values()), tokenizer.vocab_size)
            text = "hello " + "".join(alphabet)
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            saved = output.read_bytes()
            namespace["main"]()
            self.assertEqual(output.read_bytes(), saved)

    def test_prompt_protects_existing_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep.txt"
            marker.write_text("keep")
            with patch("builtins.input", return_value=directory) as prompt:
                with self.assertRaises(SystemExit):
                    builder.main([])
            self.assertIn("Dataset folder", prompt.call_args.args[0])
            self.assertEqual(marker.read_text(), "keep")

    def test_book_duplicates_and_gutenberg_notices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books = root / "originals"
            books.mkdir()
            body = " ".join(f"chapter{i} word{i} example{i}" for i in range(200))
            original = "*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n" + body + "\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nLicense notice"
            (books / "01_Example_1.txt").write_text(original)
            (books / "02_Example_2.txt").write_text(original)
            (books / "another-copy.txt").write_text(body.upper())
            (books / "trump.txt").write_text("non-book material")
            records, omitted = [], []
            with patch.object(builder, "ROOT", root / "output"), patch.object(builder, "BOOKS_DIR", books):
                builder.books(records, omitted)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(omitted), 3)
            prepared = (root / "output" / records[0]["path"]).read_text()
            self.assertNotIn("PROJECT GUTENBERG", prepared)
            self.assertIn("chapter199", prepared)
            self.assertTrue(list((root / "output/metadata/book-notices").glob("*.txt")))

    def test_code_filters_and_whitespace_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clones/test--rust"
            source.mkdir(parents=True)
            samples = {
                "a.rs": b"fn main() {\n    println!(\"Hi\");\n}\n",
                "copy.rs": b"fn main() {  \r\n    println!(\"Hi\"); \r\n}\r\n",
                "generated.rs": b"// Automatically generated; do not edit\nfn generated() {}",
                "bad.rs": b"\xff\xfe",
                "other.cpp": b"int main() {}",
                "tests/ui/error.rs": b"deliberately invalid Rust",
            }
            for name, data in samples.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            answers = ["a" * 40 + "\n"]
            records, omitted = [], []
            with patch.object(builder, "ROOT", root / "output"), patch.object(builder, "REPOS", [("test/rust", "Rust")]), patch.object(builder.subprocess, "check_output", side_effect=answers), patch.object(builder, "committed_files", return_value=((name, len(data), data, False) for name, data in samples.items())):
                repos = builder.code(root / "clones", records, omitted)
            self.assertEqual(len(records), 1)
            self.assertEqual(repos[0]["counts"]["duplicate"], 1)
            prepared = (root / "output" / records[0]["path"]).read_text()
            self.assertIn('    println!("Hi");', prepared)
            self.assertTrue(prepared.endswith("\n\n"))
            self.assertEqual(len(omitted), 5)

    def test_committed_blobs_ignore_dirty_checkout_and_preserve_case_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clones/test--c"
            source.mkdir(parents=True)
            def git(*args, data=None):
                return subprocess.check_output(["git", "-C", str(source), *args], input=data)
            git("init", "-q")
            samples = {"same.h": b"int lowercase;\n", "SAME.h": b"int uppercase;\n",
                       "LICENSE": b"Example license\n"}
            for name, data in samples.items():
                oid = git("hash-object", "-w", "--stdin", data=data).decode().strip()
                git("update-index", "--add", "--cacheinfo", f"100644,{oid},{name}")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
            (source / "same.h").write_text("local edit must not enter corpus")
            before = git("status", "--porcelain")
            records, omitted = [], []
            with patch.object(builder, "ROOT", root / "output"), patch.object(builder, "REPOS", [("test/c", "C")]):
                repos = builder.code(root / "clones", records, omitted)
            self.assertEqual(len(records), 2)
            self.assertEqual(len({r["path"].casefold() for r in records}), 2)
            for record in records:
                self.assertEqual(record["source_sha256"], builder.sha(samples[record["source"]]))
                self.assertNotIn(b"local edit", (root / "output" / record["path"]).read_bytes())
            self.assertEqual(git("status", "--porcelain"), before)
            license_path = root / "output" / repos[0]["licenses"][0]
            self.assertEqual(license_path.read_bytes(), samples["LICENSE"])

    def test_html_download_is_rejected(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"<html>not a book</html>"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(builder, "ROOT", root), patch.object(builder, "BOOKS_DIR", root / "books"), patch.object(builder, "BOOK_DOWNLOADS", [("example.txt", 11)]), patch.object(builder, "urlopen", return_value=Response()), patch.object(builder.time, "sleep"):
                with self.assertRaisesRegex(ValueError, "Missing Gutenberg text markers"):
                    builder.download_books()
            self.assertFalse((root / "books/example.txt").exists())

    def test_book_download_uses_https_text_and_resumes_from_cache(self):
        from io import BytesIO

        data = b"*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nBook text\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            with patch.object(builder, "ROOT", root), patch.object(builder, "BOOKS_DIR", root / "books"), patch.object(builder, "BOOK_DOWNLOADS", [("example.txt", 18362)]), patch.object(builder, "urlopen", return_value=BytesIO(data)) as download:
                builder.download_books()
                self.assertEqual(download.call_args.args[0].full_url, "https://www.gutenberg.org/cache/epub/18362/pg18362.txt")
                builder.download_books()
                download.assert_called_once()
            self.assertEqual((root / "books/example.txt").read_bytes(), data)
            metadata = json.loads((root / "metadata/book-downloads.json").read_text())
            self.assertEqual(metadata[0]["sha256"], builder.sha(data))


if __name__ == "__main__":
    unittest.main()
