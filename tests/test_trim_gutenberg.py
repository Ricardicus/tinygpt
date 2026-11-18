import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from other.trim_gutenberg import main, trim_text


class TrimGutenbergTests(unittest.TestCase):
    def test_removes_headers_and_multiple_footers_preserves_body_bytes(self):
        document = (b'Preface\r\n*** START OF THE PROJECT GUTENBERG EBOOK A BOOK ***\r\n'
                    b'Body\r\n\r\nMore text\r\n*** END OF THE PROJECT GUTENBERG EBOOK A BOOK ***\r\n'
                    b'License\r\n*** END OF THE PROJECT GUTENBERG EBOOK A BOOK ***\r\n')
        self.assertEqual(trim_text(document), b'Body\r\n\r\nMore text\r\n')
        self.assertEqual(trim_text(b'*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nHi\n*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n'), b'Hi\n')
        self.assertEqual(trim_text(b'*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\r\n \t\r\n  Indented\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\n'), b'  Indented\r\n')

    def test_recursion_dry_run_and_missing_markers(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            nested = root / 'nested'
            nested.mkdir()
            book = nested / 'book.txt'
            original = b'Header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nFooter'
            book.write_bytes(original)
            missing = root / 'missing.txt'
            missing.write_bytes(b'No markers')
            empty = root / 'empty.txt'
            empty.write_bytes(b'*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n')
            self.assertEqual(main(['--gutenberg-folder', str(root), '--dry-run']), 0)
            self.assertEqual(book.read_bytes(), original)
            self.assertEqual(main(['--gutenberg-folder', str(root)]), 0)
            self.assertEqual(book.read_bytes(), b'Body\n')
            self.assertEqual(missing.read_bytes(), b'No markers')
            self.assertIn(b'START', empty.read_bytes())
            self.assertEqual(main(['--gutenberg-folder', str(root)]), 0)
            self.assertEqual(book.read_bytes(), b'Body\n')
