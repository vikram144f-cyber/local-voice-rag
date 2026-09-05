"""Focused tests for upload-boundary behavior; no ML dependencies required."""

import io
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from request_validation import (  # noqa: E402
    UploadTooLargeError,
    is_pdf_file,
    sanitize_pdf_filename,
    save_stream_with_limit,
)


class RequestValidationTests(unittest.TestCase):
    def test_sanitize_pdf_filename_rejects_paths_and_non_pdfs(self):
        self.assertEqual(sanitize_pdf_filename("notes.pdf"), "notes.pdf")
        with self.assertRaises(ValueError):
            sanitize_pdf_filename("..\\notes.pdf")
        with self.assertRaises(ValueError):
            sanitize_pdf_filename("notes.txt")

    def test_pdf_magic_header_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.pdf"
            invalid = Path(directory) / "renamed.pdf"
            valid.write_bytes(b"%PDF-1.7\ncontent")
            invalid.write_bytes(b"not a pdf")
            self.assertTrue(is_pdf_file(str(valid)))
            self.assertFalse(is_pdf_file(str(invalid)))

    def test_stream_is_bounded_and_partial_file_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / "upload.pdf")
            written = save_stream_with_limit(io.BytesIO(b"12345"), destination, 5)
            self.assertEqual(written, 5)
            self.assertEqual(Path(destination).read_bytes(), b"12345")

            with self.assertRaises(UploadTooLargeError):
                save_stream_with_limit(io.BytesIO(b"123456"), destination, 5)
            self.assertFalse(Path(destination).exists())


if __name__ == "__main__":
    unittest.main()
