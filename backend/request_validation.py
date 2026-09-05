"""Small, dependency-free validation helpers for upload boundaries."""

import os
from typing import BinaryIO


class UploadTooLargeError(ValueError):
    """Raised when a streamed upload exceeds its configured byte limit."""


def sanitize_pdf_filename(filename: str) -> str:
    """Return a safe PDF basename or raise ``ValueError``.

    Upload names are retained for the document registry, but path separators,
    control characters, and non-PDF extensions are rejected at the boundary.
    """

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A filename is required.")
    if "/" in filename or "\\" in filename:
        raise ValueError("Path separators are not allowed in filenames.")

    safe_name = os.path.basename(filename)
    if safe_name != filename or safe_name in {".", ".."}:
        raise ValueError("Invalid filename.")
    if any(ord(character) < 32 for character in safe_name):
        raise ValueError("Control characters are not allowed in filenames.")
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    return safe_name


def is_pdf_file(path: str) -> bool:
    """Check the PDF magic header without trusting the filename extension."""

    try:
        with open(path, "rb") as source:
            return source.read(5) == b"%PDF-"
    except OSError:
        return False


def save_stream_with_limit(
    source: BinaryIO,
    destination: str,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Stream ``source`` to ``destination`` and remove partial files on error."""

    if max_bytes <= 0 or chunk_size <= 0:
        raise ValueError("Upload limits must be positive.")

    written = 0
    try:
        with open(destination, "wb") as target:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    raise TypeError("Upload streams must return bytes.")
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLargeError("Upload exceeds the configured size limit.")
                target.write(chunk)
    except Exception:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        raise

    return written
