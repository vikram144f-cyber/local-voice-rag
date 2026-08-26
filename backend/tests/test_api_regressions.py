"""Regression tests for the FastAPI boundary with ML modules replaced by stubs."""

import importlib
import io
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def load_main_with_lightweight_stubs():
    """Import the real API module without importing ML/model dependencies."""

    for name in ("main", "rag_core", "local_llm", "voice_input"):
        sys.modules.pop(name, None)

    rag_core = types.ModuleType("rag_core")
    rag_core.UPLOADS_DIR = tempfile.gettempdir()
    rag_core.process_upload = lambda *_args: 1
    rag_core.delete_file_and_rebuild = lambda *_args: None
    rag_core.rebuild_all_files = lambda: None
    rag_core.load_registry = lambda: []
    rag_core.load_vectorstore = lambda: None
    rag_core.retrieve_context = lambda *_args: []
    rag_core.build_prompt = lambda *_args: "prompt"
    rag_core.get_embeddings = lambda: None

    local_llm = types.ModuleType("local_llm")
    local_llm.stream_response = lambda _prompt: iter(["ok"])
    local_llm.load_llm = lambda: None

    voice_input = types.ModuleType("voice_input")
    voice_input.TEMP_AUDIO_DIR = tempfile.gettempdir()
    voice_input.transcribe_audio = lambda _path: "query"
    voice_input.listen_and_transcribe = lambda duration: f"query-{duration}"

    with patch.dict(
        sys.modules,
        {"rag_core": rag_core, "local_llm": local_llm, "voice_input": voice_input},
    ):
        return importlib.import_module("main")


class ApiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_with_lightweight_stubs()
        from fastapi import UploadFile  # noqa: PLC0415

        cls.UploadFile = UploadFile

    def make_upload(self, filename, content):
        return self.UploadFile(file=io.BytesIO(content), filename=filename)

    def test_valid_small_pdf_upload_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.UPLOADS_DIR = directory
            with patch.object(self.main, "process_upload", return_value=3) as process:
                response = self.main.upload_file(
                    self.make_upload("notes.pdf", b"%PDF-1.7\nsmall document")
                )

            self.assertEqual(response["chunks"], 3)
            self.assertEqual(Path(directory, "notes.pdf").read_bytes(), b"%PDF-1.7\nsmall document")
            process.assert_called_once()
            staged_path = process.call_args.args[0]
            self.assertTrue(Path(staged_path).parent == Path(directory))

    def test_pdf_extension_with_non_pdf_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.UPLOADS_DIR = directory
            with patch.object(self.main, "process_upload") as process:
                with self.assertRaises(self.main.HTTPException) as raised:
                    self.main.upload_file(self.make_upload("fake.pdf", b"not a pdf"))

            self.assertEqual(raised.exception.status_code, 400)
            process.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_oversized_pdf_is_rejected_during_bounded_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.UPLOADS_DIR = directory
            with patch.object(self.main, "MAX_UPLOAD_BYTES", 8), patch.object(
                self.main, "MAX_UPLOAD_SIZE_MB", 1
            ):
                with self.assertRaises(self.main.HTTPException) as raised:
                    self.main.upload_file(
                        self.make_upload("large.pdf", b"%PDF-1.7\nmore than eight bytes")
                    )

            self.assertEqual(raised.exception.status_code, 413)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_path_like_filename_cannot_escape_upload_directory(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            self.main.UPLOADS_DIR = directory
            malicious_name = str(Path(outside, "escaped.pdf"))
            with self.assertRaises(self.main.HTTPException) as raised:
                self.main.upload_file(
                    self.make_upload(malicious_name, b"%PDF-1.7\ncontent")
                )

            self.assertEqual(raised.exception.status_code, 400)
            self.assertFalse(Path(outside, "escaped.pdf").exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_indexing_failure_cleans_staged_upload_and_hides_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.UPLOADS_DIR = directory
            with patch.object(
                self.main, "process_upload", side_effect=RuntimeError("secret stack path")
            ):
                with self.assertRaises(self.main.HTTPException) as raised:
                    self.main.upload_file(
                        self.make_upload("broken.pdf", b"%PDF-1.7\ncontent")
                    )

            self.assertEqual(raised.exception.status_code, 500)
            self.assertEqual(raised.exception.detail, "PDF processing failed.")
            self.assertNotIn("secret stack path", str(raised.exception.detail))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_chat_request_top_k_bounds_and_question_validation(self):
        request = self.main.ChatRequest(query="  hello  ", top_k=3)
        self.assertEqual(request.query, "hello")
        self.assertEqual(request.top_k, 3)

        for invalid_top_k in (0, 11):
            with self.assertRaises(ValueError):
                self.main.ChatRequest(query="hello", top_k=invalid_top_k)
        for invalid_query in ("", "   "):
            with self.assertRaises(ValueError):
                self.main.ChatRequest(query=invalid_query)

    def test_streaming_exception_is_sanitized(self):
        with patch.object(
            self.main, "stream_response", side_effect=RuntimeError("secret model path")
        ):
            output = "".join(self.main._stream_answer("prompt"))

        self.assertIn("The local language model could not generate a response.", output)
        self.assertNotIn("secret model path", output)

    def test_faiss_loader_has_no_request_supplied_path(self):
        self.assertEqual(inspect.signature(self.main.load_vectorstore).parameters, {})
        self.assertNotIn("vectorstore", self.main.ChatRequest.model_fields)


if __name__ == "__main__":
    unittest.main()
