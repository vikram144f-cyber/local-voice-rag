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
    rag_core.replace_file_and_rebuild = lambda *_args: 1
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
        from fastapi.testclient import TestClient  # noqa: PLC0415

        cls.UploadFile = UploadFile
        cls.TestClient = TestClient

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

    def test_http_upload_route_accepts_valid_pdf_and_rejects_invalid_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.UPLOADS_DIR = directory
            with self.TestClient(self.main.app) as client:
                with patch.object(self.main, "process_upload", return_value=2):
                    valid = client.post(
                        "/api/upload",
                        files={"file": ("notes.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
                    )
                invalid = client.post(
                    "/api/upload",
                    files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
                )

            self.assertEqual(valid.status_code, 200)
            self.assertEqual(valid.json()["chunks"], 2)
            self.assertEqual(invalid.status_code, 400)

    def test_http_chat_route_validates_top_k_and_question(self):
        with self.TestClient(self.main.app) as client:
            zero = client.post("/api/chat", json={"query": "hello", "top_k": 0})
            excessive = client.post("/api/chat", json={"query": "hello", "top_k": 11})
            empty = client.post("/api/chat", json={"query": "   "})
            valid = client.post("/api/chat", json={"query": "hello", "top_k": 3})

        self.assertEqual(zero.status_code, 422)
        self.assertEqual(excessive.status_code, 422)
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(valid.status_code, 200)
        self.assertIn("__STATUS__generating__", valid.text)

    def test_http_cors_allows_frontend_loopback_origin(self):
        with self.TestClient(self.main.app) as client:
            response = client.options(
                "/health",
                headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://127.0.0.1:5173",
        )

    def test_http_upload_route_sanitizes_unexpected_processing_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.UPLOADS_DIR = directory
            with patch.object(
                self.main, "process_upload", side_effect=RuntimeError("secret implementation detail")
            ):
                with self.TestClient(self.main.app) as client:
                    response = client.post(
                        "/api/upload",
                        files={"file": ("broken.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
                    )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "PDF processing failed.")
        self.assertNotIn("secret implementation detail", response.text)

    def test_voice_and_prompt_limits_and_retrieval_failure_are_publicly_bounded(self):
        with patch.object(self.main, "MAX_TRANSCRIPT_CHARS", 5):
            self.assertEqual(self.main._validate_transcript("  hello  "), "hello")
            with self.assertRaises(self.main.HTTPException) as transcript_error:
                self.main._validate_transcript("too long")
            self.assertEqual(transcript_error.exception.status_code, 413)

        with patch.object(self.main, "retrieve_context", side_effect=RuntimeError("secret retrieval")):
            with self.assertRaises(self.main.HTTPException) as retrieval_error:
                self.main._retrieve_prompt("hello", "All Files", 5)
        self.assertEqual(retrieval_error.exception.status_code, 500)
        self.assertEqual(retrieval_error.exception.detail, "Retrieval failed.")
        self.assertNotIn("secret retrieval", str(retrieval_error.exception.detail))

        with patch.object(self.main, "MAX_PROMPT_CHARS", 5), patch.object(
            self.main, "build_prompt", return_value="prompt that is too long"
        ):
            with self.assertRaises(self.main.HTTPException) as prompt_error:
                self.main._retrieve_prompt("hello", "All Files", 5)
        self.assertEqual(prompt_error.exception.status_code, 413)

    def test_http_voice_upload_route_enforces_audio_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.TEMP_AUDIO_DIR = directory
            with patch.object(self.main, "MAX_AUDIO_BYTES", 4), patch.object(
                self.main, "MAX_AUDIO_SIZE_MB", 1
            ), self.TestClient(self.main.app) as client:
                response = client.post(
                    "/api/voice/upload-audio",
                    files={"audio": ("recording.webm", b"12345", "audio/webm")},
                )

            self.assertEqual(response.status_code, 413)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_http_voice_upload_route_preserves_transcript_limit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            self.main.TEMP_AUDIO_DIR = directory
            with patch.object(self.main, "MAX_TRANSCRIPT_CHARS", 4), patch.object(
                self.main, "transcribe_audio", return_value="too long"
            ), self.TestClient(self.main.app) as client:
                response = client.post(
                    "/api/voice/upload-audio",
                    files={"audio": ("recording.webm", b"audio", "audio/webm")},
                )

            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["detail"], "Transcript too long. Maximum length is 4 characters.")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_http_voice_transcribe_route_enforces_transcript_limit(self):
        with patch.object(self.main, "MAX_TRANSCRIPT_CHARS", 4), patch.object(
            self.main, "listen_and_transcribe", return_value="too long"
        ), self.TestClient(self.main.app) as client:
            response = client.post("/api/voice/transcribe", params={"duration": 5})

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "Transcript too long. Maximum length is 4 characters.")


if __name__ == "__main__":
    unittest.main()
