"""Deterministic RAG-core tests with LangChain and FAISS replaced by fakes."""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class Document:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class RetrievalStore:
    def __init__(self, documents=None, error=None):
        self.documents = documents or []
        self.error = error
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        if self.error:
            raise self.error
        self.search_kwargs = search_kwargs
        return self

    def invoke(self, _query):
        return self.documents


def import_rag_core():
    for name in (
        "rag_core",
        "dotenv",
        "langchain_community",
        "langchain_community.document_loaders",
        "langchain_community.vectorstores",
        "langchain_text_splitters",
        "langchain_huggingface",
    ):
        sys.modules.pop(name, None)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None

    document_loaders = types.ModuleType("langchain_community.document_loaders")
    document_loaders.PyPDFLoader = object
    vectorstores = types.ModuleType("langchain_community.vectorstores")
    vectorstores.FAISS = object
    splitters = types.ModuleType("langchain_text_splitters")
    splitters.RecursiveCharacterTextSplitter = object
    huggingface = types.ModuleType("langchain_huggingface")
    huggingface.HuggingFaceEmbeddings = object
    community = types.ModuleType("langchain_community")

    modules = {
        "dotenv": dotenv,
        "langchain_community": community,
        "langchain_community.document_loaders": document_loaders,
        "langchain_community.vectorstores": vectorstores,
        "langchain_text_splitters": splitters,
        "langchain_huggingface": huggingface,
    }
    with tempfile.TemporaryDirectory() as directory:
        os.environ["UPLOADS_DIR"] = str(Path(directory) / "uploads")
        os.environ["VECTORSTORE_DIR"] = str(Path(directory) / "vectorstore")
        os.environ["REGISTRY_FILE"] = str(Path(directory) / "uploaded_files.json")
        with patch.dict(sys.modules, modules):
            return importlib.import_module("rag_core")


class RagCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = import_rag_core()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("rag_core", None)

    def test_prompt_preserves_source_metadata_and_grounding_rules(self):
        documents = [
            Document(
                "The retention period is thirty days.",
                {"source_file": "handbook.pdf", "page": 2},
            )
        ]
        prompt = self.rag.build_prompt("What is the retention period?", documents)

        self.assertIn("[handbook.pdf]", prompt)
        self.assertIn("The retention period is thirty days.", prompt)
        self.assertIn("Do NOT answer questions", prompt)
        self.assertIn("What is the retention period?", prompt)

    def test_empty_context_is_an_explicit_refusal_prompt(self):
        prompt = self.rag.build_prompt("What is not in the documents?", [])

        self.assertIn(
            "I can only answer questions based on the provided documents.", prompt
        )
        self.assertIn("Do NOT answer from general knowledge.", prompt)

    def test_retrieval_filter_is_passed_to_the_vectorstore(self):
        store = RetrievalStore([Document("text", {"source_file": "handbook.pdf"})])
        documents = self.rag.retrieve_context(store, "retention", "handbook.pdf", 3)

        self.assertEqual(len(documents), 1)
        self.assertEqual(store.search_kwargs, {"k": 3, "filter": {"source_file": "handbook.pdf"}})

        self.rag.retrieve_context(store, "retention", "All Files", 5)
        self.assertEqual(store.search_kwargs, {"k": 5})

    def test_missing_store_and_retrieval_failures_return_empty_context(self):
        self.assertEqual(self.rag.retrieve_context(None, "query"), [])
        broken = RetrievalStore(error=RuntimeError("private retrieval detail"))
        self.assertEqual(self.rag.retrieve_context(broken, "query"), [])

    def test_process_upload_rejects_pdf_without_extractable_chunks(self):
        class EmptyLoader:
            def __init__(self, _path):
                pass

            def load(self):
                return [Document("", {"page": 0})]

        class EmptySplitter:
            def __init__(self, **_kwargs):
                pass

            def split_documents(self, _documents):
                return []

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.rag, "PyPDFLoader", EmptyLoader), patch.object(
                self.rag, "RecursiveCharacterTextSplitter", EmptySplitter
            ):
                with self.assertRaises(self.rag.NoExtractableTextError) as raised:
                    self.rag.process_upload(
                        str(Path(directory, "scan.pdf")), "scan.pdf"
                    )

        self.assertIn("no extractable text", str(raised.exception))

    def test_failed_replacement_restores_previous_file_and_registry_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            self.rag.UPLOADS_DIR = directory
            final_path = Path(directory, "handbook.pdf")
            staged_path = Path(directory, ".upload-handbook.pdf")
            final_path.write_bytes(b"old PDF")
            staged_path.write_bytes(b"new PDF")
            calls = []

            def fail_rebuild(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("index build failed")

            with patch.object(self.rag, "load_registry", return_value=["handbook.pdf"]), patch.object(
                self.rag, "rebuild_all_files", side_effect=fail_rebuild
            ):
                with self.assertRaises(RuntimeError):
                    self.rag.replace_file_and_rebuild(str(staged_path), "handbook.pdf")

            self.assertEqual(final_path.read_bytes(), b"old PDF")
            self.assertFalse(staged_path.exists())
            self.assertEqual(calls[0]["registry_override"], ["handbook.pdf"])
            self.assertEqual(calls[1]["registry_override"], ["handbook.pdf"])

    def test_successful_replacement_returns_rebuilt_chunk_count(self):
        with tempfile.TemporaryDirectory() as directory:
            self.rag.UPLOADS_DIR = directory
            final_path = Path(directory, "handbook.pdf")
            staged_path = Path(directory, ".upload-handbook.pdf")
            final_path.write_bytes(b"old PDF")
            staged_path.write_bytes(b"new PDF")

            with patch.object(self.rag, "load_registry", return_value=["handbook.pdf"]), patch.object(
                self.rag,
                "rebuild_all_files",
                return_value={"registry": ["handbook.pdf"], "chunks": 4},
            ), patch.object(self.rag, "save_registry") as save_registry:
                chunks = self.rag.replace_file_and_rebuild(str(staged_path), "handbook.pdf")

            self.assertEqual(chunks, 4)
            self.assertEqual(final_path.read_bytes(), b"new PDF")
            self.assertFalse(staged_path.exists())
            save_registry.assert_called_once_with(["handbook.pdf"])

    def test_successful_replacement_swaps_index_without_old_document_content(self):
        class FakeLoader:
            def __init__(self, path):
                self.path = path

            def load(self):
                return [Document(Path(self.path).read_text(encoding="utf-8"), {"page": 0})]

        class FakeSplitter:
            def __init__(self, **_kwargs):
                pass

            def split_documents(self, documents):
                return documents

        class FakeVectorStore:
            def __init__(self, documents):
                self.documents = documents

            def save_local(self, path):
                Path(path, "index.faiss").write_bytes(b"fake-index")
                Path(path, "sources.txt").write_text(
                    "\n".join(document.page_content for document in self.documents),
                    encoding="utf-8",
                )

        class FakeFaiss:
            @classmethod
            def from_documents(cls, documents, _embeddings):
                return FakeVectorStore(documents)

        with tempfile.TemporaryDirectory() as directory:
            uploads = Path(directory, "uploads")
            vectorstore = Path(directory, "vectorstore")
            uploads.mkdir()
            vectorstore.mkdir()
            registry = Path(directory, "uploaded_files.json")
            final_path = uploads / "handbook.pdf"
            staged_path = uploads / ".upload-handbook.pdf"
            final_path.write_text("old indexed content", encoding="utf-8")
            staged_path.write_text("new indexed content", encoding="utf-8")
            Path(vectorstore, "index.faiss").write_bytes(b"old-index")
            registry.write_text('["handbook.pdf"]', encoding="utf-8")

            self.rag.UPLOADS_DIR = str(uploads)
            self.rag.VECTORSTORE_DIR = str(vectorstore)
            self.rag.REGISTRY_FILE = str(registry)
            self.rag._vectorstore_cache = None
            with patch.object(self.rag, "PyPDFLoader", FakeLoader), patch.object(
                self.rag, "RecursiveCharacterTextSplitter", FakeSplitter
            ), patch.object(self.rag, "FAISS", FakeFaiss), patch.object(
                self.rag, "get_embeddings", return_value=object()
            ):
                chunks = self.rag.replace_file_and_rebuild(str(staged_path), "handbook.pdf")

            self.assertEqual(chunks, 1)
            self.assertEqual(final_path.read_text(encoding="utf-8"), "new indexed content")
            sources = Path(vectorstore, "sources.txt").read_text(encoding="utf-8")
            self.assertIn("new indexed content", sources)
            self.assertNotIn("old indexed content", sources)
            self.assertEqual(json.loads(registry.read_text(encoding="utf-8")), ["handbook.pdf"])


if __name__ == "__main__":
    unittest.main()
