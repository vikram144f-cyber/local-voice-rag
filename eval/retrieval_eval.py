#!/usr/bin/env python3
"""Evaluate source-file retrieval without inventing scores.

Use ``--results`` for a checked-in or manually captured retrieval result file,
or ``--vectorstore`` to query the app's local FAISS index. The latter may load
the embedding model and therefore may require the first-run model setup.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def reciprocal_rank(retrieved, expected):
    expected_set = set(expected)
    for index, source in enumerate(retrieved, start=1):
        if source in expected_set:
            return 1.0 / index
    return 0.0


def calculate_metrics(questions, results):
    by_id = {item["id"]: item for item in results}
    answerable = [item for item in questions if item.get("answerable", True)]
    if not answerable:
        return {
            "answerable_questions": 0,
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
            "mrr": None,
        }

    metrics = {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
    for question in answerable:
        result = by_id.get(question["id"], {})
        retrieved = result.get("retrieved_sources", [])
        expected = question.get("expected_sources", [])
        metrics["recall_at_1"] += float(bool(set(retrieved[:1]) & set(expected)))
        metrics["recall_at_3"] += float(bool(set(retrieved[:3]) & set(expected)))
        metrics["recall_at_5"] += float(bool(set(retrieved[:5]) & set(expected)))
        metrics["mrr"] += reciprocal_rank(retrieved, expected)

    count = len(answerable)
    return {
        "answerable_questions": count,
        **{key: round(value / count, 4) for key, value in metrics.items()},
        "unanswerable_questions": len(questions) - count,
    }


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def retrieve_from_vectorstore(questions, vectorstore_path):
    vectorstore_path = Path(vectorstore_path).resolve()
    if not (vectorstore_path / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index found at {vectorstore_path}")

    # rag_core reads configuration at import time. Point it at the explicitly
    # requested index rather than silently evaluating a different directory.
    os.environ["VECTORSTORE_DIR"] = str(vectorstore_path)
    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(backend_dir))
    from rag_core import load_vectorstore, retrieve_context  # pylint: disable=import-outside-toplevel

    vectorstore = load_vectorstore()
    if vectorstore is None:
        raise RuntimeError("The local vector store could not be loaded.")

    results = []
    for question in questions:
        started = time.perf_counter()
        documents = retrieve_context(vectorstore, question["query"], "All Files", 5)
        results.append(
            {
                "id": question["id"],
                "retrieved_sources": [
                    document.metadata.get("source_file", "unknown") for document in documents
                ],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Question-set JSON file")
    parser.add_argument("--results", help="Captured retrieval results JSON file")
    parser.add_argument("--vectorstore", help="Local FAISS directory to query")
    args = parser.parse_args()
    if bool(args.results) == bool(args.vectorstore):
        parser.error("provide exactly one of --results or --vectorstore")

    questions = load_json(args.questions)
    results = load_json(args.results) if args.results else retrieve_from_vectorstore(questions, args.vectorstore)
    report = calculate_metrics(questions, results)
    if args.vectorstore:
        report["source"] = "live local vectorstore query"
    else:
        report["source"] = str(Path(args.results).resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
