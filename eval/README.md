# Retrieval evaluation

This directory contains a small evaluation harness for source-file retrieval. It reports Recall@1/3/5 and mean reciprocal rank (MRR) from an explicit question set and retrieved source lists. It does not generate or assume scores.

## Question format

Copy `questions.example.json`, replace the placeholder questions and filenames, and keep `expected_sources` tied to the uploaded PDF registry. Mark questions that should be refused with `"answerable": false`; those cases are counted separately and are not included in retrieval recall.

## Evaluate captured results

Create a JSON file with one result per question:

```json
[
  {"id": "q1", "retrieved_sources": ["handbook.pdf", "policy.pdf"]}
]
```

Then run:

```bash
python eval/retrieval_eval.py \
  --questions eval/questions.example.json \
  --results path/to/retrieval-results.json
```

## Query the local FAISS index

After the backend has built `backend/vectorstore/` and its embedding model is available:

```bash
python eval/retrieval_eval.py \
  --questions eval/questions.example.json \
  --vectorstore backend/vectorstore
```

The vectorstore path is checked before querying. First-run model downloads and hardware behavior are described in the root README; no evaluation result is committed by default.
