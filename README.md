# Cultural XR: Tiwa RAG Assistant

A small, text-only RAG assistant that answers questions about Tiwa culture (artifacts, materials, clothing,
ornaments, instruments, festivals, religion, society) using **one source**: *Material Culture of the Tiwas of
Jagiroad, Assam: An Analysis* (Dewri & Saikia, 2022, IJCSPUB 12(4), pp. 283-300).

Answers cite the section and journal page. If the paper doesn't cover a question, the assistant says so
instead of guessing. Gemini does both the embeddings and the answers, so one API key is all you need.

```
PDF --build_corpus.py--> data/tiwa_chunks.jsonl --rag.py--> embed + search --> Gemini answers with [S#] citations
```

## Files

| File | Purpose |
|---|---|
| `build_corpus.py` | PDF to clean, section-aware chunks (typo fixes logged, page refs kept) |
| `rag.py` | The whole RAG pipeline and CLI: embed, retrieve, gate, generate, cite |
| `evaluate.py` | Checks retrieval and refusal on 24 questions and suggests a score threshold |
| `data/tiwa_chunks.jsonl` | The 69 chunks (67 indexed; the reference list is kept but not indexed) |
| `data/cleaning_report.json` | Every correction made, and every artifact name taken from a figure caption |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env              # add your GEMINI_API_KEY

python rag.py ask "What is a Chu Law used for?"
python rag.py chat                # interactive
python evaluate.py                # how well does it work?
python evaluate.py --full         # also prints Gemini's answers to read
```

The index is built on first run and cached in `./index`, so there is no separate ingest step.
Other options: `--no-llm` (show retrieved passages only), `--scores`, `--json`, `--embedder tfidf` (offline,
keyword-only, for testing without an API key).

To regenerate the chunks yourself: put the PDF in `data/raw/` and run `python build_corpus.py data/raw/<file>.pdf data`
(needs `pdftotext` from poppler).

## How it avoids making things up

1. **Score gate**: if even the best chunk is below `RAG_MIN_SCORE`, it refuses without calling Gemini.
2. **Grounded prompt**: Gemini sees only the retrieved chunks, must cite `[S#]`, and must answer
   `NOT_IN_DOCUMENT` if they don't help. Partial answers must say what is missing.
3. **Checked citations**: the source list shows only chunks Gemini actually cited; invented labels are dropped.
4. **Gaps are flagged, not filled**: five artifacts have no name in the paper's text, so their names come from the
   paper's own figure captions and are marked in `data_notes`. One measurement (Ramatha) is cut off in the paper.

## Calibrating

The default thresholds are untuned guesses. Run `python evaluate.py`: it reports the score ranges for answerable
and unanswerable questions and suggests a `RAG_MIN_SCORE`. With only 24 questions, add your own to `QUESTIONS`
in `evaluate.py`. Model names are overridable (`GEMINI_MODEL`, default `gemini-flash-latest`).

## Limits

- One 18-page paper: many reasonable questions will correctly get "not covered".
- The paper's typos and spelling variants (Sorailou/Saroilou, Gaogaoni/Gaogoni) are kept as written, apart from
  17 logged typo fixes.
- Text only: the paper's photos are not used; figure captions are indexed as text.
- Retrieval is a plain in-memory vector search, which is plenty for ~70 chunks. For a larger corpus, swap in
  ChromaDB or FAISS inside `Index`.

## Licence note

The chunk file contains the paper's text and the PDF is the journal's, so both are git-ignored. Check the
journal's licence before publishing either.
