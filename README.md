# DocuMind AI — NLP Service

Python FastAPI sidecar for the Smart NLP Platform (KOI). Everything the Node backend cannot do
natively — parsing, summarisation, RAG, entities, exports and evaluation — lives here.

The Express backend calls this service over HTTP through `documind-backend/src/services/nlpClient.js`.
It is not exposed to the browser.

## Requirements

- Python 3.11+ (built and tested on 3.14.7)
- A Google Gemini API key

## Setup

Create the virtual environment and install dependencies:

```bash
python -m venv .venv
```

**Activating depends on your shell** — this is the single most common thing to get wrong on Windows.

Git Bash / MINGW64 (forward slashes, and `source`):

```bash
source .venv/Scripts/activate
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```
.venv\Scripts\activate.bat
```

Pasting a Windows-style path (`c:\...\activate`) into Git Bash will fail — bash treats `\` as an
escape character and collapses the path into `c:UsersferdoDocuMind-AI...`.

With the environment active:

```bash
pip install -r requirements.txt
```

```bash
python -m spacy download en_core_web_sm
```

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Then copy the environment file and add your key:

```bash
cp .env.example .env
```

`GEMINI_API_KEY` comes from <https://aistudio.google.com/apikey>. `EMBEDDING_MODEL` runs locally
via sentence-transformers, so embeddings do not consume Gemini quota.

## Running

```bash
python -m uvicorn src.api.main:app --port 8000 --reload
```

Without activating the venv first, call its interpreter directly:

```bash
.venv/Scripts/python.exe -m uvicorn src.api.main:app --port 8000
```

Check it is alive:

```bash
curl http://127.0.0.1:8000/health
```

Interactive API docs are at <http://127.0.0.1:8000/docs>.

**Start this service before the Node backend.** Uploading a document with the sidecar down returns
`503 NLP_SERVICE_UNAVAILABLE` from Express.

## API

All responses are `{ "success": true, "data": ... }`.

| Method | Path | Purpose |
| :--- | :--- | :--- |
| GET | `/health` | liveness |
| POST | `/api/v1/parse` | multipart `file` → extracted text, page count, word count |
| POST | `/api/v1/summarize` | extractive (LexRank) **and** abstractive (Gemini) summaries |
| POST | `/api/v1/entities` | spaCy named entities + KeyBERT keyphrases |
| POST | `/api/v1/analytics` | word count, reading time, readability |
| POST | `/api/v1/rag/index` | multipart `file` + `doc_id` → chunk and embed into FAISS |
| POST | `/api/v1/rag/query` | grounded answer with page citations |
| POST | `/api/v1/export/docx` | formatted DOCX report |
| POST | `/api/v1/export/pdf` | formatted PDF report |
| POST | `/api/v1/evaluate` | ROUGE-1/2/L and BLEU against a reference summary |

Two response-shape gotchas the Node client has to respect:

- `/parse` returns **`cleaned_text`**, `raw_text`, `page_count`, `total_words`, `total_characters` —
  not `text` or `word_count`.
- The export endpoints expect the summaries as **objects** (`{"summary": "...", ...}`), not plain
  strings, or FastAPI answers `422`.

## Notes

- **The FAISS index is in-memory and per-process.** Restarting the service loses every indexed
  document, so a question asked after a restart fails until the document is re-indexed. Moving to
  a persisted store (Chroma, or pgvector in Supabase) is the fix when that starts to matter.
- `GEMINI_MODEL` must be a currently served model. `gemini-2.5-flash` was retired and returns
  `404 NOT_FOUND`, which surfaces as `"Failed to generate abstractive summary"` while the
  extractive summary still succeeds — an easy failure to miss. Current value is `gemini-3.6-flash`.
- `src/app_streamlit.py` is a standalone Streamlit UI, kept as the fallback frontend from the
  original project plan. It is not used by the React app.
