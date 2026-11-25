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

Scanned PDF OCR uses the local Tesseract executable. Install the Python dependencies from
`requirements.txt` as above, then install Tesseract and its English language data on the machine
running the NLP service.

Windows: install a 64-bit build using the [UB Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki),
include the English language data, and add the install directory to `PATH` or set `TESSERACT_CMD`
to the full path of `tesseract.exe`. Set `TESSDATA_PREFIX` to the folder containing the `eng.traineddata`
file if Tesseract cannot find it automatically. In PowerShell, verify with
`tesseract --version` and `tesseract --list-langs`.

Debian/Ubuntu: install with `sudo apt install tesseract-ocr tesseract-ocr-eng`. Verify with the
same two commands. Set `TESSDATA_PREFIX` if the language data is outside the system default path.

The OCR defaults are `OCR_LANGUAGE=eng`, `OCR_MIN_TEXT_CHARS=40`, and `OCR_DPI=200`. Set these in
`.env` when needed. Other language codes can be configured as a `+`-joined Tesseract language list
only after installing matching trained data (for example, `eng+fra`). OCR runs only on image pages
whose native extraction is below the text threshold. Normal text PDFs do not require Tesseract.
OCR output can contain recognition errors; pages processed with OCR are marked in the document
viewer so the text can be checked against the source.

Then copy the environment file and add your key:

```bash
cp .env.example .env
```

`GEMINI_API_KEY` comes from <https://aistudio.google.com/apikey>. `EMBEDDING_MODEL` runs locally
via sentence-transformers, so embeddings do not consume Gemini quota.

## Tests

Install the development requirements into the project virtual environment, then run the pipeline
tests with pytest:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

The pipeline tests exercise parsing, summaries, embeddings/RAG, spaCy entities, analytics and
exports. The spaCy English model and NLTK tokenizer data from the setup steps above must be present;
the first embedding run may also download its model.

OCR parser tests mock the OCR engine so they run without installing Tesseract. The full scanned-PDF
OCR integration path requires the system Tesseract executable and language data described above.

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

- **RAG indexes persist across service restarts.** FAISS files and page-aware chunk metadata are
  stored under `RAG_INDEX_DIR` (default `data/rag_indexes`) and loaded on demand after a restart.
  Set `RAG_INDEX_DIR` to a mounted durable volume in deployments; indexes are local to the
  service filesystem and are not shared between replicas.
- `GEMINI_MODEL` must be a currently served model. `gemini-2.5-flash` was retired and returns
  `404 NOT_FOUND`, which surfaces as `"Failed to generate abstractive summary"` while the
  extractive summary still succeeds — an easy failure to miss. Current value is `gemini-3.6-flash`.
- `src/app_streamlit.py` is a standalone Streamlit UI, kept as the fallback frontend from the
  original project plan. It is not used by the React app.
