# Hindi PDF RAG with Google Gemini

A command-line Retrieval-Augmented Generation (RAG) pipeline for answering questions from the supplied Hindi PDF about A. P. J. Abdul Kalam. It uses PyMuPDF for PDF text extraction, Unicode NFC normalization, sentence-aware chunking, `intfloat/multilingual-e5-small` embeddings, ChromaDB vector search, and Google's Gemini API for grounded answer generation.

## Architecture

1. Extract selectable PDF text page by page with PyMuPDF.
2. Normalize text to Unicode NFC while preserving Devanagari and Hindi danda (`।`).
3. Chunk text around paragraph and sentence boundaries; long chunks use overlap.
4. Embed passages with multilingual E5 and its `passage:` prefix.
5. Store text and metadata (`page`, `section`, `chunk_id`) in persistent ChromaDB.
6. Retrieve relevant passages for each question with multilingual E5 and a lightweight keyword boost.
7. Send only retrieved evidence to Gemini. The app adds page, section, chunk ID, and retrieval-score citations from retrieved metadata, rather than asking the LLM to invent citations.

## Requirements

- Python 3.10 or 3.11 recommended
- Internet access for initial model download and Gemini API calls
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup on Windows PowerShell

Open a terminal in this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pxip install -r requirements.txt
```

Set the Gemini key for the current terminal session. Replace the placeholder with your own key; do not commit or share it.

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

## Build the vector index

Run this once after installing dependencies:

```powershell
python app.py ingest
```

This creates the local `chroma_db/` index and `outputs/ingestion_summary.json`. The index is intentionally not bundled in the ZIP because it can be recreated from the supplied PDF.

## Ask a question

```powershell
python app.py ask "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"
python app.py ask "Which institution did Kalam attend to study aeronautical engineering?"
```

Answers are generated from retrieved evidence. The output includes source page, detected section (or `not_detected`), chunk ID, and retrieval score. If the evidence is insufficient, the prompt instructs Gemini to say so rather than guess.

## Run the six assignment test queries

```powershell
python app.py test
```

The command prints the answers and retrieved sources and saves `outputs/test_results.json`. Inspect this file before submitting. Gemini API availability and usage limits depend on the Google account and selected model.

## Project structure

```text
hindi_rag_assignment_gemini/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Notes and limitations

- The pipeline expects the supplied PDF to contain extractable text. Scanned PDFs may require OCR.
- Section detection is best-effort; some chunks may report `not_detected`.
- PDF table extraction and retrieval quality should be manually checked against the assignment's six questions.
- `chroma_db/` and generated JSON outputs are local runtime artifacts and are not included in this ZIP. Run ingestion and tests to recreate them.
- Never include your Gemini API key in source code, screenshots, output files, or a public repository.
