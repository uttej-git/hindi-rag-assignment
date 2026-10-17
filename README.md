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
├── eval_rerank.py
├── eval_retriever.py
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


## Notes and limitations

- **PDF extraction:** The pipeline expects the supplied PDF to contain extractable text. Scanned PDFs may require OCR.
- **Hindi text processing:** Unicode NFC normalization and Hindi-aware sentence splitting are used. Some text extracted from the PDF may contain spacing or character artifacts.
- **Section detection:** Section headings are detected using a best-effort approach. Some chunks may have incomplete or undetected section names.
- **Table extraction:** Timeline and awards tables are represented in the supplied document. PDF table extraction and retrieval accuracy may require further improvement.
- **Retrieval evaluation:** On six test queries, the baseline retrieval achieved Hit@1 of 66.7% and Hit@5 of 100%. Query expansion with Reciprocal Rank Fusion achieved Hit@1 of 83.3% and Hit@5 of 100%. These preliminary results are based on a small evaluation set.
- **Answer generation:** Answers are generated using Gemini and grounded in retrieved document passages. API quota limits may prevent all six answers from being generated in a single run.
- **Database and outputs:** The `chroma_db/` directory is excluded from version control and can be recreated by running ingestion. The `outputs/` directory contains generated JSON results and summaries included in this repository; rerunning the pipeline may update them.
- **API key security:** Store `GEMINI_API_KEY` in a local `.env` file. Never commit API keys or other secrets to the repository. Ensure `.env` is excluded by `.gitignore`.
- **Future improvements:** Improve retrieval ranking for difficult questions, validate extracted tables, expand the evaluation dataset, and handle API quota errors gracefully.
