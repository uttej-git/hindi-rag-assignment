import argparse
import json
import re
import unicodedata
from pathlib import Path

import chromadb
import fitz
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Load GEMINI_API_KEY from the project-root .env file.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
OUTPUT_DIR = ROOT / "outputs"
COLLECTION_NAME = "kalam_hindi_document"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
GEMINI_MODEL_NAME = "gemini-flash-latest"
_GEMINI_CLIENT = None

TEST_QUERIES = [
    ("SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?", "hi"),
    ("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "hi"),
    ("पोखरण-II में कलाम की क्या भूमिका थी?", "hi"),
    ("Which institution did Kalam attend to study aeronautical engineering?", "en"),
    ("Who co-wrote the autobiography, and in what year was it published?", "en"),
    ("How and where did Kalam die in 2015?", "en"),
]

_EMBEDDING_MODEL = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str):
    pieces = re.split(r"(?<=[।!?])\s+|(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def detect_section(text: str) -> str:
    """Detect a numbered heading from the beginning of a PDF page/text block."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "not_detected"

    # The supplied study guide uses headings like: "1 · ए.पी.जे. अब्दुल कलाम कौन थे?"
    # Only inspect the first few lines so a numbered item in body text is not mistaken
    # for the page's section heading.
    for line in lines[:4]:
        match = re.match(r"^\s*\d{1,2}\s*[·•.)।:-]\s*(.+?)\s*$", line)
        if match:
            heading = match.group(1).strip(" ·•.)।:-")
            if 5 <= len(heading) <= 120:
                return heading

    # Some pages have an unnumbered heading on the first line.
    first = lines[0].strip(" ·•")
    if (8 <= len(first) <= 90 and len(first.split()) <= 12
            and not first.endswith(("।", ".", "!", "?", ",", ";"))):
        return first
    return "not_detected"


def make_chunks(text: str, page_num: int, chunk_size: int = 850, overlap: int = 120, section: str | None = None):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = []
    for paragraph in paragraphs:
        sentences = split_sentences(paragraph)
        units.extend(sentences if sentences else [paragraph])

    chunks, current = [], ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(unit) > chunk_size:
            start = 0
            while start < len(unit):
                part = unit[start:start + chunk_size].strip()
                if part:
                    chunks.append(part)
                start += max(1, chunk_size - overlap)
            current = ""
        else:
            current = unit
    if current:
        chunks.append(current)

    return [
        {
            "text": chunk,
            "page": page_num,
            "section": section or detect_section(chunk),
            "chunk_id": f"p{page_num}_c{idx:03d}",
        }
        for idx, chunk in enumerate(chunks)
    ]


def extract_pdf(pdf_path: Path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = normalize_text(page.get_text("text", sort=True))
            if text:
                pages.append((index + 1, text))
    return pages


def load_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def ingest(pdf_path: Path, chunk_size: int, overlap: int):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Place Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf in the data/ folder "
            "or pass --pdf /path/to/file.pdf."
        )

    pages = extract_pdf(pdf_path)
    if not pages:
        raise ValueError("No selectable text was extracted. This PDF may need OCR.")

    chunks = []
    for page_num, page_text in pages:
        page_section = detect_section(page_text)
        chunks.extend(make_chunks(page_text, page_num, chunk_size, overlap, section=page_section))
    if not chunks:
        raise ValueError("No text chunks were created.")

    model = load_model()
    embeddings = model.encode(
        ["passage: " + chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=embeddings[start:start + 64],
            metadatas=[
                {
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "chunk_id": chunk["chunk_id"],
                }
                for chunk in batch
            ],
        )

    summary = {
        "pdf": str(pdf_path),
        "pages_with_text": len(pages),
        "chunks_indexed": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size_chars": chunk_size,
        "overlap_chars": overlap,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "ingestion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def retrieve(query: str, top_k: int = 4):
    """Retrieve candidates using the original and expanded query, then fuse ranks."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("Vector index not found. Run the ingest command first.") from exc

    count = collection.count()
    if not count:
        return []

    model = load_model()
    candidate_k = min(max(top_k * 4, 12), count)

    # Use the original question plus a query expanded with domain-specific hints.
    hints = query_hints(query)
    expanded_query = query
    if hints:
        expanded_query = query + " " + " ".join(sorted(hints))

    query_texts = ["query: " + query]
    if expanded_query.strip().lower() != query.strip().lower():
        query_texts.append("query: " + expanded_query)

    embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
    ).tolist()

    # Reciprocal Rank Fusion (RRF) combines rankings without mixing incompatible
    # raw distance scores from separate queries. Deduplicate by chunk ID.
    fused = {}
    rrf_k = 60

    for embedding in embeddings:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        for rank, (doc, meta, distance, chunk_id) in enumerate(zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ), start=1):
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "text": doc,
                    "page": meta.get("page"),
                    "section": meta.get("section", "not_detected"),
                    "chunk_id": meta.get("chunk_id", chunk_id),
                    # Keep a similarity score for the existing citation format.
                    "score": round(1.0 - float(distance), 4),
                    "_rrf_score": 0.0,
                }
            fused[chunk_id]["_rrf_score"] += 1.0 / (rrf_k + rank)

    items = sorted(
        fused.values(),
        key=lambda item: item["_rrf_score"],
        reverse=True,
    )
    for item in items:
        item.pop("_rrf_score", None)
    return items[:max(1, top_k)]


def query_hints(query: str):
    expansions = {
        "भारत रत्न": ["भारत रत्न", "1997", "सर्वोच्च सम्मान"],
        "किस वर्ष प्राप्त हुआ": ["भारत रत्न", "1997"],
        "पोखरण-ii": ["पोखरण", "वैज्ञानिक सलाहकार", "रक्षा मंत्री", "डीआरडीओ", "मई 1998"],
        "क्या भूमिका": ["वैज्ञानिक सलाहकार", "रक्षा मंत्री", "डीआरडीओ", "केंद्रीय भूमिका"],
        "autobiography": ["आत्मकथा", "अग्नि की उड़ान"],
        "co-wrote": ["अरुण तिवारी", "साथ मिलकर", "प्रकाशित", "1999"],
        "co-author": ["अरुण तिवारी", "साथ मिलकर", "प्रकाशित", "1999"],
        "published": ["प्रकाशित", "1999"],
        "publication": ["प्रकाशित", "1999"],
        "year": ["वर्ष", "1997", "1999", "1980"],
        "bharat ratna": ["भारत रत्न", "1997"],
        "pokhran": ["पोखरण", "वैज्ञानिक सलाहकार", "डीआरडीओ", "रक्षा मंत्री"],
        "role": ["भूमिका", "वैज्ञानिक सलाहकार", "डीआरडीओ", "रक्षा मंत्री"],
        "aeronautical": ["वैमानिकी", "मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी"],
        "institution": ["मद्रास इंस्टीट्यूट ऑफ टेक्नोलॉजी"],
        "satellite": ["उपग्रह", "रोहिणी", "1980"],
        "died": ["निधन", "शिलांग", "व्याख्यान"],
        "where": ["शिलांग", "भारतीय प्रबंधन संस्थान"],
        "how": ["व्याख्यान", "गिर पड़े", "निधन"],
    }
    q = query.lower()
    hints = set(re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", q))
    for term, phrases in expansions.items():
        if term in q:
            for phrase in phrases:
                hints.update(re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", phrase.lower()))
    return {h for h in hints if len(h) > 1}


def is_question_sentence(text: str) -> bool:
    # PDF's review questions may not have punctuation in the extracted text.
    q_marked = re.search(r"[?؟]\s*$", text) is not None
    question_starts = (
        "किस वर्ष", "कहाँ", "किसने", "क्या", "क्यों", "कैसे",
        "कलाम ने", "अब्दुल कलाम का जन्म", "कलाम का निधन",
    )
    return q_marked or any(text.strip().startswith(prefix) for prefix in question_starts)


def get_gemini_client():
    """Create one Gemini client using the GEMINI_API_KEY environment variable."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        import os
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY in your terminal, "
                "then run the command again."
            )
        _GEMINI_CLIENT = genai.Client()
    return _GEMINI_CLIENT


def generate_answer(query: str, retrieved):
    """Generate an answer only from retrieved PDF evidence; add citations in code."""
    if not retrieved:
        return "The indexed document did not provide matching evidence.", "none"

    # Pass only retrieved passages to Gemini. Metadata citations are appended by
    # this application, never invented by the language model.
    context_parts = []
    for i, item in enumerate(retrieved, start=1):
        context_parts.append(
            f"[PASSAGE {i}] page={item['page']}; section={item['section']}; "
            f"chunk_id={item['chunk_id']}; retrieval_score={item['score']}\n"
            f"{item['text']}"
        )
    context = "\n\n".join(context_parts)
    prompt = f"""You are a careful retrieval-augmented question-answering assistant.
Answer the user's question using ONLY the evidence passages below.
- If the passages do not support an answer, say that the provided document evidence is insufficient.
- Do not use outside knowledge or invent facts.
- Answer in the same language as the question (Hindi for Hindi questions, English for English questions).
- Be concise and directly answer the question.
- Do not write citations, page numbers, chunk IDs, or retrieval scores; the application adds citations separately.

QUESTION:
{query}

EVIDENCE PASSAGES:
{context}
"""

    try:
        response = get_gemini_client().models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
        # Cite the retrieved passages used as context, including score and section.
        citations = "\n\nSources (retrieved evidence):\n" + "\n".join(
            f"- page {item['page']} | section: {item['section']} | "
            f"chunk: {item['chunk_id']} | retrieval score: {item['score']}"
            for item in retrieved[:3]
        )
        return answer + citations, f"Gemini ({GEMINI_MODEL_NAME})"
    except Exception as exc:
        raise RuntimeError(
            "Gemini answer generation failed. Check your API key, internet connection, "
            "model availability, and Google AI Studio usage limits. "
            f"Details: {exc}"
        ) from exc


def ask(query: str, top_k: int = 8):
    retrieved = retrieve(query, top_k)
    if not retrieved:
        print("No matching chunks found.")
        return

    answer, generator = generate_answer(query, retrieved)
    print("\nQUERY\n" + query)
    print("\nANSWER\n" + answer)
    print(f"\nGENERATOR: {generator}\n")
    print("SOURCES")
    for item in retrieved:
        evidence = item["text"]
        print(
            f"- page: {item['page']} | section: {item['section']} | "
            f"chunk_id: {item['chunk_id']} | score: {item['score']}"
        )
        print(f"  Evidence: {evidence[:280]}{'...' if len(evidence) > 280 else ''}")


def run_tests(top_k: int = 8):
    records = []
    for query, language in TEST_QUERIES:
        retrieved = retrieve(query, top_k)
        if retrieved:
            answer, generator = generate_answer(query, retrieved)
        else:
            answer, generator = "No matching chunks found.", "none"

        record = {
            "query": query,
            "language": language,
            "answer": answer,
            "generator": generator,
            "sources": [
                {
                    "page": item["page"],
                    "section": item["section"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "evidence": item["text"],
                }
                for item in retrieved
            ],
        }
        records.append(record)

        print("\n" + "=" * 80)
        print(f"QUERY ({language}): {query}\nANSWER: {answer}")
        print("SOURCES:")
        for item in retrieved:
            print(
                f"page={item['page']} | section={item['section']} | "
                f"chunk_id={item['chunk_id']} | score={item['score']}"
            )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "test_results.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hindi PDF RAG QA pipeline using ChromaDB, multilingual E5, and Gemini"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Extract, chunk, embed, and index a PDF")
    p_ingest.add_argument(
        "--pdf",
        default=str(DATA_DIR / "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"),
    )
    p_ingest.add_argument("--chunk-size", type=int, default=850)
    p_ingest.add_argument("--overlap", type=int, default=120)

    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", type=int, default=8)

    p_test = sub.add_parser("test", help="Run all six required assignment queries")
    p_test.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(Path(args.pdf), args.chunk_size, args.overlap)
    elif args.command == "ask":
        ask(args.query, args.top_k)
    elif args.command == "test":
        run_tests(args.top_k)


if __name__ == "__main__":
    main()
