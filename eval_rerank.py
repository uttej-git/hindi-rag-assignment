
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "kalam_hindi_document"
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 5

TEST_CASES = [
    {
        "query": "एपीजे अब्दुल कलाम को भारत रत्न कब मिला?",
        "expanded": "डॉ एपीजे अब्दुल कलाम भारत रत्न पुरस्कार सम्मान 1997",
        "page": 16,
    },
    {
        "query": "Who co-wrote Agni Ki Udaan, and when was it published?",
        "expanded": "Agni Ki Udaan Wings of Fire co-author Arun Tiwari published 1999",
        "page": 3,
    },
    {
        "query": "SLV-III ने रोहिणी उपग्रह को कक्षा में कब स्थापित किया?",
        "expanded": "SLV III Rohini satellite placed into orbit July 1980",
        "page": 8,
    },
    {
        "query": "पोखरण-II में कलाम की क्या भूमिका थी?",
        "expanded": "APJ Abdul Kalam role Pokhran II nuclear tests DRDO 1998",
        "page": 10,
    },
    {
        "query": "Which institute did Kalam attend for aeronautical engineering?",
        "expanded": "APJ Abdul Kalam studied aeronautical engineering MIT Madras Institute of Technology",
        "page": 6,
    },
    {
        "query": "When did Kalam collapse while lecturing at IIM Shillong?",
        "expanded": "APJ Abdul Kalam collapsed while lecturing IIM Shillong July 27 2015",
        "page": 15,
    },
]


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    baseline_hits = 0
    fusion_hits = 0
    fusion_hit5 = 0

    print("\nQuery Expansion + Reciprocal-Rank Fusion")
    print("=" * 65)

    for case in TEST_CASES:
        query = case["query"]
        expected_page = case["page"]

        queries = [query, case["expanded"]]
        rankings = []

        for q in queries:
            embedding = model.encode(
                ["query: " + q],
                normalize_embeddings=True
            ).tolist()

            result = collection.query(
                query_embeddings=embedding,
                n_results=10,
                include=["metadatas"]
            )

            pages = [
                metadata.get("page")
                for metadata in result["metadatas"][0]
            ]
            rankings.append(pages)

        baseline_page = rankings[0][0] if rankings[0] else None
        baseline_hits += int(baseline_page == expected_page)

        # Reciprocal Rank Fusion: combine both result lists.
        scores = {}
        for pages in rankings:
            for rank, page in enumerate(pages, start=1):
                scores[page] = scores.get(page, 0.0) + 1.0 / (60 + rank)

        fused_pages = sorted(
            scores,
            key=lambda page: scores[page],
            reverse=True
        )

        top_page = fused_pages[0] if fused_pages else None
        fusion_hits += int(top_page == expected_page)
        fusion_hit5 += int(expected_page in fused_pages[:TOP_K])

        print(f"\nQuery: {query}")
        print(f"Expected page: {expected_page}")
        print(f"Original top page: {baseline_page}")
        print(f"Fusion top page: {top_page}")
        print(f"Fusion top 5: {fused_pages[:TOP_K]}")
        print(f"Fusion Hit@1: {'PASS' if top_page == expected_page else 'MISS'}")
        print(f"Fusion Hit@5: {'PASS' if expected_page in fused_pages[:TOP_K] else 'MISS'}")

    total = len(TEST_CASES)
    print("\n" + "=" * 65)
    print(f"Baseline Hit@1: {baseline_hits}/{total} ({baseline_hits / total:.1%})")
    print(f"Fusion Hit@1: {fusion_hits}/{total} ({fusion_hits / total:.1%})")
    print(f"Fusion Hit@5: {fusion_hit5}/{total} ({fusion_hit5 / total:.1%})")


if __name__ == "__main__":
    main()
