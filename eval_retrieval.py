
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "kalam_hindi_document"
MODEL_NAME = "intfloat/multilingual-e5-small"

# Queries and expected source pages from the existing RAG tests.
TEST_CASES = [
    {"query": "SLV-III ने रोहिणी उपग्रह को कक्षा में कब स्थापित किया?", "page": 8},
    {"query": "एपीजे अब्दुल कलाम को भारत रत्न कब मिला?", "page": 16},
    {"query": "पोखरण-II में कलाम की क्या भूमिका थी?", "page": 10},
    {"query": "Which institute did Kalam attend for aeronautical engineering?", "page": 6},
    {"query": "Who co-wrote Agni Ki Udaan, and when was it published?", "page": 3},
    {"query": "When did Kalam collapse while lecturing at IIM Shillong?", "page": 15},
]


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    hit_at_1 = 0
    hit_at_5 = 0

    print("\nRetrieval Evaluation")
    print("=" * 65)

    for case in TEST_CASES:
        query = case["query"]
        expected_page = case["page"]

        # E5 retrieval models work best with the query prefix.
        embedding = model.encode(
            ["query: " + query],
            normalize_embeddings=True
        ).tolist()

        result = collection.query(
            query_embeddings=embedding,
            n_results=5,
            include=["metadatas", "distances"]
        )

        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        retrieved_pages = [
            metadata.get("page")
            for metadata in metadatas
        ]

        found_at_1 = bool(retrieved_pages) and (
            retrieved_pages[0] == expected_page
        )
        found_at_5 = expected_page in retrieved_pages

        hit_at_1 += int(found_at_1)
        hit_at_5 += int(found_at_5)

        print(f"\nQuery: {query}")
        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Hit@1: {'PASS' if found_at_1 else 'MISS'}")
        print(f"Hit@5: {'PASS' if found_at_5 else 'MISS'}")

        if distances:
            print(f"Top result distance: {distances[0]:.4f}")

    total = len(TEST_CASES)

    print("\n" + "=" * 65)
    print(f"Hit@1: {hit_at_1}/{total} ({hit_at_1 / total:.1%})")
    print(f"Hit@5: {hit_at_5}/{total} ({hit_at_5 / total:.1%})")


if __name__ == "__main__":
    main()
