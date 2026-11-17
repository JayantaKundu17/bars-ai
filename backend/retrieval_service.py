
import json
import re
import numpy as np

from pathlib import Path
from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

client = genai.Client(api_key=GEMINI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "legal_chunks.json"
)

EMBEDDINGS_FILE = (
    BASE_DIR / "data" / "processed" / "legal_embeddings.npz"
)

EMBEDDING_MODEL = "gemini-embedding-001"

DEFAULT_TOP_K = 5


# --------------------------------------------------
# LOAD LEGAL KNOWLEDGE BASE
# --------------------------------------------------

def load_knowledge_base():
    """
    Load legal text chunks and their corresponding
    vector embeddings.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Legal chunks file not found: {CHUNKS_FILE}"
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    stored_data = np.load(EMBEDDINGS_FILE)
    embeddings = stored_data["embeddings"]

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk and embedding counts do not match. "
            "Regenerate the embeddings."
        )

    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(
            f"Unexpected embedding dimensions: {embeddings.shape}"
        )

    return chunks, embeddings


# --------------------------------------------------
# GENERATE QUERY EMBEDDING
# --------------------------------------------------

def embed_query(question):
    """
    Convert the user's legal question into a vector
    using Gemini's embedding model.
    """

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )


# --------------------------------------------------
# FILTER IRRELEVANT CHUNKS
# --------------------------------------------------

def is_useful_chunk(text):
    """
    Filter obvious footnotes, reference-only passages,
    page-number-only chunks and low-quality text.

    This is a basic heuristic filter, not a legal
    relevance classifier.
    """

    cleaned = text.strip()

    # Reject very short chunks.
    if len(cleaned) < 150:
        return False

    lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    # Common footnote and amendment-reference patterns.
    footnote_patterns = [
        r"^\d+\s+(Subs\.|Ins\.|Omitted|See the)",
        r"^\d+\.\s+(Subs\.|Ins\.|Omitted|See the)",
        r"^Subs\. by Act",
        r"^Ins\. by Act",
        r"^Omitted by Act",
        r"^See\s+the\s+",
    ]

    footnote_lines = sum(
        1
        for line in lines
        if any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in footnote_patterns
        )
    )

    # Reject chunks dominated by footnote references.
    if footnote_lines / len(lines) > 0.20:
        return False

    words = cleaned.split()

    if not words:
        return False

    # Reject chunks dominated by standalone numbers.
    number_ratio = sum(
        1
        for word in words
        if re.fullmatch(r"\d+\.?", word)
    ) / len(words)

    if number_ratio > 0.35:
        return False

    # Reject chunks with too little alphabetic content.
    alphabetic_count = sum(
        character.isalpha()
        for character in cleaned
    )

    alphabetic_ratio = alphabetic_count / len(cleaned)

    if alphabetic_ratio < 0.45:
        return False

    return True


# --------------------------------------------------
# SEARCH LEGAL DOCUMENTS
# --------------------------------------------------

def search_legal_documents(question, top_k=DEFAULT_TOP_K):
    """
    Find the most similar useful legal passages
    for a user's question using cosine similarity.

    Returns a list of passages containing:
    document, page, chunk, text and similarity.
    """

    if not question or not question.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Step 1: Load knowledge base.
    chunks, embeddings = load_knowledge_base()

    # Step 2: Generate query vector.
    query_vector = embed_query(question)

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        raise ValueError("Query embedding is zero.")

    # Step 3: Calculate embedding norms.
    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    # Step 4: Calculate cosine similarities.
    similarities = np.dot(
        embeddings,
        query_vector
    ) / (
        embedding_norms * query_norm
    )

    # Step 5: Sort by highest similarity.
    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    # Step 6: Filter and collect useful passages.
    for index in ranked_indices:

        chunk = chunks[int(index)]

        if not is_useful_chunk(chunk["text"]):
            continue

        result = chunk.copy()

        result["similarity"] = float(
            similarities[index]
        )

        results.append(result)

        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------
# TEST RETRIEVAL SERVICE
# --------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        "My landlord is refusing to return "
        "my security deposit after I vacated.",

        "What rights does a tenant have under "
        "the Transfer of Property Act?",

        "A customer purchased a defective product "
        "and the seller refuses to provide a refund."
    ]

    for question in test_questions:

        print("\n" + "=" * 65)
        print("QUESTION:", question)
        print("=" * 65)

        results = search_legal_documents(
            question,
            top_k=5
        )

        if not results:
            print("No useful legal passages found.")
            continue

        for index, result in enumerate(results, start=1):

            print(f"\nRESULT {index}")

            print("Document:", result["document"])
            print("Page:", result["page"])
            print("Chunk:", result["chunk"])

            print(
                "Similarity:",
                round(result["similarity"], 4)
            )

            print("Text:")
            print(result["text"][:600])

            print("-" * 65)
