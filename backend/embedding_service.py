import hashlib
import json
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "legal_chunks.json"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "legal_embeddings.npz"
CHECKPOINT_FILE = BASE_DIR / "data" / "processed" / "legal_embeddings_checkpoint.npz"

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIMENSIONS = 768
SAVE_EVERY = 10
PAUSE_SECONDS = 0.2


def load_chunks_and_fingerprint():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\0")

    return chunks, digest.hexdigest()


def save_checkpoint(vectors, fingerprint, total_chunks):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size == 0:
        matrix = np.empty((0, VECTOR_DIMENSIONS), dtype=np.float32)

    # Write atomically so an interruption cannot leave a half-written checkpoint.
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_file,
        embeddings=matrix,
        fingerprint=np.asarray(fingerprint),
        total_chunks=np.asarray(total_chunks),
    )
    temp_file.replace(CHECKPOINT_FILE)


def load_checkpoint(fingerprint, total_chunks):
    if not CHECKPOINT_FILE.exists():
        return []

    try:
        with np.load(CHECKPOINT_FILE, allow_pickle=False) as data:
            saved_fingerprint = str(data["fingerprint"].item())
            saved_total = int(data["total_chunks"].item())
            vectors = np.asarray(data["embeddings"], dtype=np.float32)

        if saved_fingerprint != fingerprint or saved_total != total_chunks:
            print("Checkpoint does not match the current chunks.")
            print("It will not be reused. A new checkpoint will be created.")
            return []

        if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIMENSIONS:
            print("Checkpoint vector dimensions are invalid; starting a new checkpoint.")
            return []

        if len(vectors) > total_chunks:
            print("Checkpoint has more vectors than chunks; starting a new checkpoint.")
            return []

        print(f"Loaded checkpoint: {len(vectors)}/{total_chunks} embeddings")
        return [row.copy() for row in vectors]

    except Exception as error:
        raise RuntimeError(
            f"Could not safely read checkpoint {CHECKPOINT_FILE}: {error}"
        ) from error


def generate_embedding(text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=VECTOR_DIMENSIONS,
        ),
    )
    vector = response.embeddings[0].values

    if len(vector) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Expected {VECTOR_DIMENSIONS} dimensions, received {len(vector)}"
        )

    return vector


def create_embeddings():
    chunks, fingerprint = load_chunks_and_fingerprint()
    total = len(chunks)

    if total == 0:
        raise RuntimeError("No chunks found. Run the document processor first.")

    vectors = load_checkpoint(fingerprint, total)
    completed = len(vectors)

    print(f"Total chunks: {total}")
    print(f"Completed: {completed}")
    print(f"Remaining: {total - completed}")

    for index in range(completed, total):
        try:
            vector = generate_embedding(chunks[index]["text"])
            vectors.append(vector)

            if len(vectors) % SAVE_EVERY == 0 or len(vectors) == total:
                save_checkpoint(vectors, fingerprint, total)
                print(f"Checkpoint saved: {len(vectors)}/{total}")

            if len(vectors) % 25 == 0:
                print(f"Progress: {len(vectors)}/{total}")

            time.sleep(PAUSE_SECONDS)

        except Exception as error:
            # Save every completed vector before exiting, including progress
            # since the previous periodic checkpoint.
            save_checkpoint(vectors, fingerprint, total)

            message = str(error)
            print(f"\nStopped while processing chunk {index + 1}/{total}.")
            print(f"Saved progress: {len(vectors)}/{total} embeddings.")
            print(f"Reason: {message}")

            if (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "quota" in message.lower()
            ):
                print("\nGemini quota reached.")
                print("Wait until quota resets, then run:")
                print("python -m backend.embedding_service")
                print("The script will resume from the saved checkpoint.")
                return

            raise

    final_matrix = np.asarray(vectors, dtype=np.float32)
    expected_shape = (total, VECTOR_DIMENSIONS)

    if final_matrix.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected final shape {final_matrix.shape}; expected {expected_shape}"
        )

    # Keep the checkpoint until the final output has been written successfully.
    temp_output = OUTPUT_FILE.with_suffix(".tmp.npz")
    np.savez_compressed(temp_output, embeddings=final_matrix)
    temp_output.replace(OUTPUT_FILE)

    print("\n========== EMBEDDINGS COMPLETE ==========")
    print("Total embeddings:", len(final_matrix))
    print("Vector dimensions:", final_matrix.shape[1])
    print("Saved to:", OUTPUT_FILE)

    CHECKPOINT_FILE.unlink(missing_ok=True)
    print("Checkpoint removed after successful completion.")


if __name__ == "__main__":
    create_embeddings()
