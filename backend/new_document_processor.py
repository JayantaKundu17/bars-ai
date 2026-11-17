import json
import re
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "new_legal_documents"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "new_legal_chunks.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end)
            )

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - CHUNK_OVERLAP)

    return chunks


def process_new_documents():
    all_chunks = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in:", PDF_DIR)
        print("Add new central Acts to this folder first.")
        return []

    for pdf_path in pdf_files:
        print("\nProcessing:", pdf_path.name)

        with open(pdf_path, "rb") as file:
            if file.read(5) != b"%PDF-":
                print("Skipping invalid PDF:", pdf_path.name)
                continue

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as error:
            print("Could not read PDF:", pdf_path.name, error)
            continue

        document_chunks = 0
        pages_with_text = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")

            if not page_text:
                continue

            pages_with_text += 1

            page_chunks = split_into_chunks(page_text)

            for chunk_number, chunk in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "document": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                    "retrieval_method": "bm25"
                })

                document_chunks += 1

        print("Pages:", len(reader.pages))
        print("Pages with extractable text:", pages_with_text)
        print("Chunks created:", document_chunks)

        if pages_with_text == 0:
            print("WARNING: This PDF may be scanned and require OCR.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n========== NEW DOCUMENT PROCESSING COMPLETE ==========")
    print("Total new chunks:", len(all_chunks))
    print("Saved to:", OUTPUT_FILE)

    return all_chunks


if __name__ == "__main__":
    process_new_documents()
