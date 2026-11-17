
import json
import re
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "legal_documents"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text, chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a paragraph or sentence boundary.
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end)
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def process_documents():
    all_chunks = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {PDF_DIR}"
        )

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        # Check PDF signature before parsing.
        with open(pdf_path, "rb") as file:
            if file.read(5) != b"%PDF-":
                print("Skipping invalid PDF:", pdf_path.name)
                continue

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = clean_text(page_text)

            if not page_text:
                print(f"Warning: No text on page {page_number}")
                continue

            page_chunks = split_into_chunks(page_text)

            for chunk_number, chunk in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "document": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk
                })

        print(f"Pages processed: {len(reader.pages)}")

    output_file = OUTPUT_DIR / "legal_chunks.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n========== PROCESSING COMPLETE ==========")
    print("Total chunks:", len(all_chunks))
    print("Output file:", output_file)

    return all_chunks


if __name__ == "__main__":
    process_documents()

