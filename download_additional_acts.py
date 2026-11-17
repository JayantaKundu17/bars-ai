from pathlib import Path
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "legal_documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTS = [
    (
        "bharatiya_nyaya_sanhita_2023.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf",
    ),
    (
        "bharatiya_nagarik_suraksha_sanhita_2023.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/20335/1/a2023-46.pdf",
    ),
    (
        "bharatiya_sakshya_adhiniyam_2023.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/20063/1/aa202347.pdf",
    ),
    (
        "motor_vehicles_act_1988.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1798/1/eng.pdf",
    ),
    (
        "information_technology_act_2000.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
    ),
    (
        "limitation_act_1963.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1565/5/A1963-36.pdf",
    ),
    (
        "specific_relief_act_1963.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/1583/7/A1963-47.pdf",
    ),
]


def download_pdf(filename, url):
    destination = OUTPUT_DIR / filename
    if destination.exists() and destination.stat().st_size > 1000:
        with destination.open("rb") as file:
            if file.read(5) == b"%PDF-":
                print(f"Already downloaded: {filename}")
                return

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 BARS-AI-Legal-Research-Prototype"},
    )
    print(f"Downloading {filename} ...")
    with urlopen(request, timeout=60) as response:
        content = response.read()

    if not content.startswith(b"%PDF-"):
        raise ValueError(
            f"{filename} did not return a valid PDF. "
            "Open the official India Code source URL and download it manually."
        )

    destination.write_bytes(content)
    print(f"Saved: {destination} ({len(content):,} bytes)")


if __name__ == "__main__":
    failures = []

    for filename, url in ACTS:
        try:
            download_pdf(filename, url)
        except Exception as error:
            failures.append((filename, str(error)))
            print(f"FAILED: {filename}: {error}")

    print("\nDownload summary")
    print(f"Successful or already present: {len(ACTS) - len(failures)}/{len(ACTS)}")
    if failures:
        print("\nDownload these files manually from their India Code source links:")
        for filename, error in failures:
            print(f"- {filename}: {error}")
        raise SystemExit(1)

    print("\nAll seven additional Acts are in data/legal_documents/.")
    print("Next, rebuild the text chunks and embedding index using the commands")
    print("shown in the setup instructions.")
