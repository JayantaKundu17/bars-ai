from pathlib import Path
from urllib.request import Request, urlopen
import time

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "legal_documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTS = [
    (
        "bharatiya_nagarik_suraksha_sanhita_2023.pdf",
        [
            "https://www.mha.gov.in/files/documents/250884_2english_01042024.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/21920/1/the_bharatiya_nagarik_suraksha_sanhita%2C_2023.pdf",
        ],
    ),
    (
        "bharatiya_sakshya_adhiniyam_2023.pdf",
        [
            "https://www.mha.gov.in/files/documents/250882_english_01042024_0.pdf",
            "https://www.indiacode.nic.in/indiacode/bitstream/123456789/20063/1/aa202347.pdf",
        ],
    ),
    (
        "motor_vehicles_act_1988.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1798/1/eng.pdf",
        ],
    ),
    (
        "limitation_act_1963.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1565/5/A1963-36.pdf",
        ],
    ),
]


def download_pdf(filename, urls, max_attempts=2):
    destination = OUTPUT_DIR / filename

    if destination.exists() and destination.stat().st_size > 1000:
        with destination.open("rb") as file:
            if file.read(5) == b"%PDF-":
                print(f"Already downloaded: {filename}")
                return

    last_error = None
    for url in urls:
        for attempt in range(1, max_attempts + 1):
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Accept": "application/pdf,*/*",
                        "Connection": "close",
                    },
                )
                print(f"Source: {url}")
                print(f"Downloading {filename} (attempt {attempt}/{max_attempts}) ...")

                with urlopen(request, timeout=180) as response:
                    with temporary.open("wb") as output:
                        while True:
                            chunk = response.read(64 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)

                with temporary.open("rb") as file:
                    if file.read(5) != b"%PDF-":
                        raise ValueError("The source did not return a valid PDF.")

                temporary.replace(destination)
                print(f"Saved: {destination} ({destination.stat().st_size:,} bytes)")
                return

            except Exception as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                print(f"Attempt failed: {error}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)

    raise RuntimeError(f"All official sources failed for {filename}: {last_error}")


if __name__ == "__main__":
    failures = []

    for filename, urls in ACTS:
        try:
            download_pdf(filename, urls)
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
