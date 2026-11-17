from pathlib import Path
from urllib.request import Request, urlopen
import time

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


def download_pdf(filename, url, max_attempts=4):
    destination = OUTPUT_DIR / filename

    if destination.exists() and destination.stat().st_size > 1000:
        with destination.open("rb") as file:
            if file.read(5) == b"%PDF-":
                print(f"Already downloaded: {filename}")
                return

    last_error = None

    for attempt in range(1, max_attempts + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                    "Accept": "application/pdf,*/*",
                    "Connection": "close",
                },
            )

            print(f"Downloading {filename} (attempt {attempt}/{max_attempts}) ...")

            with urlopen(request, timeout=180) as response:
                content_type = response.headers.get("Content-Type", "")
                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

            with temporary.open("rb") as file:
                signature = file.read(5)

            if signature != b"%PDF-":
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"Server did not return a PDF "
                    f"(Content-Type: {content_type or 'unknown'})."
                )

            temporary.replace(destination)
            print(f"Saved: {destination} ({destination.stat().st_size:,} bytes)")
            return

        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            print(f"Attempt {attempt} failed: {error}")

            if attempt < max_attempts:
                wait_seconds = attempt * 5
                print(f"Waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Could not download {filename} after {max_attempts} attempts: {last_error}"
    )


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
