"""
========================================================
  Phishing Email Detection – Dataset Downloader
  IICT Summer Internship Project 2
========================================================
Downloads the Phishing Email Dataset (Phishing_Email.csv)
from a verified public GitHub source and saves it to data/.
"""

import os
import urllib.request

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "SINANFIROZ/Phishing-Email-Detector/master/Phishing_Email.csv"
)
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
SAVE_PATH = os.path.join(DATA_DIR, "Phishing_Email.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── Main ─────────────────────────────────────────────────────────────────────
def download() -> str:
    """Download dataset and return local path."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SAVE_PATH):
        size_kb = os.path.getsize(SAVE_PATH) / 1024
        print(f"[INFO] Dataset already present: {SAVE_PATH}  ({size_kb:.0f} KB)")
        return SAVE_PATH

    print(f"[INFO] Downloading dataset from:\n  {DATASET_URL}")
    req = urllib.request.Request(DATASET_URL, headers=HEADERS)

    with urllib.request.urlopen(req, timeout=60) as response:
        total      = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk      = 8192
        with open(SAVE_PATH, "wb") as f:
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total > 0:
                    pct = min(downloaded / total * 100, 100)
                    kb  = downloaded / 1024
                    print(f"\r  Downloading … {pct:5.1f}%  ({kb:.0f} KB)", end="", flush=True)

    print(f"\n[OK]  Saved to: {SAVE_PATH}")
    return SAVE_PATH


def verify(path: str) -> None:
    """Quick sanity check on the downloaded file."""
    import pandas as pd
    df = pd.read_csv(path)
    expected = {"Email Text", "Email Type"}
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    print(f"[OK]  Rows   : {len(df):,}")
    print(f"[OK]  Columns: {df.columns.tolist()}")
    print(f"[OK]  Labels : {df['Email Type'].unique().tolist()}")


if __name__ == "__main__":
    path = download()
    print("[INFO] Verifying dataset integrity …")
    verify(path)
    print("\n[DONE] Dataset is ready for use.")
