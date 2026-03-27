"""Download all NTSB Class 1 AAR/AIR report PDFs by enumerating known URL patterns.

Tries all combinations of:
  - AIR{YY}{01-10}.pdf  (2022-present)
  - AAR{YY}{01-10}.pdf  (1967-2021)

Downloads newest first (AIR before AAR, recent years first).
Skips files that already exist locally.

Usage:
    cd dataset-pipeline
    ..\venv\Scripts\python -m scripts.download_aars
    ..\venv\Scripts\python -m scripts.download_aars --dry-run   # Just enumerate, don't download
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.settings import (
    DOWNLOAD_DELAY_SECONDS,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_BACKOFF_BASE,
    DOWNLOAD_TIMEOUT_SECONDS,
    ENUM_MAX_SEQUENCE,
    ENUM_YEAR_END,
    ENUM_YEAR_START,
    NTSB_REPORT_BASE_URL,
    PDF_DIR,
    REPORT_PREFIXES,
    REQUEST_HEADERS,
    ensure_dirs,
)


def build_candidate_urls() -> list[tuple[str, str]]:
    """Generate all candidate (filename, url) pairs, newest first.

    Returns:
        List of (filename, url) tuples ordered newest → oldest.
    """
    candidates: list[tuple[str, str]] = []

    for prefix in REPORT_PREFIXES:
        # Build year list: for AAR, go 21 down to 67 then 99 down to 22 (wrapping)
        # For AIR, go 25 down to 22
        if prefix == "AIR":
            years = list(range(ENUM_YEAR_END, 21, -1))  # 25, 24, 23, 22
        else:
            # AAR: 21 down to 00, then 99 down to 67
            years = list(range(21, -1, -1)) + list(range(99, ENUM_YEAR_START - 1, -1))

        for year in years:
            for seq in range(1, ENUM_MAX_SEQUENCE + 1):
                filename = f"{prefix}{year:02d}{seq:02d}.pdf"
                url = f"{NTSB_REPORT_BASE_URL}/{filename}"
                candidates.append((filename, url))

    return candidates


def download_one(url: str, dest: Path) -> bool:
    """Download a single PDF with retry logic.

    Returns:
        True if download succeeded, False otherwise.
    """
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        try:
            resp = requests.head(
                url,
                headers=REQUEST_HEADERS,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                allow_redirects=True,
            )

            # 404 or other error — this report doesn't exist
            if resp.status_code == 404:
                return False
            if resp.status_code != 200:
                return False

            # Verify it's a PDF by content-type
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                return False

            # HEAD succeeded — now GET the full file
            resp = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()

            # Verify PDF magic bytes
            if not resp.content[:5] == b"%PDF-":
                return False

            dest.write_bytes(resp.content)
            return True

        except requests.RequestException:
            if attempt < DOWNLOAD_MAX_RETRIES - 1:
                wait = DOWNLOAD_BACKOFF_BASE ** (attempt + 1)
                time.sleep(wait)
            continue

    return False


def run(dry_run: bool = False) -> None:
    """Enumerate all AAR/AIR URLs and download valid PDFs."""
    ensure_dirs()

    candidates = build_candidate_urls()
    print(f"\n{'='*60}")
    print(f"  NTSB AAR/AIR PDF Downloader")
    print(f"{'='*60}")
    print(f"  Candidates to check: {len(candidates)}")
    print(f"  Output directory:    {PDF_DIR}")

    if dry_run:
        print(f"\n  DRY RUN — listing first 20 candidates:")
        for filename, url in candidates[:20]:
            print(f"    {filename}: {url}")
        print(f"  ... and {len(candidates) - 20} more")
        return

    # Check which we already have
    existing = {f.name for f in PDF_DIR.glob("*.pdf")}
    to_check = [(f, u) for f, u in candidates if f not in existing]
    already = len(candidates) - len(to_check)

    if already > 0:
        print(f"  Already downloaded:  {already}")
    print(f"  URLs to check:      {len(to_check)}\n")

    found = 0
    failed_streak = 0
    max_streak = ENUM_MAX_SEQUENCE  # If a full year has no reports, skip rest

    pbar = tqdm(to_check, desc="Checking URLs", unit="url")
    for filename, url in pbar:
        pbar.set_postfix(found=found, file=filename)
        dest = PDF_DIR / filename

        success = download_one(url, dest)

        if success:
            found += 1
            failed_streak = 0
            size_mb = dest.stat().st_size / (1024 * 1024)
            tqdm.write(f"  Downloaded: {filename} ({size_mb:.1f} MB)")
        else:
            failed_streak += 1

        time.sleep(DOWNLOAD_DELAY_SECONDS)

    # Summary
    total = found + already
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  New downloads:     {found}")
    print(f"  Previously cached: {already}")
    print(f"  Total PDFs:        {total}")
    print(f"  Location:          {PDF_DIR}")

    # List what we have
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"), reverse=True)
    if all_pdfs:
        total_size = sum(f.stat().st_size for f in all_pdfs) / (1024 * 1024)
        print(f"  Total size:        {total_size:.0f} MB")
        print(f"\n  Reports by type:")
        air_count = sum(1 for f in all_pdfs if f.name.startswith("AIR"))
        aar_count = sum(1 for f in all_pdfs if f.name.startswith("AAR"))
        print(f"    AIR (2022+):  {air_count}")
        print(f"    AAR (pre-22): {aar_count}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all NTSB Class 1 AAR/AIR report PDFs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just show candidate URLs without downloading",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
