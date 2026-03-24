"""Centralized configuration for the NTSB dataset pipeline."""

from pathlib import Path

# ── Directory Paths ──────────────────────────────────────────────────────────

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PIPELINE_ROOT.parent  # Parent Project 1 directory

DATA_DIR = PIPELINE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = DATA_DIR / "pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"
OUTPUT_DIR = DATA_DIR / "output"
TRACKER_DB = DATA_DIR / "tracker.db"

# Parent project paths (read-only)
PARENT_DATA_DIR = PROJECT_ROOT / "data"
PARENT_SAMPLED_CSV = PARENT_DATA_DIR / "processed" / "sampled_reports.csv"
PARENT_RAW_CSV = PARENT_DATA_DIR / "raw" / "final_reports_2016-23_cons_2024-12-24.csv"

# ── NTSB Data Source ─────────────────────────────────────────────────────────

AVALL_ZIP_URL = (
    "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile"
    "?fileID=C%3A%5Cavdata%5Cavall.zip"
)

# ── Download Settings ────────────────────────────────────────────────────────

DOWNLOAD_DELAY_SECONDS = 0.5
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_BACKOFF_BASE = 2  # Exponential: 2s, 4s, 8s
DOWNLOAD_TIMEOUT_SECONDS = 60
REQUEST_HEADERS = {
    "User-Agent": "NTSB-Dataset-Builder/1.0 (academic research)"
}

# ── Extraction Settings ──────────────────────────────────────────────────────

MIN_WORD_COUNT = 200

NTSB_EXPECTED_SECTIONS = [
    "History of Flight",
    "Pilot Information",
    "Aircraft and Owner/Operator Information",
    "Meteorological Information",
    "Wreckage and Impact Information",
    "Probable Cause",
    "Findings",
]

# ── Pipeline Settings ────────────────────────────────────────────────────────

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_FULL_LIMIT = 10_000

# ── Metadata CSV Settings ────────────────────────────────────────────────────

CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8"

# ── Filtering Criteria ───────────────────────────────────────────────────────

FILTER_REPORT_STATUS = "Final"
FILTER_MIN_DATE = "2000-01-01"
FILTER_MODE = "Aviation"

# ── Ensure data directories exist ────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all required data directories if they don't exist."""
    for d in (RAW_DIR, PDF_DIR, EXTRACTED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
