"""Phase 1: Small sample test of the Docling extraction pipeline.

Downloads 20 PDFs from the parent project's sampled_reports.csv,
extracts text with Docling, and compares output quality against
the pre-extracted rep_text column.

Usage:
    cd dataset-pipeline
    python -m scripts.test_sample
    python -m scripts.test_sample --n 5   # Test with fewer reports
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Ensure the pipeline root is on sys.path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.settings import (
    DOWNLOAD_DELAY_SECONDS,
    DOWNLOAD_TIMEOUT_SECONDS,
    CSV_DELIMITER,
    CSV_ENCODING,
    DEFAULT_SAMPLE_SIZE,
    EXTRACTED_DIR,
    OUTPUT_DIR,
    PDF_DIR,
    PARENT_SAMPLED_CSV,
    REQUEST_HEADERS,
    TRACKER_DB,
    ensure_dirs,
)
from core.models import ExtractionResult, ExtractionStatus, ReportRecord
from core.tracker import ReportTracker
from extraction.cleaner import TextCleaner
from extraction.docling_extractor import DoclingExtractor
from processing.validator import QualityValidator


def load_sample(csv_path: Path, n: int) -> tuple[pd.DataFrame, list[ReportRecord]]:
    """Load n reports from the parent project's sampled CSV.

    Returns:
        Tuple of (raw DataFrame, list of ReportRecord).
    """
    df = pd.read_csv(csv_path, sep=CSV_DELIMITER, encoding=CSV_ENCODING)

    # Filter to rows that have a ReportUrl
    df = df[df["ReportUrl"].notna() & (df["ReportUrl"].str.strip() != "")]
    df = df.head(n).reset_index(drop=True)

    records: list[ReportRecord] = []
    for _, row in df.iterrows():
        records.append(ReportRecord(
            ntsb_id=str(row["NtsbNo"]),
            event_date=str(row.get("EventDate", "")),
            event_id=str(row.get("EventID", "")) if pd.notna(row.get("EventID")) else None,
            report_no=str(row.get("ReportNo", "")) if pd.notna(row.get("ReportNo")) else None,
            city=str(row.get("City", "")) if pd.notna(row.get("City")) else None,
            state=str(row.get("State", "")) if pd.notna(row.get("State")) else None,
            country=str(row.get("Country", "")) if pd.notna(row.get("Country")) else None,
            aircraft_make=str(row.get("Make", "")) if pd.notna(row.get("Make")) else None,
            aircraft_model=str(row.get("Model", "")) if pd.notna(row.get("Model")) else None,
            aircraft_category=str(row.get("AirCraftCategory", "")) if pd.notna(row.get("AirCraftCategory")) else None,
            engine_type=str(row.get("EngineType", "")) if pd.notna(row.get("EngineType")) else None,
            num_engines=int(row["NumberOfEngines"]) if pd.notna(row.get("NumberOfEngines")) else None,
            operator=str(row.get("Operator", "")) if pd.notna(row.get("Operator")) else None,
            purpose_of_flight=str(row.get("PurposeOfFlight", "")) if pd.notna(row.get("PurposeOfFlight")) else None,
            fatal_count=int(row.get("FatalInjuryCount", 0) or 0),
            serious_count=int(row.get("SeriousInjuryCount", 0) or 0),
            minor_count=int(row.get("MinorInjuryCount", 0) or 0),
            aircraft_damage=str(row.get("AirCraftDamage", "")) if pd.notna(row.get("AirCraftDamage")) else None,
            weather_condition=str(row.get("WeatherCondition", "")) if pd.notna(row.get("WeatherCondition")) else None,
            phase_of_flight=str(row.get("BroadPhaseofFlight", "")) if pd.notna(row.get("BroadPhaseofFlight")) else None,
            probable_cause=str(row.get("ProbableCause", "")) if pd.notna(row.get("ProbableCause")) else None,
            report_url=str(row["ReportUrl"]),
            docket_url=str(row.get("DocketUrl", "")) if pd.notna(row.get("DocketUrl")) else None,
        ))

    return df, records


def download_pdf(report: ReportRecord, tracker: ReportTracker) -> Path | None:
    """Download a single PDF, skipping if already downloaded.

    Returns:
        Path to the PDF file, or None on failure.
    """
    pdf_path = PDF_DIR / f"{report.ntsb_id}.pdf"

    # Skip if already downloaded
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        tracker.update_download(report.ntsb_id, str(pdf_path), "success")
        return pdf_path

    try:
        resp = requests.get(
            report.report_url,
            headers=REQUEST_HEADERS,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()

        # Verify it's actually a PDF
        if not resp.content[:5] == b"%PDF-":
            tracker.update_download(report.ntsb_id, "", "failed")
            return None

        pdf_path.write_bytes(resp.content)
        tracker.update_download(report.ntsb_id, str(pdf_path), "success")
        return pdf_path

    except Exception as e:
        tracker.update_download(report.ntsb_id, "", f"failed: {e}")
        return None


def run_sample_test(n: int) -> None:
    """Run the full Phase 1 sample test."""
    ensure_dirs()

    print(f"\n{'='*60}")
    print(f"  NTSB Dataset Pipeline — Phase 1 Sample Test ({n} reports)")
    print(f"{'='*60}\n")

    # ── Load sample ──────────────────────────────────────────────
    print(f"Loading sample from: {PARENT_SAMPLED_CSV}")
    if not PARENT_SAMPLED_CSV.exists():
        print(f"ERROR: Parent CSV not found at {PARENT_SAMPLED_CSV}")
        sys.exit(1)

    df, records = load_sample(PARENT_SAMPLED_CSV, n)
    original_texts = df["rep_text"].fillna("").tolist()
    print(f"Loaded {len(records)} reports with valid ReportUrl\n")

    # ── Initialize components ────────────────────────────────────
    tracker = ReportTracker(TRACKER_DB)
    extractor = DoclingExtractor()
    cleaner = TextCleaner()
    validator = QualityValidator()

    # Register all reports
    tracker.register_batch([r.ntsb_id for r in records])

    # ── Download PDFs ────────────────────────────────────────────
    print("Downloading PDFs...")
    pdf_paths: list[Path | None] = []
    for report in tqdm(records, desc="Downloading"):
        path = download_pdf(report, tracker)
        pdf_paths.append(path)
        if path is not None:
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    downloaded = sum(1 for p in pdf_paths if p is not None)
    print(f"Downloaded: {downloaded}/{len(records)}\n")

    # ── Extract with Docling ─────────────────────────────────────
    print("Extracting text with Docling...")
    results: list[ExtractionResult] = []
    for i, (report, pdf_path) in enumerate(
        tqdm(zip(records, pdf_paths), total=len(records), desc="Extracting")
    ):
        if pdf_path is None:
            results.append(ExtractionResult(
                status=ExtractionStatus.FAILED,
                error_message="PDF download failed",
            ))
            continue

        result = extractor.extract(pdf_path)

        # Clean the extracted text
        if result.markdown_text:
            result.markdown_text = cleaner.clean(result.markdown_text)
            result.word_count = len(result.markdown_text.split())

            # Save markdown file
            md_path = EXTRACTED_DIR / f"{report.ntsb_id}.md"
            md_path.write_text(result.markdown_text, encoding="utf-8")

            # Update tracker
            text_hash = hashlib.sha256(result.markdown_text.encode()).hexdigest()
            tracker.update_extraction(
                report.ntsb_id, result.status, result.word_count, text_hash
            )
        else:
            tracker.update_extraction(report.ntsb_id, result.status)

        results.append(result)

    # ── Generate quality report ──────────────────────────────────
    print("\nGenerating quality report...")
    quality_df = validator.generate_quality_report(records, results, original_texts)

    report_path = OUTPUT_DIR / "docling_quality_check.csv"
    quality_df.to_csv(report_path, index=False)
    print(f"Saved: {report_path}\n")

    # ── Print summary ────────────────────────────────────────────
    print(f"{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")

    status_counts = quality_df["extraction_status"].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    success_count = (quality_df["extraction_status"] == "success").sum()
    success_rate = success_count / len(quality_df) * 100
    print(f"\n  Success rate: {success_rate:.0f}% ({success_count}/{len(quality_df)})")

    if "word_count_original" in quality_df.columns:
        avg_original = quality_df["word_count_original"].mean()
        avg_docling = quality_df["word_count_docling"].mean()
        print(f"  Avg word count (original): {avg_original:.0f}")
        print(f"  Avg word count (docling):  {avg_docling:.0f}")

    avg_sections = quality_df["num_sections"].mean()
    print(f"  Avg sections detected: {avg_sections:.1f}")

    passed = quality_df["validation_passed"].sum()
    print(f"  Validation passed: {passed}/{len(quality_df)}")

    avg_time = quality_df["extraction_time_seconds"].mean()
    print(f"  Avg extraction time: {avg_time:.1f}s per report")

    print(f"\n  Tracker stats: {tracker.get_stats()}")
    print(f"{'='*60}")
    print(f"\nReview extracted files in: {EXTRACTED_DIR}")
    print(f"Review quality report at:  {report_path}")

    # Pass/fail gate
    if success_rate >= 80:
        print("\n  PASS — Proceed to Phase 2 (full pipeline)")
    else:
        print("\n  FAIL — Review errors before scaling up")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Docling sample test")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of reports to test (default: {DEFAULT_SAMPLE_SIZE})",
    )
    args = parser.parse_args()
    run_sample_test(args.n)


if __name__ == "__main__":
    main()
