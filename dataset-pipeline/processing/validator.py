"""Quality validation for extracted NTSB reports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import MIN_WORD_COUNT, NTSB_EXPECTED_SECTIONS
from core.models import ExtractionResult, ReportRecord


@dataclass
class ValidationResult:
    """Outcome of validating a single extraction."""
    passed: bool
    issues: list[str]


class QualityValidator:
    """Validates extracted text meets quality thresholds.

    Checks word count, section presence, and other quality signals
    to determine if an extraction is usable.
    """

    REQUIRED_SECTIONS = ["Probable Cause"]

    def validate(self, result: ExtractionResult) -> ValidationResult:
        """Validate a single extraction result.

        Args:
            result: The ExtractionResult to check.

        Returns:
            ValidationResult with pass/fail and list of issues.
        """
        issues: list[str] = []

        if result.word_count < MIN_WORD_COUNT:
            issues.append(
                f"Word count {result.word_count} below minimum {MIN_WORD_COUNT}"
            )

        for section in self.REQUIRED_SECTIONS:
            if section not in result.sections_found:
                issues.append(f"Missing required section: {section}")

        return ValidationResult(passed=len(issues) == 0, issues=issues)

    def generate_quality_report(
        self,
        reports: list[ReportRecord],
        results: list[ExtractionResult],
        original_texts: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate a quality comparison report for Phase 1 sample testing.

        Args:
            reports: List of ReportRecords (with metadata).
            results: Corresponding ExtractionResults from Docling.
            original_texts: Optional original rep_text for comparison.

        Returns:
            DataFrame with per-report quality metrics.
        """
        rows: list[dict] = []
        for i, (report, result) in enumerate(zip(reports, results)):
            row: dict = {
                "ntsb_id": report.ntsb_id,
                "extraction_status": result.status.value,
                "word_count_docling": result.word_count,
                "extraction_time_seconds": round(result.extraction_time_seconds, 2),
                "sections_found": ", ".join(result.sections_found),
                "num_sections": len(result.sections_found),
                "error_message": result.error_message or "",
            }

            # Check each expected section
            for section in NTSB_EXPECTED_SECTIONS:
                key = f"has_{section.lower().replace(' ', '_').replace('/', '_')}"
                row[key] = section in result.sections_found

            # Compare with original text if available
            if original_texts and i < len(original_texts):
                original = original_texts[i]
                row["word_count_original"] = len(original.split()) if original else 0
                row["word_count_diff"] = row["word_count_docling"] - row["word_count_original"]

            # Validation
            validation = self.validate(result)
            row["validation_passed"] = validation.passed
            row["validation_issues"] = "; ".join(validation.issues) if validation.issues else ""

            rows.append(row)

        return pd.DataFrame(rows)
