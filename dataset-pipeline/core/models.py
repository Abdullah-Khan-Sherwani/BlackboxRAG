"""Data models for the NTSB dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExtractionStatus(Enum):
    """Status of a report through the pipeline."""
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    SUCCESS = "success"
    PARTIAL = "partial"       # Extracted but below word threshold
    FAILED = "failed"
    SKIPPED = "skipped"       # Duplicate or no URL


class SourceType(Enum):
    """How the report text was obtained."""
    PDF_DOCLING = "pdf_docling"
    METADATA_ONLY = "metadata_only"


@dataclass
class ReportRecord:
    """Core data transfer object representing a single NTSB report."""

    ntsb_id: str
    event_date: str  # ISO format

    # Identifiers
    event_id: str | None = None
    report_no: str | None = None

    # Location
    city: str | None = None
    state: str | None = None
    country: str | None = None

    # Aircraft
    aircraft_make: str | None = None
    aircraft_model: str | None = None
    aircraft_category: str | None = None
    engine_type: str | None = None
    num_engines: int | None = None

    # Operation
    operator: str | None = None
    purpose_of_flight: str | None = None
    phase_of_flight: str | None = None
    weather_condition: str | None = None

    # Injuries & damage
    fatal_count: int = 0
    serious_count: int = 0
    minor_count: int = 0
    aircraft_damage: str | None = None

    # Investigation
    probable_cause: str | None = None
    report_url: str | None = None
    docket_url: str | None = None

    # Extracted content
    report_text: str | None = None
    source_type: SourceType = SourceType.PDF_DOCLING
    word_count: int = 0
    text_hash: str | None = None
    extraction_date: str | None = None


@dataclass
class ExtractionResult:
    """Result of extracting text from a single PDF."""

    status: ExtractionStatus
    markdown_text: str | None = None
    word_count: int = 0
    sections_found: list[str] = field(default_factory=list)
    extraction_time_seconds: float = 0.0
    error_message: str | None = None
