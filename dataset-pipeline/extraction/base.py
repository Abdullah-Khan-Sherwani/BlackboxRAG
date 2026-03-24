"""Abstract base for PDF text extractors (Strategy Pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.models import ExtractionResult


class BaseExtractor(ABC):
    """Interface for PDF-to-text extraction strategies.

    Implement this to add new extractors (Marker, MinerU, etc.)
    without modifying the rest of the pipeline.
    """

    @abstractmethod
    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract text from a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractionResult with status, text, sections found, and timing.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this extractor."""
