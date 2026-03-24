"""Docling-based PDF text extraction."""

from __future__ import annotations

import re
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

from config.settings import NTSB_EXPECTED_SECTIONS, MIN_WORD_COUNT
from core.models import ExtractionResult, ExtractionStatus
from extraction.base import BaseExtractor


class DoclingExtractor(BaseExtractor):
    """Extracts structured markdown from PDFs using IBM Docling.

    Docling provides layout-aware extraction that preserves section
    headers, tables, and reading order — critical for NTSB reports
    where sections like "Probable Cause" need to be identifiable.
    """

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    @property
    def name(self) -> str:
        return "docling"

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract markdown text from a PDF using Docling.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractionResult with markdown text, detected sections, and timing.
        """
        if not pdf_path.exists():
            return ExtractionResult(
                status=ExtractionStatus.FAILED,
                error_message=f"PDF not found: {pdf_path}",
            )

        start = time.perf_counter()
        try:
            result = self._converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()
        except Exception as e:
            return ExtractionResult(
                status=ExtractionStatus.FAILED,
                extraction_time_seconds=time.perf_counter() - start,
                error_message=f"Docling conversion failed: {e}",
            )

        elapsed = time.perf_counter() - start
        word_count = len(markdown_text.split())
        sections = self._detect_sections(markdown_text)

        if word_count < MIN_WORD_COUNT:
            status = ExtractionStatus.PARTIAL
        else:
            status = ExtractionStatus.SUCCESS

        return ExtractionResult(
            status=status,
            markdown_text=markdown_text,
            word_count=word_count,
            sections_found=sections,
            extraction_time_seconds=elapsed,
        )

    @staticmethod
    def _detect_sections(text: str) -> list[str]:
        """Detect which NTSB report sections are present in the markdown.

        Looks for markdown headers (## or **Section Name**) that match
        known NTSB report section names.
        """
        found: list[str] = []
        text_lower = text.lower()
        for section in NTSB_EXPECTED_SECTIONS:
            # Match markdown headers or bold section names
            pattern = re.compile(
                rf"(?:^#+\s*{re.escape(section)}|^\*\*{re.escape(section)}\*\*)",
                re.IGNORECASE | re.MULTILINE,
            )
            if pattern.search(text) or section.lower() in text_lower:
                found.append(section)
        return found
