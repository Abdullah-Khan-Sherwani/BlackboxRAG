"""Post-extraction text cleaning for NTSB reports."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


class TextCleaner:
    """Cleans Docling-extracted markdown while preserving useful structure.

    Removes NTSB boilerplate, normalizes whitespace, and strips artifacts
    while keeping markdown section headers intact for downstream processing.
    """

    # Boilerplate patterns commonly found in NTSB PDFs
    BOILERPLATE_PATTERNS = [
        re.compile(r"National Transportation Safety Board", re.IGNORECASE),
        re.compile(r"NTSB/\w+[-/]\d+", re.IGNORECASE),  # Report number headers
        re.compile(r"Page\s+\d+\s*of\s*\d+", re.IGNORECASE),
        re.compile(r"Form\s+6120[.\-]\d+", re.IGNORECASE),  # NTSB form numbers
        re.compile(r"OMB No\.\s*[\d-]+", re.IGNORECASE),
    ]

    def clean(self, text: str) -> str:
        """Clean extracted markdown text.

        Args:
            text: Raw markdown from Docling extraction.

        Returns:
            Cleaned text with boilerplate removed, whitespace normalized,
            and markdown headers preserved.
        """
        if not text:
            return ""

        text = self._normalize_unicode(text)
        text = self._strip_boilerplate(text)
        text = self._normalize_whitespace(text)
        text = self._strip_non_printable(text)
        return text.strip()

    def clean_file(self, file_path: Path) -> str:
        """Read and clean a single markdown file.

        Args:
            file_path: Path to the .md file.

        Returns:
            Cleaned text content.
        """
        raw = file_path.read_text(encoding="utf-8")
        cleaned = self.clean(raw)
        file_path.write_text(cleaned, encoding="utf-8")
        return cleaned

    def clean_batch(self, extracted_dir: Path) -> dict[str, int]:
        """Clean all .md files in a directory.

        Args:
            extracted_dir: Directory containing extracted .md files.

        Returns:
            Dict mapping filename to word count after cleaning.
        """
        results: dict[str, int] = {}
        for md_file in sorted(extracted_dir.glob("*.md")):
            cleaned = self.clean_file(md_file)
            results[md_file.stem] = len(cleaned.split())
        return results

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize to NFC form and replace common unicode issues."""
        text = unicodedata.normalize("NFC", text)
        # Replace common problematic characters
        replacements = {
            "\u2018": "'", "\u2019": "'",   # Smart quotes
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--",  # Dashes
            "\u00a0": " ",                   # Non-breaking space
            "\ufffd": "",                    # Replacement character
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _strip_boilerplate(self, text: str) -> str:
        """Remove known NTSB boilerplate lines."""
        lines = text.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if any(p.fullmatch(stripped) for p in self.BOILERPLATE_PATTERNS):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse excessive whitespace while preserving markdown structure."""
        # Collapse 3+ consecutive blank lines to 2 (preserve paragraph breaks)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        # Collapse multiple spaces within lines (but not leading indentation)
        text = re.sub(r"([^\n]) {2,}", r"\1 ", text)
        return text

    @staticmethod
    def _strip_non_printable(text: str) -> str:
        """Remove non-printable characters except common whitespace."""
        return "".join(
            ch for ch in text
            if ch in ("\n", "\t", "\r") or unicodedata.category(ch)[0] != "C"
        )
