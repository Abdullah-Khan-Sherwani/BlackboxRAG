"""Custom exception hierarchy for the NTSB dataset pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class DownloadError(PipelineError):
    """Raised when a PDF download fails after all retries."""


class ExtractionError(PipelineError):
    """Raised when text extraction from a PDF fails."""


class ValidationError(PipelineError):
    """Raised when extracted text fails quality checks."""


class MetadataError(PipelineError):
    """Raised when metadata fetching or parsing fails."""


class TrackerError(PipelineError):
    """Raised when the SQLite tracker encounters an error."""
