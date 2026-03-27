"""SQLite-backed pipeline state tracker (Repository Pattern)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from core.exceptions import TrackerError
from core.models import ExtractionStatus


class ReportTracker:
    """Tracks download and extraction state for every report in the pipeline.

    Uses SQLite for durability — the pipeline can be resumed at any point
    by reading the tracker state.
    """

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS processed_reports (
            ntsb_id         TEXT PRIMARY KEY,
            pdf_path        TEXT,
            download_status TEXT DEFAULT 'pending',
            download_date   TEXT,
            extract_status  TEXT DEFAULT 'pending',
            word_count      INTEGER DEFAULT 0,
            text_hash       TEXT,
            updated_at      TEXT DEFAULT (datetime('now'))
        )
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self.CREATE_TABLE_SQL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise TrackerError(f"Database error: {e}") from e
        finally:
            conn.close()

    # ── Queries ──────────────────────────────────────────────────────────

    def is_processed(self, ntsb_id: str) -> bool:
        """Check if a report has been successfully extracted."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT extract_status FROM processed_reports WHERE ntsb_id = ?",
                (ntsb_id,),
            ).fetchone()
            return row is not None and row["extract_status"] == ExtractionStatus.SUCCESS.value

    def get_status(self, ntsb_id: str) -> ExtractionStatus | None:
        """Get the extraction status for a report, or None if not tracked."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT extract_status FROM processed_reports WHERE ntsb_id = ?",
                (ntsb_id,),
            ).fetchone()
            if row is None:
                return None
            return ExtractionStatus(row["extract_status"])

    def get_download_status(self, ntsb_id: str) -> str | None:
        """Get the download status for a report."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT download_status FROM processed_reports WHERE ntsb_id = ?",
                (ntsb_id,),
            ).fetchone()
            return row["download_status"] if row else None

    def has_text_hash(self, text_hash: str) -> str | None:
        """Check if a text hash already exists. Returns ntsb_id if found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ntsb_id FROM processed_reports WHERE text_hash = ?",
                (text_hash,),
            ).fetchone()
            return row["ntsb_id"] if row else None

    # ── Batch Queries ────────────────────────────────────────────────────

    def get_pending_downloads(self) -> list[str]:
        """Get ntsb_ids that haven't been downloaded yet."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ntsb_id FROM processed_reports WHERE download_status = 'pending'"
            ).fetchall()
            return [r["ntsb_id"] for r in rows]

    def get_pending_extractions(self) -> list[str]:
        """Get ntsb_ids downloaded but not yet extracted."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ntsb_id FROM processed_reports "
                "WHERE download_status = 'success' AND extract_status = 'pending'"
            ).fetchall()
            return [r["ntsb_id"] for r in rows]

    def get_stats(self) -> dict[str, int]:
        """Get counts grouped by extraction status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT extract_status, COUNT(*) as cnt "
                "FROM processed_reports GROUP BY extract_status"
            ).fetchall()
            return {r["extract_status"]: r["cnt"] for r in rows}

    # ── Mutations ────────────────────────────────────────────────────────

    def register(self, ntsb_id: str) -> None:
        """Register a report for tracking (idempotent)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_reports (ntsb_id) VALUES (?)",
                (ntsb_id,),
            )

    def register_batch(self, ntsb_ids: list[str]) -> None:
        """Register multiple reports for tracking."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO processed_reports (ntsb_id) VALUES (?)",
                [(nid,) for nid in ntsb_ids],
            )

    def update_download(
        self, ntsb_id: str, pdf_path: str, status: str
    ) -> None:
        """Record the result of a PDF download attempt."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE processed_reports "
                "SET pdf_path = ?, download_status = ?, "
                "    download_date = datetime('now'), updated_at = datetime('now') "
                "WHERE ntsb_id = ?",
                (pdf_path, status, ntsb_id),
            )

    def update_extraction(
        self,
        ntsb_id: str,
        status: ExtractionStatus,
        word_count: int = 0,
        text_hash: str | None = None,
    ) -> None:
        """Record the result of a text extraction attempt."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE processed_reports "
                "SET extract_status = ?, word_count = ?, text_hash = ?, "
                "    updated_at = datetime('now') "
                "WHERE ntsb_id = ?",
                (status.value, word_count, text_hash, ntsb_id),
            )

    def get_pdf_path(self, ntsb_id: str) -> str | None:
        """Get the stored PDF path for a report."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pdf_path FROM processed_reports WHERE ntsb_id = ?",
                (ntsb_id,),
            ).fetchone()
            return row["pdf_path"] if row else None
