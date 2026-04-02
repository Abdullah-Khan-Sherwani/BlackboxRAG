"""
Enrich section chunks with report-level metadata from md_recursive chunks.

Reads  : data/processed/chunks_md_section.json
         data/processed/chunks_md_md_recursive.json
Writes : data/processed/chunks_md_section_enriched.json

Only report-level fields are joined (ntsb_no, event_date, make, model, state,
phase_of_flight, weather, source_filename, entity_id, context_summary).
Chunk-level fields (entities, aircraft_components, numerics) are intentionally
excluded — they are specific to each md_recursive chunk's text and would be
wrong if copied to a different chunk boundary.
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECTION_FILE  = BASE_DIR / "data" / "processed" / "chunks_md_section.json"
MDREC_FILE    = BASE_DIR / "data" / "processed" / "chunks_md_md_recursive.json"
OUT_FILE      = BASE_DIR / "data" / "processed" / "chunks_md_section_enriched.json"

REPORT_LEVEL_FIELDS = [
    "ntsb_no",
    "event_date",
    "make",
    "model",
    "state",
    "phase_of_flight",
    "weather",
    "source_filename",
    "entity_id",
    "context_summary",
]


def build_report_meta(mdrec_chunks: list[dict]) -> dict[str, dict]:
    """Build report_id -> metadata dict using the first chunk seen per report."""
    meta = {}
    for chunk in mdrec_chunks:
        rid = chunk.get("report_id")
        if rid and rid not in meta:
            meta[rid] = {f: chunk.get(f, "") for f in REPORT_LEVEL_FIELDS}
    return meta


def enrich(section_chunks: list[dict], report_meta: dict[str, dict]) -> list[dict]:
    enriched = []
    missing = set()
    for chunk in section_chunks:
        rid = chunk.get("report_id", "")
        meta = report_meta.get(rid)
        if meta is None:
            missing.add(rid)
            enriched.append(chunk)
            continue
        enriched_chunk = dict(chunk)
        enriched_chunk.update(meta)
        enriched.append(enriched_chunk)

    if missing:
        print(f"  WARNING: {len(missing)} report(s) had no metadata match: {sorted(missing)}")
    return enriched


def main():
    print("Loading section chunks...")
    with open(SECTION_FILE, "r", encoding="utf-8") as f:
        section_chunks = json.load(f)
    print(f"  {len(section_chunks)} section chunks loaded")

    print("Loading md_recursive chunks for metadata...")
    with open(MDREC_FILE, "r", encoding="utf-8") as f:
        mdrec_chunks = json.load(f)
    print(f"  {len(mdrec_chunks)} md_recursive chunks loaded")

    report_meta = build_report_meta(mdrec_chunks)
    print(f"  Built metadata lookup for {len(report_meta)} reports")

    print("Enriching section chunks...")
    enriched = enrich(section_chunks, report_meta)

    # Verify enrichment
    has_ntsb = sum(1 for c in enriched if c.get("ntsb_no"))
    print(f"  {has_ntsb}/{len(enriched)} chunks now have ntsb_no")

    print(f"Writing enriched chunks to {OUT_FILE.name}...")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"  Done — {OUT_FILE}")


if __name__ == "__main__":
    main()
