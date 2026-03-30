#!/usr/bin/env python3
"""
Regenerate section-aware chunks with fixed metadata from markdown files.
This script uses the updated chunk_markdown_section_aware() function that now
enriches chunks with ntsb_no, event_date, make, model, and other CSV metadata.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_prep.chunking import chunk_markdown_section_aware

def regenerate_section_chunks(output_path=None):
    """Regenerate section-aware chunks from markdown files."""
    if output_path is None:
        output_path = project_root / "data" / "processed" / "chunks_md_section.json"
    
    md_dir = project_root / "dataset-pipeline" / "data" / "extracted" / "extracted"
    
    if not md_dir.exists():
        print(f"ERROR: Markdown directory not found: {md_dir}")
        return False
    
    # Get all markdown files
    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        print(f"ERROR: No markdown files found in {md_dir}")
        return False
    
    print(f"Found {len(md_files)} markdown files")
    print(f"Regenerating section-aware chunks with metadata...\n")
    
    all_chunks = []
    
    for i, md_file in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] Processing {md_file.name}...", end=" ", flush=True)
        try:
            chunks = chunk_markdown_section_aware(str(md_file))
            all_chunks.extend(chunks)
            print(f"✓ {len(chunks)} chunks")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return False
    
    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Total: {len(all_chunks)} chunks generated")
    print(f"✓ Saved to: {output_path}")
    
    # Quick validation: check a few chunks for metadata
    if all_chunks:
        sample_chunk = all_chunks[0]
        print("\nSample chunk structure:")
        for key in ["chunk_id", "report_id", "ntsb_no", "event_date", "make", "model", "section_title"]:
            val = sample_chunk.get(key, "N/A")
            print(f"  {key}: {val}")
    
    return True

if __name__ == "__main__":
    success = regenerate_section_chunks()
    sys.exit(0 if success else 1)
