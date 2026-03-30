#!/usr/bin/env python3
"""
Compare: What chunks SHOULD be retrieved vs WHAT IS being retrieved
"""

import json

chunks_file = "data/processed/chunks_md_section.json"
with open(chunks_file, 'r') as f:
    chunks = json.load(f)

asiana_chunks = [c for c in chunks if c.get("ntsb_no") == "NTSB/AAR-14/01"]

# What we NEED: Chunks with the specific ARFF incident details
needed_sections = ["1.15.4 Emergency Response", "Triage of Passenger 41E", "1.15.4.2 Firefighting Tactics"]

print("="*90)
print("CHUNKS WE NEED FOR THE ANSWER:")
print("="*90)

for section_name in needed_sections:
    matching = [c for c in asiana_chunks if section_name in c.get("section_title", "")]
    if matching:
        print(f"\n{section_name}: {len(matching)} chunk(s)")
        for chunk in matching:
            text = chunk.get("text", "")
            # Look for the key information
            if "foam" in text.lower() or "rescue 37" in text.lower() or "41e" in text.lower():
                print(f"\n  KEY CHUNK FOUND: {chunk.get('chunk_id')}")
                print(f"  Text snippet: {text[:500]}...")

# What we're currently getting
print("\n\n" + "="*90)
print("WHAT GENERIC SECTIONS ARE BEING RETRIEVED (WRONG):")
print("="*90)

wrong_sections = ["1.1 History of Flight", "Introduction", "Executive Summary", "Introduction/Header"]
for section_name in wrong_sections:
    matching = [c for c in asiana_chunks if section_name in c.get("section_title", "")]
    if matching:
        print(f"\n{section_name}: {len(matching)} chunk(s)")
        print(f"  (These are too generic - they talk about flight basics, not ARFF incident)")

print("\n" + "="*90)
print("ROOT CAUSE: Semantic search is matching general 'Asiana Flight 214' terms")
print("but not drilling down to SPECIFIC sections about the ARFF response.")
print("="*90)
