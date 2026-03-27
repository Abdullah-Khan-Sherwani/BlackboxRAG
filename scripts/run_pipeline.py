import os
import argparse
import logging
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from src.data_prep.chunking import chunk_markdown_section_aware, chunk_markdown_recursive
from src.data_prep.context_generator import contextualize_chunks
from src.llm.ollama_client import is_ollama_running

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end NTSB RAG pipeline.")
    parser.add_argument("--test-batch", type=int, default=0, help="Number of files to process. 0 for all.")
    parser.add_argument("--chunk-strategy", type=str, choices=['section', 'recursive'], default='section', help="Chunking strategy to use (A or B)")
    parser.add_argument("--skip-context", action="store_true", help="Skip Context Generator (LLM calls)")
    parser.add_argument("--skip-upsert", action="store_true", help="Skip Pinecone Embedding Upsert")
    args = parser.parse_args()

    # 1. Check Ollama
    if not args.skip_context and not is_ollama_running():
        logger.error("Ollama is not running. Context generation requires Ollama!")
        sys.exit(1)

    data_dir = project_root / "dataset-pipeline" / "data" / "extracted" / "extracted"
    md_files = list(data_dir.glob("*.md"))
    
    if args.test_batch > 0:
        md_files = md_files[:args.test_batch]
        logger.info(f"Running TEST MODE on {args.test_batch} files.")
    
    logger.info(f"Loaded {len(md_files)} NTSB reports.")

    all_chunks = []
    
    # 2. Chunking
    logger.info(f"Starting chunking ({args.chunk_strategy} strategy)...")
    for md_file in md_files:
        if args.chunk_strategy == 'section':
            chunks = chunk_markdown_section_aware(str(md_file))
        else:
            chunks = chunk_markdown_recursive(str(md_file))
        all_chunks.extend(chunks)
        
    logger.info(f"Total chunks generated: {len(all_chunks)}")

    # 3. Contextual Retrieval Generation
    if not args.skip_context:
        logger.info("Starting LLM Context Generation...")
        # Since contextualize_chunks needs the WHOLE_DOCUMENT, we need to regroup
        # Alternatively, we iterate through files here and contextualize their chunks.
        
        contextualized_chunks = []
        for md_file in md_files:
            report_id = md_file.stem
            # Get chunks specific to this report
            report_chunks = [c for c in all_chunks if c.get('report_id') == report_id]
            
            if not report_chunks:
                continue
                
            with open(md_file, 'r', encoding='utf-8') as f:
                whole_doc = f.read()
                
            # We don't want to pass a 10,000 word document into Qwen.
            # Truncate the whole doc to max 4000 words.
            words = whole_doc.split()
            truncated_doc = " ".join(words[:4000]) if len(words) > 4000 else whole_doc
            
            logger.info(f"Contextualizing {len(report_chunks)} chunks for {report_id}...")
            ctx_chunks = contextualize_chunks(report_chunks, truncated_doc)
            contextualized_chunks.extend(ctx_chunks)
            
        all_chunks = contextualized_chunks
    else:
        logger.info("Skipping Context Generation.")

    # 4. Save intermediate output
    out_path = project_root / "data" / "processed" / "final_chunks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)
        
    logger.info(f"Saved {len(all_chunks)} final chunks to {out_path}")

    # 5. Embed & Upsert (Placeholder)
    if not args.skip_upsert:
        logger.info("Starting Embedding & Pinecone Upsert...")
        # TODO: Import embeddings.py and invoke upsert logic using all_chunks
        logger.warning("Upsert logic not fully implemented yet.")

    logger.info("Pipeline Execution Finished.")

if __name__ == "__main__":
    main()
