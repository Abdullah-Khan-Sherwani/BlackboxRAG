"""
Generate embeddings for chunks.
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

<<<<<<< HEAD
=======
CHUNK_FILES = {
    'fixed': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_fixed.json'),
    'recursive': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_recursive.json'),
    'semantic': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_semantic.json'),
    'parent': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_parent.json'),
}

OUTPUT_FILES = {
    'fixed': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_fixed.npz'),
    'recursive': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_recursive.npz'),
    'semantic': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_semantic.npz'),
    'parent': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_parent.npz'),
}


>>>>>>> bf61da3 (feat: implement parent chunking, diversity tuning, and network resilience)

CHUNK_FILES = {
    'fixed': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_fixed.json'),
    'recursive': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_recursive.json'),
    'semantic': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_semantic.json'),
    'parent': os.path.join(BASE_DIR, 'data', 'processed', 'chunks_parent.json'),
}

OUTPUT_FILES = {
    'fixed': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_fixed.npz'),
    'recursive': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_recursive.npz'),
    'semantic': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_semantic.npz'),
    'parent': os.path.join(BASE_DIR, 'data', 'processed', 'embeddings_parent.npz'),
}


def load_model():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    model.to(device)
    
    return model

def embed_chunks(chunks: list[dict], model=None) -> tuple[list[dict], np.ndarray]:
    """
    Take a list of chunks and return embeddings.
    Also returns the chunks, ensuring consistency.
    """
    if not model:
        model = load_model()

<<<<<<< HEAD
def embed_chunks(chunks: list[dict], model=None) -> tuple[list[dict], np.ndarray]:
    """
    Take a list of chunks and return embeddings.
    Also returns the chunks, ensuring consistency.
    """
    if not model:
        model = load_model()
=======
    for strategy, chunk_path in CHUNK_FILES.items():
        output_path = OUTPUT_FILES[strategy]

        if not os.path.exists(chunk_path):
            print(f"Skipping {strategy} - {os.path.basename(chunk_path)} not found.")
            continue
>>>>>>> bf61da3 (feat: implement parent chunking, diversity tuning, and network resilience)
        
    texts = [c.get('contextualized_text', c.get('text', '')) for c in chunks]
    
    all_embeddings = []
    print(f"Starting embedding generation for {len(texts)} chunks...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        # SentenceTransformer encode
        emb = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    return chunks, embeddings

