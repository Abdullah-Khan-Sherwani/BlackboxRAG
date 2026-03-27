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

