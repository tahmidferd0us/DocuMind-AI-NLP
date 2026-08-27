import numpy as np
from typing import List, Dict, Any, Tuple
from .splitter import DocumentChunk
from ..config import EMBEDDING_MODEL, logger

_embedder = None

def get_embedder():
    """Singleton loader for local sentence-transformers embedding model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {EMBEDDING_MODEL} (runs on CPU, 0 API quota)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

class VectorStore:
    """In-memory FAISS vector index with local embeddings for fast, quota-free semantic retrieval."""
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        self.model_name = embedding_model_name
        self.chunks: List[DocumentChunk] = []
        self.index = None
        self.dimension = None

    def build_index(self, chunks: List[DocumentChunk]):
        """Embeds all document chunks and builds a FAISS index."""
        if not chunks:
            self.chunks = []
            self.index = None
            return

        import faiss
        self.chunks = chunks
        embedder = get_embedder()
        
        texts = [chunk.content for chunk in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        self.dimension = embeddings.shape[1]
        
        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {len(chunks)} chunks, embedding dim={self.dimension}")

    def search(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """Performs cosine similarity search and returns matching chunks with similarity scores."""
        if self.index is None or not self.chunks:
            return []

        embedder = get_embedder()
        query_embedding = embedder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
