"""FAISS vector store with local disk persistence per video ID."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
from config.settings import get_settings
from ingestion.chunker import TranscriptChunk
from utils.logging_utils import get_logger

logger = get_logger("faiss_store")


class FAISSVectorStore:
    """Manages an isolated FAISS index and chunk metadata for a specific YouTube video."""

    def __init__(self, video_id: str, base_dir: Optional[Path] = None):
        self.video_id = video_id
        settings = get_settings()
        self.store_dir = (base_dir or settings.vectorstores_dir) / video_id
        self.index_path = self.store_dir / "index.faiss"
        self.metadata_path = self.store_dir / "chunks.json"

        self.index: Optional[faiss.Index] = None
        self.chunks: List[TranscriptChunk] = []

    def exists(self) -> bool:
        """Check if both the FAISS index and chunk metadata exist on disk."""
        return self.index_path.exists() and self.metadata_path.exists()

    def build_and_save(self, chunks: List[TranscriptChunk], embeddings: np.ndarray) -> None:
        """Create a new FAISS index from embeddings and persist to disk.

        Args:
            chunks: List of TranscriptChunk objects.
            embeddings: 2D numpy array of shape (N, dimension).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        if len(chunks) == 0:
            raise ValueError("Cannot build FAISS index from 0 chunks.")

        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP (Cosine) with {len(chunks)} vectors of dim {dimension} for {self.video_id}")

        # Using Inner Product on normalized vectors gives Cosine Similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))

        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Write FAISS index
        faiss.write_index(index, str(self.index_path))

        # Write chunks metadata
        chunks_data = [chunk.to_dict() for chunk in chunks]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        self.index = index
        self.chunks = chunks
        logger.info(f"Successfully saved FAISS index and metadata to {self.store_dir}")

    def load(self) -> bool:
        """Load the persisted FAISS index and chunks metadata from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.exists():
            return False

        try:
            logger.info(f"Loading cached FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))

            with open(self.metadata_path, "r", encoding="utf-8") as f:
                chunks_raw = json.load(f)

            self.chunks = [
                TranscriptChunk(
                    chunk_id=c["chunk_id"],
                    video_id=c["video_id"],
                    video_url=c["video_url"],
                    text=c["text"],
                    start_timestamp=c["start_timestamp"],
                    end_timestamp=c["end_timestamp"],
                    start_time_formatted=c["start_time_formatted"],
                    end_time_formatted=c["end_time_formatted"],
                    timestamp_url=c["timestamp_url"],
                )
                for c in chunks_raw
            ]
            logger.info(f"Loaded {len(self.chunks)} chunks for video {self.video_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading cached FAISS index for {self.video_id}: {e}")
            return False

    def similarity_search_by_vector(
        self, query_vector: np.ndarray, top_k: int = 4
    ) -> List[Tuple[TranscriptChunk, float]]:
        """Perform semantic similarity search using a normalized query vector.

        Args:
            query_vector: 1D or 2D numpy array representing query embedding.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of tuples: (TranscriptChunk, similarity_score).
        """
        if self.index is None:
            if not self.load():
                raise RuntimeError(f"FAISS index for video {self.video_id} is not loaded.")

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Ensure top_k does not exceed available chunks
        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []

        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results: List[Tuple[TranscriptChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
