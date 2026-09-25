"""Singleton Sentence-Transformers embedding service."""

from functools import lru_cache
from typing import List, Optional
import numpy as np
from config.settings import get_settings
from utils.logging_utils import get_logger

logger = get_logger("embedding_service")


class EmbeddingService:
    """Manages SentenceTransformer embedding model loading, caching, and batch inference."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self._model = None
        self._dimension: Optional[int] = None
        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")

    @property
    def model(self):
        """Lazy load the sentence transformer model once."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' into memory...")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # Determine vector dimension
            test_vec = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = int(test_vec.shape[1])
            logger.info(f"Model '{self.model_name}' successfully loaded. Embedding dimension: {self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the embedding model."""
        if self._dimension is None:
            _ = self.model  # Trigger load
        return self._dimension or 384

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized embeddings for a list of document strings.

        Args:
            texts: List of text chunks to embed.
            batch_size: Batch size for model inference.

        Returns:
            2D numpy array of shape (len(texts), dimension), normalized for cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.info(f"Generating embeddings for {len(texts)} chunks (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalizing enables cosine similarity with Inner Product index
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized embedding for a single user query.

        Args:
            query: The user question or search string.

        Returns:
            1D numpy array of shape (dimension,), normalized.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return a cached singleton EmbeddingService instance."""
    return EmbeddingService()
