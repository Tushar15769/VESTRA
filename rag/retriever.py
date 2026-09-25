"""Semantic retriever querying FAISS vector stores."""

from typing import List, Optional, Tuple
from embeddings.embedding_service import EmbeddingService, get_embedding_service
from ingestion.chunker import TranscriptChunk
from vectorstore.faiss_store import FAISSVectorStore
from utils.logging_utils import get_logger

logger = get_logger("retriever")


class TranscriptRetriever:
    """Retrieves top-k relevant transcript chunks using FAISS similarity search."""

    def __init__(
        self,
        vectorstore: FAISSVectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = 4,
    ):
        self.vectorstore = vectorstore
        self.embedding_service = embedding_service or get_embedding_service()
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Tuple[TranscriptChunk, float]]:
        """Embed the query and retrieve the top-k most similar chunks.

        Args:
            query: User's question or search query.

        Returns:
            List of (TranscriptChunk, similarity_score) sorted by relevance.
        """
        logger.info(f"Retrieving top {self.top_k} chunks for query: '{query}'")
        query_vector = self.embedding_service.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(query_vector, top_k=self.top_k)
        logger.info(f"Retrieved {len(results)} chunks (top score: {results[0][1]:.4f})" if results else "No chunks found.")
        return results
