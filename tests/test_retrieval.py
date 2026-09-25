"""Tests for FAISS vector store persistence, semantic retrieval, and RAG QA chain."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from ingestion.chunker import TranscriptChunk
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import TranscriptRetriever
from rag.qa_chain import RAGQAChain
from llm.llama_service import MockLlamaService


@pytest.fixture
def sample_chunks():
    return [
        TranscriptChunk(
            chunk_id=0,
            video_id="vid12345678",
            video_url="https://www.youtube.com/watch?v=vid12345678",
            text="First chunk discussing neural networks and backpropagation.",
            start_timestamp=0.0,
            end_timestamp=15.0,
            start_time_formatted="00:00",
            end_time_formatted="00:15",
            timestamp_url="https://www.youtube.com/watch?v=vid12345678&t=0s",
        ),
        TranscriptChunk(
            chunk_id=1,
            video_id="vid12345678",
            video_url="https://www.youtube.com/watch?v=vid12345678",
            text="Second chunk explaining attention mechanisms and transformer architecture.",
            start_timestamp=15.0,
            end_timestamp=35.0,
            start_time_formatted="00:15",
            end_time_formatted="00:35",
            timestamp_url="https://www.youtube.com/watch?v=vid12345678&t=15s",
        ),
    ]


class TestRetrievalAndFAISS:
    """Test suite for FAISS vector indexing, similarity search, and QA chaining."""

    def test_faiss_build_save_and_load(self, tmp_path: Path, sample_chunks):
        store = FAISSVectorStore(video_id="vid12345678", base_dir=tmp_path)
        assert not store.exists()

        # Create two orthogonal normalized vectors (dim=4)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        store.build_and_save(sample_chunks, embeddings)
        assert store.exists()

        # Load store from fresh instance
        new_store = FAISSVectorStore(video_id="vid12345678", base_dir=tmp_path)
        assert new_store.load() is True
        assert len(new_store.chunks) == 2

        # Search with vector close to chunk 0
        query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = new_store.similarity_search_by_vector(query_vec, top_k=2)

        assert len(results) == 2
        # Best match should be chunk 0
        top_chunk, score = results[0]
        assert top_chunk.chunk_id == 0
        assert "neural networks" in top_chunk.text
        assert score > 0.8

    def test_rag_qa_chain_with_mock_llm(self, tmp_path: Path, sample_chunks):
        store = FAISSVectorStore(video_id="vid12345678", base_dir=tmp_path)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        store.build_and_save(sample_chunks, embeddings)

        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_query.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        retriever = TranscriptRetriever(
            vectorstore=store,
            embedding_service=mock_embedding_service,
            top_k=2,
        )

        mock_llm = MockLlamaService()
        chain = RAGQAChain(retriever=retriever, llm_service=mock_llm)

        res = chain.answer_question("What is backpropagation?")
        assert "answer" in res
        assert "sources" in res
        assert len(res["sources"]) == 2
        assert res["sources"][0]["start_time_formatted"] == "00:00"
        assert "timestamp_url" in res["sources"][0]
