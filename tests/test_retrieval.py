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

    def test_candidate_expansion_and_lexical_fallback(self, tmp_path: Path):
        """Test that a relevant chunk outside top-2 FAISS results is discovered via candidate expansion & lexical fallback."""
        chunks = [
            TranscriptChunk(
                chunk_id=0,
                video_id="vid_test",
                video_url="https://www.youtube.com/watch?v=vid_test",
                text="Chunk A: Some general discussion about data structures and memory.",
                start_timestamp=0.0,
                end_timestamp=120.0,
                start_time_formatted="00:00",
                end_time_formatted="02:00",
                timestamp_url="https://www.youtube.com/watch?v=vid_test&t=0s",
            ),
            TranscriptChunk(
                chunk_id=1,
                video_id="vid_test",
                video_url="https://www.youtube.com/watch?v=vid_test",
                text="Chunk B: Overview of standard sorting algorithms and quicksort.",
                start_timestamp=300.0,
                end_timestamp=420.0,
                start_time_formatted="05:00",
                end_time_formatted="07:00",
                timestamp_url="https://www.youtube.com/watch?v=vid_test&t=300s",
            ),
            TranscriptChunk(
                chunk_id=2,
                video_id="vid_test",
                video_url="https://www.youtube.com/watch?v=vid_test",
                text="Chunk C: Another unrelated concept about graph adjacency matrices.",
                start_timestamp=720.0,
                end_timestamp=840.0,
                start_time_formatted="12:00",
                end_time_formatted="14:00",
                timestamp_url="https://www.youtube.com/watch?v=vid_test&t=720s",
            ),
            TranscriptChunk(
                chunk_id=3,
                video_id="vid_test",
                video_url="https://www.youtube.com/watch?v=vid_test",
                text="Chunk D: The speaker explains suffix arrays and how suffix indices are stored in sorted order.",
                start_timestamp=1965.0,
                end_timestamp=2085.0,
                start_time_formatted="32:45",
                end_time_formatted="34:45",
                timestamp_url="https://www.youtube.com/watch?v=vid_test&t=1965s",
            ),
        ]

        store = FAISSVectorStore(video_id="vid_test", base_dir=tmp_path)
        # Create vectors where chunk 0 and 1 are closest to query vector, but chunk 3 contains exact keyword
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.4, 0.4, 0.0, 0.0],
        ], dtype=np.float32)
        store.build_and_save(chunks, embeddings)

        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_query.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Set top_k=2 but candidates_k=4
        retriever = TranscriptRetriever(
            vectorstore=store,
            embedding_service=mock_embedding_service,
            top_k=2,
            candidates_k=4,
        )

        final_evidence, all_candidates = retriever.retrieve_candidates("What does the speaker explain about suffix arrays?")

        # Assert all 4 candidates were evaluated across the video
        assert len(all_candidates) == 4
        # Assert final evidence is capped at top_k=2
        assert len(final_evidence) == 2
        # Assert Chunk 3 (at 32:45) is ranked #1 in final evidence due to lexical match
        top_chunk, score = final_evidence[0]
        assert top_chunk.chunk_id == 3
        assert top_chunk.start_time_formatted == "32:45"
        assert "suffix arrays" in top_chunk.text

    def test_exact_keyword_query(self, tmp_path: Path):
        """Test that exact keyword queries locate the target chunk anywhere in the transcript."""
        chunks = [
            TranscriptChunk(
                chunk_id=0,
                video_id="vid_kw",
                video_url="https://www.youtube.com/watch?v=vid_kw",
                text="General talk about neural network layers and weights.",
                start_timestamp=0.0,
                end_timestamp=100.0,
                start_time_formatted="00:00",
                end_time_formatted="01:40",
                timestamp_url="https://www.youtube.com/watch?v=vid_kw&t=0s",
            ),
            TranscriptChunk(
                chunk_id=1,
                video_id="vid_kw",
                video_url="https://www.youtube.com/watch?v=vid_kw",
                text="The activation function used is ReLU, rectified linear unit.",
                start_timestamp=600.0,
                end_timestamp=700.0,
                start_time_formatted="10:00",
                end_time_formatted="11:40",
                timestamp_url="https://www.youtube.com/watch?v=vid_kw&t=600s",
            ),
        ]
        store = FAISSVectorStore(video_id="vid_kw", base_dir=tmp_path)
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.build_and_save(chunks, embeddings)

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.array([0.5, 0.5], dtype=np.float32)
        retriever = TranscriptRetriever(vectorstore=store, embedding_service=mock_emb, top_k=1)

        results = retriever.retrieve("What is ReLU?")
        assert len(results) == 1
        assert results[0][0].chunk_id == 1
        assert "ReLU" in results[0][0].text

    def test_semantic_query_without_exact_keywords(self, tmp_path: Path):
        """Test purely semantic query retrieval without exact word overlap."""
        chunks = [
            TranscriptChunk(
                chunk_id=0,
                video_id="vid_sem",
                video_url="https://www.youtube.com/watch?v=vid_sem",
                text="Optimizing cost with gradient descent.",
                start_timestamp=0.0,
                end_timestamp=60.0,
                start_time_formatted="00:00",
                end_time_formatted="01:00",
                timestamp_url="https://www.youtube.com/watch?v=vid_sem&t=0s",
            ),
            TranscriptChunk(
                chunk_id=1,
                video_id="vid_sem",
                video_url="https://www.youtube.com/watch?v=vid_sem",
                text="Linear algebra operations and vector transformations in high dimensions.",
                start_timestamp=120.0,
                end_timestamp=180.0,
                start_time_formatted="02:00",
                end_time_formatted="03:00",
                timestamp_url="https://www.youtube.com/watch?v=vid_sem&t=120s",
            ),
        ]
        store = FAISSVectorStore(video_id="vid_sem", base_dir=tmp_path)
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.build_and_save(chunks, embeddings)

        mock_emb = MagicMock()
        # Query vector aligned with chunk 1
        mock_emb.embed_query.return_value = np.array([0.1, 0.99], dtype=np.float32)
        retriever = TranscriptRetriever(vectorstore=store, embedding_service=mock_emb, top_k=1)

        results = retriever.retrieve("Matrix math and dimensional mappings")
        assert len(results) == 1
        assert results[0][0].chunk_id == 1

    def test_unsupported_question_refusal_and_no_sources(self):
        """Test unsupported query refuses without fabricating sources."""
        mock_retriever = MagicMock()
        mock_chunk = TranscriptChunk(
            chunk_id=0,
            video_id="v",
            video_url="http://y",
            text="General computing context.",
            start_timestamp=0.0,
            end_timestamp=10.0,
            start_time_formatted="00:00",
            end_time_formatted="00:10",
            timestamp_url="http://y&t=0s",
        )
        mock_retriever.retrieve.return_value = [(mock_chunk, 0.25)]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "The available video transcript does not provide enough information to answer this question."
        )

        chain = RAGQAChain(retriever=mock_retriever, llm_service=mock_llm)
        res = chain.answer_question("Explain quantum entanglement")

        assert res["is_supported"] is False
        assert res["sources"] == []
        assert len(res["retrieved_context"]) == 1

    def test_timestamp_preservation_throughout_pipeline(self, tmp_path: Path):
        """Test that timestamp metadata survives two-stage candidate retrieval and scoring."""
        chunks = [
            TranscriptChunk(
                chunk_id=0,
                video_id="vid_ts",
                video_url="https://www.youtube.com/watch?v=vid_ts",
                text="Information deep in the video at thirty-two minutes forty-five seconds.",
                start_timestamp=1965.0,
                end_timestamp=2025.0,
                start_time_formatted="32:45",
                end_time_formatted="33:45",
                timestamp_url="https://www.youtube.com/watch?v=vid_ts&t=1965s",
            )
        ]
        store = FAISSVectorStore(video_id="vid_ts", base_dir=tmp_path)
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        store.build_and_save(chunks, embeddings)

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.array([1.0, 0.0], dtype=np.float32)
        retriever = TranscriptRetriever(vectorstore=store, embedding_service=mock_emb, top_k=1)

        final_evidence, candidates = retriever.retrieve_candidates("thirty-two minutes")
        assert len(final_evidence) == 1
        chunk = final_evidence[0][0]
        assert chunk.start_timestamp == 1965.0
        assert chunk.end_timestamp == 2025.0
        assert chunk.start_time_formatted == "32:45"
        assert chunk.end_time_formatted == "33:45"
        assert chunk.timestamp_url == "https://www.youtube.com/watch?v=vid_ts&t=1965s"

    def test_duplicate_handling_between_semantic_and_lexical(self, tmp_path: Path):
        """Test that chunks appearing in both semantic and lexical candidate sets are not duplicated."""
        chunks = [
            TranscriptChunk(
                chunk_id=0,
                video_id="vid_dup",
                video_url="https://www.youtube.com/watch?v=vid_dup",
                text="Exact keyword and high semantic similarity for transformer attention.",
                start_timestamp=0.0,
                end_timestamp=30.0,
                start_time_formatted="00:00",
                end_time_formatted="00:30",
                timestamp_url="https://www.youtube.com/watch?v=vid_dup&t=0s",
            )
        ]
        store = FAISSVectorStore(video_id="vid_dup", base_dir=tmp_path)
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        store.build_and_save(chunks, embeddings)

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.array([1.0, 0.0], dtype=np.float32)
        retriever = TranscriptRetriever(vectorstore=store, embedding_service=mock_emb, top_k=4)

        final_evidence, candidates = retriever.retrieve_candidates("transformer attention")
        # Candidate IDs must be unique
        candidate_ids = [c[0].chunk_id for c in candidates]
        assert len(candidate_ids) == len(set(candidate_ids))
        assert len(final_evidence) == 1

    def test_regression_simulated_failure_chunk_b_discovered(self, tmp_path: Path):
        """Regression test for user scenario:
        Transcript contains Chunk A, Chunk B ('suffix arrays' at 32:45), Chunk C.
        Initial pure top-k=2 without candidate expansion would only return Chunk A and C.
        With candidate expansion and lexical fallback, Chunk B is discovered, is_supported=True,
        and Chunk B is cited in sources.
        """
        chunk_a = TranscriptChunk(
            chunk_id=0,
            video_id="vid_reg",
            video_url="https://www.youtube.com/watch?v=vid_reg",
            text="Chunk A: Some general discussion...",
            start_timestamp=0.0,
            end_timestamp=60.0,
            start_time_formatted="00:00",
            end_time_formatted="01:00",
            timestamp_url="https://www.youtube.com/watch?v=vid_reg&t=0s",
        )
        chunk_b = TranscriptChunk(
            chunk_id=1,
            video_id="vid_reg",
            video_url="https://www.youtube.com/watch?v=vid_reg",
            text="Chunk B: The speaker explains suffix arrays and how suffix indices are stored...",
            start_timestamp=1965.0,
            end_timestamp=2025.0,
            start_time_formatted="32:45",
            end_time_formatted="33:45",
            timestamp_url="https://www.youtube.com/watch?v=vid_reg&t=1965s",
        )
        chunk_c = TranscriptChunk(
            chunk_id=2,
            video_id="vid_reg",
            video_url="https://www.youtube.com/watch?v=vid_reg",
            text="Chunk C: Another unrelated concept...",
            start_timestamp=600.0,
            end_timestamp=660.0,
            start_time_formatted="10:00",
            end_time_formatted="11:00",
            timestamp_url="https://www.youtube.com/watch?v=vid_reg&t=600s",
        )

        store = FAISSVectorStore(video_id="vid_reg", base_dir=tmp_path)
        # Vector configuration where FAISS alone ranks Chunk A (idx 0) and Chunk C (idx 2) highest
        embeddings = np.array([
            [0.9, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ], dtype=np.float32)
        store.build_and_save([chunk_a, chunk_b, chunk_c], embeddings)

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # candidates_k=3 searches the full transcript, top_k=2 limits final evidence
        retriever = TranscriptRetriever(
            vectorstore=store,
            embedding_service=mock_emb,
            top_k=2,
            candidates_k=3,
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "Suffix arrays store sorted suffix indices to allow fast binary search across strings."
        )

        chain = RAGQAChain(retriever=retriever, llm_service=mock_llm)
        response = chain.answer_question("What does the speaker explain about suffix arrays?")

        assert response["is_supported"] is True
        assert len(response["sources"]) <= 2
        # Chunk B must be discovered and present in the sources
        source_chunk_ids = [s["chunk_id"] for s in response["sources"]]
        assert 1 in source_chunk_ids
        # Verify timestamp 32:45 is in sources
        timestamps = [s["start_time_formatted"] for s in response["sources"]]
        assert "32:45" in timestamps
