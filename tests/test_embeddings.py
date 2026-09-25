"""Tests for Sentence Transformers embedding service."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from embeddings.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite covering embedding dimensionality and normalization."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_mocked_embedding_service(self, mock_transformer_cls):
        mock_model = MagicMock()
        mock_transformer_cls.return_value = mock_model

        # Mock encode returns 384-dimensional normalized vectors
        dummy_vectors = np.random.randn(2, 384).astype(np.float32)
        dummy_vectors /= np.linalg.norm(dummy_vectors, axis=1, keepdims=True)
        mock_model.encode.return_value = dummy_vectors

        service = EmbeddingService(model_name="mock-model")
        assert service.dimension == 384

        texts = ["What is machine learning?", "How do transformers work?"]
        embeddings = service.embed_documents(texts)

        assert embeddings.shape == (2, 384)
        # Check normalization (norm should be ~1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_query(self, mock_transformer_cls):
        mock_model = MagicMock()
        mock_transformer_cls.return_value = mock_model
        dummy_vector = np.random.randn(1, 384).astype(np.float32)
        mock_model.encode.return_value = dummy_vector

        service = EmbeddingService(model_name="mock-model")
        query_vec = service.embed_query("Sample question")

        assert query_vec.shape == (384,)

    def test_empty_query_raises_error(self):
        service = EmbeddingService(model_name="mock-model")
        with pytest.raises(ValueError):
            service.embed_query("")
