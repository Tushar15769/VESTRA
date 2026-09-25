"""Tests for strict grounding directives, natural answer formatting, summary generation, and unsupported question UX."""

import pytest
from ingestion.chunker import TranscriptChunk
from rag.prompt import RAG_SYSTEM_PROMPT, build_summary_prompt, build_user_prompt
from rag.qa_chain import RAGQAChain, is_unsupported_answer
from llm.llama_service import MockLlamaService
from unittest.mock import MagicMock


@pytest.fixture
def mock_chunk():
    return TranscriptChunk(
        chunk_id=0,
        video_id="aircAruvnKk",
        video_url="https://www.youtube.com/watch?v=aircAruvnKk",
        text="A neuron is simply a thing that holds a number between 0 and 1, called its activation.",
        start_timestamp=157.0,
        end_timestamp=222.0,
        start_time_formatted="02:37",
        end_time_formatted="03:42",
        timestamp_url="https://www.youtube.com/watch?v=aircAruvnKk&t=157s",
    )


class TestGroundingDirectives:
    """Test suite ensuring strict grounding prompt directives and natural responses are enforced."""

    def test_system_prompt_grounding_rules(self):
        assert "SINGLE SOURCE OF TRUTH" in RAG_SYSTEM_PROMPT
        assert "INSUFFICIENT INFORMATION" in RAG_SYSTEM_PROMPT
        assert "pretrained knowledge" in RAG_SYSTEM_PROMPT.lower()
        assert "only factual source" in RAG_SYSTEM_PROMPT.lower()

    def test_natural_tone_no_robotic_filler(self):
        assert "NATURAL, DIRECT PRESENTATION" in RAG_SYSTEM_PROMPT
        assert "Based on the retrieved video transcript segments" in RAG_SYSTEM_PROMPT
        assert "The user already knows they are chatting with the video" in RAG_SYSTEM_PROMPT

    def test_question_type_adaptation(self):
        assert "MATCH RESPONSE TYPE TO QUESTION" in RAG_SYSTEM_PROMPT
        assert "Definition question" in RAG_SYSTEM_PROMPT
        assert "List question" in RAG_SYSTEM_PROMPT
        assert "Location question" in RAG_SYSTEM_PROMPT

    def test_user_prompt_strict_instructions(self, mock_chunk):
        prompt = build_user_prompt(
            question="What is a neuron?",
            chunks=[mock_chunk],
        )

        assert "STRICT GROUNDING INSTRUCTIONS" in prompt
        assert "only factual source" in prompt.lower()
        assert "pretrained knowledge" in prompt.lower()
        assert "02:37" in prompt
        assert "https://www.youtube.com/watch?v=aircAruvnKk&t=157s" in prompt
        assert "What is a neuron?" in prompt

    def test_summary_prompt_structure(self, mock_chunk):
        summary_prompt = build_summary_prompt([mock_chunk])
        assert "### Overview" in summary_prompt
        assert "### Key Points" in summary_prompt
        assert "### Conclusion" in summary_prompt
        assert "STRICT GROUNDING RULES" in summary_prompt

    def test_is_unsupported_answer_detection(self):
        refusal_1 = "The available video transcript does not provide enough information to answer this question."
        refusal_2 = "I couldn't find enough information about that in this video's transcript."
        refusal_3 = "The transcript does not contain any explanation of backpropagation."
        refusal_4 = "The transcript does not cover backpropagation."
        refusal_5 = "This concept is not discussed in this video."
        supported = "A neuron is a unit holding an activation value between 0 and 1."

        assert is_unsupported_answer(refusal_1) is True
        assert is_unsupported_answer(refusal_2) is True
        assert is_unsupported_answer(refusal_3) is True
        assert is_unsupported_answer(refusal_4) is True
        assert is_unsupported_answer(refusal_5) is True
        assert is_unsupported_answer(supported) is False

    def test_unsupported_question_no_unrelated_sources(self, mock_chunk):
        """Unsupported questions must NOT present unrelated retrieved chunks as Sources."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [(mock_chunk, 0.40)]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "The available video transcript does not provide enough information to answer what backpropagation is."
        )

        chain = RAGQAChain(retriever=mock_retriever, llm_service=mock_llm)
        response = chain.answer_question("What is backpropagation?")

        # Grounding check:
        assert response["is_supported"] is False
        # Chunks must NOT be presented under 'sources':
        assert response["sources"] == []
        # Chunks are retained only as debug/transparency retrieved_context:
        assert len(response["retrieved_context"]) == 1
        assert response["retrieved_context"][0]["start_time_formatted"] == "02:37"

    def test_supported_question_shows_clickable_sources(self, mock_chunk):
        """Supported questions must present relevant chunks with clickable timestamp sources."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [(mock_chunk, 0.88)]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "A neuron is a simple unit that holds an activation value between 0 and 1."
        )

        chain = RAGQAChain(retriever=mock_retriever, llm_service=mock_llm)
        response = chain.answer_question("What is a neuron?")

        assert response["is_supported"] is True
        assert len(response["sources"]) == 1
        src = response["sources"][0]
        assert src["chunk_id"] == 0
        assert src["start_time_formatted"] == "02:37"
        assert src["end_time_formatted"] == "03:42"
        assert src["timestamp_url"] == "https://www.youtube.com/watch?v=aircAruvnKk&t=157s"
        assert "label" in src
        assert src["score"] == 0.88

    def test_qa_chain_empty_retrieval_grounding(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_llm = MockLlamaService()

        chain = RAGQAChain(retriever=mock_retriever, llm_service=mock_llm)
        response = chain.answer_question("Who is the president of France?")

        assert "I couldn't find enough information" in response["answer"]
        assert response["is_supported"] is False
        assert response["sources"] == []
        assert response["retrieved_context"] == []

    def test_generate_summary_workflow(self, mock_chunk):
        mock_retriever = MagicMock()
        mock_retriever.vectorstore.chunks = [mock_chunk]
        mock_llm = MockLlamaService()

        chain = RAGQAChain(retriever=mock_retriever, llm_service=mock_llm)
        summary_res = chain.generate_summary()

        assert "answer" in summary_res
        assert summary_res["is_supported"] is True
        assert "sources" in summary_res
        assert len(summary_res["sources"]) == 1
        assert summary_res["sources"][0]["start_time_formatted"] == "02:37"
