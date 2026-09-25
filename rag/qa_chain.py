"""End-to-end RAG QA Chain coordinating retrieval, grounded generation, and video summaries."""

from typing import Any, Dict, List, Optional
import numpy as np
from ingestion.chunker import TranscriptChunk
from llm.llama_service import BaseLLMService, get_llm_service
from rag.prompt import RAG_SYSTEM_PROMPT, build_summary_prompt, build_user_prompt
from rag.retriever import TranscriptRetriever
from utils.logging_utils import get_logger

logger = get_logger("qa_chain")


def _generate_source_label(text: str, max_words: int = 7) -> str:
    """Generate a concise topic label from chunk text for clean UI source chips."""
    cleaned = text.strip()
    first_sentence = cleaned.split(".")[0].strip()
    words = first_sentence.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def is_unsupported_answer(answer: str) -> bool:
    """Check if the answer indicates the transcript lacks sufficient information."""
    lower = answer.lower()
    patterns = [
        "not provide enough information",
        "does not provide enough",
        "doesn't provide enough",
        "couldn't find enough information",
        "could not find enough information",
        "not contain enough information",
        "does not contain enough",
        "doesn't contain enough",
        "not enough information",
        "no directly relevant source",
        "cannot provide an answer",
        "does not contain any explanation",
        "transcript does not explain",
        "transcript doesn't explain",
        "transcript does not mention",
        "transcript doesn't mention",
        "transcript does not contain",
        "transcript does not cover",
        "transcript doesn't cover",
        "transcript does not discuss",
        "transcript doesn't discuss",
        "does not cover",
        "doesn't cover",
        "not covered",
        "not covered in the transcript",
        "not covered in this video",
        "not mentioned",
        "not mentioned in the transcript",
        "not mentioned in this video",
        "does not discuss",
        "doesn't discuss",
        "not discussed",
        "not discussed in",
        "available transcript does not",
        "does not describe how",
        "i can't provide a grounded explanation",
        "i cannot provide a grounded explanation",
    ]
    return any(p in lower for p in patterns)


class RAGQAChain:
    """Orchestrates query retrieval, prompt formatting, video summarization, and LLM inference."""

    def __init__(
        self,
        retriever: TranscriptRetriever,
        llm_service: Optional[BaseLLMService] = None,
    ):
        self.retriever = retriever
        self.llm_service = llm_service or get_llm_service()

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Process a user question through the RAG pipeline.

        Args:
            question: User inquiry.
            chat_history: Optional list of past chat turns [{'role': 'user'|'assistant', 'content': str}].

        Returns:
            Dictionary containing:
            - 'answer': generated text
            - 'is_supported': boolean indicating if transcript supported the answer
            - 'sources': list of supporting sources (empty if unsupported)
            - 'retrieved_context': raw retrieved chunks for transparency
        """
        logger.info(f"Processing question: '{question}'")

        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve(question)

        if not results:
            return {
                "answer": "I couldn't find enough information about that in this video's transcript.",
                "is_supported": False,
                "sources": [],
                "retrieved_context": [],
            }

        chunks: List[TranscriptChunk] = [r[0] for r in results]

        # 2. Construct grounded prompt
        prompt = build_user_prompt(question=question, chunks=chunks, chat_history=chat_history)

        # 3. Call LLM Service
        try:
            raw_answer = self.llm_service.generate(prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT)
            answer = raw_answer.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = (
                "⚠️ Unable to generate an answer. The AI provider could not process the request. "
                "Please check your LLM configuration and try again."
            )

        # 4. Check if the question was determined to be unsupported
        is_supported = not is_unsupported_answer(answer)

        # 5. Format retrieved context chunks
        retrieved_context = [
            {
                "chunk_id": chunk.chunk_id,
                "label": _generate_source_label(chunk.text),
                "text": chunk.text,
                "start_timestamp": chunk.start_timestamp,
                "end_timestamp": chunk.end_timestamp,
                "start_time_formatted": chunk.start_time_formatted,
                "end_time_formatted": chunk.end_time_formatted,
                "timestamp_url": chunk.timestamp_url,
                "score": round(score, 4),
            }
            for chunk, score in results
        ]

        # Never present chunks as sources if the answer is unsupported
        sources = retrieved_context if is_supported else []

        return {
            "answer": answer,
            "is_supported": is_supported,
            "sources": sources,
            "retrieved_context": retrieved_context,
        }

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a grounded, structured summary across the video's timeline.

        Returns:
            Dictionary containing structured summary 'answer', 'is_supported', 'sources', and 'retrieved_context'.
        """
        logger.info("Generating comprehensive video summary...")
        all_chunks = self.retriever.vectorstore.chunks

        if not all_chunks:
            return {
                "answer": "I couldn't find enough transcript information to summarize this video.",
                "is_supported": False,
                "sources": [],
                "retrieved_context": [],
            }

        # Select representative chunks across the video timeline (start, middle, end)
        num_sample = min(6, len(all_chunks))
        indices = np.linspace(0, len(all_chunks) - 1, num=num_sample, dtype=int)
        sampled_chunks = [all_chunks[i] for i in sorted(set(indices))]

        prompt = build_summary_prompt(sampled_chunks)

        try:
            raw_answer = self.llm_service.generate(prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT)
            answer = raw_answer.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            answer = (
                "⚠️ Unable to generate video summary. The AI provider could not process the request. "
                "Please check your LLM configuration and try again."
            )

        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "label": _generate_source_label(chunk.text),
                "text": chunk.text,
                "start_timestamp": chunk.start_timestamp,
                "end_timestamp": chunk.end_timestamp,
                "start_time_formatted": chunk.start_time_formatted,
                "end_time_formatted": chunk.end_time_formatted,
                "timestamp_url": chunk.timestamp_url,
                "score": 1.0,
            }
            for chunk in sampled_chunks
        ]

        return {
            "answer": answer,
            "is_supported": True,
            "sources": sources,
            "retrieved_context": sources,
        }
