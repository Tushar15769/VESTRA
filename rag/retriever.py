"""Hybrid full-transcript retriever with two-stage candidate retrieval and lexical fallback."""

import re
from typing import Dict, List, Optional, Set, Tuple
from config.settings import get_settings
from embeddings.embedding_service import EmbeddingService, get_embedding_service
from ingestion.chunker import TranscriptChunk
from vectorstore.faiss_store import FAISSVectorStore
from utils.logging_utils import get_logger

logger = get_logger("retriever")

STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your", "yours",
    "yourself", "yourselves", "tell", "explain", "describe", "discuss", "mention", "speaker", "video",
}


def extract_keywords(query: str) -> List[str]:
    """Extract informative lowercase terms from query, excluding common stop words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", query.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def extract_phrases(query: str, min_words: int = 2, max_words: int = 3) -> List[str]:
    """Extract contiguous multi-word phrases from query for exact lexical matching."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", query.lower())
    phrases = []
    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            sub = tokens[i : i + n]
            if any(t not in STOP_WORDS for t in sub):
                phrases.append(" ".join(sub))
    return list(dict.fromkeys(phrases))


def _term_in_text(term: str, text: str) -> bool:
    """Check if term or simple plural/inflection variant appears in text."""
    if term in text:
        return True
    if term.endswith("s") and len(term) > 3 and term[:-1] in text:
        return True
    if term.endswith("es") and len(term) > 4 and term[:-2] in text:
        return True
    if term.endswith("ing") and len(term) > 5 and term[:-3] in text:
        return True
    if term.endswith("ed") and len(term) > 4 and term[:-2] in text:
        return True
    return False


def compute_lexical_score(
    chunk_text: str, keywords: List[str], phrases: List[str]
) -> Tuple[float, bool]:
    """Calculate lexical relevance score [0.0, 1.0] and phrase match flag."""
    if not keywords and not phrases:
        return 0.0, False

    text_lower = chunk_text.lower()

    # Check exact multi-word phrase matches
    phrase_matched = False
    for p in phrases:
        if p in text_lower:
            phrase_matched = True
            break
        p_words = p.split()
        if p_words[-1].endswith("s") and len(p_words[-1]) > 3:
            p_sing = " ".join(p_words[:-1] + [p_words[-1][:-1]])
            if p_sing in text_lower:
                phrase_matched = True
                break

    # Check keyword matches
    matched_keywords = [k for k in keywords if _term_in_text(k, text_lower)]
    keyword_coverage = len(matched_keywords) / len(keywords) if keywords else 0.0

    if not matched_keywords and not phrase_matched:
        return 0.0, False

    total_occurrences = sum(text_lower.count(k) for k in matched_keywords)
    freq_bonus = min(0.2, total_occurrences * 0.03)

    if phrase_matched:
        score = 0.55 + 0.35 * keyword_coverage + freq_bonus
    else:
        score = 0.70 * keyword_coverage + freq_bonus

    return min(1.0, score), phrase_matched


class TranscriptRetriever:
    """Retrieves top-k relevant transcript chunks using two-stage hybrid search across the full transcript."""

    def __init__(
        self,
        vectorstore: FAISSVectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: Optional[int] = None,
        candidates_k: Optional[int] = None,
    ):
        settings = get_settings()
        self.vectorstore = vectorstore
        self.embedding_service = embedding_service or get_embedding_service()
        self.top_k = top_k if top_k is not None else settings.top_k
        self.candidates_k = (
            candidates_k
            if candidates_k is not None
            else getattr(settings, "retrieval_candidates", 10)
        )
        self.last_candidates: Optional[List[Tuple[TranscriptChunk, float]]] = None

    def retrieve_candidates(
        self, query: str
    ) -> Tuple[List[Tuple[TranscriptChunk, float]], List[Tuple[TranscriptChunk, float]]]:
        """Perform two-stage full-transcript retrieval:
        Stage 1: Retrieve candidate chunks across the entire video via FAISS + lexical fallback.
        Stage 2: Relevance-aware hybrid scoring and selection of top-k evidence chunks.

        Returns:
            Tuple of:
            - final_evidence: List of (TranscriptChunk, hybrid_score) up to top_k
            - all_candidates: List of (TranscriptChunk, score) evaluated in candidate pool
        """
        all_chunks = self.vectorstore.chunks
        if not all_chunks:
            return [], []

        # 1. Stage 1: Semantic Candidate Search across complete transcript
        candidates_limit = min(max(self.candidates_k, self.top_k), len(all_chunks))
        query_vector = self.embedding_service.embed_query(query)
        semantic_results = self.vectorstore.similarity_search_by_vector(
            query_vector, top_k=candidates_limit
        )

        semantic_score_map: Dict[int, float] = {
            chunk.chunk_id: float(score) for chunk, score in semantic_results
        }
        chunk_map: Dict[int, TranscriptChunk] = {
            chunk.chunk_id: chunk for chunk, _ in semantic_results
        }

        # 2. Stage 1: Lexical Search across all chunks in the entire video
        keywords = extract_keywords(query)
        phrases = extract_phrases(query)
        lexical_candidates: List[Tuple[TranscriptChunk, float, bool]] = []

        for chunk in all_chunks:
            chunk_map[chunk.chunk_id] = chunk
            lex_score, has_phrase = compute_lexical_score(chunk.text, keywords, phrases)
            if lex_score > 0.0:
                lexical_candidates.append((chunk, lex_score, has_phrase))

        # Sort lexical matches by score descending
        lexical_candidates.sort(key=lambda x: x[1], reverse=True)
        lexical_score_map: Dict[int, float] = {c.chunk_id: s for c, s, _ in lexical_candidates}
        lexical_phrase_map: Dict[int, bool] = {c.chunk_id: p for c, s, p in lexical_candidates}

        # 3. Adaptive Candidate Selection
        candidate_ids: Set[int] = set(semantic_score_map.keys())

        # Include top lexical matches into candidate pool (up to candidates_k)
        for chunk, lex_score, has_phrase in lexical_candidates[: self.candidates_k]:
            candidate_ids.add(chunk.chunk_id)

        min_semantic_score = min(semantic_score_map.values()) if semantic_score_map else 0.0

        # 4. Stage 2: Hybrid Scoring & Relevance Selection
        scored_candidates: List[Tuple[TranscriptChunk, float]] = []
        for cid in candidate_ids:
            chunk = chunk_map[cid]
            sem_score = semantic_score_map.get(cid, max(0.0, min_semantic_score * 0.75))
            lex_score = lexical_score_map.get(cid, 0.0)
            has_phrase = lexical_phrase_map.get(cid, False)

            # Relevance weighting
            if has_phrase:
                hybrid_score = 0.40 * sem_score + 0.60 * lex_score + 0.35
            elif lex_score > 0.0:
                hybrid_score = 0.50 * sem_score + 0.50 * lex_score + 0.15
            else:
                hybrid_score = sem_score

            scored_candidates.append((chunk, round(float(hybrid_score), 4)))

        # Sort all candidates by hybrid score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Final evidence chunks capped at top_k
        final_evidence = scored_candidates[: self.top_k]
        self.last_candidates = scored_candidates

        logger.info(
            f"Retrieval complete for '{query}': {len(scored_candidates)} candidates evaluated across "
            f"{len(all_chunks)} chunks, {len(final_evidence)} final evidence chunks selected."
        )

        return final_evidence, scored_candidates

    def retrieve(self, query: str) -> List[Tuple[TranscriptChunk, float]]:
        """Retrieve top-k final evidence chunks from full transcript search."""
        final_evidence, _ = self.retrieve_candidates(query)
        return final_evidence
