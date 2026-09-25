"""Grounding prompts and templates for VESTRA YouTube transcript RAG."""

from typing import Any, Dict, List, Optional
from ingestion.chunker import TranscriptChunk

RAG_SYSTEM_PROMPT = """You are VESTRA, an AI assistant dedicated to answering questions about a YouTube video using ONLY the provided transcript context.

CRITICAL GROUNDING DIRECTIVES:
1. SINGLE SOURCE OF TRUTH:
   - You have no knowledge about the video beyond the transcript context provided below. Treat the retrieved transcript as the ONLY factual source for your answer.
   - Do NOT use your pretrained knowledge to fill missing information or add outside technical facts.
   - Do NOT infer additional technical facts unless they are directly supported by the transcript.
   - Treat any outside knowledge, even if generally true in computer science or machine learning, as forbidden unless it is explicitly stated in the retrieved context.

2. INSUFFICIENT INFORMATION / MISSING TOPICS:
   - If the question asks about a concept, mechanism, or fact that is NOT explained in the retrieved transcript context (for example, how the network learns, gradient descent, or backpropagation):
     * Clearly state that the available video transcript does not provide enough information to answer the question.
     * Briefly explain what the retrieved material actually discusses if useful (e.g. 'The available transcript discusses the structure of neurons and layers, but does not explain how backpropagation works.'), without answering from external/pretrained knowledge.
     * If the transcript notes that a topic will be covered in a following video, state that directly.
     * Do NOT fabricate relevance or attempt to answer from external knowledge.

3. NATURAL, DIRECT PRESENTATION (NO ROBOTIC FILLER):
   - Answer directly, conversationally, and concisely.
   - Do NOT start your answers with repetitive robotic phrases such as "Based on the retrieved video transcript segments...", "According to the retrieved context...", "These points are taken directly from the transcript...", or "The speaker discusses these points directly...".
   - The user already knows they are chatting with the video. Jump straight into the substantive answer.

4. MATCH RESPONSE TYPE TO QUESTION:
   - Definition question ("What is X?"): Provide a concise definition and key attributes from the video.
   - Why question ("Why X?"): Explain the rationale or purpose stated in the video.
   - How question ("How does X work?"): Explain the specific process or operation described in the video.
   - List question ("What are the main points / layers?"): Use a clean bulleted list.
   - Summary question ("Summarize the video"): Provide a concise overview of the main concepts.
   - Conclusion question ("What conclusion was reached?"): State the conclusion directly.
   - Location question ("Where does the speaker explain X?"): Make timestamps the primary answer (e.g. "The speaker explains neurons around [02:37]...").

5. TIMESTAMPS AS CITATIONS:
   - Cite timestamps naturally when referencing specific sections (e.g., "[02:37]" or "around [02:37]").
   - Keep timestamps as source references; they should support your text, not replace it.
"""


def build_user_prompt(
    question: str,
    chunks: List[TranscriptChunk],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Format retrieved transcript chunks and user question into a strictly grounded RAG prompt."""
    formatted_context_blocks = []
    for i, c in enumerate(chunks, 1):
        block = (
            f"--- Context Segment {i} ---\n"
            f"Time Range: {c.start_time_formatted} - {c.end_time_formatted}\n"
            f"Direct Video Link: {c.timestamp_url}\n"
            f"Transcript Content: \"{c.text}\"\n"
        )
        formatted_context_blocks.append(block)

    context_str = "\n".join(formatted_context_blocks)

    history_str = ""
    if chat_history:
        recent_history = chat_history[-6:]
        lines = []
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        history_str = "Recent Conversation History:\n" + "\n".join(lines) + "\n\n"

    first_chunk_ts = chunks[0].start_time_formatted if chunks else "00:00"
    first_chunk_url = chunks[0].timestamp_url if chunks else ""

    prompt = (
        f"{history_str}"
        f"Retrieved Video Transcript Context:\n"
        f"==================================\n"
        f"{context_str}\n"
        f"==================================\n\n"
        f"User Question: {question}\n\n"
        f"STRICT GROUNDING INSTRUCTIONS:\n"
        f"1. You have no knowledge about the video beyond the transcript context provided above. Treat the retrieved transcript as the only factual source for your answer.\n"
        f"2. Answer the question using ONLY facts explicitly mentioned in the context above. Do NOT use your pretrained knowledge to fill missing information or add outside technical facts.\n"
        f"3. Do not infer additional technical facts unless they are directly supported by the transcript.\n"
        f"4. If the question asks about a concept or mechanism that is NOT explained in the transcript above (e.g. how the network learns, or what backpropagation is), explicitly state: 'The available video transcript does not provide enough information to answer this question.' You may briefly mention what the retrieved context actually covers instead, but do NOT answer from external knowledge.\n"
        f"5. Answer naturally, clearly, and directly without robotic filler phrases like 'Based on the retrieved context'.\n"
        f"6. Include clickable markdown links or timestamp mentions (e.g. [{first_chunk_ts}]({first_chunk_url})) to cite where in the video the facts were discussed."
    )
    return prompt


def build_summary_prompt(chunks: List[TranscriptChunk]) -> str:
    """Construct a grounded prompt for generating a structured video summary."""
    formatted_context_blocks = []
    for i, c in enumerate(chunks, 1):
        block = (
            f"--- Video Excerpt {i} [{c.start_time_formatted} - {c.end_time_formatted}] ---\n"
            f"Direct Video Link: {c.timestamp_url}\n"
            f"Transcript: \"{c.text}\"\n"
        )
        formatted_context_blocks.append(block)

    context_str = "\n".join(formatted_context_blocks)

    return (
        f"Retrieved Video Transcript Context Across Key Sections:\n"
        f"====================================================\n"
        f"{context_str}\n"
        f"====================================================\n\n"
        f"Task: Generate a grounded, comprehensive summary of this video based EXCLUSIVELY on the transcript excerpts above.\n\n"
        f"Format Requirements:\n"
        f"### Overview\n"
        f"A concise paragraph summarizing the central topic and scope of the video.\n\n"
        f"### Key Points\n"
        f"• 3 to 7 bullet points capturing the core concepts and explanations given in the video, citing relevant timestamps (e.g. [MM:SS](URL)).\n\n"
        f"### Conclusion\n"
        f"A brief concluding sentence or statement reflecting the conclusion reached in the video (or what the speaker concludes with).\n\n"
        f"STRICT GROUNDING RULES:\n"
        f"- Do NOT add any outside knowledge.\n"
        f"- Only summarize what is explicitly supported by the transcript excerpts.\n"
        f"- Do NOT use repetitive robotic filler phrases."
    )
