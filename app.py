"""VESTRA — Ask beyond the video. (AI-powered YouTube RAG)"""

import os
import streamlit as st
from config.settings import Settings, get_settings
from ingestion.chunker import chunk_transcript
from ingestion.transcript_cleaner import clean_transcript_segments
from ingestion.transcript_loader import (
    TranscriptLoader,
    TranscriptNotFoundError,
    TranscriptsDisabledError,
    VideoUnavailableError,
)
from ingestion.youtube_loader import VideoMetadata, YouTubeMetadataLoader
from embeddings.embedding_service import get_embedding_service
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import TranscriptRetriever
from rag.qa_chain import RAGQAChain
from llm.llama_service import get_llm_service
from utils.logging_utils import get_logger
from utils.youtube_utils import extract_video_id, is_valid_youtube_url

logger = get_logger("vestra_app")

# Page Configuration
st.set_page_config(
    page_title="VESTRA — Ask beyond the video.",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Minimal Modern UI
st.markdown(
    """
    <style>
    /* Global styles */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .vestra-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0F172A;
        margin-bottom: 0px;
        line-height: 1.1;
    }
    .vestra-tagline {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 3px;
        margin-bottom: 2px;
    }
    .vestra-subtitle {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .source-item {
        font-size: 0.92rem;
        color: #334155;
        margin-bottom: 4px;
    }
    .source-timestamp {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 1px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        text-decoration: none;
    }
    .source-timestamp:hover {
        background-color: #DBEAFE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None
    if "video_metadata" not in st.session_state:
        st.session_state.video_metadata = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "trigger_summary" not in st.session_state:
        st.session_state.trigger_summary = False


initialize_session_state()
settings = get_settings()


def render_sources_section(sources: list, retrieved_context: list, is_supported: bool = True):
    """Render grounded evidence passages or graceful fallback for unsupported questions."""
    st.markdown("**📍 Evidence from the video**")
    if not is_supported or not sources:
        st.markdown("*No directly relevant evidence was found in the video.*")
        if retrieved_context:
            with st.expander("▶ View retrieval candidates", expanded=False):
                st.caption(
                    "These are transcript passages considered during retrieval across the entire video. "
                    "None contained sufficient evidence to support this question."
                )
                for s in retrieved_context:
                    st.markdown(
                        f"**{s['start_time_formatted']} – {s['end_time_formatted']}** "
                        f"([▶️ Open in YouTube]({s['timestamp_url']}))"
                    )
                    st.info(s["text"])
    else:
        for s in sources[:4]:
            passage = s.get("passage") or s.get("label") or s["text"][:180]
            st.markdown(f"• [**{s['start_time_formatted']} ↗**]({s['timestamp_url']})")
            st.markdown(f'> *"{passage}"*')

        with st.expander("▶ View transcript evidence", expanded=False):
            st.caption("Full transcript passages supporting the answer above:")
            for s in sources[:4]:
                st.markdown(
                    f"**{s['start_time_formatted']} – {s['end_time_formatted']}** "
                    f"([▶️ Open in YouTube]({s['timestamp_url']}))"
                )
                st.info(s["text"])

        if retrieved_context:
            with st.expander("▶ View retrieval details", expanded=False):
                st.caption(
                    "These are transcript passages considered during retrieval across the entire video. "
                    "Only the final evidence shown above was provided to the LLM as supporting context."
                )
                for s in retrieved_context:
                    st.markdown(
                        f"**{s['start_time_formatted']} – {s['end_time_formatted']}** "
                        f"([▶️ Open in YouTube]({s['timestamp_url']}))"
                    )
                    st.info(s["text"])


# Sidebar: Product Branding & Video Controls
with st.sidebar:
    st.markdown("## **VESTRA**")
    st.markdown("*Ask beyond the video.*")
    st.caption("AI Video Intelligence")
    st.markdown("---")

    st.markdown("#### Active Video")
    if st.session_state.video_metadata and st.session_state.current_video_id:
        st.markdown(f"**{st.session_state.video_metadata.title[:45]}...**")
        st.caption(f"ID: `{st.session_state.current_video_id}` • {len(st.session_state.chunks)} chunks")
        if st.button("🗑️ Clear Video", use_container_width=True):
            st.session_state.current_video_id = None
            st.session_state.video_metadata = None
            st.session_state.chunks = []
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.url_input_value = ""
            st.rerun()
    else:
        st.caption("No video currently active.")

    st.markdown("---")
    st.markdown("#### Sample Videos")
    sample_videos = {
        "3Blue1Brown: Neural Networks": "https://www.youtube.com/watch?v=aircAruvnKk",
        "TED Talk: AI & Future": "https://www.youtube.com/watch?v=5dZ_lvDgevk",
        "OpenAI DevDay Highlights": "https://www.youtube.com/watch?v=U9mJuUkhUzk",
    }
    for label, url in sample_videos.items():
        if st.button(label, use_container_width=True):
            st.session_state.url_input_value = url
            st.rerun()

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# Main Header
st.markdown('<div class="vestra-title">VESTRA</div>', unsafe_allow_html=True)
st.markdown('<div class="vestra-tagline">Ask beyond the video.</div>', unsafe_allow_html=True)
st.markdown('<div class="vestra-subtitle">AI-powered YouTube RAG</div>', unsafe_allow_html=True)

# Architecture Transparency Expander
with st.expander("🧠 How VESTRA works", expanded=False):
    st.markdown(
        """
```text
YouTube Video
      ↓
   Transcript
      ↓
 Smart Chunking
      ↓
Semantic Embeddings
      ↓
FAISS + Lexical Search
      ↓
 Hybrid Reranking
      ↓
Relevant Evidence
      ↓
 GPT-OSS 20B
      ↓
 Grounded Answer
```
VESTRA grounds every answer in the video's transcript and refuses questions when sufficient evidence isn't found.
"""
    )

# Technical Architecture Expander
with st.expander("⚙️ Technical Architecture", expanded=False):
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("**Embedding Model**")
        st.caption("`all-MiniLM-L6-v2` (384-dimensional dense vectors)")
        st.markdown("**Vector Search**")
        st.caption("`FAISS · IndexFlatIP` (Cosine similarity)")
    with t_col2:
        st.markdown("**Retrieval**")
        st.caption("`Semantic + Lexical Hybrid Retrieval`")
        st.markdown("**LLM**")
        st.caption("`Groq · GPT-OSS 20B` (`openai/gpt-oss-20b`)")

# Video URL Input
url_default = getattr(st.session_state, "url_input_value", "")
col_input, col_btn = st.columns([5, 1])
with col_input:
    url_input = st.text_input(
        "Enter YouTube Video URL",
        value=url_default,
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        label_visibility="collapsed",
    )
with col_btn:
    process_btn = st.button("Process", type="primary", use_container_width=True)


def process_video_pipeline(video_url: str):
    """Execute the full RAG preparation pipeline with clean progress reporting."""
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please provide a valid YouTube link or 11-character video ID.")
        return

    vectorstore = FAISSVectorStore(video_id=video_id)
    already_indexed = vectorstore.exists()

    with st.status(f"Processing Video ({video_id})...", expanded=True) as status:
        st.write("✓ YouTube URL validated")

        metadata_loader = YouTubeMetadataLoader()
        try:
            metadata = metadata_loader.get_metadata(video_id)
            st.session_state.video_metadata = metadata
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            st.session_state.video_metadata = VideoMetadata(
                video_id=video_id,
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                title=f"Video ({video_id})",
                channel="YouTube",
                duration_seconds=0,
                duration_formatted="N/A",
                thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            )

        if already_indexed:
            st.write("✓ Existing transcript found")
            st.write("✓ Existing vector index found")
            vectorstore.load()
            st.session_state.chunks = vectorstore.chunks
        else:
            transcript_loader = TranscriptLoader()
            try:
                raw_segments = transcript_loader.load_transcript(video_id)
                st.write("✓ Transcript extracted")
            except TranscriptsDisabledError:
                status.update(label="❌ Transcripts Disabled", state="error")
                st.error("⚠️ Subtitles/transcripts are disabled for this video by the uploader.")
                return
            except VideoUnavailableError:
                status.update(label="❌ Video Unavailable", state="error")
                st.error("⚠️ This video is private, unavailable, or restricted.")
                return
            except TranscriptNotFoundError:
                status.update(label="❌ No Transcript Found", state="error")
                st.error("⚠️ No transcripts or captions were found for this video.")
                return
            except Exception as e:
                status.update(label="❌ Extraction Error", state="error")
                st.error("⚠️ Unable to extract transcript. Please verify the URL and try again.")
                return

            cleaned_segments = clean_transcript_segments(raw_segments)
            if not cleaned_segments:
                status.update(label="❌ Empty Transcript", state="error")
                st.error("⚠️ Cleaned transcript is empty.")
                return
            st.write("✓ Transcript cleaned")

            chunks = chunk_transcript(
                cleaned_segments,
                video_id=video_id,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            st.session_state.chunks = chunks
            st.write("✓ Chunks created")

            embedding_service = get_embedding_service()
            chunk_texts = [c.text for c in chunks]
            embeddings = embedding_service.embed_documents(chunk_texts)
            st.write("✓ Embeddings generated")

            vectorstore.build_and_save(chunks=chunks, embeddings=embeddings)
            st.write("✓ FAISS index ready")

        # Initialize QA Chain
        llm_service = get_llm_service(settings)
        retriever = TranscriptRetriever(
            vectorstore=vectorstore,
            top_k=settings.top_k,
            candidates_k=settings.retrieval_candidates,
        )
        st.session_state.qa_chain = RAGQAChain(retriever=retriever, llm_service=llm_service)
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = []
        status.update(label="🎉 Ready to chat", state="complete")


if process_btn and url_input:
    process_video_pipeline(url_input)

# Display Video Header Card & Actions
if st.session_state.video_metadata and st.session_state.current_video_id:
    meta: VideoMetadata = st.session_state.video_metadata
    st.markdown("---")

    card_col1, card_col2 = st.columns([1, 4])
    with card_col1:
        st.image(meta.thumbnail_url, use_container_width=True)
    with card_col2:
        st.subheader(meta.title)
        st.write(f"**{meta.channel}** • ⏱️ {meta.duration_formatted} • 📦 {len(st.session_state.chunks)} chunks")
        st.caption(f"✓ Video processed • [▶️ Watch on YouTube](https://www.youtube.com/watch?v={meta.video_id})")

    st.markdown("---")

    # Action Row: Generate Summary & Suggested Questions
    btn_col, q_col1, q_col2, q_col3, q_col4 = st.columns([1.5, 1.4, 1.4, 1.4, 1.4])
    with btn_col:
        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            st.session_state.trigger_summary = True
            st.rerun()
    with q_col1:
        if st.button("What is this video about?", use_container_width=True):
            st.session_state.pending_question = "What is this video about?"
            st.rerun()
    with q_col2:
        if st.button("What are the main points?", use_container_width=True):
            st.session_state.pending_question = "What are the main points?"
            st.rerun()
    with q_col3:
        if st.button("Explain the key concept", use_container_width=True):
            st.session_state.pending_question = "Explain the most important concept in this video."
            st.rerun()
    with q_col4:
        if st.button("What conclusion is reached?", use_container_width=True):
            st.session_state.pending_question = "What conclusion did the speaker reach?"
            st.rerun()

    st.markdown("---")

    # Handle Trigger Summary
    if st.session_state.trigger_summary:
        st.session_state.trigger_summary = False
        summary_query = "Summarize this video."
        st.session_state.chat_history.append({"role": "user", "content": summary_query})
        with st.chat_message("user"):
            st.markdown(summary_query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking from the video..."):
                chain: RAGQAChain = st.session_state.qa_chain
                resp = chain.generate_summary()
                ans_text = resp["answer"]
                srcs = resp.get("sources", [])
                ctx = resp.get("retrieved_context", srcs)
                is_supp = resp.get("is_supported", True)

                st.markdown(ans_text)
                render_sources_section(sources=srcs, retrieved_context=ctx, is_supported=is_supp)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ans_text,
                    "is_supported": is_supp,
                    "sources": srcs,
                    "retrieved_context": ctx,
                })

    # Render Conversation History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and ("sources" in msg or "retrieved_context" in msg):
                render_sources_section(
                    sources=msg.get("sources", []),
                    retrieved_context=msg.get("retrieved_context", []),
                    is_supported=msg.get("is_supported", True),
                )

    # Chat Input Handling
    user_prompt = st.chat_input("Ask a question about this video's content...")
    if st.session_state.pending_question:
        user_prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking from the video..."):
                chain: RAGQAChain = st.session_state.qa_chain
                response = chain.answer_question(
                    question=user_prompt,
                    chat_history=st.session_state.chat_history[:-1],
                )

                answer_text = response["answer"]
                sources = response.get("sources", [])
                retrieved_ctx = response.get("retrieved_context", sources)
                is_supp = response.get("is_supported", True)

                st.markdown(answer_text)
                render_sources_section(
                    sources=sources,
                    retrieved_context=retrieved_ctx,
                    is_supported=is_supp,
                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "is_supported": is_supp,
                    "sources": sources,
                    "retrieved_context": retrieved_ctx,
                })

elif not st.session_state.current_video_id:
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.5rem 1.8rem; margin-top: 1rem;">
            <h4 style="margin-top: 0; color: #1E293B; font-weight: 700;">Get started with VESTRA</h4>
            <ol style="margin-bottom: 0.8rem; padding-left: 1.2rem; color: #475569; line-height: 1.8;">
                <li>Paste any public YouTube video URL into the field above and click <strong>Process</strong>.</li>
                <li>Ask questions about the video’s content and receive strictly grounded natural answers.</li>
                <li>Verify answers with clickable transcript evidence linked directly to video timestamps.</li>
            </ol>
            <p style="margin-bottom: 0; color: #64748B; font-size: 0.9rem;">
                💡 <em>Tip: You can also choose one of the sample videos from the left sidebar to try immediately.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
