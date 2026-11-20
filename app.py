"""
Modern History & Archival Chronicle Assistant
==============================================
A Streamlit-based interactive chatbot providing fact-grounded historical
research assistance using Retrieval-Augmented Generation (RAG).
"""

import os
from pathlib import Path
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import HistoricalRAGPipeline, RAGResponse

# Load environment variables (.env)
load_dotenv()


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> HistoricalRAGPipeline:
    """
    Initialize and cache the RAG pipeline using existing FAISS vector database.
    Cached so FAISS index is only loaded once into memory across user sessions.
    """
    return HistoricalRAGPipeline(
        vector_store_path="vector_store",
        k=4,
        model_name="qwen/qwen3.8-27b",
        temperature=0.2,
    )


def init_session_state():
    """Initialize chat messages history in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome! I am your Modern History & Archival Chronicle Assistant. "
                    "Ask me questions about 19th-century Indian railways, telegraph infrastructure, "
                    "or Lord Dalhousie's railway minutes, and I will cite the exact document pages."
                ),
                "pages": [],
            }
        ]


def display_chat_message(role: str, content: str, pages: Optional[List[int]] = None):
    """Render a single chat message bubble with optional source page tags."""
    with st.chat_message(role):
        st.markdown(content)
        if pages and len(pages) > 0:
            page_tags = " ".join([f"`Page {p}`" for p in pages])
            st.caption(f"📑 **Document Sources:** {page_tags}")


def render_sidebar():
    """Render the sidebar with document and vector database metadata."""
    with st.sidebar:
        st.title("📚 Corpus & System Info")
        st.markdown(
            "**Modern History & Archival Chronicle Assistant**\n\n"
            "An AI-powered research assistant grounded in official 19th-century "
            "colonial records on Indian Railways and Telegraph infrastructure."
        )

        st.divider()
        st.subheader("📄 Document Details")
        st.markdown("- **Document Name**: `indian_railways_and_telegraph_history_50p.pdf`")
        st.markdown("- **Total Pages**: `50` pages")
        st.markdown("- **Extracted Pages**: `50` (100% text coverage)")

        st.divider()
        st.subheader("🧠 Vector Database & Model")
        st.markdown("- **Total Chunks**: `147` chunks (~640 chars avg)")
        st.markdown("- **Chunk Size / Overlap**: `800` / `150` characters")
        st.markdown("- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`")
        st.markdown("- **Vector Dimension**: `384` dimensions")

        # Check FAISS index status on disk
        faiss_dir = Path("vector_store")
        if (faiss_dir / "index.faiss").exists() and (faiss_dir / "index.pkl").exists():
            st.success("✅ **FAISS Database**: Loaded & Ready (`vector_store/`)")
        else:
            st.warning("⚠️ **FAISS Database**: Index files not detected in `vector_store/`")

        st.divider()
        st.subheader("⚡ Inference Engine")
        st.markdown("- **LLM Provider**: `Groq Cloud API`")
        st.markdown("- **Model**: `qwen/qwen3.8-27b`")

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    st.set_page_config(
        page_title="Modern History & Archival Chronicle Assistant",
        page_icon="📜",
        layout="wide",
    )

    # Render Sidebar with metadata
    render_sidebar()

    # Header & Title
    st.title("📜 Modern History & Archival Chronicle Assistant")
    st.markdown(
        "Welcome to the **Modern History & Archival Chronicle Assistant**. "
        "This interactive tool queries 19th-century archival records on Indian Railways "
        "and Telegraphs, delivering verified answers grounded strictly in source documentation "
        "with precise page citations."
    )
    st.divider()

    # Initialize chat history
    init_session_state()

    # Display past conversation history
    for msg in st.session_state.messages:
        display_chat_message(
            role=msg["role"],
            content=msg["content"],
            pages=msg.get("pages"),
        )

    # User chat input
    user_query = st.chat_input("Ask a historical question (e.g., 'What gauge was standardized for Indian railways?')...")

    if user_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using connected RAG pipeline
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching historical archives and generating grounded response..."):
                    pipeline = get_rag_pipeline()
                    rag_response: RAGResponse = pipeline.query(user_query)

                    # Display generated answer
                    st.markdown(rag_response.answer)

                    # Display retrieved source page citations
                    if rag_response.cited_pages and len(rag_response.cited_pages) > 0:
                        page_tags = " ".join([f"`Page {p}`" for p in rag_response.cited_pages])
                        st.caption(f"📑 **Document Sources:** {page_tags}")

                    # Save to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": rag_response.answer,
                            "pages": rag_response.cited_pages,
                        }
                    )
            except Exception as e:
                error_msg = f"An error occurred while generating the answer: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ {error_msg}",
                        "pages": [],
                    }
                )


if __name__ == "__main__":
    main()
