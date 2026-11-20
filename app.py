"""
Modern History & Archival Chronicle Assistant
==============================================
A Streamlit-based interactive chatbot providing fact-grounded historical
and contemporary research assistance using Hybrid RAG (PDF + Web Search).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import HistoricalRAGPipeline, HybridRAGResponse

# Load environment variables (.env)
load_dotenv()


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> HistoricalRAGPipeline:
    """
    Initialize and cache the Hybrid RAG pipeline.
    Loads FAISS index once into memory and provides access to Groq LLM + Tavily search.
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
                    "Welcome! I am your **Modern History & Archival Chronicle Assistant**.\n\n"
                    "You can ask me:\n"
                    "• **Historical questions** from our 50-page colonial archives on Indian Railways and Telegraphs (I will cite the exact page numbers).\n"
                    "• **External or contemporary questions** (I will search verified web sources and provide direct links)."
                ),
                "source_type": None,
                "pages": [],
                "web_sources": [],
            }
        ]


def display_chat_message(
    role: str,
    content: str,
    source_type: Optional[str] = None,
    pages: Optional[List[int]] = None,
    web_sources: Optional[List[Dict[str, str]]] = None,
):
    """Render a single chat message bubble with distinct source attribution."""
    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and source_type:
            st.divider()
            if source_type == "PDF":
                st.markdown("📜 **Source Type:** `Historical PDF Document`")
                if pages and len(pages) > 0:
                    page_tags = " ".join([f"`Page {p}`" for p in pages])
                    st.markdown(f"📑 **Cited Archival Pages:** {page_tags}")
            elif source_type == "External Web":
                st.markdown("🌐 **Source Type:** `External Web Search`")
                if web_sources and len(web_sources) > 0:
                    st.markdown("🔗 **Verified Web Citations:**")
                    for ws in web_sources:
                        title = ws.get("title", "Source Webpage")
                        url = ws.get("url", "#")
                        st.markdown(f"- [{title}]({url})")


def render_sidebar():
    """Render the sidebar with document, vector database, and web search metadata."""
    with st.sidebar:
        st.title("📚 Assistant Architecture")
        st.markdown(
            "**Modern History & Archival Chronicle Assistant**\n\n"
            "A hybrid intelligence assistant answering both archival historical queries "
            "and external contemporary questions with strict citation grounding."
        )

        st.divider()
        st.subheader("📄 Historical PDF Corpus")
        st.markdown("- **Document**: `indian_railways_and_telegraph_history_50p.pdf`")
        st.markdown("- **Total Pages**: `50` pages")
        st.markdown("- **Total Chunks**: `147` chunks (800 / 150 overlap)")
        st.markdown("- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`")

        # Check FAISS index status on disk
        faiss_dir = Path("vector_store")
        if (faiss_dir / "index.faiss").exists() and (faiss_dir / "index.pkl").exists():
            st.success("✅ **FAISS Vector DB**: Loaded & Ready (`vector_store/`)")
        else:
            st.warning("⚠️ **FAISS Vector DB**: Index not found in `vector_store/`")

        st.divider()
        st.subheader("🌐 External Web Search")
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            st.success("✅ **Tavily Search API**: Active & Connected")
        else:
            st.warning("⚠️ **Tavily Search API**: Missing `TAVILY_API_KEY`")

        st.divider()
        st.subheader("⚡ Inference Engine")
        st.markdown("- **LLM Provider**: `Groq Cloud API`")
        st.markdown("- **Model**: `qwen/qwen3.8-27b`")
        st.markdown("- **Routing**: `Dynamic Semantic Distance + Auto-Fallback`")

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
        "Welcome! This assistant intelligently routes your questions: queries found in our "
        "**19th-century archival corpus** cite exact PDF page numbers, while questions requiring "
        "**external/contemporary information** search verified web references."
    )
    st.divider()

    # Initialize chat history
    init_session_state()

    # Display past conversation history
    for msg in st.session_state.messages:
        display_chat_message(
            role=msg["role"],
            content=msg["content"],
            source_type=msg.get("source_type"),
            pages=msg.get("pages"),
            web_sources=msg.get("web_sources"),
        )

    # User chat input
    user_query = st.chat_input("Ask a historical question or contemporary question...")

    if user_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using connected Hybrid RAG pipeline
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing query, searching documents or web sources..."):
                    pipeline = get_rag_pipeline()
                    rag_response: HybridRAGResponse = pipeline.query(user_query)

                    # Display generated answer
                    st.markdown(rag_response.answer)

                    # Display source attribution
                    st.divider()
                    if rag_response.source_type == "PDF":
                        st.markdown("📜 **Source Type:** `Historical PDF Document`")
                        if rag_response.cited_pages and len(rag_response.cited_pages) > 0:
                            page_tags = " ".join([f"`Page {p}`" for p in rag_response.cited_pages])
                            st.markdown(f"📑 **Cited Archival Pages:** {page_tags}")
                    elif rag_response.source_type == "External Web":
                        st.markdown("🌐 **Source Type:** `External Web Search`")
                        if rag_response.web_sources and len(rag_response.web_sources) > 0:
                            st.markdown("🔗 **Verified Web Citations:**")
                            for ws in rag_response.web_sources:
                                title = ws.get("title", "Source Webpage")
                                url = ws.get("url", "#")
                                st.markdown(f"- [{title}]({url})")

                    # Save to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": rag_response.answer,
                            "source_type": rag_response.source_type,
                            "pages": rag_response.cited_pages,
                            "web_sources": rag_response.web_sources,
                        }
                    )
            except Exception as e:
                error_msg = f"An error occurred while generating the answer: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ {error_msg}",
                        "source_type": None,
                        "pages": [],
                        "web_sources": [],
                    }
                )


if __name__ == "__main__":
    main()
