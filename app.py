"""
Modern History & Archival Chronicle Assistant
==============================================
A Streamlit-based interactive chatbot providing fact-grounded historical
research assistance using Retrieval-Augmented Generation (RAG).
"""

import sys
from pathlib import Path
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()


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


def main():
    st.set_page_config(
        page_title="Modern History & Archival Chronicle Assistant",
        page_icon="📜",
        layout="wide",
    )

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

        # Generate response placeholder (wired to RAG pipeline in Task 2)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching historical archives and generating grounded response..."):
                    response_text = f"Received your question: '{user_query}'. RAG pipeline connection in progress."
                    cited_pages = [1]
                    st.markdown(response_text)
                    page_tags = " ".join([f"`Page {p}`" for p in cited_pages])
                    st.caption(f"📑 **Document Sources:** {page_tags}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            "pages": cited_pages,
                        }
                    )
            except Exception as e:
                st.error(f"An unexpected error occurred while processing your request: {e}")


if __name__ == "__main__":
    main()
