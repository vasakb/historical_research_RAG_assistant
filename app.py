"""
Historical Research Assistant - Streamlit Frontend
===================================================
Web-based interactive user interface for uploading historical PDFs,
asking questions, and viewing context-grounded answers.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY from .env)
load_dotenv()


def main():
    st.set_page_config(
        page_title="Historical Research Assistant",
        page_icon="📜",
        layout="wide",
    )

    st.title("📜 Historical Research Assistant")
    st.markdown(
        "Welcome! This assistant uses Retrieval-Augmented Generation (RAG) "
        "to answer questions from historical documents and archives."
    )

    # Sidebar for Configuration & Document Upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Enter your Groq API key or add it to a .env file.",
        )

        st.divider()
        st.header("📂 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload Historical PDF",
            type=["pdf"],
            help="Upload a research paper, historical archive, or source text.",
        )

        if uploaded_file is not None:
            st.success(f"File uploaded: `{uploaded_file.name}`")
            if st.button("Process Document", use_container_width=True):
                st.info("Document processing will be connected here.")

    # Main Chat Area
    st.subheader("💬 Ask a Historical Question")

    # Chat history container placeholder
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Once you configure your Groq key and upload a document, ask me any question about your historical text.",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about the uploaded document...")
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Assistant response placeholder
        with st.chat_message("assistant"):
            st.write("*(RAG response logic will be connected here in the next step)*")


if __name__ == "__main__":
    main()

