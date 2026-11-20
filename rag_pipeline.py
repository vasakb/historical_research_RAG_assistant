"""
RAG Pipeline Interface Module
=============================
Provides a unified interface for the Historical RAG Question Answering pipeline,
connecting the FAISS retriever with the Groq LLM.
"""

from qa_chain import (
    HistoricalRAGPipeline,
    RAGResponse,
    answer_historical_question,
    format_context_with_citations,
)

__all__ = [
    "HistoricalRAGPipeline",
    "RAGResponse",
    "answer_historical_question",
    "format_context_with_citations",
]
