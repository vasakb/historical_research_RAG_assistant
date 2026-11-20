"""
Question Answering Module (Package Export)
=========================================
Exposes RAG pipeline functions for internal src package imports.
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
