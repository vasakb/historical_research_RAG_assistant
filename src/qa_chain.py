"""
Question Answering Module (Package Export)
=========================================
Exposes RAG pipeline functions for internal src package imports.
"""

from rag_pipeline import (
    HistoricalRAGPipeline,
    HybridRAGResponse,
    RAGResponse,
    answer_historical_question,
    format_pdf_context_with_citations,
)

__all__ = [
    "HistoricalRAGPipeline",
    "HybridRAGResponse",
    "RAGResponse",
    "answer_historical_question",
    "format_pdf_context_with_citations",
]
