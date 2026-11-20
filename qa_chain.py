"""
RAG Question Answering Pipeline (Root Interface)
================================================
Re-exports the updated HybridRAGPipeline from rag_pipeline.
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
