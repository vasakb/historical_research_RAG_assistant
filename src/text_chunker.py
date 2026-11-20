"""
Text Chunking Module (Package Export)
=====================================
Exposes chunking functions for internal src package imports.
"""

from text_chunker import (
    chunk_documents,
    chunk_pages,
    chunk_pdf,
    get_text_splitter,
    verify_chunk_overlap,
)

__all__ = [
    "chunk_documents",
    "chunk_pages",
    "chunk_pdf",
    "get_text_splitter",
    "verify_chunk_overlap",
]
