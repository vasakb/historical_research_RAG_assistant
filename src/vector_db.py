"""
Vector Database Module (Package Export)
=======================================
Exposes FAISS vector store functions for internal src package imports.
"""

from vector_db import (
    create_vector_store,
    load_vector_store,
    similarity_search,
    similarity_search_with_score,
)

__all__ = [
    "create_vector_store",
    "load_vector_store",
    "similarity_search",
    "similarity_search_with_score",
]
