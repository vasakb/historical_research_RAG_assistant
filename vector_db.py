"""
Vector Database Module (FAISS)
==============================
Handles creating, saving, loading, and querying local FAISS vector stores.
Preserves chunk text, embeddings, and original metadata (page numbers).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from embeddings import get_embedding_model


def create_vector_store(
    chunks: List[Document],
    embeddings: Optional[Embeddings] = None,
    save_path: Optional[Union[str, Path]] = "vector_store",
) -> FAISS:
    """
    Create a FAISS vector index from text chunks and their embeddings.
    Optionally persists the index locally to disk.

    Args:
        chunks (List[Document]): List of chunked documents from Phase 3.
        embeddings (Optional[Embeddings]): Embedding model instance.
        save_path (Optional[str | Path]): Local folder to save index files (default 'vector_store').

    Returns:
        FAISS: The initialized FAISS vector database.
    """
    if embeddings is None:
        embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(chunks, embeddings)

    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(save_dir))

    return vector_store


def load_vector_store(
    load_path: Union[str, Path] = "vector_store",
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    """
    Load an existing local FAISS vector index from disk.

    Args:
        load_path (str | Path): Folder containing saved FAISS index files (index.faiss, index.pkl).
        embeddings (Optional[Embeddings]): Embedding model instance.

    Returns:
        FAISS: The loaded FAISS vector database.
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index folder not found at: {path.resolve()}")

    if embeddings is None:
        embeddings = get_embedding_model()

    return FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search(
    query: str,
    vector_store: FAISS,
    k: int = 4,
) -> List[Document]:
    """
    Query the vector store and retrieve the k most relevant chunks.

    Args:
        query (str): The search question or keywords.
        vector_store (FAISS): An initialized or loaded FAISS store.
        k (int): Number of top matches to retrieve (default 4).

    Returns:
        List[Document]: The k most relevant document chunks with metadata.
    """
    return vector_store.similarity_search(query, k=k)


def similarity_search_with_score(
    query: str,
    vector_store: FAISS,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Query the vector store and retrieve the k most relevant chunks with similarity/distance scores.

    Args:
        query (str): The search question or keywords.
        vector_store (FAISS): An initialized or loaded FAISS store.
        k (int): Number of top matches to retrieve (default 4).

    Returns:
        List[Tuple[Document, float]]: Pairs of (chunk, distance_score).
    """
    return vector_store.similarity_search_with_score(query, k=k)

