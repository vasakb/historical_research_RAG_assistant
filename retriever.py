"""
Retriever Module
================
Handles loading the persisted FAISS vector database and retrieving the most
relevant historical document chunks based on semantic similarity.
"""

from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from vector_db import load_vector_store
from embeddings import get_embedding_model


def get_faiss_retriever(
    vector_store_path: Union[str, Path] = "vector_store",
    k: int = 4,
):
    """
    Load the FAISS vector database and convert it into a LangChain Retriever.

    Args:
        vector_store_path (str | Path): Path to the saved FAISS index folder.
        k (int): Number of top documents to retrieve (default 4).

    Returns:
        VectorStoreRetriever: Configured retriever object.
    """
    vector_store = load_vector_store(vector_store_path)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_relevant_chunks(
    query: str,
    vector_store_path: Union[str, Path] = "vector_store",
    k: int = 4,
) -> List[Document]:
    """
    Retrieve top-k relevant document chunks for a given query, preserving page metadata.

    Args:
        query (str): The historical research question.
        vector_store_path (str | Path): Path to the saved FAISS index folder.
        k (int): Number of chunks to retrieve (default 4).

    Returns:
        List[Document]: The top matching chunks containing text and page metadata.
    """
    retriever = get_faiss_retriever(vector_store_path=vector_store_path, k=k)
    return retriever.invoke(query)


if __name__ == "__main__":
    sample_q = "What gauge was standardized for Indian railways?"
    print(f"Testing retrieval for: '{sample_q}'")
    results = retrieve_relevant_chunks(sample_q, k=4)
    print(f"Retrieved {len(results)} chunks:")
    for idx, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "Unknown")
        print(f"[{idx}] Page {page}: {doc.page_content[:150]}...")
