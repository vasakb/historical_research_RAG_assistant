"""
Text Chunking Module
====================
This module breaks down long historical documents into smaller, manageable chunks
so they fit within the embedding and LLM context limits.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split a list of documents into smaller overlapping text chunks.

    Args:
        documents (List[Document]): The raw documents loaded from the PDF.
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks.

    Returns:
        List[Document]: A list of chunked Document objects.
    """
    # TODO: Initialize RecursiveCharacterTextSplitter and split documents
    # Example:
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap
    # )
    # chunks = text_splitter.split_documents(documents)
    # return chunks
    raise NotImplementedError("Text chunking logic will be implemented here.")

