"""
Text Chunking Module
====================
Splits long extracted document text into smaller, overlapping chunks
while preserving page number metadata and document structure for RAG retrieval.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_processor import PageData, process_pdf


def get_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    """
    Configure and return a LangChain RecursiveCharacterTextSplitter.

    Args:
        chunk_size (int): Maximum number of characters per chunk (default 800).
        chunk_overlap (int): Number of overlapping characters between chunks (default 150).

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )


def chunk_pages(
    pages: List[PageData],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    source_name: Optional[str] = None,
) -> List[Document]:
    """
    Split a list of extracted PageData objects into overlapping Document chunks.
    Preserves original page numbers in chunk metadata.

    Args:
        pages (List[PageData]): Extracted page data from PDFProcessor.
        chunk_size (int): Target chunk size in characters (default 800).
        chunk_overlap (int): Target overlap size in characters (default 150).
        source_name (Optional[str]): Source identifier (e.g. file name).

    Returns:
        List[Document]: List of chunked LangChain Document objects with page metadata.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Document] = []
    chunk_counter = 0

    for page in pages:
        # Skip empty pages
        if page.is_empty or not page.text.strip():
            continue

        # Create a parent document for this page
        parent_doc = Document(
            page_content=page.text,
            metadata={
                "page": page.page_number,
                "source": source_name or "unknown",
            },
        )

        # Split this single page into chunks
        page_chunks = splitter.split_documents([parent_doc])

        # Add per-chunk index tracking while keeping original page number
        for sub_idx, chunk in enumerate(page_chunks):
            chunk.metadata["chunk_id"] = chunk_counter
            chunk.metadata["page_chunk_index"] = sub_idx
            chunk.metadata["total_page_chunks"] = len(page_chunks)
            all_chunks.append(chunk)
            chunk_counter += 1

    return all_chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split a list of existing LangChain Documents into overlapping chunks.

    Args:
        documents (List[Document]): Input LangChain documents.
        chunk_size (int): Target chunk size in characters.
        chunk_overlap (int): Target overlap size in characters.

    Returns:
        List[Document]: List of chunked Documents.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def chunk_pdf(
    file_path: Union[str, Path],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Convenience function: extracts text from a PDF and chunks it directly.

    Args:
        file_path (str | Path): Path to the target PDF document.
        chunk_size (int): Maximum chunk size in characters (default 800).
        chunk_overlap (int): Overlap in characters (default 150).

    Returns:
        List[Document]: Chunked documents ready for downstream embeddings and FAISS.
    """
    path = Path(file_path)
    processor = process_pdf(path)
    return chunk_pages(
        pages=processor.pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_name=path.name,
    )


def verify_chunk_overlap(
    chunk_a_text: str,
    chunk_b_text: str,
    max_check_len: int = 200,
) -> int:
    """
    Helper function to check the length of overlapping characters between two consecutive chunks.

    Args:
        chunk_a_text (str): Preceding chunk's text.
        chunk_b_text (str): Following chunk's text.
        max_check_len (int): Upper bound of characters to search for overlap.

    Returns:
        int: Number of characters that overlap between chunk_a's end and chunk_b's beginning.
    """
    max_len = min(len(chunk_a_text), len(chunk_b_text), max_check_len)
    for length in range(max_len, 10, -1):
        suffix = chunk_a_text[-length:]
        if suffix in chunk_b_text:
            return length
    return 0


if __name__ == "__main__":
    test_pdf = Path("data/indian_railways_and_telegraph_history_50p.pdf")
    if test_pdf.exists():
        chunks = chunk_pdf(test_pdf, chunk_size=800, chunk_overlap=150)
        print(f"Generated {len(chunks)} chunks from {test_pdf.name}")
        if chunks:
            print(f"Sample Chunk 0 (Page {chunks[0].metadata['page']}):\n{chunks[0].page_content[:300]}...")
    else:
        print(f"File not found: {test_pdf}")

