"""
PDF Processor Module
====================
This module extracts text from PDF documents using PyPDF.
It tracks page numbers, safely handles empty pages, collects document statistics,
and provides sample text preview.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from pypdf import PdfReader


@dataclass
class PageData:
    """
    Data container representing a single extracted PDF page.
    
    Attributes:
        page_number (int): 1-based page number (1, 2, 3, ...).
        text (str): The extracted plain text from the page.
        character_count (int): Number of characters in the extracted text.
        is_empty (bool): True if the page contains no readable text.
    """
    page_number: int
    text: str
    character_count: int
    is_empty: bool


class PDFProcessor:
    """
    Handles PDF loading, text extraction, page tracking, and summary statistics.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the PDF processor with the path to a PDF file.

        Args:
            file_path (str | Path): Path to the target PDF document.
        """
        self.file_path = Path(file_path)
        self.pages: List[PageData] = []
        self.total_pages: int = 0
        self.pages_with_text: int = 0
        self.empty_pages: int = 0
        self._is_processed: bool = False

    def extract_text(self) -> List[PageData]:
        """
        Read the PDF file, extract text page-by-page, and record statistics.
        Handles missing files, corrupted pages, and empty pages safely without crashing.

        Returns:
            List[PageData]: List of extracted page objects containing page numbers and text.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.file_path}")

        try:
            reader = PdfReader(str(self.file_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open or read PDF: {e}") from e

        self.total_pages = len(reader.pages)
        self.pages = []
        self.pages_with_text = 0
        self.empty_pages = 0

        # Loop through each page in the PDF (0-indexed internally, 1-indexed for humans)
        for index, page in enumerate(reader.pages):
            page_num = index + 1
            page_text = ""

            try:
                # Extract text using PyPDF
                raw_text = page.extract_text()
                # Handle cases where extract_text() returns None
                if raw_text is not None:
                    page_text = raw_text.strip()
            except Exception as page_err:
                # If an individual page fails to extract, do not crash the entire process
                print(f"[Warning] Could not extract text from Page {page_num}: {page_err}")
                page_text = ""

            # Check if the page contains any readable text
            is_empty = len(page_text) == 0

            if is_empty:
                self.empty_pages += 1
            else:
                self.pages_with_text += 1

            self.pages.append(
                PageData(
                    page_number=page_num,
                    text=page_text,
                    character_count=len(page_text),
                    is_empty=is_empty,
                )
            )

        self._is_processed = True
        return self.pages

    def get_summary(self) -> Dict[str, Union[int, str]]:
        """
        Return a summary dictionary of the extraction results.

        Returns:
            Dict[str, Any]: Total pages, pages with text, and empty pages count.
        """
        if not self._is_processed:
            self.extract_text()

        return {
            "file_name": self.file_path.name,
            "total_pages": self.total_pages,
            "pages_with_text": self.pages_with_text,
            "empty_pages": self.empty_pages,
        }

    def get_first_page_sample(self, preview_chars: Optional[int] = None) -> str:
        """
        Retrieve sample text from the first page of the document.

        Args:
            preview_chars (int, optional): If set, limits output to the first N characters.

        Returns:
            str: Extracted text from Page 1, or a placeholder if empty/unavailable.
        """
        if not self._is_processed:
            self.extract_text()

        if not self.pages:
            return "No pages extracted."

        first_page = self.pages[0]
        if first_page.is_empty:
            return f"[Page 1 is empty or contains no extractable text]"

        if preview_chars and preview_chars > 0:
            return first_page.text[:preview_chars]

        return first_page.text


# Standalone helper function for quick extraction
def process_pdf(file_path: Union[str, Path]) -> PDFProcessor:
    """
    Convenience function that initializes a PDFProcessor and immediately extracts text.

    Args:
        file_path (str | Path): Path to the PDF file.

    Returns:
        PDFProcessor: Processed instance with all extracted data.
    """
    processor = PDFProcessor(file_path)
    processor.extract_text()
    return processor


if __name__ == "__main__":
    # Default path for testing directly
    default_path = Path("data/indian_railways_and_telegraph_history_50p.pdf")
    if default_path.exists():
        print(f"--- Processing {default_path.name} ---")
        proc = process_pdf(default_path)
        summary = proc.get_summary()
        print(f"Total Pages: {summary['total_pages']}")
        print(f"Pages with Text: {summary['pages_with_text']}")
        print(f"Empty Pages: {summary['empty_pages']}")
        print("\n--- Sample Text from Page 1 ---")
        print(proc.get_first_page_sample(preview_chars=500))
    else:
        print(f"File not found: {default_path}")

