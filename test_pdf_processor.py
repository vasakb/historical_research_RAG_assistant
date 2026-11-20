"""
Test Script for PDF Text Extraction (Phase 2)
============================================
Verifies:
1. PDF file exists in the data directory.
2. PyPDF reads all 50 pages without crashing.
3. Every page records its 1-based page number.
4. Empty pages are handled gracefully.
5. Total pages count matches 50.
6. Count of pages with text is verified.
7. First page sample text is extracted and displayed.
"""

import sys
from pathlib import Path
from pdf_processor import PDFProcessor, process_pdf


def test_pdf_extraction():
    pdf_path = Path("data/indian_railways_and_telegraph_history_50p.pdf")

    print("=" * 70)
    print("🧪 Running Phase 2 PDF Extraction Test")
    print("=" * 70)

    # 1. Verify file existence
    print(f"\n[Test 1] Checking if PDF exists at: {pdf_path}")
    assert pdf_path.exists(), f"Error: PDF not found at {pdf_path}"
    print(f"  ✅ File found! Size: {pdf_path.stat().st_size:,} bytes")

    # 2. Extract text using PDFProcessor
    print("\n[Test 2] Extracting text across all pages using PyPDF...")
    processor = process_pdf(pdf_path)
    summary = processor.get_summary()

    # 3. Verify total page count
    print(f"\n[Test 3] Verifying total page count...")
    print(f"  • Total pages reported: {summary['total_pages']}")
    assert summary["total_pages"] == 50, f"Expected 50 pages, but got {summary['total_pages']}"
    print("  ✅ Total pages count is exactly 50.")

    # 4. Verify page numbers tracking
    print("\n[Test 4] Verifying page numbers and data integrity...")
    assert len(processor.pages) == 50, "Extracted pages list does not match 50 items."
    for idx, page_data in enumerate(processor.pages):
        expected_page_num = idx + 1
        assert page_data.page_number == expected_page_num, (
            f"Page mismatch at index {idx}: got {page_data.page_number}, expected {expected_page_num}"
        )
    print(f"  ✅ All 50 pages properly tracked (Page 1 through Page 50).")

    # 5. Verify text presence and empty page handling
    print("\n[Test 5] Checking text extraction and empty page counts...")
    print(f"  • Pages with text : {summary['pages_with_text']}")
    print(f"  • Empty pages     : {summary['empty_pages']}")
    assert summary["pages_with_text"] == 50, (
        f"Expected 50 pages with text, got {summary['pages_with_text']}"
    )
    assert summary["empty_pages"] == 0, (
        f"Expected 0 empty pages, got {summary['empty_pages']}"
    )
    print("  ✅ All 50 pages successfully contain extracted text.")

    # 6. Sample text verification from Page 1
    print("\n[Test 6] Inspecting sample text from Page 1...")
    sample_text = processor.get_first_page_sample()
    assert sample_text is not None and len(sample_text) > 0, "Page 1 sample text is empty!"
    assert "Broad-Gauge" in sample_text or "Dalhousie" in sample_text or "Iron Tracks" in sample_text, (
        "Expected historical keywords not found in Page 1 text."
    )
    print("  ✅ Sample text contains expected historical content.")

    print("\n" + "=" * 70)
    print("📄 SAMPLE EXTRACTED TEXT (PAGE 1):")
    print("=" * 70)
    print(sample_text.strip())
    print("=" * 70)
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_pdf_extraction()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)

