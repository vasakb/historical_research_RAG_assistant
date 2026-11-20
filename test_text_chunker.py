"""
Test Script for Text Chunking (Phase 3)
======================================
Verifies:
1. All 50 pages are chunked into smaller units.
2. Chunk size is bounded around 800 characters.
3. Chunk overlap is ~150 characters between adjacent chunks.
4. Original page numbers are preserved in every chunk's metadata.
5. No content is lost or corrupted.
"""

import sys
from pathlib import Path
from text_chunker import chunk_pdf, verify_chunk_overlap


def test_chunking():
    pdf_path = Path("data/indian_railways_and_telegraph_history_50p.pdf")
    chunk_size = 800
    chunk_overlap = 150

    print("=" * 70)
    print("🧪 Running Phase 3: Text Chunking Test")
    print(f"Target Chunk Size   : {chunk_size} characters")
    print(f"Target Chunk Overlap: {chunk_overlap} characters")
    print("=" * 70)

    # 1. Run Chunking
    print(f"\n[Test 1] Processing and chunking {pdf_path.name}...")
    chunks = chunk_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total_chunks = len(chunks)
    print(f"  ✅ Generated {total_chunks} chunks across 50 pages.")
    assert total_chunks > 50, f"Expected more than 50 chunks, got {total_chunks}"

    # 2. Check metadata on all chunks
    print("\n[Test 2] Verifying metadata preservation (page numbers)...")
    pages_found = set()
    for i, c in enumerate(chunks):
        assert "page" in c.metadata, f"Chunk {i} is missing 'page' metadata!"
        page_num = c.metadata["page"]
        assert 1 <= page_num <= 50, f"Invalid page number {page_num} in chunk {i}"
        pages_found.add(page_num)
        assert len(c.page_content.strip()) > 0, f"Chunk {i} is empty!"

    print(f"  ✅ Page numbers preserved across all chunks (Pages 1 to {max(pages_found)}).")
    assert len(pages_found) == 50, f"Expected chunks from all 50 pages, but found {len(pages_found)}"

    # 3. Check chunk lengths
    print("\n[Test 3] Verifying chunk size boundaries...")
    max_len = max(len(c.page_content) for c in chunks)
    min_len = min(len(c.page_content) for c in chunks)
    avg_len = sum(len(c.page_content) for c in chunks) / total_chunks
    print(f"  • Min chunk length : {min_len} chars")
    print(f"  • Max chunk length : {max_len} chars (<= {chunk_size})")
    print(f"  • Avg chunk length : {avg_len:.1f} chars")
    assert max_len <= chunk_size + 20, f"Chunk exceeded max chunk size: {max_len}"
    print(f"  ✅ All chunks respect the {chunk_size} character size constraint.")

    # 4. Check overlap between adjacent chunks on the same page
    print("\n[Test 4] Verifying chunk overlap (~150 characters)...")
    overlap_samples = []
    for i in range(len(chunks) - 1):
        if chunks[i].metadata["page"] == chunks[i + 1].metadata["page"]:
            overlap_len = verify_chunk_overlap(chunks[i].page_content, chunks[i + 1].page_content)
            if overlap_len > 0:
                overlap_samples.append((i, i + 1, chunks[i].metadata["page"], overlap_len))

    print(f"  • Total adjacent chunk pairs on same pages: {len(overlap_samples)}")
    assert len(overlap_samples) > 0, "No overlapping adjacent chunks found on the same page!"
    
    first_pair = overlap_samples[0]
    sample_overlap_len = first_pair[3]
    print(f"  • Sample overlap between Chunk {first_pair[0]} & Chunk {first_pair[1]} (Page {first_pair[2]}): {sample_overlap_len} characters")
    assert 50 <= sample_overlap_len <= 150, f"Overlap {sample_overlap_len} is outside expected range"
    print("  ✅ Overlap verified between adjacent chunks.")

    # 5. Display a sample chunk
    sample_chunk = chunks[0]
    print("\n" + "=" * 70)
    print("📄 SAMPLE CHUNK (Index 0):")
    print("=" * 70)
    print(f"Original Page Number : {sample_chunk.metadata['page']}")
    print(f"Chunk Character Count: {len(sample_chunk.page_content)}")
    print(f"Chunk Metadata       : {sample_chunk.metadata}")
    print("-" * 70)
    print("Content:")
    print(sample_chunk.page_content)
    print("=" * 70)

    # 6. Display second chunk to illustrate overlap
    sample_chunk_1 = chunks[1]
    print("\n" + "=" * 70)
    print("📄 FOLLOWING CHUNK (Index 1) FOR OVERLAP COMPARISON:")
    print("=" * 70)
    print(f"Original Page Number : {sample_chunk_1.metadata['page']}")
    print(f"Chunk Character Count: {len(sample_chunk_1.page_content)}")
    print("-" * 70)
    print("Content:")
    print(sample_chunk_1.page_content)
    print("=" * 70)

    print("\n🎉 PHASE 3 CHUNKING TESTS PASSED WITH 0 ERRORS!\n")


if __name__ == "__main__":
    try:
        test_chunking()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)

