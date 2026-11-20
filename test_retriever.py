"""
Test Script for Task 1: FAISS Retriever
=======================================
Verifies:
1. FAISS index loads properly from vector_store/.
2. Retrieves top 4 relevant chunks for a question.
3. Every chunk preserves page-number metadata.
4. Content is relevant to the question.
"""

import sys
from pathlib import Path
from retriever import get_faiss_retriever, retrieve_relevant_chunks


def test_faiss_retriever():
    print("=" * 70)
    print("🧪 Running Task 1: FAISS Document Retriever Test")
    print("=" * 70)

    vector_store_path = Path("vector_store")
    assert vector_store_path.exists(), f"vector_store directory not found at {vector_store_path}"

    test_query = "What gauge was standardized for Indian railways?"
    print(f"\n[Test 1] Querying retriever with: \"{test_query}\" (k=4)...")

    results = retrieve_relevant_chunks(test_query, vector_store_path=vector_store_path, k=4)

    # 1. Check count
    assert len(results) == 4, f"Expected 4 chunks, got {len(results)}"
    print(f"  ✅ Successfully retrieved exactly 4 chunks.")

    # 2. Check metadata
    print("\n[Test 2] Checking page-number metadata on all retrieved chunks...")
    for i, doc in enumerate(results, 1):
        assert "page" in doc.metadata, f"Chunk {i} is missing 'page' metadata!"
        page_num = doc.metadata["page"]
        assert isinstance(page_num, int) and 1 <= page_num <= 50, f"Invalid page number: {page_num}"
        print(f"  • Result #{i}: Page {page_num} | Length: {len(doc.page_content)} chars")

    print("  ✅ All retrieved chunks preserve valid page numbers.")

    # 3. Check relevance
    print("\n[Test 3] Verifying content relevance...")
    combined_content = " ".join(doc.page_content for doc in results)
    assert "Broad-Gauge" in combined_content or "5 Feet 6 Inches" in combined_content or "Dalhousie" in combined_content, (
        "Expected historical keywords not found in retrieved chunks!"
    )
    print("  ✅ Retrieved chunks contain relevant context (Broad-Gauge / Dalhousie).")

    print("\n" + "=" * 70)
    print("📄 SAMPLE RETRIEVED CHUNK (Rank 1):")
    print("=" * 70)
    top_doc = results[0]
    print(f"Page Number : {top_doc.metadata['page']}")
    print(f"Metadata    : {top_doc.metadata}")
    print("-" * 70)
    print(top_doc.page_content[:400] + "...")
    print("=" * 70)
    print("\n🎉 TASK 1: RETRIEVER TEST PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_faiss_retriever()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
