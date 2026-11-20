"""
Test Script for Phase 4: Embeddings & FAISS Vector Database
===========================================================
Verifies:
1. Loads 147 chunks from Phase 3 text chunker.
2. Initializes Hugging Face embedding model (sentence-transformers/all-MiniLM-L6-v2).
3. Verifies embedding vector dimensions (384).
4. Creates FAISS vector database and stores all chunk vectors with metadata.
5. Persists FAISS database locally into vector_store/.
6. Successfully loads the persisted FAISS database from disk.
7. Performs similarity search for a historical query and retrieves the 4 most relevant chunks.
"""

import sys
from pathlib import Path
from embeddings import get_embedding_model
from text_chunker import chunk_pdf
from vector_db import create_vector_store, load_vector_store, similarity_search_with_score


def test_phase_4_vector_db():
    pdf_path = Path("data/indian_railways_and_telegraph_history_50p.pdf")
    vector_store_dir = Path("vector_store")
    sample_query = "What was the broad gauge standardization and Lord Dalhousie railway minute?"

    print("=" * 75)
    print("🧪 Running Phase 4: Embeddings & FAISS Vector Database Test")
    print("=" * 75)

    # 1. Generate text chunks from Phase 3
    print("\n[Step 1] Generating Phase 3 text chunks from 50-page PDF...")
    assert pdf_path.exists(), f"PDF not found at {pdf_path}"
    chunks = chunk_pdf(pdf_path, chunk_size=800, chunk_overlap=150)
    total_chunks = len(chunks)
    print(f"  ✅ Extracted & chunked: {total_chunks} chunks ready for embedding.")
    assert total_chunks > 0, "No chunks generated!"

    # 2. Initialize Embedding Model
    print("\n[Step 2] Initializing Hugging Face embedding model (all-MiniLM-L6-v2)...")
    embeddings = get_embedding_model()
    test_vec = embeddings.embed_query("Historical railway infrastructure in colonial India")
    embedding_dim = len(test_vec)
    print(f"  ✅ Embedding model ready. Vector dimension: {embedding_dim}")
    assert embedding_dim == 384, f"Expected 384 dimensions, got {embedding_dim}"

    # 3. Create FAISS Vector Store & Save
    print(f"\n[Step 3] Embedding {total_chunks} chunks and building FAISS vector database...")
    vector_store = create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        save_path=vector_store_dir,
    )
    print(f"  ✅ FAISS database created and saved to: {vector_store_dir.resolve()}")
    assert (vector_store_dir / "index.faiss").exists(), "index.faiss was not saved!"
    assert (vector_store_dir / "index.pkl").exists(), "index.pkl was not saved!"
    print(f"  • Found saved files: index.faiss ({ (vector_store_dir / 'index.faiss').stat().st_size:,} bytes) and index.pkl")

    # 4. Test Loading Saved Database
    print(f"\n[Step 4] Loading FAISS database from disk ({vector_store_dir})...")
    loaded_vector_store = load_vector_store(
        load_path=vector_store_dir,
        embeddings=embeddings,
    )
    print(f"  ✅ Successfully loaded persisted vector store into memory.")

    # 5. Perform Similarity Search (k=4)
    print(f"\n[Step 5] Running similarity search on loaded database...")
    print(f"  • Sample Query: \"{sample_query}\"")
    print(f"  • Top k: 4")

    search_results = similarity_search_with_score(
        query=sample_query,
        vector_store=loaded_vector_store,
        k=4,
    )

    assert len(search_results) == 4, f"Expected 4 results, got {len(search_results)}"
    print(f"  ✅ Retrieved {len(search_results)} most relevant chunks.")

    print("\n" + "=" * 75)
    print(f"🔍 TOP 4 SIMILARITY SEARCH RESULTS FOR: \"{sample_query}\"")
    print("=" * 75)

    for rank, (doc, score) in enumerate(search_results, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        print(f"\nResult #{rank} | Page: {page_num} | Chunk ID: {chunk_id} | Distance Score: {score:.4f}")
        print("-" * 75)
        # Show first 250 characters of chunk
        preview = doc.page_content.strip().replace("\n", " ")
        print(f"{preview[:300]}...")

    print("\n" + "=" * 75)
    print("🎉 PHASE 4 TEST COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    try:
        test_phase_4_vector_db()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)

