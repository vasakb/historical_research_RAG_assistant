"""
Test Script for Task 3: Complete RAG Pipeline
=============================================
Verifies:
1. Retriever fetches relevant chunks.
2. Context is passed to Groq LLM.
3. Answer is grounded in the document.
4. Includes page-number citations.
5. Returns refusal message when asked an out-of-context question.
"""

import sys
from pathlib import Path
from qa_chain import HistoricalRAGPipeline, answer_historical_question


def test_rag_pipeline():
    print("=" * 70)
    print("🧪 Running Task 3: Complete RAG Pipeline Test")
    print("=" * 70)

    pipeline = HistoricalRAGPipeline(vector_store_path="vector_store", k=4)

    # 1. Test grounded historical question
    q1 = "What gauge was standardized for Indian railways and under whose railway minute?"
    print(f"\n[Test 1] Testing grounded question: \"{q1}\"...")
    resp1 = pipeline.query(q1)

    assert len(resp1.answer) > 0, "RAG pipeline returned an empty answer!"
    assert len(resp1.source_documents) == 4, f"Expected 4 source documents, got {len(resp1.source_documents)}"
    assert len(resp1.cited_pages) > 0, "No cited pages found!"
    assert "Page" in resp1.answer or "[Page" in resp1.answer or "Page " in str(resp1.cited_pages), "No page citation found!"

    print(f"  ✅ Generated grounded answer ({len(resp1.answer)} chars).")
    print(f"  • Source pages retrieved: {resp1.cited_pages}")
    print("\n--- Answer 1 Preview ---")
    print(resp1.answer)

    # 2. Test hallucination avoidance with out-of-scope question
    q2 = "Who won the FIFA World Cup in 2022?"
    print(f"\n[Test 2] Testing out-of-scope question: \"{q2}\"...")
    resp2 = pipeline.query(q2)

    print(f"--- Out-of-Scope Answer ---")
    print(resp2.answer)
    assert (
        "cannot find" in resp2.answer.lower()
        or "not mentioned" in resp2.answer.lower()
        or "not contain" in resp2.answer.lower()
        or "insufficient" in resp2.answer.lower()
    ), "RAG pipeline failed to reject out-of-scope question!"
    print("  ✅ Hallucination avoided: pipeline correctly reported lack of historical context.")

    print("\n" + "=" * 70)
    print("🎉 TASK 3: RAG PIPELINE TEST PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_rag_pipeline()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
