"""
Test Script for Task 2: Hybrid Question Answering
=================================================
Verifies:
1. Historical PDF question routes to "PDF", cites pages, and generates grounded answer.
2. Contemporary question routes to "External Web", cites titles & URLs.
3. Distinction between source types is preserved.
"""

import sys
from rag_pipeline import HistoricalRAGPipeline


def test_hybrid_qa():
    print("=" * 70)
    print("🧪 Running Task 2: Hybrid Question Answering Test")
    print("=" * 70)

    pipeline = HistoricalRAGPipeline(vector_store_path="vector_store", k=4)

    # 1. Test PDF Question
    q_pdf = "What exact annual guaranteed interest rate was promised to private British railway companies under the 1849 contract terms?"
    print(f"\n[Test 1] Testing PDF question: \"{q_pdf}\"...")
    resp_pdf = pipeline.query(q_pdf)

    print(f"  • Source Type: {resp_pdf.source_type}")
    print(f"  • Cited Pages: {resp_pdf.cited_pages}")
    print(f"  • Answer:\n{resp_pdf.answer}\n")

    assert resp_pdf.source_type == "PDF", f"Expected source_type 'PDF', got {resp_pdf.source_type}"
    assert 13 in resp_pdf.cited_pages, f"Expected Page 13 in cited pages, got {resp_pdf.cited_pages}"
    assert "5%" in resp_pdf.answer, "Expected '5%' interest rate in answer!"
    print("  ✅ PDF Question verified: routed to PDF, accurately cited Page 13.")

    # 2. Test External Web Question
    q_web = "Which country won the 2022 FIFA World Cup?"
    print(f"\n[Test 2] Testing External question: \"{q_web}\"...")
    resp_web = pipeline.query(q_web)

    print(f"  • Source Type: {resp_web.source_type}")
    print(f"  • Web Sources ({len(resp_web.web_sources)}):")
    for s in resp_web.web_sources:
        print(f"    - {s['title']} ({s['url']})")
    print(f"  • Answer:\n{resp_web.answer}\n")

    assert resp_web.source_type == "External Web", f"Expected source_type 'External Web', got {resp_web.source_type}"
    assert len(resp_web.web_sources) > 0, "No web sources returned!"
    assert "argentina" in resp_web.answer.lower(), "Expected 'Argentina' in answer!"
    print("  ✅ External Question verified: routed to Web Search with source URLs.")

    print("\n" + "=" * 70)
    print("🎉 TASK 2: HYBRID QUESTION ANSWERING TEST PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_hybrid_qa()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
