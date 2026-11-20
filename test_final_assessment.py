"""
Phase 7 Final Assessment Test Suite
===================================
Tests all 5 benchmark assessment questions against the Hybrid RAG assistant:
1. Guaranteed interest rate under 1849 contract (Expected: PDF, Page 13, 5% per annum)
2. Bhor Ghat engineering obstacle & gradient (Expected: PDF, Page 26, Battery Hill, gradient ratio 1 in 37)
3. Vande Bharat Executive Chair Car ticket price New Delhi - Varanasi (Expected: External Web, railway fare source, dynamic pricing disclaimer)
4. Diamond Harbour experimental telegraph cable supervisor (Expected: PDF, Page 43, Dr. William Brooke O'Shaughnessy)
5. 2022 FIFA World Cup winner (Expected: External Web, FIFA/encyclopedia source, Argentina)
"""

import sys
from rag_pipeline import HistoricalRAGPipeline


def run_final_assessment():
    print("=" * 80)
    print("🧪 Running Phase 7 Final Assessment Test Suite")
    print("=" * 80)

    pipeline = HistoricalRAGPipeline(vector_store_path="vector_store", k=4)

    test_cases = [
        {
            "id": 1,
            "question": "What exact annual guaranteed interest rate was promised to private British railway companies under the 1849 contract terms?",
            "expected_source": "PDF",
            "expected_page": 13,
            "must_contain": ["5%", "per annum"],
        },
        {
            "id": 2,
            "question": "What engineering obstacle on the Bhor Ghat incline required a reversing station, and what was its gradient ratio?",
            "expected_source": "PDF",
            "expected_page": 26,
            "must_contain": ["Battery Hill", "1 in 37"],
        },
        {
            "id": 3,
            "question": "What is the Executive Chair Car ticket price between New Delhi and Varanasi on Vande Bharat Express?",
            "expected_source": "External Web",
            "expected_page": None,
            "must_contain": ["Executive Chair Car", "Vande Bharat"],
        },
        {
            "id": 4,
            "question": "Who supervised the experimental Diamond Harbour underwater telegraph cable in 1851?",
            "expected_source": "PDF",
            "expected_page": 43,
            "must_contain": ["William Brooke O'Shaughnessy", "Calcutta Mint"],
        },
        {
            "id": 5,
            "question": "Which country won the 2022 FIFA World Cup?",
            "expected_source": "External Web",
            "expected_page": None,
            "must_contain": ["Argentina"],
        },
    ]

    report = []

    for tc in test_cases:
        print(f"\n" + "=" * 80)
        print(f"📌 QUESTION #{tc['id']}")
        print(f"Q: {tc['question']}")
        print(f"Expected Source: {tc['expected_source']}")
        print("-" * 80)

        response = pipeline.query(tc["question"])

        print(f"Selected Source Type: {response.source_type}")
        print(f"Answer:\n{response.answer}\n")

        # 1. Verify Source Type
        assert response.source_type == tc["expected_source"], (
            f"Expected {tc['expected_source']}, got {response.source_type}"
        )

        # 2. Verify Citations
        if tc["expected_source"] == "PDF":
            assert tc["expected_page"] in response.cited_pages, (
                f"Expected Page {tc['expected_page']} in cited pages {response.cited_pages}"
            )
            print(f"✅ Cited Pages: {response.cited_pages} (Target Page {tc['expected_page']} verified)")
        else:
            assert len(response.web_sources) > 0, "No web citations found!"
            print(f"✅ Web Sources Cited ({len(response.web_sources)}):")
            for ws in response.web_sources:
                print(f"   • {ws['title']} -> {ws['url']}")

        # 3. Verify Content
        norm_answer = response.answer.lower().replace("’", "'")
        for kw in tc["must_contain"]:
            norm_kw = kw.lower().replace("’", "'")
            assert norm_kw in norm_answer, (
                f"Missing expected fact '{kw}' in answer!"
            )
        print(f"✅ Required facts verified: {tc['must_contain']}")

        report.append({
            "id": tc["id"],
            "question": tc["question"],
            "source_type": response.source_type,
            "answer": response.answer,
            "cited_pages": response.cited_pages,
            "web_sources": response.web_sources,
        })

    print("\n" + "=" * 80)
    print("🏆 FINAL ASSESSMENT SUMMARY: 5/5 PASSED")
    print("=" * 80)
    for r in report:
        src = f"PDF (Page {r['cited_pages']})" if r["source_type"] == "PDF" else f"External Web ({len(r['web_sources'])} URLs)"
        print(f"Q{r['id']}: {r['question'][:65]}... -> [{src}] ✅ PASS")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_final_assessment()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
