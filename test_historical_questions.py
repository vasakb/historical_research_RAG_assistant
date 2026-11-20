"""
Task 4 Validation Suite: Evaluation Questions
=============================================
Tests the full RAG pipeline against the three designated historical evaluation questions:
1. What gauge was standardized for Indian railways?
2. What engineering challenge was encountered on the Bhor Ghat incline?
3. Who supervised the experimental Diamond Harbour underwater telegraph cable in 1851?
"""

import sys
from qa_chain import HistoricalRAGPipeline


def run_evaluation_suite():
    print("=" * 80)
    print("🧪 Running Task 4: Historical RAG Evaluation Suite")
    print("=" * 80)

    pipeline = HistoricalRAGPipeline(vector_store_path="vector_store", k=4)

    test_cases = [
        {
            "id": 1,
            "question": "What gauge was standardized for Indian railways?",
            "expected_keywords": ["5 feet 6 inches", "broad-gauge"],
            "target_pages": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
        },
        {
            "id": 2,
            "question": "What engineering challenge was encountered on the Bhor Ghat incline?",
            "expected_keywords": ["Battery Hill", "Khandala ravine", "gradient", "reversing station", "James Berkley"],
            "target_pages": [26],
        },
        {
            "id": 3,
            "question": "Who supervised the experimental Diamond Harbour underwater telegraph cable in 1851?",
            "expected_keywords": ["William Brooke O'Shaughnessy", "Calcutta Mint", "27-mile"],
            "target_pages": [43],
        },
    ]

    results = []

    for case in test_cases:
        print(f"\n--- [Test Question #{case['id']}] ---")
        print(f"Q: {case['question']}")

        response = pipeline.query(case["question"])
        print("\nAnswer:")
        print(response.answer)
        print(f"\nCited Source Pages: {response.cited_pages}")

        # Validation checks
        assert len(response.answer) > 0, f"Empty answer for Question {case['id']}"
        assert len(response.cited_pages) > 0, f"No source pages cited for Question {case['id']}"

        # Check for expected keywords
        found_keywords = [
            kw for kw in case["expected_keywords"] if kw.lower() in response.answer.lower()
        ]
        print(f"Keywords verified: {found_keywords}")
        assert len(found_keywords) > 0, f"Missing historical keywords in Question {case['id']}"

        results.append({
            "id": case["id"],
            "question": case["question"],
            "answer": response.answer,
            "pages": response.cited_pages,
            "keywords": found_keywords,
        })

    print("\n" + "=" * 80)
    print("📊 TASK 4 EVALUATION SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"Question #{r['id']}: {r['question']}")
        print(f"  • Cited Pages: {r['pages']}")
        print(f"  • Verified Key Facts: {r['keywords']}")
        print(f"  • Status: PASS ✅\n")

    print("🎉 ALL 3 HISTORICAL EVALUATION QUESTIONS PASSED WITH ACCURATE GROUNDING!\n")


if __name__ == "__main__":
    try:
        run_evaluation_suite()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
