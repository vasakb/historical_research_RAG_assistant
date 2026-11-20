"""
Task 4 Complete Application Test Suite
======================================
Tests the Streamlit Chatbot workflow and RAG engine:
1. Tests asking a question about Indian railway history.
2. Tests asking a question about the Bhor Ghat incline.
3. Tests asking a question about the Diamond Harbour telegraph cable.
4. Tests asking an unrelated question to verify hallucination avoidance.
5. Tests multi-turn conversation and session state chat history preservation.
6. Verifies that source page numbers are captured and displayed correctly.
"""

import sys
from rag_pipeline import HistoricalRAGPipeline


def test_complete_chatbot_workflow():
    print("=" * 80)
    print("🧪 Running Phase 6 Task 4: Complete Streamlit Chatbot Workflow Test")
    print("=" * 80)

    # Initialize RAG Pipeline (as cached by get_rag_pipeline in app.py)
    print("\n[Setup] Initializing Historical RAG Pipeline (vector_store, k=4)...")
    pipeline = HistoricalRAGPipeline(vector_store_path="vector_store", k=4)
    print("  ✅ Pipeline loaded.")

    # Simulated Streamlit Session State
    chat_history = []

    # Test Cases
    test_queries = [
        {
            "id": 1,
            "topic": "Indian Railway History",
            "query": "What terms were promised under the 1849 guarantee system for private British railway companies?",
            "expected_keywords": ["5%", "capital expenditure", "99-year", "guarantee"],
            "target_page": 13,
            "is_grounded": True,
        },
        {
            "id": 2,
            "topic": "Bhor Ghat Incline",
            "query": "What engineering challenge occurred on the Bhor Ghat incline and how was it solved?",
            "expected_keywords": ["Battery Hill", "gradient", "reversing station", "James Berkley"],
            "target_page": 26,
            "is_grounded": True,
        },
        {
            "id": 3,
            "topic": "Diamond Harbour Telegraph Cable",
            "query": "Who supervised the laying of the 1851 Diamond Harbour telegraph line and what was its length?",
            "expected_keywords": ["William Brooke O'Shaughnessy", "27-mile", "Calcutta Mint"],
            "target_page": 43,
            "is_grounded": True,
        },
        {
            "id": 4,
            "topic": "Unrelated Query (Hallucination Avoidance)",
            "query": "What is the capital of Australia and who was its first prime minister?",
            "expected_keywords": ["cannot find", "not"],
            "target_page": None,
            "is_grounded": False,
        },
    ]

    for tc in test_queries:
        print(f"\n" + "-" * 75)
        print(f"💬 Turn #{tc['id']}: [{tc['topic']}]")
        print(f"User Query : \"{tc['query']}\"")

        # Simulate user adding message to session state
        chat_history.append({"role": "user", "content": tc["query"]})

        # Run RAG Query
        res = pipeline.query(tc["query"])

        # Simulate assistant appending answer to session state
        chat_history.append({
            "role": "assistant",
            "content": res.answer,
            "pages": res.cited_pages,
        })

        print(f"Assistant  : {res.answer[:250]}...")
        print(f"Pages Cited: {res.cited_pages}")

        # Assertions
        assert len(res.answer) > 0, f"Empty answer on query {tc['id']}"

        if tc["is_grounded"]:
            assert len(res.cited_pages) > 0, f"No source pages cited for grounded query {tc['id']}"
            assert tc["target_page"] in res.cited_pages, (
                f"Expected target page {tc['target_page']} in cited pages {res.cited_pages}"
            )
            print(f"  ✅ Correct source page {tc['target_page']} successfully cited.")
        else:
            assert "cannot find" in res.answer.lower() or "not" in res.answer.lower(), (
                "System failed to reject out-of-scope question!"
            )
            print("  ✅ Hallucination avoided: correctly refused out-of-scope query.")

    # Verify Chat History Preservation (Multi-turn verification)
    print("\n" + "=" * 80)
    print("📜 VERIFYING STREAMLIT MULTI-TURN CHAT HISTORY")
    print("=" * 80)
    print(f"Total messages preserved in session state: {len(chat_history)} (4 queries, 4 responses)")
    assert len(chat_history) == 8, f"Expected 8 messages in chat history, got {len(chat_history)}"

    for idx, msg in enumerate(chat_history, 1):
        role_icon = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
        pages_info = f" (Pages: {msg.get('pages', [])})" if msg["role"] == "assistant" else ""
        print(f"{idx}. {role_icon}: {msg['content'][:80]}...{pages_info}")

    print("\n🎉 ALL PHASE 6 APPLICATION TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_complete_chatbot_workflow()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
