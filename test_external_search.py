"""
Test Script for Task 1: External Search Integration
===================================================
Verifies:
1. Tavily API key is loaded securely from .env (never printed or exposed).
2. Tavily web search connects and retrieves live web results.
3. Every result preserves source title, valid URL, and relevant content.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from web_search import get_tavily_api_key, search_web, format_web_results_for_context


def test_external_search():
    print("=" * 70)
    print("🧪 Running Task 1: External Search Integration Test")
    print("=" * 70)

    # 1. Verify API Key
    print("\n[Test 1] Verifying Tavily API key configuration in .env...")
    load_dotenv()
    key = get_tavily_api_key()
    assert key is not None and len(key) > 0, "TAVILY_API_KEY is missing!"
    assert key.startswith("tvly-"), "TAVILY_API_KEY does not have expected 'tvly-' prefix!"
    print(f"  ✅ TAVILY_API_KEY detected securely (length: {len(key)} chars, masked: tvly-***).")

    # 2. Test Live Web Search
    test_query = "Who won the 2022 FIFA World Cup?"
    print(f"\n[Test 2] Querying external web search: \"{test_query}\"...")

    results = search_web(test_query, max_results=3)
    assert len(results) > 0, "Web search returned 0 results!"
    print(f"  ✅ Retrieved {len(results)} live web results.")

    # 3. Verify Titles, URLs, and Content
    print("\n[Test 3] Verifying source metadata (titles, URLs, snippets)...")
    for idx, r in enumerate(results, 1):
        assert len(r.title) > 0, f"Result #{idx} missing title!"
        assert r.url.startswith("http://") or r.url.startswith("https://"), (
            f"Result #{idx} has invalid URL: {r.url}"
        )
        assert len(r.content) > 0, f"Result #{idx} has empty content!"
        print(f"  • Source #{idx}: {r.title}")
        print(f"    URL: {r.url}")

    # Check for expected fact in results
    combined_content = " ".join(r.content for r in results).lower()
    assert "argentina" in combined_content, "Expected term 'Argentina' not found in search results!"
    print(f"  ✅ Relevant fact ('Argentina') confirmed in retrieved search content.")

    # 4. Check Context Formatting
    print("\n[Test 4] Verifying context formatting...")
    formatted = format_web_results_for_context(results)
    assert "Web Source #" in formatted, "Formatted context missing source headers!"
    print("  ✅ Context formatting verified.")

    print("\n" + "=" * 70)
    print("🎉 TASK 1: EXTERNAL SEARCH TEST PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_external_search()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
