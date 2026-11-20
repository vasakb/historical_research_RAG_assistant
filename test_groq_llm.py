"""
Test Script for Task 2: Groq LLM Integration
============================================
Verifies:
1. Groq API key is successfully loaded from .env.
2. The key is never printed or exposed in output.
3. ChatGroq connects to Groq API and generates a valid test response.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from llm import get_groq_llm, test_llm_connection


def test_groq_integration():
    print("=" * 70)
    print("🧪 Running Task 2: Groq LLM Integration Test")
    print("=" * 70)

    # 1. Check API Key presence without printing it
    print("\n[Test 1] Checking Groq API key configuration in .env...")
    load_dotenv()
    import os
    key = os.getenv("GROQ_API_KEY")
    assert key is not None and len(key.strip()) > 0, "GROQ_API_KEY is missing from environment!"
    assert key.startswith("gsk_"), "GROQ_API_KEY does not start with expected 'gsk_' prefix!"
    print(f"  ✅ GROQ_API_KEY detected securely (length: {len(key)} chars, secret masked: gsk_***).")

    # 2. Initialize LLM instance
    print("\n[Test 2] Initializing ChatGroq instance...")
    llm = get_groq_llm(model_name="qwen/qwen3.8-27b", temperature=0.2)
    assert llm is not None, "Failed to initialize ChatGroq instance"
    print(f"  ✅ ChatGroq initialized with model '{llm.model_name}'.")

    # 3. Test basic LLM response
    print("\n[Test 3] Sending basic test query to Groq LLM...")
    prompt = "In 20 words or less, what is the role of railways in history?"
    response = llm.invoke(prompt)
    response_text = response.content.strip()

    assert len(response_text) > 0, "LLM returned an empty response!"
    print(f"  ✅ Received valid response from Groq API ({len(response_text)} chars).")

    print("\n" + "=" * 70)
    print("🤖 GROQ LLM TEST RESPONSE:")
    print("=" * 70)
    print(response_text)
    print("=" * 70)
    print("\n🎉 TASK 2: GROQ LLM INTEGRATION TEST PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    try:
        test_groq_integration()
    except AssertionError as ae:
        print(f"\n❌ Assertion Error: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)
