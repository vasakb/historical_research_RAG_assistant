"""
Groq LLM Integration Module
============================
Connects to the Groq API using LangChain's ChatGroq.
Loads API key securely from .env without exposing or logging it.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()


def get_groq_llm(
    model_name: str = "qwen/qwen3.8-27b",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> ChatGroq:
    """
    Initialize and return a ChatGroq LLM instance.

    Args:
        model_name (str): Groq-supported model identifier (defaults to 'qwen/qwen3.8-27b').
        temperature (float): Sampling temperature (default 0.2 for factual historical responses).
        api_key (Optional[str]): Explicit API key or loads GROQ_API_KEY from environment.

    Returns:
        ChatGroq: Initialized LangChain ChatGroq model instance.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Groq API key not found. Please set GROQ_API_KEY in your .env file."
        )

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=key,
        max_retries=2,
    )


def test_llm_connection(prompt: str = "Explain the role of railways in history in one sentence.") -> str:
    """
    Send a simple test prompt to the Groq LLM to verify connectivity and authentication.

    Args:
        prompt (str): A brief test prompt.

    Returns:
        str: Response text from the LLM.
    """
    llm = get_groq_llm()
    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":
    print("Testing Groq LLM connection...")
    reply = test_llm_connection()
    print(f"Groq LLM response:\n{reply}")
