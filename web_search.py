"""
External Web Search Module
==========================
Integrates Tavily Web Search API to retrieve reliable web sources,
preserving page titles, URLs, and text content for hybrid RAG citation.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()


@dataclass
class WebSearchResult:
    """
    Container for an individual web search result.

    Attributes:
        title (str): Title of the webpage.
        url (str): Source web link (URL).
        content (str): Relevant snippet / text extracted from the page.
        score (Optional[float]): Relevance score if provided by the search API.
    """
    title: str
    url: str
    content: str
    score: Optional[float] = None


def get_tavily_api_key() -> str:
    """
    Retrieve Tavily API key from environment securely without exposing or logging it.

    Returns:
        str: Valid Tavily API key.
    """
    key = os.getenv("TAVILY_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Tavily API key not found. Please set TAVILY_API_KEY in your .env file."
        )
    return key.strip()


def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> List[WebSearchResult]:
    """
    Execute a web search using Tavily API and return structured results with titles and URLs.

    Args:
        query (str): The search query.
        max_results (int): Number of web results to retrieve (default 5).
        search_depth (str): 'basic' or 'advanced' (default 'basic').

    Returns:
        List[WebSearchResult]: List of search results with titles, URLs, and snippet text.
    """
    api_key = get_tavily_api_key()

    # Attempt search via official tavily client first, fallback to direct REST API
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=False,
        )
        raw_results = response.get("results", [])
    except Exception:
        import requests
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        raw_results = resp.json().get("results", [])

    results: List[WebSearchResult] = []
    for item in raw_results:
        results.append(
            WebSearchResult(
                title=item.get("title", "Untitled Source").strip(),
                url=item.get("url", "").strip(),
                content=item.get("content", "").strip(),
                score=item.get("score"),
            )
        )

    return results


def format_web_results_for_context(results: List[WebSearchResult]) -> str:
    """
    Format web search results into clean context blocks for the LLM.

    Args:
        results (List[WebSearchResult]): List of search results.

    Returns:
        str: Formatted context blocks with source title and URL citations.
    """
    if not results:
        return "No external web sources found."

    formatted_blocks = []
    for idx, r in enumerate(results, 1):
        formatted_blocks.append(
            f"--- [Web Source #{idx}: {r.title}] ---\n"
            f"URL: {r.url}\n"
            f"Content: {r.content}\n"
        )
    return "\n\n".join(formatted_blocks)


if __name__ == "__main__":
    q = "2022 FIFA World Cup winner"
    print(f"Testing web search for: '{q}'...")
    res = search_web(q, max_results=3)
    print(f"Retrieved {len(res)} results:")
    for r in res:
        print(f"- {r.title} ({r.url})")
