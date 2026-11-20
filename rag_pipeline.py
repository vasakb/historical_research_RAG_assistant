"""
Hybrid RAG Pipeline Module
==========================
Combines FAISS vector retrieval on historical PDF documents with Tavily external
web search, dynamically routing questions to the appropriate source.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vector_db import load_vector_store, similarity_search_with_score
from llm import get_groq_llm
from web_search import search_web, format_web_results_for_context, WebSearchResult


PDF_SYSTEM_PROMPT = """You are a meticulous historical research assistant analyzing official archives on 19th-century Indian railways and telegraphs.

Instructions:
1. Answer the user's question using ONLY the retrieved historical excerpts provided below.
2. For every factual claim, explicitly cite the original page number using the format [Page X] (e.g., [Page 13]).
3. If the provided context does not contain sufficient information to answer the question, clearly state:
   "I cannot find the answer to this question in the provided historical document."
4. Do not speculate, extrapolate, or introduce outside historical knowledge not stated in the context.

Retrieved Historical Context:
{context}
"""

WEB_SYSTEM_PROMPT = """You are a reliable research assistant answering questions using external web search results.

Instructions:
1. Answer the question accurately using the provided web search excerpts.
2. For train fares or ticket prices (such as the Vande Bharat Express), state the typical or published Executive Chair Car (EC) fare range, identify the train route/number where available, and clearly explain that exact prices vary depending on travel date, dynamic pricing, and catering options.
3. Be concise, objective, and factual.
4. Do not invent facts, fares, or sources.

Retrieved Web Sources:
{context}
"""


@dataclass
class HybridRAGResponse:
    """
    Unified response object for both PDF-based and web-based answers.

    Attributes:
        question (str): The original user question.
        answer (str): The factual generated answer.
        source_type (str): Either "PDF" or "External Web".
        cited_pages (List[int]): Page numbers cited from the PDF.
        web_sources (List[Dict[str, str]]): List of web sources with 'title' and 'url'.
    """
    question: str
    answer: str
    source_type: str  # "PDF" or "External Web"
    cited_pages: List[int] = field(default_factory=list)
    web_sources: List[Dict[str, str]] = field(default_factory=list)


def format_pdf_context_with_citations(docs: List[Document]) -> str:
    """Format retrieved PDF chunks with clear page markers."""
    blocks = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content.strip()
        blocks.append(f"--- [Historical Excerpt from Page {page}] ---\n{content}")
    return "\n\n".join(blocks)


class HistoricalRAGPipeline:
    """
    Hybrid RAG Pipeline: Intelligently routes queries between local FAISS
    historical PDF vector storage and external web search.
    """

    def __init__(
        self,
        vector_store_path: Union[str, Path] = "vector_store",
        k: int = 4,
        model_name: str = "qwen/qwen3.8-27b",
        temperature: float = 0.2,
    ):
        self.vector_store_path = Path(vector_store_path)
        self.k = k
        self.vector_store = load_vector_store(self.vector_store_path)
        self.llm = get_groq_llm(model_name=model_name, temperature=temperature)

        self.pdf_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PDF_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self.web_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", WEB_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        self.pdf_chain = self.pdf_prompt | self.llm | StrOutputParser()
        self.web_chain = self.web_prompt | self.llm | StrOutputParser()

    def route_query(self, question: str) -> Tuple[str, List[Tuple[Document, float]]]:
        """
        Determine whether a query should be routed to PDF or External Web.

        Uses FAISS semantic distance:
        - If top distance <= 0.95: High semantic similarity with 19th-century corpus -> Route to PDF.
        - If top distance > 0.95: Query is external/contemporary -> Route to External Web.

        Returns:
            Tuple[str, List[Tuple[Document, float]]]: ('PDF' or 'EXTERNAL', retrieved_matches)
        """
        matches = similarity_search_with_score(
            query=question,
            vector_store=self.vector_store,
            k=self.k,
        )

        if not matches:
            return "EXTERNAL", []

        top_score = matches[0][1]

        # Semantic distance threshold: below 0.95 indicates content related to historical PDF corpus
        if top_score <= 0.95:
            return "PDF", matches
        else:
            return "EXTERNAL", matches

    def query_pdf(self, question: str, matches: Optional[List[Tuple[Document, float]]] = None) -> HybridRAGResponse:
        """Execute PDF-based RAG query."""
        if matches is None:
            matches = similarity_search_with_score(question, self.vector_store, k=self.k)

        docs = [doc for doc, _ in matches]
        context = format_pdf_context_with_citations(docs)
        answer = self.pdf_chain.invoke({"context": context, "question": question}).strip()

        cited_pages = sorted(list({doc.metadata["page"] for doc in docs if "page" in doc.metadata}))

        return HybridRAGResponse(
            question=question,
            answer=answer,
            source_type="PDF",
            cited_pages=cited_pages,
            web_sources=[],
        )

    def query_web(self, question: str, prefix_note: Optional[str] = None) -> HybridRAGResponse:
        """Execute external web search RAG query."""
        web_results = search_web(question, max_results=4)
        context = format_web_results_for_context(web_results)
        answer = self.web_chain.invoke({"context": context, "question": question}).strip()

        if prefix_note:
            answer = f"{prefix_note}\n\n{answer}"

        sources = [{"title": r.title, "url": r.url} for r in web_results]

        return HybridRAGResponse(
            question=question,
            answer=answer,
            source_type="External Web",
            cited_pages=[],
            web_sources=sources,
        )

    def query(self, question: str) -> HybridRAGResponse:
        """
        Hybrid query handler:
        1. Checks semantic relevance against PDF vector store.
        2. Routes to PDF if relevant.
        3. If PDF does not contain sufficient answer, falls back to web search with clear notice.
        4. If semantically external, routes directly to web search.
        """
        route, matches = self.route_query(question)

        if route == "PDF":
            pdf_response = self.query_pdf(question, matches)
            # Check if PDF response reports lack of evidence
            ans_lower = pdf_response.answer.lower()
            if "cannot find" in ans_lower or "insufficient" in ans_lower:
                note = "*(Note: The 19th-century archival PDF does not contain this information. Retrieved via External Web Search)*"
                return self.query_web(question, prefix_note=note)
            return pdf_response
        else:
            return self.query_web(question)


# Alias for backwards compatibility
RAGResponse = HybridRAGResponse


def answer_historical_question(
    question: str,
    vector_store_path: Union[str, Path] = "vector_store",
    k: int = 4,
) -> HybridRAGResponse:
    """Convenience function to run a hybrid RAG query."""
    pipeline = HistoricalRAGPipeline(vector_store_path=vector_store_path, k=k)
    return pipeline.query(question)
