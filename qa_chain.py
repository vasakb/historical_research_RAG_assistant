"""
RAG Question Answering Pipeline
===============================
Combines FAISS semantic retrieval with the Groq LLM to generate
fact-grounded answers with page-number citations from historical documents.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import get_faiss_retriever, retrieve_relevant_chunks
from llm import get_groq_llm


RAG_SYSTEM_PROMPT = """You are a meticulous historical research assistant analyzing official archives on 19th-century Indian railways and telegraphs.

Instructions:
1. Answer the user's question using ONLY the retrieved historical excerpts provided below.
2. For every factual claim, explicitly cite the original page number using the format [Page X] (e.g., [Page 13]).
3. If the provided context does not contain sufficient information to answer the question, clearly state:
   "I cannot find the answer to this question in the provided historical document."
4. Do not speculate, extrapolate, or introduce outside historical knowledge not stated in the context.

Retrieved Historical Context:
{context}
"""


@dataclass
class RAGResponse:
    """
    Container for the completed RAG query answer and citation metadata.
    """
    question: str
    answer: str
    source_documents: List[Document]
    cited_pages: List[int]


def format_context_with_citations(docs: List[Document]) -> str:
    """
    Format retrieved LangChain documents into a labeled context string with page headers.

    Args:
        docs (List[Document]): List of retrieved document chunks.

    Returns:
        str: Formatted context blocks.
    """
    blocks = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content.strip()
        blocks.append(f"--- [Historical Excerpt from Page {page}] ---\n{content}")
    return "\n\n".join(blocks)


class HistoricalRAGPipeline:
    """
    End-to-end RAG pipeline managing FAISS retrieval and Groq LLM answer generation.
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
        self.retriever = get_faiss_retriever(self.vector_store_path, k=self.k)
        self.llm = get_groq_llm(model_name=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> RAGResponse:
        """
        Execute full RAG query:
        1. Retrieve top-k chunks from FAISS.
        2. Format context with page markers.
        3. Prompt Groq LLM.
        4. Return answer with page citations.

        Args:
            question (str): The research question.

        Returns:
            RAGResponse: Object containing question, answer text, and source docs.
        """
        # 1. Retrieve top-k relevant chunks
        docs = self.retriever.invoke(question)

        # 2. Format context
        context = format_context_with_citations(docs)

        # 3. Generate answer
        answer = self.chain.invoke({"context": context, "question": question})

        # 4. Extract cited page numbers from retrieved docs
        pages = sorted(list({doc.metadata["page"] for doc in docs if "page" in doc.metadata}))

        return RAGResponse(
            question=question,
            answer=answer.strip(),
            source_documents=docs,
            cited_pages=pages,
        )


def answer_historical_question(
    question: str,
    vector_store_path: Union[str, Path] = "vector_store",
    k: int = 4,
) -> RAGResponse:
    """
    Convenience function to run a RAG query on the historical corpus.

    Args:
        question (str): The research question.
        vector_store_path (str | Path): Path to the FAISS index folder.
        k (int): Number of chunks to retrieve (default 4).

    Returns:
        RAGResponse: Complete answer and citation details.
    """
    pipeline = HistoricalRAGPipeline(vector_store_path=vector_store_path, k=k)
    return pipeline.query(question)


if __name__ == "__main__":
    q = "What gauge was standardized for Indian railways?"
    print(f"Question: {q}\n")
    res = answer_historical_question(q)
    print(f"Answer:\n{res.answer}\n")
    print(f"Source Pages: {res.cited_pages}")
