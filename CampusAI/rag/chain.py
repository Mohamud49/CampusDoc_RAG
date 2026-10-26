"""
rag/chain.py — Conversational RAG pipeline (retrieve + generate)
----------------------------------------------------------------
Uses history-aware retriever + strict context-only answering.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from rag.llm_factory import get_chat_llm
from rag.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from vectorstore.chroma_store import get_retriever

# Each retrieved chunk is labeled so the LLM can cite filename + page
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "Document: {filename}\nPage: {page_number}\n\n{page_content}"
)


def build_conversational_rag_chain():
    """
    Build LangChain conversational retrieval chain (production pattern).

    Somali: Isku xir raadinta Chroma iyo jawaabta LLM — taariikhda wada sheekaysiga waa la isticmaalaa.
    English: Follow-up questions like "What about exams?" still work thanks to history-aware retrieval.
    """
    llm = get_chat_llm()
    retriever = get_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONDENSE_QUESTION_PROMPT
    )

    document_chain = create_stuff_documents_chain(
        llm,
        QA_PROMPT,
        document_prompt=DOCUMENT_PROMPT,
    )

    return create_retrieval_chain(history_aware_retriever, document_chain)


def ask_question(
    chain,
    question: str,
    chat_history: List,
) -> Tuple[str, List[Document]]:
    """
    Run one chat turn.

    Parameters
    ----------
    chain : retrieval chain from build_conversational_rag_chain()
    question : str
        User message.
    chat_history : list
        LangChain messages (HumanMessage / AIMessage) from prior turns.

    Returns
    -------
    (answer, source_documents)

    Somali: Soo celi jawaabta iyo qaybaha PDF-ka ee laga helay.
    """
    result = chain.invoke({"input": question, "chat_history": chat_history})
    answer = result.get("answer", "")
    sources: List[Document] = result.get("context", [])
    return answer, sources


def format_source_label(doc: Document) -> str:
    """Short label for Streamlit expander title."""
    meta = doc.metadata or {}
    filename = meta.get("filename") or meta.get("source", "Document")
    page = meta.get("page_number") or meta.get("page", "?")
    return f"{filename} — Page {page}"
