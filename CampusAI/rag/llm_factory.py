"""
rag/llm_factory.py — Create LLM from Groq, OpenAI, or Ollama
------------------------------------------------------------
Set LLM_PROVIDER in .env to switch providers without code changes.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

import config


def get_chat_llm() -> BaseChatModel:
    """
    Factory: returns the configured chat model.

    Raises
    ------
    ValueError
        If provider is unknown or API key is missing (Groq/OpenAI).

    Somali: Dooro LLM-ka aad isticmaasho (.env) — Groq, OpenAI, ama Ollama.
    """
    provider = config.LLM_PROVIDER

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is empty. Add it to .env or set LLM_PROVIDER=ollama."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1,
        )

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is empty. Add it to .env.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=0.1,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use groq, openai, or ollama."
    )
