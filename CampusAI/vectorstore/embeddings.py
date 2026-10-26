"""
vectorstore/embeddings.py — HuggingFace embedding model (free, local)
---------------------------------------------------------------------
Embeddings turn text into vectors for similarity search.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformers model once (cached in memory).

    First run downloads the model from HuggingFace — may take a few minutes.

    Somali: Model-kan wuxuu qoraalka u beddelaa tirooyin (vectors) si loo raadiyo.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
