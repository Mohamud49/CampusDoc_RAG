"""
ingestion/chunking.py — Split long PDF text into searchable chunks
--------------------------------------------------------------------
Recursive splitting keeps paragraphs and sentences intact when possible.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks with overlap.

    Metadata (filename, page) is copied to each chunk automatically
    by RecursiveCharacterTextSplitter.

    Somali: U kala jar qoraalka dheer qaybo yaryar si search-ku u fududaado.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(documents)
