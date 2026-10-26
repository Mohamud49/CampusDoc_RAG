"""
vectorstore/chroma_store.py — Persistent numpy vector index
-----------------------------------------------------------
ChromaDB's hnswlib DLL is blocked by Windows Application Control on this machine,
so CampusAI stores embeddings with NumPy (already used by sentence-transformers).
Public functions stay the same so the rest of the app does not change.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

import config
from vectorstore.embeddings import get_embedding_model

INDEX_DIR = config.DATA_DIR / "vector_index"
DOCS_FILE = INDEX_DIR / "documents.json"
EMB_FILE = INDEX_DIR / "embeddings.npy"


class NumpyVectorStore(VectorStore):
    """Cosine-similarity store persisted as JSON + .npy (no native HNSW DLL)."""

    def __init__(self, persist_dir: Path, embedding: Embeddings) -> None:
        self.persist_dir = Path(persist_dir)
        self._embedding = embedding
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._load()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _load(self) -> None:
        if not DOCS_FILE.exists() or not EMB_FILE.exists():
            return
        try:
            payload = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
            docs = payload.get("documents", [])
            matrix = np.load(EMB_FILE)
        except (OSError, json.JSONDecodeError, ValueError):
            return
        if not docs or matrix.ndim != 2 or matrix.shape[0] != len(docs):
            return
        self._ids = [str(d.get("id", uuid.uuid4())) for d in docs]
        self._texts = [d.get("text", "") for d in docs]
        self._metadatas = [d.get("metadata") or {} for d in docs]
        self._matrix = matrix.astype(np.float32, copy=False)

    def _save(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [
                {"id": i, "text": t, "metadata": m}
                for i, t, m in zip(self._ids, self._texts, self._metadatas)
            ]
        }
        DOCS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if self._matrix is not None and len(self._ids):
            np.save(EMB_FILE, self._matrix)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        text_list = list(texts)
        if not text_list:
            return []
        meta_list = metadatas or [{} for _ in text_list]
        id_list = ids or [str(uuid.uuid4()) for _ in text_list]
        vectors = np.asarray(
            self._embedding.embed_documents(text_list), dtype=np.float32
        )
        if self._matrix is None or self._matrix.size == 0:
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        self._ids.extend(id_list)
        self._texts.extend(text_list)
        self._metadatas.extend(meta_list)
        self._save()
        return id_list

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        if self._matrix is None or not self._texts:
            return []
        query_vec = np.asarray(self._embedding.embed_query(query), dtype=np.float32)
        # Embeddings are L2-normalized in embeddings.py → cosine = dot product
        scores = self._matrix @ query_vec
        k = max(1, min(int(k), len(self._texts)))
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Document(page_content=self._texts[i], metadata=self._metadatas[i])
            for i in top
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        persist_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> "NumpyVectorStore":
        store = cls(persist_dir or INDEX_DIR, embedding)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


def get_vector_store() -> NumpyVectorStore:
    """Open or create the on-disk numpy index."""
    config.ensure_directories()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return NumpyVectorStore(INDEX_DIR, get_embedding_model())


def index_documents(chunks: List[Document], reset: bool = False) -> int:
    """Add document chunks to the vector store."""
    if not chunks:
        return 0
    if reset:
        clear_vector_store()
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def clear_vector_store() -> None:
    """Remove all indexed vectors (use before full re-ingest)."""
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def get_retriever(search_kwargs: Optional[dict] = None):
    """Build a retriever for the RAG chain (top-k similar chunks)."""
    kwargs = search_kwargs or {"k": config.TOP_K}
    store = get_vector_store()
    return store.as_retriever(search_type="similarity", search_kwargs=kwargs)


def collection_has_documents() -> bool:
    """Return True if at least one vector exists on disk."""
    if not DOCS_FILE.exists():
        return False
    try:
        payload = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
        return bool(payload.get("documents"))
    except (OSError, json.JSONDecodeError):
        return False
