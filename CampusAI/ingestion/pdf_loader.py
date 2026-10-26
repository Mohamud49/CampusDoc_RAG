"""
ingestion/pdf_loader.py — Load PDF handbooks into LangChain Documents
---------------------------------------------------------------------
Each page becomes text with metadata: source filename and page number.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_files(file_paths: List[Path]) -> List[Document]:
    """
    Load one or more PDF files.

    Parameters
    ----------
    file_paths : list of Path
        Paths to PDF files on disk.

    Returns
    -------
    list[Document]
        All pages from all PDFs, each with metadata.

    Somali: Akhri PDF-yada oo soo celi qoraalka bog kasta.
    """
    all_documents: List[Document] = []

    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            continue

        loader = PyPDFLoader(str(path))
        pages = loader.load()

        # Ensure every chunk knows which file it came from
        for doc in pages:
            doc.metadata.setdefault("source", path.name)
            doc.metadata["filename"] = path.name
            # PyPDFLoader usually sets "page" (0-based); expose human-friendly page
            page_idx = doc.metadata.get("page", 0)
            doc.metadata["page_number"] = int(page_idx) + 1

        all_documents.extend(pages)

    return all_documents


def save_uploaded_pdfs(uploaded_files, destination_dir: Path) -> List[Path]:
    """
    Write Streamlit UploadedFile objects to disk.

    Parameters
    ----------
    uploaded_files : streamlit UploadedFile list
    destination_dir : Path
        Folder where PDFs are stored (e.g. data/uploads).

    Returns
    -------
    list[Path]
        Saved file paths ready for load_pdf_files().
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for uploaded in uploaded_files or []:
        if not uploaded.name.lower().endswith(".pdf"):
            continue
        target = destination_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved.append(target)

    return saved
