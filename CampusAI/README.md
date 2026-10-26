# CampusAI

RAG-based university academic rules and syllabus querying system (MCA minor project).

**Stack:** Python 3.10+, Streamlit, LangChain, ChromaDB, HuggingFace embeddings, Groq / OpenAI / Ollama.

---

## Step 1 — Requirements

See `requirements.txt` for pinned dependencies.

## Step 2 — Virtual environment & install

**Windows (PowerShell):**

```powershell
cd C:\Users\moham\CampusAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
# Edit .env — add GROQ_API_KEY or set LLM_PROVIDER=ollama
```

**macOS / Linux:**

```bash
cd CampusAI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### LLM options

| Provider | `.env` |
|----------|--------|
| **Groq** (recommended, free tier) | `LLM_PROVIDER=groq` + `GROQ_API_KEY` |
| **Ollama** (local) | `LLM_PROVIDER=ollama`, run `ollama pull llama3.2` |
| **OpenAI** | `LLM_PROVIDER=openai` + `OPENAI_API_KEY` |

First run downloads the embedding model (`all-MiniLM-L6-v2`) — requires internet once.

## Step 3 — Run locally

```powershell
streamlit run app.py
```

Browser opens at `http://localhost:8501`.

Log in first:

| Role | Username | Password | Access |
|------|----------|----------|--------|
| **admin** | `admin` | `admin123` | Upload/index PDFs and use the chatbot |
| **user** | `student` | `student123` | Chatbot only (no PDF upload) |

1. As **admin**, upload PDF handbooks in the **sidebar** and click **Index documents**.
2. As **student**, ask questions in the chat; expand **Retrieved sources** for citations.

## Project layout

```
CampusAI/
├── app.py                 # Streamlit UI (main entry)
├── config.py              # Settings from .env
├── requirements.txt
├── ingestion/
│   ├── pdf_loader.py      # PDF → Documents
│   └── chunking.py        # Text splitting
├── vectorstore/
│   ├── embeddings.py      # HuggingFace embeddings
│   └── chroma_store.py    # Chroma persistence
└── rag/
    ├── prompts.py         # Zero-hallucination system prompt
    ├── llm_factory.py     # Groq / OpenAI / Ollama
    └── chain.py           # Conversational RAG chain
```

## Push to GitHub

```powershell
cd C:\Users\moham\CampusAI
git init
git add .
git commit -m "Initial CampusAI RAG application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CampusAI.git
git push -u origin main
```

Do **not** commit `.env` or `data/` (already in `.gitignore`).

---

## License

Academic / educational use for MCA minor project.
