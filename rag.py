"""Core RAG logic: load PDFs -> chunk -> embed -> store -> retrieve -> answer."""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
COLLECTION = "pdf_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 4))

ANSWER_PROMPT = """You are a helpful assistant that answers questions using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

CONDENSE_PROMPT = """Given the chat history and a follow-up question, rewrite the follow-up
as a standalone question. Return only the rewritten question.

Chat history:
{history}

Follow-up question: {question}

Standalone question:"""


# ---------- Phase 2: load ----------
def load_pdfs(folder: Path = DATA_DIR):
    docs = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs


# ---------- Phase 3: chunk ----------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# ---------- Phase 4: embed + store ----------
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def build_vectorstore(folder: Path = DATA_DIR):
    """(Re)build the vector DB from all PDFs in `folder`. Returns chunk count."""
    docs = load_pdfs(folder)
    if not docs:
        raise ValueError(f"No readable PDFs found in '{folder}'. Add PDFs first.")
    chunks = split_documents(docs)
    if not any(c.page_content.strip() for c in chunks):
        raise ValueError("No text could be extracted. Scanned PDFs need OCR.")

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )
    try:
        store.delete_collection()
    except Exception:
        pass
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )
    store.add_documents(chunks)
    return len(chunks)


def load_vectorstore():
    if not Path(DB_DIR).exists():
        raise FileNotFoundError("Vector DB not found. Run `python ingest.py` first.")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )


def reset_db():
    shutil.rmtree(DB_DIR, ignore_errors=True)


# ---------- Phase 5: retrieve ----------
def retrieve(store, question: str, k: int = TOP_K):
    return store.similarity_search(question, k=k)


# ---------- Phase 6: generate ----------
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-haiku-5-5", temperature=0, max_tokens=1000)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.2", temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use anthropic, openai or ollama.")


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _format_history(history, max_turns: int = 4) -> str:
    recent = history[-max_turns * 2:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def answer_question(store, llm, question: str, history=None):
    """Return (answer, source_chunks). `history` is a list of {role, content}."""
    history = history or []

    # Memory: turn follow-ups like "what about its cost?" into standalone questions
    search_query = question
    if history:
        search_query = _text(
            llm.invoke(CONDENSE_PROMPT.format(history=_format_history(history), question=question))
        ).strip() or question

    chunks = retrieve(store, search_query)
    context = "\n\n---\n\n".join(c.page_content for c in chunks)
    reply = llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
    return _text(reply).strip(), chunks


def format_source(chunk) -> str:
    name = Path(chunk.metadata.get("source", "unknown")).name
    page = chunk.metadata.get("page")
    return f"{name}, page {page + 1}" if isinstance(page, int) else name
