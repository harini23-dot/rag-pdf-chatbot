"""Phases 2-4: read PDFs from data/, chunk them, embed them, save to chroma_db/."""
from rag import build_vectorstore

if __name__ == "__main__":
    print("Reading PDFs from data/ and building the vector database...")
    n = build_vectorstore()
    print(f"Done. Stored {n} chunks in chroma_db/")
