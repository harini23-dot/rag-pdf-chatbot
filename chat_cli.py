"""Phases 5-6: command-line chatbot. Run `python ingest.py` first."""
from rag import answer_question, format_source, get_llm, load_vectorstore

def main():
    store, llm = load_vectorstore(), get_llm()
    history = []
    print("PDF chatbot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        answer, chunks = answer_question(store, llm, q, history)
        print(f"\nBot: {answer}")
        print("Sources:", "; ".join(sorted({format_source(c) for c in chunks})), "\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]

if __name__ == "__main__":
    main()
