"""Phase 8: score the bot on your own test questions.

Edit eval/questions.json with ~10 questions from YOUR PDFs. A question passes if
the answer contains every keyword in `expected_keywords` (case-insensitive).
Try different CHUNK_SIZE / TOP_K values in .env and compare scores.
"""
import json
from pathlib import Path

from rag import TOP_K, CHUNK_SIZE, answer_question, get_llm, load_vectorstore

def main():
    cases = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    store, llm = load_vectorstore(), get_llm()
    passed = 0
    for i, c in enumerate(cases, 1):
        answer, _ = answer_question(store, llm, c["question"])
        ok = all(k.lower() in answer.lower() for k in c["expected_keywords"])
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] Q{i}: {c['question']}")
        if not ok:
            print(f"        expected: {c['expected_keywords']}\n        got: {answer[:200]}")
    print(f"\nScore: {passed}/{len(cases)} ({100 * passed / len(cases):.0f}%)  "
          f"[chunk_size={CHUNK_SIZE}, top_k={TOP_K}]")

if __name__ == "__main__":
    main()
