"""How often does retrieval find the right page? No LLM involved, so it's free to run.

    python scripts/eval_retrieval.py

This is what I used to pick TOP_K and MIN_SIMILARITY in app/config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.retriever import Retriever  # noqa: E402

# Question -> page (in a PDF viewer) that answers it.
LABELLED = [
    ("How much does the bootcamp cost?", 16),
    ("How is my final grade calculated?", 14),
    ("When are the live classes?", 11),
    ("What laptop do I need?", 5),
    ("What happens if my Capitec loan is declined?", 17),
    ("Who will be my tutor and can I change them?", 20),
    ("What are the tutor support hours?", 20),
    ("What mark do I need to get placed at a company?", 24),
    ("How much do the paid internships pay?", 24),
    ("What do the colours on my calendar mean?", 9),
    ("How do I check my XP rank on Discord?", 22),
    ("Where do I submit my assignments?", 13),
    ("When is orientation day?", 7),
    ("Who do I email about laptop advice?", 5),
    ("What name should I use on Discord?", 19),
    ("How do I join the Tuesday Zoom class?", 12),
    ("What are the payment options?", 16),
    ("Will the live classes be recorded?", 11),
    ("What will I learn in weeks 11 to 15?", 6),
    ("When is the deadline to pay fees?", 16),
    ("What should I do if Capitec hasn't contacted me?", 17),
    ("How do I get access to my calendar?", 10),
    ("Where are recorded lectures posted?", 13),
    ("What is Zaio's office address?", 26),
    ("How much do junior developers earn?", 24),
    ("How will you communicate with us before orientation?", 18),
    ("Which Discord levels get a certificate?", 22),
    ("Who is the lead instructor?", 15),
]

# Not in the handbook. Their best score shows why the threshold alone can't
# decide what's answerable.
UNANSWERABLE = [
    "What is the attendance requirement?",
    "What is the refund policy?",
    "Can I get an extension on an assignment?",
    "Is there a dress code for live classes?",
    "What is the pass mark for the MCQs?",
    "What is the capital of France?",
    "How do I cook pasta?",
]


def main():
    retriever = Retriever()
    top1 = topk = 0
    lowest = 1.0

    print(f"Top {config.TOP_K} pages per question (similarity of best hit)\n")
    for question, page in LABELLED:
        hits = retriever.search(question)
        pages = [h.page for h in hits]
        top1 += pages[0] == page
        topk += page in pages
        lowest = min(lowest, hits[0].similarity)
        mark = "ok  " if pages[0] == page else ("top5" if page in pages else "MISS")
        print(f"  {mark} {hits[0].similarity:.2f}  want p{page:<3} got {pages}  {question}")

    print("\nUnanswerable")
    for question in UNANSWERABLE:
        best = retriever.search(question)[0]
        gate = "blocked" if best.similarity < config.MIN_SIMILARITY else "-> LLM decides"
        print(f"  {best.similarity:.2f}  {gate:<15} {question}")

    n = len(LABELLED)
    print(f"\nRight page ranked first: {top1}/{n}")
    print(f"Right page in top {config.TOP_K}:     {topk}/{n}")
    print(f"Lowest best-hit similarity on an answerable question: {lowest:.2f} "
          f"(threshold is {config.MIN_SIMILARITY})")


if __name__ == "__main__":
    main()
