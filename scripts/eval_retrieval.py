"""How often does retrieval put the right page in front of the LLM? No LLM involved, so it's free.

    python scripts/eval_retrieval.py            # the settings in app/config.py
    python scripts/eval_retrieval.py --compare  # pooled vs per-source search at a few sizes

This is what I used to pick TOP_K_PER_SOURCE, the reranker and MIN_SIMILARITY
in app/config.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.retriever import HANDBOOK, WEBSITE, Retriever  # noqa: E402

# Question -> handbook page (as a PDF viewer numbers it) that answers it.
HANDBOOK_QUESTIONS = [
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

# Question -> website paths that answer it. A trailing * matches any path
# starting with that prefix (the "15% off upfront" line is on every course page).
WEBSITE_QUESTIONS = [
    # Bootcamps and QCTO qualifications are both "courses", so either kind of page answers it.
    ("What courses does ZAIO offer?", ["/", "/bootcamps", "/compare-courses", "/qualifications*"]),
    ("How much does the Fullstack Web Development bootcamp cost?",
     ["/compare-courses", "/tuition/fullstack-web-development", "/fullstack-bootcamp"]),
    ("What is Zaio's refund policy?", ["/terms"]),
    ("How long is the Data Science bootcamp?", ["/compare-courses", "/datascience-bootcamp"]),
    ("Can I finance 100% of my tuition?", ["/tuition-financing", "/flexible-tuition", "/tuition/*", "/manati"]),
    ("How quickly does Manati respond to a loan application?",
     ["/tuition-financing", "/flexible-tuition", "/tuition/*", "/application-process"]),
    ("What happens if I miss a tuition payment?", ["/tuition-financing", "/flexible-tuition", "/application-process"]),
    ("How long does it take from applying to being enrolled?", ["/application-process"]),
    ("When does the next Cloud & DevOps bootcamp start?", ["/cloud-devops-engineer-bootcamp"]),
    ("Where was the Zaio open day in Cape Town?", ["/events/zaio-open-day-capetown"]),
    ("Which certification do I get from the Cybersecurity bootcamp?", ["/compare-courses", "/cybersecurity-bootcamp"]),
    ("Can I switch to a different bootcamp after enrolling?", ["/compare-courses"]),
    ("How old do I need to be to enrol?", ["/terms"]),
    ("Who is the CEO of Zaio?", ["/aboutus"]),
    ("What is a QCTO occupational certificate?", ["/qualifications*"]),
    ("How does Capitec education finance work?", ["/capitec-finance"]),
    ("What will I learn in the Cloud & DevOps Engineering bootcamp?", ["/cloud-devops-engineer-bootcamp", "/compare-courses"]),
    ("Does Zaio offer training for companies?", ["/company", "/skillsdevelopment"]),
    ("Is there a discount for paying the full tuition upfront?", ["/tuition-financing", "/flexible-tuition", "/*-bootcamp"]),
    ("What is The Weekly Commit?", ["/newsletter"]),
    ("How much is the Software Development occupational certificate?",
     ["/qualifications/occupational-certificate-software-development"]),
    ("What do I need to be admitted to the Software Development occupational certificate?",
     ["/qualifications/occupational-certificate-software-development"]),
]

# In neither source. Their best score shows why the threshold alone can't
# decide what's answerable.
UNANSWERABLE = [
    "What is the attendance requirement?",
    "Can I get an extension on an assignment?",
    "Is there a dress code for live classes?",
    "What is the pass mark for the MCQs?",
    "What is the Wi-Fi password at the Cape Town hub?",
    "Can I bring my dog to the live classes?",
    "How many parking bays are at the Durban hub?",
    "Does Zaio offer medical aid for students?",
    "Who won the 2026 Zaio hackathon?",
    "What is the student-to-tutor ratio?",
    "Can I defer my studies to next year?",
    "Is there a student discount on laptops?",
    "What is the capital of France?",
    "How do I cook pasta?",
]


def path_of(url: str) -> str:
    return url.removeprefix(config.WEBSITE_URL) or "/"


def url_matches(url: str, accepted: list[str]) -> bool:
    path = path_of(url)
    for pattern in accepted:
        if pattern.startswith("/*"):
            if path.endswith(pattern[2:]):
                return True
        elif pattern.endswith("*"):
            if path.startswith(pattern[:-1]):
                return True
        elif path == pattern:
            return True
    return False


def first_right_rank(hits, question_index: int) -> int:
    """1-based rank of the first chunk from the right page (99 if none)."""
    if question_index < len(HANDBOOK_QUESTIONS):
        page = HANDBOOK_QUESTIONS[question_index][1]
        ok = [h.source == HANDBOOK and h.page == page for h in hits]
    else:
        accepted = WEBSITE_QUESTIONS[question_index - len(HANDBOOK_QUESTIONS)][1]
        ok = [h.source == WEBSITE and url_matches(h.url, accepted) for h in hits]
    return ok.index(True) + 1 if True in ok else 99


def score(retriever: Retriever, verbose: bool = False) -> dict:
    """Recall of the right page among the chunks the LLM would be given."""
    handbook_hits = website_hits = context = local_skips = 0
    lowest = 1.0
    for question, page in HANDBOOK_QUESTIONS:
        hits = retriever.search(question)
        context += len(hits)
        lowest = min(lowest, best_similarity(hits))
        local_skips += local_model_skips(hits)
        found = page in [h.page for h in hits if h.source == HANDBOOK]
        handbook_hits += found
        if verbose:
            got = [h.page for h in hits if h.source == HANDBOOK]
            print(f"  {'ok  ' if found else 'MISS'} {best_similarity(hits):.2f}  want p{page:<3} handbook pages {got}  {question}")
    for question, accepted in WEBSITE_QUESTIONS:
        hits = retriever.search(question)
        context += len(hits)
        lowest = min(lowest, best_similarity(hits))
        local_skips += local_model_skips(hits)
        urls = [h.url for h in hits if h.source == WEBSITE]
        found = any(url_matches(u, accepted) for u in urls)
        website_hits += found
        if verbose:
            got = [path_of(u) for u in dict.fromkeys(urls)]
            print(f"  {'ok  ' if found else 'MISS'} {best_similarity(hits):.2f}  want {accepted[0]:<32} got {got[:3]}  {question}")
    n = len(HANDBOOK_QUESTIONS) + len(WEBSITE_QUESTIONS)
    return {"handbook": handbook_hits, "website": website_hits, "context": context / n, "lowest": lowest,
            "local_skips": local_skips}


def best_similarity(hits) -> float:
    # max, not hits[0]: after reranking the first hit isn't the most similar one
    return max(h.similarity for h in hits)


def local_model_skips(hits) -> bool:
    """Would the local model's reranker check stop this question? (see app/local_llm.py)"""
    return hits[0].rerank_score is not None and hits[0].rerank_score < config.LOCAL_MIN_RERANK_SCORE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="compare search strategies instead")
    args = parser.parse_args()
    nh, nw = len(HANDBOOK_QUESTIONS), len(WEBSITE_QUESTIONS)

    if args.compare:
        print(f"{'strategy':<24} {'handbook':>9} {'website':>9} {'excerpts sent':>14}")
        for label, kwargs in [
            ("pooled top 5", {"top_k": 5, "per_source": None}),
            ("pooled top 8", {"top_k": 8, "per_source": None}),
            ("pooled top 10", {"top_k": 10, "per_source": None}),
            ("per source, 3 each", {"per_source": 3}),
            ("per source, 4 each", {"per_source": 4}),
            ("per source, 5 each", {"per_source": 5}),
        ]:
            s = score(Retriever(rerank=False, **kwargs))
            print(f"{label:<24} {s['handbook']:>5}/{nh} {s['website']:>6}/{nw} {s['context']:>14.1f}")

        questions = [q for q, _ in HANDBOOK_QUESTIONS + WEBSITE_QUESTIONS]
        print(f"\nWhere the right page ranks among the {2 * config.TOP_K_PER_SOURCE} excerpts")
        print(f"{'order':<24} {'1st':>7} {'top 2':>7} {'top 3':>7}")
        for label, retriever in [("similarity", Retriever(rerank=False)), ("reranked", Retriever())]:
            ranks = [first_right_rank(retriever.search(q), i) for i, q in enumerate(questions)]
            n = len(ranks)
            print(f"{label:<24} " + " ".join(f"{sum(r <= k for r in ranks):>3}/{n}" for k in (1, 2, 3)))
        return

    retriever = Retriever()
    print(f"Handbook and website questions ({config.TOP_K_PER_SOURCE} chunks per source)\n")
    s = score(retriever, verbose=True)

    print("\nUnanswerable (best similarity, best rerank score)")
    past_gate = local_stopped = 0
    for question in UNANSWERABLE:
        hits = retriever.search(question)
        similarity = best_similarity(hits)
        if similarity < config.MIN_SIMILARITY:
            gate = "blocked (similarity)"
        else:
            past_gate += 1
            local_stopped += local_model_skips(hits)
            gate = "local model: skipped" if local_model_skips(hits) else "-> LLM decides"
        print(f"  {similarity:.2f}  {hits[0].rerank_score:6.2f}  {gate:<21} {question}")

    print(f"\nRight handbook page sent to the LLM: {s['handbook']}/{nh}")
    print(f"Right website page sent to the LLM:  {s['website']}/{nw}")
    print(f"Lowest best-hit similarity on an answerable question: {s['lowest']:.2f} "
          f"(threshold is {config.MIN_SIMILARITY})")
    print(f"Local model's reranker check stops {local_stopped} of the {past_gate} unanswerable questions that get "
          f"past the similarity gate, and {s['local_skips']} of the {nh + nw} answerable ones")


if __name__ == "__main__":
    main()
