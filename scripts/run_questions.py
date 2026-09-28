"""Part 4: put real questions through POST /ask and write TEST_RESULTS.md.

    python scripts/run_questions.py                              # in-process, no server needed
    python scripts/run_questions.py --url http://localhost:8000  # against a running server

Uses whichever LLM the API would use (Claude with ANTHROPIC_API_KEY set,
otherwise the local model), and needs a built index.
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.llm import describe_llm  # noqa: E402

HANDBOOK, WEBSITE, EITHER, NEITHER = "Handbook", "Website", "Both", "Neither"

# (kind, question, what an acceptable source looks like)
#   Handbook: page numbers, Website: paths on zaio.io, Both: any source, Neither: no source
QUESTIONS = [
    (HANDBOOK, "What are the minimum laptop requirements?", [5]),
    (HANDBOOK, "How is my final grade calculated?", [14]),
    (HANDBOOK, "What happens if my Capitec loan application fails?", [17]),
    (HANDBOOK, "I want a different tutor. Who do I speak to?", [20]),
    (HANDBOOK, "What does a yellow tile on my calendar mean?", [9]),
    (WEBSITE, "What courses does ZAIO offer?", ["/", "/bootcamps", "/compare-courses", "/qualifications*"]),
    # In the handbook-only version this was a "not found" - the answer is on the website.
    (WEBSITE, "What is the refund policy?", ["/terms"]),
    (WEBSITE, "How long is the Data Science bootcamp?", ["/compare-courses", "/datascience-bootcamp"]),
    (WEBSITE, "When does the next Cloud & DevOps bootcamp start?", ["/cloud-devops-engineer-bootcamp"]),
    (WEBSITE, "Who is the CEO of Zaio?", ["/aboutus"]),
    (WEBSITE, "How quickly does Manati respond to a loan application?",
     ["/tuition-financing", "/flexible-tuition", "/tuition/*", "/application-process"]),
    # The handbook (one 2025 intake) says R 38,950; the website now says R42,850 upfront.
    (EITHER, "How much does the Fullstack AI Engineer bootcamp cost?", None),
    (NEITHER, "What is the attendance requirement?", None),
    (NEITHER, "Is there a dress code for live classes?", None),
    (NEITHER, "What is the pass mark for the MCQs?", None),  # MCQs are on page 14, the pass mark isn't
    (NEITHER, "What is the capital of France?", None),
]


def make_poster(url):
    if url:
        import httpx

        http = httpx.Client(base_url=url, timeout=600)
        return lambda q: http.post("/ask", json={"question": q})

    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app, raise_server_exceptions=False)
    return lambda q: client.post("/ask", json={"question": q})


def path_matches(url: str, accepted: list[str]) -> bool:
    path = url.removeprefix(config.WEBSITE_URL) or "/"
    return any(path.startswith(a[:-1]) if a.endswith("*") else path == a for a in accepted)


def check(kind, expected, source) -> bool:
    if kind == NEITHER:
        return source is None
    if source is None:
        return False
    if kind == EITHER:
        return True
    if kind == HANDBOOK:
        pages = source.removeprefix("Student Handbook - ").replace("Pages", "").replace("Page", "")
        return source.startswith("Student Handbook") and any(str(p) in pages.replace(",", " ").split()
                                                             for p in expected)
    return source.startswith(config.WEBSITE_URL) and path_matches(source, expected)


def describe_expected(kind, expected) -> str:
    if kind == HANDBOOK:
        return "Handbook p" + ", ".join(str(p) for p in expected)
    if kind == WEBSITE:
        return " or ".join(expected[:2]) + (" ..." if len(expected) > 2 else "")
    return {EITHER: "Either source", NEITHER: "Not found"}[kind]


def cell(text):
    return (text or "-").replace("|", "\\|").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="base URL of a running server; default runs the app in-process")
    parser.add_argument("--out", type=Path, default=ROOT / "TEST_RESULTS.md")
    args = parser.parse_args()

    post = make_poster(args.url)
    rows, passed, seconds = [], 0, []
    for i, (kind, question, expected) in enumerate(QUESTIONS, start=1):
        started = time.perf_counter()
        response = post(question)
        seconds.append(time.perf_counter() - started)
        body = response.json()
        if response.status_code != 200:
            sys.exit(f"Q{i} failed with {response.status_code}: {body.get('error')}")

        ok = check(kind, expected, body["source"])
        passed += ok
        print(f"[{'pass' if ok else 'FAIL'}] {question}  ({seconds[-1]:.0f}s)\n       {body['source']}: {body['answer']}\n")
        rows.append(f"| {i} | {kind} | {cell(question)} | {cell(body['source'])} | {cell(body['answer'])} "
                    f"| {describe_expected(kind, expected)} | {'✅' if ok else '❌'} |")

    counts = {k: sum(1 for q in QUESTIONS if q[0] == k) for k in (HANDBOOK, WEBSITE, EITHER, NEITHER)}
    report = [
        "# Test results",
        "",
        f"Generated by `scripts/run_questions.py` on {date.today():%d %B %Y}. Every question went through "
        f"`POST /ask`, so this covers retrieval, generation and the API together.",
        "",
        f"- LLM: `{describe_llm()}`",
        f"- Embeddings: `{config.EMBEDDING_MODEL}`, {config.TOP_K_PER_SOURCE} chunks from each source, "
        f"reranked with `{config.RERANK_MODEL}`, similarity gate {config.MIN_SIMILARITY}",
        f"- Questions: {counts[HANDBOOK]} answered by the handbook, {counts[WEBSITE]} by the website, "
        f"{counts[EITHER]} by both (they disagree), {counts[NEITHER]} by neither",
        f"- **{passed}/{len(QUESTIONS)}** cited an acceptable source, or correctly said the answer isn't there",
        f"- Average time per question: {sum(seconds) / len(seconds):.0f}s",
        "",
        "✅/❌ checks the source only. Whether each answer is actually right is noted under the table.",
        "",
        "| # | Needs | Question | Retrieved source | Generated answer | Expected source | |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    args.out.write_text("\n".join(report), encoding="utf-8")
    print(f"{passed}/{len(QUESTIONS)} as expected. Written to {args.out.name}")


if __name__ == "__main__":
    main()
