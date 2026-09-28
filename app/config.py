"""Settings for the handbook assistant.

Anything that can reasonably change between machines comes from the
environment (see .env.example). The rest are tuning values I picked after
looking at the handbook and running the test questions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

HANDBOOK_PATH = Path(os.getenv("HANDBOOK_PATH", ROOT / "data" / "handbook.pdf"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = "knowledge_base"  # handbook and website chunks together

# --- the website ---
WEBSITE_URL = "https://www.zaio.io"
WEBSITE_PAGES_PATH = Path(os.getenv("WEBSITE_PAGES_PATH", ROOT / "data" / "website" / "pages.json"))

# The site's sitemap lists 49 pages; the cap only stops a runaway crawl.
CRAWL_MAX_PAGES = 150
CRAWL_DELAY_SECONDS = 0.5

# /blog is a separate Ghost site with ~70 SEO articles ("Best coding bootcamps
# in South Africa 2025", "Zaio vs HyperionDev"...). Many are dated or about other
# schools, so course facts come from the main site only.
CRAWL_SKIP_PREFIXES = ("/blog",)

# A block that appears on this many pages is site furniture (the financing
# banner, "join our community", the shared FAQ) and is only indexed once.
REPEATED_SECTION_PAGES = 3

# Pages with less text than this after cleaning are empty shells (a
# "coming soon" page, a form with no copy) and aren't worth a chunk.
MIN_PAGE_CHARS = 150

# Small, fast, runs on CPU. 384-dim vectors.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# "auto" uses Claude when ANTHROPIC_API_KEY is set and the local model otherwise.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
# The local model has to fit next to everything else in 8 GB of RAM, which
# rules out anything much over 1B parameters. Of the two I tried,
# Qwen2.5-0.5B-Instruct said NOT FOUND to 6 of 7 answerable test questions
# (and got the 7th wrong); Qwen3-0.6B actually answers. Its settings are
# explained in app/local_llm.py.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-0.6B")
LOCAL_MAX_EXCERPTS = 3
LOCAL_MAX_NEW_TOKENS = 200
# The local model answers almost anything it's asked, so it only gets asked
# when the reranker thinks an excerpt fits (score above 0). In
# scripts/eval_retrieval.py that stops 10 of the 12 unanswerable questions
# that get past MIN_SIMILARITY, and 4 of the 50 answerable ones.
LOCAL_MIN_RERANK_SCORE = 0.0

# Pages in this handbook are short (60-1500 characters), so a chunk of ~600
# characters is usually one or two paragraphs - enough to hold a complete
# answer like the fee breakdown without dragging in the next topic.
CHUNK_SIZE = 600
CHUNK_OVERLAP_SENTENCES = 1

# Each source is searched separately and the results merged (see
# app/retriever.py). On the 50 labelled questions in scripts/eval_retrieval.py:
#
#   strategy             handbook  website  excerpts
#   pooled top 5           27/28    21/22       5
#   pooled top 10          27/28    21/22      10
#   per source, 4 each     27/28    21/22       8
#   per source, 5 each     28/28    21/22      10
#
# The one handbook question that separates them is "What mark do I need to
# get placed?": page 24 is only the 5th best handbook chunk, and in a pooled
# search /skillsdevelopment and the qualification pages push it out of even
# the top 10. The website miss is "How old do I need to be to enrol?" - the
# answer is buried in the long /terms page.
TOP_K_PER_SOURCE = 5
TOP_K = 10  # only used if per-source search is switched off

# Reorders those 10 chunks so the one that answers the question comes first
# (see app/reranker.py). RERANK_MODEL= (empty) switches it off.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# If even the best chunk scores below this, the question isn't about Zaio at
# all ("what is the capital of France?" scores 0.11, "how do I cook pasta?"
# 0.19) and we answer "not found" without calling the LLM. Kept low on
# purpose: the lowest answerable question scores 0.36, and some unanswerable
# ones score higher than that ("pass mark for the MCQs?" is 0.47 because MCQs
# are on page 14), so similarity alone can't decide - the LLM makes the call.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

NOT_FOUND_ANSWER = "I could not find that information in the available knowledge base."

MAX_QUESTION_LENGTH = 500
