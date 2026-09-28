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
COLLECTION_NAME = "handbook"

# Small, fast, runs on CPU. 384-dim vectors.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")

# Pages in this handbook are short (60-1500 characters), so a chunk of ~600
# characters is usually one or two paragraphs - enough to hold a complete
# answer like the fee breakdown without dragging in the next topic.
CHUNK_SIZE = 600
CHUNK_OVERLAP_SENTENCES = 1

# On the 28 labelled questions in scripts/eval_retrieval.py the right page is
# the top hit 24 times, in the top 4 for 27 and in the top 5 for all 28.
# The miss at 4 is "What mark do I need to get placed?" - page 24 comes 5th.
TOP_K = 5

# If even the best chunk scores below this, the question isn't about the
# handbook ("what is the capital of France?" scores 0.10) and we answer
# "not found" without calling the LLM. Kept low on purpose: the lowest
# answerable question scores 0.33, and some unanswerable ones score higher
# than that ("pass mark for the MCQs?" is 0.47 because MCQs are on page 14),
# so similarity alone can't decide - the LLM makes the final call.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

NOT_FOUND_ANSWER = "I could not find that information in the handbook."

MAX_QUESTION_LENGTH = 500
