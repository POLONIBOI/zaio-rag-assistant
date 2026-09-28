# Zaio Handbook Assistant

An API that answers student questions from the Full-Stack AI Engineer Bootcamp handbook. It finds the part of the handbook that covers the question, has Claude write the answer using only that text, and says which page it came from. If the handbook doesn't cover the question, it says so instead of guessing.

```
POST /ask  {"question": "How much does the bootcamp cost?"}
→ {"answer": "The total fee is R 38,950, payable in full before orientation day.", "source": "Page 16"}

POST /ask  {"question": "What is the refund policy?"}
→ {"answer": "I could not find that information in the handbook.", "source": null}
```

## Running it

Needs Python 3.10 or newer (built on 3.12) and an Anthropic API key.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # then put your ANTHROPIC_API_KEY in .env

python -m app.ingest              # build the vector index (first run downloads a ~90 MB embedding model)
uvicorn app.api:app --reload      # http://localhost:8000, interactive docs at /docs
```

## How it works

```
Part 1 - python -m app.ingest (once)
  data/handbook.pdf -> pypdf, per page -> clean -> chunk -> embed (all-MiniLM-L6-v2) -> ChromaDB
                                                             49 chunks from 25 pages

Part 2 - every POST /ask
  question -> embed -> ChromaDB, top 5 chunks
           -> best chunk scores under 0.25?  -> "I could not find that information in the handbook."
           -> Claude, given only those 5 excerpts, returns {found, answer, pages}
           -> found is false?                -> "I could not find that information in the handbook."
           -> {"answer": ..., "source": "Page 16"}
```

| Brief | Where |
|---|---|
| Part 1: load, extract, chunk, embed, store | `app/ingest.py`, `app/embeddings.py`, `app/vector_store.py` |
| Part 2: embed question, search, retrieve, generate | `app/retriever.py`, `app/llm.py`, `app/assistant.py` |
| Part 3: `POST /ask` | `app/api.py` |
| Part 4: test questions | `scripts/run_questions.py` → `TEST_RESULTS.md` |
| Part 5: JSON in/out, invalid requests | `app/api.py`, `tests/test_api.py` |

## Why it's built this way

### Cleaning is specific to this PDF

The raw pypdf output of the handbook made a few problems obvious:

- Page numbers are printed letter-spaced (`1 3`), and so are some headings (`C O N T A C T`). These get joined back together and bare page numbers are dropped.
- Each page's heading (`FEES`, `GRADE BREAKDOWN`, ...) comes out at the *bottom* of the page text. The cleaner pulls it out and stores it as the chunk's `section`. Pages that continue a topic inherit the previous heading (page 17 is more about fees). The section name is also put in front of the text that gets embedded, which helps short chunks match.
- In the sprint table on page 6, `HTML`, `CSS` and `NPM` are on lines of their own and looked like headings. They're excluded by name.
- Page 2, the table of contents, is skipped. It's only headings and page numbers, and it kept outranking the real fees page.

`python -m app.ingest --preview` prints every chunk so you can check the result.

### Page numbers

`source` uses the PDF's own page numbers, the ones a PDF viewer shows. On pages 5–14 the number printed on the page is one lower, so "Page 14" is the grade breakdown page, printed as "13".

### Chunking

Chunks are made per page, so each chunk has exactly one page number. Within a page, whole sentences are packed into chunks of up to 600 characters, and the last sentence of each chunk is repeated at the start of the next. The pages are short (60–1,500 characters), so most pages become one to three chunks, and an answer like the fee breakdown stays in one piece.

### Picking the embedding model

I labelled 28 questions with the page that answers them (`scripts/eval_retrieval.py`) and compared four sentence-transformers models on them:

| Model | Right page ranked first | Right page in top 5 |
|---|---|---|
| **all-MiniLM-L6-v2** (used) | 24/28 | 28/28 |
| BAAI/bge-small-en-v1.5 | 24/28 | 28/28 |
| multi-qa-MiniLM-L6-cos-v1 | 24/28 | 27/28 |
| all-mpnet-base-v2 | 24/28 | 26/28 |

MiniLM ties for best and is the fastest on a CPU. bge-small puts all of its scores in a narrow band ("What is the capital of France?" still scores 0.43), so a similarity cut-off means very little with it. Only one question needs all five results: for "What mark do I need to get placed at a company?", page 24 comes fifth. That's why `TOP_K` is 5.

### Two checks before "not found"

Similarity alone can't separate answerable questions from unanswerable ones here. "What is the pass mark for the MCQs?" scores 0.47 because page 14 talks about MCQs, while "What mark do I need to get placed?", which *is* answered, scores only 0.33. So there are two checks:

1. If even the best chunk scores under 0.25, the question isn't about the handbook ("capital of France" 0.10, "how do I cook pasta" 0.19). The API returns the fallback straight away without calling Claude.
2. Otherwise Claude gets the top 5 excerpts and must reply with structured JSON: `found`, `answer`, `pages`. The system prompt tells it to answer only from the excerpts, and that a related excerpt isn't an answer. If `found` is false, the API returns the exact fallback sentence, so a workflow can match on it (or just check that `source` is null).

### Where `source` comes from

`source` is the page Claude says it used, not simply the top search hit, because the best-scoring chunk isn't always the one with the answer. For "Who do I email about laptop advice?", page 20 scores highest, but the email address (asif@zaio.io) is on page 5. Any page Claude cites that wasn't among the retrieved excerpts is thrown away. If nothing usable is left, the top hit's page is used.

### The LLM call

Claude Opus 5 (`claude-opus-5`) through the official `anthropic` SDK:

- Effort is set to `low`. These are short factual lookups, and it keeps the response time reasonable behind an n8n webhook.
- Anthropic's server-side fallback is on. If a safety filter ever declines a request, the API retries it on another model instead of failing.

You can change the model with `CLAUDE_MODEL` in `.env`.

## API

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When are the live classes?"}'
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" `
  -Body '{"question": "When are the live classes?"}'
```

Every response is JSON. Errors always have the form `{"error": "..."}`:

| Status | When |
|---|---|
| 200 | Answered, or not in the handbook (`source` is `null`) |
| 400 | Body isn't valid JSON, or isn't a JSON object (plain text, form data, a list, no body) |
| 422 | `question` is missing, empty, not a string, or longer than 500 characters |
| 404 / 405 | Wrong path, or wrong method (e.g. `GET /ask`) |
| 502 | The Claude API call failed (the details go to the server log, not the caller) |
| 503 | Index not built yet, or `ANTHROPIC_API_KEY` not set |

Surrounding whitespace in `question` is trimmed, and extra fields in the body are ignored.

### `GET /health`

`{"status": "ok", "chunks": 49}`, or a 503 explaining that the index needs building.

## Tests

```bash
pytest
```

66 tests:

| File | What it checks |
|---|---|
| `tests/test_ingest.py` | Cleaning and chunking, including on the real PDF: 26 pages, the fees chunk lands on page 16 under "Fees", the contents page is skipped, no chunk is over 600 characters |
| `tests/test_retrieval.py` | Real embeddings and ChromaDB: the top hit is the right page for 8 questions, and off-topic questions fall under the threshold. Skipped until `python -m app.ingest` has run |
| `tests/test_assistant.py` | The two "not found" paths and how `source` is chosen, using a fake retriever and LLM |
| `tests/test_llm.py` | Prompt building, parsing Claude's JSON, refusals, API failures, a missing key (fake client) |
| `tests/test_api.py` | Each kind of invalid request gets a clear JSON error, and server-side problems map to 502/503 |

No test calls the Claude API, so the suite costs nothing and runs offline once the embedding model has been downloaded.

Two scripts go further:

- `python scripts/eval_retrieval.py` shows retrieval accuracy on the 28 labelled questions. No API key needed.
- `python scripts/run_questions.py` puts 12 questions through `POST /ask` and writes `TEST_RESULTS.md` with the question, the source page and the answer. Nine of the questions are answered in the handbook and three aren't. One of the three is the brief's own example, "What is the attendance requirement?": the handbook covers live-class times and recordings but never states an attendance requirement.

## Using it from n8n

In an **HTTP Request** node:

- Method `POST`, URL `http://localhost:8000/ask`. If n8n runs in Docker, use `http://host.docker.internal:8000/ask`, because `localhost` inside the container is the container itself.
- Send the body as JSON: `{"question": "{{ $json.question }}"}`.
- `answer` and `source` in the reply can be passed straight on. `source` is `null` when the handbook doesn't cover the question, which makes a simple IF-node condition.

## Limitations

- The handbook is the May 2025 cohort edition and still contains template placeholders: the start date is literally `<DATE>` and the orientation link is `<TO BE ADDED>`. The assistant repeats what the handbook says, placeholders included.
- The handbook contradicts itself on length (7 months on page 7, 7.5 months on page 8), so an answer about duration depends on which page is retrieved.
- Only text is extracted. The example calendar (page 8) and the Offerzen salary graph (page 25) are images, so anything shown only in them can't be answered.
- It takes one question at a time and doesn't remember the conversation.

## Project layout

```
app/
  config.py         settings and tuning values, each with the reason for it
  ingest.py         Part 1 pipeline (python -m app.ingest)
  embeddings.py     the sentence-transformers model, shared by ingest and queries
  vector_store.py   ChromaDB, persisted in chroma_db/
  retriever.py      question -> top-k chunks with similarity scores
  llm.py            the Claude call and its structured output
  assistant.py      retrieval + generation + the "not found" rules
  api.py            FastAPI app
data/handbook.pdf
scripts/            eval_retrieval.py, run_questions.py
tests/
```
