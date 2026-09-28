# Zaio Student Assistant

An API that answers questions about Zaio from two sources: the Full-Stack AI Engineer Bootcamp student handbook (PDF) and the Zaio website, www.zaio.io. It finds the passages that cover the question in either source, has an LLM write the answer using only those passages, and returns where the answer came from: a handbook page or a zaio.io URL. If neither source covers the question, it says so instead of guessing.

This extends the handbook-only assistant from the previous practical (its two commits are the start of this repo's history).

```
POST /ask  {"question": "What is the refund policy?"}
→ {"answer": "The refund policy states that full refunds are available if requested before the bootcamp
              start date, partial refunds are available within the first 30 days with a 25% deduction, and
              no refunds are provided after 3 months of enrollment.",
   "source": "https://www.zaio.io/terms"}

POST /ask  {"question": "What are the minimum laptop requirements?"}
→ {"answer": "The minimum laptop requirements are a dual-core Intel i5 or Apple M1 processor with at least
              4 or 8 GB of RAM, and a solid-state drive (SSD) with at least 256 GB of storage.",
   "source": "Student Handbook - Page 5"}

POST /ask  {"question": "Is there a dress code for live classes?"}
→ {"answer": "I could not find that information in the available knowledge base.", "source": null}
```

These are real responses from the test run (local model, no API key). The first question was a "not found" in the handbook-only version.

## Where each part of the brief is

| Part | Marks | What it asks | Where |
|---|---|---|---|
| 1. Extend the knowledge base | 25 | Keep the PDF, crawl and clean zaio.io, chunk and embed both, one vector DB, metadata | `app/crawl.py`, `app/ingest.py`, `data/website/pages.json` |
| 2. Retrieve information | 25 | Embed the question, search both sources, answer from the context only, return the source, refuse when it isn't there | `app/retriever.py`, `app/reranker.py`, `app/llm.py`, `app/local_llm.py`, `app/assistant.py` |
| 3. Update the API | 20 | `POST /ask` returns `answer` and the most appropriate `source` | `app/api.py` |
| 4. Test the assistant | 15 | Handbook, website and unanswerable questions, each with question, source and answer | `scripts/run_questions.py` → `TEST_RESULTS.md` |
| 5. n8n | 10 | Take a question, call the API, show or forward the answer | `n8n/ask-zaio-workflow.json` |

## Running it

Needs Python 3.10 or newer (built on 3.12). An Anthropic API key is optional: without one, a small local model writes the answers instead (see [The LLM](#the-llm-claude-or-a-local-model)).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # optional: put your ANTHROPIC_API_KEY in .env

python -m app.ingest              # index the handbook + the website snapshot
uvicorn app.api:app --reload      # http://localhost:8000, interactive docs at /docs
```

The first run downloads the embedding model (~90 MB), the reranker (~90 MB) and, without an API key, Qwen3-0.6B (~1.5 GB).

The website is already crawled: `data/website/pages.json` is a snapshot taken on 28 September 2026. To crawl it again:

```bash
playwright install chromium       # once, ~115 MB
python -m app.crawl               # ~5 minutes, rewrites data/website/pages.json
python -m app.ingest
```

## How it works

```
Part 1 - python -m app.crawl, then python -m app.ingest
  www.zaio.io -> Playwright (headless Chromium), robots.txt respected -> clean HTML -> sections
              -> data/website/pages.json                             42 pages
  data/handbook.pdf -> pypdf, per page -> clean
  both -> chunk (<= 600 chars) -> embed (all-MiniLM-L6-v2) -> one ChromaDB collection
                                   49 handbook chunks + 342 website chunks, with source/page/url metadata

Part 2 - every POST /ask
  question -> embed -> ChromaDB: best 5 handbook chunks + best 5 website chunks
           -> cross-encoder reranks the 10 chunks
           -> best similarity under 0.25?  -> "I could not find that information in the available knowledge base."
           -> LLM, given only those excerpts, answers or says it isn't there
              (local model only: not asked at all if the best rerank score is under 0)
           -> not there?                     -> the same "could not find" sentence, source null
           -> {"answer": ..., "source": "Student Handbook - Page 16"} or {..., "source": "https://www.zaio.io/terms"}
```

## Part 1: getting the website into the knowledge base

### Crawling (`app/crawl.py`)

The brief suggests Puppeteer. The rest of the project is Python, so I used Playwright, which is the same idea (a real browser driven from code) with a Python API. A plain HTTP fetch wouldn't have been enough: `/refundpolicy`, `/affiliate-programme` and `/coming-soon` are served by Zaio's old single-page app, and without JavaScript they are an empty `<div id="root">`. In the browser they turn out to be client-side redirects to a sign-up page.

- **Where it starts.** The home page, plus every URL in the site's page sitemap (49 of them). From each page it follows links that stay on www.zaio.io, breadth first, with half a second between pages. Query strings and `#fragments` are dropped so the same page isn't crawled twice.
- **robots.txt.** zaio.io asks crawlers to stay out of `/app/`, `/course/`, `/static/`, `/welcome` and `/home`. Python's built-in `urllib.robotparser` gets this file wrong: it stops at the first matching rule, and the file lists `Allow: /` first, so it says `/app/` is fine. `RobotsRules` matches the way RFC 9309 says (longest rule wins), and `tests/test_crawl.py` has a test showing the difference on this exact file. Redirects are checked too: `/affiliate-programme` lands on `/app/getstarted`, so it's skipped.
- **The blog is left out.** `/blog` is a separate Ghost site with about 70 SEO articles ("Best coding bootcamps in South Africa 2025", "Zaio vs HyperionDev", "Zaio vs WeThinkCode"). Many are dated or about other schools, and an answer about fees or start dates should come from the course pages.
- **Result.** 47 pages rendered; 5 dropped as near-empty (the two certificate-verification forms, `/consultation`, and two pages that rendered blank); 42 kept.

The crawl result is saved to `data/website/pages.json` and committed. Indexing and the tests then don't need the network, and every answer can be traced back to a known snapshot of the site.

### Cleaning

`clean_html()` works on the rendered HTML:

- **Site chrome goes.** `<header>`, `<footer>`, `<nav>`, `<aside>`, dialogs and the cookie banner, plus scripts, styles, SVG icons and iframes. Only `<main>` is kept.
- **FAQ answers stay.** Every FAQ on the site is a collapsed accordion. Playwright's `inner_text()` skips hidden elements and would have dropped every answer, so the text comes from the HTML instead.
- **Words don't run together.** The site lays out labels as sibling `<span>`s spaced with CSS, e.g. `<span>With</span><span>NQF Level 4 certification</span>`. Read naively that's "WithNQF Level 4". Every tag boundary becomes whitespace.
- **Headings split the page into sections.** Each `h1`–`h4` starts a section, and a heading is flattened to one line first. The FAQ questions are `<button>`s inside `<h2>`s, and some headings contain `<br>`.
- **The comparison table is regrouped per course.** `/compare-courses` is a CSS-grid table. Read top to bottom it's "Monthly tuition R4,178 /mo R4,178 /mo R3,213 /mo ..." with nothing saying whose price is whose. It's rewritten as "Data Science: Monthly tuition R3,213 /mo; Upfront tuition R32,950; Duration ±30 weeks (≈ 7 months); ...".
- **Repeated blocks are kept once.** The course pages share the same financing banner, "join our community" block and FAQ. A block that appears on 3 or more pages is only kept on the first page that has it. Without this, the same FAQ answer, indexed once per course page, would fill the search results with copies of itself. It removed 27% of the text (183,461 → 134,377 characters).

`python -m app.ingest --preview website` prints every chunk so the result can be checked by eye.

### Chunking and metadata

The handbook is chunked exactly as before: per page, whole sentences packed into chunks of up to 600 characters, one sentence of overlap.

Web pages are chunked per section. A section's heading is kept in the chunk text ("How do the live classes work? We offer live classes twice a week..."), because on this site the heading is often half the answer. Short neighbouring sections are packed together up to 600 characters. A longer section is split on sentences, with its heading repeated on every piece. Chunks never cross a page, so each one has exactly one URL. The page title is added to the text that gets embedded, since it's often the only place the topic is named ("Refund Policy", "Flexible Tuition Options for Data Science").

Both sources go into one ChromaDB collection, `knowledge_base`: 391 chunks, 49 from the handbook and 342 from the website. Each chunk's metadata says where it came from:

| | `source` | `page` | `url` | `title` | `section` |
|---|---|---|---|---|---|
| handbook chunk | `"handbook"` | 16 | - | - | `"Fees"` |
| website chunk | `"website"` | - | `"https://www.zaio.io/terms"` | `"Terms and Conditions"` | `"Zaio Terms and Conditions"` |

ChromaDB can't store `None`, so each source only gets the fields it has.

## Part 2: finding the answer

### Searching both sources

The question is embedded with the same model as the chunks, and each source is searched separately: the best 5 handbook chunks and the best 5 website chunks. I compared this with a single pooled search on 50 labelled questions (28 handbook, 22 website; `python scripts/eval_retrieval.py --compare`):

| strategy | right handbook page sent to the LLM | right website page sent to the LLM | excerpts |
|---|---|---|---|
| pooled top 5 | 27/28 | 21/22 | 5 |
| pooled top 10 | 27/28 | 21/22 | 10 |
| per source, 4 each | 27/28 | 21/22 | 8 |
| **per source, 5 each** (used) | **28/28** | **21/22** | 10 |

The totals are close. The difference is the kind of question that got lost. The website has seven times as many chunks as the handbook, and for "What mark do I need to get placed at a company?" a pooled top 10 is eight website chunks (`/skillsdevelopment` and the qualification pages) and handbook page 21 twice. Page 24, which has the answer, never reaches the LLM. Searching each source separately also means that when the handbook and the website disagree, the LLM sees both. The one website miss is "How old do I need to be to enrol?": the answer is buried in the long `/terms` page.

### Reranking

The embedding search compares vectors that were made separately. A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB) reads the question and each chunk together and reorders the 10 candidates. It costs about a quarter of a second per question:

| order of the 10 excerpts | right page first | in the top 2 | in the top 3 |
|---|---|---|---|
| by similarity | 39/50 | 43/50 | 46/50 |
| reranked | 44/50 | 47/50 | 49/50 |

Reranking moved the right page for 9 of the 50 questions: 8 up, 1 down. The biggest move is "What mark do I need to get placed at a company?". By similarity, handbook page 24 is 10th of the 10 excerpts, behind `/skillsdevelopment`; reranked, it's first. "Who do I email about laptop advice?" goes from 4th to 2nd, which puts page 5 (the one with asif@zaio.io) inside the 3 excerpts the local model sees. The one that got worse is "Which certification do I get from the Cybersecurity bootcamp?", from 1st to 3rd.

### When to say "not found"

Same idea as the handbook version, with the numbers redone for both sources:

1. If even the best chunk scores under 0.25 similarity, the question isn't about Zaio at all ("What is the capital of France?" 0.11, "How do I cook pasta?" 0.19). The API answers "not found" without calling the LLM. The lowest-scoring answerable question is at 0.36.
2. Otherwise the LLM decides, because similarity can't. "What is the pass mark for the MCQs?" scores 0.47 (page 14 is about MCQs, but gives only their weighting), higher than several questions that are answered.

I also checked whether the reranker's score could make this call. It can't: answerable questions go as low as -8.5 (the laptop email) and unanswerable ones as high as +3.5 ("Does Zaio offer medical aid for students?"). It's used as a safety net for the local model only (below).

When the answer isn't there, the response is always exactly `"I could not find that information in the available knowledge base."` with `"source": null`, so n8n can branch on it.

### Which source is returned

`source` is a single string, as in the brief's examples:

- a handbook answer → `"Student Handbook - Page 18"`, or `"Student Handbook - Pages 16, 17"` when it uses two pages;
- a website answer → the page URL, e.g. `"https://www.zaio.io/terms"`.

The LLM lists the excerpts it actually used, most important first, and the first one decides which kind of source it is. That isn't always the top search hit: even after reranking, the right page comes first for 44 of the 50 labelled questions, not all of them. Excerpt numbers that weren't sent are ignored. If nothing usable is left, the best-ranked chunk is used.

### When the handbook and the website disagree

They do. The handbook was written for one intake (May 2025) and quotes R 38,950 for the bootcamp. The website now lists R42,850 upfront for the Fullstack AI Engineer bootcamp. The handbook's start date is a `<DATE>` placeholder, while the website gives 28 September 2026 for the same bootcamp. Claude's prompt tells it to give both figures and say which source says what, rather than silently pick one.

### The LLM: Claude, or a local model

`LLM_PROVIDER=auto` (the default) uses Claude when `ANTHROPIC_API_KEY` is set and a local model when it isn't. The project was built on a machine without a key, and whoever marks it may not have one either.

**Claude** (`claude-opus-5`, `app/llm.py`) is the one this is designed for. It gets all 10 excerpts, numbered and labelled with their page or URL, and must reply with structured JSON: `found`, `answer`, `excerpts`. Effort is `low`, since these are short lookups behind a webhook, and Anthropic's server-side fallback is on in case a safety classifier declines a request.

**The local model** (`Qwen/Qwen3-0.6B`, `app/local_llm.py`) runs on the CPU in 8 GB of RAM. That rules out anything much bigger than 1B parameters, and a model that small needs a simpler job:

- It only answers or says NOT FOUND. Asked to cite excerpt numbers too, it started writing "The number of the excerpt is." into its answers. The source is worked out afterwards: whichever excerpt shares the most distinctive words with the answer, with figures and emails counting three times.
- It sees the top 3 reranked excerpts, not 10.
- It isn't asked at all if the reranker's best score is below 0. Asked anyway, it answered "What is the pass mark for the MCQs?" with "15%", which is the MCQ weighting. Of the 12 unanswerable test questions that get past the similarity gate, this check stops 10. It also stops 4 of the 50 answerable ones (`python scripts/eval_retrieval.py` prints both).
- I tried Qwen2.5-0.5B-Instruct first: it said NOT FOUND to 6 of 7 answerable test questions. Qwen3's thinking mode would help the 0.6B check its own answers, but on this laptop ten questions hadn't finished after half an hour, so it's off.

Its mistakes are listed in `TEST_RESULTS.md`.

## Part 3: the API

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What courses does ZAIO offer?"}'
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" `
  -Body '{"question": "What courses does ZAIO offer?"}'
```

The response is always `{"answer": ..., "source": ...}`, with `source` a handbook page, a zaio.io URL, or `null`. Errors are always `{"error": "..."}`:

| Status | When |
|---|---|
| 200 | Answered, or not in either source (`source` is `null`) |
| 400 | Body isn't valid JSON, or isn't a JSON object |
| 422 | `question` is missing, empty, not a string, or longer than 500 characters |
| 404 / 405 | Wrong path, or wrong method (e.g. `GET /ask`) |
| 502 | The LLM call failed (details go to the server log, not the caller) |
| 503 | Index not built yet, or `LLM_PROVIDER=claude` without an API key |

Surrounding whitespace in `question` is trimmed and extra fields are ignored, because n8n often sends extra fields from earlier nodes.

### `GET /health`

```json
{"status": "ok", "chunks": {"handbook": 49, "website": 342}, "llm": "local: Qwen/Qwen3-0.6B"}
```

## Part 4: tests

```bash
pytest
```

166 tests, all passing (output in [pytest_results.txt](pytest_results.txt)):

| File | Tests | What it checks |
|---|---|---|
| `tests/test_crawl.py` | 45 | robots.txt rules on zaio.io's own file, including the case the standard library gets wrong; URL normalising; HTML cleaning on markup shaped like the real pages (chrome removed, collapsed FAQ answers kept, span spacing, the grid table); repeated-block removal; the committed snapshot (only allowed pages, no blog, the FAQ is there) |
| `tests/test_ingest.py` | 23 | Handbook cleaning and chunking as before, plus website chunking: headings folded in, packing, splitting, metadata per source, ids unique across both sources |
| `tests/test_retriever.py` | 4 | Per-source queries and merging, with a fake ChromaDB |
| `tests/test_retrieval.py` | 21 | The real index: right handbook page and right website page for labelled questions, page 24 reaching the LLM, what reranking moves, off-topic questions under the gate. Skipped until `python -m app.ingest` has run |
| `tests/test_assistant.py` | 16 | The two "not found" paths, the exact fallback sentence, how `source` is chosen and formatted |
| `tests/test_llm.py` | 12 | Claude's prompt and structured output (fake client), refusals, API failures, choosing Claude or the local model |
| `tests/test_local_llm.py` | 19 | The local model's prompt, reply clean-up, source attribution and the reranker check (fake model) |
| `tests/test_api.py` | 26 | Every kind of invalid request gets a clear JSON error; handbook and website sources; `/health` |

No test calls the Claude API or loads the local LLM, so the suite costs nothing. It takes about a minute, most of it loading the embedding model and reranker for the retrieval tests.

Two scripts go further:

- `python scripts/eval_retrieval.py` (add `--compare` for the strategy tables) measures retrieval on the 50 labelled questions. The numbers in this README come from it.
- `python scripts/run_questions.py` is Part 4. It puts 16 questions through `POST /ask` and writes [TEST_RESULTS.md](TEST_RESULTS.md) with the question, the source and the generated answer: 5 answered by the handbook, 6 by the website, 1 by both (they disagree on the price) and 4 by neither.

With the local model, 15 of the 16 cited an acceptable source or correctly said "not found". Checked by hand, 9 answers are right, 5 partly right and 2 wrong. `TEST_RESULTS.md` goes through each one, including why the two wrong ones went wrong.

## Part 5: n8n

`n8n/ask-zaio-workflow.json`. In n8n, import it (Import from File), publish it, and open the form's production URL from the first node.

```
Student asks a question (Form) -> Settings -> Ask the RAG API (POST /ask) -> Read the reply -> What came back?
    answered   -> Show the answer          (answer + source)
    not found  -> Discord webhook set? -> Tell the tutors on Discord -> Say it wasn't found
    error      -> Show an error
```

- **Input.** An n8n Form with one question field. No credentials to set up.
- **Settings.** `apiUrl` is `http://host.docker.internal:8000` for n8n in Docker (inside a container, `localhost` is the container itself), or `http://localhost:8000` for n8n installed with npm. Start the API with `uvicorn app.api:app --host 0.0.0.0 --port 8000` so Docker can reach it.
- **The call.** HTTP Request, `POST {apiUrl}/ask` with `{"question": ...}`, with a 5-minute timeout because of the local model. It's set never to throw, so an HTTP error or an API that's down comes through as data, and the **Read the reply** code node sorts every reply into `answered`, `not found` or `error`.
- **Presenting it.** The answer and its source are shown on the form's result page.
- **Forwarding it.** If `discordWebhookUrl` is set (Discord channel settings, Integrations, Webhooks), questions the assistant can't answer are posted to that channel. The handbook names Discord as the bootcamp's main platform, so that's where a tutor would see it and follow up. If the post fails, the student still gets their page.

I tested it in n8n 2.40.7 (Docker) against the running API:

| Test | What the student saw |
|---|---|
| "Who is the CEO of Zaio?" | **Here's what I found.** The CEO of Zaio is Mvelo Hlophe. Source: https://www.zaio.io/aboutus |
| "Is there a dress code for live classes?" | **Not in the knowledge base.** I could not find that information in the available knowledge base. It isn't in the student handbook or on zaio.io. Ask your tutor on Discord, or email hello@zaio.io. |
| Any question, API stopped | **The assistant is offline.** Your question couldn't be answered right now. Please try again in a minute, or ask a tutor on Discord. |
| "Can I bring my dog to the live classes?", webhook pointed at a local stand-in for Discord | The not-found page. The stand-in received `{"username": "Zaio assistant", "content": "A student asked something the assistant could not answer from the handbook or zaio.io:\n> Can I bring my dog to the live classes?"}`, which is the JSON a Discord webhook takes |

## Limitations

- **The website snapshot ages.** It's from 28 September 2026. Prices, start dates and events change, so re-crawl before relying on it.
- **The handbook is the May 2025 edition** and still contains template placeholders (`<DATE>`, `<TO BE ADDED>`), which the assistant repeats.
- **Text only.** Images aren't read: the handbook's calendar and salary graph, and on the website anything shown only in a picture.
- **The blog isn't indexed** (see Crawling), so questions answered only there come back as "not found".
- **The local model makes mistakes** (2 wrong and 5 partly right out of 16 in `TEST_RESULTS.md`). It's a fallback so the project runs without a key.
- **Claude hasn't been run for real.** There was no API key on the machine this was built on, so the Claude path is only covered by unit tests with a fake client. Put a key in `.env` and run `python scripts/run_questions.py` to see its answers.
- It takes one question at a time and doesn't remember the conversation.

## Project layout

```
app/
  config.py         settings and tuning values, each with the reason for it
  crawl.py          Part 1: crawl + clean zaio.io (python -m app.crawl)
  ingest.py         Part 1: chunk both sources, embed, store (python -m app.ingest)
  embeddings.py     the sentence-transformers model, shared by ingest and queries
  vector_store.py   ChromaDB, persisted in chroma_db/
  retriever.py      question -> best chunks from each source
  reranker.py       cross-encoder that reorders them
  llm.py            Claude, and choosing between Claude and the local model
  local_llm.py      the local model
  assistant.py      retrieval + generation + the "not found" rules + the source
  api.py            FastAPI app
data/
  handbook.pdf
  website/pages.json   cleaned crawl of www.zaio.io
n8n/ask-zaio-workflow.json
scripts/            eval_retrieval.py, run_questions.py
tests/
```
