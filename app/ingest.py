"""Part 1: turn the handbook PDF and the crawled website into chunks in ChromaDB.

    python -m app.ingest                         # build (or rebuild) the index
    python -m app.ingest --preview handbook      # print chunks without embedding anything
    python -m app.ingest --preview website

Handbook: load PDF -> extract text per page -> clean -> chunk -> embed -> store.
Website:  data/website/pages.json (from `python -m app.crawl`) -> chunk -> embed -> store.

Both go into the same collection. A chunk never crosses a PDF page or a web
page, so every chunk has exactly one page number or one URL and the API can
always say where an answer came from.
"""

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app import config

HANDBOOK = "handbook"
WEBSITE = "website"

# The handbook prints page numbers letter-spaced ("1 3") and some headings the
# same way ("C O N T A C T"). Words inside those lines are two spaces apart.
LETTER_SPACED = re.compile(r"(?:\S {1,2})+\S")
PAGE_NUMBER = re.compile(r"\d{1,2}")
# Section titles are full lines in capitals, e.g. "GRADE BREAKDOWN". In this
# PDF they sit at the bottom of the page, after the body text.
HEADING = re.compile(r"[A-Z][A-Z& ]{2,}")
# ...except in the sprint table on page 6, where these are course topics that
# happen to be on a line of their own.
NOT_HEADINGS = {"HTML", "CSS", "NPM"}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")

# Page 2 is the table of contents: nothing but section names and page numbers
# that don't match the PDF. Left in, it outranks the real pages for questions
# like "what are the fees?" and the answer ends up citing the wrong page.
SKIP_SECTIONS = {"Contents"}


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as shown in a PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # HANDBOOK or WEBSITE
    section: str
    index: int  # position within its page
    page: int | None = None  # handbook only
    url: str | None = None  # website only
    title: str = ""  # website page title

    @property
    def id(self) -> str:
        if self.source == HANDBOOK:
            return f"handbook-p{self.page}-c{self.index}"
        slug = self.url.removeprefix(config.WEBSITE_URL).strip("/").replace("/", "-") or "home"
        return f"website-{slug}-c{self.index}"

    @property
    def embedding_text(self) -> str:
        # Prefixing the section name gives short chunks more to match on:
        # "Option 2: Financing with Capitec..." is easier to find for a
        # question about fees when the vector also knows it's under "Fees".
        # Web chunks get the page title too ("Refund Policy", "Data Science
        # Tuition"), which is often the only place the topic is named.
        if self.source == HANDBOOK:
            return f"{self.section}: {self.text}"
        return f"{self.title}: {self.text}" if self.title else self.text

    @property
    def metadata(self) -> dict:
        # Chroma can't store None, so each source only gets the fields it has.
        meta = {"source": self.source, "section": self.section}
        if self.source == HANDBOOK:
            meta["page"] = self.page
        else:
            meta["url"] = self.url
            meta["title"] = self.title
        return meta


# --- handbook ---------------------------------------------------------------

def load_pages(pdf_path: Path) -> list[Page]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Handbook not found at {pdf_path}")
    reader = PdfReader(pdf_path)
    return [Page(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def unspace(line: str) -> str:
    """'C O N T A C T' -> 'CONTACT', '1 3' -> '13'. Other lines are returned as-is."""
    if not LETTER_SPACED.fullmatch(line) or any(len(tok) > 1 for tok in line.split()):
        return line
    return " ".join(word.replace(" ", "") for word in re.split(r" {2,}", line))


def clean_page(raw: str) -> tuple[str, str]:
    """Split a page's raw text into (section title, body text)."""
    headings, body = [], []
    for line in raw.splitlines():
        line = unspace(line.strip())
        if not line or PAGE_NUMBER.fullmatch(line):
            continue
        if HEADING.fullmatch(line) and line not in NOT_HEADINGS:
            headings.append(line)
        else:
            body.append(line)

    text = "\n".join(body)
    text = re.sub(r"-\n(?=[a-z])", "-", text)  # "problem-\nsolving" -> "problem-solving"
    text = re.sub(r"\s+", " ", text).strip()
    section = " ".join(headings).title()
    return section, text


def split_into_chunks(text: str, size: int = config.CHUNK_SIZE,
                      overlap: int = config.CHUNK_OVERLAP_SENTENCES) -> list[str]:
    """Pack whole sentences into chunks of at most ~`size` characters.

    The last `overlap` sentence(s) of a chunk are repeated at the start of the
    next one so an answer that straddles the cut is still found in one piece.
    Sentences longer than `size` (the sprint table has no full stops at all)
    are wrapped on word boundaries.
    """
    pieces = []
    for sentence in SENTENCE_END.split(text):
        sentence = sentence.strip()
        if len(sentence) > size:
            pieces.extend(textwrap.wrap(sentence, size))
        elif sentence:
            pieces.append(sentence)

    chunks, current = [], []
    for piece in pieces:
        if current and len(" ".join([*current, piece])) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            if len(" ".join([*current, piece])) > size:
                current = []  # no room for the overlap - start clean
        current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    chunks = []
    section = ""
    for page in pages:
        title, text = clean_page(page.text)
        # Pages that continue a topic (e.g. page 17, more on fees) have no
        # heading of their own, so they inherit the previous one.
        section = title or section
        if not text or section in SKIP_SECTIONS:
            continue
        for i, piece in enumerate(split_into_chunks(text)):
            chunks.append(Chunk(text=piece, source=HANDBOOK, page=page.number, section=section, index=i))
    return chunks


# --- website ----------------------------------------------------------------

def with_heading(heading: str, text: str) -> str:
    """'How do live classes work?' + 'Twice a week...' -> one readable line.

    The heading stays in the chunk text because on this site it's often half
    the answer: an FAQ answer that starts "No, our bootcamps are designed for
    beginners!" means nothing without the question above it.
    """
    if not heading:
        return text
    return f"{heading} {text}" if heading[-1] in ".?!:" else f"{heading}: {text}"


def chunk_web_page(page, size: int = config.CHUNK_SIZE) -> list[Chunk]:
    """Sections of one web page -> chunks of up to `size` characters.

    Neighbouring short sections are packed together (a course card is often
    a heading and one line), and a section longer than `size` is split on
    sentences with its heading repeated on every piece.
    """
    blocks: list[tuple[str, str]] = []  # (heading, text) with the heading already folded in
    for section in page.sections:
        full = with_heading(section.heading, section.text)
        if len(full) <= size:
            blocks.append((section.heading, full))
            continue
        room = size - len(section.heading) - 2
        for piece in split_into_chunks(section.text, size=room):
            blocks.append((section.heading, with_heading(section.heading, piece)))

    chunks, texts, heading = [], [], ""

    def flush():
        if texts:
            chunks.append(Chunk(text=" ".join(texts), source=WEBSITE, url=page.url, title=page.title,
                                section=heading or page.title, index=len(chunks)))

    for block_heading, text in blocks:
        if texts and len(" ".join([*texts, text])) > size:
            flush()
            texts = []
        if not texts:
            heading = block_heading
        texts.append(text)
    flush()
    return chunks


def chunk_website(pages) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_web_page(page)]


# --- index ------------------------------------------------------------------

def load_all_chunks(pdf_path: Path = config.HANDBOOK_PATH,
                    website_path: Path = config.WEBSITE_PAGES_PATH) -> list[Chunk]:
    from app.crawl import load_pages as load_web_pages

    return chunk_pages(load_pages(pdf_path)) + chunk_website(load_web_pages(website_path))


def build_index(pdf_path: Path = config.HANDBOOK_PATH,
                website_path: Path = config.WEBSITE_PAGES_PATH) -> list[Chunk]:
    from app import vector_store
    from app.embeddings import embed_texts

    chunks = load_all_chunks(pdf_path, website_path)
    vectors = embed_texts([c.embedding_text for c in chunks])

    collection = vector_store.reset_collection()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the handbook and the website into ChromaDB.")
    parser.add_argument("--pdf", type=Path, default=config.HANDBOOK_PATH)
    parser.add_argument("--website", type=Path, default=config.WEBSITE_PAGES_PATH)
    parser.add_argument("--preview", choices=[HANDBOOK, WEBSITE],
                        help="print the cleaned chunks of one source and stop (no embeddings, no database)")
    args = parser.parse_args()

    if args.preview:
        for chunk in load_all_chunks(args.pdf, args.website):
            if chunk.source == args.preview:
                where = f"page {chunk.page}" if chunk.page else chunk.url
                print(f"--- {chunk.id} | {where} | {chunk.section} | {len(chunk.text)} chars")
                print(chunk.text)
        return

    chunks = build_index(args.pdf, args.website)
    handbook = [c for c in chunks if c.source == HANDBOOK]
    website = [c for c in chunks if c.source == WEBSITE]
    print(f"Indexed {len(chunks)} chunks into '{config.COLLECTION_NAME}':")
    print(f"  handbook  {len(handbook):>4} chunks from {len({c.page for c in handbook})} pages of {args.pdf.name}")
    print(f"  website   {len(website):>4} chunks from {len({c.url for c in website})} pages of {config.WEBSITE_URL}")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Stored in: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
