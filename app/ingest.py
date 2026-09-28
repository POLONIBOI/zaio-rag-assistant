"""Part 1: turn the handbook PDF into searchable chunks in ChromaDB.

    python -m app.ingest             # build (or rebuild) the index
    python -m app.ingest --preview   # print the chunks without embedding anything

Pipeline: load PDF -> extract text per page -> clean -> chunk -> embed -> store.
Chunks never cross a page boundary, so every chunk carries exactly one page
number and the API can always say where an answer came from.
"""

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app import config

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
    page: int
    section: str
    index: int  # position within its page

    @property
    def id(self) -> str:
        return f"p{self.page}-c{self.index}"

    @property
    def embedding_text(self) -> str:
        # Prefixing the section name gives short chunks more to match on:
        # "Option 2: Financing with Capitec..." is easier to find for a
        # question about fees when the vector also knows it's under "Fees".
        return f"{self.section}: {self.text}"


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
            chunks.append(Chunk(text=piece, page=page.number, section=section, index=i))
    return chunks


def build_index(pdf_path: Path = config.HANDBOOK_PATH) -> list[Chunk]:
    from app import vector_store
    from app.embeddings import embed_texts

    chunks = chunk_pages(load_pages(pdf_path))
    vectors = embed_texts([c.embedding_text for c in chunks])

    collection = vector_store.reset_collection()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"page": c.page, "section": c.section, "source": "handbook"} for c in chunks],
    )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the student handbook into ChromaDB.")
    parser.add_argument("--pdf", type=Path, default=config.HANDBOOK_PATH)
    parser.add_argument("--preview", action="store_true",
                        help="print the cleaned chunks and stop (no embeddings, no database)")
    args = parser.parse_args()

    if args.preview:
        for chunk in chunk_pages(load_pages(args.pdf)):
            print(f"--- {chunk.id} | {chunk.section} | {len(chunk.text)} chars")
            print(chunk.text)
        return

    chunks = build_index(args.pdf)
    pages = sorted({c.page for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(pages)} pages of {args.pdf.name}")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Stored in: {config.CHROMA_DIR} (collection '{config.COLLECTION_NAME}')")


if __name__ == "__main__":
    main()
