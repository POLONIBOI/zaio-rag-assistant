import pytest

from app import config
from app.crawl import Section, WebPage
from app.crawl import load_pages as load_web_pages
from app.ingest import (HANDBOOK, WEBSITE, Chunk, Page, chunk_pages, chunk_web_page, chunk_website, clean_page,
                        load_all_chunks, load_pages, split_into_chunks, unspace, with_heading)

SITE = "https://www.zaio.io"


def test_unspace_joins_letter_spaced_text():
    assert unspace("C O N T A C T") == "CONTACT"
    assert unspace("E X C I T I N G  T I M E S  A W A I T") == "EXCITING TIMES AWAIT"
    assert unspace("1 3") == "13"


def test_unspace_leaves_normal_lines_alone():
    assert unspace("Total Fees: R 38,950") == "Total Fees: R 38,950"
    assert unspace("Option 2: Financing with Capitec and Manati") == "Option 2: Financing with Capitec and Manati"


def test_clean_page_separates_heading_and_drops_page_number():
    raw = "1 3\nThis is how we will determine the final grade.\nAssignments (25%):\nGRADE BREAKDOWN"
    section, body = clean_page(raw)
    assert section == "Grade Breakdown"
    assert body == "This is how we will determine the final grade. Assignments (25%):"


def test_clean_page_joins_multi_line_heading():
    section, _ = clean_page("Discord will be the primary platform.\nCOMMUNICATION\nCHANNELS")
    assert section == "Communication Channels"


def test_clean_page_rejoins_hyphenated_line_breaks():
    _, body = clean_page("algorithms, data structures, and problem-\nsolving techniques.")
    assert "problem-solving" in body


def test_sprint_table_topics_are_not_mistaken_for_headings():
    section, body = clean_page("Orientation\nHTML\nCSS\nGit\nJavaScript\nTHE SPRINTS")
    assert section == "The Sprints"
    assert "HTML CSS Git" in body


def test_short_text_is_a_single_chunk():
    assert split_into_chunks("One sentence. Another one.", size=100) == ["One sentence. Another one."]


def test_chunks_stay_under_size_and_overlap_by_a_sentence():
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    chunks = split_into_chunks(" ".join(sentences), size=120, overlap=1)

    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = previous.split(". ")[-1]
        assert current.startswith(last_sentence.rstrip("."))


def test_sentence_longer_than_chunk_is_wrapped_on_words():
    long_sentence = " ".join(["word"] * 100)  # 499 chars, no full stop
    chunks = split_into_chunks(long_sentence, size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == long_sentence.split()


def test_pages_without_heading_inherit_previous_section():
    pages = [
        Page(16, "Total Fees: R 38,950\nFEES"),
        Page(17, "Should your Capitec loan fail, we will take your application to Manati."),
    ]
    chunks = chunk_pages(pages)
    assert [(c.page, c.section) for c in chunks] == [(16, "Fees"), (17, "Fees")]


def test_table_of_contents_is_skipped():
    pages = [Page(2, "Welcome\n03.\nTable of Contents\nCONTENTS\nFees\n10."), Page(3, "Hello.\nWELCOME")]
    assert {c.page for c in chunk_pages(pages)} == {3}


def test_real_handbook():
    pages = load_pages(config.HANDBOOK_PATH)
    chunks = chunk_pages(pages)

    assert len(pages) == 26
    assert all(len(c.text) <= config.CHUNK_SIZE for c in chunks)
    assert 2 not in {c.page for c in chunks}
    assert len({c.id for c in chunks}) == len(chunks)

    fees = [c for c in chunks if "R 38,950" in c.text]
    assert len(fees) == 1
    assert (fees[0].page, fees[0].section) == (16, "Fees")


# --- website ----------------------------------------------------------------

@pytest.mark.parametrize("heading, text, expected", [
    ("How do the live classes work?", "Twice a week.", "How do the live classes work? Twice a week."),
    ("Upfront Payment", "Pay early and get a discount", "Upfront Payment: Pay early and get a discount"),
    ("", "No heading here.", "No heading here."),
])
def test_with_heading(heading, text, expected):
    assert with_heading(heading, text) == expected


def test_short_sections_are_packed_together():
    page = WebPage(SITE + "/tuition-financing", "Flexible Tuition Plans", [
        Section("Upfront Payment", "Invest in yourself, pay early and get a discount"),
        Section("Loans", "Apply for a loan with our financial partners"),
    ])
    [chunk] = chunk_web_page(page, size=200)

    assert chunk.text == ("Upfront Payment: Invest in yourself, pay early and get a discount "
                          "Loans: Apply for a loan with our financial partners")
    assert (chunk.source, chunk.url, chunk.title, chunk.section) == (
        WEBSITE, SITE + "/tuition-financing", "Flexible Tuition Plans", "Upfront Payment")


def test_long_section_is_split_with_its_heading_on_every_piece():
    text = " ".join(f"Refund rule number {i} applies here." for i in range(30))
    page = WebPage(SITE + "/terms", "Terms and Conditions", [Section("Refund Policy", text)])
    chunks = chunk_web_page(page, size=150)

    assert len(chunks) > 1
    assert all(len(c.text) <= 150 for c in chunks)
    assert all(c.text.startswith("Refund Policy: ") for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_section_without_heading_is_labelled_with_the_page_title():
    page = WebPage(SITE + "/apply-now", "Apply Now", [Section("", "Takes 3 minutes.")])
    assert chunk_web_page(page)[0].section == "Apply Now"


def test_metadata_only_has_the_fields_each_source_uses():
    # Chroma rejects None, so a handbook chunk can't carry url=None.
    book = Chunk(text="Total Fees: R 38,950", source=HANDBOOK, section="Fees", index=0, page=16)
    web = Chunk(text="...", source=WEBSITE, section="Refunds", index=2, url=SITE + "/terms", title="Terms")

    assert book.metadata == {"source": "handbook", "section": "Fees", "page": 16}
    assert web.metadata == {"source": "website", "section": "Refunds", "url": SITE + "/terms", "title": "Terms"}
    assert (book.id, web.id) == ("handbook-p16-c0", "website-terms-c2")


def test_website_ids_use_the_path():
    home = Chunk(text="...", source=WEBSITE, section="", index=0, url=SITE, title="Zaio")
    nested = Chunk(text="...", source=WEBSITE, section="", index=1, url=SITE + "/tuition/data-science", title="")
    assert (home.id, nested.id) == ("website-home-c0", "website-tuition-data-science-c1")


def test_page_title_is_part_of_what_gets_embedded():
    web = Chunk(text="Full refunds before the start date.", source=WEBSITE, section="", index=0,
                url=SITE + "/terms", title="Terms and Conditions")
    assert web.embedding_text == "Terms and Conditions: Full refunds before the start date."


def test_real_website_snapshot():
    pages = load_web_pages(config.WEBSITE_PAGES_PATH)
    chunks = chunk_website(pages)

    assert len(chunks) > 100
    assert all(len(c.text) <= config.CHUNK_SIZE for c in chunks)
    assert {c.url for c in chunks} == {p.url for p in pages}

    terms = [c for c in chunks if c.url == SITE + "/terms" and "Partial refunds" in c.text]
    assert terms, "the refund terms should be in a chunk of their own page"


def test_every_chunk_id_is_unique_across_both_sources():
    chunks = load_all_chunks()
    assert {c.source for c in chunks} == {HANDBOOK, WEBSITE}
    assert len({c.id for c in chunks}) == len(chunks)
