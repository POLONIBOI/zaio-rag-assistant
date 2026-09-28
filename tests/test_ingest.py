from app import config
from app.ingest import Page, chunk_pages, clean_page, load_pages, split_into_chunks, unspace


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
