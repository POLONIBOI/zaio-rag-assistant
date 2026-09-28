import pytest

from app import config
from app.assistant import Assistant, format_source
from app.llm import LLMAnswer
from app.retriever import RetrievedChunk

TERMS = "https://www.zaio.io/terms"
COMPARE = "https://www.zaio.io/compare-courses"


def handbook(page, similarity=0.5, text="..."):
    return RetrievedChunk(text=text, source="handbook", section="Fees", page=page, similarity=similarity)


def website(url, similarity=0.5, text="..."):
    return RetrievedChunk(text=text, source="website", section="", url=url, title="Page", similarity=similarity)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, question):
        return self.chunks


class FakeLLM:
    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def answer(self, question, chunks):
        self.calls.append((question, chunks))
        return self.reply


def make(chunks, reply=None):
    llm = FakeLLM(reply)
    return Assistant(FakeRetriever(chunks), llm, min_similarity=0.25), llm


def test_handbook_answer_cites_the_page():
    assistant, llm = make(
        [handbook(16, 0.59), website(COMPARE, 0.55), handbook(17, 0.47)],
        LLMAnswer(found=True, answer="The total fee is R 38,950.", excerpts=[1]),
    )
    result = assistant.ask("How much does the bootcamp cost?")

    assert result.answer == "The total fee is R 38,950."
    assert result.source == "Student Handbook - Page 16"
    question, sent = llm.calls[0]
    assert question == "How much does the bootcamp cost?"
    assert len(sent) == 3


def test_website_answer_cites_the_url():
    assistant, _ = make(
        [handbook(16, 0.41), website(TERMS, 0.52)],
        LLMAnswer(found=True, answer="Partial refunds are available within the first 30 days.", excerpts=[2]),
    )
    assert assistant.ask("What is the refund policy?").source == TERMS


def test_first_cited_excerpt_decides_the_source():
    chunks = [handbook(16), website(COMPARE)]
    web_first, _ = make(chunks, LLMAnswer(found=True, answer="...", excerpts=[2, 1]))
    handbook_first, _ = make(chunks, LLMAnswer(found=True, answer="...", excerpts=[1, 2]))

    assert web_first.ask("fees?").source == COMPARE
    assert handbook_first.ask("fees?").source == "Student Handbook - Page 16"


def test_answer_from_two_handbook_pages():
    assistant, _ = make(
        [handbook(16), website(COMPARE), handbook(17)],
        LLMAnswer(found=True, answer="...", excerpts=[3, 2, 1]),
    )
    assert assistant.ask("Tell me about financing").source == "Student Handbook - Pages 16, 17"


def test_off_topic_question_never_reaches_the_llm():
    assistant, llm = make([website(COMPARE, 0.12), handbook(6, 0.10)])
    result = assistant.ask("What is the capital of France?")

    assert result.answer == config.NOT_FOUND_ANSWER
    assert result.source is None
    assert llm.calls == []


def test_empty_retrieval_is_not_found():
    assistant, llm = make([])
    assert assistant.ask("anything").answer == config.NOT_FOUND_ANSWER
    assert llm.calls == []


def test_llm_saying_not_found_gives_the_standard_message():
    assistant, _ = make(
        [handbook(14, 0.47)],
        LLMAnswer(found=False, answer="The handbook only gives the MCQ weighting.", excerpts=[1]),
    )
    result = assistant.ask("What is the pass mark for the MCQs?")
    assert result.answer == "I could not find that information in the available knowledge base."
    assert result.source is None


def test_blank_answer_is_treated_as_not_found():
    assistant, _ = make([handbook(14)], LLMAnswer(found=True, answer="   ", excerpts=[1]))
    assert assistant.ask("q").answer == config.NOT_FOUND_ANSWER


def test_excerpt_numbers_that_were_not_sent_are_ignored():
    assistant, _ = make(
        [handbook(24, 0.33), website(TERMS, 0.30)],
        LLMAnswer(found=True, answer="A final mark of 70% and above.", excerpts=[7, 0, -1, 1]),
    )
    assert assistant.ask("What mark do I need to get placed?").source == "Student Handbook - Page 24"


def test_no_usable_citation_falls_back_to_best_chunk():
    assistant, _ = make([website(COMPARE, 0.71), handbook(11, 0.49)],
                        LLMAnswer(found=True, answer="From R1,950 a month.", excerpts=[]))
    assert assistant.ask("What is the cheapest bootcamp?").source == COMPARE


@pytest.mark.parametrize("cited, expected", [
    ([handbook(12)], "Student Handbook - Page 12"),
    ([handbook(17), handbook(16)], "Student Handbook - Pages 16, 17"),
    ([handbook(20), handbook(20)], "Student Handbook - Page 20"),
    ([handbook(16), website(COMPARE), handbook(17)], "Student Handbook - Pages 16, 17"),
    ([website(TERMS), handbook(16)], TERMS),
    ([website(TERMS), website(COMPARE)], TERMS),
])
def test_format_source(cited, expected):
    assert format_source(cited) == expected
