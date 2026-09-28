import pytest

from app import config
from app.assistant import HandbookAssistant, format_source
from app.llm import LLMAnswer
from app.retriever import RetrievedChunk


def chunk(page, similarity, text="..."):
    return RetrievedChunk(text=text, page=page, section="Fees", similarity=similarity)


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
    return HandbookAssistant(FakeRetriever(chunks), llm, min_similarity=0.25), llm


def test_answer_uses_the_pages_the_llm_cited():
    assistant, llm = make(
        [chunk(16, 0.59), chunk(17, 0.47)],
        LLMAnswer(found=True, answer="The total fee is R 38,950.", pages=[16]),
    )
    result = assistant.ask("How much does the bootcamp cost?")

    assert result.answer == "The total fee is R 38,950."
    assert result.source == "Page 16"
    question, sent_chunks = llm.calls[0]
    assert question == "How much does the bootcamp cost?"
    assert [c.page for c in sent_chunks] == [16, 17]


def test_answer_from_two_pages():
    assistant, _ = make(
        [chunk(16, 0.5), chunk(17, 0.5)],
        LLMAnswer(found=True, answer="...", pages=[17, 16]),
    )
    assert assistant.ask("Tell me about financing").source == "Pages 16, 17"


def test_off_topic_question_never_reaches_the_llm():
    assistant, llm = make([chunk(6, 0.10), chunk(25, 0.08)])
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
        [chunk(14, 0.47)],
        LLMAnswer(found=False, answer="The handbook only gives the MCQ weighting.", pages=[14]),
    )
    result = assistant.ask("What is the pass mark for the MCQs?")
    assert result.answer == config.NOT_FOUND_ANSWER
    assert result.source is None


def test_blank_answer_is_treated_as_not_found():
    assistant, _ = make([chunk(14, 0.5)], LLMAnswer(found=True, answer="   ", pages=[14]))
    assert assistant.ask("q").answer == config.NOT_FOUND_ANSWER


def test_pages_that_were_not_retrieved_are_ignored():
    assistant, _ = make(
        [chunk(24, 0.33), chunk(21, 0.30)],
        LLMAnswer(found=True, answer="A final mark of 70% and above.", pages=[12, 24]),
    )
    assert assistant.ask("What mark do I need to get placed?").source == "Page 24"


def test_no_usable_pages_falls_back_to_best_chunk():
    assistant, _ = make([chunk(11, 0.71), chunk(12, 0.49)],
                        LLMAnswer(found=True, answer="Tuesdays and Thursdays.", pages=[]))
    assert assistant.ask("When are live classes?").source == "Page 11"


@pytest.mark.parametrize("pages, expected", [
    ([12], "Page 12"),
    ([17, 16], "Pages 16, 17"),
    ([20, 20], "Page 20"),
])
def test_format_source(pages, expected):
    assert format_source(pages) == expected
