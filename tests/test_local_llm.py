"""The local model's prompt, reply parsing and source attribution, with a fake model (nothing is downloaded)."""

import pytest

from app import config
from app.llm import LLMError
from app.local_llm import LocalAnswerer, attribute, build_messages, parse_reply
from app.retriever import RetrievedChunk

CLASSES = RetrievedChunk(text="Live classes are on Tuesdays 9 am - 11 am and Thursdays 6 pm - 8 pm.",
                         source="handbook", section="Live Classes", page=11, similarity=0.7)
REFUNDS = RetrievedChunk(text="Partial refunds are available within the first 30 days, subject to a 25% deduction.",
                         source="website", section="Terms", url="https://www.zaio.io/terms",
                         title="Terms and Conditions", similarity=0.5)
LAPTOP = RetrievedChunk(text="For laptop advice, email asif@zaio.io before orientation.", source="handbook",
                        section="Hardware", page=5, similarity=0.4)
CHUNKS = [CLASSES, REFUNDS, LAPTOP]


def test_question_comes_before_and_after_the_excerpts():
    system, user = build_messages("When are classes?", CHUNKS)
    content = user["content"]

    assert system["role"] == "system"
    assert content.startswith("Question: When are classes?")
    assert "Excerpt 1 (Student Handbook, page 11):\nLive classes are on" in content
    assert "Excerpt 2 (Zaio website, Terms and Conditions):\nPartial refunds" in content
    assert content.index("Excerpt 3") < content.rindex("Question: When are classes?") < content.index("NOT FOUND")


@pytest.mark.parametrize("answer, expected", [
    ("Classes are on Tuesdays and Thursdays.", [1]),
    ("You get a partial refund within 30 days, minus 25%.", [2]),
    ("Email asif@zaio.io.", [3]),
    ("Yes.", []),  # nothing distinctive to go on
])
def test_attribute_picks_the_excerpt_the_answer_came_from(answer, expected):
    assert attribute(answer, CHUNKS) == expected


def test_figures_outweigh_ordinary_words():
    # "refunds" and "available" match excerpt 1's wording here, but "30 days"
    # and "25%" only appear in excerpt 2.
    chunks = [RetrievedChunk(text="Refunds are available on request.", source="handbook", section="",
                             page=3, similarity=0.6), REFUNDS]
    assert attribute("Refunds are available within 30 days, minus 25%.", chunks) == [2]


def test_parse_answer():
    result = parse_reply("Classes are on Tuesdays and Thursdays.", CHUNKS)
    assert (result.found, result.answer, result.excerpts) == (True, "Classes are on Tuesdays and Thursdays.", [1])


def test_parse_drops_prompt_echoes_and_an_empty_think_block():
    reply = ("<think>\n\n</think>\n\n"
             "Based on the excerpts, partial refunds are available within 30 days (as stated in Excerpt 2).")
    result = parse_reply(reply, CHUNKS)
    assert result.answer == "Partial refunds are available within 30 days."
    assert result.excerpts == [2]


def test_no_dangling_comma_after_removing_an_excerpt_reference():
    # A real reply from the test run: "The Data Science bootcamp runs over 7 months,."
    result = parse_reply("The Data Science bootcamp runs over 7 months, as stated in Excerpt 1.", CHUNKS)
    assert result.answer == "The Data Science bootcamp runs over 7 months."


@pytest.mark.parametrize("reply", [
    "NOT FOUND",
    "Not found.",
    "  ",
    "The excerpts don't mention a dress code. Therefore: NOT FOUND.",
    "<think>\n\n</think>\n\nNOT FOUND",
])
def test_parse_not_found(reply):
    assert parse_reply(reply, CHUNKS).found is False


def test_answer_uses_the_fake_model():
    seen = []

    def generate(messages):
        seen.append(messages)
        return "For laptop advice, email asif@zaio.io."

    result = LocalAnswerer(generate=generate).answer("Who do I email about laptops?", CHUNKS)
    assert (result.found, result.excerpts) == (True, [3])
    assert len(seen) == 1


def test_only_the_top_excerpts_are_sent(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MAX_EXCERPTS", 1)
    seen = []
    LocalAnswerer(generate=lambda m: seen.append(m) or "NOT FOUND").answer("q", CHUNKS)
    prompt = seen[0][1]["content"]
    assert "Live classes are on Tuesdays" in prompt
    assert "Excerpt 2" not in prompt


def test_model_is_not_asked_when_no_excerpt_fits():
    # A negative rerank score means the reranker doesn't think the excerpt
    # answers the question; a small model would answer anyway.
    weak = [RetrievedChunk(text="MCQs make up 15% of the final grade.", source="handbook", section="Grades",
                           page=14, similarity=0.47, rerank_score=-1.3)]
    calls = []
    result = LocalAnswerer(generate=lambda m: calls.append(m) or "The pass mark is 15%.").answer("MCQ pass mark?", weak)
    assert result.found is False
    assert calls == []


def test_model_is_asked_when_an_excerpt_fits():
    strong = [RetrievedChunk(text="Partial refunds within 30 days.", source="website", section="",
                             url="https://www.zaio.io/terms", similarity=0.5, rerank_score=4.2)]
    assert LocalAnswerer(generate=lambda m: "Within 30 days.").answer("Refunds?", strong).found is True


def test_model_failure_becomes_llm_error():
    def broken(messages):
        raise RuntimeError("out of memory")

    with pytest.raises(LLMError, match="out of memory"):
        LocalAnswerer(generate=broken).answer("q", CHUNKS)
