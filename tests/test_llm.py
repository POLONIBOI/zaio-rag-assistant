import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app import llm
from app.llm import ANSWER_SCHEMA, ClaudeAnswerer, LLMError, MissingAPIKeyError, build_prompt, make_answerer
from app.retriever import RetrievedChunk

CHUNKS = [
    RetrievedChunk(text="Total Fees: R 38,950", source="handbook", section="Fees", page=16, similarity=0.59),
    RetrievedChunk(text="Full refunds are available if cancellation happens before the start date.",
                   source="website", section="Zaio Terms and Conditions", url="https://www.zaio.io/terms",
                   title="Terms and Conditions", similarity=0.52),
]


def text_block(text):
    return SimpleNamespace(type="text", text=text)


class FakeClient:
    """Stands in for anthropic.Anthropic so these tests never hit the network."""

    def __init__(self, content=None, stop_reason="end_turn", raises=None, api_key="test-key"):
        self.api_key = api_key
        self.auth_token = None
        self.requests = []
        self._response = SimpleNamespace(content=content or [], stop_reason=stop_reason)
        self._raises = raises
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self._raises:
            raise self._raises
        return self._response


def reply(found, answer, excerpts):
    return [text_block(json.dumps({"found": found, "answer": answer, "excerpts": excerpts}))]


def test_prompt_numbers_every_excerpt_and_names_its_source():
    prompt = build_prompt("Can I get a refund?", CHUNKS)
    assert ('<excerpt id="1" source="Student Handbook, page 16" section="Fees">\nTotal Fees: R 38,950\n</excerpt>'
            in prompt)
    assert '<excerpt id="2" source="https://www.zaio.io/terms" page_title="Terms and Conditions"' in prompt
    assert prompt.endswith("Question: Can I get a refund?")


def test_parses_structured_answer():
    client = FakeClient(reply(True, "Full refunds before the start date.", [2]))
    result = ClaudeAnswerer(client, model="claude-opus-5").answer("Refunds?", CHUNKS)

    assert (result.found, result.answer, result.excerpts) == (True, "Full refunds before the start date.", [2])


def test_request_asks_for_json_matching_the_schema():
    client = FakeClient(reply(True, "x", [1]))
    ClaudeAnswerer(client, model="claude-opus-5").answer("Cost?", CHUNKS)

    request = client.requests[0]
    assert request["model"] == "claude-opus-5"
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert "R 38,950" in request["messages"][0]["content"]
    assert "https://www.zaio.io/terms" in request["messages"][0]["content"]


def test_reads_the_last_text_block_after_a_fallback():
    content = [
        text_block('{"found": tr'),  # partial output from the declined model
        SimpleNamespace(type="fallback"),
        *reply(True, "From the fallback model.", [1]),
    ]
    result = ClaudeAnswerer(FakeClient(content)).answer("q", CHUNKS)
    assert result.answer == "From the fallback model."


def test_refusal_is_treated_as_not_found():
    result = ClaudeAnswerer(FakeClient(stop_reason="refusal")).answer("q", CHUNKS)
    assert result.found is False


def test_malformed_output_raises():
    with pytest.raises(LLMError):
        ClaudeAnswerer(FakeClient([text_block("not json")], stop_reason="max_tokens")).answer("q", CHUNKS)


def test_api_failure_raises_llm_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(raises=anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError):
        ClaudeAnswerer(client).answer("q", CHUNKS)


def test_missing_api_key_is_reported_up_front():
    with pytest.raises(MissingAPIKeyError, match="LLM_PROVIDER=local"):
        ClaudeAnswerer(FakeClient(api_key=None))


# --- choosing the LLM -------------------------------------------------------

class FakeLocal:
    name = "local"


@pytest.fixture
def no_model_download(monkeypatch):
    import app.local_llm

    monkeypatch.setattr(app.local_llm, "LocalAnswerer", FakeLocal)


def test_auto_uses_the_local_model_without_a_key(monkeypatch, no_model_download):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_answerer("auto").name == "local"


def test_auto_uses_claude_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm, "ClaudeAnswerer", lambda: SimpleNamespace(name="claude"))
    assert make_answerer("auto").name == "claude"


def test_local_can_be_forced_even_with_a_key(monkeypatch, no_model_download):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert make_answerer("local").name == "local"


def test_unknown_provider_is_an_error():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        make_answerer("gpt")
