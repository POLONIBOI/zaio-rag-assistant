import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.llm import ANSWER_SCHEMA, ClaudeAnswerer, LLMError, MissingAPIKeyError, build_prompt
from app.retriever import RetrievedChunk

CHUNKS = [
    RetrievedChunk(text="Total Fees: R 38,950", page=16, section="Fees", similarity=0.59),
    RetrievedChunk(text="Should your Capitec loan fail...", page=17, section="Fees", similarity=0.47),
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


def reply(found, answer, pages):
    return [text_block(json.dumps({"found": found, "answer": answer, "pages": pages}))]


def test_prompt_labels_every_excerpt_with_its_page():
    prompt = build_prompt("How much does it cost?", CHUNKS)
    assert '<excerpt page="16" section="Fees">\nTotal Fees: R 38,950\n</excerpt>' in prompt
    assert '<excerpt page="17"' in prompt
    assert prompt.endswith("Student question: How much does it cost?")


def test_parses_structured_answer():
    client = FakeClient(reply(True, "R 38,950.", [16]))
    result = ClaudeAnswerer(client, model="claude-opus-5").answer("Cost?", CHUNKS)

    assert (result.found, result.answer, result.pages) == (True, "R 38,950.", [16])


def test_request_asks_for_json_matching_the_schema():
    client = FakeClient(reply(True, "x", [16]))
    ClaudeAnswerer(client, model="claude-opus-5").answer("Cost?", CHUNKS)

    request = client.requests[0]
    assert request["model"] == "claude-opus-5"
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert "R 38,950" in request["messages"][0]["content"]


def test_reads_the_last_text_block_after_a_fallback():
    content = [
        text_block('{"found": tr'),  # partial output from the declined model
        SimpleNamespace(type="fallback"),
        *reply(True, "From the fallback model.", [17]),
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
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        ClaudeAnswerer(FakeClient(api_key=None))
