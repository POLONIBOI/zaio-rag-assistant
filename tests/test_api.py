import pytest
from fastapi.testclient import TestClient

from app import api, config, vector_store
from app.api import app
from app.assistant import NOT_FOUND, Answer
from app.llm import LLMError, MissingAPIKeyError
from app.vector_store import IndexNotBuiltError


class FakeAssistant:
    def __init__(self, result=None, raises=None):
        self.result = result or Answer(answer="The total fee is R 38,950.", source="Student Handbook - Page 16")
        self.raises = raises
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        if self.raises:
            raise self.raises
        return self.result


@pytest.fixture
def assistant(monkeypatch):
    fake = FakeAssistant()
    monkeypatch.setattr(api, "get_assistant", lambda: fake)
    return fake


@pytest.fixture
def client(assistant):
    return TestClient(app)


# --- happy path -----------------------------------------------------------

def test_ask_returns_answer_and_source(client, assistant):
    response = client.post("/ask", json={"question": "How much does the bootcamp cost?"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"answer": "The total fee is R 38,950.", "source": "Student Handbook - Page 16"}
    assert assistant.questions == ["How much does the bootcamp cost?"]


def test_website_answer_has_the_url_as_source(client, assistant):
    assistant.result = Answer(answer="ZAIO offers six bootcamps.", source="https://www.zaio.io/bootcamps")
    response = client.post("/ask", json={"question": "What courses does ZAIO offer?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "ZAIO offers six bootcamps.", "source": "https://www.zaio.io/bootcamps"}


def test_not_found_has_null_source(client, assistant):
    assistant.result = NOT_FOUND
    response = client.post("/ask", json={"question": "Is there a dress code?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I could not find that information in the available knowledge base.",
        "source": None,
    }


def test_question_is_trimmed(client, assistant):
    client.post("/ask", json={"question": "   When are live classes?  \n"})
    assert assistant.questions == ["When are live classes?"]


def test_extra_fields_are_ignored(client):
    # n8n often sends along extra fields from earlier nodes.
    response = client.post("/ask", json={"question": "Fees?", "sessionId": "abc", "user": "n8n"})
    assert response.status_code == 200


# --- invalid requests -----------------------------------------------------

@pytest.mark.parametrize("body, status, message", [
    ({}, 422, "The 'question' field is required"),
    ({"q": "Fees?"}, 422, "The 'question' field is required"),
    ({"question": ""}, 422, "can't be empty"),
    ({"question": "    "}, 422, "can't be empty"),
    ({"question": 42}, 422, "must be a string"),
    ({"question": None}, 422, "must be a string"),
    ({"question": "x" * (config.MAX_QUESTION_LENGTH + 1)}, 422, "can't be longer than"),
    (["What are the fees?"], 400, "must be a JSON object"),
    ("What are the fees?", 400, "must be a JSON object"),
])
def test_invalid_body(client, assistant, body, status, message):
    response = client.post("/ask", json=body)

    assert response.status_code == status
    assert message in response.json()["error"]
    assert assistant.questions == []


def test_malformed_json(client):
    response = client.post("/ask", content='{"question": "fees?"', headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "isn't valid JSON" in response.json()["error"]


def test_empty_body(client):
    response = client.post("/ask")
    assert response.status_code == 400
    assert "must be a JSON object" in response.json()["error"]


@pytest.mark.parametrize("kwargs", [
    {"content": "What are the fees?", "headers": {"Content-Type": "text/plain"}},
    {"data": {"question": "What are the fees?"}},  # form-encoded, e.g. an HTML form
])
def test_non_json_body(client, assistant, kwargs):
    response = client.post("/ask", **kwargs)
    assert response.status_code in (400, 422)
    assert "error" in response.json()
    assert assistant.questions == []


def test_bad_request_is_reported_even_when_server_is_misconfigured(monkeypatch):
    # Found by hand: with no API key, a request missing "question" came back
    # 503 instead of 422, because the assistant was built before validation.
    def broken():
        raise MissingAPIKeyError()

    monkeypatch.setattr(api, "get_assistant", broken)
    response = TestClient(app).post("/ask", json={"q": "Fees?"})
    assert response.status_code == 422


def test_wrong_method_is_json(client):
    response = client.get("/ask")
    assert response.status_code == 405
    assert response.json() == {"error": "Method Not Allowed"}


def test_unknown_route_is_json(client):
    response = client.post("/question", json={"question": "Fees?"})
    assert response.status_code == 404
    assert response.json() == {"error": "Not Found"}


# --- the server's own problems --------------------------------------------

def test_llm_failure_is_a_502(client, assistant):
    assistant.raises = LLMError("Claude API request failed: connection reset")
    response = client.post("/ask", json={"question": "Fees?"})

    assert response.status_code == 502
    # internal details stay in the server log
    assert response.json() == {"error": "The language model couldn't answer right now. Please try again in a moment."}


def test_missing_api_key_is_a_503(client, assistant):
    assistant.raises = MissingAPIKeyError()
    response = client.post("/ask", json={"question": "Fees?"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["error"]


def test_missing_index_is_a_503(client, assistant):
    assistant.raises = IndexNotBuiltError()
    response = client.post("/ask", json={"question": "Fees?"})

    assert response.status_code == 503
    assert "python -m app.ingest" in response.json()["error"]


# --- /health --------------------------------------------------------------

class FakeCollection:
    def get(self, include):
        return {"metadatas": [{"source": "handbook"}] * 2 + [{"source": "website"}] * 3}


def test_health_counts_chunks_per_source(monkeypatch):
    monkeypatch.setattr(vector_store, "get_collection", lambda: FakeCollection())
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["chunks"] == {"handbook": 2, "website": 3}


def test_health_explains_a_missing_index(monkeypatch):
    def missing():
        raise IndexNotBuiltError()

    monkeypatch.setattr(vector_store, "get_collection", missing)
    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert "python -m app.ingest" in response.json()["error"]
