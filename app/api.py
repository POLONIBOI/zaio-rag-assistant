"""Parts 3 and 5: the HTTP API.

    uvicorn app.api:app --reload

Every response is JSON, errors included, so an n8n HTTP Request node can
branch on the status code and always read a body. Errors look like:

    {"error": "'question' can't be empty."}
"""

import logging
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import config
from app.assistant import HandbookAssistant
from app.llm import LLMError, MissingAPIKeyError
from app.vector_store import IndexNotBuiltError

log = logging.getLogger("handbook.api")

app = FastAPI(
    title="Zaio Handbook Assistant",
    description="Answers student questions from the Full-Stack AI Engineer Bootcamp handbook.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=config.MAX_QUESTION_LENGTH,
        examples=["What is the attendance requirement?"],
    )


class AskResponse(BaseModel):
    answer: str
    source: str | None = Field(examples=["Page 12"])


class ErrorResponse(BaseModel):
    error: str


@lru_cache(maxsize=1)
def get_assistant() -> HandbookAssistant:
    # Built on the first request, not at import time, so the server still
    # starts (and /health can explain the problem) if the index is missing.
    from app.llm import ClaudeAnswerer
    from app.retriever import Retriever

    return HandbookAssistant(Retriever(), ClaudeAnswerer())


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ask(body: AskRequest):
    # Not a Depends(): FastAPI resolves dependencies before it reports body
    # errors, so a server without an API key would answer a malformed request
    # with 503 instead of telling the caller what's wrong with it.
    result = get_assistant().ask(body.question)
    return AskResponse(answer=result.answer, source=result.source)


@app.get("/health")
def health():
    from app.vector_store import get_collection

    try:
        chunks = get_collection().count()
    except IndexNotBuiltError as exc:
        return error(503, str(exc))
    return {"status": "ok", "chunks": chunks}


# --- error handling -------------------------------------------------------

def describe_validation_error(err: dict) -> tuple[int, str]:
    """Turn pydantic's error list entry into one sentence a person can act on."""
    kind, loc = err["type"], err["loc"]
    example = '{"question": "What are the fees?"}'

    if kind == "json_invalid":
        return 400, f"The request body isn't valid JSON. Expected something like {example}."
    if loc == ("body",):
        # No body, a non-JSON body (text/plain, form data) or JSON that isn't an object
        return 400, f"The request body must be a JSON object, like {example}."
    if loc[-1] == "question":
        messages = {
            "missing": f"The 'question' field is required, like {example}.",
            "string_type": "'question' must be a string.",
            "string_too_short": "'question' can't be empty.",
            "string_too_long": f"'question' can't be longer than {config.MAX_QUESTION_LENGTH} characters.",
        }
        if kind in messages:
            return 422, messages[kind]
    return 422, f"Invalid request: {err['msg']}"


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return error(*describe_validation_error(exc.errors()[0]))


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request: Request, exc: StarletteHTTPException):
    # 404 for unknown paths, 405 for e.g. GET /ask - same {"error": ...} shape.
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail},
                        headers=getattr(exc, "headers", None))


@app.exception_handler(IndexNotBuiltError)
async def on_missing_index(request: Request, exc: IndexNotBuiltError):
    return error(503, str(exc))


@app.exception_handler(MissingAPIKeyError)
async def on_missing_key(request: Request, exc: MissingAPIKeyError):
    return error(503, str(exc))


@app.exception_handler(LLMError)
async def on_llm_error(request: Request, exc: LLMError):
    # The full reason goes to the server log; the caller gets a clean message.
    log.warning("LLM call failed: %s", exc)
    return error(502, "The language model couldn't answer right now. Please try again in a moment.")
