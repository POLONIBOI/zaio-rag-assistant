// Tests for the live version's logic, with a fake OpenAI (no network, no key).
//   node --test     (from the netlify/ folder)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ANSWER_SCHEMA, HttpError, MAX_QUESTION_LENGTH, NOT_FOUND_ANSWER, ask, buildPrompt, formatSource, openaiClient,
  readQuestion, search,
} from "./lib/rag.mjs";

const unit = (...xs) => { const n = Math.hypot(...xs); return xs.map((x) => x / n); };

const FEES = { source: "handbook", page: 16, section: "Fees", text: "Total Fees: R 38,950", vector: unit(1, 0, 0) };
const FEES_2 = { source: "handbook", page: 17, section: "Fees", text: "Should your Capitec loan fail...", vector: unit(0.9, 0.1, 0) };
const TERMS = { source: "website", url: "https://www.zaio.io/terms", title: "Terms and Conditions", section: "Refunds",
  text: "Partial refunds are available within the first 30 days.", vector: unit(0, 1, 0) };
const INDEX = [FEES, FEES_2, TERMS];

function fakeOpenAI({ vector = unit(1, 0.2, 0), reply = { found: true, answer: "R 38,950.", excerpts: [1] } } = {}) {
  const calls = { embed: [], answer: [] };
  return {
    calls,
    async embed(texts) { calls.embed.push(texts); return [vector]; },
    async answer(system, user, schema) { calls.answer.push({ system, user, schema }); return reply; },
  };
}

// --- request validation ----------------------------------------------------

test("a valid body gives the trimmed question", () => {
  assert.equal(readQuestion('{"question": "  When are live classes? \\n", "sessionId": "x"}'), "When are live classes?");
});

for (const [body, status, message] of [
  ['{"question": "fees?"', 400, "isn't valid JSON"],
  ['["fees?"]', 400, "must be a JSON object"],
  ['"fees?"', 400, "must be a JSON object"],
  ["null", 400, "must be a JSON object"],
  ["{}", 422, "The 'question' field is required"],
  ['{"question": 42}', 422, "must be a string"],
  ['{"question": "   "}', 422, "can't be empty"],
  [JSON.stringify({ question: "x".repeat(MAX_QUESTION_LENGTH + 1) }), 422, "can't be longer than"],
]) {
  test(`bad body ${body.slice(0, 30)} -> ${status}`, () => {
    assert.throws(() => readQuestion(body), (err) => err instanceof HttpError && err.status === status && err.message.includes(message));
  });
}

// --- retrieval ---------------------------------------------------------------

test("each source is searched separately, best first", () => {
  const hits = search(unit(1, 0.5, 0), INDEX, 1);
  // page 17 (0.94) beats page 16 (0.89) for this question; /terms is the best website chunk
  assert.deepEqual(hits.map((h) => h.page ?? h.url), [17, "https://www.zaio.io/terms"]);
  assert.ok(hits[0].similarity > hits[1].similarity);
});

// --- source format, same as app/assistant.py -----------------------------

test("formatSource", () => {
  assert.equal(formatSource([FEES]), "Student Handbook - Page 16");
  assert.equal(formatSource([FEES_2, TERMS, FEES]), "Student Handbook - Pages 16, 17");
  assert.equal(formatSource([TERMS, FEES]), "https://www.zaio.io/terms");
});

test("the prompt numbers every excerpt and names its source", () => {
  const prompt = buildPrompt("Refunds?", [FEES, TERMS]);
  assert.match(prompt, /<excerpt id="1" source="Student Handbook, page 16" section="Fees">\nTotal Fees: R 38,950\n<\/excerpt>/);
  assert.match(prompt, /<excerpt id="2" source="https:\/\/www.zaio.io\/terms" page_title="Terms and Conditions"/);
  assert.ok(prompt.endsWith("Question: Refunds?"));
});

// --- ask() -------------------------------------------------------------------

test("answer cites the excerpt the model used", async () => {
  const openai = fakeOpenAI({ reply: { found: true, answer: " Partial refunds within 30 days. ", excerpts: [3] } });
  // Ranked page 17, page 16, /terms: the model cited the third, not the top hit.
  const result = await ask("Refunds?", { index: INDEX, openai, minSimilarity: 0.1 });
  assert.deepEqual(result, { answer: "Partial refunds within 30 days.", source: "https://www.zaio.io/terms" });
  assert.deepEqual(openai.calls.answer[0].schema, ANSWER_SCHEMA);
});

test("off-topic question never reaches the chat model", async () => {
  const openai = fakeOpenAI({ vector: unit(0, 0, 1) });
  assert.deepEqual(await ask("Capital of France?", { index: INDEX, openai, minSimilarity: 0.2 }),
    { answer: NOT_FOUND_ANSWER, source: null });
  assert.equal(openai.calls.answer.length, 0);
});

test("model saying not found gives the standard sentence", async () => {
  const openai = fakeOpenAI({ reply: { found: false, answer: "Only the MCQ weighting is given.", excerpts: [1] } });
  assert.deepEqual(await ask("MCQ pass mark?", { index: INDEX, openai, minSimilarity: 0.1 }),
    { answer: NOT_FOUND_ANSWER, source: null });
});

test("excerpt numbers that weren't sent fall back to the best match", async () => {
  const openai = fakeOpenAI({ vector: unit(1, 0, 0), reply: { found: true, answer: "R 38,950.", excerpts: [0, 9] } });
  const result = await ask("Fees?", { index: INDEX, openai, minSimilarity: 0.1 });
  assert.equal(result.source, "Student Handbook - Page 16");
});

// --- the OpenAI client -------------------------------------------------------

test("missing key is a 503", () => {
  assert.throws(() => openaiClient(""), (err) => err.status === 503);
});

test("chat request asks for the JSON schema and parses the reply", async () => {
  const sent = [];
  const fetchImpl = async (url, init) => {
    sent.push({ url, body: JSON.parse(init.body) });
    return new Response(JSON.stringify({ choices: [{ message: { content: '{"found":true,"answer":"x","excerpts":[1]}' } }] }));
  };
  const reply = await openaiClient("sk-test", fetchImpl).answer("sys", "user", ANSWER_SCHEMA);
  assert.deepEqual(reply, { found: true, answer: "x", excerpts: [1] });
  assert.equal(sent[0].url, "https://api.openai.com/v1/chat/completions");
  assert.deepEqual(sent[0].body.response_format.json_schema.schema, ANSWER_SCHEMA);
  assert.equal(sent[0].body.response_format.json_schema.strict, true);
});

test("a refusal counts as not found", async () => {
  const fetchImpl = async () => new Response(JSON.stringify({ choices: [{ message: { refusal: "no", content: null } }] }));
  assert.equal((await openaiClient("sk-test", fetchImpl).answer("s", "u", ANSWER_SCHEMA)).found, false);
});

test("OpenAI errors become a 502 without leaking the key", async () => {
  const fetchImpl = async () => new Response("rate limited", { status: 429 });
  await assert.rejects(openaiClient("sk-secret", fetchImpl).embed(["q"]),
    (err) => err.status === 502 && err.message.includes("429") && !err.message.includes("sk-secret"));
});
