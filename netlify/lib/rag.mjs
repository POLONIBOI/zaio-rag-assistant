// The live version of POST /ask, for Netlify Functions.
//
// Same pipeline as the Python app (app/assistant.py), with the parts Netlify
// can't run swapped out: OpenAI embeddings instead of sentence-transformers,
// a JSON file searched in memory instead of ChromaDB, and gpt-4o-mini instead
// of Claude or the local model. The chunks themselves come from the Python
// code (scripts/export_chunks.py), so both versions answer from the same text.

export const NOT_FOUND_ANSWER = "I could not find that information in the available knowledge base.";
export const MAX_QUESTION_LENGTH = 500;
export const PER_SOURCE = 5;
export const EMBEDDING_MODEL = "text-embedding-3-small";
export const CHAT_MODEL = "gpt-4o-mini";
// Below this, the question isn't about Zaio at all and the chat model isn't
// called. From scripts/eval_live_index.py: "capital of France" scores 0.18 and
// "how do I cook pasta" 0.17, while the lowest answerable question scores 0.37.
// Near misses ("pass mark for the MCQs?" 0.48) are left to the chat model.
export const MIN_SIMILARITY = 0.25;

export const SYSTEM_PROMPT = `You answer questions from students and prospective students of Zaio, a South \
African tech bootcamp. You are given numbered excerpts from two sources: the \
student handbook of the Full-Stack AI Engineer Bootcamp (tagged with its page \
number) and Zaio's website (tagged with the page's URL).

Answer only from those excerpts. If they don't contain the answer, set "found" \
to false - don't fill the gap from general knowledge or from what bootcamps \
usually do, because a student may act on what you say. A related excerpt is \
not an answer: if someone asks for the MCQ pass mark and the handbook only \
gives the MCQ weighting, the answer is not there.

The handbook was written for one intake of one bootcamp, while the website \
covers all of Zaio's current courses. If the two disagree (fees, for example), \
give both and say which source says what.

Write the way a tutor would reply on Discord: a short, direct answer, usually \
one to three sentences. Keep figures, times, links and email addresses exactly \
as the excerpts write them. In "excerpts", list the numbers of the excerpts you \
actually used, most important first.`;

export const ANSWER_SCHEMA = {
  type: "object",
  properties: {
    found: { type: "boolean" },
    answer: { type: "string" },
    excerpts: { type: "array", items: { type: "integer" } },
  },
  required: ["found", "answer", "excerpts"],
  additionalProperties: false,
};

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// --- request validation: same rules and wording as app/api.py -------------

const EXAMPLE = '{"question": "What are the fees?"}';

export function readQuestion(rawBody) {
  let body;
  try {
    body = JSON.parse(rawBody);
  } catch {
    throw new HttpError(400, `The request body isn't valid JSON. Expected something like ${EXAMPLE}.`);
  }
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    throw new HttpError(400, `The request body must be a JSON object, like ${EXAMPLE}.`);
  }
  if (!("question" in body)) {
    throw new HttpError(422, `The 'question' field is required, like ${EXAMPLE}.`);
  }
  if (typeof body.question !== "string") throw new HttpError(422, "'question' must be a string.");
  const question = body.question.trim();
  if (!question) throw new HttpError(422, "'question' can't be empty.");
  if (question.length > MAX_QUESTION_LENGTH) {
    throw new HttpError(422, `'question' can't be longer than ${MAX_QUESTION_LENGTH} characters.`);
  }
  return question;
}

// --- retrieval ---------------------------------------------------------------

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// OpenAI embeddings come back with length 1, so the dot product is the cosine
// similarity. Each source is ranked separately and the best 5 of each are kept,
// for the same reason as in app/retriever.py: in one pooled list the website's
// 342 chunks crowd out the handbook's 49.
export function search(questionVector, index, perSource = PER_SOURCE) {
  const hits = [];
  for (const source of ["handbook", "website"]) {
    const ranked = index
      .filter((chunk) => chunk.source === source)
      .map((chunk) => ({ ...chunk, similarity: dot(questionVector, chunk.vector) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, perSource);
    hits.push(...ranked);
  }
  return hits.sort((a, b) => b.similarity - a.similarity);
}

// --- generation --------------------------------------------------------------

function describe(chunk) {
  if (chunk.source === "handbook") return `source="Student Handbook, page ${chunk.page}" section="${chunk.section}"`;
  return `source="${chunk.url}" page_title="${chunk.title}" section="${chunk.section}"`;
}

export function buildPrompt(question, chunks) {
  const excerpts = chunks
    .map((c, i) => `<excerpt id="${i + 1}" ${describe(c)}>\n${c.text}\n</excerpt>`)
    .join("\n\n");
  return `Excerpts:\n\n${excerpts}\n\nQuestion: ${question}`;
}

// Same format as app/assistant.py: the first excerpt the model cited decides
// the kind of source; handbook answers list every handbook page they used.
export function formatSource(cited) {
  const primary = cited[0];
  if (primary.source !== "handbook") return primary.url;
  const pages = [...new Set(cited.filter((c) => c.source === "handbook").map((c) => c.page))].sort((a, b) => a - b);
  return pages.length === 1 ? `Student Handbook - Page ${pages[0]}` : `Student Handbook - Pages ${pages.join(", ")}`;
}

export async function ask(question, { index, openai, minSimilarity = MIN_SIMILARITY }) {
  const [vector] = await openai.embed([question]);
  const chunks = search(vector, index);

  // Nothing in either source is even close - don't spend a chat call on it.
  if (!chunks.length || Math.max(...chunks.map((c) => c.similarity)) < minSimilarity) {
    return { answer: NOT_FOUND_ANSWER, source: null };
  }

  const reply = await openai.answer(SYSTEM_PROMPT, buildPrompt(question, chunks), ANSWER_SCHEMA);
  if (!reply.found || !reply.answer.trim()) return { answer: NOT_FOUND_ANSWER, source: null };

  // Only trust excerpt numbers we actually sent; otherwise use the best match.
  const cited = [...new Set(reply.excerpts)].filter((n) => n >= 1 && n <= chunks.length).map((n) => chunks[n - 1]);
  return { answer: reply.answer.trim(), source: formatSource(cited.length ? cited : [chunks[0]]) };
}

// --- the OpenAI calls --------------------------------------------------------

export function openaiClient(apiKey, fetchImpl = fetch) {
  if (!apiKey) throw new HttpError(503, "OPENAI_API_KEY is not set on the server.");

  async function call(path, payload) {
    let response;
    try {
      response = await fetchImpl(`https://api.openai.com/v1/${path}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      throw new HttpError(502, `OpenAI request failed: ${err.message}`);
    }
    if (!response.ok) {
      throw new HttpError(502, `OpenAI ${path} returned ${response.status}: ${(await response.text()).slice(0, 300)}`);
    }
    return response.json();
  }

  return {
    async embed(texts) {
      const data = await call("embeddings", { model: EMBEDDING_MODEL, input: texts });
      return data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
    },
    async answer(system, user, schema) {
      const data = await call("chat/completions", {
        model: CHAT_MODEL,
        temperature: 0,
        max_tokens: 500,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        response_format: { type: "json_schema", json_schema: { name: "answer", strict: true, schema } },
      });
      const message = data.choices?.[0]?.message;
      if (message?.refusal) return { found: false, answer: "", excerpts: [] };
      try {
        return JSON.parse(message?.content ?? "");
      } catch {
        throw new HttpError(502, "Unexpected response from OpenAI.");
      }
    },
  };
}
