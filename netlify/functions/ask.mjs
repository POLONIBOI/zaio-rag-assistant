// POST /ask - same request and response as the Python API (app/api.py).
import index from "../data/index.json";
import { ask, HttpError, openaiClient, readQuestion } from "../lib/rag.mjs";

// Open to any origin, so n8n or another site can call it like the Python API.
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

export default async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  if (req.method !== "POST") return json(405, { error: "Method Not Allowed" });

  try {
    const question = readQuestion(await req.text());
    const result = await ask(question, { index, openai: openaiClient(process.env.OPENAI_API_KEY) });
    return json(200, result);
  } catch (err) {
    if (err instanceof HttpError && err.status === 502) {
      // The full reason goes to the function log; the caller gets a clean message.
      console.error(err.message);
      return json(502, { error: "The language model couldn't answer right now. Please try again in a moment." });
    }
    if (err instanceof HttpError) return json(err.status, { error: err.message });
    console.error(err);
    return json(500, { error: "Something went wrong on the server." });
  }
};

export const config = { path: "/ask" };
