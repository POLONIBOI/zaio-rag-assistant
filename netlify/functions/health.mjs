// GET /health - what's loaded, like the Python API's /health.
import index from "../data/index.json";
import { CHAT_MODEL, EMBEDDING_MODEL } from "../lib/rag.mjs";

export default async () => {
  const chunks = { handbook: 0, website: 0 };
  for (const chunk of index) chunks[chunk.source] += 1;
  return Response.json(
    { status: "ok", chunks, llm: `openai: ${CHAT_MODEL}`, embeddings: `openai: ${EMBEDDING_MODEL}` },
    { headers: { "Access-Control-Allow-Origin": "*" } },
  );
};

export const config = { path: "/health" };
