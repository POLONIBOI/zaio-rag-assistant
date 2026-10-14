// Netlify build step: embed data/chunks.json with OpenAI -> data/index.json.
//
//   node build-index.mjs        (run from the netlify/ folder; needs OPENAI_API_KEY)
//
// The index is built here rather than committed, so the API key only has to
// live in Netlify's environment variables. Embedding all 391 chunks is about
// 60,000 tokens, which costs a fraction of a cent per deploy.
import { readFile, writeFile } from "node:fs/promises";
import { EMBEDDING_MODEL, openaiClient } from "./lib/rag.mjs";

const here = new URL(".", import.meta.url);
const BATCH = 100;

// Locally, read the key from the project's .env if it isn't already set.
if (!process.env.OPENAI_API_KEY) {
  try {
    const env = await readFile(new URL("../.env", here), "utf8");
    const match = env.match(/^OPENAI_API_KEY=(.+)$/m);
    if (match) process.env.OPENAI_API_KEY = match[1].trim().replace(/^["']|["']$/g, "");
  } catch {}
}
if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is not set. Add it in Netlify (Site configuration > Environment variables) or to .env.");
  process.exit(1);
}

const chunks = JSON.parse(await readFile(new URL("data/chunks.json", here), "utf8"));
const openai = openaiClient(process.env.OPENAI_API_KEY);
const index = [];

for (let start = 0; start < chunks.length; start += BATCH) {
  const batch = chunks.slice(start, start + BATCH);
  const vectors = await openai.embed(batch.map((c) => c.embedding_text));
  batch.forEach((chunk, i) => {
    const { embedding_text, ...rest } = chunk;
    index.push({ ...rest, vector: vectors[i].map((x) => Math.round(x * 1e6) / 1e6) });
  });
}

await writeFile(new URL("data/index.json", here), JSON.stringify(index));
const count = (s) => index.filter((c) => c.source === s).length;
console.log(`Embedded ${index.length} chunks with ${EMBEDDING_MODEL} (${count("handbook")} handbook, ${count("website")} website)`);
