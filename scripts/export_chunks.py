"""Write the chunks the Python pipeline makes to netlify/data/chunks.json.

    python scripts/export_chunks.py

The live version on Netlify can't run PyTorch or ChromaDB, but it should
answer from exactly the same text. So the chunking stays here, and the
Netlify build (netlify/build-index.mjs) embeds this file with OpenAI.
Run it again after `python -m app.crawl` or a handbook change.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingest import load_all_chunks  # noqa: E402

OUT = ROOT / "netlify" / "data" / "chunks.json"


def main():
    chunks = load_all_chunks()
    rows = [
        {
            "id": c.id,
            "source": c.source,
            "section": c.section,
            "page": c.page,
            "url": c.url,
            "title": c.title,
            "text": c.text,
            "embedding_text": c.embedding_text,
        }
        for c in chunks
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {s: sum(r["source"] == s for r in rows) for s in ("handbook", "website")}
    print(f"Wrote {len(rows)} chunks to {OUT.relative_to(ROOT)} ({counts['handbook']} handbook, {counts['website']} website)")


if __name__ == "__main__":
    main()
