"""
RAG pipeline over the NASA Systems Engineering Handbook.

Pipeline stages:
  1. Extract text from the PDF
  2. Split into overlapping chunks
  3. Embed each chunk locally (sentence-transformers, no API key needed)
  4. On a query: embed it, retrieve the most similar chunks (cosine similarity, plain numpy)
  5. Feed retrieved chunks + question to Claude, answer grounded in that context only

Setup:
  pip install pypdf sentence-transformers numpy anthropic
  export ANTHROPIC_API_KEY=your_key_here

Usage:
  python rag_pipeline.py --pdf nasa_systems_engineering_handbook.pdf --build
  python rag_pipeline.py --ask "What is the difference between verification and validation?"
"""

import argparse
import json
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs locally, no API key


def extract_text(pdf_path: str) -> str:
    """Pull all text out of the PDF, page by page."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word chunks. Overlap matters: it stops an
    answer-relevant sentence from being sliced in half at a chunk boundary.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def build_index(pdf_path: str):
    """Stage 1-3: extract, chunk, embed, save to disk."""
    print(f"Extracting text from {pdf_path}...")
    text = extract_text(pdf_path)
    print(f"Extracted {len(text):,} characters")

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks (this is the slow step, runs once)...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)

    print(f"Saved {len(chunks)} chunk embeddings to {EMBEDDINGS_FILE}")
    print("Index build complete. You can now run --ask")


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Stage 4: embed the query, find the most similar chunks via cosine similarity."""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError("No index found. Run with --build first.")

    embeddings = np.load(EMBEDDINGS_FILE)
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    query_vec = model.encode([query])[0]

    # cosine similarity: normalize both sides, then dot product
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = chunk_norms @ query_norm

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def answer(query: str, top_k: int = 5) -> str:
    """Stage 4-5: retrieve context, then ask Claude to answer grounded in it only."""
    context_chunks = retrieve(query, top_k=top_k)
    context = "\n\n---\n\n".join(context_chunks)

    client = Anthropic()  # reads ANTHROPIC_API_KEY from environment
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so directly rather than guessing.

Context:
{context}

Question: {query}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to the PDF to index")
    parser.add_argument("--build", action="store_true", help="Build the index from --pdf")
    parser.add_argument("--ask", help="Ask a question against the already-built index")
    parser.add_argument("--top_k", type=int, default=5, help="How many chunks to retrieve")
    args = parser.parse_args()

    if args.build:
        if not args.pdf:
            raise SystemExit("--build requires --pdf <path>")
        build_index(args.pdf)
    elif args.ask:
        print(answer(args.ask, top_k=args.top_k))
    else:
        print("Use --build --pdf <path> first, then --ask '<question>'")
