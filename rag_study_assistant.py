import argparse
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from openai import OpenAI


STUDY_DIR = Path(os.environ.get("STUDY_DIR", os.path.join(tempfile.gettempdir(), "study_materials")))
INDEX_PATH = Path(os.environ.get("INDEX_PATH", os.path.join(tempfile.gettempdir(), "study_index.pkl")))
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None


def load_study_files() -> List[Path]:
    if not STUDY_DIR.exists():
        raise FileNotFoundError(
            f"Study materials folder '{STUDY_DIR}' does not exist. "
            f"Create it next to this script and put your PDFs or .txt files inside."
        )

    files: List[Path] = []
    for ext in ("*.pdf", "*.txt"):
        files.extend(STUDY_DIR.glob(ext))

    if not files:
        raise FileNotFoundError(
            f"No study files found in '{STUDY_DIR}'. "
            f"Add some .pdf or .txt files and run ingest again."
        )
    return files


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 128) -> List[str]:
    # Very rough token approximation using words
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max_tokens - overlap
    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def build_chunks() -> List[Chunk]:
    files = load_study_files()
    chunks: List[Chunk] = []

    for f in files:
        if f.suffix.lower() == ".txt":
            text = read_txt(f)
            for ch in chunk_text(text):
                if ch.strip():
                    chunks.append(Chunk(text=ch, source=f.name, page=None))
        elif f.suffix.lower() == ".pdf":
            for page_num, page_text in read_pdf(f):
                for ch in chunk_text(page_text):
                    if ch.strip():
                        chunks.append(Chunk(text=ch, source=f.name, page=page_num))
    if not chunks:
        raise RuntimeError("No usable text could be extracted from the study materials.")

    return chunks


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def ingest() -> None:
    print(f"Loading study materials from '{STUDY_DIR}'...")
    chunks = build_chunks()
    print(f"Found {len(chunks)} text chunks. Computing embeddings...")

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    index_data = {
        "embeddings": embeddings.astype(np.float32),
        "chunks": chunks,
        "embed_model": EMBED_MODEL_NAME,
    }

    with INDEX_PATH.open("wb") as f:
        pickle.dump(index_data, f)

    print(f"Index saved to '{INDEX_PATH}'. You can now run in chat mode.")


def load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file '{INDEX_PATH}' not found. Run with 'ingest' first."
        )
    with INDEX_PATH.open("rb") as f:
        index_data = pickle.load(f)

    if index_data.get("embed_model") != EMBED_MODEL_NAME:
        print(
            "Warning: index was built with a different embedding model. "
            "Consider re-running ingest."
        )
    return index_data


def retrieve(query: str, k: int = 5, allowed_sources: Optional[List[str]] = None) -> List[Chunk]:
    index = load_index()
    embeddings: np.ndarray = index["embeddings"]
    chunks: List[Chunk] = index["chunks"]

    embedder = get_embedder()
    query_emb = embedder.encode([query], convert_to_numpy=True)

    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sims)[::-1]

    top_chunks: List[Chunk] = []
    for idx in top_indices:
        if sims[idx] <= 0:
            continue
            
        chunk = chunks[int(idx)]
        if allowed_sources is not None and chunk.source not in allowed_sources:
            continue
            
        top_chunks.append(chunk)
        if len(top_chunks) >= k:
            break
            
    return top_chunks


def build_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return ""

    parts: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        location = f"{ch.source}"
        if ch.page is not None:
            location += f", page {ch.page}"
        parts.append(f"[{i}] From {location}:\n{ch.text}")
    return "\n\n".join(parts)


def get_groq_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it to your Groq API key before using chat mode."
        )
    # Groq provides an OpenAI-compatible API.
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def answer_question(question: str, chunks: List[Chunk]) -> str:
    context = build_context(chunks)
    if not context:
        return "the study material not found"

    client = get_groq_client()

    system_prompt = (
        "You are a helpful and intelligent study assistant. You must answer questions "
        "using ONLY the provided study material excerpts. "
        "Structure your answers to be highly attractive, readable, and well-formatted, using Markdown features like headings, bullet points, italics, or bold text where appropriate (similar to ChatGPT). "
        "Do NOT mention the names of the source files, filenames, or page numbers in your response. Just provide the direct, beautifully formatted answer. "
        "If the answer to the question is not clearly supported by the material, explicitly state EXACTLY: "
        "'the study material not found'. "
        "Do not use outside knowledge."
    )

    user_prompt = (
        f"Study material excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question beautifully and comprehensively using only the excerpts above. Do NOT mention which study material or page was used. If it cannot be answered from the excerpts, output EXACTLY 'the study material not found'."
    )

    completion = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()


def chat_loop() -> None:
    print("Study Material Assistant (RAG)")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            chunks = retrieve(question, k=5)
            answer = answer_question(question, chunks)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print("\nAssistant:\n")
        print(answer)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG-based Study Material Assistant"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Index the study materials")
    subparsers.add_parser("chat", help="Ask questions about the study materials")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest()
    elif args.command == "chat":
        chat_loop()


if __name__ == "__main__":
    main()

