import os
import re
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
MD_FILE = "Shakti Userhand Book for Bot.md"
MODEL_PATH = r"D:\Machine Learning and LLMs\LLMs\all-MiniLM-L6-v2"
OUT_FAISS = "shakti.faiss"
OUT_META = "shakti_meta.json"

MAX_WORDS = 110
# ---------------------------------------


def load_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------- MARKER PARSER ----------------
def parse_marker_markdown(md):
    """
    Converts Marker markdown into clean semantic blocks
    Removes:
    - Page numbers
    - PAGE_BREAK
    - 'This page is intentionally left blank'
    """
    elements = []
    buffer = []
    in_table = False

    lines = md.split("\n")

    def flush():
        nonlocal buffer
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                elements.append(text)
            buffer = []

    for line in lines:
        clean = line.strip().lower()

        # 🚫 Kill all page junk
        if (
            "[page_break]" in clean
            or re.search(r"page\s+\d+\s+of\s+\d+", clean)
            or "intentionally left blank" in clean
        ):
            flush()
            continue

        # Headings
        if line.startswith("#"):
            flush()
            elements.append(line.strip())
            continue

        # Images / figures
        if line.startswith("![](") or re.search(r"\b(fig|figure)\b", line, re.I):
            flush()
            elements.append("[IMAGE] " + line.strip())
            continue

        # Tables
        if "|" in line:
            if not in_table:
                flush()
                in_table = True
            buffer.append(line)
            continue
        else:
            if in_table:
                flush()
                in_table = False

        # Bullet points
        if line.strip().startswith(("-", "•", "*")):
            flush()
            elements.append(line.strip())
            continue

        # Normal text
        if line.strip() == "":
            flush()
        else:
            buffer.append(line)

    flush()
    return elements


# ---------------- SMART CHUNKER ----------------
def chunk_elements(elements, max_words=220):
    chunks = []
    current = []
    current_words = 0

    def wc(x):
        return len(x.split())

    for el in elements:

        # Safety re-check
        if "intentionally left blank" in el.lower():
            continue

        # Tables, headings & images must be atomic
        if el.startswith(("[IMAGE]", "#")) or "|" in el:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_words = 0
            chunks.append(el)
            continue

        w = wc(el)

        if current_words + w > max_words:
            chunks.append("\n".join(current))
            current = [el]
            current_words = w
        else:
            current.append(el)
            current_words += w

    if current:
        chunks.append("\n".join(current))

    return chunks


# ---------------- MAIN PIPELINE ----------------
def main():
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n📄 Loading markdown...")
    md = load_md(MD_FILE)

    print("📐 Parsing Marker output...")
    elements = parse_marker_markdown(md)
    print("Semantic elements:", len(elements))

    print("✂ Chunking...")
    chunks = chunk_elements(elements, MAX_WORDS)
    print("Final chunks:", len(chunks))

    print("\n🚀 Loading embedding model...")
    model = SentenceTransformer(MODEL_PATH, device="cuda")

    print("🧠 Encoding...")
    embeddings = []
    for c in tqdm(chunks):
        emb = model.encode(c, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print("\n⚡ Building FAISS...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("💾 Writing FAISS index...")
    faiss.write_index(index, OUT_FAISS)

    print("💾 Writing metadata...")
    meta = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Military-grade RAG DB created")
    print("Index:", OUT_FAISS)
    print("Meta :", OUT_META)


if __name__ == "__main__":
    main()
