import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import markdown
from bs4 import BeautifulSoup

# ---------------- CONFIG ----------------
MD_FILE = "Shakti Userhand Book for Bot.md"
MODEL_PATH = r"D:\Machine Learning and LLMs\LLMs\all-MiniLM-L6-v2"
OUT_FAISS = "doc.index"
OUT_META = "doc_meta.json"
CHUNK_SIZE = 500
OVERLAP = 100
# ---------------------------------------


def load_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def markdown_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n")

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n📄 Loading markdown...")
    md = load_markdown(MD_FILE)

    print("🧹 Converting to clean text...")
    text = markdown_to_text(md)

    print("✂ Chunking...")
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print(f"Total chunks: {len(chunks)}")

    print("\n🚀 Loading embedding model from disk...")
    model = SentenceTransformer(MODEL_PATH, device="cuda")

    print("🧠 Encoding chunks on GPU...")
    embeddings = []

    for chunk in tqdm(chunks):
        emb = model.encode(chunk, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    print("\n⚡ Building FAISS index (CPU)...")
    index = faiss.IndexFlatIP(dim)

    print("📥 Adding vectors...")
    index.add(embeddings)

    print("💾 Saving FAISS index...")
    faiss.write_index(index, OUT_FAISS)

    print("💾 Saving chunk metadata...")
    meta = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Vector DB ready")
    print("FAISS:", OUT_FAISS)
    print("META :", OUT_META)


if __name__ == "__main__":
    main()
