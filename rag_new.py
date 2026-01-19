import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FAISS_INDEX_PATH = "shakti.faiss"
DOCUMENTS_PATH = "output.jsonl"          # JSONL
METADATA_PATH = "shakti_meta.json"      # JSONL

EMBEDDING_MODEL = "D:/Machine Learning and LLMs/LLMs/all-MiniLM-L6-v2"
LLM_MODEL = "D:/Machine Learning and LLMs/LLMs/Mistral-7B-Instruct-v0.2"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.75
MAX_NEW_TOKENS = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# LOAD FAISS INDEX (CPU — REQUIRED ON WINDOWS)
# -------------------------------------------------
print("Loading FAISS index (CPU)...")
index = faiss.read_index(FAISS_INDEX_PATH)

# -------------------------------------------------
# JSONL LOADER (FIX FOR Extra data ERROR)
# -------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# -------------------------------------------------
# LOAD DOCUMENTS & METADATA (JSONL)
# -------------------------------------------------
documents = load_jsonl(DOCUMENTS_PATH)
metadata = load_jsonl(METADATA_PATH)

assert len(documents) == len(metadata), "Documents and metadata mismatch"

# -------------------------------------------------
# LOAD EMBEDDING MODEL (CUDA)
# -------------------------------------------------
print("Loading embedding model on", DEVICE)
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

def embed_query(text: str):
    vec = embedder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vec.astype("float32")

# -------------------------------------------------
# RETRIEVAL (STRICT)
# -------------------------------------------------
def retrieve(query: str):
    query_vector = embed_query(query)

    distances, indices = index.search(query_vector, TOP_K)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        similarity = 1 - dist  # cosine similarity

        if similarity >= SIMILARITY_THRESHOLD:
            results.append({
                "text": documents[idx]["text"],   # FIXED
                "metadata": metadata[idx],
                "score": similarity
            })

    return results

# -------------------------------------------------
# LOAD MISTRAL LLM (CUDA)
# -------------------------------------------------
print("Loading Mistral LLM on CUDA...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.1
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# STRICT ANTI-HALLUCINATION PROMPT
# -------------------------------------------------
def build_prompt(context, query):
    context_block = "\n\n".join(
        f"[Source: {c['metadata']['source']} | Page: {c['metadata']['page']}]\n{c['text']}"
        for c in context
    )

    return f"""
You are a document-based QA system.

MANDATORY RULES:
- Use ONLY the provided context
- If the answer is missing, say:
  "I don't know based on the provided documents."
- Do NOT use prior knowledge
- Do NOT infer or guess
- Cite sources exactly

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:
"""

# -------------------------------------------------
# RAG PIPELINE
# -------------------------------------------------
def answer(query: str):
    retrieved = retrieve(query)

    if not retrieved:
        return "I don't know based on the provided documents."

    prompt = build_prompt(retrieved, query)
    return generate(prompt)

# -------------------------------------------------
# INTERACTIVE MODE
# -------------------------------------------------
if __name__ == "__main__":
    print("\nCUDA RAG system ready (FAISS CPU + CUDA LLM). Type 'exit' to quit.\n")
    while True:
        q = input("Question: ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", answer(q))
        print("\n" + "=" * 70 + "\n")
