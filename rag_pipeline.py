import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------- PATHS ----------------
LLM_PATH = r"D:/Machine Learning and LLMs/LLMs/Mistral-7B-Instruct-v0.2"
EMBED_PATH = r"D:/Machine Learning and LLMs/LLMs/all-MiniLM-L6-v2"

FAISS_INDEX = r"shakti.faiss"
META_JSON = r"shakti_meta.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
print("Loading embedder...")
embedder = SentenceTransformer(EMBED_PATH, device=device)

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300
)

# ---------------- LOAD FAISS ----------------
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX)

with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

# normalize metadata
if isinstance(meta[0], dict):
    chunks = [m["text"] for m in meta]
else:
    chunks = meta

print(f"Loaded {len(chunks)} chunks")

# ---------------- SEARCH ----------------
def retrieve(query, k=25):
    """
    Retrieve top-k chunks from FAISS (k=25 for maximum coverage)
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), k)
    results = [chunks[idx] for idx in I[0] if idx != -1]
    return results

# ---------------- CONTEXT CLEANING ----------------
def clean_context(context: str) -> str:
    """
    Keep descriptive sentences, remove purely structural lines.
    Allows sentences with chapter/figure/table references if they contain actual description.
    """
    lines = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        # remove purely structural lines
        if lower.startswith(("table of contents", "index")):
            continue
        if lower.startswith(("figure", "table")) and len(line.split()) < 6:
            continue
        # keep descriptive sentences
        lines.append(line)
    return "\n".join(lines)

# ---------------- ANSWER GENERATION ----------------
def generate_answer(context: str, question: str) -> str:
    context = clean_context(context)

    prompt = f"""[INST]
You are a technical assistant.

Answer the user's question **directly and factually** using **only the provided context**.

Rules:
- Use descriptive sentences from the context.
- Do NOT invent information.
- Do NOT include chapter/figure/table references literally.
- If the context does not contain enough information, say exactly:
  "I don't have information on that."
- Keep answers concise, clear, and human-readable.

### CONTEXT:
{context}

### QUESTION:
{question}

### ANSWER:
[/INST]"""

    response = llm_pipeline(
        prompt,
        do_sample=False,  # deterministic, greedy
        max_new_tokens=300,
        eos_token_id=tokenizer.eos_token_id
    )

    answer = response[0].get("generated_text", "").strip()
    answer = answer.replace("[INST]", "").replace("[/INST]", "").strip()
    if "### ANSWER:" in answer:
        answer = answer.split("### ANSWER:")[-1].strip()

    # Safety filter
    if not answer or len(answer.split()) < 5:
        return "I don't have information on that."

    return answer

# ---------------- RAG PIPELINE ----------------
def ask(question: str) -> str:
    docs = retrieve(question, k=25)
    print("DEBUG: Retrieved Chunks:\n", docs)  # debug to check FAISS retrieval

    if not docs:
        return "I don't have information on that."

    context = "\n\n".join(docs)
    return generate_answer(context, question)

# ---------------- CHAT ----------------
print("\nRAG READY (Direct, Context-Grounded, Human-Readable Answers)\n")

while True:
    q = input(">>> ")
    if q.lower() in ["exit", "quit"]:
        break

    print("\n", ask(q), "\n")
