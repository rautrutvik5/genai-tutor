# ===== MEMBER A owns this file (chunking, embeddings, retrieval) =====
import os, json, re
import numpy as np, requests
from pypdf import PdfReader
from dotenv import load_dotenv
from prompts import SYSTEM_RAG, SYSTEM_PLAIN, FEWSHOT

load_dotenv()
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM = os.getenv("LLM_MODEL", "llama3.2")
EMB = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0))
# Page numbers = numbers PRINTED in the handbook footer (cover = 0). Cover/TOC/overview pages (0-3) mention every topic and practice-question/reference pages (32-34) have questions but no answers -> retrieval noise
SKIP_PAGES = {int(x) for x in os.getenv("SKIP_PAGES", "0,1,2,3,32,33,34").split(",") if x.strip()}
INDEX_DIR = "index"

def clean(text):
    """Remove the repeating header 'Generative AI - Handbook' and footer 'Page N'."""
    text = re.sub(r"Generative AI\s*-\s*Handbook", " ", text)
    text = re.sub(r"Page\s*\d+\s*$", " ", text.strip())
    return text

def read_pdf(path):
    return [(i, clean(p.extract_text() or "")) for i, p in enumerate(PdfReader(path).pages) if i not in SKIP_PAGES]

def chunk_pages(pages, size=None, overlap=None):
    """Fixed-size character chunks with overlap, kept per page so we can cite page numbers."""
    size = size or CHUNK_SIZE
    overlap = OVERLAP if overlap is None else overlap
    chunks = []
    for page, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append({"id": len(chunks), "page": page, "text": piece})
            start += size - overlap
    return chunks

def embed(text, kind="document"):
    """kind = 'document' (chunks) or 'query' (questions). nomic-embed-text works best with these prefixes."""
    if "nomic" in EMB:
        text = f"search_{kind}: {text}"
    r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": EMB, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]

def build_index(pdf_path):
    chunks = chunk_pages(read_pdf(pdf_path))
    vecs = np.array([embed(c["text"]) for c in chunks], dtype="float32")
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(f"{INDEX_DIR}/vecs.npy", vecs)
    json.dump(chunks, open(f"{INDEX_DIR}/chunks.json", "w"))
    return len(chunks)

def load_index():
    return np.load(f"{INDEX_DIR}/vecs.npy"), json.load(open(f"{INDEX_DIR}/chunks.json"))

def retrieve(query, k=TOP_K):
    vecs, chunks = load_index()
    q = np.array(embed(query, "query"), dtype="float32")
    sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(-sims)[:k]
    return [{**chunks[i], "score": float(sims[i])} for i in top]

def chat(messages):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM, "messages": messages, "stream": False,
        "options": {"temperature": 0.2}}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]

def answer(question, use_rag=True, k=TOP_K):
    if not use_rag:
        msgs = [{"role": "system", "content": SYSTEM_PLAIN}, {"role": "user", "content": question}]
        return {"answer": chat(msgs), "chunks": []}
    chunks = retrieve(question, k)
    # Guard: if even the best chunk is not similar enough, refuse without asking the LLM
    if MIN_SCORE and chunks[0]["score"] < MIN_SCORE:
        return {"answer": "Not found in the notes.", "chunks": chunks}
    ctx = "\n".join(f"[p.{c['page']}] {c['text']}" for c in chunks)
    msgs = [{"role": "system", "content": SYSTEM_RAG}, *FEWSHOT,
            {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"}]
    return {"answer": chat(msgs), "chunks": chunks}

if __name__ == "__main__":
    import sys
    print("Chunks indexed:", build_index(sys.argv[1] if len(sys.argv) > 1 else "data/handbook.pdf"))
