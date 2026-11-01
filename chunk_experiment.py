# ===== MEMBER A: chunk size / top-k comparison -> goes straight into the report =====
import csv, numpy as np, rag
pages = rag.read_pdf("data/handbook.pdf")
qs = [r for r in csv.DictReader(open("questions.csv", encoding="utf-8")) if r["in_docs"] == "yes"]
qvecs = [np.array(rag.embed(q["question"], "query"), dtype="float32") for q in qs]
print("| chunk size | overlap | #chunks | top-k | retrieval hit rate |\n|---|---|---|---|---|")
for size in (400, 800, 1200):
    overlap = int(size * 0.18)
    chunks = rag.chunk_pages(pages, size, overlap)
    vecs = np.array([rag.embed(c["text"], "document") for c in chunks], dtype="float32")
    for k in (2, 4, 6):
        hits = 0
        for q, qv in zip(qs, qvecs):
            sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-9)
            got = {str(chunks[i]["page"]) for i in np.argsort(-sims)[:k]}
            hits += any(p in got for p in q["expected_page"].split(";"))
        print(f"| {size} | {overlap} | {len(chunks)} | {k} | {hits}/{len(qs)} |")
