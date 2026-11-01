# ===== MEMBER D owns this file (evaluation) =====
import csv, rag
rows = list(csv.DictReader(open("questions.csv", encoding="utf-8")))
out = []
for r in rows:
    print("Q", r["id"])
    a = rag.answer(r["question"], use_rag=True)
    b = rag.answer(r["question"], use_rag=False)
    pages = [str(c["page"]) for c in a["chunks"]]
    exp = [p for p in r["expected_page"].split(";") if p]
    hit = (any(p in pages for p in exp) if exp else "n/a")
    refused = "not found in the notes" in a["answer"].lower()
    out.append({
        "id": r["id"], "question": r["question"], "in_docs": r["in_docs"], "expected": r["expected"],
        "rag_answer": a["answer"].replace("\n", " "), "retrieved_pages": ",".join(pages),
        "retrieval_hit": hit, "rag_refused": refused,
        "norag_answer": b["answer"].replace("\n", " "),
        "rag_correct(Y/N)": "", "norag_correct(Y/N)": "", "hallucinated(Y/N)": "", "notes": ""})
with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
ind = [o for o in out if o["in_docs"] == "yes"]; outd = [o for o in out if o["in_docs"] == "no"]
print(f"Retrieval hit rate: {sum(o['retrieval_hit'] is True for o in ind)}/{len(ind)}")
print(f"Correct refusals on out-of-doc: {sum(o['rag_refused'] for o in outd)}/{len(outd)}")
print("Now open results.csv and fill the Y/N columns by hand.")
