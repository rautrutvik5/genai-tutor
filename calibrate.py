# Finds a MIN_SCORE so off-topic questions are refused automatically, and saves it in .env
import csv, os, numpy as np, rag
rows = list(csv.DictReader(open("questions.csv", encoding="utf-8")))
EXTRA_OUT = ["Who won the cricket world cup?", "What is the capital of France?",
             "How do I cook pasta?", "What is the latest iPhone model?", "Explain how a car engine works"]
score = lambda q: rag.retrieve(q, 1)[0]["score"]
ins = [score(r["question"]) for r in rows if r["in_docs"] == "yes"]
outs = [score(r["question"]) for r in rows if r["in_docs"] == "no"] + [score(q) for q in EXTRA_OUT]
print(f"In-docs   best-chunk score: min {min(ins):.3f}  avg {np.mean(ins):.3f}")
print(f"Off-topic best-chunk score: max {max(outs):.3f}  avg {np.mean(outs):.3f}")
thr = (min(ins) + max(outs)) / 2 if min(ins) > max(outs) else max(outs)
if min(ins) <= max(outs):
    print("WARNING: scores overlap; some real questions may get refused. Check results after.")
lines = open(".env").read().splitlines() if os.path.exists(".env") else []
lines = [l for l in lines if not l.startswith("MIN_SCORE")] + [f"MIN_SCORE={thr:.3f}"]
open(".env", "w").write("\n".join(lines) + "\n")
print(f"Saved MIN_SCORE={thr:.3f} in .env")
