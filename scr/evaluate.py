"""Check how well retrieval and refusal work.

    python evaluate.py            retrieval + refusal check (cheap: embeddings only)
    python evaluate.py --full     also run Gemini on every question and print the answers to read

It prints the score ranges of answerable vs unanswerable questions and suggests a
RAG_MIN_SCORE for your embedder. Add your own questions to QUESTIONS.
"""
import argparse

from rag import ask, setup

# (question, what a correct retrieval contains)  or  (question, None) = the paper cannot answer this
# Strings match a chunk's title first, otherwise a phrase in its text.
QUESTIONS = [
    ("What is a Chu Law and what is it made from?", ["Chu Law"]),
    ("What is a Dao made of?", ["Dao"]),
    ("What is the Thagla?", ["Thagla"]),
    ("What is Kachong worn by Tiwa women?", ["Kachong"]),
    ("What do the Tiwas use to serve rice beer?", ["Tran", "Lawbe"]),
    ("Which instrument is played in the Barat festivals?", ["Sorailou"]),
    ("What is Ghar Deota Puja?", ["Ghar Deota Puja"]),
    ("Who are the Hill Tiwas and where do they live?", ["Hill Tiwas lives"]),
    ("What is the staple food of the Tiwas?", ["staple food"]),
    ("What is Pahar puja or Dong puja?", ["Dong puja"]),
    ("What tools do the Tiwas use for fishing?", ["Jakoi", "Khaloi", "Sepa", "Palo"]),
    ("Which raw materials do the Tiwas use to make things?", ["Bamboo", "Wood", "Cane"]),
    ("What is a Pangsi?", ["Pangsi"]),
    ("How is the Jakoria-Saroilou different from the Sorailou?", ["Jakoria-Saroilou"]),
    ("What is Khuri lo made of?", ["cowry"]),
    ("How do the Tiwas build their houses?", ["choraghar"]),
    ("Tell me about the Jon Beel Mela", ["Jon Beel"]),
    ("Which ornament is made from old coins?", ["Sigi lo"]),
    ("Who won the FIFA World Cup in 2022?", None),
    ("How do I reverse a list in Python?", None),
    ("What are the Tiwa funeral rituals?", None),
    ("How do Tiwas conduct wedding ceremonies?", None),
    ("How much does a traditional Kachong cost?", None),
    ("What is the capital of France?", None),
]


def is_match(chunk, expected):
    text = chunk["text"].lower()
    return chunk.get("title") in expected or any(e.lower() in text for e in expected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", choices=["gemini", "tfidf"])
    p.add_argument("--llm", choices=["gemini", "groq"], help="LLM for --full (default: gemini)")
    p.add_argument("--full", action="store_true", help="also call Gemini and print answers")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    index, llm, min_score = setup(args.embedder, no_llm=not args.full, llm_name=args.llm)
    found, rr, ans_best, ref_best, misses, leaks = 0, 0.0, [], [], [], []

    for q, expected in QUESTIONS:
        hits = index.search(q, args.k)
        best = hits[0]["score"] if hits else 0.0
        if expected is None:
            ref_best.append(best)
            if best >= min_score:
                leaks.append((q, best))
            continue
        ans_best.append(best)
        rank = next((h["label"] for h in hits if is_match(h["chunk"], expected)), None)
        if rank:
            found += 1
            rr += 1 / int(rank[1:])
        else:
            misses.append(q)

    n_a, n_r = len(ans_best), len(ref_best)
    print(f"min_score = {min_score}   top-k = {args.k}\n")
    print(f"Answerable ({n_a}):   right chunk in top-{args.k}: {found}/{n_a}   MRR {rr / n_a:.2f}")
    print(f"                    pass the score gate:   {sum(s >= min_score for s in ans_best)}/{n_a}")
    print(f"Unanswerable ({n_r}): stopped by score gate: {n_r - len(leaks)}/{n_r}")
    lo, hi = min(ans_best), max(ref_best)
    print(f"\nBest-score range  answerable: {min(ans_best):.2f}-{max(ans_best):.2f}   "
          f"unanswerable: {min(ref_best):.2f}-{max(ref_best):.2f}")
    if lo > hi:
        print(f"Clean split. Suggested RAG_MIN_SCORE = {(lo + hi) / 2:.2f}")
    else:
        print(f"Ranges overlap, so no threshold separates them perfectly. A value near {hi:.2f} stops "
              f"every unanswerable question but loses some answerable ones;\nthe model's own "
              f"NOT_IN_DOCUMENT reply is the second safety net (see --full).")
    for q in misses:
        print("  MISS (right chunk not retrieved):", q)
    for q, s in leaks:
        print(f"  PASSED GATE ({s:.2f}, should be refused):", q)

    if args.full:
        print("\n" + "=" * 70 + "\nFULL ANSWERS (read these yourself)\n" + "=" * 70)
        for q, expected in QUESTIONS:
            a = ask(q, index, min_score, args.k, llm)
            tag = "SHOULD ANSWER" if expected else "SHOULD REFUSE"
            print(f"\n[{tag}] {q}\n -> grounded={a['grounded']} ({a['reason']})\n    {a['answer'][:400]}")
            print("    pages:", ", ".join(s["page_ref"] for s in a["sources"]) or "-")


if __name__ == "__main__":
    main()