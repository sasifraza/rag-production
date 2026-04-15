import json
import re
from statistics import mean

from app.rag_pipeline import RAGPipeline


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text: str) -> set[str]:
    text = normalize(text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())


def answer_overlap_score(answer: str, ground_truth: str) -> float:
    gt_tokens = token_set(ground_truth)
    ans_tokens = token_set(answer)

    if not gt_tokens:
        return 0.0

    overlap = gt_tokens.intersection(ans_tokens)
    return len(overlap) / len(gt_tokens)


def citation_present(answer: str) -> float:
    return 1.0 if re.search(r"\[\d+\]", answer) else 0.0


def context_hit_score(sources: list[dict], ground_truth: str) -> float:
    gt_tokens = token_set(ground_truth)
    if not gt_tokens:
        return 0.0

    source_text = " ".join(s["content"] for s in sources)
    src_tokens = token_set(source_text)

    overlap = gt_tokens.intersection(src_tokens)
    return len(overlap) / len(gt_tokens)


def evaluate_one(pipeline: RAGPipeline, item: dict) -> dict:
    result = pipeline.ask(item["question"])
    answer = result["answer"]
    sources = result["sources"]
    ground_truth = item["ground_truth"]

    overlap = answer_overlap_score(answer, ground_truth)
    citation = citation_present(answer)
    context_hit = context_hit_score(sources, ground_truth)

    # weighted composite score
    total = 0.5 * overlap + 0.2 * citation + 0.3 * context_hit

    return {
        "question": item["question"],
        "ground_truth": ground_truth,
        "answer": answer,
        "sources": [s["source"] for s in sources],
        "answer_overlap": round(overlap, 4),
        "citation_present": round(citation, 4),
        "context_hit": round(context_hit, 4),
        "total_score": round(total, 4),
    }


def main() -> None:
    with open("eval/dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    pipeline = RAGPipeline()

    rows = [evaluate_one(pipeline, item) for item in data]

    print("\n=== EVALUATION REPORT ===")
    for row in rows:
        print("\n-----------------------------------")
        print("Q:", row["question"])
        print("GT:", row["ground_truth"])
        print("ANS:", row["answer"])
        print("SRC:", row["sources"])
        print("answer_overlap:", row["answer_overlap"])
        print("citation_present:", row["citation_present"])
        print("context_hit:", row["context_hit"])
        print("total_score:", row["total_score"])

    avg_overlap = mean(r["answer_overlap"] for r in rows)
    avg_citation = mean(r["citation_present"] for r in rows)
    avg_context = mean(r["context_hit"] for r in rows)
    avg_total = mean(r["total_score"] for r in rows)

    print("\n=== SUMMARY ===")
    print(f"avg_answer_overlap: {avg_overlap:.4f}")
    print(f"avg_citation_present: {avg_citation:.4f}")
    print(f"avg_context_hit: {avg_context:.4f}")
    print(f"avg_total_score: {avg_total:.4f}")

    threshold = 0.70
    if avg_total < threshold:
        raise ValueError(
            f"Evaluation failed. avg_total_score={avg_total:.4f} is below threshold={threshold:.2f}"
        )


if __name__ == "__main__":
    main()