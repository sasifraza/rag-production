import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.pipeline.rag_pipeline import RAGPipeline


def load_dataset(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def build_ragas_dataset(pipeline: RAGPipeline, samples: list) -> Dataset:
    rows = []
    for sample in samples:
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        print(f"Running: {question}")
        response = pipeline.run(question)
        rows.append({
            "question": question,
            "answer": response.answer,
            "contexts": [doc.page_content for doc in response.sources],
            "ground_truth": ground_truth,
        })
    return Dataset.from_list(rows)


def main():
    print("Loading pipeline...")
    pipeline = RAGPipeline()

    print("Loading eval dataset...")
    samples = load_dataset("eval/dataset.json")

    print(f"Running {len(samples)} eval samples...\n")
    dataset = build_ragas_dataset(pipeline, samples)

    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    df = result.to_pandas()
    print("\n── Results ──")
    print(df.to_string())
    print("\n── Mean Scores ──")
    print(df[["faithfulness", "answer_relevancy",
              "context_precision", "context_recall"]].mean())

    df.to_csv("eval/results.csv", index=False)
    print("\nSaved to eval/results.csv")


if __name__ == "__main__":
    main()