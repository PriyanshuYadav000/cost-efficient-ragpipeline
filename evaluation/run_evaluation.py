
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import BASE_DIR
from app.store import search_mmr


EVAL_DIR = BASE_DIR / "evaluation"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
CHUNKS_FILE = EVAL_DIR / "sample_chunks.csv"
RESULTS_FILE = EVAL_DIR / "retrieval_results.json"


def normalize(text):
    return " ".join(text.casefold().split())


def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_relevant_ids(question, chunks):
    relevant_ids = set()

    for snippet in question["relevant_texts"]:
        snippet = normalize(snippet)

        matches = [
            chunk["id"]
            for chunk in chunks
            if snippet in normalize(chunk["text"])
        ]

        if not matches:
            raise ValueError(
                f"Could not match a relevance label for "
                f"{question['id']}: {snippet}"
            )

        relevant_ids.update(matches)

    return relevant_ids


def recall_at_k(retrieved, relevant):
    if not relevant:
        return 0.0

    return len(set(retrieved) & relevant) / len(relevant)


def hit_rate_at_k(retrieved, relevant):
    return float(bool(set(retrieved) & relevant))


def reciprocal_rank(retrieved, relevant):
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    gains = [
        1 if chunk_id in relevant else 0
        for chunk_id in retrieved[:k]
    ]

    dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )

    ideal_gains = [1] * min(len(relevant), k)

    idcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )

    return dcg / idcg if idcg else 0.0


def context_precision(retrieved, relevant):
    if not retrieved:
        return 0.0

    return sum(
        chunk_id in relevant for chunk_id in retrieved
    ) / len(retrieved)


def percentile(values, percent):
    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    return (
        values[lower] * (upper - position)
        + values[upper] * (position - lower)
    )


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    chunks = load_chunks()
    questions = dataset["questions"]
    k = dataset["top_k"]
    metadata_filter = dataset.get("metadata_filter")

    if not 15 <= len(questions) <= 30:
        raise ValueError("Use between 15 and 30 evaluation questions")

    if not chunks:
        raise ValueError("No evaluation chunks found")

    question_results = []

    for question in questions:
        relevant = find_relevant_ids(question, chunks)

        start = time.perf_counter()

        retrieved_chunks = search_mmr(
            question["question"],
            top_k=k,
            metadata_filter=metadata_filter
        )

        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [
            chunk["id"] for chunk in retrieved_chunks
        ]

        result = {
            "id": question["id"],
            "question": question["question"],
            "relevant_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_sources": [
                chunk["metadata"].get("source", "")
                for chunk in retrieved_chunks
            ],
            "retrieval_latency_ms": round(latency_ms, 2),
            "metrics": {
                f"recall_at_{k}": recall_at_k(
                    retrieved_ids, relevant
                ),
                f"hit_rate_at_{k}": hit_rate_at_k(
                    retrieved_ids, relevant
                ),
                "mrr": reciprocal_rank(
                    retrieved_ids, relevant
                ),
                f"ndcg_at_{k}": ndcg_at_k(
                    retrieved_ids, relevant, k
                ),
                "context_precision": context_precision(
                    retrieved_ids, relevant
                )
            }
        }

        question_results.append(result)

        print(
            f"{question['id']}: "
            f"Recall={result['metrics'][f'recall_at_{k}']:.3f}, "
            f"MRR={result['metrics']['mrr']:.3f}, "
            f"latency={latency_ms:.1f} ms"
        )

    metric_names = list(question_results[0]["metrics"])

    averages = {
        name: round(
            sum(result["metrics"][name] for result in question_results)
            / len(question_results),
            4
        )
        for name in metric_names
    }

    latencies = [
        result["retrieval_latency_ms"]
        for result in question_results
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "top_k": k,
        "metadata_filter": metadata_filter,
        "evaluation_sources": sorted({
            chunk["source"] for chunk in chunks
        }),
        "metrics": averages,
        "retrieval_latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2)
        },
        "per_question": question_results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nAverage retrieval metrics:")
    print(json.dumps(averages, indent=2))
    print("\nRetrieval latency:")
    print(json.dumps(report["retrieval_latency_ms"], indent=2))
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
