
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

from groq import Groq

from app.config import BASE_DIR, GROQ_API_KEY, JUDGE_MODEL
from app.rag import answer_question


EVAL_DIR = BASE_DIR / "evaluation"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "answer_results.json"


def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference):
    pred_words = normalize(prediction).split()
    ref_words = normalize(reference).split()

    if not pred_words or not ref_words:
        return float(pred_words == ref_words)

    common = Counter(pred_words) & Counter(ref_words)
    matches = sum(common.values())

    if matches == 0:
        return 0.0

    precision = matches / len(pred_words)
    recall = matches / len(ref_words)

    return 2 * precision * recall / (precision + recall)


def parse_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            raise ValueError("Judge did not return valid JSON")

        return json.loads(text[start:end + 1])


def judge_answer(client, question, expected, answer, sources):
    context = "\n\n".join(
        f"[{s['citation']}] {s['source']}, page {s['page']}\n{s['text']}"
        for s in sources
    )

    prompt = f"""
Evaluate this answer using the supplied source excerpts.

Question:
{question}

Reference answer:
{expected}

Generated answer:
{answer}

Source excerpts:
{context if context else "No source excerpts were retrieved."}

Score both criteria from 1 to 5.

Faithfulness:
5 = all factual claims are supported by the excerpts.
3 = partly supported, with some questionable claims.
1 = mostly unsupported or contradicted by the excerpts.

Relevance:
5 = directly answers the question.
3 = partly answers it or includes substantial irrelevant content.
1 = does not answer the question.

The reference answer helps assess relevance, but source excerpts
must be used to judge faithfulness.

Return only valid JSON with this structure:
{{
  "faithfulness_score": 1,
  "faithfulness_reason": "reason based on the excerpts",
  "relevance_score": 1,
  "relevance_reason": "reason based on the question"
}}
"""

    
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict answer-quality evaluator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_completion_tokens=1000,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content or ""
    verdict = parse_json(raw)

    faithfulness = int(verdict["faithfulness_score"])
    relevance = int(verdict["relevance_score"])

    if not 1 <= faithfulness <= 5 or not 1 <= relevance <= 5:
        raise ValueError("Judge scores must be between 1 and 5")

    return {
        "faithfulness_score": faithfulness,
        "faithfulness_reason": verdict["faithfulness_reason"],
        "relevance_score": relevance,
        "relevance_reason": verdict["relevance_reason"],
        "judge_raw_response": raw,
        "judge_prompt": prompt,
        "judge_tokens": (
            response.usage.total_tokens if response.usage else 0
        )
    }


def percentile(values, percent):
    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return round(values[lower], 2)

    result = (
        values[lower] * (upper - position)
        + values[upper] * (position - lower)
    )

    return round(result, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions for a quick test"
    )
    args = parser.parse_args()

    if not GROQ_API_KEY:
        raise ValueError("Set GROQ_API_KEY in your .env file")

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[:args.limit]

    client = Groq(api_key=GROQ_API_KEY)
    results = []

    for item in questions:
        print(f"Evaluating {item['id']}...")

        # Generate a real answer using the existing RAG pipeline.
        result = answer_question(
            question=item["question"],
            top_k=dataset["top_k"],
            metadata_filter=dataset.get("metadata_filter")
        )

        answer = result["answer"]
        expected = item["expected_answer"]

        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_answer": expected,
            "generated_answer": answer,
            "exact_match": exact_match(answer, expected),
            "token_f1": round(token_f1(answer, expected), 4),
            "sources": result["sources"],
            "chunks_retrieved": result["chunks_retrieved"],
            "retrieval_ms": result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
            "total_latency_ms": result["total_latency_ms"],
            "generator_tokens": result["total_tokens"]
        }

        try:
            verdict = judge_answer(
                client,
                item["question"],
                expected,
                answer,
                result["sources"]
            )
            row["judge"] = verdict
        except Exception as e:
            row["judge"] = {
                "error": str(e),
                "faithfulness_score": None,
                "relevance_score": None
            }
            print(f"Judge failed for {item['id']}: {e}")

        results.append(row)

        print(
            f"EM={row['exact_match']:.0f}, "
            f"F1={row['token_f1']:.3f}, "
            f"retrieval={row['retrieval_ms']:.1f} ms"
        )

    valid_judges = [
        row["judge"] for row in results
        if row["judge"].get("faithfulness_score") is not None
        and row["judge"].get("relevance_score") is not None
    ]

    summary = {
        "questions_run": len(results),
        "valid_judge_results": len(valid_judges),
        "exact_match_mean": round(
            sum(row["exact_match"] for row in results) / len(results), 4
        ) if results else None,
        "token_f1_mean": round(
            sum(row["token_f1"] for row in results) / len(results), 4
        ) if results else None,
        "faithfulness_mean_1_to_5": round(
            sum(j["faithfulness_score"] for j in valid_judges)
            / len(valid_judges), 3
        ) if valid_judges else None,
        "relevance_mean_1_to_5": round(
            sum(j["relevance_score"] for j in valid_judges)
            / len(valid_judges), 3
        ) if valid_judges else None,
        "retrieval_p50_ms": percentile(
            [row["retrieval_ms"] for row in results], 50
        ),
        "retrieval_p95_ms": percentile(
            [row["retrieval_ms"] for row in results], 95
        ),
        "end_to_end_p50_ms": percentile(
            [row["total_latency_ms"] for row in results], 50
        ),
        "end_to_end_p95_ms": percentile(
            [row["total_latency_ms"] for row in results], 95
        ),
        "generator_tokens": sum(
            row["generator_tokens"] for row in results
        ),
        "judge_tokens": sum(
            j.get("judge_tokens", 0)
            for j in valid_judges
        )
    }

    report = {
        "generator_model": "configured by GROQ_MODEL",
        "judge_model": JUDGE_MODEL,
        "summary": summary,
        "results": results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
