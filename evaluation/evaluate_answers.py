
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
CHECKPOINT_FILE = EVAL_DIR / "answer_results_checkpoint.json"
PARTIAL_FILE = EVAL_DIR / "answer_results_partial.json"


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
        help="Evaluate only the first N questions"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved checkpoint"
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

    if args.limit is not None and args.resume:
        parser.error("Use --limit for a preview, or --resume for the full evaluation.")

    checkpoint_file = (
        CHECKPOINT_FILE if args.limit is None
        else EVAL_DIR / "answer_results_preview_checkpoint.json"
    )

    output_file = (
        RESULTS_FILE if args.limit is None
        else EVAL_DIR / "answer_results_preview.json"
    )

    partial_file = (
        PARTIAL_FILE if args.limit is None
        else EVAL_DIR / "answer_results_preview_partial.json"
    )

    question_ids = {item["id"] for item in questions}
    results_by_id = {}

    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, encoding="utf-8") as f:
            checkpoint = json.load(f)

        results_by_id = {
            row["id"]: row
            for row in checkpoint.get("results", [])
            if row.get("id") in question_ids
        }

        print(
            f"Resuming evaluation: "
            f"{len(results_by_id)}/{len(questions)} cases already saved."
        )
    elif args.resume:
        print("No checkpoint found. Starting a new evaluation.")

    def ordered_results():
        return [
            results_by_id[item["id"]]
            for item in questions
            if item["id"] in results_by_id
        ]

    def save_checkpoint():
        checkpoint = {
            "judge_model": JUDGE_MODEL,
            "results": ordered_results()
        }

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def valid_judge(row):
        judge = row.get("judge") or {}

        return (
            judge.get("faithfulness_score") is not None
            and judge.get("relevance_score") is not None
        )

    def is_rate_limit_error(error):
        message = str(error).lower()

        return (
            getattr(error, "status_code", None) == 429
            or "rate_limit" in message
            or "rate limit" in message
            or "429" in message
        )

    client = Groq(api_key=GROQ_API_KEY)
    stopped_early = False

    for item in questions:
        question_id = item["id"]
        existing = results_by_id.get(question_id)

        if existing and valid_judge(existing):
            print(f"Skipping {question_id}: already evaluated.")
            continue

        if existing and existing.get("generated_answer") is not None:
            row = existing
            print(f"Retrying judge for {question_id} using saved answer.")

        else:
            print(f"Evaluating {question_id}...")

            try:
                result = answer_question(
                    question=item["question"],
                    top_k=dataset["top_k"],
                    metadata_filter=dataset.get("metadata_filter")
                )

            except Exception as error:
                save_checkpoint()

                print(
                    f"Answer generation failed for {question_id}: {error}"
                )
                print("Completed results have been checkpointed.")

                if is_rate_limit_error(error):
                    print("Stopped because of an API rate limit.")

                stopped_early = True
                break

            answer = result["answer"]
            expected = item["expected_answer"]

            row = {
                "id": question_id,
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
                item["expected_answer"],
                row["generated_answer"],
                row["sources"]
            )

            row["judge"] = verdict

        except Exception as error:
            row["judge"] = {
                "error": str(error),
                "faithfulness_score": None,
                "relevance_score": None
            }

            print(f"Judge failed for {question_id}: {error}")

        results_by_id[question_id] = row
        save_checkpoint()

        print(
            f"EM={row['exact_match']:.0f}, "
            f"F1={row['token_f1']:.3f}, "
            f"retrieval={row['retrieval_ms']:.1f} ms"
        )

        if row["judge"].get("faithfulness_score") is None:
            if is_rate_limit_error(
                row["judge"].get("error", "")
            ):
                print("Stopped because of an API rate limit.")
                stopped_early = True
                break

    results = ordered_results()

    valid_judges = [
        row["judge"] for row in results
        if valid_judge(row)
    ]

    summary = {
        "questions_run": len(results),
        "questions_requested": len(questions),
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
            row.get("generator_tokens", 0) for row in results
        ),

        "judge_tokens": sum(
            j.get("judge_tokens", 0) for j in valid_judges
        )
    }

    complete = (
        len(results) == len(questions)
        and len(valid_judges) == len(questions)
    )

    report = {
        "generator_model": "configured by GROQ_MODEL",
        "judge_model": JUDGE_MODEL,
        "complete": complete,
        "summary": summary,
        "results": results
    }

    report_file = output_file if complete else partial_file

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report to: {report_file}")
    print(f"Checkpoint saved to: {checkpoint_file}")

    if not complete or stopped_early:
        print(
            "\nEvaluation is incomplete. After the API limit clears, "
            "run: python -m evaluation.evaluate_answers --resume"
        )
    else:
        print("\nAll requested questions have valid judge results.")


if __name__ == "__main__":
    main()
