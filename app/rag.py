
import time

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.store import search_mmr

def answer_question(question, top_k=None, metadata_filter=None):
    start_time = time.perf_counter()
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not GROQ_API_KEY:
        raise ValueError("Please add GROQ_API_KEY to your .env file")

    retrieval_start = time.perf_counter()

    chunks = search_mmr(
        query=question,
        top_k=top_k,
        metadata_filter=metadata_filter
    )

    retrieval_time = (time.perf_counter() - retrieval_start) * 1000

    if not chunks:
        return {
            "answer": "I could not find enough relevant information in the uploaded documents to answer this question.",
            "sources": [],
            "chunks_retrieved": 0,
            "retrieval_ms": round(retrieval_time, 2),
            "generation_ms": 0,
            "total_latency_ms": round(
                (time.perf_counter() - start_time) * 1000, 2
            ),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    context = []
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        citation = f"S{i}"

        context.append(
            f"[{citation}]\n"
            f"Source: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Content: {chunk['text']}"
        )

        sources.append({
            "citation": citation,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk_index": metadata["chunk_index"],
            "score": chunk["score"],
            "text": chunk["text"]
        })

    prompt = f"""
Answer the question using only the source excerpts below.

Rules:
- Do not use outside knowledge.
- If the excerpts do not contain the answer, say that the information
  is not available in the uploaded documents.
- Cite supporting excerpts using their IDs, such as [S1].
- Do not invent facts or citations.
- Keep the answer clear and concise.

Question:
{question}

Source excerpts:
{"".join(chr(10) + item + chr(10) for item in context)}
"""

    client = Groq(api_key=GROQ_API_KEY)

    generation_start = time.perf_counter()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Follow the supplied source excerpts and instructions."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_completion_tokens=800
    )

    generation_time = (
        time.perf_counter() - generation_start
    ) * 1000

    answer = response.choices[0].message.content or ""

    usage = response.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    return {
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(chunks),
        "retrieval_ms": round(retrieval_time, 2),
        "generation_ms": round(generation_time, 2),
        "total_latency_ms": round(
            (time.perf_counter() - start_time) * 1000, 2
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
