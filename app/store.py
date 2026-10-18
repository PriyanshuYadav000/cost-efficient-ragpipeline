
import chromadb
import numpy as np
from app.config import (CHROMA_DIR,DEFAULT_TOP_K,MMR_FETCH_K,MMR_LAMBDA,MIN_RELEVANCE,)
from app.embeddings import embed_texts, embed_query

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def add_documents(documents):
    if not documents:
        return {"added": 0, "message": "No chunks to add"}

    document_id = documents[0]["metadata"]["document_id"]
    source = documents[0]["metadata"]["source"]

    existing = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    if existing["ids"]:
        return {
            "added": 0,
            "message": "Document already indexed",
            "total_vectors": collection.count()
        }

    texts = [item["text"] for item in documents]
    ids = [item["id"] for item in documents]
    metadata = [item["metadata"] for item in documents]

    # Generate embeddings before changing the existing index.
    vectors = embed_texts(texts)

    old_file = collection.get(
        where={"source": source},
        include=["metadatas"]
    )

    if old_file["ids"]:
        collection.delete(where={"source": source})

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadata,
        embeddings=vectors
    )

    return {
        "added": len(ids),
        "message": "Document indexed",
        "total_vectors": collection.count()
    }


def search_mmr(query, top_k=None, metadata_filter=None):
    if top_k is None:
        top_k = DEFAULT_TOP_K

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if collection.count() == 0:
        return []

    query_vector = np.array(embed_query(query), dtype=float)

    search_args = {
        "query_embeddings": [query_vector.tolist()],
        "n_results": min(MMR_FETCH_K, collection.count()),
        "include": ["documents", "metadatas", "embeddings"]
    }

    if metadata_filter:
        search_args["where"] = metadata_filter

    results = collection.query(**search_args)

    ids = results["ids"][0]

    if not ids:
        return []

    texts = results["documents"][0]
    metadata = results["metadatas"][0]
    vectors = np.array(results["embeddings"][0], dtype=float)

    query_norm = max(np.linalg.norm(query_vector), 1e-12)
    vector_norms = np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )

    query_vector = query_vector / query_norm
    vectors = vectors / vector_norms

    relevance = vectors @ query_vector

    selected = [int(np.argmax(relevance))]
    limit = min(top_k, len(ids))

    while len(selected) < limit:
        similarity = vectors @ vectors[selected].T
        redundancy = similarity.max(axis=1)

        scores = (
            MMR_LAMBDA * relevance
            - (1 - MMR_LAMBDA) * redundancy
        )

        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))

    chunks = []

    for index in selected:
        score = float(relevance[index])

        if score < MIN_RELEVANCE:
            continue

        chunks.append({
            "id": ids[index],
            "text": texts[index],
            "metadata": metadata[index],
            "score": round(score, 4)
        })

    return chunks


def get_stats():
    return {
        "total_vectors": collection.count(),
        "collection": collection.name
    }


def list_documents():
    data = collection.get(include=["metadatas"])
    documents = {}

    for metadata in data["metadatas"]:
        if metadata:
            source = metadata["source"]
            documents[source] = {
                "source": source,
                "document_id": metadata["document_id"],
                "file_type": metadata["file_type"]
            }

    return list(documents.values())
