
import csv

from app.config import BASE_DIR
from app.store import collection


sources = [
    "sample_handbook.md",
    "it_operations.md",
]

data = collection.get(
    where={"source": {"$in": sources}},
    include=["documents", "metadatas"]
)

rows = []

for doc_id, text, metadata in zip(
    data["ids"],
    data["documents"],
    data["metadatas"]
):
    rows.append({
        "id": doc_id,
        "source": metadata["source"],
        "page": metadata["page"],
        "chunk_index": metadata["chunk_index"],
        "text": text
    })

rows.sort(
    key=lambda row: (
        row["source"],
        int(row["chunk_index"])
    )
)

output_file = BASE_DIR / "evaluation" / "sample_chunks.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "source",
            "page",
            "chunk_index",
            "text"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

found_sources = sorted(set(row["source"] for row in rows))

print("Exported chunks:", len(rows))
print("Sources found:", found_sources)

missing = [source for source in sources if source not in found_sources]

if missing:
    print("Sources not found in ChromaDB:", missing)

print("Saved to:", output_file)
