
import csv
import json
from datetime import date
from pathlib import Path

from app.config import BASE_DIR


# Storage assumptions for all three scales
DIMENSIONS = 384
BYTES_PER_FLOAT = 4
AVG_TEXT_BYTES = 1024
AVG_METADATA_BYTES = 512
INDEX_OVERHEAD = 1.5

bytes_per_vector = (
    DIMENSIONS * BYTES_PER_FLOAT
    + AVG_TEXT_BYTES
    + AVG_METADATA_BYTES
) * INDEX_OVERHEAD


# These are example hosting sizes, not measured benchmarks.
plans = [
    {
        "vectors": 100_000,
        "chroma_plan": "2 GiB RAM, 1 vCPU, 50 GiB SSD",
        "chroma_cost": 12.00,
        "managed_plan": "PostgreSQL Standard, 1 GiB RAM",
        "managed_cost": 15.15,
        "notes": "Small workload; managed plan includes 10-30 GiB storage."
    },
    {
        "vectors": 1_000_000,
        "chroma_plan": "8 GiB RAM, 4 vCPU, 160 GiB SSD",
        "chroma_cost": 48.00,
        "managed_plan": "PostgreSQL Standard, 2 GiB RAM",
        "managed_cost": 30.45,
        "notes": "Managed plan includes 30-60 GiB storage."
    },
    {
        "vectors": 10_000_000,
        "chroma_plan": "32 GiB RAM, 8 vCPU, 100 GiB SSD",
        "chroma_cost": 252.00,
        "managed_plan": "PostgreSQL Standard, 4 GiB RAM",
        "managed_cost": 60.90,
        "notes": (
            "Managed plan includes 60-120 GiB storage. "
            "Validate RAM and latency; a larger plan may be needed."
        )
    }
]


def estimate_storage(vectors):
    total_bytes = vectors * bytes_per_vector

    return {
        "storage_gb": round(total_bytes / 1_000_000_000, 2),
        "storage_gib": round(total_bytes / (1024 ** 3), 2)
    }


def main():
    output_dir = BASE_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for plan in plans:
        storage = estimate_storage(plan["vectors"])

        # Positive means Chroma hosting is cheaper.
        savings = plan["managed_cost"] - plan["chroma_cost"]

        result = {
            "vectors": plan["vectors"],
            "embedding_dimensions": DIMENSIONS,
            "estimated_storage_gb": storage["storage_gb"],
            "estimated_storage_gib": storage["storage_gib"],
            "chroma_host_plan": plan["chroma_plan"],
            "chroma_monthly_usd": plan["chroma_cost"],
            "managed_database_plan": plan["managed_plan"],
            "managed_monthly_usd": plan["managed_cost"],
            "chroma_savings_vs_managed_usd": round(savings, 2),
            "notes": plan["notes"]
        }

        results.append(result)

    assumptions = {
        "generated_on": date.today().isoformat(),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": DIMENSIONS,
        "float_bytes": BYTES_PER_FLOAT,
        "average_chunk_text_bytes": AVG_TEXT_BYTES,
        "average_metadata_bytes": AVG_METADATA_BYTES,
        "index_storage_overhead_factor": INDEX_OVERHEAD,
        "estimated_bytes_per_vector": bytes_per_vector,
        "currency": "USD",
        "month": "Monthly list-price estimate",
        "excludes": [
            "API and LLM charges",
            "Embedding-generation charges",
            "Taxes",
            "Monitoring and backup add-ons",
            "Engineering and maintenance labour",
            "Additional storage or compute beyond the selected plans"
        ],
        "chroma_price_source": (
            "https://www.digitalocean.com/pricing/droplets"
        ),
        "managed_price_source": (
            "https://www.digitalocean.com/pricing/managed-databases"
        ),
        "limitations": (
            "Storage is a planning estimate, not a measured ChromaDB "
            "database size. Hosting plans are not performance-equivalent. "
            "Benchmark latency and verify memory requirements before "
            "claiming production capacity."
        ),
        "results": results
    }

    json_file = output_dir / "cost_comparison.json"
    csv_file = output_dir / "cost_comparison.csv"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(assumptions, f, indent=2)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("Monthly cost comparison")
    print("-" * 76)

    for result in results:
        print(f"\nVectors: {result['vectors']:,}")
        print(f"Estimated storage: {result['estimated_storage_gib']} GiB")
        print(f"Chroma hosting: ${result['chroma_monthly_usd']:.2f}/month")
        print(
            "Managed PostgreSQL + pgvector: "
            f"${result['managed_monthly_usd']:.2f}/month"
        )
        print(
            "Savings (managed cost - Chroma cost): "
            f"${result['chroma_savings_vs_managed_usd']:.2f}/month"
        )
        print("Note:", result["notes"])

    print(f"\nJSON results: {json_file}")
    print(f"CSV results: {csv_file}")


if __name__ == "__main__":
    main()
