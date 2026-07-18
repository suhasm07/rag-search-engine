import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, GOLDEN_DATASET_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load golden dataset
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    # Load documents and initialize hybrid search
    documents = load_movies()
    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])

        # Run RRF search
        results = hs.rrf_search(query, k=60, limit=limit)

        # Get retrieved titles
        retrieved_titles = [r["title"] for r in results]

        # Calculate precision
        relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant_docs)

        precision = relevant_retrieved / limit if limit > 0 else 0.0
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()


if __name__ == "__main__":
    main()
