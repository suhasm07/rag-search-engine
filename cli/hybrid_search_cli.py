import argparse

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid weighted search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic search (0-1)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid RRF search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if not args.scores:
                return
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['sem_score']:.3f}")
                print(f"  {result['document']}...")
        case "rrf-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results, start=1):
                bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
                sem_rank = result["sem_rank"] if result["sem_rank"] is not None else "N/A"
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
