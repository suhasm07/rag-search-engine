import argparse

from lib.hybrid_search import (
    HybridSearch,
    enhance_query_expand,
    enhance_query_rewrite,
    enhance_query_spell,
    evaluate_results,
    normalize_scores,
    rerank_batch,
    rerank_cross_encoder,
    rerank_individual,
)
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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method",
    )
    rrf_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate results using LLM",
    )

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
            query = args.query
            debug = args.debug

            if debug:
                print(f"[DEBUG] Original query: '{query}'")

            # Enhancement
            match args.enhance:
                case "spell":
                    enhanced_query = enhance_query_spell(query)
                    print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'\n")
                    query = enhanced_query
                case "rewrite":
                    enhanced_query = enhance_query_rewrite(query)
                    print(f"Enhanced query (rewrite): '{query}' -> '{enhanced_query}'\n")
                    query = enhanced_query
                case "expand":
                    enhanced_query = enhance_query_expand(query)
                    print(f"Enhanced query (expand): '{query}' -> '{enhanced_query}'\n")
                    query = enhanced_query

            if debug and args.enhance:
                print(f"[DEBUG] Query after enhancement: '{query}")

            # Fetch more results if re-ranking
            fetch_limit = args.limit * 5 if args.rerank_method else args.limit

            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, args.k, fetch_limit)

            if debug:
                print(f"\n[DEBUG] RRF results (top {fetch_limit}):")
                for i, r in enumerate(results, start=1):
                    bm25 = r["bm25_rank"] if r["bm25_rank"] is not None else "N/A"
                    sem = r["sem_rank"] if r["sem_rank"] is not None else "N/A"
                    print(f"  {i}. {r['title']} | RRF: {r['rrf_score']:.3f} | BM25 Rank: {bm25} | Sem Rank: {sem}")
                print()


            # Re-rank if requested
            if args.rerank_method == "individual":
                print(f"Re-ranking top {fetch_limit} results using individual method...")
                results = rerank_individual(query, results)
                results = results[:args.limit]
                if debug:
                    print(f"\n[DEBUG] Results after individual re-ranking:")
                    for i, r in enumerate(results, start=1):
                        print(f"  {i}. {r['title']} | Re-rank Score: {r['rerank_score']:.3f}")
                    print()

                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                for i, result in enumerate(results, start=1):
                    bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
                    sem_rank = result["sem_rank"] if result["sem_rank"] is not None else "N/A"
                    print(f"{i}. {result['title']}")
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                    print(f"   {result['document']}...")

            elif args.rerank_method == "batch":
                print(f"Re-ranking top {fetch_limit} results using batch method...")
                results = rerank_batch(query, results)
                results = results[:args.limit]
                if debug:
                    print(f"\n[DEBUG] Results after batch re-ranking:")
                    for i, r in enumerate(results, start=1):
                        print(f"  {i}. {r['title']} | Re-rank Rank: {r['rerank_rank']}")
                    print()

                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                for i, result in enumerate(results, start=1):
                    bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
                    sem_rank = result["sem_rank"] if result["sem_rank"] is not None else "N/A"
                    print(f"{i}. {result['title']}")
                    print(f"   Re-rank Rank: {result['rerank_rank']}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                    print(f"   {result['document']}...")

            elif args.rerank_method == "cross_encoder":
                    print(f"Re-ranking top {fetch_limit} results using cross_encoder method...")
                    results = rerank_cross_encoder(query, results)
                    results = results[:args.limit]
                    if debug:
                        print(f"\n[DEBUG] Results after cross-encoder re-ranking:")
                        for i, r in enumerate(results, start=1):
                            print(f"  {i}. {r['title']} | Cross Encoder Score: {r['cross_encoder_score']:.3f}")
                        print()

                    print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                    for i, result in enumerate(results, start=1):
                        bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
                        sem_rank = result["sem_rank"] if result["sem_rank"] is not None else "N/A"
                        print(f"{i}. {result['title']}")
                        print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                        print(f"   RRF Score: {result['rrf_score']:.3f}")
                        print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                        print(f"   {result['document']}...")
            else:
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                for i, result in enumerate(results, start=1):
                    bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
                    sem_rank = result["sem_rank"] if result["sem_rank"] is not None else "N/A"
                    print(f"{i}. {result['title']}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                    print(f"   {result['document']}...")

            if args.evaluate:
                scores = evaluate_results(query, results)
                print("\nEvaluation Report:")
                for i, (result, score) in enumerate(zip(results, scores), start=1):
                    print(f"{i}. {result['title']}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
