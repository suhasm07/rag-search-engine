import argparse

from lib.keyword_search import build_command, bm25_idf_command, bm25_search_command, bm25_tf_command, idf_command, search_command, tf_command, tfidf_command
from lib.search_utils import BM25_B, BM25_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # build command
    subparsers.add_parser("build", help="Build the inverted index")

    # search command
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # tf command
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    # idf command
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    # tfidf command
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")


    # bm25 idf command
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # bm25 tf command
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    # bm25 search command
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")


    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")
            print()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            results = bm25_search_command(args.query, args.limit)
            for i, (movie, score) in enumerate(results, start=1):
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
