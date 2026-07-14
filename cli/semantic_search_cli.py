import argparse

from lib.semantic_search import ChunkedSemanticSearch, SemanticSearch, chunk_text, semantic_chunk_text, embed_query_text, embed_text, semantic_chunk_text, verify_embeddings, verify_model
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model is loaded correctly")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a text input")
    embed_parser.add_argument("input_text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies semantically")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed-size pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text on sentence boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences to overlap")

    subparsers.add_parser("embed_chunks", help="Build or load chunk embeddings")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using chunk embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results")


    args = parser.parse_args()


    match getattr(args, "command", None):
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.input_text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            ss = SemanticSearch()
            documents = load_movies()
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                description_preview = result["description"][:100]
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {description_preview}...\n")
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
