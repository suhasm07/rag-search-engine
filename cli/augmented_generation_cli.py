import argparse

from lib.hybrid_search import HybridSearch, get_llm_client
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    citations_parser = subparsers.add_parser(
        "citations", help="Answer query with cited sources"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    question_parser = subparsers.add_parser("question", help="Answer a question conversationally")
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results")


    args = parser.parse_args()

    match getattr(args, "command", None):
        case "rag":
            query = args.query

            # Load movies and run RRF search
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=5)

            # Print search results
            print("Search Results:")
            for result in results:
                print(f"- {result['title']}")

            # Build docs string for prompt
            docs = "\n".join(
                [f"- {r['title']}: {r['document']}" for r in results]
            )

            # Build prompt
            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

            # Call LLM
            client = get_llm_client()
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            print("\nRAG Response:")
            print(response.choices[0].message.content.strip())

        case "summarize":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            # Build results string for prompt
            results_str = "\n".join(
                [f"- {r['title']}: {r['document']}" for r in results]
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results_str}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            client = get_llm_client()
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            print("\nLLM Summary:")
            print(response.choices[0].message.content.strip())


        case "citations":
            query = args.query
            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            # Number the documents so LLM can cite them as [1], [2], etc.
            documents = "\n".join(
                [f"[{i+1}] {r['title']}: {r['document']}"
                 for i, r in enumerate(results)]
            )

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            client = get_llm_client()
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            print("\nLLM Answer:")
            print(response.choices[0].message.content.strip())


        case "question":
            question = args.question
            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.rrf_search(question, k=60, limit=args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            context = "\n".join(
                [f"- {r['title']}: {r['document']}" for r in results]
            )

            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            client = get_llm_client()
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )
            print("\nAnswer:")
            print(response.choices[0].message.content.strip())


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
