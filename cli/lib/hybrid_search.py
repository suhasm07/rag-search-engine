import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)




class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        fetch_limit = limit * 500

        bm25_results = self.semantic_search.search_chunks(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        # Normalize BM25 scores
        bm25_scores = [r["score"] for r in bm25_results]
        norm_bm25 = normalize_scores(bm25_scores)

        # Normalize semantic scores
        sem_scores = [r["score"] for r in semantic_results]
        norm_sem = normalize_scores(sem_scores)

        # build combined doc map
        doc_map: dict[int, dict] = {}

        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            doc_map[doc_id] = {
                "result": result,
                "bm25_score": norm_bm25[i],
                "sem_score": 0.0,
            }

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id in doc_map:
                doc_map[doc_id]["sem_score"] = norm_sem[i]
            else:
                doc_map[doc_id] = {
                    "result": result,
                    "bm25_score": 0.0,
                    "sem_score": norm_sem[i],
                }


        # Calculate hybrid scores
        for doc_id, entry in doc_map.items():
            entry["hybrid_score"] = (
                alpha * entry["bm25_score"] + (1 - alpha) * entry["sem_score"]
            )

        # Sort by hybrid score descending
        sorted_docs = sorted(
            doc_map.values(), key=lambda x: x["hybrid_score"], reverse=True
        )


        # Format results
        results = []
        for entry in sorted_docs[:limit]:
            result = entry["result"]
            results.append({
                "id": result["id"],
                "title": result["title"],
                "document": result["document"],
                "hybrid_score": entry["hybrid_score"],
                "bm25_score": entry["bm25_score"],
                "sem_score": entry["sem_score"],
            })

        return results


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        fetch_limit = limit * 500

        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        doc_map: dict[int, dict] = {}

        # Process BM25 results with ranks
        # bm25_search returns (movie_dict, score) tuples
        for rank, (doc, score) in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = {
                "result": {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc.get("description", "")[:100],
                    "score": score,
                },
                "bm25_rank": rank,
                "sem_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        # Process semantic results with ranks, summing RRF if already seen
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id in doc_map:
                doc_map[doc_id]["sem_rank"] = rank
                doc_map[doc_id]["rrf_score"] += rrf_score(rank, k)
            else:
                doc_map[doc_id] = {
                    "result": result,
                    "bm25_rank": None,
                    "sem_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }

        sorted_docs = sorted(
            doc_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        results = []
        for entry in sorted_docs[:limit]:
            result = entry["result"]
            results.append({
                "id": result["id"],
                "title": result["title"],
                "document": result["document"],
                "rrf_score": entry["rrf_score"],
                "bm25_rank": entry["bm25_rank"],
                "sem_rank": entry["sem_rank"],
            })

        return results







# LLM
def get_llm_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def enhance_query_spell(query: str) -> str:
    client = get_llm_client()
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def enhance_query_rewrite(query: str) -> str:
    client = get_llm_client()
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def enhance_query_expand(query: str) -> str:
    client = get_llm_client()
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    expanded_terms = response.choices[0].message.content.strip()
    return f"{query} {expanded_terms}"



# RERANKING
def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    client = get_llm_client()
    reranked = []
    for doc in results:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        try:
            score = float(raw)
        except ValueError:
            score = 0.0
        reranked.append({**doc, "rerank_score": score})
        time.sleep(3)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# Reranking batch
def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    client = get_llm_client()

    # Build the document list string for the prompt
    doc_list_str = "\n".join(
        [f"ID: {doc['id']} | {doc.get('title', '')} - {doc.get('document', '')}"
         for doc in results]
    )

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    try:
        ranked_ids = json.loads(raw)
    except json.JSONDecodeError:
        # If parsing fails, return original order
        return results

    # Build a map from doc ID to result
    id_to_result = {doc["id"]: doc for doc in results}

    # Sort results by the ranked order, assign rerank_rank
    reranked = []
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in id_to_result:
            reranked.append({**id_to_result[doc_id], "rerank_rank": rank})

    # Append any docs not mentioned by LLM at the end
    ranked_id_set = set(ranked_ids)
    for doc in results:
        if doc["id"] not in ranked_id_set:
            reranked.append({**doc, "rerank_rank": len(reranked) + 1})

    return reranked


# Rerank CROSS ENCODER
def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    from sentence_transformers import CrossEncoder

    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
    scores = cross_encoder.predict(pairs)

    reranked = []
    for i, doc in enumerate(results):
        reranked.append({**doc, "cross_encoder_score": float(scores[i])})

    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked
