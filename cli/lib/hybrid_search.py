import os

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


