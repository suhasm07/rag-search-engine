import os
import pickle
import math
from collections import Counter

from .search_utils import BM25_B, BM25_K1, CACHE_DIR, load_movies, tokenize_text


INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_length.pkl")


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = INDEX_PATH


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)


    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)


    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)


    def load(self) -> None:
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError(
                "Index files not found. Run 'build' command first."
            )
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTHS_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)


    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf


    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)


    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[dict, float]]:
        query_tokens = tokenize_text(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            total_score = 0.0
            for token in query_tokens:
                total_score += self.bm25(doc_id, token)
            if total_score > 0:
                scores[doc_id] = total_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            results.append((self.docmap[doc_id], score))
        return results
