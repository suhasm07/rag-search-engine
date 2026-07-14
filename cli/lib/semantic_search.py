import os
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies

EMBEDDINGS_PATH = "cache/movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = "cache/chunk_embeddings.npy"
CHUNK_METADATA_PATH = "cache/chunk_metadata.json"


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm1)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace")
        result = self.model.encode([text])
        return result[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        import os
        os.makedirs("cache", exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        import os
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings)== len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        query_embedding = self.generate_embedding(query)
        scored = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored.append((score, self.documents[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:limit]:
            results.append({
                "score" : score,
                "title" : doc["title"],
                "description" : doc["description"],
            })
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for doc in self.documents:
            description = doc.get("description", "")
            if not description.strip():
                continue
            chunks = semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)
            movie_idx = self.documents.index(doc)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        from .search_utils import format_search_result, SCORE_PRECISION
        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            meta = self.chunk_metadata[i]
            chunk_scores.append({
            "chunk_idx": meta["chunk_idx"],
            "movie_idx": meta["movie_idx"],
            "score": score,
            })

        # Keep best chunk score per movie
        movie_scores: dict[int, dict] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in movie_scores or chunk_score["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = chunk_score

        sorted_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)
        top_movies = sorted_movies[:limit]

        results = []
        for entry in top_movies:
            doc = self.documents[entry["movie_idx"]]
            result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"][:100],
                score=entry["score"],
            )
            results.append(result)
        return results


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    from .search_utils import load_movies
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


#CHUNKING
def chunk_text(text: str, chunk_size: int = 200, overlap: int=0) -> list[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        # Skip if this chunk only contains words already seen in the previous overlap
        if i > 0 and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks

# Semantic chunking
def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # if only one text and doesn't end with punctuation, keep whole text
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    # Strip each sentence and filter empty ones
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than max_chunk_size ({max_chunk_size})")
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + max_chunk_size]
        if not chunk_sentences:
            break
        if i > 0 and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks
