from .inverted_index import InvertedIndex
from .search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, load_movies, remove_stopwords, stem_tokens, tokenize_text

def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(
            f"Expected exactly one token from '{term}', got {len(tokens)}: {tokens}"
        )
    return tokens[0]

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def tf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    print(idx.get_tf(doc_id, token))



def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    query_tokens = tokenize_text(query)
    seen_ids: set[int] = set()
    results: list[dict] = []

    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(idx.docmap[doc_id])
                if len(results) >= limit:
                    return results
    return results


def idf_command(term: str) -> None:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    idf = idx.get_idf(token)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tfidf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    token = tokenize_term(term)
    tf_idf = idx.get_tfidf(doc_id, token)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf}': {tf_idf:.2f}")


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0
    token = tokenize_term(term)
    return idx.get_bm25_idf(token)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 0.0
    token = tokenize_term(term)
    return idx.get_bm25_tf(doc_id, token, k1, b)


def bm25_search_command(query: str, limit: int = 5) -> list[tuple[dict, float]]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    return idx.bm25_search(query, limit)
