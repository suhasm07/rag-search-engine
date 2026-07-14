import json
import os
import string
from typing import Any, TypedDict

from nltk.stem import PorterStemmer


class Movie(TypedDict):
    id: int
    title: str
    description: str


DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
_STOPWORDS: set[str] | None = None
_STEMMER = PorterStemmer()

#TF Saturation
BM25_K1 = 1.5
# Document length normalization
BM25_B = 0.75

def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def remove_punctuation(text: str) -> str:
    return text.translate(_PUNCTUATION_TABLE)

def tokenize(text: str) -> list[str]:
    cleaned = remove_punctuation(text.lower())
    tokens = cleaned.split()
    return [t for t in tokens if t]

def load_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        with open(STOPWORDS_PATH, "r") as f:
            raw_words = f.read().splitlines()
        _STOPWORDS = {remove_punctuation(w.lower()) for w in raw_words if w}
    return _STOPWORDS

def remove_stopwords(tokens: list[str]) -> list[str]:
    stopwords = load_stopwords()
    return [t for t in tokens if t not in stopwords]

def stem_tokens(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(t) for t in tokens]

def tokenize_text(text: str) -> list[str]:
    return stem_tokens(remove_stopwords(tokenize(text)))




SCORE_PRECISION = 3

def format_search_result(
    doc_id: int,
    title: str,
    document: str,
    score: float,
    **metadata,
) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
