"""Small reproducible RAG retriever. Default: local TF-IDF, no LLM required."""

from dataclasses import asdict, dataclass
from pathlib import Path
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = PROJECT_ROOT / "knowledge"


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    id: str
    source: str
    heading: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_passages(folder: Path = DEFAULT_DOCS) -> list[Passage]:
    """Split each Markdown document by headings to retain source references."""
    if not folder.exists():
        raise FileNotFoundError(f"Папка базы знаний не найдена: {folder}")
    passages: list[Passage] = []
    for file in sorted(folder.glob("*.md")):
        heading = file.stem
        lines: list[str] = []
        count = 0

        def flush() -> None:
            nonlocal count
            body = "\n".join(lines).strip()
            if body:
                count += 1
                passages.append(Passage(
                    id=f"{file.name}:{count}", source=file.name,
                    heading=heading, text=body,
                ))

        for line in file.read_text(encoding="utf-8").splitlines():
            if line.startswith("# ") or line.startswith("## "):
                flush()
                lines.clear()
                heading = line.lstrip("# ").strip()
            else:
                lines.append(line)
        flush()
    if not passages:
        raise ValueError("База знаний пуста")
    return passages


class Retriever:
    def __init__(self, folder: Path = DEFAULT_DOCS, backend: str | None = None):
        self.passages = load_passages(folder)
        self.backend = backend or os.getenv("RETRIEVER", "tfidf")
        texts = [f"{p.heading}\n{p.text}" for p in self.passages]
        if self.backend == "tfidf":
            self.encoder = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.encoder.fit_transform(texts)
        elif self.backend == "embeddings":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Установите pip install -r requirements-embeddings.txt"
                ) from exc
            self.encoder = SentenceTransformer(os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ))
            self.matrix = self.encoder.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError("RETRIEVER должен быть tfidf или embeddings")

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        if not query.strip():
            return []
        if not 1 <= limit <= 8:
            raise ValueError("limit должен быть от 1 до 8")
        if self.backend == "tfidf":
            query_vec = self.encoder.transform([query])
            scores = cosine_similarity(query_vec, self.matrix)[0]
        else:
            query_vec = self.encoder.encode([query], normalize_embeddings=True)[0]
            scores = np.asarray(self.matrix @ query_vec)
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchHit(p.id, p.source, p.heading, p.text, round(float(scores[i]), 4))
            for i in indices
            if scores[i] > 0
            for p in [self.passages[int(i)]]
        ]
