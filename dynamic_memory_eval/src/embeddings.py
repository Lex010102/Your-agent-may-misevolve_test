from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Embedding config for BAAI/bge-large-en-v1.5.
    This embedding is local (not OpenRouter chat completion).
    """

    model_name: str = "BAAI/bge-large-en-v1.5"
    use_fp16: bool = True
    query_instruction_for_retrieval: str = "Represent this sentence for searching relevant passages:"


class BGEEmbedder:
    """
    Minimal wrapper around SentenceTransformers for BGE models.
    We use local HuggingFace/SentenceTransformers (not OpenRouter chat completion),
    as required by Dynamic 2.0 when OpenRouter embedding is unavailable.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        # Lazy import to keep startup light if embeddings aren't used.
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model: Any = SentenceTransformer(cfg.model_name)

    def embed_texts(self, texts: List[str], *, as_query: bool = False, normalize: bool = True) -> List[List[float]]:
        """
        Returns list[list[float]] of shape (n, dim).
        If normalize=True, embeddings are L2-normalized, so dot product == cosine similarity.
        """
        if as_query:
            texts_in = [f"{self.cfg.query_instruction_for_retrieval} {t}" for t in texts]
        else:
            texts_in = texts

        # sentence-transformers can normalize embeddings internally
        vecs = self._model.encode(
            texts_in,
            normalize_embeddings=normalize,
            convert_to_numpy=False,
            show_progress_bar=False,
        )

        out: List[List[float]] = []
        for v in list(vecs):
            out.append([float(x) for x in list(v)])
        return out

    def embed_text(self, text: str, *, as_query: bool = True, normalize: bool = True) -> List[float]:
        arr = self.embed_texts([text], as_query=as_query, normalize=normalize)
        return arr[0]


_global_embedder: Optional[BGEEmbedder] = None


def load_embedding_model(cfg: Optional[EmbeddingConfig] = None) -> BGEEmbedder:
    """
    Cached loader for the embedding model.
    """
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = BGEEmbedder(cfg or EmbeddingConfig())
    return _global_embedder

