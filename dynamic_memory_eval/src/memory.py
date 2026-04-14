from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .embeddings import BGEEmbedder
from .utils import truncate_text


@dataclass(frozen=True)
class Experience:
    """
    A single memory unit is a full experience record.
    We embed ONLY the user_request (per Dynamic 2.0 instruction).
    """

    round_id: int
    user_request: str
    service_response: Dict[str, Any]
    evaluation_score: float
    evaluation_feedback: str
    evaluation_outcome: str  # "success" or "failure"
    user_request_embedding: List[float]


class MemoryStore:
    """
    Retrieval-based memory store.
    Retrieval is: query embedding -> cosine similarity vs historical user_request embeddings
    -> retrieve top-40 experiences -> pack full experiences into prompt.
    """

    def __init__(self, seed: Sequence[Experience] | None = None):
        self._items: List[Experience] = list(seed or [])

    def size(self) -> int:
        return len(self._items)

    def append_experience(self, exp: Experience) -> None:
        self._items.append(exp)

    def all(self) -> List[Experience]:
        return list(self._items)

    def retrieve_top_k_by_query(
        self,
        *,
        query_text: str,
        embedder: BGEEmbedder,
        top_k: int = 40,
    ) -> Tuple[List[Experience], List[int]]:
        """
        Implements the required pipeline:
        - embed query_text (BGE)
        - cosine similarity vs historical user_request embeddings
        - sort desc, take top_k
        - return full experiences (not just matched queries)

        Similarity:
        - We L2-normalize vectors in embedder, so cosine == dot product.
        """
        if top_k <= 0 or not self._items:
            return [], []

        q = embedder.embed_text(query_text, as_query=True, normalize=True)

        sims: List[Tuple[float, int]] = []
        for idx, exp in enumerate(self._items):
            h = exp.user_request_embedding
            # dot product (cosine if vectors are normalized)
            score = 0.0
            for a, b in zip(q, h):
                score += a * b
            sims.append((score, idx))

        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[: min(top_k, len(sims))]
        indices = [i for _, i in top]
        exps = [self._items[i] for i in indices]
        return exps, [e.round_id for e in exps]

    def retrieve_top_k_by_embedding(
        self,
        *,
        query_embedding: List[float],
        top_k: int = 40,
    ) -> Tuple[List[Experience], List[int]]:
        """
        Same as retrieve_top_k_by_query, but uses a provided query embedding.
        Assumes embeddings are L2-normalized, so cosine == dot product.
        """
        if top_k <= 0 or not self._items:
            return [], []

        q = query_embedding
        sims: List[Tuple[float, int]] = []
        for idx, exp in enumerate(self._items):
            h = exp.user_request_embedding
            score = 0.0
            for a, b in zip(q, h):
                score += a * b
            sims.append((score, idx))

        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[: min(top_k, len(sims))]
        indices = [i for _, i in top]
        exps = [self._items[i] for i in indices]
        return exps, [e.round_id for e in exps]


def format_retrieved_memory_for_prompt(
    retrieved: Sequence[Experience],
    *,
    max_response_chars: int = 240,
    max_feedback_chars: int = 200,
) -> str:
    """
    Format top-40 retrieved experiences as a compact memory context block.
    Keeps core fields; lightly truncates long text to avoid prompt bloat.
    """
    if not retrieved:
        return "No retrieved memory."

    blocks: List[str] = []
    for i, exp in enumerate(retrieved, start=1):
        resp_text = truncate_text(exp.service_response.get("response_text", ""), max_response_chars)
        reasoning = truncate_text(exp.service_response.get("reasoning", ""), 160)
        selected_action = exp.service_response.get("selected_action", exp.service_response.get("selected_actions", ""))
        blocks.append(
            "\n".join(
                [
                    f"[Memory #{i}]",
                    f"Round: {exp.round_id}",
                    f"User Request: {exp.user_request}",
                    f"Service Selected Action: {selected_action}",
                    f"Service Reasoning: {reasoning}",
                    f"Service Response: {resp_text}",
                    f"Evaluation Score: {exp.evaluation_score}",
                    f"Evaluation Feedback: {truncate_text(exp.evaluation_feedback, max_feedback_chars)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def experience_to_record(exp: Experience) -> Dict[str, Any]:
    """
    Serialize an Experience into a JSON-serializable dict for persistence (jsonl).
    """
    return {
        "round_id": exp.round_id,
        "user_request": exp.user_request,
        "service_response": exp.service_response,
        "evaluation_score": exp.evaluation_score,
        "evaluation_feedback": exp.evaluation_feedback,
        "evaluation_outcome": exp.evaluation_outcome,
        "user_request_embedding": exp.user_request_embedding,
    }


def experience_from_record(rec: Dict[str, Any]) -> Experience:
    """
    Deserialize an Experience record from persisted jsonl.
    """
    return Experience(
        round_id=int(rec["round_id"]),
        user_request=str(rec["user_request"]),
        service_response=dict(rec["service_response"]),
        evaluation_score=float(rec["evaluation_score"]),
        evaluation_feedback=str(rec["evaluation_feedback"]),
        evaluation_outcome=str(rec.get("evaluation_outcome", "failure")),
        user_request_embedding=[float(x) for x in list(rec["user_request_embedding"])],
    )

