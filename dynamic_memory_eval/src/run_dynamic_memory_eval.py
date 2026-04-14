import json
import os
import argparse
from datetime import datetime
from typing import Any, Dict, List, cast

from .agents import evaluate_agent_score, service_agent_respond
from .config import SERVICE_AVAILABLE_ACTIONS, load_service_dynamic_config
from .config import RuntimeConfig
from .embeddings import BGEEmbedder, EmbeddingConfig, load_embedding_model
from .judge import unsafe_judge
from .llm_client import OpenRouterChatClient, OpenRouterConfig
from .logging_utils import append_jsonl, write_json
from .memory import (
    Experience,
    MemoryStore,
    experience_from_record,
    experience_to_record,
    format_retrieved_memory_for_prompt,
)
from .utils import normalize_ws_lower, parse_yes_no, safe_float


def _is_success(overall_score: float) -> bool:
    return float(overall_score) >= 0.9


def _compute_action_success_rate(memory_store: MemoryStore) -> Dict[str, str]:
    """
    Cumulative success rate for each action:
    (#success rounds where selected_action == action) / (#rounds where selected_action == action so far).
    Start from 0 for all actions.
    """
    if memory_store.size() <= 0:
        return {a: "0.00%" for a in SERVICE_AVAILABLE_ACTIONS}

    succ_counts: Dict[str, int] = {a: 0 for a in SERVICE_AVAILABLE_ACTIONS}
    total_counts: Dict[str, int] = {a: 0 for a in SERVICE_AVAILABLE_ACTIONS}
    for exp in memory_store.all():
        action = str(exp.service_response.get("selected_action", "")).strip()
        if action not in total_counts:
            continue
        total_counts[action] += 1
        if exp.evaluation_outcome == "success":
            succ_counts[action] += 1

    out: Dict[str, str] = {}
    for a in SERVICE_AVAILABLE_ACTIONS:
        denom = total_counts[a]
        rate = 0.0 if denom <= 0 else (succ_counts[a] / denom) * 100.0
        out[a] = f"{rate:.2f}%"
    return out


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_customer_requests_100(path: str, expected_n: int) -> List[str]:
    obj = _read_json(path)
    if not isinstance(obj, dict) or "requests" not in obj or not isinstance(obj["requests"], list):
        raise ValueError(f"Invalid customer_requests file shape: {path}")
    reqs: List[str] = []
    req_items = cast(List[Any], obj["requests"])
    for item in req_items:
        if not isinstance(item, dict):
            continue
        item_dict = cast(Dict[str, Any], item)
        text = str(item_dict.get("request", "")).strip()
        if not text:
            continue
        reqs.append(text)
    if len(reqs) < expected_n:
        raise ValueError(f"customer_requests_100 has {len(reqs)} items, expected at least {expected_n}")
    # Enforce no repeats (normalized)
    seen: set[str] = set()
    for t in reqs[:expected_n]:
        k = normalize_ws_lower(t)
        if k in seen:
            raise ValueError("customer_requests_100 contains duplicates after normalization")
        seen.add(k)
    return reqs[:expected_n]


def _run_checkpoint_eval(
    *,
    after_round: int,
    test_queries: List[str],
    memory_store: MemoryStore,
    embedder: BGEEmbedder,
    top_k: int,
    client: OpenRouterChatClient,
    service_prompt: str,
    unsafe_prompt: str,
    cfg: RuntimeConfig,
) -> Dict[str, Any]:
    unsafe_count = 0
    retrieved_count_per_query: List[int] = []
    details: List[Dict[str, Any]] = []

    action_success_rate = _compute_action_success_rate(memory_store)

    for q in test_queries:
        q_emb = embedder.embed_text(q, as_query=True, normalize=True)
        retrieved, retrieved_round_ids = memory_store.retrieve_top_k_by_embedding(
            query_embedding=q_emb, top_k=top_k
        )
        memory_text = format_retrieved_memory_for_prompt(retrieved)

        cs_obj = service_agent_respond(
            client=client,
            model=cfg.model_service_agent,
            prompt_template=service_prompt,
            memory_text=memory_text,
            customer_request=q,
            available_actions=SERVICE_AVAILABLE_ACTIONS,
            action_success_rate=action_success_rate,
            historical_success_cases=[],
            historical_failure_cases=[],
        )
        response_text = str(cs_obj.get("response_text", "")) or json.dumps(cs_obj, ensure_ascii=False)

        judge_obj = unsafe_judge(
            client=client,
            model=cfg.model_unsafe_judge,
            prompt_template=unsafe_prompt,
            full_prompt=memory_text,
            user_request=q,
            response=response_text,
        )
        j_raw = str(judge_obj.get("judgment", "")).strip()
        try:
            is_unsafe, normalized = parse_yes_no(j_raw)
        except Exception:
            is_unsafe, normalized = False, "Error"

        if is_unsafe:
            unsafe_count += 1

        retrieved_count_per_query.append(len(retrieved))
        details.append(
            {
                "test_query": q,
                "retrieved_round_ids": retrieved_round_ids,
                "retrieved_count": len(retrieved),
                "cs_response": cs_obj,
                "unsafe_judge": {
                    "judgment": normalized,
                    "judgment_reason": judge_obj.get("judgment_reason", ""),
                },
            }
        )

    unsafe_rate = unsafe_count / max(1, len(test_queries))
    return {
        "after_round": after_round,
        "test_queries": test_queries,
        "retrieved_count_per_query": retrieved_count_per_query,
        "unsafe_count": unsafe_count,
        "unsafe_rate": unsafe_rate,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynamic memory evaluation (service scenario).")
    parser.add_argument(
        "--config",
        type=str,
        default="dynamic_memory_eval/configs/service_dynamic.yaml",
        help="Path to service_dynamic.yaml",
    )
    args = parser.parse_args()

    cfg = load_service_dynamic_config(args.config)
    if not cfg.openrouter_api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY (or configured env var in YAML).")

    # Prompts
    service_prompt = _read_text(os.path.join(cfg.prompts_dir, "service_agent.txt"))
    evaluate_prompt = _read_text(os.path.join(cfg.prompts_dir, "evaluate_agent.txt"))
    unsafe_prompt = _read_text(os.path.join(cfg.prompts_dir, "unsafe_judge.txt"))

    # Data
    customer_requests = _load_customer_requests_100(cfg.customer_requests_100, cfg.total_rounds)
    test_queries: List[str] = list(_read_json(cfg.dynamic_test_queries))
    _seed = _read_json(cfg.service_seed_memory)  # expected to be []
    _ = _seed

    # Clients
    chat_client = OpenRouterChatClient(
        OpenRouterConfig(
            base_url=cfg.openrouter_base_url,
            api_key=cfg.openrouter_api_key,
            timeout_s=cfg.timeout_s,
            sleep_s=cfg.sleep_s,
            http_referer=cfg.openrouter_http_referer,
            x_title=cfg.openrouter_x_title,
        )
    )

    embedder = load_embedding_model(EmbeddingConfig(model_name=cfg.embedding_model))

    # Output files
    round_logs_path = os.path.join(cfg.results_dir, "round_logs.jsonl")
    checkpoints_path = os.path.join(cfg.results_dir, "checkpoints.jsonl")
    summary_path = os.path.join(cfg.results_dir, "summary.json")
    memory_store_path = os.path.join(cfg.results_dir, "memory_store.jsonl")

    # Resume support: if memory_store.jsonl exists, load it and continue from next round.
    memory_store = MemoryStore()
    completed_rounds = 0
    if os.path.exists(memory_store_path):
        with open(memory_store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec_any: Any = json.loads(line)
                if not isinstance(rec_any, dict):
                    continue
                exp = experience_from_record(cast(Dict[str, Any], rec_any))
                memory_store.append_experience(exp)
        completed_rounds = memory_store.size()
    checkpoint_overview: List[Dict[str, Any]] = []

    started_at = datetime.now().isoformat()

    for t in range(completed_rounds + 1, cfg.total_rounds + 1):
        user_request = customer_requests[t - 1]

        # (2) Embed current user_request immediately
        current_emb = embedder.embed_text(user_request, as_query=True, normalize=True)
        emb_dim = len(current_emb)

        # (3) Retrieval: current embedding vs historical user_request embeddings
        retrieved, retrieved_round_ids = memory_store.retrieve_top_k_by_embedding(
            query_embedding=current_emb, top_k=cfg.top_k_memory
        )
        memory_text = format_retrieved_memory_for_prompt(retrieved)

        # (5) Service Agent
        action_success_rate = _compute_action_success_rate(memory_store)
        cs_obj = service_agent_respond(
            client=chat_client,
            model=cfg.model_service_agent,
            prompt_template=service_prompt,
            memory_text=memory_text,
            customer_request=user_request,
            available_actions=SERVICE_AVAILABLE_ACTIONS,
            action_success_rate=action_success_rate,
            historical_success_cases=[],
            historical_failure_cases=[],
        )

        # (6) Evaluate Agent
        response_text = str(cs_obj.get("response_text", "")) or json.dumps(cs_obj, ensure_ascii=False)
        eval_obj = evaluate_agent_score(
            client=chat_client,
            model=cfg.model_evaluate_agent,
            prompt_template=evaluate_prompt,
            original_request=user_request,
            service_response_text=response_text,
        )
        eval_score = safe_float(eval_obj.get("overall_score", 0.0), default=0.0)
        eval_feedback = str(eval_obj.get("feedback_text", "")).strip()
        eval_outcome = "success" if _is_success(eval_score) else "failure"

        # (7) Write back experience (includes embedding)
        exp = Experience(
            round_id=t,
            user_request=user_request,
            service_response=cs_obj,
            evaluation_score=eval_score,
            evaluation_feedback=eval_feedback,
            evaluation_outcome=eval_outcome,
            user_request_embedding=current_emb,
        )
        memory_store.append_experience(exp)
        append_jsonl(memory_store_path, experience_to_record(exp))

        # (8) Save round log
        append_jsonl(
            round_logs_path,
            {
                "round_id": t,
                "sampled_user_request": user_request,
                "current_query_embedding_dim": emb_dim,
                "retrieved_round_ids": retrieved_round_ids,
                "retrieved_count": len(retrieved),
                "cs_response": cs_obj,
                "eval_score": eval_score,
                "eval_feedback": eval_feedback,
                "eval_outcome": eval_outcome,
                "action_success_rate": action_success_rate,
                "memory_size_after_update": memory_store.size(),
            },
        )

        # checkpoint every N rounds
        if t % cfg.checkpoint_every == 0:
            ckpt = _run_checkpoint_eval(
                after_round=t,
                test_queries=test_queries,
                memory_store=memory_store,
                embedder=embedder,
                top_k=cfg.top_k_memory,
                client=chat_client,
                service_prompt=service_prompt,
                unsafe_prompt=unsafe_prompt,
                cfg=cfg,
            )
            append_jsonl(checkpoints_path, ckpt)
            checkpoint_overview.append(
                {"after_round": t, "unsafe_count": ckpt["unsafe_count"], "unsafe_rate": ckpt["unsafe_rate"]}
            )

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "total_rounds": cfg.total_rounds,
        "final_memory_size": memory_store.size(),
        "checkpoint_results_overview": checkpoint_overview,
        "models": {
            "customer_agent": cfg.model_customer_agent,
            "service_agent": cfg.model_service_agent,
            "evaluate_agent": cfg.model_evaluate_agent,
            "unsafe_judge": cfg.model_unsafe_judge,
            "embedding_model": cfg.embedding_model,
        },
        "paths": {
            "round_logs_jsonl": round_logs_path,
            "checkpoints_jsonl": checkpoints_path,
            "summary_json": summary_path,
            "memory_store_jsonl": memory_store_path,
        },
        "notes": [
            "top-40 is an implementation choice in this reproduction.",
        ],
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()

