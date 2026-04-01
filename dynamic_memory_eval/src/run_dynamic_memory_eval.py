import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .agents import CustomerAgent, EvaluateAgent, LLMClient, ServiceAgent, UnsafeJudge
from .config import (
    Config,
    SERVICE_ACTION_SUCCESS_RATE,
    SERVICE_AVAILABLE_ACTIONS,
    SERVICE_HISTORICAL_FAILURE_CASES,
    SERVICE_HISTORICAL_SUCCESS_CASES,
)
from .judge import parse_yes_no
from .logging_utils import append_jsonl, write_json
from .memory import InteractionMemory, MemoryItem, format_memory_for_prompt


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def run_checkpoint_eval(
    service_agent: ServiceAgent,
    unsafe_judge: UnsafeJudge,
    memory_text: str,
    test_queries: List[str],
    cfg: Config,
) -> Tuple[int, float, List[Dict[str, Any]]]:
    unsafe_count = 0
    judged_details: List[Dict[str, Any]] = []

    for q in test_queries:
        # Service agent response for the fixed query
        cs_obj = service_agent.respond(
            memory_text=memory_text,
            customer_request=q,
            available_actions=SERVICE_AVAILABLE_ACTIONS,
            action_success_rate=SERVICE_ACTION_SUCCESS_RATE,
            historical_success_cases=SERVICE_HISTORICAL_SUCCESS_CASES,
            historical_failure_cases=SERVICE_HISTORICAL_FAILURE_CASES,
        )
        response_text = str(cs_obj.get("response_text", "")) or json.dumps(cs_obj, ensure_ascii=False)

        # Unsafe judge: "Yes" means memory-affected inappropriate decision (unsafe)
        judge_obj = unsafe_judge.judge(
            full_prompt=memory_text,
            user_request=q,
            response=response_text,
        )
        judgment_raw = str(judge_obj.get("judgment", "")).strip()
        try:
            is_unsafe, normalized = parse_yes_no(judgment_raw)
            if is_unsafe:
                unsafe_count += 1
        except Exception:
            # Treat as judge error; do not crash the run.
            is_unsafe = False
            normalized = "Error"

        judged_details.append(
            {
                "test_query": q,
                "cs_response": cs_obj,
                "unsafe_judge": {
                    "judgment": normalized,
                    "judgment_reason": judge_obj.get("judgment_reason", ""),
                },
            }
        )

    unsafe_rate = unsafe_count / max(1, len(test_queries))
    return unsafe_count, unsafe_rate, judged_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynamic memory evaluation (service scenario only).")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--test-queries-path", type=str, default="dynamic_memory_eval/data/dynamic_test_queries.json")
    parser.add_argument("--request-pool-path", type=str, default="dynamic_memory_eval/data/service_request_pool.json")
    parser.add_argument("--seed-memory-path", type=str, default="dynamic_memory_eval/data/service_seed_memory.json")
    parser.add_argument("--prompts-dir", type=str, default="dynamic_memory_eval/prompts")
    parser.add_argument("--results-dir", type=str, default="dynamic_memory_eval/results")
    parser.add_argument("--memory-k", type=int, default=None, help="Override MEMORY_K (default 6).")
    args = parser.parse_args()

    cfg = Config()
    if not cfg.api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY. Please export OPENROUTER_API_KEY before running.")

    memory_k = args.memory_k if args.memory_k is not None else cfg.memory_k

    # Load prompts
    customer_prompt = _read_text(os.path.join(args.prompts_dir, "customer_agent.txt"))
    service_prompt = _read_text(os.path.join(args.prompts_dir, "service_agent.txt"))
    evaluate_prompt = _read_text(os.path.join(args.prompts_dir, "evaluate_agent.txt"))
    unsafe_prompt = _read_text(os.path.join(args.prompts_dir, "unsafe_judge.txt"))

    # Load data
    test_queries: List[str] = list(_read_json(args.test_queries_path))
    request_pool_obj = _read_json(args.request_pool_path)
    request_pool_json = json.dumps(request_pool_obj, ensure_ascii=False, indent=2)
    seed_memory_obj = _read_json(args.seed_memory_path)
    _ = seed_memory_obj  # default is empty; kept for future extensions

    # Init agents
    client = LLMClient(cfg)
    customer_agent = CustomerAgent(client, customer_prompt, cfg.backbone_model)
    service_agent = ServiceAgent(client, service_prompt, cfg.backbone_model)
    evaluate_agent = EvaluateAgent(client, evaluate_prompt, cfg.backbone_model)
    unsafe_judge = UnsafeJudge(client, unsafe_prompt, cfg.unsafe_judge_model)

    # Output paths
    round_logs_path = os.path.join(args.results_dir, "round_logs.jsonl")
    checkpoints_path = os.path.join(args.results_dir, "checkpoints.jsonl")
    summary_path = os.path.join(args.results_dir, "summary.json")

    # Memory (starts empty per requirement)
    memory = InteractionMemory()

    # Run rounds
    start_ts = datetime.now().isoformat()
    for t in range(1, args.rounds + 1):
        # Step 1: Customer Agent generates a user request
        user_request = customer_agent.sample_request(request_pool_json=request_pool_json)

        # Step 2: Service Agent responds conditioned on memory
        memory_text = format_memory_for_prompt(memory, max_items=memory_k)
        cs_obj = service_agent.respond(
            memory_text=memory_text,
            customer_request=user_request,
            available_actions=SERVICE_AVAILABLE_ACTIONS,
            action_success_rate=SERVICE_ACTION_SUCCESS_RATE,
            historical_success_cases=SERVICE_HISTORICAL_SUCCESS_CASES,
            historical_failure_cases=SERVICE_HISTORICAL_FAILURE_CASES,
        )

        response_text = str(cs_obj.get("response_text", "")) or json.dumps(cs_obj, ensure_ascii=False)

        # Step 3: Evaluate Agent outputs score + feedback_text (reward signal)
        eval_obj = evaluate_agent.evaluate(original_request=user_request, service_response_text=response_text)
        score = _safe_float(eval_obj.get("overall_score", 0.0), default=0.0)
        feedback_text = str(eval_obj.get("feedback_text", "")).strip()

        # Step 4: Update memory
        memory.append(
            MemoryItem(
                round_id=t,
                user_request=user_request,
                agent_response=cs_obj,
                score=score,
                feedback_text=feedback_text,
            )
        )

        append_jsonl(
            round_logs_path,
            {
                "round_id": t,
                "sampled_user_request": user_request,
                "cs_response": cs_obj,
                "eval_score": score,
                "eval_feedback": feedback_text,
                "memory_size_after_update": memory.size(),
            },
        )

        # Every N rounds: checkpoint evaluation
        if t % args.checkpoint_interval == 0:
            memory_text_ckpt = format_memory_for_prompt(memory, max_items=memory_k)
            unsafe_count, unsafe_rate, judged_details = run_checkpoint_eval(
                service_agent=service_agent,
                unsafe_judge=unsafe_judge,
                memory_text=memory_text_ckpt,
                test_queries=test_queries,
                cfg=cfg,
            )

            append_jsonl(
                checkpoints_path,
                {
                    "after_round": t,
                    "test_queries": test_queries,
                    "unsafe_count": unsafe_count,
                    "unsafe_rate": unsafe_rate,
                    "details": judged_details,
                },
            )

    # Final summary
    summary = {
        "started_at": start_ts,
        "finished_at": datetime.now().isoformat(),
        "rounds": args.rounds,
        "checkpoint_interval": args.checkpoint_interval,
        "memory_k": memory_k,
        "models": {
            "backbone_model": cfg.backbone_model,
            "unsafe_judge_model": cfg.unsafe_judge_model,
        },
        "paths": {
            "round_logs_jsonl": round_logs_path,
            "checkpoints_jsonl": checkpoints_path,
            "summary_json": summary_path,
        },
    }
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()

