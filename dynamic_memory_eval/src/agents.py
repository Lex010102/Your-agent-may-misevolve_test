import json
from typing import Any, Dict, List

from .llm_client import OpenRouterChatClient
from .utils import extract_json_object, render_prompt_template


def service_agent_respond(
    *,
    client: OpenRouterChatClient,
    model: str,
    prompt_template: str,
    memory_text: str,
    customer_request: str,
    available_actions: List[str],
    action_success_rate: Dict[str, str],
    historical_success_cases: List[Dict[str, Any]],
    historical_failure_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = render_prompt_template(
        prompt_template,
        {
            "memory": memory_text,
            "customer_request": customer_request,
            "available_actions": json.dumps(available_actions, ensure_ascii=False),
            "action_success_rate": json.dumps(action_success_rate, ensure_ascii=False, indent=2),
            "historical_success_cases": json.dumps(historical_success_cases, ensure_ascii=False, indent=2),
            "historical_failure_cases": json.dumps(historical_failure_cases, ensure_ascii=False, indent=2),
        },
    )
    raw = client.generate(model=model, user_prompt=prompt, temperature=0.7, max_completion_tokens=1024)
    obj = extract_json_object(raw)
    client.sleep()
    return obj


def evaluate_agent_score(
    *,
    client: OpenRouterChatClient,
    model: str,
    prompt_template: str,
    original_request: str,
    service_response_text: str,
) -> Dict[str, Any]:
    prompt = render_prompt_template(
        prompt_template,
        {
            "original_request": original_request,
            "service_response": service_response_text,
        },
    )
    raw = client.generate(model=model, user_prompt=prompt, temperature=0.0, max_completion_tokens=512)
    obj = extract_json_object(raw)
    client.sleep()
    return obj

