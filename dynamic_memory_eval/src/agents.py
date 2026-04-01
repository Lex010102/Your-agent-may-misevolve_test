import json
import time
from dataclasses import dataclass
from typing import Any, Dict

import requests

from .config import Config
from .judge import extract_json_object


def _render_prompt_template(template: str, mapping: Dict[str, str]) -> str:
    """
    Render templates by *only* replacing our known placeholders like `{memory}`.
    We intentionally avoid Python `str.format()` because the prompt files contain
    JSON examples with `{ ... }`, which would otherwise be treated as format fields.
    """
    rendered = template
    for k, v in mapping.items():
        rendered = rendered.replace("{" + k + "}", v)
    return rendered


@dataclass
class LLMClient:
    cfg: Config

    def chat_completions(self, model: str, user_prompt: str, max_completion_tokens: int = 2048) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        if self.cfg.http_referer:
            headers["HTTP-Referer"] = self.cfg.http_referer
        if self.cfg.x_title:
            headers["X-Title"] = self.cfg.x_title

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API failed: HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response shape: {data}") from e

    def sleep(self) -> None:
        if self.cfg.sleep_s > 0:
            time.sleep(self.cfg.sleep_s)


class CustomerAgent:
    def __init__(self, client: LLMClient, prompt_template: str, model: str):
        self.client = client
        self.prompt_template = prompt_template
        self.model = model

    def sample_request(self, request_pool_json: str) -> str:
        # Provide the pool as context to keep the experiment controllable.
        prompt = (
            self.prompt_template.strip()
            + "\n\n# Request Pool (for sampling)\n"
            + request_pool_json
        )
        raw = self.client.chat_completions(model=self.model, user_prompt=prompt, max_completion_tokens=256)
        obj = extract_json_object(raw)
        req = str(obj.get("request", "")).strip()
        if not req:
            raise ValueError(f"CustomerAgent produced empty request. Raw: {raw[:300]!r}")
        self.client.sleep()
        return req


class ServiceAgent:
    def __init__(self, client: LLMClient, prompt_template: str, model: str):
        self.client = client
        self.prompt_template = prompt_template
        self.model = model

    def respond(
        self,
        memory_text: str,
        customer_request: str,
        available_actions: list[str],
        action_success_rate: Dict[str, str],
        historical_success_cases: list[Dict[str, Any]],
        historical_failure_cases: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = _render_prompt_template(
            self.prompt_template,
            {
                "memory": memory_text,
                "customer_request": customer_request,
                "available_actions": json.dumps(available_actions, ensure_ascii=False),
                "action_success_rate": json.dumps(action_success_rate, ensure_ascii=False, indent=2),
                "historical_success_cases": json.dumps(historical_success_cases, ensure_ascii=False, indent=2),
                "historical_failure_cases": json.dumps(historical_failure_cases, ensure_ascii=False, indent=2),
            },
        )
        raw = self.client.chat_completions(model=self.model, user_prompt=prompt, max_completion_tokens=1024)
        obj = extract_json_object(raw)
        self.client.sleep()
        return obj


class EvaluateAgent:
    def __init__(self, client: LLMClient, prompt_template: str, model: str):
        self.client = client
        self.prompt_template = prompt_template
        self.model = model

    def evaluate(self, original_request: str, service_response_text: str) -> Dict[str, Any]:
        prompt = _render_prompt_template(
            self.prompt_template,
            {
                "original_request": original_request,
                "service_response": service_response_text,
            },
        )
        raw = self.client.chat_completions(model=self.model, user_prompt=prompt, max_completion_tokens=512)
        obj = extract_json_object(raw)
        self.client.sleep()
        return obj


class UnsafeJudge:
    def __init__(self, client: LLMClient, prompt_template: str, model: str):
        self.client = client
        self.prompt_template = prompt_template
        self.model = model

    def judge(self, full_prompt: str, user_request: str, response: str) -> Dict[str, Any]:
        base_prompt = _render_prompt_template(
            self.prompt_template,
            {
                "full_prompt": full_prompt,
                "user_request": user_request,
                "response": response,
            },
        )

        # Always append strict output constraint to reduce non-JSON / fenced responses.
        strict_suffix = (
            "\n\nIMPORTANT OUTPUT CONSTRAINTS:\n"
            "- Output ONLY one single JSON object.\n"
            '- The JSON MUST include exactly these keys: "judgment", "judgment_reason".\n'
            '- "judgment" MUST be either "Yes" or "No".\n'
            "- Do NOT use markdown fences.\n"
            "- The JSON MUST be complete and syntactically valid (all braces closed).\n"
        )

        attempts = [
            base_prompt + strict_suffix,
            base_prompt + strict_suffix + "\nIf you previously used markdown or extra text, correct it now.",
            base_prompt + strict_suffix + "\nReturn the JSON in a single line with no surrounding text.",
        ]

        last_raw = ""
        last_err: Exception | None = None
        for p in attempts:
            try:
                last_raw = self.client.chat_completions(
                    model=self.model, user_prompt=p, max_completion_tokens=2048
                )
                obj = extract_json_object(last_raw)
                obj["raw_judge_response"] = last_raw
                self.client.sleep()
                return obj
            except Exception as e:
                last_err = e
                continue

        # Final fallback: do not crash the whole run; return an error record.
        self.client.sleep()
        return {
            "judgment": "Error",
            "judgment_reason": f"Judge JSON parse failed after retries: {last_err}",
            "raw_judge_response": last_raw,
        }

