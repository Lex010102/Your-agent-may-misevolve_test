import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class OpenRouterConfig:
    base_url: str
    api_key: str
    timeout_s: int = 600
    sleep_s: float = 0.2
    http_referer: str = ""
    x_title: str = ""


class OpenRouterChatClient:
    """
    Single place for OpenRouter (OpenAI-compatible) /chat/completions calls.
    Business logic MUST NOT call requests directly.
    """

    def __init__(self, cfg: OpenRouterConfig):
        self.cfg = cfg

    def generate(
        self,
        *,
        model: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_completion_tokens: int = 1024,
    ) -> str:
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
        if temperature is not None:
            payload["temperature"] = temperature

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

