import re
from typing import Any, Dict

from .llm_client import OpenRouterChatClient
from .utils import extract_json_object


def _unescape_json_string_inner(inner: str) -> str:
    """Decode JSON string contents (the part between quotes), including truncated tail."""
    out: list[str] = []
    i = 0
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            e = inner[i + 1]
            if e == "n":
                out.append("\n")
            elif e == "r":
                out.append("\r")
            elif e == "t":
                out.append("\t")
            elif e in '"\\/':
                out.append(e)
            elif e == "u" and i + 5 < len(inner):
                try:
                    out.append(chr(int(inner[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except Exception:
                    out.append(inner[i])
                    out.append(inner[i + 1])
                    i += 2
                    continue
            else:
                out.append(inner[i])
                out.append(inner[i + 1])
            i += 2
            continue
        out.append(inner[i])
        i += 1
    return "".join(out)


def _parse_json_string_value(raw: str, start: int) -> tuple[str, int]:
    """Read a JSON string value starting at opening quote at raw[start]. Unterminated string runs to EOF."""
    if start >= len(raw) or raw[start] != '"':
        return "", start
    i = start + 1
    chunks: list[str] = []
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            chunks.append(raw[i : i + 2])
            i += 2
            continue
        if c == '"':
            return _unescape_json_string_inner("".join(chunks)), i + 1
        chunks.append(c)
        i += 1
    return _unescape_json_string_inner("".join(chunks)), i


def _fallback_parse_unsafe_judge(raw: str) -> Dict[str, Any]:
    """
    When the model wraps JSON in prose, truncates output, or emits slightly invalid JSON,
    still recover judgment + reason for checkpoint logging.
    """
    jm = re.search(r'"judgment"\s*:\s*"(Yes|No)"', raw, flags=re.IGNORECASE)
    if not jm:
        jm = re.search(r'"judgment"\s*:\s*"(Yes|No)', raw, flags=re.IGNORECASE)
    if not jm:
        raise ValueError(f"unsafe_judge: no judgment field in model output: {raw[:500]!r}")
    judgment = jm.group(1).strip().capitalize()

    rm = re.search(r'"judgment_reason"\s*:\s*"', raw)
    reason = ""
    if rm:
        reason, _ = _parse_json_string_value(raw, rm.end() - 1)
    return {"judgment": judgment, "judgment_reason": reason}


def unsafe_judge(
    *,
    client: OpenRouterChatClient,
    model: str,
    prompt_template: str,
    full_prompt: str,
    user_request: str,
    response: str,
) -> Dict[str, Any]:
    """
    Unsafe judge used ONLY in checkpoint evaluation.
    Output MUST be JSON: {"judgment":"Yes|No","judgment_reason":"..."}
    """
    prompt = (
        prompt_template.replace("{full_prompt}", full_prompt)
        .replace("{user_request}", user_request)
        .replace("{response}", response)
    )
    raw = client.generate(model=model, user_prompt=prompt, temperature=0.0, max_completion_tokens=4096)
    try:
        obj = extract_json_object(raw)
    except ValueError:
        obj = _fallback_parse_unsafe_judge(raw)
    client.sleep()
    return obj

