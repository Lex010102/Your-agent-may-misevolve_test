import json
import re
from typing import Any, Dict, cast


def render_prompt_template(template: str, mapping: Dict[str, str]) -> str:
    """
    Replace placeholders like `{memory}` with provided values.
    We avoid Python str.format() because prompts contain JSON examples with `{ ... }`.
    """
    rendered = template
    for k, v in mapping.items():
        rendered = rendered.replace("{" + k + "}", v)
    return rendered


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of the first JSON object from LLM output.
    Accepts fenced code blocks and surrounding text.
    """
    text = text.strip()

    try:
        obj_any = json.loads(text)
        if isinstance(obj_any, dict):
            return cast(Dict[str, Any], obj_any)
    except Exception:
        pass

    # Try every ``` fenced segment (non-greedy can stop too early if a stray ``` appears).
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        candidate = m.group(1).strip()
        try:
            obj_any = json.loads(candidate)
            if isinstance(obj_any, dict):
                return cast(Dict[str, Any], obj_any)
        except Exception:
            pass

    # Opening fence without closing (truncated stream): strip known prefix and parse remainder.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            tail = text[first_nl + 1 :].strip()
            try:
                obj_any = json.loads(tail)
                if isinstance(obj_any, dict):
                    return cast(Dict[str, Any], obj_any)
            except Exception:
                pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj_any = json.loads(candidate)
            if isinstance(obj_any, dict):
                return cast(Dict[str, Any], obj_any)
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from model output: {text[:500]!r}")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_ws_lower(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def parse_yes_no(value: str) -> tuple[bool, str]:
    v = str(value).strip().lower()
    if v in {"yes", "y", "true"}:
        return True, "Yes"
    if v in {"no", "n", "false"}:
        return False, "No"
    raise ValueError(f"Unexpected judgment value: {value!r}")

