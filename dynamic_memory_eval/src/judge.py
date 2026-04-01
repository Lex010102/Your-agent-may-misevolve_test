import json
import re
from typing import Any, Dict, Tuple, cast


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of the first JSON object from LLM output.
    Accepts fenced code blocks and surrounding text.
    """
    try:
        obj_any = json.loads(text)
        if isinstance(obj_any, dict):
            return cast(Dict[str, Any], obj_any)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            obj_any = json.loads(candidate)
            if isinstance(obj_any, dict):
                return cast(Dict[str, Any], obj_any)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        obj_any = json.loads(candidate)
        if isinstance(obj_any, dict):
            return cast(Dict[str, Any], obj_any)

    raise ValueError(f"Could not parse JSON from model output: {text[:500]!r}")


def parse_yes_no(judgment: str) -> Tuple[bool, str]:
    j = judgment.strip().lower()
    if j in {"yes", "y", "true"}:
        return True, "Yes"
    if j in {"no", "n", "false"}:
        return False, "No"
    raise ValueError(f"Unexpected judgment value: {judgment!r}")

