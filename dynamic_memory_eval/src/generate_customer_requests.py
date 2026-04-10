import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, cast

from .config import load_service_dynamic_config
from .llm_client import OpenRouterChatClient, OpenRouterConfig
from .utils import extract_json_object, normalize_ws_lower


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate N unique customer requests (request bank).")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--config", type=str, default="dynamic_memory_eval/configs/service_dynamic.yaml")
    parser.add_argument(
        "--output-path",
        type=str,
        default="dynamic_memory_eval/generated/customer_requests_100.json",
    )
    args = parser.parse_args()

    cfg = load_service_dynamic_config(args.config)
    if not cfg.openrouter_api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY (or configured env var in YAML).")

    request_bank_prompt = _read_text(os.path.join(cfg.prompts_dir, "request_bank_agent.txt"))
    request_pool_obj = _read_json("dynamic_memory_eval/data/service_request_pool.json")
    request_pool_json = json.dumps(request_pool_obj, ensure_ascii=False, indent=2)

    client = OpenRouterChatClient(
        OpenRouterConfig(
            base_url=cfg.openrouter_base_url,
            api_key=cfg.openrouter_api_key,
            timeout_s=cfg.timeout_s,
            sleep_s=cfg.sleep_s,
            http_referer=cfg.openrouter_http_referer,
            x_title=cfg.openrouter_x_title,
        )
    )

    # Chunked generation to make "100 unique" achievable.
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    chunk_size = 20
    next_id = 1
    max_chunk_attempts = 8

    while next_id <= args.n:
        end_id = min(args.n, next_id + chunk_size - 1)
        existing_requests_json = json.dumps([it["request"] for it in out], ensure_ascii=False, indent=2)

        expected_ids = list(range(next_id, end_id + 1))

        last_err: Exception | None = None
        for attempt in range(1, max_chunk_attempts + 1):
            try:
                prompt = request_bank_prompt
                prompt = prompt.replace("{n_requests}", str(args.n))
                prompt = prompt.replace("{start_id}", str(next_id))
                prompt = prompt.replace("{end_id}", str(end_id))
                prompt = prompt.replace("{request_pool_json}", request_pool_json)
                prompt = prompt.replace("{existing_requests_json}", existing_requests_json)
                if attempt > 1:
                    prompt += "\n\nIMPORTANT: Regenerate this chunk with entirely NEW requests. Do not reuse any previous phrasing.\n"

                raw = client.generate(
                    model=cfg.model_customer_agent,
                    user_prompt=prompt,
                    temperature=1.0,
                    max_completion_tokens=2048,
                )
                obj = extract_json_object(raw)
                reqs_any: Any = obj.get("requests")
                if not isinstance(reqs_any, list):
                    raise RuntimeError("request_bank_agent output missing 'requests' list")
                reqs = cast(List[Any], reqs_any)

                if len(reqs) != len(expected_ids):
                    raise RuntimeError(f"Expected {len(expected_ids)} items, got {len(reqs)}")

                # Validate the whole chunk first (no partial writes).
                new_items: List[Dict[str, Any]] = []
                new_keys: set[str] = set()
                for expected_id, item_any in zip(expected_ids, reqs, strict=True):
                    if not isinstance(item_any, dict):
                        raise RuntimeError(f"Bad request item: {item_any!r}")
                    item = cast(Dict[str, Any], item_any)
                    if item.get("id") != expected_id:
                        raise RuntimeError(
                            f"Bad id sequence: expected {expected_id}, got {item.get('id')!r}"
                        )
                    text = str(item.get("request", "")).strip()
                    if not text:
                        raise RuntimeError(f"Empty request at id {expected_id}")
                    key = normalize_ws_lower(text)
                    if key in seen or key in new_keys:
                        raise RuntimeError(f"Duplicate request detected at id {expected_id}")
                    new_keys.add(key)
                    new_items.append({"id": expected_id, "request": text})

                # Commit chunk only after full validation.
                for it in new_items:
                    seen.add(normalize_ws_lower(it["request"]))
                    out.append(it)
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise RuntimeError(
                f"Failed to generate unique chunk ids {next_id}..{end_id} after {max_chunk_attempts} attempts: {last_err}"
            ) from last_err

        next_id = end_id + 1

    payload = {
        "generated_at": datetime.now().isoformat(),
        "n": args.n,
        "model": cfg.model_customer_agent,
        "requests": out,
    }

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output_path} ({args.n} unique requests)")


if __name__ == "__main__":
    main()

