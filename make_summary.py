#!/usr/bin/env python3
"""
Generate summary.json from existing round_logs.jsonl / checkpoints.jsonl
(without re-running the eval). Run from repo root:

  python make_summary.py
  python make_summary.py --results-dir dynamic_memory_eval/results_full_100
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write summary.json for a results directory.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("dynamic_memory_eval", "results_full_100"),
        help="Directory containing round_logs.jsonl and checkpoints.jsonl",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help='Value for summary "rounds" (configured target; may exceed completed rounds)',
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help='Value for summary "checkpoint_interval"',
    )
    args = parser.parse_args()

    os.chdir(_repo_root())

    from dynamic_memory_eval.src.config import Config

    cfg = Config()
    results_dir = args.results_dir
    round_logs_path = os.path.join(results_dir, "round_logs.jsonl")
    checkpoints_path = os.path.join(results_dir, "checkpoints.jsonl")
    summary_path = os.path.join(results_dir, "summary.json")

    if not os.path.isfile(round_logs_path):
        raise SystemExit(f"Missing {round_logs_path}")

    with open(round_logs_path, "r", encoding="utf-8") as f:
        log_lines = [ln for ln in f if ln.strip()]

    completed = len(log_lines)
    last_round_id = None
    if log_lines:
        try:
            last_round_id = json.loads(log_lines[-1]).get("round_id")
        except json.JSONDecodeError:
            pass

    last_ckpt_after = None
    if os.path.isfile(checkpoints_path):
        with open(checkpoints_path, "r", encoding="utf-8") as f:
            ckpt_lines = [ln for ln in f if ln.strip()]
        if ckpt_lines:
            try:
                last_ckpt_after = json.loads(ckpt_lines[-1]).get("after_round")
            except json.JSONDecodeError:
                pass

    st = os.stat(round_logs_path)
    started_ts = getattr(st, "st_birthtime", st.st_mtime)

    summary: Dict[str, Any] = {
        "started_at": datetime.fromtimestamp(started_ts).isoformat(),
        "finished_at": datetime.now().isoformat(),
        "rounds": args.rounds,
        "checkpoint_interval": args.checkpoint_interval,
        "memory_k": cfg.memory_k,
        "models": {
            "backbone_model": cfg.backbone_model,
            "unsafe_judge_model": cfg.unsafe_judge_model,
        },
        "paths": {
            "round_logs_jsonl": round_logs_path,
            "checkpoints_jsonl": checkpoints_path,
            "summary_json": summary_path,
        },
        "completed_rounds": completed,
        "last_round_id_in_logs": last_round_id,
        "last_after_round_in_checkpoints": last_ckpt_after,
        "status": "complete" if completed >= args.rounds else "partial",
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {summary_path}")
    print(f"  completed_rounds={completed}, target rounds={args.rounds}, status={summary['status']}")


if __name__ == "__main__":
    main()
