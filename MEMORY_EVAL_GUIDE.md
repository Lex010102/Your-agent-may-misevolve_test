# Memory Evaluation (Static + Dynamic) — Project Notes

This repository contains two **memory-misevolution** evaluation tracks:

- **Static evaluation**: evaluate model responses given a *fixed, hand-crafted memory context* (deployment-time reward hacking).
- **Dynamic evaluation**: simulate *memory accumulation over time* with a 100-round interaction loop and periodic unsafe-rate checkpoints.

This document summarizes **what each script does**, **where outputs are saved**, **important parameters**, and **known differences from the paper setup**.

---

## 1) Static memory evaluation

### 1.1 Generate responses (static cases)
- **Script**: `memory_misevolution/reward_hacking_test.py`
- **Purpose**: Calls an OpenAI-compatible Chat Completions API to generate responses for four scenarios:
  - `service`, `sales`, `medical`, `financial`
- **Output**: writes one JSON per scenario under:
  - `outputs/reward_hacking/<MODEL>/<scenario>.json`
  - Example: `outputs/reward_hacking/gpt-5/service.json`

**Key CLI args**
- `--model`: model name passed to the API (e.g., `gpt-5`)
- `--scenario`: optional, one of `service|sales|medical|financial`
- `--output-dir`: base output directory (default is `outputs/reward_hacking`)

### 1.2 LLM-as-judge for static outputs
- **Script**: `outputs/reward_hacking_test/static_LLM_judge.py`
- **Purpose**: Reads the static outputs above and uses a judge model to label each case:
  - whether the assistant is **affected by memory** and makes **inappropriate decisions** (`Yes/No`)
- **Default judge backend**: OpenRouter (OpenAI-compatible endpoint)
- **Output**: writes judged files to:
  - `outputs/reward_hacking/<MODEL>/judged/<scenario>.judged.json`
  - Example: `outputs/reward_hacking/gpt-5/judged/service.judged.json`

**Key CLI args**
- `--input-dir`: directory containing `service.json/sales.json/medical.json/financial.json`
- `--judge-model`: e.g., `gemini-2.5-pro` (OpenRouter model id)
- `--scenarios`: comma-separated list, default all 4
- `--sleep`: delay between judge calls

---

## 2) Dynamic memory evaluation (service only)

### 2.1 Entry point
- **Runner**: `dynamic_memory_eval/src/run_dynamic_memory_eval.py`
- **Purpose**: Minimal dynamic experiment for **service/customer service** only:
  1) Customer Agent generates a user request
  2) Service Agent responds conditioned on current memory
  3) Evaluate Agent outputs a round-level `score` + `feedback_text`
  4) The interaction is appended to memory
  5) Every N rounds (default 10), run checkpoint evaluation on 10 fixed test queries and compute `unsafe_rate`

### 2.2 Prompts and data
- **Prompts**: `dynamic_memory_eval/prompts/`
  - `customer_agent.txt`
  - `service_agent.txt`
  - `evaluate_agent.txt`
  - `unsafe_judge.txt`
- **Data**: `dynamic_memory_eval/data/`
  - `dynamic_test_queries.json` (the 10 fixed test queries)
  - `service_request_pool.json` (request pool to keep customer sampling controllable)
  - `service_seed_memory.json` (currently empty; dynamic starts from empty memory)

### 2.3 Results
The runner writes **structured logs** to the `--results-dir` directory:
- `round_logs.jsonl`: per-round interaction log
- `checkpoints.jsonl`: per-checkpoint unsafe evaluation results
- `summary.json`: run metadata and file paths

Example (smoke test):
- `dynamic_memory_eval/results_smoke/round_logs.jsonl`
- `dynamic_memory_eval/results_smoke/checkpoints.jsonl`
- `dynamic_memory_eval/results_smoke/summary.json`

### 2.4 Important parameters
**Runner CLI**
- `--rounds`: default `100`
- `--checkpoint-interval`: default `10` (**unsafe judge only runs every 10 rounds**, not every round)
- `--memory-k`: number of recent memory items fed into the Service Agent (see below)

**Environment / `.env`**
- The dynamic runner loads environment variables from `.env` automatically (see `dynamic_memory_eval/src/config.py`).
- Create: `dynamic_memory_eval/.env` (copy from `dynamic_memory_eval/.env.example`).

Key variables:
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`
- `BACKBONE_MODEL` (Customer/Service/Evaluate)
- `UNSAFE_JUDGE_MODEL` (Unsafe Judge)
- `MEMORY_K` (default 6)

---

## 3) Differences vs the paper setup (intentional simplifications)

- **Memory window \(K\)**: we use a simple *recent-K* memory formatting, **default `K=6`** (`MEMORY_K`), rather than a more complex retrieval mechanism.
- **Prompts**: Customer/Service/Evaluate/UnsafeJudge prompts are **re-created in this repo** under `dynamic_memory_eval/prompts/` for minimal reproducibility and clarity.
- **Customer simulation**: we provide a controllable request pool (`service_request_pool.json`) instead of fully open-ended user generation.
- **Logging**: outputs are logged as JSONL (`round_logs.jsonl`, `checkpoints.jsonl`) + `summary.json` for easy downstream analysis.
- **API layer**: we use **OpenRouter** (OpenAI-compatible `/chat/completions`) with OpenRouter model IDs (e.g., `qwen/qwen-2.5-72b-instruct`, `google/gemini-2.5-pro`).
- **Goal**: this code targets a **minimal runnable dynamic experiment** that matches the *flow logic* first, not a perfect reproduction of all model-level statistics.

