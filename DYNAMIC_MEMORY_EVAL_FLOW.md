## Dynamic memory evaluation：主流程与 checkpoint 流程（含用到的文件）

本文档总结当前仓库里 **dynamic memory evaluation（service/customer service 场景）** 的两条执行路径：
- **普通主流程**：跑满 100 轮（或配置的 `total_rounds`）
- **Checkpoint 流程**：每隔 N 轮（默认 10 轮）对固定 10 条 test queries 计算一次 unsafe rate

所有流程的主入口脚本是 `dynamic_memory_eval/src/run_dynamic_memory_eval.py`。

---

## 1. 普通主流程（100 轮）

### 1.1 入口与配置

- **入口脚本**：`dynamic_memory_eval/src/run_dynamic_memory_eval.py`
- **配置文件**：`dynamic_memory_eval/configs/service_dynamic.yaml`
- **配置加载**：`dynamic_memory_eval/src/config.py`（`load_service_dynamic_config`）

配置中决定：
- `total_rounds`（默认 100）
- `checkpoint_every`（默认 10）
- `top_k_memory`（固定 top-40；实现选择）
- 模型名（Service/Evaluate/UnsafeJudge）与 embedding 模型名
- 路径：results 输出目录、customer requests 文件、test queries 文件、prompts 目录等

### 1.2 每轮 round t 的执行步骤（严格顺序）

> 下面每一步同时列出“用到的文件/模块”。

#### (1) 取出本轮 user_request（严格按 1..100 顺序，不重复）
- **数据文件**：`dynamic_memory_eval/generated/customer_requests_100.json`
- **读取逻辑**：`dynamic_memory_eval/src/run_dynamic_memory_eval.py`（`_load_customer_requests_100`）

#### (2) 对当前 user_request 做 embedding（BAAI/bge-large-en-v1.5）
- **embedding 封装**：`dynamic_memory_eval/src/embeddings.py`
- **模型名来源**：`dynamic_memory_eval/configs/service_dynamic.yaml` → `dynamic_memory_eval/src/config.py`
- **调用点**：`dynamic_memory_eval/src/run_dynamic_memory_eval.py`（每轮 `embedder.embed_text(...)`）

约束：
- **只对 `user_request` 做 embedding**
- embedding 是本地模型（HuggingFace/SentenceTransformers），不走 OpenRouter chat completion

#### (3) 用 current_query_embedding 做相似度检索，取 top-40 historical experiences
- **Memory store**：`dynamic_memory_eval/src/memory.py`（`MemoryStore`）
- **检索函数**：`MemoryStore.retrieve_top_k_by_embedding(...)`
- **相似度**：cosine similarity（实现方式：向量 L2-normalize 后用 dot product 排序，越大越相似）
- **retrieval depth**：`top_k_memory`（默认 40）

#### (4) 将 top-40 完整 experiences 格式化为 memory context
- **格式化函数**：`dynamic_memory_eval/src/memory.py`（`format_retrieved_memory_for_prompt`）
- **内容**：每条 memory 含 Round / User Request / Service Response / Eval Score / Eval Feedback 等核心字段
- **轻量截断**：为避免 prompt 过长，会对 response/feedback 做字符级截断（不删核心字段）

#### (5) Service Agent 读取 current request + retrieved memory context，输出 JSON response
- **prompt 文件**：`dynamic_memory_eval/prompts/service_agent.txt`
- **agent 封装**：`dynamic_memory_eval/src/agents.py`（`service_agent_respond`）
- **LLM client（统一 OpenRouter chat）**：`dynamic_memory_eval/src/llm_client.py`（`OpenRouterChatClient.generate`）

Service Agent 输出格式（严格 JSON）：
```json
{
  "selected_action": "...",
  "reasoning": "...",
  "response_text": "..."
}
```

#### (6) Evaluate Agent 读取 original_request + service_response，输出 reward-like feedback（JSON）
- **prompt 文件**：`dynamic_memory_eval/prompts/evaluate_agent.txt`
- **agent 封装**：`dynamic_memory_eval/src/agents.py`（`evaluate_agent_score`）
- **LLM client**：`dynamic_memory_eval/src/llm_client.py`

Evaluate Agent 输出格式（严格 JSON）：
```json
{
  "overall_score": 0.0,
  "feedback_text": "..."
}
```

#### (7) 将本轮完整 experience 写入 memory store（包含 embedding）
- **experience 结构**：`dynamic_memory_eval/src/memory.py`（`Experience`）
- **写回**：`MemoryStore.append_experience(...)`

memory record 至少包含：
- `round_id`
- `user_request`
- `service_response`
- `evaluation_score`
- `evaluation_feedback`
- `user_request_embedding`（list[float]）

#### (8) 保存本轮结构化日志（jsonl）
- **落盘工具**：`dynamic_memory_eval/src/logging_utils.py`（`append_jsonl`）
- **输出文件**：`<results_dir>/round_logs.jsonl`

`round_logs.jsonl` 每行至少包含：
- `round_id`
- `sampled_user_request`
- `current_query_embedding_dim`
- `retrieved_round_ids`
- `retrieved_count`
- `cs_response`
- `eval_score`
- `eval_feedback`
- `memory_size_after_update`

---

## 2. Checkpoint 流程（每 10 轮一次）

### 2.1 触发点
- **触发逻辑**：`dynamic_memory_eval/src/run_dynamic_memory_eval.py`
  - `if t % checkpoint_every == 0: ...`

### 2.2 固定 10 条 test queries
- **数据文件**：`dynamic_memory_eval/data/dynamic_test_queries.json`

### 2.3 对每条 test query 的执行步骤

对每条 `test_query`，执行以下流程（与主流程的 retrieval 部分一致）：

1) **test query embedding**
- `dynamic_memory_eval/src/embeddings.py`

2) **retrieval top-40 experiences**
- `dynamic_memory_eval/src/memory.py`（只基于历史 `user_request_embedding`）

3) **Service Agent 基于 test query + retrieved memory 生成 response**
- prompt：`dynamic_memory_eval/prompts/service_agent.txt`
- agent：`dynamic_memory_eval/src/agents.py`
- client：`dynamic_memory_eval/src/llm_client.py`

4) **Unsafe Judge 判断是否 unsafe（Yes/No）**
- **prompt 文件**：`dynamic_memory_eval/prompts/unsafe_judge.txt`
- **judge 实现**：`dynamic_memory_eval/src/judge.py`（`unsafe_judge`）
- **LLM client**：`dynamic_memory_eval/src/llm_client.py`

Unsafe Judge 输出格式（严格 JSON）：
```json
{
  "judgment": "Yes",
  "judgment_reason": "..."
}
```

> 注意：Unsafe Judge **不写回 memory**，只用于 checkpoint safety evaluation。

### 2.4 checkpoint 统计与落盘
- `unsafe_count`
- `unsafe_rate = unsafe_count / 10`
- `retrieved_count_per_query`

输出文件：
- **`<results_dir>/checkpoints.jsonl`**
  - 由 `dynamic_memory_eval/src/logging_utils.py` 写入

---

## 3. 输出文件（results）

由 `configs/service_dynamic.yaml` 的 `paths.results_dir` 控制，至少包含：
- `round_logs.jsonl`
- `checkpoints.jsonl`
- `summary.json`

其中 `summary.json` 由 `dynamic_memory_eval/src/run_dynamic_memory_eval.py` 在 run 结束时写出，包含：
- `total_rounds`
- `final_memory_size`
- `checkpoint_results_overview`

---

## 4. 重要说明

- 本实现显式遵守 retrieval pipeline：
  **current query embedding → match historical user_request embeddings → retrieve top-40 full experiences → pack into prompt**
- `top-40` 在代码与配置中固定为实现选择，并在配置注释与 summary notes 中标注：
  **“top-40 is an implementation choice in this reproduction.”**

