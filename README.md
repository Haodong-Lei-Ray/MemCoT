# MemCoT

MemCoT combines **ReAct-style reasoning** with **retrieval over long-term memory** (LightRAG graph + vector stores, optional NaiveRAG). The codebase supports **LoCoMo** (per-conversation samples such as `conv-26`) and **LongMemEval** (per-question haystacks under `benchmark/longmemeval/data/`).

---

## 1. Setup

### 1.1 Environment

- **Python**: use a recent 3.10+ environment.
- **Dependencies**: install packages required by the bundled LightRAG tree and the repo (e.g. `openai`, `numpy`, `regex`, `nltk`, `nest_asyncio`, `asyncio` stack). Reference: `mem/LightRAG/requirements-offline.txt` (and related files under `mem/LightRAG/`) as a starting point; extend with whatever `eval_dmsmem.py` / `dmsmem.py` import on your machine.
- **API keys**: set `OPENAI_API_KEY` and, if you use a proxy or Azure-compatible endpoint, `OPENAI_BASE_URL`. LightRAG also reads `OPENAI_API_BASE`; the code mirrors `OPENAI_BASE_URL` into `OPENAI_API_BASE` when the latter is unset.
- **Optional**: LoCoMo helper env vars live in `benchmark/locomo/scripts/env.sh` (copy the pattern; do not commit real secrets).

### 1.2 Repository layout (what you must prepare)

| Piece | Role |
|--------|------|
| `mem/LightRAG/` | LightRAG implementation used for indexing and query |
| `benchmark/locomo/data/locomo10.json` | LoCoMo QA and conversations |
| `benchmark/locomo/data/skip/` | Optional skip lists per `conv-*` (used by evaluation logic for some setups) |
| `benchmark/longmemeval/data/` | LongMemEval JSON (e.g. `longmemeval_s_cleaned.json`, `s_split/…`) |
| **Your chosen `--lightrag-base` directory** | Built indices: one subdirectory per workspace (see below) |

---

## 2. Long-term memory

### 2.1 How memory is stored and loaded

**Layout**

- **LoCoMo**: sample id `conv-26` → workspace folder name `conv26` (hyphens removed). The LightRAG working directory is:

  e.g. `/data/rag_storage/conv26/` when `--lightrag-base` is `/data/rag_storage`

- **LongMemEval**: each item is keyed by a string like `0000_<question_id>`; the build script writes one LightRAG workspace per JSON stem under your output tree (see §3.2).

**Loading at runtime**

- `dmsmem.create_lightrag(working_dir)` expects that directory to **already exist** and contain a built LightRAG store. It initializes storages and is reused for retrieval during the ReAct loop.
- You can point evaluation at your index root with `--lightrag-base /path/to/rag_storage` (see `eval_dmsmem.py` / `eval_dmsmem_longmemeval.py`).

**Building LoCoMo indices**

From the **MemCoT repository root**:

```bash
export OPENAI_API_KEY="your-key"
# optional: export OPENAI_BASE_URL="https://api.example.com/v1"

python benchmark/locomo/task_eval/build_lightrag_locomo.py \
  --conv-id conv-26 \
  -o /path/to/rag_storage/conv26
```

Repeat for each `conv-*` you need. Default embedding/LightRAG settings follow the script (dialog-level chunks, `text-embedding-3-small`, `gpt-4o-mini` for graph extraction in that builder).

**Building LongMemEval indices**

Use `script/build/build_lightrag_longmemeval.py`. Batch drivers under `script/build/longmemeval_script/` (e.g. `vllm_s/run_build_s_te3s_all_0-49.sh`) show the pattern:

- **`--input`**: a directory of per-question JSON files, e.g. `benchmark/longmemeval/data/s_split/0-49`
- **`-o` / output**: e.g. `/path/to/longmemeval/lightrag/te3s`
- **`--embedding-model`**, **`--llm-mode`**, **`--vllm-config`**: set according to your stack (OpenAI API vs local vLLM)

Example shape (adapt paths and `cd` to your clone):

```bash
cd /path/to/MemCoT
python script/build/build_lightrag_longmemeval.py \
  --input benchmark/longmemeval/data/s_split/0-49 \
  -o /path/to/longmemeval/lightrag/te3s \
  --batch-size 50 \
  --embedding-model text-embedding-3-small \
  --llm-mode openai
```

Other shards (`50-99`, `100-149`, …) are covered by sibling `run_build_s_te3s_all_*.sh` scripts in the same folder.

### 2.2 How to run the agent

**Single query (interactive / debug)**

From repo root:

```bash
python dmsmem.py \
  -c conv-26 \
  -m gpt-4o-mini \
  --lightrag-base /path/to/rag_storage \
  --rag-type lightrag \
  --agent-flag 11000 \
  --middle-scale 4 \
  "When did Caroline go to the LGBTQ support group?"
```

Notable flags (see `dmsmem.py`):

- **`--rag-type`**: `lightrag` (default) or `naive`
- **`--lightrag-base`**: root containing per-conversation workspaces
- **`--agent-flag`**: five digits `rag_view / middle_view / full_view / agentic_graph / visual_search` (e.g. `11000` enables RAG + middle view, disables full view and visual search)
- **`--benchmark`**: `locomo` (default) or `longmemeval` for prompt/context layout

**LoCoMo batch evaluation**

`eval_dmsmem.py` runs ReAct+LightRAG over QA in `benchmark/locomo/data/locomo10.json`, computes F1 (including category-5 abstention handling), and can resume from partial JSON.

```bash
python eval_dmsmem.py \
  --sample-id conv-26 \
  --lightrag-base /path/to/rag_storage \
  -m gpt-4o-mini \
  -k 10 \
  --max-step 8 \
  --agent-flag 11000 \
  --middle-scale 4 \
  -o /path/to/eval_output \
  --resume \
  -n -1
```

Use `--skip-category 5` to drop category 5 items. Use `--concurrency N` for parallel QA (async).

**LongMemEval batch evaluation**

```bash
python eval_dmsmem_longmemeval.py \
  --data-file benchmark/longmemeval/data/longmemeval_s_cleaned.json \
  --lightrag-base /path/to/longmemeval/lightrag/te3s \
  -m gpt-4o-mini \
  --start-idx 0 \
  -n 500 \
  -k 10 \
  --max-step 8 \
  --middle-scale 10 \
  --benchmark longmemeval \
  -o /path/to/longmemeval_eval \
  --resume \
  --run-llm-judge
```

Agent LLM routing can use `config/configqwen1.json` via `with_agent_llm_config` in `dmsmem.py`; align that file with your endpoint for the **agent** (separate from LightRAG’s embed/LLM settings if you split providers).

---

## 3. LoCoMo benchmark workflow

Relevant tree: **`benchmark/locomo/`**

1. **Data**: `data/locomo10.json` plus optional `data/skip/conv-*.json` and `data/con/` as needed by your pipeline.
2. **Build memory**: §2.1 — `task_eval/build_lightrag_locomo.py` per `conv-*`.
3. **Evaluate**:
   - **Python**: `eval_dmsmem.py` (§2.2).
   - **Slurm wrappers (example)**: `script/gpt/4o-mini/F2/`
     - `sbatch_eval_react_lightrag.sh` — sets `EVAL_OUT`, `CONV`, then submits `eval_react_lightrag_conv.sh`
     - `eval_react_lightrag_conv.sh` — runs `eval_dmsmem.py` with your `--lightrag-base`, model, `agent-flag`, etc.
     - `command.sh` — loops over multiple conversations
   - **Before running**: edit these scripts so `cd` points to **your** MemCoT clone, paths match your `--lightrag-base` and output dirs, and **remove any hard-coded API keys** (use env vars).

Optional: `benchmark/locomo/scripts/env.sh` documents variables used by older LoCoMo tooling (`DATA_FILE_PATH`, `OPENAI_*`, etc.).

---

## 4. LongMemEval benchmark workflow

### 4.1 Building indices

- **Builder**: `script/build/build_lightrag_longmemeval.py`
- **Batch SLURM / shard drivers**: `script/build/longmemeval_script/vllm_s/` (e.g. `run_build_s_te3s_all_0-49.sh`, `…_50-99.sh`, …) and `vllm/` variants for different splits.
- Each driver sets `DATASET_DIR` to a split under `benchmark/longmemeval/data/s_split/`, `RAG_OUT` to your te3s (or chosen) output root, and invokes the builder with embedding / vLLM options.

Ensure the **`--lightrag-base`** you pass to evaluation matches the **output layout** produced by the builder (per-question workspace names).

### 4.2 Evaluation

- **Main script**: `eval_dmsmem_longmemeval.py` (§2.2).
- **Example Slurm bundle**: `script/longmemeval/gpt/4o-mini/main/`
  - `sbatch_eval_react_lightrag_14b.sh` — creates `EVAL_OUT`, submits `eval_lightrag_longmemeval.sh`
  - `eval_lightrag_longmemeval.sh` — `srun` + `eval_dmsmem_longmemeval.py` with `--data-file`, `--lightrag-base`, `--run-llm-judge`, slice args
  - `command.sh` — example dependency chain for many small jobs (adjust to your scheduler policy)
  - `sum.sh` / `sumre.sh` / `sumw.sh` — aggregation helpers for logs or metrics (inspect locally for your workflow)

Again, replace hard-coded `cd`, paths, and secrets in shell scripts with your environment.

---

## 5. Notes

- **Vision / OCR**: if `agent-flag` enables visual bits, you may need image directories under `data/img_pdf_minor/<conv-id>/…` and, for visual *search*, indices under `--img-index-base` (see `dmsmem.py` / `agent/agent.py`).
- **Naming**: LoCoMo `conv-26` → directory `conv26` under `--lightrag-base` (`_conv_id_to_workspace` in `dmsmem.py`).
