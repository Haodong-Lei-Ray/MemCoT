# MemCoT LongMemEval index reconstruction

This is a reconstructed builder, validated against one complete released index.
It recovers the observed document format and chunk hashes. The historical
extraction model, generation settings, and long-turn splitting policy are not
fully recoverable from this sample, so this is not a claim of identical paper
results or identical regenerated graphs/embeddings.

## Released indexes and small downloads

The [released dataset](https://huggingface.co/datasets/Rayleihaodong/MemCoT-longmemeval/tree/176d2f5d7be0b9fc5bd131810643ed260d37dd1d)
contains a single `te3s.tar.zst`, 33,717,961,213 bytes (31.40 GiB).
Its HTTP server supports byte ranges. The archive's first index is
`te3s/0484_5809eb10/`; archive order is **not** dataset order.

The first 80 MiB is enough to extract that entire index. No full snapshot or
`git lfs pull` is necessary for format inspection. This command downloads at
most 80 MiB of compressed data and requires only Python 3 and the `zstd` CLI:

```bash
python download_hf_prefix.py --output ./prebuilt-sample --mib 80 --indexes 1
```

It pins the dataset revision, checks the Range response, extracts only regular
files with safe paths, and records a manifest. A truncated last index is not
reported as complete. `--prefix-file PATH` reuses a local compressed prefix
without downloading it again.

The resulting `./prebuilt-sample/te3s` is the `rag_base` for the extracted
question. It does not contain question 0, and cannot support the evaluator's
default first-N run. For all questions, download and extract the whole archive,
or build the missing indexes from the full original dataset.

## Build from original LongMemEval-S

Run these from the MemCoT repository root, adjusting the script path if needed.
Use Python >=3.10 and install this repository's vendored LightRAG rather than
an unrelated newer release:

```bash
pip install -e ./mem/LightRAG 'openai>=2,<3'
```

Obtain `longmemeval_s_cleaned.json` from the
[original dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main).
It is about 277 MB; it is separate from the 33.7 GB prebuilt index archive.
Preparing inputs does not need model packages or API credentials:

```bash
python benchmark/longmemeval/tool/build_lightrag_longmemeval.py prepare \
  --source benchmark/longmemeval/data/longmemeval_s_cleaned.json \
  --output-base ./memory/longmemeval/lightrag/te3s \
  --dataset-dir benchmark/longmemeval/dataset \
  --limit 1
```

Remove `--limit 1` for all questions. `--start-index` and `--question-id` filter
the original list without renumbering directories. A single-entry source file
requires `--index-offset ORIGINAL_INDEX`, preventing accidental renumbering.

Building calls both an extraction LLM and `text-embedding-3-small`. Configure
`OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL` / `OPENAI_API_BASE`. Use
`--llm-base-url` and `--embedding-base-url` for separate compatible endpoints.
The extraction model must be explicitly selected because its historical value
is unknown. For example, **as a new choice rather than a recovered setting**:

```bash
python benchmark/longmemeval/tool/build_lightrag_longmemeval.py build \
  --source benchmark/longmemeval/data/longmemeval_s_cleaned.json \
  --output-base ./memory/longmemeval/lightrag/te3s \
  --dataset-dir benchmark/longmemeval/dataset \
  --llm-model gpt-4o-mini --limit 1
```

Each question uses a separate process to isolate LightRAG shared namespaces.
Re-running with the same inputs and model allows LightRAG checkpoint recovery.
Use a fresh output directory when changing inputs/model; do not mix new builds
into historical index directories. `_BUILD_DONE` is written only after actual
document, chunk, vector-ID, and processed-status checks pass.

The observed format is one complete turn per chunk. Turns exceeding the
embedding API's 8192-token input limit cause an explicit error; the builder does
not silently truncate or invent an unverified historical splitting policy.
Numeric session-position IDs for repeated session IDs follow the surviving
agent parser comments but were not observed in the sampled question.

## Verify an existing index

```bash
python benchmark/longmemeval/tool/build_lightrag_longmemeval.py verify \
  --source benchmark/longmemeval/data/longmemeval_s_cleaned.json \
  --output-base ./memory/longmemeval/lightrag/te3s --limit 1
```

For the downloaded sample use its raw single-row JSON, `--index-offset 484`,
and the extracted `te3s` directory as `--output-base`. Verification needs only
the standard library. It checks coverage and metadata, not semantic graph
quality, regenerated embedding equality, or retrieval metrics.

## What was validated

- Question 484 / `5809eb10`: all 606 original turn IDs, escaped texts, and dates
  match `kv_store_full_docs.json` exactly.
- These contain 600 distinct texts. All 600 MD5 chunk IDs and their stored
  contents match. The six repeated texts explain why 606 documents have 600
  content-addressed chunks.
- The stored token counts match `o200k_base` for all 600 chunks. An unsplit turn
  has 2915 tokens, ruling out blindly using the normal 1200-token chunker.
- Chunk/entity/relationship vectors have 1536 dimensions. The `te3s` name and
  local scripts support choosing `text-embedding-3-small`, but dimensions alone
  cannot mathematically identify the embedding model.
- Cached prompts explicitly specify English and entity types
  `Person, Event, Location, Organization, Activity, Concept, Date, Other`.
- 4601 historical `failed` status records are all marked as duplicate inserts.
  All 606 real document IDs are processed; this is not 4601 missing turns.
- `test_offline_build.py` successfully built two isolated indexes using the real
  vendored LightRAG against a local fake API, including repeated text, embedded
  quotes, and a turn longer than 1200 tokens. No paid model API was called.

The complete archive and all 500 historical indexes were not audited.

## Remaining evaluator integration issues in the checked-out repository

The builder generates `{idx:04d}_{question_id}` directories as required by
`run_eval_in_benchmark.py`. Set `config/rag/longmemevallightrag.json`'s `rag_base`
to the local output, and use the same original dataset order.

The current checkout has additional issues independent of index construction:

1. `agent/conversation.py` still loads `benchmark/LongMemEval/dataset`, whereas
   the tracked benchmark directory is lowercase `benchmark/longmemeval`.
2. It refers to a missing `module_unit/LongMemEval/.../format_history.py` path.
3. Its LongMemEval `full_conv` becomes a formatted string, but
   `MiddleViewAgent.get_context_window4longmemeval` and
   `agent/prompt.py::conv_answer_agent_prompt` require the raw entry dictionary.
4. The shipped run script and RAG config contain author-specific absolute paths.

These must be repaired before advertising a complete evaluator reproduction
command. This reconstruction adds index tools and does not modify that runtime.
