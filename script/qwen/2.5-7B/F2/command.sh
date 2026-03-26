#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-26 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-30 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-43 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-44 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-47 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-49 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-50 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-41 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-42 10
"${SCRIPT_DIR}/sbatch_eval_react_lightrag.sh" conv-48 10