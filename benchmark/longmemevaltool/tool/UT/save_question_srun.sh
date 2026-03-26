#!/usr/bin/env bash
# 用 SLURM srun 运行 save_question.py（也可 sbatch 提交）。
#
# 示例:
#   sbatch save_question_srun.sh --all
#   sbatch save_question_srun.sh -i 0
#   sbatch save_question_srun.sh -i 1 --one-based
#   sbatch save_question_srun.sh --all --input /path/to.json --output /path/to/out
#
# 已有交互式分配时:
#   bash save_question_srun.sh --all
#
# 可选环境变量:
#   SLURM_PARTITION  覆盖默认分区（默认 DataFrontier_Knowledge，与 tool/buildrag/script 一致）

#SBATCH --job-name=save_q
#SBATCH --output=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/UT/save_question_%j.out
#SBATCH --error=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/UT/save_question_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -p DataFrontier_Knowledge

set -euo pipefail

# sbatch 会把脚本拷到 /var/spool/slurmd/...，BASH_SOURCE 不在 UT 目录；优先用提交目录或固定路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/save_question.py" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/save_question.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
  else
    SCRIPT_DIR="/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/tool/UT"
  fi
fi
[[ -f "${SCRIPT_DIR}/save_question.py" ]] || {
  echo "找不到 save_question.py，SCRIPT_DIR=${SCRIPT_DIR}" >&2
  exit 1
}
PY="${SCRIPT_DIR}/save_question.py"
LOG_PARENT="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/UT"
mkdir -p "$LOG_PARENT"

PARTITION="${SLURM_PARTITION:-DataFrontier_Knowledge}"

if [[ $# -eq 0 ]]; then
  echo "用法: $0 [-i N | --all] [其它 save_question.py 参数]" >&2
  echo "未传参数时默认执行: --all" >&2
  set -- --all
fi

# 无 SLURM 作业时（登录节点直接 bash 本脚本），srun 需自带分区/资源；sbatch/salloc 内不再重复指定
# 注意: set -u 下空数组 "${SRUN_EXTRA[@]}" 会报 unbound variable，须分支展开
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --partition="$PARTITION" --cpus-per-task=2 --time=02:00:00 python3 "$PY" "$@"
else
  # 已在 sbatch/salloc 分配内：直接 srun 起 step（与 buildrag 脚本一致）
  exec srun python3 "$PY" "$@"
fi
