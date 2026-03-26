#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/m"
DST_DIR="/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/m_split"
NPARTS=5

RESET=0
DRY_RUN=0

usage() {
  echo "Usage: bash $0 [--reset] [--dry-run]"
  echo "  --reset   清理 DST_DIR 下本次要写入的分目录（0-...~4-...）"
  echo "  --dry-run 只打印分配情况，不复制文件"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset) RESET=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "SRC_DIR 不存在: $SRC_DIR"
  exit 1
fi

mkdir -p "$DST_DIR"

mapfile -t files < <(ls -1 "${SRC_DIR}"/*.json 2>/dev/null | sort)
total="${#files[@]}"
if [[ "$total" -eq 0 ]]; then
  echo "没有找到 json 文件: ${SRC_DIR}/*.json"
  exit 1
fi

echo "SRC: $SRC_DIR"
echo "DST: $DST_DIR"
echo "TOTAL files: $total"
echo "NPARTS: $NPARTS"

for ((p=0; p<NPARTS; p++)); do
  start=$(( p * total / NPARTS ))
  end=$(( (p + 1) * total / NPARTS - 1 ))
  if [[ "$start" -gt "$end" ]]; then
    continue
  fi

  dir_name="${start}-${end}"
  out_dir="${DST_DIR}/${dir_name}"
  echo "Part ${p}: idx [${start}, ${end}] -> ${out_dir}"

  if [[ $RESET -eq 1 ]]; then
    rm -rf "$out_dir"
  fi
  mkdir -p "$out_dir"

  if [[ $DRY_RUN -eq 1 ]]; then
    continue
  fi

  for ((i=start; i<=end; i++)); do
    f="${files[$i]}"
    # 只复制文件到对应分目录，不改变文件名
    cp -f "$f" "$out_dir/"
  done
done

echo "完成"

