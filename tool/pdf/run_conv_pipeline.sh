#!/usr/bin/env bash
# 批量运行 JSON->PDF 和 PDF->图片 流程，支持所有 conv-xx

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/petrelfs/leihaodong/ICML/locomo/data}"
CON_DIR="${DATA_ROOT}/con"
# SKIP_EXISTING=1 时，JSON->PDF 阶段跳过已存在的 PDF
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 用法：./run_conv_pipeline.sh [conv-id1] [conv-id2] ...
# 若不传参，则自动处理 con/ 下所有 conv-*.json（排除 locomo10pure.json）
# SKIP_EXISTING=1 ./run_conv_pipeline.sh  启用跳过已存在 PDF

if [ $# -gt 0 ]; then
    CONV_IDS=("$@")
else
    # 自动扫描 conv-*.json
    CONV_IDS=()
    for f in "${CON_DIR}"/conv-*.json; do
        [ -f "$f" ] || continue
        base=$(basename "$f" .json)
        CONV_IDS+=("$base")
    done
    if [ ${#CONV_IDS[@]} -eq 0 ]; then
        echo "未找到 ${CON_DIR}/conv-*.json，请手动指定 conv-id"
        exit 1
    fi
fi

echo "将处理: ${CONV_IDS[*]}"
echo "DATA_ROOT=$DATA_ROOT"
echo "SKIP_EXISTING=$SKIP_EXISTING"
echo "---"

SKIP_ARG=""
[ "$SKIP_EXISTING" = "1" ] || [ "$SKIP_EXISTING" = "true" ] || [ "$SKIP_EXISTING" = "yes" ] && SKIP_ARG="--skip-existing"

for conv_id in "${CONV_IDS[@]}"; do
    # echo "[$conv_id] Step 1: JSON -> PDF"
    # python "${SCRIPT_DIR}/json_conv26_to_pdf.py" --conv-id "$conv_id" --data-root "$DATA_ROOT" $SKIP_ARG || exit 1

    echo "[$conv_id] Step 2: PDF -> Images"
    # 显式指定 pdf_dir，避免和默认目录不一致导致扫不到 pdf
    python "${SCRIPT_DIR}/pdf_to_images_conv26.py" --conv-id "$conv_id" --data-root "$DATA_ROOT" \
        --pdf-dir "${DATA_ROOT}/pdf_minor/${conv_id}" || exit 1

    echo "[$conv_id] 完成"
    echo "---"
done

echo "全部完成。"
