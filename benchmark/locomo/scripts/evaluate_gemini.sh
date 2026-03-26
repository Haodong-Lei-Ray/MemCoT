# sets necessary environment variables
source scripts/env.sh

# Evaluate Gemini Pro
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model gemini-pro-1.0 --batch-size 20
