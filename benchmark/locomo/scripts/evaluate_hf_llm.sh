# sets necessary environment variables
source scripts/env.sh


# run models
for MODEL in llama3-chat-70b llama3-chat-70b mistral-instruct-7b-32k-v2; do
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
        --model $MODEL --use-4bit --batch-size 1
done
