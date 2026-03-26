# sets necessary environment variables
source scripts/env.sh

# gets observations using gpt-3.5-turbo and extract DRAGON embeddings for RAG database
python task_eval/get_facts.py --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$OBS_OUTPUT_FILE \
    --prompt-dir $PROMPT_DIR --emb-dir $EMB_DIR --use-date --overwrite