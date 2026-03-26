# save generated outputs to this location
OUT_DIR=./outputs

# save embeddings to this location
EMB_DIR=./outputs

# path to LoCoMo data file
DATA_FILE_PATH=./data/locomo10.json

# filenames for different outputs
QA_OUTPUT_FILE=locomo10_qa.json
OBS_OUTPUT_FILE=locomo10_observation.json
SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# path to folder containing prompts and in-context examples
PROMPT_DIR=./prompt_examples

# OpenAI API Key
# export OPENAI_API_KEY="sk-a3iAhoJ4w30ykivaD6PSn97QdRoPsIRQ2zbNawJBYPtPyMLX"
# export OPENAI_API_KEY="sk-SL2cgXG9HXPxJ2KRAQFBrmbcd2odqU5hXszr0EkwCQNfx6Xr"
# export OPENAI_API_KEY="sk-qyXyGlik3PnDddF68mOXbaA3OqdguG7PAkD8sUiwJHYfv44U"
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1"     # 或其他地址

# Google API Key
export GOOGLE_API_KEY=

# Anthropic API Key
export ANTHROPIC_API_KEY=

# HuggingFace Token
export HF_TOKEN=
