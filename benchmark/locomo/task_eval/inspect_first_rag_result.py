"""
查看第一个 LoCoMo 问题的 RAG 检索结果
运行: python3 task_eval/inspect_first_rag_result.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import argparse
from global_methods import set_openai_key
from task_eval.rag_utils import get_embeddings
from task_eval.gpt_utils import prepare_for_rag, get_rag_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default='/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json')
    parser.add_argument('--emb-dir', default='/mnt/petrelfs/leihaodong/ICML/exp/memory')
    parser.add_argument('--rag-mode', default='hybridrag', choices=['hybridrag', 'observation', 'dialog'])
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', default='openai')
    args = parser.parse_args()

    set_openai_key()

    # 加载数据
    with open(args.data_file) as f:
        samples = json.load(f)
    data = samples[0]

    # 第一个问题
    first_qa = data['qa'][0]
    first_qa = data['qa'][1]
    question = first_qa['question']
    answer = first_qa.get('answer', 'N/A')
    evidence = first_qa.get('evidence', [])

    print("=" * 60)
    print("第一个 LoCoMo 问题")
    print("=" * 60)
    print(f"Sample ID: {data['sample_id']}")
    print(f"Question: {question}")
    print(f"Ground Truth Answer: {answer}")
    print(f"Evidence (dia_id): {evidence}")
    print()

    # 准备 RAG - 只对第一个问题获取 embedding
    args.data_file = args.data_file
    args.use_rag = True

    # 直接加载 database，只为第一个问题获取 embedding
    if args.rag_mode == 'hybridrag':
        memory_dir = '/mnt/petrelfs/leihaodong/ICML/exp/memory'
        pkl_file = f'{memory_dir}/locomo10_observation_{data["sample_id"]}.pkl'
        with open(pkl_file, 'rb') as f:
            database = pickle.load(f)
        question_embeddings = get_embeddings(args.retriever, [question], 'query')
    else:
        database, question_embeddings = prepare_for_rag(args, data)

    query_vector = question_embeddings[0]
    if hasattr(query_vector, 'numpy'):
        query_vector = query_vector.numpy()
    elif hasattr(query_vector, 'cpu'):
        query_vector = query_vector.cpu().numpy()

    # RAG 检索
    query_context, context_ids = get_rag_context(
        database, query_vector, args,
        query_text=question if args.rag_mode == 'hybridrag' else None
    )

    print("=" * 60)
    print(f"RAG 检索结果 (top_k={args.top_k})")
    print("=" * 60)
    print("检索到的 context_ids:", context_ids)
    print()
    print("检索到的内容:")
    print("-" * 60)
    print(query_context)
    print("-" * 60)


if __name__ == '__main__':
    main()
