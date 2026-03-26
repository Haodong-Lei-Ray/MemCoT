import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import pickle
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from global_methods import set_openai_key
from task_eval.rag_utils import get_embeddings


def extract_keywords(text):
    """
    从文本中提取关键词
    使用简单的规则：提取名词、重要动词和形容词
    """
    # 转换为小写并移除标点
    text_lower = text.lower()
    # 移除常见的停用词
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # 提取单词（至少3个字符）
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    # 过滤停用词
    keywords = [w for w in words if w not in stop_words]
    # 去重但保持顺序
    seen = set()
    unique_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique_keywords.append(w)
    return unique_keywords


def build_keyword_index(contexts):
    """
    构建关键词倒排索引
    返回: {keyword: [indices]}
    """
    keyword_index = defaultdict(list)
    for idx, context in enumerate(contexts):
        keywords = extract_keywords(context)
        for keyword in keywords:
            keyword_index[keyword].append(idx)
    return keyword_index


def main():
    # 设置OpenAI API key
    set_openai_key()
    
    # 输入文件路径
    input_file = '/mnt/petrelfs/leihaodong/ICML/locomo/data/multimodal_dialog/example/locomo10_observation.json'
    # 输出目录
    output_dir = '/mnt/petrelfs/leihaodong/ICML/exp/memory'
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取observation数据
    print(f"读取observation数据: {input_file}")
    with open(input_file, 'r') as f:
        observations_data = json.load(f)
    
    # 处理每个sample
    all_contexts = []
    all_dia_ids = []
    all_date_times = []
    
    for sample in tqdm(observations_data, desc="处理samples"):
        sample_id = sample['sample_id']
        
        # 提取所有observations
        contexts = []
        dia_ids = []
        date_times = []
        
        # 遍历所有session
        for key in sample.keys():
            if 'observation' in key:
                session_num = key.split('_')[1]  # session_1_observation -> 1
                date_time_key = f'session_{session_num}_date_time'
                
                # 获取日期时间（如果存在）
                date_time = sample.get(date_time_key, '')
                
                observation_dict = sample[key]
                for speaker, obs_list in observation_dict.items():
                    for obs_item in obs_list:
                        if isinstance(obs_item, list) and len(obs_item) > 0:
                            context_text = obs_item[0]
                            context_dia_ids = obs_item[1:] if len(obs_item) > 1 else []
                            
                            contexts.append(context_text)
                            dia_ids.append(','.join(context_dia_ids) if context_dia_ids else '')
                            date_times.append(date_time)
        
        # 生成embeddings
        print(f"为sample {sample_id}生成embeddings ({len(contexts)}条observations)...")
        if len(contexts) > 0:
            # 使用日期时间+observation作为输入
            inputs = [f"{dt}: {ctx}" if dt else ctx for dt, ctx in zip(date_times, contexts)]
            embeddings = get_embeddings('openai', inputs, 'context')
            # 确保embeddings形状正确
            if embeddings.shape[0] != len(contexts):
                print(f"Warning: embeddings shape {embeddings.shape} doesn't match contexts length {len(contexts)}")
                # 如果embeddings是1D，需要reshape
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
                # 如果embeddings数量不匹配，只取前len(contexts)个
                if embeddings.shape[0] > len(contexts):
                    embeddings = embeddings[:len(contexts)]
                elif embeddings.shape[0] < len(contexts):
                    print(f"Error: Not enough embeddings generated!")
                    continue
            
            # 构建关键词索引
            print(f"构建关键词索引...")
            keyword_index = build_keyword_index(contexts)
            
            # 保存数据库
            database = {
                'embeddings': embeddings,
                'context': contexts,
                'dia_id': dia_ids,
                'date_time': date_times,
                'keyword_index': dict(keyword_index)  # 转换为普通dict以便序列化
            }
            
            output_file = os.path.join(output_dir, f'locomo10_observation_{sample_id}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(database, f)
            
            print(f"已保存到: {output_file}")
            print(f"  - embeddings shape: {embeddings.shape}")
            print(f"  - contexts数量: {len(contexts)}")
            print(f"  - 关键词数量: {len(keyword_index)}")
            
            # 同时保存到all_*中用于全局索引（如果需要）
            all_contexts.extend(contexts)
            all_dia_ids.extend(dia_ids)
            all_date_times.extend(date_times)
    
    # 可选：创建全局索引（如果需要跨sample检索）
    if len(all_contexts) > 0:
        print(f"\n创建全局索引 ({len(all_contexts)}条observations)...")
        all_inputs = [f"{dt}: {ctx}" if dt else ctx for dt, ctx in zip(all_date_times, all_contexts)]
        all_embeddings = get_embeddings('openai', all_inputs, 'context')
        all_keyword_index = build_keyword_index(all_contexts)
        
        global_database = {
            'embeddings': all_embeddings,
            'context': all_contexts,
            'dia_id': all_dia_ids,
            'date_time': all_date_times,
            'keyword_index': dict(all_keyword_index)
        }
        
        global_output_file = os.path.join(output_dir, 'locomo10_observation_global.pkl')
        with open(global_output_file, 'wb') as f:
            pickle.dump(global_database, f)
        
        print(f"已保存全局索引到: {global_output_file}")
        print(f"  - embeddings shape: {all_embeddings.shape}")
        print(f"  - 关键词数量: {len(all_keyword_index)}")


if __name__ == '__main__':
    main()
