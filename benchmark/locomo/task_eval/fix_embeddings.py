"""
修复已生成的embeddings文件，只保留正确数量的embeddings
"""
import pickle
import os

memory_dir = '/mnt/petrelfs/leihaodong/ICML/exp/memory'
pkl_files = [f for f in os.listdir(memory_dir) if f.startswith('locomo10_observation_') and f.endswith('.pkl')]

for pkl_file in pkl_files:
    filepath = os.path.join(memory_dir, pkl_file)
    print(f"处理文件: {pkl_file}")
    
    with open(filepath, 'rb') as f:
        db = pickle.load(f)
    
    n_contexts = len(db['context'])
    n_embeddings = db['embeddings'].shape[0]
    
    print(f"  Contexts: {n_contexts}, Embeddings: {n_embeddings}")
    
    if n_embeddings != n_contexts:
        print(f"  修复中: 保留前{n_contexts}个embeddings...")
        db['embeddings'] = db['embeddings'][:n_contexts]
        
        # 重新构建keyword_index以确保一致性
        from build_hybrid_rag_db import build_keyword_index
        db['keyword_index'] = dict(build_keyword_index(db['context']))
        
        with open(filepath, 'wb') as f:
            pickle.dump(db, f)
        
        print(f"  已修复: embeddings shape = {db['embeddings'].shape}")
    else:
        print(f"  无需修复")
