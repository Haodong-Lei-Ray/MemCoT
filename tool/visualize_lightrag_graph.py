#!/usr/bin/env python3
"""将 LightRAG 的 GraphML 图谱可视化为交互式 HTML。"""

import argparse
import os

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError:
    print("请安装依赖: pip install networkx pyvis")
    raise

# 按实体类型着色，便于区分
ENTITY_TYPE_COLORS = {
    "person": "#4CAF50",      # 绿
    "event": "#2196F3",       # 蓝
    "location": "#FF9800",    # 橙
    "organization": "#9C27B0",  # 紫
    "activity": "#00BCD4",    # 青
    "concept": "#E91E63",     # 粉
    "date": "#795548",        # 棕
    "book": "#607D8B",        # 灰蓝
    "artifact": "#3F51B5",    # 靛蓝
    "other": "#9E9E9E",       # 灰
}


def main():
    parser = argparse.ArgumentParser(description="LightRAG 图谱可视化")
    parser.add_argument(
        "--graph-dir",
        default="/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage/conv26",
        help="存储 GraphML 的目录",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 HTML 路径，默认输出到 graph-dir 下的 knowledge_graph.html",
    )
    args = parser.parse_args()

    graph_path = os.path.join(args.graph_dir, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graph_path):
        print(f"错误: 未找到 {graph_path}")
        return 1

    output_path = args.output or os.path.join(args.graph_dir, "knowledge_graph.html")
    print(f"加载图谱: {graph_path}")
    G = nx.read_graphml(graph_path)

    # Pyvis 交互式网络
    net = Network(height="95vh", width="100%", bgcolor="#1a1a2e")
    net.from_nx(G)

    # 按实体类型着色，并设置悬停提示
    for node in net.nodes:
        etype = (node.get("entity_type") or "other").lower()
        node["color"] = ENTITY_TYPE_COLORS.get(etype, ENTITY_TYPE_COLORS["other"])
        desc = node.get("description", "")
        if desc:
            # 截断过长的 description
            if len(desc) > 200:
                desc = desc[:200] + "..."
            node["title"] = f"<b>{node.get('id', '')}</b><br>类型: {node.get('entity_type', '')}<br><br>{desc}"
        else:
            node["title"] = f"<b>{node.get('id', '')}</b><br>类型: {node.get('entity_type', '')}"

    # 边悬停提示
    for edge in net.edges:
        desc = edge.get("description", "")
        if desc:
            edge["title"] = desc[:150] + ("..." if len(desc) > 150 else "")

    net.save_graph(output_path)
    # 移除会 404 的 lib/bindings/utils.js 引用（vis-network 已从 CDN 加载）
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace('<script src="lib/bindings/utils.js"></script>\n            ', "")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成: {output_path}")
    print("在浏览器中打开该 HTML 即可交互查看图谱（可拖拽、缩放、悬停查看详情）")
    return 0


if __name__ == "__main__":
    exit(main())
