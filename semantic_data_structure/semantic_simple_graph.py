import logging
import pickle

from matplotlib import pyplot as plt
from data_type import WritingDataType as DataType
from semantic_map import BaseSemanticMap as SemanticMap
import networkx as nx

class BaseSemanticSimpleGraph:
    """
    使用 "semantic_map + graph_relations dict" 的轻量图实现。
    每个节点记录 {"parents": {pKey: relation}, "children": {cKey: relation}, "links": {lKey: relation}}。
    """

    def __init__(self, semantic_map=None):
        self.graph_relations = (
            {}
        )  # { nodeKey: {"parents":{}, "children":{}, "links":{}} }

        self.semantic_map = semantic_map if semantic_map else SemanticMap()

    def _ensure_node(self, key):
        if key not in self.graph_relations:
            self.graph_relations[key] = {"parents": {}, "children": {}, "links": {}}

    def add_node(
        self,
        key: str,
        value: dict,
        datatype: DataType,
        parent_keys=None,
        parent_relation="contains",
    ):
        """
        新增节点 (key, value)，可选地指定若干 parent_keys, 并用 parent_relation 表示每个父节点与本节点的关系。
        同时插入到 semantic_map。
        """
        # 1) SemanticMap 插入
        self.semantic_map.insert(key, value, datatype)

        # 2) 关系处理
        self._ensure_node(key)
        if parent_keys:
            for p in parent_keys:
                self._ensure_node(p)
                # 父->子
                self.graph_relations[p]["children"][key] = parent_relation
                # 子->父
                self.graph_relations[key]["parents"][p] = parent_relation

    def insert_edge(self, src_key, dst_key, relation="link"):
        """
        类似 networkx.insert_edge: 在 SimpleGraph 中添加一条边 (src->dst)。
        对于双向或无向，你可再加一条 (dst->src) 或者使用 self.link_nodes() 做互相引用。
        """
        self._ensure_node(src_key)
        self._ensure_node(dst_key)
        # 这条边可以理解为 "src -> dst"
        # 你可以决定是否要 “dst -> src” 对称
        self.graph_relations[src_key]["links"][dst_key] = relation

    def link_nodes(self, key1, key2, relation="link"):
        """
        连接两个节点 (key1 <-> key2)，并指定关系类型。
        """
        self.insert_edge(key1, key2, relation)
        self.insert_edge(key2, key1, relation)

    def build_index(self):
        """
        构建索引。
        """
        self.semantic_map.build_index()

    def retrieve_similar_nodes(self, query, k=5):
        """
        检索与查询相似的节点。
        """
        result = self.semantic_map.retrieve_similar(query, k)
        return result

    def delete_node(self, key: str):
        if key in self.graph_relations:
            for parent in list(self.graph_relations[key]["parents"].keys()):
                del self.graph_relations[parent]["children"][key]
            for child in list(self.graph_relations[key]["children"].keys()):
                del self.graph_relations[child]["parents"][key]
            for link in list(self.graph_relations[key]["links"].keys()):
                del self.graph_relations[link]["links"][key]
            del self.graph_relations[key]
            self.semantic_map.delete(key)

    def get_node(self, key: str):
        if key in self.graph_relations:
            return self.semantic_map.get(key)
        return None

    def retrieve_subtree(self, root_key: str):
        subtree = {}
        nodes_to_visit = [root_key]
        while nodes_to_visit:
            current = nodes_to_visit.pop()
            if current in subtree:
                continue
            subtree[current] = self.graph_relations[current]
            nodes_to_visit.extend(self.graph_relations[current]["children"].keys())
        return subtree

    def get_children(self, key: str):
        if key in self.graph_relations:
            return self.graph_relations[key]["children"]
        return {}

    def get_parents(self, key: str):
        if key in self.graph_relations:
            return self.graph_relations[key]["parents"]
        return {}

    def get_links(self, key: str):
        if key in self.graph_relations:
            return self.graph_relations[key]["links"]
        return {}

    def save_graph(self, data_path: str):
        with open(data_path, "wb") as f:
            pickle.dump(self.graph_relations, f)
        logging.info(f"Graph relations saved to {data_path}")

    def load_graph(self, data_path: str):
        with open(data_path, "rb") as f:
            self.graph_relations = pickle.load(f)
        logging.info(f"Graph relations loaded from {data_path}")

    def display(self):
        print("Nodes:")
        for node, attr in self.graph_relations.items():
            print(f"  {node}: {attr}")
        print("Edges:")
        for node, rels in self.graph_relations.items():
            for child, relation in rels.get("children", {}).items():
                print(f"  {node} -> {child}, relation: {relation}")
            for src, relation in rels.get("links", {}).items():
                print(f"  {node} -> {src}, relation: {relation}")

    def display_Graph(self):
        G = nx.DiGraph()
        for node, attrs in self.graph_relations.items():
            G.add_node(node, **attrs)
        for parent, attr in self.graph_relations.items():
            for child, relation in attr.get("children", {}).items():
                G.add_edge(parent, child, relation=relation, edge_type="child")
            for src, relation in attr.get("links", {}).items():
                G.add_edge(parent, src, relation=relation, edge_type="link")
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10)
        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.show()

    def auto_generate_subgraphs(self):
        """
        根据节点数据类型自动生成子图。
        使用预定义映射规则对节点分类，将节点添加到对应子图，
        并在同一子图中收集其内部的边（包括父子关系和关联链接）。
        """
        # 清空已有子图
        self.sub_graphs = {}
        # 定义数据类型与子图标签的对应关系
        subgraph_mapping = {
            DataType.Character: "Knowledge Memory Subgraph",
            DataType.Location: "Episodic Memory Subgraph",
            DataType.Plot: "Episodic Memory Subgraph"
        }
        # 遍历 semantic_map 中所有节点，添加到对应子图中
        for key, value, dt, emb in self.semantic_map.data:
            label = subgraph_mapping.get(dt, "Other Subgraph")
            if label not in self.sub_graphs:
                self.sub_graphs[label] = {"nodes": set(), "edges": []}
            self.sub_graphs[label]["nodes"].add(key)
        
        # 遍历 graph_relations，若边两端节点均存在于同一子图，则添加此边信息
        for src, attrs in self.graph_relations.items():
            # 处理父子关系边
            for child, relation in attrs.get("children", {}).items():
                for label, subgraph in self.sub_graphs.items():
                    if src in subgraph["nodes"] and child in subgraph["nodes"]:
                        subgraph["edges"].append((src, child, relation))
            # 处理关联关系边
            for link, relation in attrs.get("links", {}).items():
                for label, subgraph in self.sub_graphs.items():
                    if src in subgraph["nodes"] and link in subgraph["nodes"]:
                        subgraph["edges"].append((src, link, relation))
        
        # 输出生成的子图信息
        print("自动生成的子图:")
        for label, data in self.sub_graphs.items():
            print(f"{label}:")
            print("  节点:", data["nodes"])
            print("  边:", data["edges"])