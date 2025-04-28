from semantic_simple_graph import BaseSemanticSimpleGraph

import logging
import pickle

from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from IssueManagerSemanticMap import IssueManagerSemanticMap


class IssueManagerSemanticGraph(BaseSemanticSimpleGraph):
    """管理显式与隐式关系图的增强实现"""

    def __init__(self, semantic_map: IssueManagerSemanticMap):
        self.smap = semantic_map
        self.explicit_edges: Dict[str, Dict[str, List[Dict]]] = (
            {}
        )  # {source: {target: [relations]}}
        self.implicit_edges: Dict[str, Dict[str, List[Dict]]] = (
            {}
        )  # {source: {target: [relations]}}

    def _add_explicit_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        weight: float = 1.0,
        metadata: Dict = None,
    ):
        """添加显式关系边（带权值和元数据）"""
        # if source not in self.smap.reverse_index:
        #     logging.warning(f"Source uid {source} not in SemanticMap")
        #     return
        # if target not in self.smap.reverse_index:
        #     logging.warning(f"Target uid {target} not in SemanticMap")

        type2ms = {
            "pr": "github_prs",
            "issue": "github_issues",
            "commit": "github_commits",
            "contributor": "github_contributors",
            "code": "github_code",
        }

        source_ms = source.split("_")[1]
        target_ms = target.split("_")[1]
        if source not in self.smap.memoryspaces[type2ms.get(source_ms)].units:
            logging.warning(f"Source uid {source} not in SemanticMap")
            return
        if target not in self.smap.memoryspaces[type2ms.get(target_ms)].units:
            logging.warning(f"Target uid {target} not in SemanticMap")
        if metadata is None:
            metadata = {}

        # 自动记录时间戳
        metadata.setdefault("created_at", datetime.now())

        # 更新边存储
        if source not in self.explicit_edges:
            self.explicit_edges[source] = {}
        if target not in self.explicit_edges[source]:
            self.explicit_edges[source][target] = []

        # 防止重复添加
        existing = next(
            (e for e in self.explicit_edges[source][target] if e["type"] == rel_type),
            None,
        )
        if not existing:
            self.explicit_edges[source][target].append(
                {"type": rel_type, "weight": weight, "metadata": metadata}
            )

    def _add_implicit_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        score: float,
    ):
        """添加隐式边"""
        if source not in self.implicit_edges:
            self.implicit_edges[source] = {}
        if target not in self.implicit_edges[source]:
            self.implicit_edges[source][target] = []

        self.implicit_edges[source][target].append(
            {"type": rel_type, "score": score, "inferred_at": datetime.now()}
        )

    def find_relations(
        self,
        source: str = None,
        target: str = None,
        rel_type: str = None,
    ) -> List[Dict]:
        """查找满足条件的关系"""
        results = []

        if not any([source, target, rel_type]):
            return results

        # 搜索所有显式边
        for s in self.explicit_edges:
            if source and s != source:
                continue
            for t in self.explicit_edges[s]:
                if target and t != target:
                    continue
                for rel in self.explicit_edges[s][t]:
                    if rel_type and rel["type"] != rel_type:
                        continue
                    results.append({"source": s, "target": t, **rel})

        # 搜索所有隐式边
        for s in self.implicit_edges:
            if source and s != source:
                continue
            for t in self.implicit_edges[s]:
                if target and t != target:
                    continue
                for rel in self.implicit_edges[s][t]:
                    if rel_type and rel["type"] != rel_type:
                        continue
                    results.append({"source": s, "target": t, **rel})

        return results

    def infer_implicit_edges(
        self,
        ms_name1: str,
        ms_name2: str = None,
        similarity_threshold: float = 0.9,
    ):
        """推断两个指定类型节点的隐式边（比如代码相似性）"""
        # 获取目标类型的所有节点
        nodes1 = [
            (uid, unit)
            for ms in self.smap.memoryspaces.values()
            if ms.name == ms_name1
            for uid, unit in ms.units.items()
        ]
        embeddings1 = {uid: unit.embedding for uid, unit in nodes1}

        if not ms_name2:
            valid_pairs = [
                (u1, u2) for u1 in embeddings1 for u2 in embeddings1 if u1 != u2
            ]

            for uid1, uid2 in valid_pairs:
                sim = cosine_similarity([embeddings1[uid1]], [embeddings1[uid2]])[0][0]
                if sim >= similarity_threshold:
                    self._add_implicit_edge(uid1, uid2, "semantic_similarity", sim)

        else:
            nodes2 = [
                (uid, unit)
                for ms in self.smap.memoryspaces.values()
                if ms.name == ms_name2
                for uid, unit in ms.units.items()
            ]
            embeddings2 = {uid: unit.embedding for uid, unit in nodes2}

            for uid1 in embeddings1:
                for uid2 in embeddings2:
                    sim = cosine_similarity([embeddings1[uid1]], [embeddings2[uid2]])[
                        0
                    ][0]
                    if sim >= similarity_threshold:
                        self._add_implicit_edge(uid1, uid2, "semantic_similarity", sim)

    def save_graph(self, data_path: str):
        with open(data_path, "wb") as f:
            pickle.dump([self.explicit_edges, self.implicit_edges], f)
        logging.info(f"Graph relations saved to {data_path}")

    def load_graph(self, data_path: str):
        with open(data_path, "rb") as f:
            self.explicit_edges, self.implicit_edges = pickle.load(f)
        logging.info(f"Graph relations loaded from {data_path}")
