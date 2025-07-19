from typing import Optional, Any, List, Set, Dict, Tuple
import os
import logging
from datetime import datetime
import pickle
from collections import Counter

import numpy as np
import networkx as nx

from core.memory_unit import MemoryUnit
from core.memory_space import MemorySpace
from core.semantic_map import SemanticMap
from core.neo4j_operator import Neo4jOperator


class SemanticGraph:
    """
    语义图谱 (SemanticGraph) 结合了 SemanticMap 的向量存储/搜索能力和 NetworkX 的图结构管理能力。
    它存储记忆单元作为节点，并允许在它们之间定义显式的命名关系。
    查询可以利用显式图遍历和隐式语义相似性。
    """

    def __init__(self, semantic_map_instance: Optional[SemanticMap] = None):
        """
        初始化语义图谱。
        参数:
            semantic_map_instance (Optional[SemanticMap]): 一个 SemanticMap 实例。
                                                          如果为 None，将创建一个新的默认 SemanticMap。
        """
        self.semantic_map: SemanticMap = (
            semantic_map_instance if semantic_map_instance else SemanticMap()
        )
        self.nx_graph: nx.DiGraph = (
            nx.DiGraph()
        )  # 使用 NetworkX有向图存储节点和显式关系
        # self.rel_types: Set[str] = set()  # 移除关系类型集合
        logging.info("SemanticGraph 已初始化。")

        # 添加Neo4j连接跟踪
        self._neo4j_connection: Optional[Neo4jOperator] = None  # 指向Neo4j的连接
        self._modified_relationships = (
            set()
        )  # 修改过的关系 (source_id, target_id, rel_type)
        self._deleted_relationships = set()  # 删除的关系
        self._modified_units = set()  # 修改过的记忆单元
        self._deleted_units = set()  # 删除的记忆单元

        # 内存管理配置
        self._max_nodes_in_memory = 10000  # 内存中最大节点数
        self._nodes_access_counts = {}  # 节点访问计数，用于LFU算法
        self._nodes_last_accessed = {}  # 节点最后访问时间，用于LRU算法
        self._nodes_dirty_flag = set()  # 标记内存中已修改但尚未同步的节点

        # 关系内存管理
        self._max_relationships_in_memory = 100000  # 内存中最大关系数
        self._relationship_cache = (
            {}
        )  # 缓存关系属性 {(source_id, target_id, rel_type): properties}
        self._relationships_access_counts = {}  # 关系访问计数
        self._relationships_last_accessed = {}  # 关系最后访问时间

    def connect_to_neo4j(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "k4s9k4s9",
        database: str = "neo4j",
        # Milvus连接参数
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_user: str = "",
        milvus_password: str = "",
        milvus_collection: str = "hippo",
    ) -> bool:
        """
        连接到Neo4j数据库，同时设置Milvus连接
        """
        try:
            # 创建Neo4jOperator，它内部已经集成了Milvus连接
            self._neo4j_connection = Neo4jOperator(
                neo4j_uri=uri,
                neo4j_user=user,
                neo4j_password=password,
                neo4j_database=database,
                milvus_host=milvus_host,
                milvus_port=milvus_port,
                milvus_user=milvus_user,
                milvus_password=milvus_password,
                milvus_collection=milvus_collection,
                embedding_dim=self.semantic_map.embedding_dim,
            )

            if self._neo4j_connection.neo4j_connected:
                logging.info(f"已连接到Neo4j: {uri}")

                # 设置SemanticMap的外部存储为Neo4jOperator的Milvus连接
                self.semantic_map._external_storage = (
                    self._neo4j_connection.milvus_operator
                )
                logging.info("SemanticMap已连接到Milvus外部存储")

                return True
            else:
                self._neo4j_connection = None
                logging.error("连接到Neo4j失败")
                return False
        except Exception as e:
            logging.error(f"初始化Neo4j连接失败: {e}")
            self._neo4j_connection = None
            return False

    def add_unit(
        self,
        unit: MemoryUnit,
        explicit_content_for_embedding: Optional[Any] = None,
        content_type_for_embedding: Optional[str] = None,
        space_names: Optional[List[str]] = None,
        rebuild_semantic_map_index_immediately: bool = False,
    ):
        """
        向图谱添加一个记忆单元 (节点)。
        单元也会被添加到内部的 SemanticMap 中。
        """
        # 检查内存限制，必要时触发换页
        if len(self.semantic_map.memory_units) >= self._max_nodes_in_memory:
            self.swap_out_nodes(count=int(self._max_nodes_in_memory * 0.1))  # 换出10%

        # 1. 将单元添加到 SemanticMap
        self.semantic_map.add_unit(
            unit,
            explicit_content_for_embedding,
            content_type_for_embedding,
            space_names,
            rebuild_index_immediately=rebuild_semantic_map_index_immediately,
        )

        # 2. 将单元ID作为节点添加到 NetworkX 图中
        if not self.nx_graph.has_node(unit.uid):
            # 在节点上存储一些基本属性
            node_attrs = {
                "uid": unit.uid,
                "created": str(datetime.now()),
                **{
                    k: v
                    for k, v in unit.raw_data.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            self.nx_graph.add_node(unit.uid, **node_attrs)
            logging.debug(f"节点 '{unit.uid}' 已添加到 NetworkX 图。")
        else:
            # 如果节点已存在，更新其属性
            node_attrs = {
                "uid": unit.uid,
                "updated": str(datetime.now()),
                **{
                    k: v
                    for k, v in unit.raw_data.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            nx.set_node_attributes(self.nx_graph, {unit.uid: node_attrs})
            logging.debug(f"节点 '{unit.uid}' 的属性已在 NetworkX 图中更新。")

        # 3. 更新访问统计和标记为脏数据
        self._nodes_access_counts[unit.uid] = (
            self._nodes_access_counts.get(unit.uid, 0) + 1
        )
        self._nodes_last_accessed[unit.uid] = datetime.now().timestamp()
        self._nodes_dirty_flag.add(unit.uid)
        self._modified_units.add(unit.uid)

    def add_relationship(
        self,
        source_uid: str,
        target_uid: str,
        relationship_name: str,
        bidirectional: bool = False,
        **kwargs: Any,
    ):
        """
        在两个已存在的记忆单元 (节点) 或 MemorySpace 之间添加一条显式关系 (边)。
        支持 MemoryUnit <-> MemoryUnit, MemorySpace <-> MemorySpace, MemoryUnit <-> MemorySpace。
        """

        # 判断节点类型和id
        def get_node_id(uid):
            # MemorySpace: 以ms:前缀
            if uid in self.semantic_map.memory_units:
                return uid, "memory_unit"
            elif uid in self.semantic_map.memory_spaces:
                return f"ms:{uid}", "memory_space"
            elif isinstance(uid, str) and uid.startswith("ms:"):
                ms_name = uid[3:]
                if ms_name in self.semantic_map.memory_spaces:
                    return uid, "memory_space"
            return uid, None

        src_id, src_type = get_node_id(source_uid)
        tgt_id, tgt_type = get_node_id(target_uid)

        # 检查源节点和目标节点是否存在
        src_exists = (
            src_type == "memory_unit" and self.semantic_map.get_unit(src_id) is not None
        ) or (
            src_type == "memory_space" and src_id[3:] in self.semantic_map.memory_spaces
        )
        tgt_exists = (
            tgt_type == "memory_unit" and self.semantic_map.get_unit(tgt_id) is not None
        ) or (
            tgt_type == "memory_space" and tgt_id[3:] in self.semantic_map.memory_spaces
        )
        if not src_exists:
            logging.error(f"源节点 '{source_uid}' 不存在。无法添加关系。")
            return False
        if not tgt_exists:
            logging.error(f"目标节点 '{target_uid}' 不存在。无法添加关系。")
            return False

        # 确保节点存在于nx_graph中
        if not self.nx_graph.has_node(src_id):
            if src_type == "memory_unit":
                unit = self.semantic_map.get_unit(src_id)
                if unit is not None:
                    self.nx_graph.add_node(
                        src_id,
                        uid=unit.uid,
                        type="memory_unit",
                        created=str(datetime.now()),
                    )
            elif src_type == "memory_space":
                ms = self.semantic_map.memory_spaces.get(src_id[3:])
                if ms is not None:
                    self.nx_graph.add_node(
                        src_id,
                        name=ms.name,
                        type="memory_space",
                        created=str(datetime.now()),
                    )
        if not self.nx_graph.has_node(tgt_id):
            if tgt_type == "memory_unit":
                unit = self.semantic_map.get_unit(tgt_id)
                if unit is not None:
                    self.nx_graph.add_node(
                        tgt_id,
                        uid=unit.uid,
                        type="memory_unit",
                        created=str(datetime.now()),
                    )
            elif tgt_type == "memory_space":
                ms = self.semantic_map.memory_spaces.get(tgt_id[3:])
                if ms is not None:
                    self.nx_graph.add_node(
                        tgt_id,
                        name=ms.name,
                        type="memory_space",
                        created=str(datetime.now()),
                    )

        # 添加关系到NetworkX图
        # 只保留GML支持的属性类型
        def filter_gml_attrs(attrs):
            return {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in attrs.items()
                if v is not None
            }

        edge_attributes = {
            "type": relationship_name,
            "created": str(datetime.now()),
            **filter_gml_attrs(kwargs),
        }
        self.nx_graph.add_edge(src_id, tgt_id, **edge_attributes)

        # 缓存关系属性
        self.swap_in_relationship(src_id, tgt_id, relationship_name, kwargs)

        # 记录修改
        self._modified_relationships.add((src_id, tgt_id, relationship_name))

        # 记录关系类型
        # self.rel_types.add(relationship_name) # 移除

        logging.info(
            f"已添加从 '{src_id}' 到 '{tgt_id}' 的关系 '{relationship_name}'。"
        )

        if bidirectional:
            # 添加反向关系
            self.nx_graph.add_edge(tgt_id, src_id, **edge_attributes)
            self.swap_in_relationship(tgt_id, src_id, relationship_name, kwargs)
            self._modified_relationships.add((tgt_id, src_id, relationship_name))
            logging.info(
                f"已添加从 '{tgt_id}' 到 '{src_id}' 的双向关系 '{relationship_name}'。"
            )

        return True

    def get_relationship(
        self, source_uid: str, target_uid: str, relationship_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        检查两个节点之间是否存在指定类型的关系，并返回关系属性

        Args:
            source_uid: 源节点UID
            target_uid: 目标节点UID
            relationship_type: 关系类型

        Returns:
            关系属性字典，如果关系不存在则返回None
        """
        try:
            # 首先检查内存中的关系缓存
            rel_key = (source_uid, target_uid, relationship_type)
            if rel_key in self._relationship_cache:
                # 更新访问统计
                self._relationships_access_counts[rel_key] = (
                    self._relationships_access_counts.get(rel_key, 0) + 1
                )
                self._relationships_last_accessed[rel_key] = datetime.now().timestamp()
                return self._relationship_cache[rel_key]

            # 检查NetworkX图中是否存在该关系
            if self.nx_graph.has_edge(source_uid, target_uid):
                edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
                if edge_data and edge_data.get("type") == relationship_type:
                    # 提取关系属性（排除系统字段）
                    relationship_properties = {
                        k: v
                        for k, v in edge_data.items()
                        if k not in ["type", "created", "updated"]
                    }

                    # 添加到缓存
                    self.swap_in_relationship(
                        source_uid,
                        target_uid,
                        relationship_type,
                        relationship_properties,
                    )

                    return relationship_properties

            # 如果内存中没有，尝试从Neo4j加载
            if self._neo4j_connection and self._neo4j_connection.neo4j_connected:
                try:
                    rels = self._neo4j_connection.get_relationships(
                        source_id=source_uid,
                        target_id=target_uid,
                        relationship_type=relationship_type,
                        limit=1,
                    )
                    if rels:
                        relationship_data = rels[0].get("properties", {})
                        # 添加到内存缓存
                        self.swap_in_relationship(
                            source_uid, target_uid, relationship_type, relationship_data
                        )
                        # 如果NetworkX图中没有这个关系，也添加进去
                        if not self.nx_graph.has_edge(source_uid, target_uid):
                            edge_attributes = {
                                "type": relationship_type,
                                "loaded_from_neo4j": str(datetime.now()),
                                **{
                                    k: (
                                        str(v)
                                        if not isinstance(v, (str, int, float, bool))
                                        else v
                                    )
                                    for k, v in relationship_data.items()
                                    if v is not None
                                },
                            }
                            self.nx_graph.add_edge(
                                source_uid, target_uid, **edge_attributes
                            )
                        return relationship_data
                except Exception as e:
                    logging.warning(f"从Neo4j加载关系失败: {e}")

            # 关系不存在
            return None

        except Exception as e:
            logging.error(
                f"检查关系失败 ({source_uid} -[{relationship_type}]-> {target_uid}): {e}"
            )
            return None

    def delete_unit(
        self, uid: str, rebuild_semantic_map_index_immediately: bool = False
    ):
        """从图谱和底层的 SemanticMap 中删除一个记忆单元及其所有相关关系。"""
        # 1. 从 SemanticMap 删除
        self.semantic_map.delete_unit(
            uid, rebuild_index_immediately=rebuild_semantic_map_index_immediately
        )

        # 2. 从 NetworkX 图中删除节点 (这会自动删除所有相关的边)
        if self.nx_graph.has_node(uid):
            # 记录即将删除的关系
            for neighbor in list(self.nx_graph.neighbors(uid)):
                edge_data = self.nx_graph.get_edge_data(uid, neighbor)
                if edge_data:
                    rel_type = edge_data.get("type", "RELATED_TO")
                    self._deleted_relationships.add((uid, neighbor, rel_type))

            # 记录反向关系
            for predecessor in list(self.nx_graph.predecessors(uid)):
                edge_data = self.nx_graph.get_edge_data(predecessor, uid)
                if edge_data:
                    rel_type = edge_data.get("type", "RELATED_TO")
                    self._deleted_relationships.add((predecessor, uid, rel_type))

            self.nx_graph.remove_node(uid)
            logging.info(f"节点 '{uid}' 及其关系已从 NetworkX 图中删除。")
        else:
            logging.warning(f"尝试从 NetworkX 图中删除不存在的节点 '{uid}'。")

        # 3. 清理内存管理相关数据
        self._deleted_units.add(uid)
        if uid in self._modified_units:
            self._modified_units.remove(uid)
        if uid in self._nodes_access_counts:
            del self._nodes_access_counts[uid]
        if uid in self._nodes_last_accessed:
            del self._nodes_last_accessed[uid]
        if uid in self._nodes_dirty_flag:
            self._nodes_dirty_flag.remove(uid)

    def delete_relationship(
        self, source_uid: str, target_uid: str, relationship_name: Optional[str] = None
    ):
        """
        删除两个节点之间的特定关系或所有关系。
        """
        if not self.nx_graph.has_edge(source_uid, target_uid):
            logging.warning(f"节点 '{source_uid}' 和 '{target_uid}' 之间没有直接边。")
            return False

        if relationship_name:
            edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
            if edge_data and edge_data.get("type") == relationship_name:
                self.nx_graph.remove_edge(source_uid, target_uid)
                self._deleted_relationships.add(
                    (source_uid, target_uid, relationship_name)
                )

                # 从关系缓存中移除
                rel_key = (source_uid, target_uid, relationship_name)
                if rel_key in self._relationship_cache:
                    del self._relationship_cache[rel_key]
                if rel_key in self._relationships_access_counts:
                    del self._relationships_access_counts[rel_key]
                if rel_key in self._relationships_last_accessed:
                    del self._relationships_last_accessed[rel_key]

                logging.info(
                    f"已删除从 '{source_uid}' 到 '{target_uid}' 的关系 '{relationship_name}'。"
                )
                return True
            else:
                logging.warning(
                    f"未找到从 '{source_uid}' 到 '{target_uid}' 的名为 '{relationship_name}' 的关系。"
                )
                return False
        else:
            # 删除所有直接边
            edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
            if edge_data:
                rel_type = edge_data.get("type", "RELATED_TO")
                self._deleted_relationships.add((source_uid, target_uid, rel_type))

            self.nx_graph.remove_edge(source_uid, target_uid)
            logging.info(f"已删除从 '{source_uid}' 到 '{target_uid}' 的所有直接关系。")
            return True

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        """智能获取节点，当节点不在内存时从外部存储加载"""
        # 先检查是否在内存中
        unit = self.semantic_map.get_unit(uid)
        if unit:
            # 更新访问统计
            self._nodes_access_counts[uid] = self._nodes_access_counts.get(uid, 0) + 1
            self._nodes_last_accessed[uid] = datetime.now().timestamp()
            return unit

        # 不在内存中，尝试从外部存储加载
        loaded_count = self.swap_in_nodes([uid])
        if loaded_count > 0:
            unit = self.semantic_map.get_unit(uid)
            if unit:
                return unit

        logging.warning(f"节点 '{uid}' 在内存和外部存储中均不存在")
        return None

    # ==============================
    # 新增方法：兼容新的 MemorySpace 架构
    # ==============================

    def create_memory_space_in_map(self, space_name: str) -> MemorySpace:
        """在底层的 SemanticMap 中创建或获取一个记忆空间。"""
        return self.semantic_map.create_memory_space(space_name)

    def add_unit_to_space_in_map(self, unit_or_uid, space_name: str):
        """将一个记忆单元添加到 SemanticMap 中的指定记忆空间。"""
        self.semantic_map.add_unit_to_space(unit_or_uid, space_name)

    def remove_unit_from_space_in_map(self, unit_or_uid, space_name: str):
        """从 SemanticMap 中的指定记忆空间移除一个记忆单元。"""
        self.semantic_map.remove_unit_from_space(unit_or_uid, space_name)

    def get_units_in_memory_space(
        self, ms_names, recursive: bool = True
    ) -> List[MemoryUnit]:
        # 支持ms_names为["ms:xxx", ...]格式
        if isinstance(ms_names, str):
            ms_names = [ms_names]
        normed = []
        for name in ms_names:
            if name.startswith("ms:"):
                normed.append(name[3:])
            else:
                normed.append(name)
        return self.semantic_map.get_units_in_memory_space(normed, recursive=recursive)

    def search_similarity_in_graph(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 5,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
        return_score: bool = False,  # 新增参数，默认为False
        candidate_units: Optional[List[Any]] = None,  # 新增参数，支持MemoryUnit或uid
    ):
        normed = None
        if ms_names:
            normed = [(n[3:] if n.startswith("ms:") else n) for n in ms_names]

        # 处理 candidate_units，转换为 candidate_uids
        candidate_uids = None
        if candidate_units is not None:
            candidate_uids = []
            for u in candidate_units:
                if hasattr(u, "uid"):
                    candidate_uids.append(u.uid)
                else:
                    candidate_uids.append(str(u))

        if query_text is not None:
            results = self.semantic_map.search_similarity_by_text(
                query_text, top_k, normed, candidate_uids
            )
        elif query_embedding is not None:
            results = self.semantic_map.search_similarity_by_vector(
                query_embedding, top_k, normed, candidate_uids
            )
        elif query_image_path is not None:
            results = self.semantic_map.search_similarity_by_image(
                query_image_path, top_k, normed, candidate_uids
            )
        else:
            logging.warning(
                "必须提供 query_text, query_embedding 或 query_image_path 之一进行相似性搜索。"
            )
            return []

        if return_score:
            return results
        else:
            return [unit for unit, _ in results]

    # ==============================
    # 内存管理和换页策略（保持原有功能）
    # ==============================

    def swap_out_nodes(
        self,
        count: int = 100,
        strategy: str = "LRU",
        query_context: Optional[str] = None,
    ) -> int:
        """
        将不常用节点从内存移出到外部存储
        """
        if not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法执行节点换出")
            return 0

        current_count = len(self.semantic_map.memory_units)
        if current_count == 0:
            logging.debug("内存中没有节点可以换出")
            return 0

        # 使用 SemanticMap 的 swap_out 方法
        before_count = len(self.semantic_map.memory_units)
        self.semantic_map.swap_out(
            count=count, strategy=strategy, query_context=query_context
        )
        after_count = len(self.semantic_map.memory_units)
        removed_count = before_count - after_count

        # 清理 SemanticGraph 中相关的访问统计
        for uid in list(self._nodes_access_counts.keys()):
            if uid not in self.semantic_map.memory_units:
                self._nodes_access_counts.pop(uid, None)
                self._nodes_last_accessed.pop(uid, None)
                self._nodes_dirty_flag.discard(uid)

        logging.info(f"已将 {removed_count} 个节点从内存移出")
        return removed_count

    def swap_in_nodes(self, node_ids: List[str]) -> int:
        """
        从外部存储加载节点到内存
        """
        if not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法从外部加载节点")
            return 0

        # 过滤已在内存中的节点
        ids_to_load = [
            uid for uid in node_ids if uid not in self.semantic_map.memory_units
        ]
        if not ids_to_load:
            return 0

        # 从外部存储加载节点
        loaded_units = self.semantic_map._external_storage.get_units_batch(ids_to_load)

        # 添加到内存
        for unit in loaded_units:
            # 检查内存限制
            if len(self.semantic_map.memory_units) >= self._max_nodes_in_memory:
                self.swap_out_nodes(count=1)

            self.semantic_map.memory_units[unit.uid] = unit
            self._nodes_access_counts[unit.uid] = 1
            self._nodes_last_accessed[unit.uid] = datetime.now().timestamp()

            # 添加到FAISS索引
            if unit.embedding is not None and self.semantic_map.faiss_index:
                try:
                    internal_id = self.semantic_map._internal_faiss_id_counter
                    self.semantic_map._internal_faiss_id_counter += 1
                    self.semantic_map._uid_to_internal_faiss_id[unit.uid] = internal_id

                    vector = unit.embedding.reshape(1, -1).astype(np.float32)
                    ids = np.array([internal_id], dtype=np.int64)
                    self.semantic_map.faiss_index.add_with_ids(vector, ids)
                    logging.debug(f"节点 '{unit.uid}' 的向量已添加到FAISS索引")
                except Exception as e:
                    logging.error(f"向FAISS索引添加向量失败: {e}")

        logging.info(f"已从外部存储加载 {len(loaded_units)} 个节点到内存")
        return len(loaded_units)

    def swap_in_relationship(
        self, source_uid: str, target_uid: str, relationship_type: str, properties: dict
    ):
        """将关系缓存到内存"""
        rel_key = (source_uid, target_uid, relationship_type)
        self._relationship_cache[rel_key] = properties
        self._relationships_access_counts[rel_key] = 1
        self._relationships_last_accessed[rel_key] = datetime.now().timestamp()

        # 检查缓存大小，必要时清理
        if len(self._relationship_cache) > self._max_relationships_in_memory:
            self.swap_out_relationship(int(self._max_relationships_in_memory * 0.2))

    def swap_out_relationship(self, count: int = 100, strategy: str = "LRU"):
        """
        将关系从内存换出到Neo4j数据库
        """
        if not self._relationship_cache:
            logging.debug("关系缓存为空，无需换出")
            return

        if not self._neo4j_connection or not self._neo4j_connection.neo4j_connected:
            logging.warning("未连接到Neo4j数据库，无法执行关系换出")
            # 如果没有数据库连接，降级为简单清理缓存
            self._clear_relationship_cache_fallback(count)
            return

        # 根据策略确定要换出的关系
        if strategy == "LRU":
            sorted_rels = sorted(
                self._relationship_cache.keys(),
                key=lambda k: self._relationships_last_accessed.get(k, 0),
            )
        elif strategy == "LFU":
            sorted_rels = sorted(
                self._relationship_cache.keys(),
                key=lambda k: self._relationships_access_counts.get(k, 0),
            )
        else:
            logging.warning(f"不支持的换出策略: {strategy}，使用LRU")
            sorted_rels = sorted(
                self._relationship_cache.keys(),
                key=lambda k: self._relationships_last_accessed.get(k, 0),
            )

        # 限制换出数量
        actual_count = min(count, len(sorted_rels))
        relationships_to_swap_out = sorted_rels[:actual_count]

        synced_count = 0
        removed_count = 0

        # 将关系同步到Neo4j数据库
        for rel_key in relationships_to_swap_out:
            source_uid, target_uid, rel_type = rel_key

            try:
                # 获取关系属性
                properties = self._relationship_cache.get(rel_key, {})

                # 同步到Neo4j数据库
                if self._neo4j_connection and self._neo4j_connection.neo4j_connected:
                    if self._neo4j_connection.add_relationship(
                        source_uid, target_uid, rel_type, properties=properties
                    ):
                        synced_count += 1
                        logging.debug(
                            f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 已同步到Neo4j"
                        )
                    else:
                        logging.warning(
                            f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 同步到Neo4j失败"
                        )
                else:
                    logging.warning(
                        f"未连接到Neo4j，无法同步关系 ({source_uid} -[{rel_type}]-> {target_uid})"
                    )

                # 从内存缓存中移除关系
                if rel_key in self._relationship_cache:
                    del self._relationship_cache[rel_key]
                    removed_count += 1
                if rel_key in self._relationships_access_counts:
                    del self._relationships_access_counts[rel_key]
                if rel_key in self._relationships_last_accessed:
                    del self._relationships_last_accessed[rel_key]

                logging.debug(
                    f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 已从内存缓存中移除"
                )

            except Exception as e:
                logging.error(
                    f"换出关系 ({source_uid} -[{rel_type}]-> {target_uid}) 时出错: {e}"
                )

        logging.info(
            f"使用{strategy}策略已将 {removed_count} 个关系从内存换出，其中 {synced_count} 个已同步到Neo4j数据库"
        )

    def _clear_relationship_cache_fallback(self, count: int = 100):
        """
        备用关系缓存清理方法（当没有数据库连接时使用）
        """
        if not self._relationship_cache:
            return

        # 按最后访问时间排序
        sorted_rels = sorted(
            self._relationship_cache.keys(),
            key=lambda k: self._relationships_last_accessed.get(k, 0),
        )

        # 移除最旧的关系
        removed_count = 0
        for rel_key in sorted_rels[:count]:
            if rel_key in self._relationship_cache:
                del self._relationship_cache[rel_key]
                removed_count += 1
            if rel_key in self._relationships_access_counts:
                del self._relationships_access_counts[rel_key]
            if rel_key in self._relationships_last_accessed:
                del self._relationships_last_accessed[rel_key]

        logging.debug(f"已清理 {removed_count} 个关系缓存（备用模式）")

    # ==============================
    # 数据库同步方法（更新兼容性）
    # ==============================

    def sync_to_external(
        self,
        force_full_sync: bool = False,
        auto_detect_first_sync: bool = True,
        sync_nodes: bool = True,
        sync_relationships: bool = True,
    ) -> Dict[str, int]:
        """
        智能同步数据到外部存储(Neo4j和Milvus)
        """
        stats = {
            "nodes_synced": 0,
            "nodes_failed": 0,
            "relationships_synced": 0,
            "relationships_failed": 0,
            "sync_mode": "incremental",
        }

        # 1. 同步节点到Milvus（通过SemanticMap）
        if sync_nodes and self.semantic_map._external_storage:
            semantic_map_stats = self.semantic_map.sync_to_external(force_full_sync)
            stats["nodes_synced"] = (
                semantic_map_stats[0]
                if isinstance(semantic_map_stats, tuple)
                else semantic_map_stats.get("nodes_synced", 0)
            )
            stats["nodes_failed"] = (
                semantic_map_stats[1]
                if isinstance(semantic_map_stats, tuple)
                else semantic_map_stats.get("nodes_failed", 0)
            )

        # 2. 同步关系到Neo4j
        if sync_relationships and self._neo4j_connection:
            rel_stats = self._sync_relationships_to_neo4j(force_full_sync)
            stats["relationships_synced"] += rel_stats["relationships_synced"]
            stats["relationships_failed"] += rel_stats["relationships_failed"]

        # 3. 更新同步状态
        if stats["nodes_synced"] > 0 or stats["relationships_synced"] > 0:
            if force_full_sync:
                self._clear_all_dirty_flags()

        # 4. 记录同步结果
        total_success = stats["nodes_synced"] + stats["relationships_synced"]
        total_failed = stats["nodes_failed"] + stats["relationships_failed"]

        sync_mode = "full" if force_full_sync else "incremental"
        stats["sync_mode"] = sync_mode

        logging.info(
            f"SemanticGraph同步完成 ({sync_mode})。"
            f"节点: 成功={stats['nodes_synced']}, 失败={stats['nodes_failed']}; "
            f"关系: 成功={stats['relationships_synced']}, 失败={stats['relationships_failed']}; "
            f"总计: 成功={total_success}, 失败={total_failed}"
        )

        return stats

    def _sync_relationships_to_neo4j(self, force_full_sync: bool) -> Dict[str, int]:
        """同步关系到Neo4j"""
        stats = {"relationships_synced": 0, "relationships_failed": 0}

        # 确定要同步的关系
        if force_full_sync:
            # 全量同步：同步所有NetworkX图中的关系
            relationships_to_sync = []
            for source, target, data in self.nx_graph.edges(data=True):
                rel_type = data.get("type", "RELATED_TO")
                relationships_to_sync.append((source, target, rel_type))
            logging.info(
                f"全量同步: 准备同步 {len(relationships_to_sync)} 个关系到Neo4j"
            )
        else:
            # 增量同步：只同步修改过的关系
            relationships_to_sync = list(self._modified_relationships)
            logging.info(
                f"增量同步: 准备同步 {len(relationships_to_sync)} 个修改的关系到Neo4j"
            )

        # 同步关系
        for source_uid, target_uid, rel_type in relationships_to_sync:
            try:
                # 获取关系属性
                edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
                if not edge_data:
                    stats["relationships_failed"] += 1
                    continue

                # 过滤掉系统属性
                properties = {
                    k: v
                    for k, v in edge_data.items()
                    if k not in ["type", "created", "updated"]
                }

                if self._neo4j_connection and self._neo4j_connection.neo4j_connected:
                    if not self._neo4j_connection.ensure_node_exists(
                        source_uid, node_type="MemoryUnit"
                    ):
                        logging.warning(f"确保源节点 '{source_uid}' 存在失败")
                    if not self._neo4j_connection.ensure_node_exists(
                        target_uid, node_type="MemoryUnit"
                    ):
                        logging.warning(f"确保目标节点 '{target_uid}' 存在失败")

                    # 同步到Neo4j
                    if self._neo4j_connection.add_relationship(
                        source_uid, target_uid, rel_type, properties=properties
                    ):
                        stats["relationships_synced"] += 1
                        self._modified_relationships.discard(
                            (source_uid, target_uid, rel_type)
                        )
                        logging.debug(
                            f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 已同步到Neo4j"
                        )
                    else:
                        stats["relationships_failed"] += 1
                        logging.warning(
                            f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 同步到Neo4j失败"
                        )
                else:
                    stats["relationships_failed"] += 1
                    logging.warning(
                        f"未连接到Neo4j，无法同步关系 ({source_uid} -[{rel_type}]-> {target_uid})"
                    )

            except Exception as e:
                stats["relationships_failed"] += 1
                logging.error(
                    f"同步关系 ({source_uid} -[{rel_type}]-> {target_uid}) 时出错: {e}"
                )

        # 处理删除的关系
        if self._deleted_relationships:
            deleted_relationships = list(self._deleted_relationships)
            logging.info(f"处理 {len(deleted_relationships)} 个已删除的关系")

            for source_uid, target_uid, rel_type in deleted_relationships:
                if self._neo4j_connection and self._neo4j_connection.neo4j_connected:
                    try:
                        if self._neo4j_connection.delete_relationship(
                            source_uid, target_uid, relationship_type=rel_type
                        ):
                            stats["relationships_synced"] += 1
                            self._deleted_relationships.discard(
                                (source_uid, target_uid, rel_type)
                            )
                            logging.debug(
                                f"已删除关系 ({source_uid} -[{rel_type}]-> {target_uid}) 从Neo4j中移除"
                            )
                        else:
                            stats["relationships_failed"] += 1
                            logging.warning(
                                f"从Neo4j删除关系 ({source_uid} -[{rel_type}]-> {target_uid}) 失败"
                            )
                    except Exception as e:
                        stats["relationships_failed"] += 1
                        logging.error(
                            f"从Neo4j删除关系 ({source_uid} -[{rel_type}]-> {target_uid}) 时出错: {e}"
                        )
                else:
                    stats["relationships_failed"] += 1
                    logging.warning(
                        f"未连接到Neo4j，无法删除关系 ({source_uid} -[{rel_type}]-> {target_uid})"
                    )

        return stats

    def _clear_all_dirty_flags(self):
        """清理所有修改标记（用于全量同步后）"""
        self._nodes_dirty_flag.clear()
        self._modified_units.clear()
        self._deleted_units.clear()
        self._modified_relationships.clear()
        self._deleted_relationships.clear()
        logging.debug("已清理所有修改标记")

    # 保留原有方法作为便利函数
    def incremental_export(self) -> Dict[str, int]:
        """增量导出修改过的节点和关系到外部存储"""
        return self.sync_to_external(
            force_full_sync=False, auto_detect_first_sync=False
        )

    def full_export(self) -> Dict[str, int]:
        """完整导出所有节点和关系到外部存储"""
        return self.sync_to_external(force_full_sync=True, auto_detect_first_sync=False)

    # ==============================
    # 保持原有的查询和遍历API
    # ==============================

    def get_unit_data(self, uid: str) -> Optional[MemoryUnit]:
        """从底层的 SemanticMap 检索记忆单元对象。"""
        return self.get_unit(uid)

    def build_semantic_map_index(self):
        """构建底层 SemanticMap 的 FAISS 索引。"""
        self.semantic_map.build_index()

    def get_all_relations(self) -> List[str]:
        """
        返回所有已注册的显式关系类型列表（动态收集）。
        """
        rel_types = set()
        for _, _, data in self.nx_graph.edges(data=True):
            rel_type = data.get("type")
            if rel_type:
                rel_types.add(rel_type)
        return list(rel_types)

    def get_all_relations_with_samples(self, samples_per_type: int = 2) -> dict:
        # 1. 分类收集所有边
        rel_samples = {}
        seen_pairs = set()
        for u, v, data in self.nx_graph.edges(data=True):
            rel_type = data.get("type")
            if not rel_type:
                continue
            # 2. 检查是否已收集足够样本
            if rel_type not in rel_samples:
                rel_samples[rel_type] = []
            if len(rel_samples[rel_type]) >= samples_per_type:
                continue
            # 3. 判断是否为双向
            if (v, u, rel_type) in seen_pairs:
                # 已经作为双向收录过
                continue
            if self.nx_graph.has_edge(v, u):
                rev_data = self.nx_graph.get_edge_data(v, u)
                if rev_data and rev_data.get("type") == rel_type:
                    # 双向
                    rel_samples[rel_type].append(f"{u} <-> {v}")
                    seen_pairs.add((u, v, rel_type))
                    seen_pairs.add((v, u, rel_type))
                    continue
            # 单向
            rel_samples[rel_type].append(f"{u} -> {v}")
            seen_pairs.add((u, v, rel_type))
        return rel_samples

    def traverse_explicit_nodes(
        self,
        uid: str,
        relationship_type: Optional[str] = None,
        direction: str = "successors",
        ms_names: Optional[List[str]] = None,
    ) -> List[MemoryUnit]:
        """
        遍历与给定节点通过显式关系连接的邻居节点。
        """
        if not self.nx_graph.has_node(uid):
            logging.warning(f"节点 '{uid}' 不在图中，无法遍历。")
            return []

        neighbor_ids: Set[str] = set()
        if direction == "successors":
            for successor in self.nx_graph.successors(uid):
                if relationship_type:
                    edge_data = self.nx_graph.get_edge_data(uid, successor)
                    if edge_data and edge_data.get("type") == relationship_type:
                        neighbor_ids.add(successor)
                else:
                    neighbor_ids.add(successor)
        elif direction == "predecessors":
            for predecessor in self.nx_graph.predecessors(uid):
                if relationship_type:
                    edge_data = self.nx_graph.get_edge_data(predecessor, uid)
                    if edge_data and edge_data.get("type") == relationship_type:
                        neighbor_ids.add(predecessor)
                else:
                    neighbor_ids.add(predecessor)
        elif direction == "all":
            all_neighbors_temp = set(self.nx_graph.successors(uid))
            all_neighbors_temp.update(self.nx_graph.predecessors(uid))

            for neighbor in all_neighbors_temp:
                passes_filter = False
                if not relationship_type:
                    passes_filter = True
                else:
                    if (
                        self.nx_graph.has_edge(uid, neighbor)
                        and self.nx_graph.get_edge_data(uid, neighbor).get("type")
                        == relationship_type
                    ):
                        passes_filter = True
                    elif (
                        self.nx_graph.has_edge(neighbor, uid)
                        and self.nx_graph.get_edge_data(neighbor, uid).get("type")
                        == relationship_type
                    ):
                        passes_filter = True
                if passes_filter:
                    neighbor_ids.add(neighbor)
        else:
            logging.warning(
                f"无效的遍历方向: '{direction}'。应为 'successors', 'predecessors', 或 'all'。"
            )
            return []

        # 根据 ms_names 过滤 (如果提供)
        if ms_names:
            ms_units = set()
            for name in ms_names:
                if name.startswith("ms:"):
                    space_name = name[3:]
                else:
                    space_name = name
                space = self.semantic_map.get_memory_space(space_name)
                if space:
                    space_uids = space.get_all_unit_uids(recursive=True)
                    ms_units.update(space_uids)
                else:
                    logging.warning(f"记忆空间 '{space_name}' 未找到，已忽略")
            neighbor_ids.intersection_update(ms_units)

        # 获取 MemoryUnit 对象
        results: List[MemoryUnit] = []
        for nid in neighbor_ids:
            unit = self.get_unit(nid)  # 使用智能获取方法
            if unit:
                results.append(unit)
        return results

    def traverse_implicit_nodes(
        self, uid: str, k: int = 5, ms_names: Optional[List[str]] = None
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        基于语义相似性查找与给定节点隐式相关的节点。
        """
        start_unit = self.get_unit(uid)
        if not start_unit or start_unit.embedding is None:
            logging.warning(f"节点 '{uid}' 不存在或没有向量，无法进行隐式遍历。")
            return []

        normed = None
        if ms_names:
            normed = [(n[3:] if n.startswith("ms:") else n) for n in ms_names]
        similar_units_with_scores = self.semantic_map.search_similarity_by_vector(
            start_unit.embedding,
            k=k + 1,  # 获取稍多一些，以防 uid 是最相似的
            ms_names=normed,
        )

        results: List[Tuple[MemoryUnit, float]] = []
        for unit, score in similar_units_with_scores:
            if unit.uid != uid:  # 排除起始节点本身
                results.append((unit, score))
            if len(results) >= k:  # 如果已达到k个结果
                break
        return results

    def display_graph_summary(self):
        """打印图谱的摘要信息。"""
        num_map_units = len(self.semantic_map.memory_units)
        num_map_indexed = (
            self.semantic_map.faiss_index.ntotal if self.semantic_map.faiss_index else 0
        )
        num_map_spaces = len(self.semantic_map.memory_spaces)

        num_graph_nodes = self.nx_graph.number_of_nodes()
        num_graph_edges = self.nx_graph.number_of_edges()

        # 添加内存管理统计
        num_dirty_nodes = len(self._nodes_dirty_flag)
        num_modified_units = len(self._modified_units)
        num_deleted_units = len(self._deleted_units)
        num_modified_relationships = len(self._modified_relationships)
        num_deleted_relationships = len(self._deleted_relationships)

        summary = (
            f"--- SemanticGraph 摘要 ---\n"
            f"SemanticMap:\n"
            f"  - 记忆单元总数: {num_map_units}\n"
            f"  - 已索引向量数: {num_map_indexed}\n"
            f"  - 记忆空间数: {num_map_spaces} ({list(self.semantic_map.memory_spaces.keys())})\n"
            f"NetworkX Graph:\n"
            f"  - 节点数: {num_graph_nodes}\n"
            f"  - 边数 (关系数): {num_graph_edges}\n"
            f"内存管理:\n"
            f"  - 脏节点数: {num_dirty_nodes}\n"
            f"  - 待同步修改单元: {num_modified_units}\n"
            f"  - 待同步删除单元: {num_deleted_units}\n"
            f"  - 待同步修改关系: {num_modified_relationships}\n"
            f"  - 待同步删除关系: {num_deleted_relationships}\n"
            f"外部存储连接:\n"
            f"  - Neo4j: {'已连接' if self._neo4j_connection and self._neo4j_connection.neo4j_connected else '未连接'}\n"
            f"  - Milvus: {'已连接' if self.semantic_map._external_storage else '未连接'}\n"
            f"---------------------------\n"
        )
        print(summary)
        logging.info(summary.replace("\n", " | "))

    # ==============================
    # 与Hippo保持一致的方法
    # ==============================

    def get_all_memory_space_names(self) -> List[str]:
        """
        获取所有MemorySpace名称（递归，去重）。
        """
        return self.semantic_map.get_all_memory_space_names()

    def get_memory_space_structures(self) -> List[dict]:
        """
        递归导出所有MemorySpace嵌套结构（树/嵌套dict），
        每个ms展示：名称、unit uid列表、所有unit的raw_data字段全集、子空间。
        返回列表，每个元素为一个ms的结构。
        """
        return self.semantic_map.get_memory_space_structures()

    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        """去重单元列表"""
        return self.semantic_map.deduplicate_units(units)

    def filter_memory_units(
        self,
        candidate_units: Optional[List[MemoryUnit]] = None,
        filter_condition: Optional[dict] = None,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        # 支持ms_names为["ms:xxx", ...]格式
        normed = None
        if ms_names:
            normed = [(n[3:] if n.startswith("ms:") else n) for n in ms_names]
        return self.semantic_map.filter_memory_units(
            candidate_units=candidate_units,
            filter_condition=filter_condition,
            ms_names=normed,
            recursive=recursive,
        )

    def get_explicit_neighbors(
        self,
        uids: List[str],
        rel_type: Optional[str] = None,
        ms_names: Optional[List[str]] = None,
        direction: str = "successors",
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """
        获取显式邻居节点
        """
        result = []
        for uid in uids:
            if direction == "successors":
                neighbors = self.nx_graph.successors(uid)
            elif direction == "predecessors":
                neighbors = self.nx_graph.predecessors(uid)
            else:
                neighbors = set(self.nx_graph.successors(uid)) | set(
                    self.nx_graph.predecessors(uid)
                )
            for n in neighbors:
                if rel_type:
                    edge = (
                        self.nx_graph.get_edge_data(uid, n)
                        if direction != "predecessors"
                        else self.nx_graph.get_edge_data(n, uid)
                    )
                    if not edge or edge.get("type") != rel_type:
                        continue
                unit = self.semantic_map.get_unit(n)
                if unit is not None:
                    result.append(unit)

        if ms_names:
            ms_units = set(
                u.uid
                for u in self.get_units_in_memory_space(ms_names, recursive=recursive)
            )
            result = [u for u in result if u.uid in ms_units]
        return result

    def get_implicit_neighbors(
        self,
        uids: List[str],
        top_k: int = 5,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """
        获取多个起点单元的隐式邻居节点（基于语义相似性），对每个uid分别top-k，合并去重。
        """
        if isinstance(uids, str):
            uids = [uids]
        all_results = []
        seen = set()
        for uid in uids:
            unit = self.semantic_map.get_unit(uid)
            if not unit or unit.embedding is None:
                continue
            # 获取top_k相似单元
            if ms_names:
                units_with_scores = self.semantic_map.search_similarity_by_vector(
                    unit.embedding, k=top_k + 1, ms_names=ms_names
                )
            else:
                units_with_scores = self.semantic_map.search_similarity_by_vector(
                    unit.embedding, k=top_k + 1
                )
            for neighbor, _ in units_with_scores:
                if neighbor.uid != uid and neighbor.uid not in seen:
                    seen.add(neighbor.uid)
                    all_results.append(neighbor)
                if len(all_results) >= top_k * len(uids):
                    break
        return all_results

    # ==============================
    # 持久化方法（更新兼容性）
    # ==============================

    def save_graph(self, directory_path: str):
        """保存 SemanticGraph 的状态到指定目录。"""
        os.makedirs(directory_path, exist_ok=True)

        # 1. 保存 SemanticMap
        self.semantic_map.save_map(os.path.join(directory_path, "semantic_map_data"))

        # 2. 保存 NetworkX 图
        nx_graph_file = os.path.join(directory_path, "semantic_graph.gml")
        try:
            nx.write_gml(self.nx_graph, nx_graph_file)
        except Exception as e:
            logging.warning(f"以 GML 格式保存 NetworkX 图失败: {e}。尝试使用 pickle。")
            nx_graph_file_pkl = os.path.join(directory_path, "semantic_graph.pkl")
            with open(nx_graph_file_pkl, "wb") as f:
                pickle.dump(self.nx_graph, f)
            logging.info(f"NetworkX 图已使用 pickle 保存到 '{nx_graph_file_pkl}'。")

        # 3. 保存内存管理状态
        management_state = {
            "_nodes_access_counts": self._nodes_access_counts,
            "_nodes_last_accessed": self._nodes_last_accessed,
            "_nodes_dirty_flag": list(self._nodes_dirty_flag),
            "_modified_units": list(self._modified_units),
            "_deleted_units": list(self._deleted_units),
            "_modified_relationships": list(self._modified_relationships),
            "_deleted_relationships": list(self._deleted_relationships),
            "_relationship_cache": self._relationship_cache,
            "_relationships_access_counts": self._relationships_access_counts,
            "_relationships_last_accessed": self._relationships_last_accessed,
        }

        with open(os.path.join(directory_path, "management_state.pkl"), "wb") as f:
            pickle.dump(management_state, f)

        logging.info(f"SemanticGraph 已保存到目录: '{directory_path}'")

    @classmethod
    def load_graph(
        cls,
        directory_path: str,
        image_embedding_model_name: Optional[str] = None,
        text_embedding_model_name: Optional[str] = None,
    ) -> "SemanticGraph":
        """从指定目录加载 SemanticGraph 的状态。"""
        # 1. 加载 SemanticMap
        loaded_map = SemanticMap.load_map(
            os.path.join(directory_path, "semantic_map_data"),
            image_embedding_model_name=image_embedding_model_name,
            text_embedding_model_name=text_embedding_model_name,
        )

        instance = cls(semantic_map_instance=loaded_map)

        # 2. 加载 NetworkX 图
        nx_graph_file_gml = os.path.join(directory_path, "semantic_graph.gml")
        nx_graph_file_pkl = os.path.join(directory_path, "semantic_graph.pkl")

        if os.path.exists(nx_graph_file_gml):
            try:
                instance.nx_graph = nx.read_gml(nx_graph_file_gml)
                logging.info(f"NetworkX 图已从 GML 文件 '{nx_graph_file_gml}' 加载。")
            except Exception as e:
                logging.error(
                    f"从 GML 文件加载 NetworkX 图失败: {e}。检查是否存在 pickle 文件。"
                )
                if os.path.exists(nx_graph_file_pkl):
                    with open(nx_graph_file_pkl, "rb") as f:
                        instance.nx_graph = pickle.load(f)
                    logging.info(
                        f"NetworkX 图已从 pickle 文件 '{nx_graph_file_pkl}' 加载。"
                    )
                else:
                    logging.warning(
                        "GML 和 pickle 格式的 NetworkX 图文件均未找到或加载失败。图将为空。"
                    )
                    instance.nx_graph = nx.DiGraph()
        elif os.path.exists(nx_graph_file_pkl):
            with open(nx_graph_file_pkl, "rb") as f:
                instance.nx_graph = pickle.load(f)
            logging.info(f"NetworkX 图已从 pickle 文件 '{nx_graph_file_pkl}' 加载。")
        else:
            logging.warning(
                f"NetworkX 图文件在 '{directory_path}' 中未找到。图将为空。"
            )
            instance.nx_graph = nx.DiGraph()

        # 3. 加载内存管理状态
        management_state_file = os.path.join(directory_path, "management_state.pkl")
        if os.path.exists(management_state_file):
            try:
                with open(management_state_file, "rb") as f:
                    state = pickle.load(f)

                instance._nodes_access_counts = state.get("_nodes_access_counts", {})
                instance._nodes_last_accessed = state.get("_nodes_last_accessed", {})
                instance._nodes_dirty_flag = set(state.get("_nodes_dirty_flag", []))
                instance._modified_units = set(state.get("_modified_units", []))
                instance._deleted_units = set(state.get("_deleted_units", []))
                instance._modified_relationships = set(
                    state.get("_modified_relationships", [])
                )
                instance._deleted_relationships = set(
                    state.get("_deleted_relationships", [])
                )
                instance._relationship_cache = state.get("_relationship_cache", {})
                instance._relationships_access_counts = state.get(
                    "_relationships_access_counts", {}
                )
                instance._relationships_last_accessed = state.get(
                    "_relationships_last_accessed", {}
                )

                logging.info("内存管理状态已加载")
            except Exception as e:
                logging.warning(f"加载内存管理状态失败: {e}")

        logging.info(f"SemanticGraph 已从目录 '{directory_path}' 加载。")
        # 自动恢复 rel_types
        # instance.rel_types = set() # 移除
        return instance

    # ==============================
    # 兼容旧接口的方法
    # ==============================

    def export_to_neo4j(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> bool:
        """将SemanticGraph中的节点和关系导出到Neo4j数据库
        注意：此方法已废弃，推荐使用 sync_to_external() 方法"""

        logging.warning(
            "export_to_neo4j() 方法已废弃，推荐使用 sync_to_external() 方法"
        )

        try:
            # 创建Neo4j操作类
            neo4j_op = Neo4jOperator(
                neo4j_uri=uri,
                neo4j_user=user,
                neo4j_password=password,
                neo4j_database=database,
            )

            if not neo4j_op.neo4j_connected:
                logging.error("连接Neo4j失败，无法导出数据")
                return False

            # 导出所有记忆单元
            unit_success_count = 0
            for uid, unit in self.semantic_map.memory_units.items():
                if neo4j_op.add_unit(unit):
                    unit_success_count += 1
                else:
                    logging.warning(f"导出记忆单元 '{uid}' 到Neo4j失败")

            # 导出所有关系
            rel_success_count = 0
            for source, target, data in self.nx_graph.edges(data=True):
                # 获取关系类型，默认为"RELATED_TO"
                rel_type = data.get("type", "RELATED_TO")
                # 移除type，因为它已用作关系类型
                properties = {k: v for k, v in data.items() if k != "type"}

                if neo4j_op.add_relationship(source, target, rel_type, properties):
                    rel_success_count += 1
                else:
                    logging.warning(
                        f"导出从 '{source}' 到 '{target}' 的 '{rel_type}' 关系到Neo4j失败"
                    )

            logging.info(
                f"成功导出 {unit_success_count}/{len(self.semantic_map.memory_units)} 个记忆单元和 {rel_success_count}/{self.nx_graph.number_of_edges()} 个关系到Neo4j"
            )
            neo4j_op.close()

            return unit_success_count > 0 or rel_success_count > 0

        except Exception as e:
            logging.error(f"导出到Neo4j失败: {e}")
        return False

    def units_union(self, *args):
        """
        语义图谱层的并集操作，委托给SemanticMap。
        """
        return self.semantic_map.units_union(*args)

    def units_intersection(self, *args):
        """
        语义图谱层的交集操作，委托给SemanticMap。
        """
        return self.semantic_map.units_intersection(*args)

    def units_difference(self, arg1, arg2):
        """
        语义图谱层的差集操作，委托给SemanticMap。
        """
        return self.semantic_map.units_difference(arg1, arg2)

    def aggregate_results(
        self, memory_units: List[MemoryUnit]
    ) -> Dict[MemoryUnit, int]:
        counter = Counter(memory_units)
        return dict(counter)

    def add_memory_space_in_map(self, space: "MemorySpace"):
        """
        委托到底层SemanticMap，将MemorySpace对象注册到SemanticMap下。
        """
        return self.semantic_map.add_memory_space(space)
