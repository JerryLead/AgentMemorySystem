# Hippo.py

from datetime import datetime
import os
import pickle
import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx
from ..milvus_operator import MilvusOperator
from ..neo4j_operator import Neo4jOperator
from ..memory_unit import MemoryUnit
from .semantic_map import SemanticMap
from .semantic_map import MemorySpace


class SemanticGraph:
    """
    语义图谱 (SemanticGraph) 结合了 SemanticMap 的向量存储/搜索能力和 NetworkX 的图结构管理能力。
    它存储内存单元作为节点，并允许在它们之间定义显式的命名关系。
    查询可以利用显式图遍历和隐式语义相似性。
    """
    def __init__(self, semantic_map_instance: Optional[SemanticMap] = None):
        """
        初始化语义图谱。
        参数:
            semantic_map_instance (Optional[SemanticMap]): 一个 SemanticMap 实例。
                                                          如果为 None，将创建一个新的默认 SemanticMap。
        """
        self.semantic_map: SemanticMap = semantic_map_instance if semantic_map_instance else SemanticMap()
        self.nx_graph: nx.DiGraph = nx.DiGraph() # 使用 NetworkX有向图存储节点和显式关系
        logging.info("SemanticGraph 已初始化。")
        # 添加Neo4j连接跟踪
        self._neo4j_connection:Neo4jOperator = None  # 指向Neo4j的连接
        self._modified_relationships = set()  # 修改过的关系 (source_id, target_id, rel_type)
        self._deleted_relationships = set()   # 删除的关系
        self._modified_units = set()  # 修改过的内存单元
        self._deleted_units = set() # 删除的内存单元
        self._max_nodes_in_memory = 10000  # 内存中最大节点数
        self._nodes_access_counts = {}     # 节点访问计数，用于LRU算法
        self._nodes_last_accessed = {}     # 节点最后访问时间，用于LRU算法
        self._nodes_dirty_flag = set()     # 标记内存中已修改但尚未同步的节点
        # 关系内存管理
        self._max_relationships_in_memory = 100000  # 内存中最大关系数
        self._relationship_cache = {}  # 缓存关系属性 {(source_id, target_id, rel_type): properties}
        self._relationships_access_counts = {}  # 关系访问计数
        self._relationships_last_accessed = {}  # 关系最后访问时间

    def page_out_nodes(self, count: int = 100, strategy: str = "LRU") -> int:
        """
        将不常用节点从内存移出
        
        参数:
            count: 要移出的节点数量
            strategy: 换出策略，"LRU"(最近最少使用)或"LFU"(最不常使用)
        
        返回:
            实际移出的节点数量
        """
        if not hasattr(self.semantic_map, '_external_storage') or not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法执行节点换出")
            return 0
            
        if len(self.semantic_map.memory_units) <= self._max_nodes_in_memory:
            logging.debug("当前节点数量未超过最大限制，无需换出")
            return 0
        
        # 确定要换出的节点
        candidates = list(self.semantic_map.memory_units.keys())
        if strategy == "LRU":
            # 按最后访问时间排序
            candidates.sort(key=lambda uid: self._nodes_last_accessed.get(uid, 0))
        elif strategy == "LFU":
            # 按访问频率排序
            candidates.sort(key=lambda uid: self._nodes_access_counts.get(uid, 0))
        else:
            logging.warning(f"不支持的换出策略: {strategy}，使用LRU")
            candidates.sort(key=lambda uid: self._nodes_last_accessed.get(uid, 0))
        
        # 限制换出数量
        nodes_to_page_out = candidates[:min(count, len(candidates))]
        
        # 同步已修改的节点到外部存储
        synced_count = 0
        for uid in nodes_to_page_out:
            if uid in self._nodes_dirty_flag:
                unit = self.semantic_map.memory_units.get(uid)
                if unit:
                    # 同步到Milvus
                    space_names = []
                    for space_name, space in self.semantic_map.memory_spaces.items():
                        if uid in space.get_memory_uids():
                            space_names.append(space_name)
                    
                    success = self.semantic_map._external_storage.add_unit(unit, space_names)
                    if success:
                        self._nodes_dirty_flag.remove(uid)
                        synced_count += 1
                        logging.debug(f"节点 '{uid}' 已同步到外部存储")
        
        # 从内存中移除这些节点(但保留在图结构中)
        for uid in nodes_to_page_out:
            if uid in self.semantic_map.memory_units:
                del self.semantic_map.memory_units[uid]
                if uid in self._nodes_access_counts:
                    del self._nodes_access_counts[uid]
                if uid in self._nodes_last_accessed:
                    del self._nodes_last_accessed[uid]
                logging.debug(f"节点 '{uid}' 已从内存中移出")
        
        logging.info(f"已将 {len(nodes_to_page_out)} 个节点从内存移出，其中 {synced_count} 个节点已同步到外部存储")
        return len(nodes_to_page_out)

    def page_in_nodes(self, node_ids: List[str]) -> int:
        """
        从外部存储加载节点到内存
        
        参数:
            node_ids: 要加载的节点ID列表
        
        返回:
            成功加载的节点数量
        """
        if not hasattr(self.semantic_map, '_external_storage') or not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法从外部加载节点")
            return 0
        
        # 过滤已在内存中的节点
        ids_to_load = [uid for uid in node_ids if uid not in self.semantic_map.memory_units]
        if not ids_to_load:
            return 0
        
        # 从Milvus加载节点
        loaded_units = self.semantic_map._external_storage.get_units_batch(ids_to_load)
        
        # 添加到内存
        for unit in loaded_units:
            self.semantic_map.memory_units[unit.uid] = unit
            self._nodes_access_counts[unit.uid] = 1
            self._nodes_last_accessed[unit.uid] = datetime.now().timestamp()
            
            # 如果有向量，可能需要更新FAISS索引
            if unit.embedding is not None:
                # 选择一种方式更新索引
                # 方式1: 单独添加向量到已有索引(如果FAISS索引支持增量更新)
                if self.semantic_map.faiss_index and hasattr(self.semantic_map.faiss_index, 'add_with_ids'):
                    try:
                        internal_id = self.semantic_map._internal_faiss_id_counter
                        self.semantic_map._internal_faiss_id_counter += 1
                        self.semantic_map._uid_to_internal_faiss_id[unit.uid] = internal_id
                        
                        vector = unit.embedding.reshape(1, -1).astype(np.float32)
                        ids = np.array([internal_id], dtype=np.int64)
                        self.semantic_map.faiss_index.add_with_ids(vector, ids)
                    except Exception as e:
                        logging.error(f"向FAISS索引添加向量失败: {e}")
                
                # 方式2: 标记为需要重建索引，但不立即重建
                # 可以设置一个标志，在一定数量的更改后触发重建
        
        logging.info(f"已从外部存储加载 {len(loaded_units)} 个节点到内存")
        
        # 如果加载后内存节点数超过限制，触发节点换出
        if len(self.semantic_map.memory_units) > self._max_nodes_in_memory:
            nodes_to_page_out = len(self.semantic_map.memory_units) - self._max_nodes_in_memory
            self.page_out_nodes(count=nodes_to_page_out)
        
        return len(loaded_units)

    def _cache_relationship(self, source_uid: str, target_uid: str, relationship_type: str, properties: dict):
        """将关系缓存到内存"""
        rel_key = (source_uid, target_uid, relationship_type)
        self._relationship_cache[rel_key] = properties
        self._relationships_access_counts[rel_key] = 1
        self._relationships_last_accessed[rel_key] = datetime.now().timestamp()
        
        # 检查缓存大小，必要时清理
        if len(self._relationship_cache) > self._max_relationships_in_memory:
            self._clear_relationship_cache(int(self._max_relationships_in_memory * 0.2))  # 清理20%的关系缓存

    def _clear_relationship_cache(self, count: int = 100):
        """清理关系缓存"""
        if not self._relationship_cache:
            return
            
        # 按最后访问时间排序
        sorted_rels = sorted(
            self._relationship_cache.keys(),
            key=lambda k: self._relationships_last_accessed.get(k, 0)
        )
        
        # 移除最旧的关系
        for rel_key in sorted_rels[:count]:
            if rel_key in self._relationship_cache:
                del self._relationship_cache[rel_key]
            if rel_key in self._relationships_access_counts:
                del self._relationships_access_counts[rel_key]
            if rel_key in self._relationships_last_accessed:
                del self._relationships_last_accessed[rel_key]
        
        logging.debug(f"已清理 {count} 个关系缓存")

    def add_relationship(self, source_uid: str, target_uid: str, relationship_name: str, 
                        bidirectional: bool = False, **kwargs):
        """添加关系并跟踪修改"""
        # 添加到NetworkX图
        edge_attributes = {"type": relationship_name, **kwargs}
        self.nx_graph.add_edge(source_uid, target_uid, **edge_attributes)
        
        # 缓存关系属性
        self._cache_relationship(source_uid, target_uid, relationship_name, kwargs)
        
        # 记录修改
        self._modified_relationships.add((source_uid, target_uid, relationship_name))
        
        # 如果是双向关系
        if bidirectional:
            self.nx_graph.add_edge(target_uid, source_uid, **edge_attributes)
            self._cache_relationship(target_uid, source_uid, relationship_name, kwargs)
            self._modified_relationships.add((target_uid, source_uid, relationship_name))
            
        logging.info(f"已添加从 '{source_uid}' 到 '{target_uid}' 的关系 '{relationship_name}'")
        return True

    def get_relationship(self, source_uid: str, target_uid: str, relationship_name: Optional[str] = None) -> Dict:
        """获取关系属性，必要时从Neo4j加载"""
        # 先检查内存中是否有此关系
        if relationship_name:
            rel_key = (source_uid, target_uid, relationship_name)
            if rel_key in self._relationship_cache:
                # 更新访问统计
                self._relationships_access_counts[rel_key] = self._relationships_access_counts.get(rel_key, 0) + 1
                self._relationships_last_accessed[rel_key] = datetime.now().timestamp()
                return self._relationship_cache[rel_key]
            
            # 从NetworkX获取
            if self.nx_graph.has_edge(source_uid, target_uid):
                edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
                if edge_data and edge_data.get("type") == relationship_name:
                    # 缓存并返回
                    properties = {k: v for k, v in edge_data.items() if k != "type"}
                    self._cache_relationship(source_uid, target_uid, relationship_name, properties)
                    return properties
        else:
            # 获取所有类型的关系
            if self.nx_graph.has_edge(source_uid, target_uid):
                return self.nx_graph.get_edge_data(source_uid, target_uid)
        
        # 内存中没有，尝试从Neo4j加载
        if self._neo4j_connection:
            try:
                # 假设Neo4jOperator有get_relationship方法
                properties = self._neo4j_connection.get_relationship_properties(
                    source_uid, target_uid, relationship_name
                )
                if properties:
                    # 添加到NetworkX和缓存
                    if relationship_name:
                        self.nx_graph.add_edge(source_uid, target_uid, type=relationship_name, **properties)
                        self._cache_relationship(source_uid, target_uid, relationship_name, properties)
                    return properties
            except Exception as e:
                logging.error(f"从Neo4j加载关系失败: {e}")
        
        return {}  # 关系不存在

    def sync_to_external(self, force_full_sync: bool = False) -> Dict[str, int]:
        """
        将修改同步到外部存储(Neo4j和Milvus)
        
        参数:
            force_full_sync: 是否强制全量同步
            
        返回:
            同步统计信息
        """
        stats = {
            "nodes_synced": 0,
            "nodes_failed": 0,
            "relationships_synced": 0,
            "relationships_failed": 0
        }
        
        # 1. 同步节点到Milvus
        if hasattr(self.semantic_map, '_external_storage') and self.semantic_map._external_storage:
            # 确定要同步的节点
            nodes_to_sync = list(self.semantic_map.memory_units.keys()) if force_full_sync else list(self._nodes_dirty_flag)
            
            for uid in nodes_to_sync:
                unit = self.semantic_map.memory_units.get(uid)
                if not unit:
                    continue
                
                # 获取节点所属空间
                space_names = []
                for space_name, space in self.semantic_map.memory_spaces.items():
                    if uid in space.get_memory_uids():
                        space_names.append(space_name)
                
                # 同步到Milvus
                if self.semantic_map._external_storage.add_unit(unit, space_names):
                    stats["nodes_synced"] += 1
                    if uid in self._nodes_dirty_flag:
                        self._nodes_dirty_flag.remove(uid)
                else:
                    stats["nodes_failed"] += 1
            
            # 处理已删除的节点
            for uid in self.semantic_map._deleted_units:
                if self.semantic_map._external_storage.delete_unit(uid):
                    stats["nodes_synced"] += 1
                else:
                    stats["nodes_failed"] += 1
        
        # 2. 同步关系到Neo4j
        if self._neo4j_connection:
            # 处理修改的关系
            for source_uid, target_uid, rel_type in self._modified_relationships:
                # 获取关系属性
                edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
                if not edge_data:
                    continue
                
                properties = {k: v for k, v in edge_data.items() if k != "type"}
                
                # 同步到Neo4j
                if self._neo4j_connection.add_relationship(source_uid, target_uid, rel_type, properties):
                    stats["relationships_synced"] += 1
                    if (source_uid, target_uid, rel_type) in self._modified_relationships:
                        self._modified_relationships.remove((source_uid, target_uid, rel_type))
                else:
                    stats["relationships_failed"] += 1
            
            # 处理删除的关系
            for source_uid, target_uid, rel_type in self._deleted_relationships:
                if self._neo4j_connection.delete_relationship(source_uid, target_uid, rel_type):
                    stats["relationships_synced"] += 1
                    if (source_uid, target_uid, rel_type) in self._deleted_relationships:
                        self._deleted_relationships.remove((source_uid, target_uid, rel_type))
                else:
                    stats["relationships_failed"] += 1
        
        logging.info(f"同步完成。节点: 成功={stats['nodes_synced']}, 失败={stats['nodes_failed']}; "
                    f"关系: 成功={stats['relationships_synced']}, 失败={stats['relationships_failed']}")
        
        return stats
    
    def full_export(self) -> Dict[str, int]:
        """完整导出所有节点和关系到外部存储"""
        logging.info("开始全量导出...")
        
        # 获取所有还未加载到内存的节点ID
        if self._neo4j_connection:
            try:
                # 获取Neo4j中的所有节点ID
                all_node_ids = self._neo4j_connection.get_all_node_ids()
                
                # 过滤出不在内存中的节点
                missing_ids = [uid for uid in all_node_ids if uid not in self.semantic_map.memory_units]
                
                # 分批加载这些节点
                batch_size = 100
                for i in range(0, len(missing_ids), batch_size):
                    batch_ids = missing_ids[i:i+batch_size]
                    self.page_in_nodes(batch_ids)
                    
            except Exception as e:
                logging.error(f"获取所有节点ID失败: {e}")
                return {"nodes_synced": 0, "nodes_failed": 0, "relationships_synced": 0, "relationships_failed": 0}
        
        # 执行增量导出来同步所有数据
        return self.incremental_export()

    def incremental_export(self) -> Dict[str, int]:
        """增量导出修改过的节点和关系到外部存储"""
        return self.sync_to_external(force_full_sync=False)

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
        self.page_in_nodes([uid])
        unit = self.semantic_map.get_unit(uid)
        
        if unit:
            return unit
        else:
            logging.warning(f"节点 '{uid}' 在内存和外部存储中均不存在")
            return None

    def connect_to_neo4j(self,
                    uri: str = "bolt://localhost:7687", 
                    user: str = "neo4j", 
                    password: str = "password",
                    database: str = "neo4j"):
        """
        连接到Neo4j数据库
        """
        try:
            self._neo4j_connection = Neo4jOperator(
                neo4j_uri=uri,
                neo4j_user=user,
                neo4j_password=password,
                neo4j_database=database
            )
            
            if self._neo4j_connection.neo4j_connected:
                logging.info(f"已连接到Neo4j: {uri}")
                # 同时连接SemanticMap到Milvus
                self.semantic_map.connect_external_storage(
                    storage_type="milvus",
                    host="localhost",  # 可以参数化这些默认值
                    port="19530"
                )
                return True
            else:
                self._neo4j_connection = None
                logging.error("连接到Neo4j失败")
                return False
        except Exception as e:
            logging.error(f"初始化Neo4j连接失败: {e}")
            return False

    def add_relationship(self, source_uid: str, target_uid: str, relationship_name: str, bidirectional: bool = False, **kwargs):
        # 现有代码
        
        # 添加到修改跟踪
        self._modified_relationships.add((source_uid, target_uid, relationship_name))
        if bidirectional:
            self._modified_relationships.add((target_uid, source_uid, relationship_name))

    def delete_relationship(self, source_uid: str, target_uid: str, relationship_name: Optional[str] = None):
        # 现有代码
        
        # 添加到删除跟踪
        if relationship_name:
            self._deleted_relationships.add((source_uid, target_uid, relationship_name))
        else:
            # 如果没有指定关系名称，则添加所有相关关系到删除跟踪
            for _, _, data in self.nx_graph.edges([source_uid, target_uid], data=True):
                rel_type = data.get("type", "RELATED_TO")
                self._deleted_relationships.add((source_uid, target_uid, rel_type))

    def add_unit(self,
                unit: MemoryUnit,
                explicit_content_for_embedding: Optional[Any] = None,
                content_type_for_embedding: Optional[str] = None,
                space_names: Optional[List[str]] = None,
                rebuild_semantic_map_index_immediately: bool = False):
        """
        向图谱添加一个内存单元 (节点)。
        单元也会被添加到内部的 SemanticMap 中。
        参数:
            unit (MemoryUnit): 要添加的内存单元。
            explicit_content_for_embedding, content_type_for_embedding: 传递给 SemanticMap 用于嵌入生成。
            space_names (Optional[List[str]]): 要将此单元添加到的 SemanticMap 中的内存空间名称。
            rebuild_semantic_map_index_immediately (bool): 是否在添加后立即重建 SemanticMap 的 FAISS 索引。
        """
        # 1. 将单元添加到 SemanticMap
        self.semantic_map.add_unit(
            unit,
            explicit_content_for_embedding,
            content_type_for_embedding,
            space_names,
            rebuild_index_immediately=rebuild_semantic_map_index_immediately # 注意这里传递的是否立即重建map索引
        )
        
        # 2. 将单元ID作为节点添加到 NetworkX 图中 (如果尚不存在)
        if not self.nx_graph.has_node(unit.uid):
            # 可以在节点上存储来自 unit.raw_data 的一些属性，如果需要的话
            self.nx_graph.add_node(unit.uid, **unit.raw_data) 
            logging.debug(f"节点 '{unit.uid}' 已添加到 NetworkX 图。")
        else: # 如果节点已存在，可以选择更新其属性
            nx.set_node_attributes(self.nx_graph, {unit.uid: unit.raw_data})
            logging.debug(f"节点 '{unit.uid}' 的属性已在 NetworkX 图中更新。")


    def add_relationship(self,
                         source_uid: str,
                         target_uid: str,
                         relationship_name: str,
                         bidirectional: bool = False,
                         **kwargs: Any): # 允许添加其他关系属性
        """
        在两个已存在的内存单元 (节点) 之间添加一条显式关系 (边)。
        参数:
            source_uid (str): 源节点的ID。
            target_uid (str): 目标节点的ID。
            relationship_name (str): 关系的名称 (例如 "连接到", "依赖于", "父子")。
            bidirectional (bool): 如果为 True，则添加一条从 target 到 source 的具有相同名称的反向关系。
            **kwargs: 任何其他要存储为边属性的键值对。
        """
        if not self.semantic_map.get_unit(source_uid):
            logging.error(f"源节点 '{source_uid}' 不存在于 SemanticMap 中。无法添加关系。")
            return
        if not self.semantic_map.get_unit(target_uid):
            logging.error(f"目标节点 '{target_uid}' 不存在于 SemanticMap 中。无法添加关系。")
            return
        
        # 确保节点也存在于nx_graph中 (通常 add_unit 会处理)
        if not self.nx_graph.has_node(source_uid): self.nx_graph.add_node(source_uid)
        if not self.nx_graph.has_node(target_uid): self.nx_graph.add_node(target_uid)

        # 使用 relationship_name 作为边的 'type' 或 'label' 属性
        edge_attributes = {"type": relationship_name, **kwargs}
        self.nx_graph.add_edge(source_uid, target_uid, **edge_attributes)
        logging.info(f"已添加从 '{source_uid}' 到 '{target_uid}' 的关系 '{relationship_name}'。")

        if bidirectional:
            self.nx_graph.add_edge(target_uid, source_uid, **edge_attributes) # 注意：如果关系有方向性，反向关系可能需要不同名称/属性
            logging.info(f"已添加从 '{target_uid}' 到 '{source_uid}' 的双向关系 '{relationship_name}'。")

    def delete_unit(self, uid: str, rebuild_semantic_map_index_immediately: bool = False):
        """从图谱和底层的 SemanticMap 中删除一个内存单元及其所有相关关系。"""
        # 1. 从 SemanticMap 删除
        self.semantic_map.delete_unit(uid, rebuild_index_immediately=rebuild_semantic_map_index_immediately)
        
        # 2. 从 NetworkX 图中删除节点 (这会自动删除所有相关的边)
        if self.nx_graph.has_node(uid):
            self.nx_graph.remove_node(uid)
            logging.info(f"节点 '{uid}' 及其关系已从 NetworkX 图中删除。")
        else:
            logging.warning(f"尝试从 NetworkX 图中删除不存在的节点 '{uid}'。")

    def delete_relationship(self, source_uid: str, target_uid: str, relationship_name: Optional[str] = None):
        """
        删除两个节点之间的特定关系或所有关系。
        参数:
            source_uid (str): 源节点ID。
            target_uid (str): Target 节点ID。
            relationship_name (Optional[str]): 如果提供，则只删除具有此名称 (作为'type'属性) 的关系。
                                               否则，删除这两个节点之间的所有直接关系。
        """
        if not self.nx_graph.has_edge(source_uid, target_uid):
            logging.warning(f"节点 '{source_uid}' 和 '{target_uid}' 之间没有直接边。")
            return

        if relationship_name:
            # NetworkX DiGraph 可以有平行边，但 add_edge 通常会替换。
            # 如果允许多个同名关系，则需要更复杂的删除逻辑。
            # 假设每个 (source, target) 对之间特定类型的关系是唯一的。
            edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
            # 对于有向图，通常只有一个直接边。如果有多条边（MultiDiGraph），则需要迭代。
            if edge_data and edge_data.get("type") == relationship_name:
                self.nx_graph.remove_edge(source_uid, target_uid)
                logging.info(f"已删除从 '{source_uid}' 到 '{target_uid}' 的关系 '{relationship_name}'。")
            else:
                logging.warning(f"未找到从 '{source_uid}' 到 '{target_uid}' 的名为 '{relationship_name}' 的关系。")
        else: # 删除所有直接边
            self.nx_graph.remove_edge(source_uid, target_uid)
            logging.info(f"已删除从 '{source_uid}' 到 '{target_uid}' 的所有直接关系。")


    def get_unit_data(self, uid: str) -> Optional[MemoryUnit]:
        """从底层的 SemanticMap 检索内存单元对象。"""
        return self.semantic_map.get_unit(uid)

    def build_semantic_map_index(self):
        """构建底层 SemanticMap 的 FAISS 索引。"""
        self.semantic_map.build_index()

    # --- 查询API ---
    def search_similarity_in_graph(self,
                                   query_text: Optional[str] = None,
                                   query_embedding: Optional[np.ndarray] = None,
                                   query_image_path: Optional[str] = None,
                                   k: int = 5,
                                   space_name: Optional[str] = None,
                                   filter_uids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        在图谱中执行语义相似性搜索 (委托给 SemanticMap)。
        参数:
            query_text (Optional[str]): 查询文本。
            query_embedding (Optional[np.ndarray]): 查询向量。
            query_image_path (Optional[str]): 查询图像的路径。
            k (int): 返回结果数量。
            space_name (Optional[str]): 限制在 SemanticMap 中的特定内存空间内搜索。
            filter_uids (Optional[Set[str]]): 进一步限制搜索范围的单元ID集合。
        返回:
            List[Tuple[MemoryUnit, float]]: (MemoryUnit, 相似度得分) 列表。
        """
        if query_embedding is not None:
            return self.semantic_map.search_similarity_by_embedding(query_embedding, k, space_name, filter_uids)
        elif query_text is not None:
            return self.semantic_map.search_similarity_by_text(query_text, k, space_name, filter_uids)
        elif query_image_path is not None:
            return self.semantic_map.search_similarity_by_image(query_image_path, k, space_name, filter_uids)
        else:
            logging.warning("必须提供 query_text, query_embedding 或 query_image_path 之一进行相似性搜索。")
            return []

    def traverse_explicit_nodes(self,
                                uid: str,
                                relationship_type: Optional[str] = None, # 对应于边的 'type' 属性
                                direction: str = "successors", # "successors", "predecessors", or "all"
                                space_name: Optional[str] = None) -> List[MemoryUnit]:
        """
        遍历与给定节点通过显式关系连接的邻居节点。
        参数:
            uid (str): 起始节点的ID。
            relationship_type (Optional[str]): 要筛选的关系类型 (边属性 'type')。如果为 None，则不按类型筛选。
            direction (str): 遍历方向:
                             "successors" (默认): 查找 uid 指向的节点 (子节点/出边)。
                             "predecessors": 查找指向 uid 的节点 (父节点/入边)。
                             "all": 查找双向的邻居。
            space_name (Optional[str]): 如果提供，则仅返回那些也存在于 SemanticMap 中指定内存空间的邻居。
        返回:
            List[MemoryUnit]: 符合条件的邻居 MemoryUnit 对象列表。
        """
        if not self.nx_graph.has_node(uid):
            logging.warning(f"节点 '{uid}' 不在图中，无法遍历。")
            return []

        neighbor_ids: Set[str] = set()
        if direction == "successors":
            for successor in self.nx_graph.successors(uid):
                if relationship_type:
                    edge_data = self.nx_graph.get_edge_data(uid, successor)
                    # 对于有向图，通常只有一个直接边。如果是MultiDiGraph，需要检查所有边。
                    # 假设默认的DiGraph，get_edge_data返回第一个找到的边的属性。
                    # 如果一个 (u,v) 对有多条不同类型的边，这个逻辑需要调整为检查所有边。
                    # 对于简单的DiGraph，如果 (u,v) 存在，则只有一条边。
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
            # 获取所有邻居 (包括前驱和后继)
            all_neighbors_temp = set(self.nx_graph.successors(uid))
            all_neighbors_temp.update(self.nx_graph.predecessors(uid))
            
            for neighbor in all_neighbors_temp:
                # 检查 (uid, neighbor) 或 (neighbor, uid) 的边
                passes_filter = False
                if not relationship_type:
                    passes_filter = True
                else:
                    if self.nx_graph.has_edge(uid, neighbor) and self.nx_graph.get_edge_data(uid, neighbor).get("type") == relationship_type:
                        passes_filter = True
                    elif self.nx_graph.has_edge(neighbor, uid) and self.nx_graph.get_edge_data(neighbor, uid).get("type") == relationship_type:
                        passes_filter = True
                if passes_filter:
                    neighbor_ids.add(neighbor)
        else:
            logging.warning(f"无效的遍历方向: '{direction}'。应为 'successors', 'predecessors', 或 'all'。")
            return []

        # 根据 space_name 过滤 (如果提供)
        if space_name:
            space = self.semantic_map.get_memory_space(space_name)
            if space:
                space_uids = space.get_memory_uids()
                neighbor_ids.intersection_update(space_uids) # 只保留也在空间内的ID
            else:
                logging.warning(f"内存空间 '{space_name}' 未找到，无法按空间过滤邻居。")
                return [] # 如果指定了空间但空间不存在，则不返回任何结果

        # 获取 MemoryUnit 对象
        results: List[MemoryUnit] = []
        for nid in neighbor_ids:
            unit = self.semantic_map.get_unit(nid)
            if unit:
                results.append(unit)
        return results

    def traverse_implicit_nodes(self,
                                uid: str,
                                k: int = 5,
                                space_name: Optional[str] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        基于语义相似性查找与给定节点隐式相关的节点。
        参数:
            uid (str): 起始节点的ID。
            k (int): 要查找的相似邻居数量。
            space_name (Optional[str]): 如果提供，则在 SemanticMap 中的指定内存空间内限制搜索。
        返回:
            List[Tuple[MemoryUnit, float]]: (MemoryUnit, 相似度得分) 列表，不包括 uid 本身。
        """
        start_unit = self.semantic_map.get_unit(uid)
        if not start_unit or start_unit.embedding is None:
            logging.warning(f"节点 '{uid}' 不存在或没有向量，无法进行隐式遍历。")
            return []

        # 搜索时排除自身 (如果 SemanticMap 的搜索结果可能包含查询项本身)
        # k+1 然后过滤，或者让 SemanticMap 的搜索处理过滤（如果它支持的话）
        # 这里假设 search_similarity_by_embedding 返回的结果不包含查询向量本身（除非它在数据集中且非常相似）
        # 通常，我们会获取 k+1 个结果，然后手动排除 uid。
        
        # 创建一个过滤器，排除 uid 本身
        filter_ids_to_exclude_self = {uid for uid in self.semantic_map.memory_units.keys() if uid != uid}
        
        # 如果指定了空间，则 filter_ids_to_exclude_self 会被 search_similarity_by_embedding 内部的 space_name 逻辑覆盖或合并。
        # 我们需要确保 uid 本身被排除。
        # 一个更简单的方法是获取k+1个结果，然后从结果中移除uid。

        similar_units_with_scores = self.semantic_map.search_similarity_by_embedding(
            start_unit.embedding,
            k=k + 1, # 获取稍多一些，以防 uid 是最相似的
            space_name=space_name
        )
        
        results: List[Tuple[MemoryUnit, float]] = []
        for unit, score in similar_units_with_scores:
            if unit.uid != uid: # 排除起始节点本身
                results.append((unit, score))
            if len(results) >= k: # 如果已达到k个结果
                break
        return results

    # --- MemorySpace 相关 (通过 SemanticMap 操作) ---
    def create_memory_space_in_map(self, space_name: str) -> MemorySpace:
        """在底层的 SemanticMap 中创建或获取一个内存空间。"""
        return self.semantic_map.create_memory_space(space_name)

    def add_unit_to_space_in_map(self, uid: str, space_name: str):
        """将一个内存单元添加到 SemanticMap 中的指定内存空间。"""
        self.semantic_map.add_unit_to_space(uid, space_name)

    # --- 持久化 ---
    def save_graph(self, directory_path: str):
        """
        将 SemanticGraph 的状态保存到指定目录。
        包括 SemanticMap 的数据和 NetworkX 图的结构。
        参数:
            directory_path (str): 保存文件的目录。
        """
        os.makedirs(directory_path, exist_ok=True)
        
        # 1. 保存 SemanticMap
        self.semantic_map.save_map(os.path.join(directory_path, "semantic_map_data"))
        
        # 2. 保存 NetworkX 图 (例如，使用 GML 或 pickle)
        nx_graph_file = os.path.join(directory_path, "semantic_graph.gml")
        try:
            nx.write_gml(self.nx_graph, nx_graph_file)
        except Exception as e: # GML 可能不支持所有数据类型作为属性，pickle 更通用
            logging.warning(f"以 GML 格式保存 NetworkX 图失败: {e}。尝试使用 pickle。")
            nx_graph_file_pkl = os.path.join(directory_path, "semantic_graph.pkl")
            with open(nx_graph_file_pkl, "wb") as f:
                pickle.dump(self.nx_graph, f)
            logging.info(f"NetworkX 图已使用 pickle 保存到 '{nx_graph_file_pkl}'。")

        logging.info(f"SemanticGraph 已保存到目录: '{directory_path}'")

    @classmethod
    def load_graph(cls, directory_path: str,
                   image_embedding_model_name: Optional[str] = None,
                   text_embedding_model_name: Optional[str] = None) -> 'SemanticGraph':
        """
        从指定目录加载 SemanticGraph 的状态。
        参数:
            directory_path (str): 从中加载文件的目录。
            image_embedding_model_name, text_embedding_model_name: 可选，用于覆盖加载 SemanticMap 时的模型名称。
        返回:
            SemanticGraph: 加载的 SemanticGraph 实例。
        """
        # 1. 加载 SemanticMap
        loaded_map = SemanticMap.load_map(
            os.path.join(directory_path, "semantic_map_data"),
            image_embedding_model_name=image_embedding_model_name,
            text_embedding_model_name=text_embedding_model_name
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
                logging.error(f"从 GML 文件加载 NetworkX 图失败: {e}。检查是否存在 pickle 文件。")
                if os.path.exists(nx_graph_file_pkl):
                    with open(nx_graph_file_pkl, "rb") as f:
                        instance.nx_graph = pickle.load(f)
                    logging.info(f"NetworkX 图已从 pickle 文件 '{nx_graph_file_pkl}' 加载。")
                else:
                    logging.warning("GML 和 pickle 格式的 NetworkX 图文件均未找到或加载失败。图将为空。")
                    instance.nx_graph = nx.DiGraph() # 创建一个空图
        elif os.path.exists(nx_graph_file_pkl):
             with open(nx_graph_file_pkl, "rb") as f:
                instance.nx_graph = pickle.load(f)
             logging.info(f"NetworkX 图已从 pickle 文件 '{nx_graph_file_pkl}' 加载。")
        else:
            logging.warning(f"NetworkX 图文件 (GML 或 pickle) 在 '{directory_path}' 中未找到。图将为空。")
            instance.nx_graph = nx.DiGraph()

        logging.info(f"SemanticGraph 已从目录 '{directory_path}' 加载。")
        return instance
    
    def export_to_neo4j(
        self, 
        uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password",
        database: str = "neo4j"
        ) -> bool:
        """将SemanticGraph中的节点和关系导出到Neo4j数据库"""
        # try:
        #     # 延迟导入，避免强制依赖
        #     from neo4j_operator import Neo4jOperator
        # except ImportError:
        #     logging.error("未找到neo4j_operator模块，请确保已安装neo4j并创建了neo4j_operator.py")
        #     return False
        
        try:
            # 创建Neo4j操作类 - 修改参数名以匹配Neo4jOperator的定义
            neo4j_op = Neo4jOperator(
                neo4j_uri=uri,           # 从uri改为neo4j_uri
                neo4j_user=user,         # 从user改为neo4j_user
                neo4j_password=password, # 从password改为neo4j_password
                neo4j_database=database  # 从database改为neo4j_database
            )
            
            if not neo4j_op.neo4j_connected:
                logging.error("连接Neo4j失败，无法导出数据")
                return False
            
            # 导出所有内存单元
            unit_success_count = 0
            for uid, unit in self.semantic_map.memory_units.items():
                if neo4j_op.add_unit(unit):
                    unit_success_count += 1
                else:
                    logging.warning(f"导出内存单元 '{uid}' 到Neo4j失败")
            
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
                    logging.warning(f"导出从 '{source}' 到 '{target}' 的 '{rel_type}' 关系到Neo4j失败")
            
            logging.info(f"成功导出 {unit_success_count}/{len(self.semantic_map.memory_units)} 个内存单元和 {rel_success_count}/{self.nx_graph.number_of_edges()} 个关系到Neo4j")
            neo4j_op.close()
            
            return unit_success_count > 0 or rel_success_count > 0
            
        except Exception as e:
            logging.error(f"导出到Neo4j失败: {e}")
        return False

    def display_graph_summary(self):
        """打印图谱的摘要信息。"""
        num_map_units = len(self.semantic_map.memory_units)
        num_map_indexed = self.semantic_map.faiss_index.ntotal if self.semantic_map.faiss_index else 0
        num_map_spaces = len(self.semantic_map.memory_spaces)
        
        num_graph_nodes = self.nx_graph.number_of_nodes()
        num_graph_edges = self.nx_graph.number_of_edges()

        summary = (
            f"--- SemanticGraph 摘要 ---\n"
            f"SemanticMap:\n"
            f"  - 内存单元总数: {num_map_units}\n"
            f"  - 已索引向量数: {num_map_indexed}\n"
            f"  - 内存空间数: {num_map_spaces} ({list(self.semantic_map.memory_spaces.keys())})\n"
            f"NetworkX Graph:\n"
            f"  - 节点数: {num_graph_nodes}\n"
            f"  - 边数 (关系数): {num_graph_edges}\n"
            f"---------------------------\n"
        )
        print(summary)
        logging.info(summary.replace("\n", " | "))