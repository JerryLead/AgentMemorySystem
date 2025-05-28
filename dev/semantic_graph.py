import logging
from datetime import datetime
import os
import pickle
from typing import Dict, Any, Optional, List, Set, Tuple

import numpy as np
import networkx as nx

from .semantic_map import SemanticMap
from .neo4j_operator import Neo4jOperator
from .memory_unit import MemoryUnit
from .memory_space import MemorySpace

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
        self._neo4j_connection: Optional[Neo4jOperator] = None  # 指向Neo4j的连接
        self._modified_relationships = set()  # 修改过的关系 (source_id, target_id, rel_type)
        self._deleted_relationships = set()   # 删除的关系
        self._modified_units = set()  # 修改过的内存单元
        self._deleted_units = set() # 删除的内存单元
        
        # 内存管理配置
        self._max_nodes_in_memory = 10000  # 内存中最大节点数
        self._nodes_access_counts = {}     # 节点访问计数，用于LFU算法
        self._nodes_last_accessed = {}     # 节点最后访问时间，用于LRU算法
        self._nodes_dirty_flag = set()     # 标记内存中已修改但尚未同步的节点
        
        # 关系内存管理
        self._max_relationships_in_memory = 100000  # 内存中最大关系数
        self._relationship_cache = {}  # 缓存关系属性 {(source_id, target_id, rel_type): properties}
        self._relationships_access_counts = {}  # 关系访问计数
        self._relationships_last_accessed = {}  # 关系最后访问时间

    def connect_to_neo4j(self,
                        uri: str = "bolt://localhost:7687", 
                        user: str = "neo4j", 
                        password: str = "password",
                        database: str = "neo4j",
                        # Milvus连接参数
                        milvus_host: str = "localhost",
                        milvus_port: str = "19530",
                        milvus_user: str = "",
                        milvus_password: str = "",
                        milvus_collection: str = "hippo_memory_units") -> bool:
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
                embedding_dim=self.semantic_map.embedding_dim
            )
            
            if self._neo4j_connection.neo4j_connected:
                logging.info(f"已连接到Neo4j: {uri}")
                
                # 设置SemanticMap的外部存储为Neo4jOperator的Milvus连接
                self.semantic_map._external_storage = self._neo4j_connection.milvus_operator
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

    def add_unit(self,
                unit: MemoryUnit,
                explicit_content_for_embedding: Optional[Any] = None,
                content_type_for_embedding: Optional[str] = None,
                space_names: Optional[List[str]] = None,
                rebuild_semantic_map_index_immediately: bool = False):
        """
        向图谱添加一个内存单元 (节点)。
        单元也会被添加到内部的 SemanticMap 中。
        """
        # 检查内存限制，必要时触发换页
        if len(self.semantic_map.memory_units) >= self._max_nodes_in_memory:
            self.page_out_nodes(count=int(self._max_nodes_in_memory * 0.1))  # 换出10%
        
        # 1. 将单元添加到 SemanticMap
        self.semantic_map.add_unit(
            unit,
            explicit_content_for_embedding,
            content_type_for_embedding,
            space_names,
            rebuild_index_immediately=rebuild_semantic_map_index_immediately
        )
        
        # 2. 将单元ID作为节点添加到 NetworkX 图中
        if not self.nx_graph.has_node(unit.uid):
            # 在节点上存储一些基本属性
            node_attrs = {
                "uid": unit.uid,
                "created": str(datetime.now()),
                **{k: v for k, v in unit.raw_data.items() if isinstance(v, (str, int, float, bool))}
            }
            self.nx_graph.add_node(unit.uid, **node_attrs)
            logging.debug(f"节点 '{unit.uid}' 已添加到 NetworkX 图。")
        else:
            # 如果节点已存在，更新其属性
            node_attrs = {
                "uid": unit.uid,
                "updated": str(datetime.now()),
                **{k: v for k, v in unit.raw_data.items() if isinstance(v, (str, int, float, bool))}
            }
            nx.set_node_attributes(self.nx_graph, {unit.uid: node_attrs})
            logging.debug(f"节点 '{unit.uid}' 的属性已在 NetworkX 图中更新。")
        
        # 3. 更新访问统计和标记为脏数据
        self._nodes_access_counts[unit.uid] = self._nodes_access_counts.get(unit.uid, 0) + 1
        self._nodes_last_accessed[unit.uid] = datetime.now().timestamp()
        self._nodes_dirty_flag.add(unit.uid)
        self._modified_units.add(unit.uid)

    def add_relationship(self,
                         source_uid: str,
                         target_uid: str,
                         relationship_name: str,
                         bidirectional: bool = False,
                         **kwargs: Any):
        """
        在两个已存在的内存单元 (节点) 之间添加一条显式关系 (边)。
        """
        # 检查源节点和目标节点是否存在
        source_unit = self.get_unit(source_uid)
        target_unit = self.get_unit(target_uid)
        
        if not source_unit:
            logging.error(f"源节点 '{source_uid}' 不存在。无法添加关系。")
            return False
        if not target_unit:
            logging.error(f"目标节点 '{target_uid}' 不存在。无法添加关系。")
            return False
        
        # 确保节点也存在于nx_graph中
        if not self.nx_graph.has_node(source_uid):
            self.nx_graph.add_node(source_uid, uid=source_uid)
        if not self.nx_graph.has_node(target_uid):
            self.nx_graph.add_node(target_uid, uid=target_uid)

        # 添加关系到NetworkX图
        edge_attributes = {"type": relationship_name, "created": str(datetime.now()), **kwargs}
        self.nx_graph.add_edge(source_uid, target_uid, **edge_attributes)
        
        # 缓存关系属性
        self._cache_relationship(source_uid, target_uid, relationship_name, kwargs)
        
        # 记录修改
        self._modified_relationships.add((source_uid, target_uid, relationship_name))
        
        logging.info(f"已添加从 '{source_uid}' 到 '{target_uid}' 的关系 '{relationship_name}'。")

        if bidirectional:
            # 添加反向关系
            self.nx_graph.add_edge(target_uid, source_uid, **edge_attributes)
            self._cache_relationship(target_uid, source_uid, relationship_name, kwargs)
            self._modified_relationships.add((target_uid, source_uid, relationship_name))
            logging.info(f"已添加从 '{target_uid}' 到 '{source_uid}' 的双向关系 '{relationship_name}'。")
        
        return True

    def delete_unit(self, uid: str, rebuild_semantic_map_index_immediately: bool = False):
        """从图谱和底层的 SemanticMap 中删除一个内存单元及其所有相关关系。"""
        # 1. 从 SemanticMap 删除
        self.semantic_map.delete_unit(uid, rebuild_index_immediately=rebuild_semantic_map_index_immediately)
        
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

    def delete_relationship(self, source_uid: str, target_uid: str, relationship_name: Optional[str] = None):
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
                self._deleted_relationships.add((source_uid, target_uid, relationship_name))
                
                # 从关系缓存中移除
                rel_key = (source_uid, target_uid, relationship_name)
                if rel_key in self._relationship_cache:
                    del self._relationship_cache[rel_key]
                if rel_key in self._relationships_access_counts:
                    del self._relationships_access_counts[rel_key]
                if rel_key in self._relationships_last_accessed:
                    del self._relationships_last_accessed[rel_key]
                
                logging.info(f"已删除从 '{source_uid}' 到 '{target_uid}' 的关系 '{relationship_name}'。")
                return True
            else:
                logging.warning(f"未找到从 '{source_uid}' 到 '{target_uid}' 的名为 '{relationship_name}' 的关系。")
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
        loaded_count = self.page_in_nodes([uid])
        if loaded_count > 0:
            unit = self.semantic_map.get_unit(uid)
            if unit:
                return unit
        
        logging.warning(f"节点 '{uid}' 在内存和外部存储中均不存在")
        return None
    

    ### 内存管理和换页策略 ###

    def page_out_nodes(self, count: int = 100, strategy: str = "LRU") -> int:
        """
        将不常用节点从内存移出到外部存储
        """
        if not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法执行节点换出")
            return 0
            
        # 修复：即使未超过限制也允许换出，用于测试
        current_count = len(self.semantic_map.memory_units)
        if current_count == 0:
            logging.debug("内存中没有节点可以换出")
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
        
        # 限制换出数量，但至少换出1个（如果有的话）
        actual_count = min(count, len(candidates))
        if actual_count == 0:
            return 0
            
        nodes_to_page_out = candidates[:actual_count]
        
        # 同步已修改的节点到外部存储
        synced_count = 0
        for uid in nodes_to_page_out:
            if uid in self._nodes_dirty_flag:
                unit = self.semantic_map.memory_units.get(uid)
                if unit:
                    # 获取节点所属空间
                    space_names = []
                    for space_name, space in self.semantic_map.memory_spaces.items():
                        if uid in space.get_memory_uids():
                            space_names.append(space_name)
                    
                    # 同步到外部存储
                    success = self.semantic_map._external_storage.add_unit(unit, space_names)
                    if success:
                        self._nodes_dirty_flag.discard(uid)  # 使用discard避免KeyError
                        synced_count += 1
                        logging.debug(f"节点 '{uid}' 已同步到外部存储")
        
        # 从内存中移除这些节点(但保留在图结构中)
        removed_count = 0
        for uid in nodes_to_page_out:
            if uid in self.semantic_map.memory_units:
                del self.semantic_map.memory_units[uid]
                removed_count += 1
                
                # 从FAISS索引中移除
                if uid in self.semantic_map._uid_to_internal_faiss_id and self.semantic_map.faiss_index:
                    try:
                        internal_id = self.semantic_map._uid_to_internal_faiss_id[uid]
                        if hasattr(self.semantic_map.faiss_index, 'remove_ids'):
                            self.semantic_map.faiss_index.remove_ids(np.array([internal_id], dtype=np.int64))
                        del self.semantic_map._uid_to_internal_faiss_id[uid]
                        logging.debug(f"节点 '{uid}' 已从FAISS索引中移除")
                    except Exception as e:
                        logging.error(f"从FAISS索引移除节点 '{uid}' 失败: {e}")
                
                # 清理访问统计
                self._nodes_access_counts.pop(uid, None)
                self._nodes_last_accessed.pop(uid, None)
                logging.debug(f"节点 '{uid}' 已从内存中移出")
        
        logging.info(f"已将 {removed_count} 个节点从内存移出，其中 {synced_count} 个节点已同步到外部存储")
        return removed_count

    def page_in_nodes(self, node_ids: List[str]) -> int:
        """
        从外部存储加载节点到内存
        """
        if not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法从外部加载节点")
            return 0
        
        # 过滤已在内存中的节点
        ids_to_load = [uid for uid in node_ids if uid not in self.semantic_map.memory_units]
        if not ids_to_load:
            return 0
        
        # 从外部存储加载节点
        loaded_units = self.semantic_map._external_storage.get_units_batch(ids_to_load)
        
        # 添加到内存
        for unit in loaded_units:
            # 检查内存限制
            if len(self.semantic_map.memory_units) >= self._max_nodes_in_memory:
                self.page_out_nodes(count=1)
            
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

    def _cache_relationship(self, source_uid: str, target_uid: str, relationship_type: str, properties: dict):
        """将关系缓存到内存"""
        rel_key = (source_uid, target_uid, relationship_type)
        self._relationship_cache[rel_key] = properties
        self._relationships_access_counts[rel_key] = 1
        self._relationships_last_accessed[rel_key] = datetime.now().timestamp()
        
        # 检查缓存大小，必要时清理
        if len(self._relationship_cache) > self._max_relationships_in_memory:
            self._clear_relationship_cache(int(self._max_relationships_in_memory * 0.2))

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

    ### LLM缓存顾问相关功能 ###
    
    def set_llm_cache_advisor(self, llm_client, model_name: str = "gpt-4"):
        """为语义图设置LLM缓存顾问"""
        self.semantic_map.set_llm_cache_advisor(llm_client, model_name)
        logging.info("SemanticGraph LLM缓存顾问已启用")

    def unpersist_nodes(self, count: int = 100, strategy: str = "LLM", query_context: Optional[str] = None) -> int:
        """
        将不常用节点从内存移除并持久化到外部存储
        支持LLM策略进行智能决策
        """
        if not self.semantic_map._external_storage:
            logging.warning("未连接外部存储，无法执行节点换出")
            return 0
            
        current_count = len(self.semantic_map.memory_units)
        if current_count == 0:
            logging.debug("内存中没有节点可以换出")
            return 0
        
        if strategy == "LLM" and hasattr(self.semantic_map, '_llm_cache_advisor'):
            # 使用LLM策略
            self.semantic_map._unpersist_least_used_units(count, query_context)
            return count
        else:
            # 使用传统策略 (LRU/LFU)
            candidates = list(self.semantic_map.memory_units.keys())
            if strategy == "LRU":
                candidates.sort(key=lambda uid: self._nodes_last_accessed.get(uid, 0))
            elif strategy == "LFU":
                candidates.sort(key=lambda uid: self._nodes_access_counts.get(uid, 0))
            
            actual_count = min(count, len(candidates))
            if actual_count == 0:
                return 0
                
            nodes_to_page_out = candidates[:actual_count]
            
            # 执行换出逻辑...
            return len(nodes_to_page_out)

    def search_with_llm_cache(self, 
                            query_text: str,
                            k: int = 5,
                            space_name: Optional[str] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        带LLM缓存优化的搜索
        """
        # 执行搜索
        results = self.semantic_map.search_similarity_by_text(query_text, k, space_name)
        
        # 记录查询和访问的单元
        accessed_uids = [unit.uid for unit, _ in results]
        self.semantic_map.record_query(query_text, accessed_uids)
        
        # 可选：基于查询预取相关单元
        if hasattr(self.semantic_map, '_llm_cache_advisor') and self.semantic_map._llm_cache_advisor:
            try:
                # 获取可能需要预取的单元ID（从外部存储）
                if self.semantic_map._external_storage:
                    # 这里需要实现获取外部存储中所有单元ID的方法
                    # available_units = self.semantic_map._external_storage.get_all_unit_ids()
                    # prefetch_uids = self.semantic_map._llm_cache_advisor.recommend_prefetch(
                    #     query_text, available_units, prefetch_count=3
                    # )
                    # self.cache_nodes(prefetch_uids)
                    pass
            except Exception as e:
                logging.error(f"LLM预取推荐失败: {e}")
        
        return results
    
    ### 数据库操作
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
                    properties = {k: v for k, v in edge_data.items() if k not in ["type", "created", "updated"]}
                    self._cache_relationship(source_uid, target_uid, relationship_name, properties)
                    return properties
        else:
            # 获取所有类型的关系
            if self.nx_graph.has_edge(source_uid, target_uid):
                return self.nx_graph.get_edge_data(source_uid, target_uid)
        
        # 内存中没有，尝试从Neo4j加载
        if self._neo4j_connection:
            try:
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
        
        return {}

    def sync_to_external(self, force_full_sync: bool = False) -> Dict[str, int]:
        """
        将修改同步到外部存储(Neo4j和Milvus)
        """
        stats = {
            "nodes_synced": 0,
            "nodes_failed": 0,
            "relationships_synced": 0,
            "relationships_failed": 0
        }
        
        # 1. 同步节点到Milvus
        if self.semantic_map._external_storage:
            # 确定要同步的节点
            nodes_to_sync = list(self.semantic_map.memory_units.keys()) if force_full_sync else list(self._nodes_dirty_flag)
            
            for uid in nodes_to_sync:
                unit = self.semantic_map.memory_units.get(uid)
                if not unit:
                    stats["nodes_failed"] += 1
                    continue
                
                # 获取节点所属空间
                space_names = []
                for space_name, space in self.semantic_map.memory_spaces.items():
                    if uid in space.get_memory_uids():
                        space_names.append(space_name)
                
                # 同步到Milvus
                if self.semantic_map._external_storage.add_unit(unit, space_names):
                    stats["nodes_synced"] += 1
                    self._nodes_dirty_flag.discard(uid)
                else:
                    stats["nodes_failed"] += 1
            
            # 处理已删除的节点
            for uid in list(self._deleted_units):
                if self.semantic_map._external_storage.delete_unit(uid):
                    self._deleted_units.discard(uid)
                    stats["nodes_synced"] += 1
                else:
                    stats["nodes_failed"] += 1
        
        # 2. 同步关系到Neo4j
        if self._neo4j_connection:
            # 处理修改的关系
            for source_uid, target_uid, rel_type in list(self._modified_relationships):
                # 获取关系属性
                edge_data = self.nx_graph.get_edge_data(source_uid, target_uid)
                if not edge_data:
                    stats["relationships_failed"] += 1
                    continue
                
                # 过滤掉系统属性
                properties = {k: v for k, v in edge_data.items() 
                             if k not in ["type", "created", "updated"]}
                
                # 同步到Neo4j
                if self._neo4j_connection.add_relationship(source_uid, target_uid, rel_type, properties):
                    stats["relationships_synced"] += 1
                    self._modified_relationships.discard((source_uid, target_uid, rel_type))
                else:
                    stats["relationships_failed"] += 1
            
            # 处理删除的关系
            for source_uid, target_uid, rel_type in list(self._deleted_relationships):
                if self._neo4j_connection.delete_relationship(source_uid, target_uid, rel_type):
                    stats["relationships_synced"] += 1
                    self._deleted_relationships.discard((source_uid, target_uid, rel_type))
                else:
                    stats["relationships_failed"] += 1
        
        logging.info(f"同步完成。节点: 成功={stats['nodes_synced']}, 失败={stats['nodes_failed']}; "
                    f"关系: 成功={stats['relationships_synced']}, 失败={stats['relationships_failed']}")
        
        return stats

    def incremental_export(self) -> Dict[str, int]:
        """增量导出修改过的节点和关系到外部存储"""
        return self.sync_to_external(force_full_sync=False)

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
                if missing_ids:
                    batch_size = 100
                    for i in range(0, len(missing_ids), batch_size):
                        batch_ids = missing_ids[i:i+batch_size]
                        self.page_in_nodes(batch_ids)
                    
                    logging.info(f"已从外部存储加载 {len(missing_ids)} 个缺失节点")
                    
            except Exception as e:
                logging.error(f"获取所有节点ID失败: {e}")
        
        # 执行全量同步
        return self.sync_to_external(force_full_sync=True)

    # 保持原有的查询API不变
    def get_unit_data(self, uid: str) -> Optional[MemoryUnit]:
        """从底层的 SemanticMap 检索内存单元对象。"""
        return self.get_unit(uid)

    def build_semantic_map_index(self):
        """构建底层 SemanticMap 的 FAISS 索引。"""
        self.semantic_map.build_index()

    def search_similarity_in_graph(self,
                                   query_text: Optional[str] = None,
                                   query_embedding: Optional[np.ndarray] = None,
                                   query_image_path: Optional[str] = None,
                                   k: int = 5,
                                   space_name: Optional[str] = None,
                                   filter_uids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        在图谱中执行语义相似性搜索 (委托给 SemanticMap)。
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
                                relationship_type: Optional[str] = None,
                                direction: str = "successors",
                                space_name: Optional[str] = None) -> List[MemoryUnit]:
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
                neighbor_ids.intersection_update(space_uids)
            else:
                logging.warning(f"内存空间 '{space_name}' 未找到，无法按空间过滤邻居。")
                return []

        # 获取 MemoryUnit 对象
        results: List[MemoryUnit] = []
        for nid in neighbor_ids:
            unit = self.get_unit(nid)  # 使用智能获取方法
            if unit:
                results.append(unit)
        return results

    def traverse_implicit_nodes(self,
                                uid: str,
                                k: int = 5,
                                space_name: Optional[str] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        基于语义相似性查找与给定节点隐式相关的节点。
        """
        start_unit = self.get_unit(uid)
        if not start_unit or start_unit.embedding is None:
            logging.warning(f"节点 '{uid}' 不存在或没有向量，无法进行隐式遍历。")
            return []

        similar_units_with_scores = self.semantic_map.search_similarity_by_embedding(
            start_unit.embedding,
            k=k + 1,  # 获取稍多一些，以防 uid 是最相似的
            space_name=space_name
        )
        
        results: List[Tuple[MemoryUnit, float]] = []
        for unit, score in similar_units_with_scores:
            if unit.uid != uid:  # 排除起始节点本身
                results.append((unit, score))
            if len(results) >= k:  # 如果已达到k个结果
                break
        return results

    # MemorySpace 相关 (通过 SemanticMap 操作)
    def create_memory_space_in_map(self, space_name: str) -> MemorySpace:
        """在底层的 SemanticMap 中创建或获取一个内存空间。"""
        return self.semantic_map.create_memory_space(space_name)

    def add_unit_to_space_in_map(self, uid: str, space_name: str):
        """将一个内存单元添加到 SemanticMap 中的指定内存空间。"""
        self.semantic_map.add_unit_to_space(uid, space_name)

    def display_graph_summary(self):
        """打印图谱的摘要信息。"""
        num_map_units = len(self.semantic_map.memory_units)
        num_map_indexed = self.semantic_map.faiss_index.ntotal if self.semantic_map.faiss_index else 0
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
            f"  - 内存单元总数: {num_map_units}\n"
            f"  - 已索引向量数: {num_map_indexed}\n"
            f"  - 内存空间数: {num_map_spaces} ({list(self.semantic_map.memory_spaces.keys())})\n"
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

    # 保持原有的持久化方法
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
            "_relationships_last_accessed": self._relationships_last_accessed
        }
        
        with open(os.path.join(directory_path, "management_state.pkl"), "wb") as f:
            pickle.dump(management_state, f)

        logging.info(f"SemanticGraph 已保存到目录: '{directory_path}'")

    def export_to_neo4j(
        self, 
        uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password",
        database: str = "neo4j"
        ) -> bool:
        """将SemanticGraph中的节点和关系导出到Neo4j数据库"""
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

    @classmethod
    def load_graph(cls, directory_path: str,
                   image_embedding_model_name: Optional[str] = None,
                   text_embedding_model_name: Optional[str] = None) -> 'SemanticGraph':
        """从指定目录加载 SemanticGraph 的状态。"""
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
                    instance.nx_graph = nx.DiGraph()
        elif os.path.exists(nx_graph_file_pkl):
             with open(nx_graph_file_pkl, "rb") as f:
                instance.nx_graph = pickle.load(f)
             logging.info(f"NetworkX 图已从 pickle 文件 '{nx_graph_file_pkl}' 加载。")
        else:
            logging.warning(f"NetworkX 图文件在 '{directory_path}' 中未找到。图将为空。")
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
                instance._modified_relationships = set(state.get("_modified_relationships", []))
                instance._deleted_relationships = set(state.get("_deleted_relationships", []))
                instance._relationship_cache = state.get("_relationship_cache", {})
                instance._relationships_access_counts = state.get("_relationships_access_counts", {})
                instance._relationships_last_accessed = state.get("_relationships_last_accessed", {})
                
                logging.info("内存管理状态已加载")
            except Exception as e:
                logging.warning(f"加载内存管理状态失败: {e}")

        logging.info(f"SemanticGraph 已从目录 '{directory_path}' 加载。")
        return instance