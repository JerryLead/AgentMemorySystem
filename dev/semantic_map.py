from datetime import datetime
import os
import pickle
import logging
from typing import Dict, Any, Optional, List, Set, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image

from .milvus_operator import MilvusOperator
from .memory_unit import MemoryUnit
from .memory_space import MemorySpace
from .llm_cache import LLMCacheAdvisor
from .llm_cache import CacheAnalysisData

class SemanticMap:
    """
    语义地图 (SemanticMap) 负责存储内存单元及其向量嵌入，并支持基于相似度的搜索。
    它还管理内存空间，允许在特定上下文中进行操作。
    类似于一个向量数据库。
    """
    DEFAULT_TEXT_EMBEDDING_KEY = "text_content" # MemoryUnit.raw_data 中用于文本嵌入的默认键
    DEFAULT_IMAGE_EMBEDDING_KEY = "image_path" # MemoryUnit.raw_data 中用于图像嵌入的默认键

    def __init__(self,
                 image_embedding_model_name: str = "clip-ViT-B-32",
                 text_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
                 embedding_dim: int = 512,
                 faiss_index_type: str = "IDMap,Flat"): # 使用IDMap来支持按ID删除
        """
        初始化语义地图。
        参数:
            image_embedding_model_name (str): 用于图像嵌入的 SentenceTransformer 模型名称。
            text_embedding_model_name (str): 用于文本嵌入的 SentenceTransformer 模型名称。
            embedding_dim (int): 嵌入向量的维度。
            faiss_index_type (str): FAISS 索引工厂字符串 (例如 "IDMap,Flat", "IDMap,HNSW32,Flat")。
                                    "IDMap" 前缀对于按ID删除和高效空间过滤很重要。
        """
        self.embedding_dim: int = embedding_dim
        self.faiss_index_type: str = faiss_index_type
        
        self.memory_units: Dict[str, MemoryUnit] = {} # 存储 MemoryUnit 对象，键为 uid
        self.memory_spaces: Dict[str, MemorySpace] = {} # 存储 MemorySpace 对象，键为 space_name

        self.faiss_index: Optional[faiss.Index] = None # FAISS 索引
        
        # FAISS ID 到 MemoryUnit ID 的映射 (以及反向映射)
        # FAISS 内部使用从0开始的连续整数ID。我们需要将这些ID映射回我们的MemoryUnit ID。
        # 当使用 IndexIDMap 时，我们传递给 add_with_ids 的 ID 可以是我们自己的整数ID。
        # 我们将使用一个内部计数器或 MemoryUnit ID 的哈希（如果需要稳定且非连续的ID）
        # 为简单起见，我们将维护一个从 MemoryUnit.id 到一个内部 faiss_id (int64) 的映射。
        self._uid_to_internal_faiss_id: Dict[str, int] = {}
        self._internal_faiss_id_counter: int = 0 # 用于生成唯一的内部 FAISS ID

        # 添加以下跟踪变量
        self._modified_units = set()  # 被修改的单元ID集合
        self._deleted_units = set()   # 已删除但尚未同步到磁盘的单元ID
        self._last_sync_time = None   # 上次与外部存储同步的时间
        self._external_storage:MilvusOperator = None # 指向外部存储的连接（Milvus）
        self._max_memory_units = 10000 # 内存中最大单元数，超过则触发换页
        self._access_counts = {}      # 记录每个单元的访问次数，用于LRU策略

        try:
            self.text_model = SentenceTransformer(text_embedding_model_name)
            self.image_model = SentenceTransformer(image_embedding_model_name)
            # 验证嵌入维度是否与模型输出匹配 (可选，但推荐)
            # test_text_emb = self.text_model.encode("test")
            # if test_text_emb.shape[0] != embedding_dim:
            #     logging.warning(f"文本模型输出维度 {test_text_emb.shape[0]} 与指定的 embedding_dim {embedding_dim} 不符。")
        except Exception as e:
            logging.error(f"初始化 SentenceTransformer 模型失败: {e}")
            raise
            
        self._init_faiss_index()
        logging.info(f"SemanticMap 已初始化。嵌入维度: {self.embedding_dim}, FAISS索引类型: '{self.faiss_index_type}'。")

    def _init_faiss_index(self):
        """初始化或重新初始化 FAISS 索引。"""
        try:
            # IndexIDMap2 允许我们使用自己的64位整数ID
            # 我们将使用一个内部计数器生成的ID，并维护 uid -> internal_faiss_id 的映射
            base_index = faiss.index_factory(self.embedding_dim, self.faiss_index_type.replace("IDMap,","").replace("IDMap2,","")) # 例如 "Flat" or "HNSW32,Flat"
            if "IDMap" not in self.faiss_index_type and "IDMap2" not in self.faiss_index_type:
                 logging.warning(f"FAISS索引类型 '{self.faiss_index_type}' 不包含 'IDMap' 或 'IDMap2'。建议使用它们以支持按ID删除和高效过滤。将尝试包装基础索引。")
                 # 如果用户没有指定IDMap，我们尝试添加它
                 if hasattr(faiss, 'IndexIDMap2'):
                     self.faiss_index = faiss.IndexIDMap2(base_index)
                 else:
                     self.faiss_index = faiss.IndexIDMap(base_index) # 后备
            else: # 用户已指定IDMap或IDMap2
                 self.faiss_index = faiss.index_factory(self.embedding_dim, self.faiss_index_type)

            if self.faiss_index_type.startswith("IDMap,") and not hasattr(faiss, 'IndexIDMap'): # 检查兼容性
                logging.warning("请求了 IndexIDMap 但可能不可用，尝试 IndexIDMap2。")
                # 进一步的兼容性逻辑可能需要

            logging.info(f"FAISS 索引 '{self.faiss_index_type}' 已初始化。总向量数: {self.faiss_index.ntotal if self.faiss_index else 0}")
        except Exception as e:
            logging.error(f"初始化 FAISS 索引 '{self.faiss_index_type}' 失败: {e}")
            self.faiss_index = None # 确保索引在失败时为None
            raise

    ### 嵌入生成函数

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """为给定文本生成嵌入向量。"""
        if not text or not isinstance(text, str):
            logging.warning("尝试为无效文本生成嵌入。")
            return None
        try:
            emb = self.text_model.encode(text)
            return emb.astype(np.float32)
        except Exception as e:
            logging.error(f"生成文本嵌入失败: '{text[:50]}...' - {e}")
            return None

    def _get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """为给定图像路径生成嵌入向量。"""
        if not image_path or not isinstance(image_path, str):
            logging.warning("尝试为无效图像路径生成嵌入。")
            return None
        if not os.path.isfile(image_path):
            logging.error(f"图像文件未找到: {image_path}")
            return None
        try:
            img = Image.open(image_path)
            emb = self.image_model.encode(img)
            return emb.astype(np.float32)
        except Exception as e:
            logging.error(f"生成图像嵌入失败: '{image_path}' - {e}")
            return None

    def _generate_embedding_for_unit(self, unit: MemoryUnit,
                                     explicit_content: Optional[Any] = None,
                                     content_type: Optional[str] = None) -> Optional[np.ndarray]:
        """
        为内存单元生成嵌入向量。
        参数:
            unit (MemoryUnit): 要生成嵌入的内存单元。
            explicit_content (Optional[Any]): 用于嵌入的显式内容 (文本字符串或图像路径)。
            content_type (Optional[str]): "text" 或 "image_path"，指示 explicit_content 的类型。
        返回:
            np.ndarray 或 None: 生成的嵌入向量，如果失败则为 None。
        """
        embedding = None
        if explicit_content and content_type:
            if content_type == "text":
                embedding = self._get_text_embedding(str(explicit_content))
            elif content_type == "image_path":
                embedding = self._get_image_embedding(str(explicit_content))
            else:
                logging.warning(f"不支持的内容类型 '{content_type}' 用于为单元 '{unit.uid}' 生成嵌入。")
        else: # 尝试从 unit.raw_data 推断
            image_path = unit.raw_data.get(self.DEFAULT_IMAGE_EMBEDDING_KEY)
            text_content = unit.raw_data.get(self.DEFAULT_TEXT_EMBEDDING_KEY)

            if image_path and isinstance(image_path, str):
                embedding = self._get_image_embedding(image_path)
                if embedding is not None:
                    logging.debug(f"已为单元 '{unit.uid}' 从其值中的图像路径 '{image_path}' 生成嵌入。")
            
            if embedding is None and text_content and isinstance(text_content, str): # 如果图像嵌入失败或未提供图像，则尝试文本
                embedding = self._get_text_embedding(text_content)
                if embedding is not None:
                    logging.debug(f"已为单元 '{unit.uid}' 从其值中的文本内容生成嵌入。")
            
            if embedding is None:
                logging.warning(f"无法从单元 '{unit.uid}' 的值中找到合适的内容来生成嵌入。请检查键 '{self.DEFAULT_IMAGE_EMBEDDING_KEY}' 或 '{self.DEFAULT_TEXT_EMBEDDING_KEY}'。")
        
        if embedding is not None and embedding.shape[0] != self.embedding_dim:
            logging.error(f"为单元 '{unit.uid}' 生成的嵌入维度 ({embedding.shape[0]}) 与期望维度 ({self.embedding_dim}) 不符。")
            return None
        return embedding
    
    ### 数据库操作函数

    def connect_external_storage(self, 
                            storage_type: str = "milvus",
                            host: str = "localhost", 
                            port: str = "19530",
                            user: str = "", 
                            password: str = "",
                            collection_name: str = "hippo_memory_units"):
        """
        连接到外部存储系统
        参数:
            storage_type: 存储类型，当前支持 "milvus"
            其他参数: 特定存储系统的连接参数
        """
        if storage_type.lower() == "milvus":
            try:
                self._external_storage = MilvusOperator(
                    host=host, 
                    port=port,
                    user=user, 
                    password=password,
                    collection_name=collection_name,
                    embedding_dim=self.embedding_dim
                )
                if self._external_storage.is_connected:
                    # 确保集合已创建
                    self._external_storage.create_collection()
                    logging.info(f"已连接到外部存储: Milvus ({host}:{port})")
                    return True
                else:
                    self._external_storage = None
                    logging.error("连接到Milvus失败")
                    return False
            except Exception as e:
                logging.error(f"初始化外部存储连接失败: {e}")
                self._external_storage = None
                return False
        else:
            logging.error(f"不支持的存储类型: {storage_type}")
            return False

    def _record_unit_access(self, uid: str):
        """记录单元被访问，更新访问计数"""
        if uid in self.memory_units:
            self._access_counts[uid] = self._access_counts.get(uid, 0) + 1

    def _sync_unit_to_external(self, uid: str) -> bool:
        """将单元同步到外部存储(Milvus)"""
        if not self._external_storage:
            return False
            
        unit = self.memory_units.get(uid)
        if not unit:
            logging.warning(f"无法同步不存在的单元 '{uid}'")
            return False
            
        try:
            # 找出单元所属的所有空间
            space_names = []
            for space_name, space in self.memory_spaces.items():
                if uid in space.get_memory_uids():
                    space_names.append(space_name)
            
            # 使用MilvusOperator添加/更新单元
            success = self._external_storage.add_unit(unit, space_names)
            if success:
                logging.debug(f"单元 '{uid}' 已同步到外部存储")
                return True
            else:
                logging.error(f"同步单元 '{uid}' 到外部存储失败")
                return False
        except Exception as e:
            logging.error(f"同步单元 '{uid}' 到外部存储时出错: {e}")
            return False

    def sync_to_external(self, force_full_sync: bool = False):
        """
        将修改过的单元同步到外部存储
        参数:
            force_full_sync: 如果为True，则同步所有单元，而不仅是修改过的单元
        返回:
            (success_count, fail_count): 成功和失败的单元数量
        """
        if not self._external_storage:
            logging.error("未连接到外部存储，无法同步")
            return 0, 0
            
        success_count = 0
        fail_count = 0
        
        # 处理修改过的单元
        units_to_sync = list(self.memory_units.keys()) if force_full_sync else list(self._modified_units)
        for uid in units_to_sync:
            if uid in self.memory_units:  # 确保单元仍然存在
                if self._sync_unit_to_external(uid):
                    success_count += 1
                    if uid in self._modified_units:
                        self._modified_units.remove(uid)
                else:
                    fail_count += 1
        
        # 处理删除的单元
        deleted_uids = list(self._deleted_units)
        for uid in deleted_uids:
            try:
                if self._external_storage.delete_unit(uid):
                    self._deleted_units.remove(uid)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logging.error(f"从外部存储删除单元 '{uid}' 失败: {e}")
                fail_count += 1
        
        self._last_sync_time = datetime.now()
        logging.info(f"与外部存储同步完成。成功: {success_count}, 失败: {fail_count}")
        return success_count, fail_count

    def load_from_external(self, 
                 filter_space: Optional[str] = None,
                 limit: int = 1000,
                 replace_existing: bool = False):
        """
        从外部存储加载单元到内存
        """
        if not self._external_storage:
            logging.error("未连接到外部存储，无法加载")
            return 0
        
        try:
            # 根据空间过滤加载单元
            if filter_space:
                units = self._external_storage.get_units_by_space(filter_space)[:limit]
            else:
                # 为了避免一次性加载太多，我们需要实现分页加载
                # 这里简化处理，获取有限数量的单元
                logging.warning("从外部存储加载所有单元可能消耗大量内存，建议指定空间过滤")
                # 由于MilvusOperator没有get_all_unit_ids方法，我们跳过这个实现
                return 0
            
            load_count = 0
            # 将加载的单元添加到内存中
            for unit in units:
                if unit.uid not in self.memory_units or replace_existing:
                    # 检查内存限制
                    if len(self.memory_units) >= self._max_memory_units:
                        self.swap_out(count=1)
                    
                    self.memory_units[unit.uid] = unit
                    self._access_counts[unit.uid] = 0  # 初始化访问计数
                    load_count += 1
                    
                    # 处理单元的空间归属
                    if hasattr(unit, 'metadata') and isinstance(unit.metadata, dict):
                        spaces = unit.metadata.get('spaces', [])
                        for space_name in spaces:
                            self.add_unit_to_space(unit.uid, space_name)
            
            # 如果加载了单元，重建索引
            if load_count > 0:
                logging.info(f"从外部存储加载了 {load_count} 个单元")
                self.build_index()
                
            return load_count
            
        except Exception as e:
            logging.error(f"从外部存储加载单元失败: {e}")
            return 0
        
    ### 内存上下文管理函数,新增对于LLM strategy的支持

    def set_llm_cache_advisor(self, llm_client, model_name: str = "gpt-4"):
        """设置LLM缓存顾问"""
        self._llm_cache_advisor = LLMCacheAdvisor(llm_client, model_name)
        self._recent_queries = []
        self._last_accessed = {}
        logging.info("LLM缓存顾问已启用")

    def swap_out(self, count: int = 100, strategy: str = "LRU", query_context: Optional[str] = None):
        """
        将最少使用的单元从内存移出到外部存储
        参数:
            count: 要移出的单元数量
            strategy: 换出策略，支持 "LLM"、"LRU"、"LFU"
            query_context: 当前查询上下文（用于LLM策略）
        """
        if not self._external_storage:
            logging.warning("没有配置外部存储，无法执行换页操作")
            return
        
        if len(self.memory_units) == 0:
            logging.debug("内存中没有单元可以换出")
            return
        
        # 根据策略确定要换出的单元
        units_to_page_out = []
        
        if strategy == "LLM" and hasattr(self, '_llm_cache_advisor') and self._llm_cache_advisor:
            try:
                # 使用LLM进行智能决策
                analysis_data = self._llm_cache_advisor.analyze_cache_context(
                    memory_units=self.memory_units,
                    access_counts=self._access_counts,
                    last_accessed=getattr(self, '_last_accessed', {}),
                    current_query_context=query_context
                )
                
                recommended_uids = self._llm_cache_advisor.recommend_eviction(
                    analysis_data=analysis_data,
                    eviction_count=count,
                    current_query_context=query_context,
                    recent_queries=getattr(self, '_recent_queries', [])
                )
                
                units_to_page_out = [(uid, 0) for uid in recommended_uids]
                logging.info(f"LLM推荐换出单元: {recommended_uids}")
                
            except Exception as e:
                logging.error(f"LLM缓存决策失败: {e}，降级使用LRU算法")
                strategy = "LRU"  # 降级到LRU
        
        # 如果是LRU、LFU策略，或者LLM策略失败后的降级
        if strategy in ["LRU", "LFU"] or not units_to_page_out:
            if strategy == "LRU":
                # 按最后访问时间排序，最少最近使用的优先换出
                sorted_units = sorted(
                    [(uid, getattr(self, '_last_accessed', {}).get(uid, 0)) for uid in self.memory_units.keys()],
                    key=lambda x: x[1]  # 按最后访问时间排序，时间越早越优先换出
                )
            elif strategy == "LFU":
                # 按访问频率排序，访问次数最少的优先换出
                sorted_units = sorted(
                    [(uid, self._access_counts.get(uid, 0)) for uid in self.memory_units.keys()],
                    key=lambda x: x[1]  # 按访问次数排序，次数越少越优先换出
                )
            else:  # 默认使用LRU
                logging.warning(f"不支持的换出策略: {strategy}，使用LRU")
                sorted_units = sorted(
                    [(uid, getattr(self, '_last_accessed', {}).get(uid, 0)) for uid in self.memory_units.keys()],
                    key=lambda x: x[1]
                )
            
            # 限制换出数量
            actual_count = min(count, len(sorted_units))
            units_to_page_out = sorted_units[:actual_count]
        
        if not units_to_page_out:
            logging.debug("没有找到可换出的单元")
            return
        
        # 执行换出操作
        synced_count = 0
        removed_count = 0
        
        for uid, _ in units_to_page_out:
            unit = self.memory_units.get(uid)
            if not unit:
                continue
                
            # 获取单元所属空间
            space_names = []
            for space_name, space in self.memory_spaces.items():
                if uid in space.get_memory_uids():
                    space_names.append(space_name)
            
            # 同步到外部存储
            if self._external_storage.add_unit(unit, space_names):
                synced_count += 1
                self._modified_units.discard(uid)
            
            # 从FAISS索引中移除
            if uid in self._uid_to_internal_faiss_id and self.faiss_index:
                try:
                    internal_id = self._uid_to_internal_faiss_id[uid]
                    if hasattr(self.faiss_index, 'remove_ids'):
                        self.faiss_index.remove_ids(np.array([internal_id], dtype=np.int64))
                    del self._uid_to_internal_faiss_id[uid]
                except Exception as e:
                    logging.error(f"从FAISS索引移除单元 '{uid}' 失败: {e}")
            
            # 从内存中移除
            if uid in self.memory_units:
                del self.memory_units[uid]
                removed_count += 1
                
            # 清理访问计数和时间记录
            self._access_counts.pop(uid, None)
            if hasattr(self, '_last_accessed'):
                self._last_accessed.pop(uid, None)
                
            logging.debug(f"单元 '{uid}' 已从内存换出（策略: {strategy}）")

        logging.info(f"使用{strategy}策略已将 {removed_count} 个单元从内存换出，其中 {synced_count} 个已同步到外部存储")

    def _load_unit_from_external(self, uid: str) -> Optional[MemoryUnit]:
        """从外部存储(Milvus)加载单元到内存"""
        if not self._external_storage:
            return None
            
        try:
            # 使用MilvusOperator加载单元
            unit = self._external_storage.get_unit(uid)
            if unit:
                logging.debug(f"单元 '{uid}' 已从外部存储加载到内存")
                return unit
        except Exception as e:
            logging.error(f"从外部存储加载单元 '{uid}' 失败: {e}")
        
        return None
    
    # def _unpersist_least_used_units(self, 
    #                                 count: int = 100, 
    #                                 query_context: Optional[str] = None):
    #     """
    #     使用LLM智能决策要换出的单元
    #     参数:
    #         count: 要移出的单元数量  
    #         query_context: 当前查询上下文
    #     """
    #     if not self._external_storage:
    #         logging.warning("没有配置外部存储，无法执行换页操作")
    #         return
        
    #     if len(self.memory_units) == 0:
    #         logging.debug("内存中没有单元可以换出")
    #         return
            
    #     # 检查是否配置了LLM顾问
    #     if hasattr(self, '_llm_cache_advisor') and self._llm_cache_advisor:
    #         try:
    #             # 使用LLM进行智能决策
    #             analysis_data = self._llm_cache_advisor.analyze_cache_context(
    #                 memory_units=self.memory_units,
    #                 access_counts=self._access_counts,
    #                 last_accessed=getattr(self, '_last_accessed', {}),
    #                 current_query_context=query_context
    #             )
                
    #             recommended_uids = self._llm_cache_advisor.recommend_eviction(
    #                 analysis_data=analysis_data,
    #                 eviction_count=count,
    #                 current_query_context=query_context,
    #                 recent_queries=getattr(self, '_recent_queries', [])
    #             )
                
    #             units_to_page_out = [(uid, 0) for uid in recommended_uids]
    #             logging.info(f"LLM推荐换出单元: {recommended_uids}")
                
    #         except Exception as e:
    #             logging.error(f"LLM缓存决策失败: {e}，使用传统算法")
    #             # 降级到传统算法
    #             sorted_units = sorted(
    #                 [(uid, self._access_counts.get(uid, 0)) for uid in self.memory_units.keys()],
    #                 key=lambda x: x[1]
    #             )
    #             actual_count = min(count, len(sorted_units))
    #             units_to_page_out = sorted_units[:actual_count]
    #     else:
    #         # 使用传统LRU算法
    #         sorted_units = sorted(
    #             [(uid, self._access_counts.get(uid, 0)) for uid in self.memory_units.keys()],
    #             key=lambda x: x[1]
    #         )
    #         actual_count = min(count, len(sorted_units))
    #         units_to_page_out = sorted_units[:actual_count]
        
    #     # 执行换出操作
    #     synced_count = 0
    #     removed_count = 0
        
    #     for uid, _ in units_to_page_out:
    #         unit = self.memory_units.get(uid)
    #         if not unit:
    #             continue
                
    #         # 获取单元所属空间
    #         space_names = []
    #         for space_name, space in self.memory_spaces.items():
    #             if uid in space.get_memory_uids():
    #                 space_names.append(space_name)
            
    #         # 同步到外部存储
    #         if self._external_storage.add_unit(unit, space_names):
    #             synced_count += 1
    #             self._modified_units.discard(uid)
            
    #         # 从FAISS索引中移除
    #         if uid in self._uid_to_internal_faiss_id and self.faiss_index:
    #             try:
    #                 internal_id = self._uid_to_internal_faiss_id[uid]
    #                 if hasattr(self.faiss_index, 'remove_ids'):
    #                     self.faiss_index.remove_ids(np.array([internal_id], dtype=np.int64))
    #                 del self._uid_to_internal_faiss_id[uid]
    #             except Exception as e:
    #                 logging.error(f"从FAISS索引移除单元 '{uid}' 失败: {e}")
            
    #         # 从内存中移除
    #         if uid in self.memory_units:
    #             del self.memory_units[uid]
    #             removed_count += 1
                
    #         # 清理访问计数
    #         self._access_counts.pop(uid, None)
                
    #         logging.debug(f"单元 '{uid}' 已从内存页出")

    #     logging.info(f"已将 {removed_count} 个单元从内存页出，其中 {synced_count} 个已同步到外部存储")

    def record_query(self, query: str, accessed_uids: List[str]):
        """记录查询和访问的单元"""
        self._recent_queries.append(query)
        if len(self._recent_queries) > 10:  # 只保留最近10个查询
            self._recent_queries = self._recent_queries[-10:]
        
        # 更新访问时间
        current_time = datetime.now().timestamp()
        for uid in accessed_uids:
            self._last_accessed[uid] = current_time
            if hasattr(self, '_llm_cache_advisor') and self._llm_cache_advisor:
                self._llm_cache_advisor.record_access(uid, query)
    
    ### 内存单元操作函数
    
    def add_unit(self,
            unit: MemoryUnit,
            explicit_content_for_embedding: Optional[Any] = None,
            content_type_for_embedding: Optional[str] = None,
            space_names: Optional[List[str]] = None,
            rebuild_index_immediately: bool = False
        ):
        """
        向语义地图添加一个新的内存单元。
        """
        if not isinstance(unit, MemoryUnit):
            logging.error("尝试添加的不是 MemoryUnit 对象。")
            return
        
        # 检查内存限制，必要时触发换页
        if len(self.memory_units) >= self._max_memory_units:
            page_out_count = max(1, int(self._max_memory_units * 0.1))  # 至少换出1个
            self.swap_out(count=page_out_count)
        
        if explicit_content_for_embedding is None:
            logging.debug(f"未提供显式内容用于嵌入，尝试从单元 '{unit.uid}' 的值中推断")
            # 尝试从raw_data中提取文本内容
            if self.DEFAULT_TEXT_EMBEDDING_KEY in unit.raw_data:
                explicit_content_for_embedding = unit.raw_data[self.DEFAULT_TEXT_EMBEDDING_KEY]
                content_type_for_embedding = "text"
            elif self.DEFAULT_IMAGE_EMBEDDING_KEY in unit.raw_data:
                explicit_content_for_embedding = unit.raw_data[self.DEFAULT_IMAGE_EMBEDDING_KEY]
                content_type_for_embedding = "image_path"
            else:
                # 如果没有标准键，使用整个raw_data的字符串表示
                explicit_content_for_embedding = str(unit.raw_data)
                content_type_for_embedding = "text"

        # 生成或更新嵌入
        new_embedding = self._generate_embedding_for_unit(unit, explicit_content_for_embedding, content_type_for_embedding)
        
        if new_embedding is None:
            logging.warning(f"未能为内存单元 '{unit.uid}' 生成嵌入。该单元将被添加，但无法用于相似性搜索。")
        unit.embedding = new_embedding

        # 添加到内存
        self.memory_units[unit.uid] = unit
        logging.info(f"内存单元 '{unit.uid}' 已添加/更新到 SemanticMap。")

        # 添加到指定空间
        if space_names:
            for space_name in space_names:
                self.add_unit_to_space(unit.uid, space_name)
        
        # 重建索引(如果需要)
        if rebuild_index_immediately:
            self.build_index()

        # 添加到修改跟踪
        self._modified_units.add(unit.uid)
        self._access_counts[unit.uid] = self._access_counts.get(unit.uid, 0) + 1

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        """增强get_unit方法，记录访问并支持从外部存储加载"""
        unit = self.memory_units.get(uid)
        if unit:
            # 如果内存中有，记录访问并返回
            self._record_unit_access(uid)
            return unit
        elif self._external_storage:
            # 如果内存中没有但有外部存储，尝试加载
            unit = self._load_unit_from_external(uid)
            if unit:
                # 检查内存是否已满，必要时换页
                if len(self.memory_units) >= self._max_memory_units:
                    self.swap_out(count=1)
                
                self.memory_units[uid] = unit
                self._record_unit_access(uid)
                
                # 重新添加到FAISS索引
                if unit.embedding is not None and self.faiss_index:
                    try:
                        internal_id = self._internal_faiss_id_counter
                        self._internal_faiss_id_counter += 1
                        self._uid_to_internal_faiss_id[uid] = internal_id
                        
                        vector = unit.embedding.reshape(1, -1).astype(np.float32)
                        ids = np.array([internal_id], dtype=np.int64)
                        self.faiss_index.add_with_ids(vector, ids)
                        logging.debug(f"单元 '{uid}' 的向量已添加到FAISS索引")
                    except Exception as e:
                        logging.error(f"向FAISS索引添加向量失败: {e}")
                
                return unit
        return None

    def delete_unit(self, uid: str, rebuild_index_immediately: bool = False):
        """从语义地图中删除一个内存单元。"""
        if uid in self.memory_units:
            del self.memory_units[uid]
            
            # 从所有内存空间中移除
            for space_name, space_obj in self.memory_spaces.items():
                space_obj.remove_unit(uid)
            
            # 从FAISS索引中移除
            if self.faiss_index and uid in self._uid_to_internal_faiss_id:
                internal_id_to_remove = self._uid_to_internal_faiss_id[uid]
                try:
                    if hasattr(self.faiss_index, 'remove_ids'):
                        self.faiss_index.remove_ids(np.array([internal_id_to_remove], dtype=np.int64))
                        logging.debug(f"内存单元 '{uid}' (内部FAISS ID: {internal_id_to_remove}) 已从FAISS索引中移除。")
                    del self._uid_to_internal_faiss_id[uid]
                except Exception as e:
                    logging.error(f"从FAISS索引中移除单元 '{uid}' 失败: {e}。建议重建索引。")
            else:
                logging.warning(f"单元 '{uid}' 不在FAISS ID映射中，可能未被索引或已被移除。")

            logging.info(f"内存单元 '{uid}' 已从 SemanticMap 删除。")
            if rebuild_index_immediately:
                self.build_index()
        else:
            logging.warning(f"尝试删除不存在的内存单元ID '{uid}'。")

        # 添加到删除跟踪
        self._deleted_units.add(uid)
        if uid in self._modified_units:
            self._modified_units.remove(uid)
        if uid in self._access_counts:
            del self._access_counts[uid]

    def build_index(self):
        """
        根据当前所有具有有效嵌入的内存单元构建（或重建）FAISS索引。
        """
        if not self.memory_units:
            logging.info("没有内存单元可用于构建索引。")
            if self.faiss_index: self.faiss_index.reset() # 清空现有索引
            self._uid_to_internal_faiss_id.clear()
            self._internal_faiss_id_counter = 0
            return

        valid_embeddings: List[np.ndarray] = []
        internal_faiss_ids_for_index: List[int] = []
        
        # 重置映射和计数器，因为我们要重建
        self._uid_to_internal_faiss_id.clear()
        # self._internal_faiss_id_counter = 0 # 如果希望ID在多次重建中保持某种程度的稳定性，则不要重置计数器，除非ID用完。
                                          # 但对于 IndexIDMap，每次重建时使用新的连续ID可能更简单。

        current_internal_id = 0
        for uid, unit in self.memory_units.items():
            if unit.embedding is not None and unit.embedding.shape[0] == self.embedding_dim:
                valid_embeddings.append(unit.embedding)
                # 分配一个新的内部FAISS ID
                self._uid_to_internal_faiss_id[uid] = current_internal_id
                internal_faiss_ids_for_index.append(current_internal_id)
                current_internal_id += 1
            else:
                logging.debug(f"内存单元 '{uid}' 没有有效向量，将不被包含在FAISS索引中。")

        if not valid_embeddings:
            logging.info("没有有效的嵌入可用于构建FAISS索引。")
            if self.faiss_index: self.faiss_index.reset()
            return

        embeddings_np = np.array(valid_embeddings).astype(np.float32)
        ids_np = np.array(internal_faiss_ids_for_index, dtype=np.int64)

        # 重新初始化FAISS索引以确保它是干净的
        self._init_faiss_index()
        if not self.faiss_index: # 如果初始化失败
            logging.error("FAISS索引未初始化，无法构建。")
            return

        # 如果索引类型需要训练 (例如 IVF 系列)
        if "IVF" in self.faiss_index_type and not self.faiss_index.is_trained:
            logging.info(f"正在训练FAISS索引 ('{self.faiss_index_type}')...")
            # 确保训练数据至少有一定数量的向量，具体取决于 nlist
            if embeddings_np.shape[0] < getattr(self.faiss_index, 'nlist', 1): # 简单检查
                 logging.warning(f"训练数据太少 ({embeddings_np.shape[0]} 个向量) 对于 IVF 索引。可能导致错误或性能不佳。")
            if embeddings_np.shape[0] > 0 :
                self.faiss_index.train(embeddings_np)
                logging.info("FAISS索引训练完成。")
            else:
                logging.error("没有数据用于训练FAISS索引。")
                return


        self.faiss_index.add_with_ids(embeddings_np, ids_np)
        logging.info(f"FAISS索引已成功构建/重建。包含 {self.faiss_index.ntotal} 个向量。")
        # 更新内部ID计数器，以防将来增量添加（尽管当前build_index是完全重建）
        self._internal_faiss_id_counter = current_internal_id


    def search_similarity_by_embedding(self,
                                query_embedding: np.ndarray,
                                k: int = 5,
                                space_name: Optional[str] = None,
                                filter_uids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        通过查询向量在语义地图中搜索相似的内存单元。
        参数:
            query_embedding (np.ndarray): 用于查询的嵌入向量。
            k (int): 要返回的最相似单元的数量。
            space_name (Optional[str]): 如果提供，则仅在指定的内存空间内搜索。
            filter_uids (Optional[Set[str]]): 一个可选的单元ID集合，用于进一步限制搜索范围（仅搜索这些ID对应的单元）。
                                                如果同时提供了 space_name 和 filter_uids，则取它们的交集。
        返回:
            List[Tuple[MemoryUnit, float]]: 一个元组列表，每个元组包含 (相似的MemoryUnit, 相似度得分/距离)。
                                            列表按相似度降序排列 (距离越小越相似)。
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logging.warning("FAISS索引未构建或为空。无法执行搜索。")
            return []
        if query_embedding is None or query_embedding.shape[0] != self.embedding_dim:
            logging.error(f"查询向量无效或维度不匹配 (期望 {self.embedding_dim})。")
            return []

        query_embedding_np = query_embedding.reshape(1, -1).astype(np.float32)
        
        search_params = None
        final_target_internal_ids_np = None

        # 确定搜索范围 (所有单元，特定空间内的单元，或特定ID列表中的单元)
        candidate_uids: Optional[Set[str]] = None
        if space_name:
            space = self.get_memory_space(space_name)
            if space:
                candidate_uids = space.get_memory_uids().copy() # 获取副本
            else:
                logging.warning(f"内存空间 '{space_name}' 未找到。将执行全局搜索或基于 filter_uids 的搜索。")
                # 如果空间不存在，则不应返回任何结果，除非filter_uids也为空
                if not filter_uids: return []


        if filter_uids:
            if candidate_uids is not None:
                candidate_uids.intersection_update(filter_uids) # 取交集
            else:
                candidate_uids = filter_uids.copy()
        
        if candidate_uids is not None: # 如果有任何过滤条件
            if not candidate_uids: # 如果过滤后候选集为空
                logging.info("根据空间和/或ID过滤器，没有候选内存单元可供搜索。")
                return []
            
            # 将候选的 MemoryUnit ID 转换为内部 FAISS ID
            target_internal_faiss_ids = [self._uid_to_internal_faiss_id[uid] 
                                        for uid in candidate_uids if uid in self._uid_to_internal_faiss_id]
            if not target_internal_faiss_ids:
                logging.info("候选单元ID在FAISS索引中没有对应的内部ID。")
                return []

            final_target_internal_ids_np = np.array(target_internal_faiss_ids, dtype=np.int64)
            id_selector = faiss.IDSelectorArray(final_target_internal_ids_np)
            search_params = faiss.SearchParametersIVF() if "IVF" in self.faiss_index_type else faiss.SearchParameters()
            search_params.sel = id_selector
            # 调整k值，使其不超过候选集大小
            k = min(k, len(final_target_internal_ids_np))


        if k == 0: return [] # 如果有效k为0，则不搜索

        try:
            # 尝试使用搜索参数
            distances, internal_faiss_indices = self.faiss_index.search(query_embedding_np, k, params=search_params)
        except RuntimeError as e:
            if "search params not supported for this index" in str(e):
                # 索引不支持搜索参数，使用不带参数的搜索
                logging.warning(f"当前FAISS索引类型 '{self.faiss_index_type}' 不支持搜索参数。执行不带过滤器的搜索。")
                if space_name or filter_uids:
                    logging.warning("空间和ID过滤将在搜索后手动应用。")
                distances, internal_faiss_indices = self.faiss_index.search(query_embedding_np, k)
            else:
                # 如果是其他类型的错误，则重新抛出
                raise
        
        results: List[Tuple[MemoryUnit, float]] = []
        # 反向映射：从内部FAISS ID找到MemoryUnit ID
        # 创建一个从 internal_faiss_id 到 uid 的临时反向映射
        internal_id_to_uid_map = {v: k for k, v in self._uid_to_internal_faiss_id.items()}

        for i in range(internal_faiss_indices.shape[1]):
            internal_id = internal_faiss_indices[0, i]
            if internal_id == -1: # FAISS中表示没有更多结果
                continue
            
            uid = internal_id_to_uid_map.get(internal_id)
            if uid:
                unit = self.get_unit(uid)
                if unit:
                    results.append((unit, float(distances[0, i])))
                else: # 理论上不应发生，因为internal_id应该有效
                    logging.warning(f"在FAISS搜索结果中找到内部ID {internal_id}，但在内存单元字典中找不到对应的单元ID '{uid}'。")
            else:
                logging.warning(f"在FAISS搜索结果中找到无法映射回单元ID的内部ID {internal_id}。")
        
        return results

    def search_similarity_by_text(self, query_text: str, k: int = 5, space_name: Optional[str] = None, filter_uids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """通过查询文本进行相似性搜索。"""
        query_embedding = self._get_text_embedding(query_text)
        if query_embedding is None:
            return []
        return self.search_similarity_by_embedding(query_embedding, k, space_name, filter_uids)

    def search_similarity_by_image(self, image_path: str, k: int = 5, space_name: Optional[str] = None, filter_uids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """通过查询图像路径进行相似性搜索。"""
        query_embedding = self._get_image_embedding(image_path)
        if query_embedding is None:
            return []
        return self.search_similarity_by_embedding(query_embedding, k, space_name, filter_uids)

    # --- MemorySpace 管理方法 ---
    def create_memory_space(self, space_name: str) -> MemorySpace:
        """创建或获取一个内存空间。"""
        if space_name not in self.memory_spaces:
            self.memory_spaces[space_name] = MemorySpace(name=space_name)
            logging.info(f"内存空间 '{space_name}' 已创建。")
        return self.memory_spaces[space_name]

    def get_memory_space(self, space_name: str) -> Optional[MemorySpace]:
        """通过名称获取内存空间。"""
        return self.memory_spaces.get(space_name)

    def add_unit_to_space(self, uid: str, space_name: str):
        """将一个内存单元添加到一个内存空间。"""
        if uid not in self.memory_units:
            logging.warning(f"尝试将不存在的内存单元 '{uid}' 添加到空间 '{space_name}'。")
            return
        space = self.create_memory_space(space_name) # 如果空间不存在则创建
        space.add_unit(uid)

    def remove_unit_from_space(self, uid: str, space_name: str):
        """从一个内存空间移除一个内存单元。"""
        space = self.get_memory_space(space_name)
        if space:
            space.remove_unit(uid)
        else:
            logging.warning(f"尝试从不存在的内存空间 '{space_name}' 移除单元 '{uid}'。")

    # --- 持久化 ---
    def save_map(self, directory_path: str):
        """
        将 SemanticMap 的状态保存到指定目录。
        会保存 memory_units, memory_spaces, FAISS 索引, 以及 _uid_to_internal_faiss_id 映射。
        参数:
            directory_path (str): 用于保存文件的目录路径。如果不存在，将尝试创建。
        """
        os.makedirs(directory_path, exist_ok=True)
        
        # 1. 保存 MemoryUnit 和 MemorySpace 对象 (使用 pickle)
        with open(os.path.join(directory_path, "semantic_map_data.pkl"), "wb") as f:
            pickle.dump({
                "memory_units": self.memory_units,
                "memory_spaces": self.memory_spaces,
                "_uid_to_internal_faiss_id": self._uid_to_internal_faiss_id,
                "_internal_faiss_id_counter": self._internal_faiss_id_counter,
                "embedding_dim": self.embedding_dim,
                "faiss_index_type": self.faiss_index_type
            }, f)
        
        # 2. 保存 FAISS 索引
        if self.faiss_index:
            faiss.write_index(self.faiss_index, os.path.join(directory_path, "semantic_map.faissindex"))
        
        logging.info(f"SemanticMap 已保存到目录: '{directory_path}'")

    @classmethod
    def load_map(cls, directory_path: str,
                 image_embedding_model_name: Optional[str] = None, # 加载时可以覆盖模型名称
                 text_embedding_model_name: Optional[str] = None) -> 'SemanticMap':
        """
        从指定目录加载 SemanticMap 的状态。
        参数:
            directory_path (str): 从中加载文件的目录路径。
            image_embedding_model_name (Optional[str]): 可选，用于覆盖保存的图像模型名称。
            text_embedding_model_name (Optional[str]): 可选，用于覆盖保存的文本模型名称。
        返回:
            SemanticMap: 加载的 SemanticMap 实例。
        """
        data_file = os.path.join(directory_path, "semantic_map_data.pkl")
        index_file = os.path.join(directory_path, "semantic_map.faissindex")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"SemanticMap 数据文件未找到: {data_file}")

        with open(data_file, "rb") as f:
            saved_state = pickle.load(f)

        # 使用保存的参数或加载时提供的参数初始化实例
        img_model = image_embedding_model_name if image_embedding_model_name else "clip-ViT-B-32" # 默认值
        txt_model = text_embedding_model_name if text_embedding_model_name else "sentence-transformers/clip-ViT-B-32-multilingual-v1" # 默认值
        
        # 从保存的状态中获取参数，如果存在的话
        embedding_dim = saved_state.get("embedding_dim", 512)
        faiss_index_type = saved_state.get("faiss_index_type", "IDMap,Flat") # 确保有默认值

        # 如果加载时提供了模型名称，则使用它们
        # (注意：如果模型名称与保存时的不一致，嵌入维度也可能需要调整，这里简化处理)
        # 实际应用中，模型和维度应保持一致或有兼容性策略

        instance = cls(
            image_embedding_model_name=img_model, # 实际使用时应确保与保存时一致或兼容
            text_embedding_model_name=txt_model,  # 同上
            embedding_dim=embedding_dim,
            faiss_index_type=faiss_index_type
        )
        
        instance.memory_units = saved_state["memory_units"]
        instance.memory_spaces = saved_state["memory_spaces"]
        instance._uid_to_internal_faiss_id = saved_state.get("_uid_to_internal_faiss_id", {})
        instance._internal_faiss_id_counter = saved_state.get("_internal_faiss_id_counter", 0)

        if os.path.exists(index_file):
            try:
                instance.faiss_index = faiss.read_index(index_file)
                logging.info(f"FAISS 索引已从 '{index_file}' 加载。包含 {instance.faiss_index.ntotal} 个向量。")
                # 验证维度
                if instance.faiss_index.d != instance.embedding_dim:
                    logging.warning(f"加载的FAISS索引维度 ({instance.faiss_index.d}) 与 SemanticMap 期望维度 ({instance.embedding_dim}) 不符。可能需要重建索引。")
            except Exception as e:
                logging.error(f"加载 FAISS 索引失败: {e}。索引将为空，可能需要重建。")
                instance.faiss_index = None # 确保失败时索引为空
                instance._init_faiss_index() # 尝试重新初始化一个空索引结构
        else:
            logging.warning(f"FAISS 索引文件 '{index_file}' 未找到。索引将为空，需要重建。")
            instance._init_faiss_index() # 初始化一个空索引结构
            
        logging.info(f"SemanticMap 已从目录 '{directory_path}' 加载。")
        return instance
    
    def export_to_milvus(
        self, 
        host: str = "localhost", 
        port: str = "19530",
        user: str = "", 
        password: str = "",
        collection_name: str = "hippo_memory_units"
    ) -> bool:
        """
        将SemanticMap中的内存单元导出到Milvus数据库
        
        参数:
            host: Milvus服务器地址
            port: Milvus服务器端口
            user: 用户名（如果需要认证）
            password: 密码（如果需要认证）
            collection_name: 集合名称
        
        返回:
            导出是否成功
        """
        # try:
        #     # 延迟导入，避免强制依赖
        #     from milvus_operator import MilvusOperator
        # except ImportError:
        #     logging.error("未找到milvus_operator模块，请确保已安装pymilvus并创建了milvus_operator.py")
        #     return False
        
        try:
            # 创建Milvus操作类
            milvus_op = MilvusOperator(
                host=host, 
                port=port,
                user=user, 
                password=password,
                collection_name=collection_name,
                embedding_dim=self.embedding_dim
            )
            
            if not milvus_op.is_connected:
                logging.error("连接Milvus失败，无法导出数据")
                return False
            
            # 创建集合
            if not milvus_op.create_collection():
                logging.error("创建Milvus集合失败，无法导出数据")
                return False
            
            # 导出所有内存单元
            success_count = 0
            for uid, unit in self.memory_units.items():
                # 查找单元所属的所有空间
                space_names = []
                for space_name, space in self.memory_spaces.items():
                    if uid in space.get_memory_uids():
                        space_names.append(space_name)
                
                # 添加到Milvus
                if milvus_op.add_unit(unit, space_names):
                    success_count += 1
                else:
                    logging.warning(f"导出内存单元 '{uid}' 到Milvus失败")
            
            logging.info(f"成功导出 {success_count}/{len(self.memory_units)} 个内存单元到Milvus")
            milvus_op.close()
            
            return success_count > 0
            
        except Exception as e:
            logging.error(f"导出到Milvus失败: {e}")
            return False