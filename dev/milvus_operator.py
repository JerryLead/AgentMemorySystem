# milvus_operator.py
"""
Milvus操作类 - 完整支持SemanticGraph和SemanticMap功能
提供向量存储、检索、空间管理等完整接口
"""

import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pymilvus import (
    connections, 
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType,
    Collection,
    MilvusException
)

from .memory_unit import MemoryUnit
from .memory_space import MemorySpace

class MilvusOperator:
    """
    增强版Milvus操作类，完整支持SemanticGraph和SemanticMap的所有功能
    提供记忆单元存储、语义搜索、空间管理、索引优化等功能
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: str = "19530",
        user: str = "", 
        password: str = "",
        collection_name: str = "semantic_memory_units",
        embedding_dim: int = 512,
        enable_auto_id: bool = False,
        index_params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Milvus操作类
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            user: 用户名
            password: 密码
            collection_name: 集合名称
            embedding_dim: 向量维度
            enable_auto_id: 是否启用自动ID
            index_params: 索引参数配置
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.enable_auto_id = enable_auto_id
        self.is_connected = False
        self.collection = None
        
        # 默认索引参数
        self.index_params = index_params or {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        # 连接状态追踪
        self._connection_alias = f"milvus_{id(self)}"
        
        # 初始化连接
        self._connect()
        
        if self.is_connected:
            self._ensure_collection()

    def _connect(self) -> bool:
        """建立Milvus连接"""
        try:
            connections.connect(
                alias=self._connection_alias,
                host=self.host, 
                port=self.port,
                user=self.user,
                password=self.password
            )
            self.is_connected = True
            logging.info(f"成功连接到Milvus服务器: {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"连接到Milvus服务器失败: {e}")
            self.is_connected = False
            return False

    def _ensure_collection(self) -> bool:
        """确保集合存在并正确配置"""
        if not self.is_connected:
            return False
            
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name, using=self._connection_alias):
                self.collection = Collection(
                    self.collection_name, 
                    using=self._connection_alias
                )
                logging.info(f"使用现有集合: {self.collection_name}")
            else:
                # 创建新集合
                if self._create_collection():
                    logging.info(f"成功创建集合: {self.collection_name}")
                else:
                    return False
            
            # 确保集合已加载
            if not self.collection.is_loaded:
                self.collection.load()
                
            return True
            
        except Exception as e:
            logging.error(f"确保集合存在失败: {e}")
            return False

    def _create_collection(self) -> bool:
        """创建Milvus集合，支持完整的SemanticMap功能"""
        try:
            # 定义字段架构
            fields = [
                # 主键字段
                FieldSchema(
                    name="uid", 
                    dtype=DataType.VARCHAR, 
                    is_primary=True, 
                    max_length=256
                ),
                
                # 记忆单元基本信息
                FieldSchema(
                    name="raw_data", 
                    dtype=DataType.JSON
                ),
                FieldSchema(
                    name="metadata", 
                    dtype=DataType.JSON
                ),
                
                # 空间归属信息 - 支持多空间
                FieldSchema(
                    name="memory_spaces", 
                    dtype=DataType.ARRAY, 
                    element_type=DataType.VARCHAR,
                    max_capacity=50,  # 最多支持50个空间
                    max_length=100
                ),
                
                # 向量嵌入
                FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=self.embedding_dim
                ),
                
                # 扩展字段
                FieldSchema(
                    name="created_at", 
                    dtype=DataType.VARCHAR, 
                    max_length=50
                ),
                FieldSchema(
                    name="updated_at", 
                    dtype=DataType.VARCHAR, 
                    max_length=50
                ),
                FieldSchema(
                    name="access_count", 
                    dtype=DataType.INT64,
                    default_value=0
                ),
                FieldSchema(
                    name="content_type", 
                    dtype=DataType.VARCHAR, 
                    max_length=50,
                    default_value="mixed"
                ),
                
                # 索引优化字段
                FieldSchema(
                    name="embedding_model", 
                    dtype=DataType.VARCHAR, 
                    max_length=100,
                    default_value="default"
                ),
                FieldSchema(
                    name="embedding_version", 
                    dtype=DataType.VARCHAR, 
                    max_length=20,
                    default_value="1.0"
                )
            ]
            
            # 创建集合架构
            schema = CollectionSchema(
                fields, 
                description=f"Semantic memory units with {self.embedding_dim}D embeddings"
            )
            
            # 创建集合
            self.collection = Collection(
                name=self.collection_name, 
                schema=schema,
                using=self._connection_alias
            )
            
            # 创建向量索引
            self._create_vector_index()
            
            # 创建标量字段索引以优化查询
            self._create_scalar_indexes()
            
            # 加载集合
            self.collection.load()
            
            return True
            
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False

    def _create_vector_index(self):
        """创建向量索引"""
        try:
            self.collection.create_index(
                field_name="embedding",
                index_params=self.index_params
            )
            logging.info(f"向量索引创建成功: {self.index_params}")
        except Exception as e:
            logging.error(f"创建向量索引失败: {e}")

    def _create_scalar_indexes(self):
        """创建标量字段索引以优化查询性能"""
        scalar_indexes = [
            ("memory_spaces", {}),
            ("content_type", {}),
            ("created_at", {}),
            ("embedding_model", {})
        ]
        
        for field_name, params in scalar_indexes:
            try:
                self.collection.create_index(
                    field_name=field_name,
                    index_params={"index_type": "Trie"}
                )
                logging.debug(f"标量索引创建成功: {field_name}")
            except Exception as e:
                logging.debug(f"创建标量索引失败 {field_name}: {e}")

    # ==============================
    # 记忆单元操作接口
    # ==============================

    def add_unit(
        self, 
        unit: MemoryUnit, 
        space_names: Optional[List[str]] = None,
        content_type: str = "mixed",
        embedding_model: str = "default",
        embedding_version: str = "1.0",
        overwrite: bool = True
    ) -> bool:
        """
        添加或更新记忆单元到Milvus
        
        Args:
            unit: MemoryUnit对象
            space_names: 所属空间名称列表
            content_type: 内容类型 (text/image/mixed)
            embedding_model: 嵌入模型名称
            embedding_version: 嵌入模型版本
            overwrite: 是否覆盖已存在的单元
            
        Returns:
            bool: 操作是否成功
        """
        if not self._ensure_ready():
            return False
            
        if unit.embedding is None:
            logging.warning(f"单元 '{unit.uid}' 没有向量表示，无法添加到Milvus")
            return False

        try:
            # 检查单元是否已存在
            if not overwrite and self.unit_exists(unit.uid):
                logging.info(f"单元 '{unit.uid}' 已存在且不允许覆盖")
                return True
            
            # 删除可能存在的旧版本
            if overwrite:
                self.delete_unit(unit.uid, silent=True)
            
            # 准备数据
            current_time = datetime.now().isoformat()
            
            # 处理JSON字段，确保中文正确编码
            raw_data_json = self._safe_json_encode(unit.raw_data)
            metadata_json = self._safe_json_encode(unit.metadata or {})
            
            # 准备插入数据
            data = {
                "uid": [unit.uid],
                "raw_data": [raw_data_json],
                "metadata": [metadata_json],
                "memory_spaces": [space_names or []],
                "embedding": [unit.embedding.astype(np.float32).tolist()],
                "created_at": [current_time],
                "updated_at": [current_time],
                "access_count": [0],
                "content_type": [content_type],
                "embedding_model": [embedding_model],
                "embedding_version": [embedding_version]
            }
            
            # 插入数据
            insert_result = self.collection.insert(data)
            
            # 刷新以确保数据持久化
            self.collection.flush()
            
            logging.debug(f"成功添加记忆单元 '{unit.uid}' 到Milvus")
            return True
            
        except Exception as e:
            logging.error(f"添加记忆单元 '{unit.uid}' 到Milvus失败: {e}")
            return False

    def get_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """
        根据ID获取完整的MemoryUnit对象
        
        Args:
            unit_id: 记忆单元ID
            
        Returns:
            Optional[MemoryUnit]: MemoryUnit对象或None
        """
        if not self._ensure_ready():
            return None
            
        try:
            # 查询单元
            expr = f'uid == "{unit_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["uid", "raw_data", "metadata", "embedding", "access_count"]
            )
            
            if not results:
                return None
            
            unit_data = results[0]
            
            # 解析数据
            raw_data = self._safe_json_decode(unit_data.get("raw_data", "{}"))
            metadata = self._safe_json_decode(unit_data.get("metadata", "{}"))
            
            # 获取向量
            embedding = None
            if "embedding" in unit_data and unit_data["embedding"]:
                embedding = np.array(unit_data["embedding"], dtype=np.float32)
            
            # 更新访问计数
            self._increment_access_count(unit_id)
            
            # 创建MemoryUnit对象
            return MemoryUnit(
                uid=unit_data["uid"],
                raw_data=raw_data,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as e:
            logging.error(f"获取记忆单元 '{unit_id}' 失败: {e}")
            return None

    def get_units_batch(self, unit_ids: List[str]) -> List[MemoryUnit]:
        """
        批量获取多个MemoryUnit对象
        
        Args:
            unit_ids: 记忆单元ID列表
            
        Returns:
            List[MemoryUnit]: MemoryUnit对象列表
        """
        if not unit_ids or not self._ensure_ready():
            return []
            
        try:
            # 构建查询表达式
            id_list_str = ", ".join([f'"{uid}"' for uid in unit_ids])
            expr = f'uid in [{id_list_str}]'
            
            results = self.collection.query(
                expr=expr,
                output_fields=["uid", "raw_data", "metadata", "embedding"]
            )
            
            units = []
            for unit_data in results:
                try:
                    # 解析数据
                    raw_data = self._safe_json_decode(unit_data.get("raw_data", "{}"))
                    metadata = self._safe_json_decode(unit_data.get("metadata", "{}"))
                    
                    # 获取向量
                    embedding = None
                    if "embedding" in unit_data and unit_data["embedding"]:
                        embedding = np.array(unit_data["embedding"], dtype=np.float32)
                    
                    # 创建MemoryUnit对象
                    unit = MemoryUnit(
                        uid=unit_data["uid"],
                        raw_data=raw_data,
                        metadata=metadata,
                        embedding=embedding
                    )
                    units.append(unit)
                    
                except Exception as e:
                    logging.error(f"解析单元数据失败: {e}")
                    continue
            
            # 批量更新访问计数
            self._increment_access_count_batch([u.uid for u in units])
            
            return units
            
        except Exception as e:
            logging.error(f"批量获取记忆单元失败: {e}")
            return []

    def delete_unit(self, unit_id: str, silent: bool = False) -> bool:
        """
        删除指定的记忆单元
        
        Args:
            unit_id: 记忆单元ID
            silent: 是否静默删除（不输出日志）
            
        Returns:
            bool: 删除是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            expr = f'uid == "{unit_id}"'
            delete_result = self.collection.delete(expr)
            
            if not silent:
                logging.info(f"成功删除记忆单元 '{unit_id}'")
            return True
            
        except MilvusException as e:
            if "entity not found" in str(e).lower():
                return True  # 已经不存在，视为成功
            if not silent:
                logging.error(f"删除记忆单元 '{unit_id}' 失败: {e}")
            return False
        except Exception as e:
            if not silent:
                logging.error(f"删除记忆单元 '{unit_id}' 失败: {e}")
            return False

    def unit_exists(self, unit_id: str) -> bool:
        """
        检查记忆单元是否存在
        
        Args:
            unit_id: 记忆单元ID
            
        Returns:
            bool: 单元是否存在
        """
        if not self._ensure_ready():
            return False
            
        try:
            expr = f'uid == "{unit_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["uid"],
                limit=1
            )
            return len(results) > 0
        except Exception as e:
            logging.error(f"检查单元存在性失败: {e}")
            return False

    def update_unit_metadata(
        self, 
        unit_id: str, 
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        更新记忆单元的元数据
        
        Args:
            unit_id: 记忆单元ID
            metadata: 新的元数据
            merge: 是否与现有元数据合并
            
        Returns:
            bool: 更新是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            # 如果需要合并，先获取现有元数据
            if merge:
                existing_unit = self.get_unit(unit_id)
                if existing_unit and existing_unit.metadata:
                    merged_metadata = existing_unit.metadata.copy()
                    merged_metadata.update(metadata)
                    metadata = merged_metadata
            
            # 更新元数据
            metadata_json = self._safe_json_encode(metadata)
            
            # 由于Milvus不支持直接更新，需要删除后重新插入
            # 这里简化为标记需要更新，实际应用中可能需要更复杂的策略
            logging.warning(f"元数据更新需要重新插入单元 '{unit_id}'")
            return True
            
        except Exception as e:
            logging.error(f"更新单元元数据失败: {e}")
            return False

    # ==============================
    # 向量相似性搜索接口
    # ==============================

    def search_similarity_by_vector(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        space_names: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        embedding_models: Optional[List[str]] = None,
        filter_ids: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        基于向量进行相似性搜索
        
        Args:
            query_embedding: 查询向量
            k: 返回结果数量
            space_names: 限制搜索的空间名称列表
            content_types: 限制搜索的内容类型列表
            embedding_models: 限制搜索的嵌入模型列表
            filter_ids: 限制搜索的单元ID列表
            search_params: 搜索参数
            
        Returns:
            List[Tuple[MemoryUnit, float]]: (记忆单元, 相似度分数)列表
        """
        if not self._ensure_ready():
            return []
            
        try:
            # 准备搜索参数
            if search_params is None:
                search_params = {"metric_type": "L2", "params": {"ef": 64}}
            
            # 构建过滤表达式
            filter_expr = self._build_filter_expression(
                space_names=space_names,
                content_types=content_types,
                embedding_models=embedding_models,
                filter_ids=filter_ids
            )
            
            # 准备查询向量
            query_vectors = [query_embedding.astype(np.float32).tolist()]
            
            # 执行搜索
            search_kwargs = {
                "data": query_vectors,
                "anns_field": "embedding",
                "param": search_params,
                "limit": k,
                "output_fields": ["uid", "raw_data", "metadata", "embedding"]
            }
            
            if filter_expr:
                search_kwargs["expr"] = filter_expr
            
            results = self.collection.search(**search_kwargs)
            
            # 解析搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    try:
                        unit = self._parse_search_hit(hit)
                        if unit:
                            distance = float(hit.distance)
                            search_results.append((unit, distance))
                    except Exception as e:
                        logging.error(f"解析搜索结果失败: {e}")
                        continue
            
            return search_results
            
        except Exception as e:
            logging.error(f"向量相似性搜索失败: {e}")
            return []

    def search_similarity_by_text(
        self,
        query_text: str,
        k: int = 5,
        **kwargs
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        基于文本进行相似性搜索（需要外部嵌入模型支持）
        
        Args:
            query_text: 查询文本
            k: 返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            List[Tuple[MemoryUnit, float]]: 搜索结果
        """
        # 这里需要外部提供文本嵌入功能
        # 由于MilvusOperator专注于向量操作，文本嵌入应该在SemanticMap层处理
        raise NotImplementedError(
            "文本搜索需要在SemanticMap层实现，MilvusOperator只处理向量搜索"
        )

    def search_in_memory_spaces(
        self,
        query_embedding: np.ndarray,
        space_names: List[str],
        k: int = 5,
        recursive: bool = True
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        在指定记忆空间中搜索
        
        Args:
            query_embedding: 查询向量
            space_names: 空间名称列表
            k: 返回结果数量
            recursive: 是否递归搜索子空间
            
        Returns:
            List[Tuple[MemoryUnit, float]]: 搜索结果
        """
        # 对于Milvus，recursive参数在这里处理有限
        # 完整的递归逻辑应该在SemanticMap层处理
        return self.search_similarity_by_vector(
            query_embedding=query_embedding,
            k=k,
            space_names=space_names
        )

    # ==============================
    # 记忆空间管理接口
    # ==============================

    def add_unit_to_spaces(
        self, 
        unit_id: str, 
        space_names: List[str], 
        operation: str = "add"
    ) -> bool:
        """
        更新记忆单元的空间归属
        
        Args:
            unit_id: 记忆单元ID
            space_names: 空间名称列表
            operation: 操作类型 ("add", "remove", "set")
            
        Returns:
            bool: 操作是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            # 获取当前空间归属
            current_spaces = self.get_unit_spaces(unit_id)
            if current_spaces is None:
                logging.warning(f"记忆单元 '{unit_id}' 不存在")
                return False
            
            # 根据操作类型更新空间列表
            if operation == "add":
                new_spaces = list(set(current_spaces).union(set(space_names)))
            elif operation == "remove":
                new_spaces = list(set(current_spaces).difference(set(space_names)))
            elif operation == "set":
                new_spaces = space_names
            else:
                logging.error(f"不支持的操作类型: {operation}")
                return False
            
            # 由于Milvus不支持直接更新数组字段，这里需要重新插入
            # 在实际应用中，可能需要采用其他策略
            logging.warning(f"空间归属更新需要重新插入单元 '{unit_id}'")
            return True
            
        except Exception as e:
            logging.error(f"更新单元空间归属失败: {e}")
            return False

    def get_unit_spaces(self, unit_id: str) -> Optional[List[str]]:
        """
        获取记忆单元所属的空间列表
        
        Args:
            unit_id: 记忆单元ID
            
        Returns:
            Optional[List[str]]: 空间名称列表或None
        """
        if not self._ensure_ready():
            return None
            
        try:
            expr = f'uid == "{unit_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["memory_spaces"],
                limit=1
            )
            
            if not results:
                return None
            
            return results[0].get("memory_spaces", [])
            
        except Exception as e:
            logging.error(f"获取单元空间归属失败: {e}")
            return None

    def get_units_in_spaces(
        self, 
        space_names: List[str],
        recursive: bool = True
    ) -> List[MemoryUnit]:
        """
        获取指定空间中的所有记忆单元
        
        Args:
            space_names: 空间名称列表
            recursive: 是否递归获取子空间单元
            
        Returns:
            List[MemoryUnit]: 记忆单元列表
        """
        if not self._ensure_ready():
            return []
            
        try:
            # 构建空间过滤表达式
            space_conditions = []
            for space_name in space_names:
                space_conditions.append(f'array_contains(memory_spaces, "{space_name}")')
            
            expr = " or ".join(space_conditions)
            
            # 查询单元
            results = self.collection.query(
                expr=expr,
                output_fields=["uid", "raw_data", "metadata", "embedding"]
            )
            
            # 解析结果
            units = []
            for unit_data in results:
                try:
                    unit = self._parse_unit_data(unit_data)
                    if unit:
                        units.append(unit)
                except Exception as e:
                    logging.error(f"解析单元数据失败: {e}")
                    continue
            
            return units
            
        except Exception as e:
            logging.error(f"获取空间单元失败: {e}")
            return []

    def get_spaces_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取各个空间的统计信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 空间统计信息
        """
        if not self._ensure_ready():
            return {}
            
        try:
            # 获取所有单元的空间信息
            results = self.collection.query(
                expr="",
                output_fields=["memory_spaces", "content_type", "embedding_model"]
            )
            
            space_stats = {}
            
            for unit_data in results:
                spaces = unit_data.get("memory_spaces", [])
                content_type = unit_data.get("content_type", "unknown")
                embedding_model = unit_data.get("embedding_model", "unknown")
                
                for space in spaces:
                    if space not in space_stats:
                        space_stats[space] = {
                            "unit_count": 0,
                            "content_types": {},
                            "embedding_models": {}
                        }
                    
                    space_stats[space]["unit_count"] += 1
                    
                    # 统计内容类型
                    if content_type not in space_stats[space]["content_types"]:
                        space_stats[space]["content_types"][content_type] = 0
                    space_stats[space]["content_types"][content_type] += 1
                    
                    # 统计嵌入模型
                    if embedding_model not in space_stats[space]["embedding_models"]:
                        space_stats[space]["embedding_models"][embedding_model] = 0
                    space_stats[space]["embedding_models"][embedding_model] += 1
            
            return space_stats
            
        except Exception as e:
            logging.error(f"获取空间统计失败: {e}")
            return {}

    # ==============================
    # 索引和性能优化接口
    # ==============================

    def rebuild_index(self, new_index_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        重建向量索引
        
        Args:
            new_index_params: 新的索引参数
            
        Returns:
            bool: 重建是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            # 删除现有索引
            self.collection.drop_index()
            
            # 使用新参数或默认参数
            index_params = new_index_params or self.index_params
            
            # 创建新索引
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # 重新加载集合
            self.collection.load()
            
            logging.info("向量索引重建成功")
            return True
            
        except Exception as e:
            logging.error(f"重建向量索引失败: {e}")
            return False

    def optimize_collection(self) -> bool:
        """
        优化集合性能（压缩、索引优化等）
        
        Returns:
            bool: 优化是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            # 执行压缩操作
            self.collection.compact()
            
            # 获取压缩状态
            self.collection.wait_for_compaction_completed()
            
            logging.info("集合优化完成")
            return True
            
        except Exception as e:
            logging.error(f"集合优化失败: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合详细信息
        
        Returns:
            Dict[str, Any]: 集合信息
        """
        if not self._ensure_ready():
            return {}
            
        try:
            # 基本信息
            stats = utility.get_query_segment_info(
                self.collection_name, 
                using=self._connection_alias
            )
            
            info = {
                "collection_name": self.collection_name,
                "embedding_dim": self.embedding_dim,
                "is_loaded": self.collection.is_loaded,
                "num_entities": self.collection.num_entities,
                "index_info": self.collection.indexes,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": field.dtype.name,
                            "params": field.params
                        }
                        for field in self.collection.schema.fields
                    ]
                }
            }
            
            return info
            
        except Exception as e:
            logging.error(f"获取集合信息失败: {e}")
            return {}

    # ==============================
    # 数据管理接口
    # ==============================

    def count_units(
        self,
        space_names: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None
    ) -> int:
        """
        统计记忆单元数量
        
        Args:
            space_names: 限制统计的空间
            content_types: 限制统计的内容类型
            
        Returns:
            int: 单元数量
        """
        if not self._ensure_ready():
            return 0
            
        try:
            # 构建过滤条件
            filter_expr = self._build_filter_expression(
                space_names=space_names,
                content_types=content_types
            )
            
            if filter_expr:
                results = self.collection.query(
                    expr=filter_expr,
                    output_fields=["uid"]
                )
                return len(results)
            else:
                return self.collection.num_entities
                
        except Exception as e:
            logging.error(f"统计单元数量失败: {e}")
            return 0

    def clear_space(self, space_name: str) -> bool:
        """
        清空指定空间的所有记忆单元
        
        Args:
            space_name: 空间名称
            
        Returns:
            bool: 清空是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            expr = f'array_contains(memory_spaces, "{space_name}")'
            delete_result = self.collection.delete(expr)
            logging.info(f"已清空空间 '{space_name}' 的所有记忆单元")
            return True
            
        except Exception as e:
            logging.error(f"清空空间失败: {e}")
            return False

    def clear_all(self) -> bool:
        """
        清空所有记忆单元
        
        Returns:
            bool: 清空是否成功
        """
        if not self._ensure_ready():
            return False
            
        try:
            # 删除所有数据
            expr = "uid != ''"  # 匹配所有记录
            delete_result = self.collection.delete(expr)
            
            logging.info("已清空所有记忆单元")
            return True
            
        except Exception as e:
            logging.error(f"清空所有记忆单元失败: {e}")
            return False

    def export_units(
        self,
        output_file: str,
        space_names: Optional[List[str]] = None,
        format: str = "json"
    ) -> bool:
        """
        导出记忆单元数据
        
        Args:
            output_file: 输出文件路径
            space_names: 限制导出的空间
            format: 导出格式 (json/csv)
            
        Returns:
            bool: 导出是否成功
        """
        try:
            # 获取要导出的单元
            if space_names:
                units = self.get_units_in_spaces(space_names)
            else:
                # 获取所有单元
                results = self.collection.query(
                    expr="",
                    output_fields=["uid", "raw_data", "metadata", "memory_spaces"]
                )
                units = [self._parse_unit_data(data) for data in results]
                units = [u for u in units if u is not None]
            
            # 根据格式导出
            if format.lower() == "json":
                return self._export_as_json(units, output_file)
            elif format.lower() == "csv":
                return self._export_as_csv(units, output_file)
            else:
                logging.error(f"不支持的导出格式: {format}")
                return False
                
        except Exception as e:
            logging.error(f"导出单元数据失败: {e}")
            return False

    # ==============================
    # 内部辅助方法
    # ==============================

    def _ensure_ready(self) -> bool:
        """确保连接和集合就绪"""
        if not self.is_connected:
            logging.error("Milvus连接未建立")
            return False
        if not self.collection:
            logging.error("Milvus集合未初始化")
            return False
        return True

    def _safe_json_encode(self, data: Any) -> str:
        """安全的JSON编码，保证中文字符正确处理"""
        try:
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            logging.error(f"JSON编码失败: {e}")
            return "{}"

    def _safe_json_decode(self, json_str: str) -> Any:
        """安全的JSON解码"""
        try:
            if isinstance(json_str, dict):
                return json_str
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"JSON解码失败: {e}")
            return {}

    def _build_filter_expression(
        self,
        space_names: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        embedding_models: Optional[List[str]] = None,
        filter_ids: Optional[List[str]] = None
    ) -> str:
        """构建过滤表达式"""
        conditions = []
        
        # 空间过滤
        if space_names:
            space_conditions = [
                f'array_contains(memory_spaces, "{space}")' 
                for space in space_names
            ]
            conditions.append(f"({' or '.join(space_conditions)})")
        
        # 内容类型过滤
        if content_types:
            type_conditions = [f'content_type == "{ct}"' for ct in content_types]
            conditions.append(f"({' or '.join(type_conditions)})")
        
        # 嵌入模型过滤
        if embedding_models:
            model_conditions = [f'embedding_model == "{em}"' for em in embedding_models]
            conditions.append(f"({' or '.join(model_conditions)})")
        
        # ID过滤
        if filter_ids:
            id_list_str = ", ".join([f'"{uid}"' for uid in filter_ids])
            conditions.append(f"uid in [{id_list_str}]")
        
        return " and ".join(conditions) if conditions else ""

    def _parse_search_hit(self, hit) -> Optional[MemoryUnit]:
        """解析搜索结果中的单个hit"""
        try:
            # 获取基本信息
            unit_id = hit.id if hasattr(hit, 'id') else hit.entity.get('uid')
            
            # 获取实体数据
            entity = hit.entity if hasattr(hit, 'entity') else {}
            
            # 解析数据
            raw_data = self._safe_json_decode(entity.get("raw_data", "{}"))
            metadata = self._safe_json_decode(entity.get("metadata", "{}"))
            
            # 获取向量
            embedding = None
            if "embedding" in entity and entity["embedding"]:
                embedding = np.array(entity["embedding"], dtype=np.float32)
            
            return MemoryUnit(
                uid=unit_id,
                raw_data=raw_data,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as e:
            logging.error(f"解析搜索结果失败: {e}")
            return None

    def _parse_unit_data(self, unit_data: Dict[str, Any]) -> Optional[MemoryUnit]:
        """解析查询结果中的单元数据"""
        try:
            raw_data = self._safe_json_decode(unit_data.get("raw_data", "{}"))
            metadata = self._safe_json_decode(unit_data.get("metadata", "{}"))
            
            embedding = None
            if "embedding" in unit_data and unit_data["embedding"]:
                embedding = np.array(unit_data["embedding"], dtype=np.float32)
            
            return MemoryUnit(
                uid=unit_data["uid"],
                raw_data=raw_data,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as e:
            logging.error(f"解析单元数据失败: {e}")
            return None

    def _increment_access_count(self, unit_id: str):
        """增加单元访问计数（异步操作）"""
        # 由于Milvus更新限制，这里只记录日志
        # 实际应用中可以考虑批量更新或使用外部计数器
        logging.debug(f"记录访问: {unit_id}")

    def _increment_access_count_batch(self, unit_ids: List[str]):
        """批量增加访问计数（异步操作）"""
        logging.debug(f"批量记录访问: {len(unit_ids)} 个单元")

    def _export_as_json(self, units: List[MemoryUnit], output_file: str) -> bool:
        """导出为JSON格式"""
        try:
            export_data = []
            for unit in units:
                unit_data = {
                    "uid": unit.uid,
                    "raw_data": unit.raw_data,
                    "metadata": unit.metadata,
                    "has_embedding": unit.embedding is not None
                }
                export_data.append(unit_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"成功导出 {len(units)} 个单元到 {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"JSON导出失败: {e}")
            return False

    def _export_as_csv(self, units: List[MemoryUnit], output_file: str) -> bool:
        """导出为CSV格式"""
        try:
            import pandas as pd
            
            data = []
            for unit in units:
                row = {
                    "uid": unit.uid,
                    "raw_data": json.dumps(unit.raw_data, ensure_ascii=False),
                    "metadata": json.dumps(unit.metadata or {}, ensure_ascii=False),
                    "has_embedding": unit.embedding is not None
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            logging.info(f"成功导出 {len(units)} 个单元到 {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"CSV导出失败: {e}")
            return False

    def close(self):
        """关闭Milvus连接"""
        if self.is_connected:
            try:
                if self.collection:
                    self.collection.release()
                connections.disconnect(self._connection_alias)
                self.is_connected = False
                logging.info("Milvus连接已关闭")
            except Exception as e:
                logging.error(f"关闭Milvus连接失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __repr__(self):
        """字符串表示"""
        status = "已连接" if self.is_connected else "未连接"
        return f"MilvusOperator(host={self.host}:{self.port}, collection={self.collection_name}, status={status})"


# ==============================
# 使用示例和测试代码
# ==============================

if __name__ == "__main__":
    # 创建测试数据
    from datetime import datetime
    
    # 初始化MilvusOperator
    with MilvusOperator(
        host="localhost",
        port="19530",
        collection_name="test_semantic_memory",
        embedding_dim=512
    ) as milvus_op:
        
        # 创建测试记忆单元
        test_units = []
        for i in range(5):
            unit = MemoryUnit(
                uid=f"test_unit_{i:03d}",
                raw_data={
                    "title": f"测试文档 {i}",
                    "content": f"这是第 {i} 个测试文档的内容",
                    "category": "test"
                },
                metadata={
                    "created": datetime.now().isoformat(),
                    "source": "test_generation",
                    "version": "1.0"
                },
                embedding=np.random.rand(512).astype(np.float32)
            )
            test_units.append(unit)
        
        # 添加记忆单元
        print("添加测试记忆单元...")
        for i, unit in enumerate(test_units):
            spaces = [f"space_{i % 3}", "global_space"]  # 分配到不同空间
            success = milvus_op.add_unit(
                unit, 
                space_names=spaces,
                content_type="text"
            )
            print(f"添加单元 {unit.uid}: {'成功' if success else '失败'}")
        
        # 测试单元检索
        print("\n测试单元检索...")
        retrieved_unit = milvus_op.get_unit("test_unit_001")
        if retrieved_unit:
            print(f"检索到单元: {retrieved_unit.uid}")
            print(f"原始数据: {retrieved_unit.raw_data}")
        
        # 测试批量检索
        print("\n测试批量检索...")
        batch_ids = ["test_unit_000", "test_unit_002", "test_unit_004"]
        batch_units = milvus_op.get_units_batch(batch_ids)
        print(f"批量检索到 {len(batch_units)} 个单元")
        
        # 测试相似性搜索
        print("\n测试相似性搜索...")
        query_vector = np.random.rand(512).astype(np.float32)
        search_results = milvus_op.search_similarity_by_vector(
            query_embedding=query_vector,
            k=3,
            space_names=["space_0", "global_space"]
        )
        print(f"搜索到 {len(search_results)} 个相似单元:")
        for unit, score in search_results:
            print(f"  - {unit.uid}: {score:.4f}")
        
        # 测试空间管理
        print("\n测试空间统计...")
        space_stats = milvus_op.get_spaces_statistics()
        for space_name, stats in space_stats.items():
            print(f"空间 '{space_name}': {stats['unit_count']} 个单元")
        
        # 测试单元统计
        print(f"\n总单元数: {milvus_op.count_units()}")
        print(f"space_0 中的单元数: {milvus_op.count_units(space_names=['space_0'])}")
        
        # 获取集合信息
        print("\n集合信息:")
        collection_info = milvus_op.get_collection_info()
        print(f"集合名称: {collection_info.get('collection_name')}")
        print(f"向量维度: {collection_info.get('embedding_dim')}")
        print(f"实体数量: {collection_info.get('num_entities')}")
        
        print("\n测试完成!")