# Hippo.py

import os
import pickle
import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx

# 配置日志记录器
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# --- 基础数据结构 ---

class MemoryUnit:
    """
    内存单元 (MemoryUnit) 代表系统中的一个基本信息片段。
    它包含一个唯一的ID、一个存储具体数据的字典以及一个可选的向量表示。
    """
    def __init__(self, unit_id: str, value: Dict[str, Any], vector: Optional[np.ndarray] = None):
        """
        初始化一个内存单元。
        参数:
            unit_id (str): 内存单元的唯一标识符。
            value (Dict[str, Any]): 包含内存单元具体内容的字典。
                                     例如: {"text": "一些描述", "image_path": "路径/到/图片.jpg", "type": "文档"}
            vector (Optional[np.ndarray]): 该单元的向量表示。如果为None，则可以由SemanticMap生成。
        """
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError("内存单元ID (unit_id) 不能为空字符串。")
        if not isinstance(value, dict):
            raise ValueError("内存单元值 (value) 必须是一个字典。")

        self.id: str = unit_id
        self.value: Dict[str, Any] = value
        self.vector: Optional[np.ndarray] = vector

    def __repr__(self) -> str:
        vector_shape = self.vector.shape if self.vector is not None else None
        return f"MemoryUnit(id='{self.id}', value={self.value}, vector_shape={vector_shape})"

class MemorySpace:
    """
    内存空间 (MemorySpace) 用于组织和管理一组相关的内存单元。
    一个内存单元可以属于多个内存空间。
    """
    def __init__(self, name: str):
        """
        初始化一个内存空间。
        参数:
            name (str): 内存空间的名称，应唯一。
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("内存空间名称 (name) 不能为空字符串。")
        self.name: str = name
        self.memory_unit_ids: Set[str] = set() # 存储属于此空间的 MemoryUnit 的 ID

    def add_memory_unit(self, unit_id: str):
        """向此内存空间添加一个内存单元的ID。"""
        if not isinstance(unit_id, str) or not unit_id.strip():
            logging.warning(f"尝试向内存空间 '{self.name}' 添加无效的单元ID。")
            return
        self.memory_unit_ids.add(unit_id)
        logging.debug(f"内存单元 '{unit_id}' 已添加到内存空间 '{self.name}'。")

    def remove_memory_unit(self, unit_id: str):
        """从此内存空间移除一个内存单元的ID。"""
        if unit_id in self.memory_unit_ids:
            self.memory_unit_ids.remove(unit_id)
            logging.debug(f"内存单元 '{unit_id}' 已从内存空间 '{self.name}' 移除。")
        else:
            logging.warning(f"尝试从内存空间 '{self.name}' 移除不存在的内存单元ID '{unit_id}'。")

    def get_memory_unit_ids(self) -> Set[str]:
        """获取此内存空间中所有内存单元的ID集合。"""
        return self.memory_unit_ids

    def __repr__(self) -> str:
        return f"MemorySpace(name='{self.name}', unit_count={len(self.memory_unit_ids)})"

# --- 语义地图 ---

class SemanticMap:
    """
    语义地图 (SemanticMap) 负责存储内存单元及其向量嵌入，并支持基于相似度的搜索。
    它还管理内存空间，允许在特定上下文中进行操作。
    类似于一个向量数据库。
    """
    DEFAULT_TEXT_EMBEDDING_KEY = "text_content" # MemoryUnit.value 中用于文本嵌入的默认键
    DEFAULT_IMAGE_EMBEDDING_KEY = "image_path" # MemoryUnit.value 中用于图像嵌入的默认键

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
        
        self.memory_units: Dict[str, MemoryUnit] = {} # 存储 MemoryUnit 对象，键为 unit_id
        self.memory_spaces: Dict[str, MemorySpace] = {} # 存储 MemorySpace 对象，键为 space_name

        self.faiss_index: Optional[faiss.Index] = None # FAISS 索引
        
        # FAISS ID 到 MemoryUnit ID 的映射 (以及反向映射)
        # FAISS 内部使用从0开始的连续整数ID。我们需要将这些ID映射回我们的MemoryUnit ID。
        # 当使用 IndexIDMap 时，我们传递给 add_with_ids 的 ID 可以是我们自己的整数ID。
        # 我们将使用一个内部计数器或 MemoryUnit ID 的哈希（如果需要稳定且非连续的ID）
        # 为简单起见，我们将维护一个从 MemoryUnit.id 到一个内部 faiss_id (int64) 的映射。
        self._unit_id_to_internal_faiss_id: Dict[str, int] = {}
        self._internal_faiss_id_counter: int = 0 # 用于生成唯一的内部 FAISS ID

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
            # 我们将使用一个内部计数器生成的ID，并维护 unit_id -> internal_faiss_id 的映射
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
                logging.warning(f"不支持的内容类型 '{content_type}' 用于为单元 '{unit.id}' 生成嵌入。")
        else: # 尝试从 unit.value 推断
            image_path = unit.value.get(self.DEFAULT_IMAGE_EMBEDDING_KEY)
            text_content = unit.value.get(self.DEFAULT_TEXT_EMBEDDING_KEY)

            if image_path and isinstance(image_path, str):
                embedding = self._get_image_embedding(image_path)
                if embedding is not None:
                    logging.debug(f"已为单元 '{unit.id}' 从其值中的图像路径 '{image_path}' 生成嵌入。")
            
            if embedding is None and text_content and isinstance(text_content, str): # 如果图像嵌入失败或未提供图像，则尝试文本
                embedding = self._get_text_embedding(text_content)
                if embedding is not None:
                    logging.debug(f"已为单元 '{unit.id}' 从其值中的文本内容生成嵌入。")
            
            if embedding is None:
                logging.warning(f"无法从单元 '{unit.id}' 的值中找到合适的内容来生成嵌入。请检查键 '{self.DEFAULT_IMAGE_EMBEDDING_KEY}' 或 '{self.DEFAULT_TEXT_EMBEDDING_KEY}'。")
        
        if embedding is not None and embedding.shape[0] != self.embedding_dim:
            logging.error(f"为单元 '{unit.id}' 生成的嵌入维度 ({embedding.shape[0]}) 与期望维度 ({self.embedding_dim}) 不符。")
            return None
        return embedding

    def add_memory_unit(self,
                        unit: MemoryUnit,
                        explicit_content_for_embedding: Optional[Any] = None,
                        content_type_for_embedding: Optional[str] = None, # "text" or "image_path"
                        space_names: Optional[List[str]] = None,
                        rebuild_index_immediately: bool = False):
        """
        向语义地图添加一个新的内存单元。
        如果单元已存在，则其值和嵌入将被更新。
        参数:
            unit (MemoryUnit): 要添加或更新的内存单元对象。
            explicit_content_for_embedding (Optional[Any]): 用于生成嵌入的显式内容。如果提供，将覆盖从 unit.value 推断的内容。
            content_type_for_embedding (Optional[str]): explicit_content_for_embedding 的类型 ("text" 或 "image_path")。
            space_names (Optional[List[str]]): 要将此单元添加到的内存空间的名称列表。如果空间不存在，将创建它们。
            rebuild_index_immediately (bool): 是否在添加后立即重建FAISS索引。对于批量添加，建议设置为False，并在最后统一重建。
        """
        if not isinstance(unit, MemoryUnit):
            logging.error("尝试添加的不是 MemoryUnit 对象。")
            return

        # 生成或更新嵌入
        new_embedding = self._generate_embedding_for_unit(unit, explicit_content_for_embedding, content_type_for_embedding)
        
        if new_embedding is None:
            logging.warning(f"未能为内存单元 '{unit.id}' 生成嵌入。该单元将被添加，但无法用于相似性搜索，除非后续更新嵌入并重建索引。")
        unit.vector = new_embedding # 更新单元对象中的向量

        # 如果单元已存在，可能需要从FAISS中移除旧向量 (如果向量已更改)
        # 为了简单起见，如果 rebuild_index_immediately 为 True，或者依赖于后续的 build_index() 调用来处理更新。
        # 更精细的更新会在这里处理 self.faiss_index.remove_ids()。

        self.memory_units[unit.id] = unit
        logging.info(f"内存单元 '{unit.id}' 已添加/更新到 SemanticMap。")

        if space_names:
            for space_name in space_names:
                self.add_unit_to_space(unit.id, space_name)
        
        if rebuild_index_immediately:
            self.build_index() # 立即重建索引 (可能效率不高，除非是单个添加)

    def get_memory_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """通过ID检索内存单元。"""
        return self.memory_units.get(unit_id)

    def delete_memory_unit(self, unit_id: str, rebuild_index_immediately: bool = False):
        """从语义地图中删除一个内存单元。"""
        if unit_id in self.memory_units:
            del self.memory_units[unit_id]
            
            # 从所有内存空间中移除
            for space in self.memory_spaces.values():
                space.remove_memory_unit(unit_id)
            
            # 从FAISS索引中移除 (如果已构建且包含该单元)
            if self.faiss_index and unit_id in self._unit_id_to_internal_faiss_id:
                internal_id_to_remove = self._unit_id_to_internal_faiss_id[unit_id]
                try:
                    # FAISS 的 remove_ids 需要一个包含ID的numpy数组
                    self.faiss_index.remove_ids(np.array([internal_id_to_remove], dtype=np.int64))
                    logging.info(f"内存单元 '{unit_id}' (内部FAISS ID: {internal_id_to_remove}) 已从FAISS索引中移除。")
                    del self._unit_id_to_internal_faiss_id[unit_id]
                    # 注意：简单的 remove_ids 可能会导致ID空间不连续，这对于某些FAISS索引类型可能是个问题。
                    # IndexIDMap 通常能处理这个问题。
                except Exception as e:
                    logging.error(f"从FAISS索引中移除单元 '{unit_id}' 失败: {e}。建议重建索引。")
            else:
                logging.warning(f"单元 '{unit_id}' 不在FAISS ID映射中，可能未被索引或已被移除。")

            logging.info(f"内存单元 '{unit_id}' 已从 SemanticMap 删除。")
            if rebuild_index_immediately:
                self.build_index()
        else:
            logging.warning(f"尝试删除不存在的内存单元ID '{unit_id}'。")

    def build_index(self):
        """
        根据当前所有具有有效嵌入的内存单元构建（或重建）FAISS索引。
        """
        if not self.memory_units:
            logging.info("没有内存单元可用于构建索引。")
            if self.faiss_index: self.faiss_index.reset() # 清空现有索引
            self._unit_id_to_internal_faiss_id.clear()
            self._internal_faiss_id_counter = 0
            return

        valid_embeddings: List[np.ndarray] = []
        internal_faiss_ids_for_index: List[int] = []
        
        # 重置映射和计数器，因为我们要重建
        self._unit_id_to_internal_faiss_id.clear()
        # self._internal_faiss_id_counter = 0 # 如果希望ID在多次重建中保持某种程度的稳定性，则不要重置计数器，除非ID用完。
                                          # 但对于 IndexIDMap，每次重建时使用新的连续ID可能更简单。

        current_internal_id = 0
        for unit_id, unit in self.memory_units.items():
            if unit.vector is not None and unit.vector.shape[0] == self.embedding_dim:
                valid_embeddings.append(unit.vector)
                # 分配一个新的内部FAISS ID
                self._unit_id_to_internal_faiss_id[unit_id] = current_internal_id
                internal_faiss_ids_for_index.append(current_internal_id)
                current_internal_id += 1
            else:
                logging.debug(f"内存单元 '{unit_id}' 没有有效向量，将不被包含在FAISS索引中。")

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


    def search_similarity_by_vector(self,
                                query_vector: np.ndarray,
                                k: int = 5,
                                space_name: Optional[str] = None,
                                filter_unit_ids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        通过查询向量在语义地图中搜索相似的内存单元。
        参数:
            query_vector (np.ndarray): 用于查询的嵌入向量。
            k (int): 要返回的最相似单元的数量。
            space_name (Optional[str]): 如果提供，则仅在指定的内存空间内搜索。
            filter_unit_ids (Optional[Set[str]]): 一个可选的单元ID集合，用于进一步限制搜索范围（仅搜索这些ID对应的单元）。
                                                如果同时提供了 space_name 和 filter_unit_ids，则取它们的交集。
        返回:
            List[Tuple[MemoryUnit, float]]: 一个元组列表，每个元组包含 (相似的MemoryUnit, 相似度得分/距离)。
                                            列表按相似度降序排列 (距离越小越相似)。
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logging.warning("FAISS索引未构建或为空。无法执行搜索。")
            return []
        if query_vector is None or query_vector.shape[0] != self.embedding_dim:
            logging.error(f"查询向量无效或维度不匹配 (期望 {self.embedding_dim})。")
            return []

        query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
        
        search_params = None
        final_target_internal_ids_np = None

        # 确定搜索范围 (所有单元，特定空间内的单元，或特定ID列表中的单元)
        candidate_unit_ids: Optional[Set[str]] = None
        if space_name:
            space = self.get_memory_space(space_name)
            if space:
                candidate_unit_ids = space.get_memory_unit_ids().copy() # 获取副本
            else:
                logging.warning(f"内存空间 '{space_name}' 未找到。将执行全局搜索或基于 filter_unit_ids 的搜索。")
                # 如果空间不存在，则不应返回任何结果，除非filter_unit_ids也为空
                if not filter_unit_ids: return []


        if filter_unit_ids:
            if candidate_unit_ids is not None:
                candidate_unit_ids.intersection_update(filter_unit_ids) # 取交集
            else:
                candidate_unit_ids = filter_unit_ids.copy()
        
        if candidate_unit_ids is not None: # 如果有任何过滤条件
            if not candidate_unit_ids: # 如果过滤后候选集为空
                logging.info("根据空间和/或ID过滤器，没有候选内存单元可供搜索。")
                return []
            
            # 将候选的 MemoryUnit ID 转换为内部 FAISS ID
            target_internal_faiss_ids = [self._unit_id_to_internal_faiss_id[uid] 
                                        for uid in candidate_unit_ids if uid in self._unit_id_to_internal_faiss_id]
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
            distances, internal_faiss_indices = self.faiss_index.search(query_vector_np, k, params=search_params)
        except RuntimeError as e:
            if "search params not supported for this index" in str(e):
                # 索引不支持搜索参数，使用不带参数的搜索
                logging.warning(f"当前FAISS索引类型 '{self.faiss_index_type}' 不支持搜索参数。执行不带过滤器的搜索。")
                if space_name or filter_unit_ids:
                    logging.warning("空间和ID过滤将在搜索后手动应用。")
                distances, internal_faiss_indices = self.faiss_index.search(query_vector_np, k)
            else:
                # 如果是其他类型的错误，则重新抛出
                raise
        
        results: List[Tuple[MemoryUnit, float]] = []
        # 反向映射：从内部FAISS ID找到MemoryUnit ID
        # 创建一个从 internal_faiss_id 到 unit_id 的临时反向映射
        internal_id_to_unit_id_map = {v: k for k, v in self._unit_id_to_internal_faiss_id.items()}

        for i in range(internal_faiss_indices.shape[1]):
            internal_id = internal_faiss_indices[0, i]
            if internal_id == -1: # FAISS中表示没有更多结果
                continue
            
            unit_id = internal_id_to_unit_id_map.get(internal_id)
            if unit_id:
                unit = self.get_memory_unit(unit_id)
                if unit:
                    results.append((unit, float(distances[0, i])))
                else: # 理论上不应发生，因为internal_id应该有效
                    logging.warning(f"在FAISS搜索结果中找到内部ID {internal_id}，但在内存单元字典中找不到对应的单元ID '{unit_id}'。")
            else:
                logging.warning(f"在FAISS搜索结果中找到无法映射回单元ID的内部ID {internal_id}。")
        
        return results

    def search_similarity_by_text(self, query_text: str, k: int = 5, space_name: Optional[str] = None, filter_unit_ids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """通过查询文本进行相似性搜索。"""
        query_vector = self._get_text_embedding(query_text)
        if query_vector is None:
            return []
        return self.search_similarity_by_vector(query_vector, k, space_name, filter_unit_ids)

    def search_similarity_by_image(self, image_path: str, k: int = 5, space_name: Optional[str] = None, filter_unit_ids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """通过查询图像路径进行相似性搜索。"""
        query_vector = self._get_image_embedding(image_path)
        if query_vector is None:
            return []
        return self.search_similarity_by_vector(query_vector, k, space_name, filter_unit_ids)

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

    def add_unit_to_space(self, unit_id: str, space_name: str):
        """将一个内存单元添加到一个内存空间。"""
        if unit_id not in self.memory_units:
            logging.warning(f"尝试将不存在的内存单元 '{unit_id}' 添加到空间 '{space_name}'。")
            return
        space = self.create_memory_space(space_name) # 如果空间不存在则创建
        space.add_memory_unit(unit_id)

    def remove_unit_from_space(self, unit_id: str, space_name: str):
        """从一个内存空间移除一个内存单元。"""
        space = self.get_memory_space(space_name)
        if space:
            space.remove_memory_unit(unit_id)
        else:
            logging.warning(f"尝试从不存在的内存空间 '{space_name}' 移除单元 '{unit_id}'。")

    # --- 持久化 ---
    def save_map(self, directory_path: str):
        """
        将 SemanticMap 的状态保存到指定目录。
        会保存 memory_units, memory_spaces, FAISS 索引, 以及 _unit_id_to_internal_faiss_id 映射。
        参数:
            directory_path (str): 用于保存文件的目录路径。如果不存在，将尝试创建。
        """
        os.makedirs(directory_path, exist_ok=True)
        
        # 1. 保存 MemoryUnit 和 MemorySpace 对象 (使用 pickle)
        with open(os.path.join(directory_path, "semantic_map_data.pkl"), "wb") as f:
            pickle.dump({
                "memory_units": self.memory_units,
                "memory_spaces": self.memory_spaces,
                "_unit_id_to_internal_faiss_id": self._unit_id_to_internal_faiss_id,
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
        instance._unit_id_to_internal_faiss_id = saved_state.get("_unit_id_to_internal_faiss_id", {})
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
        try:
            # 延迟导入，避免强制依赖
            from milvus_operator import MilvusOperator
        except ImportError:
            logging.error("未找到milvus_operator模块，请确保已安装pymilvus并创建了milvus_operator.py")
            return False
        
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
            for unit_id, unit in self.memory_units.items():
                # 查找单元所属的所有空间
                space_names = []
                for space_name, space in self.memory_spaces.items():
                    if unit_id in space.get_memory_unit_ids():
                        space_names.append(space_name)
                
                # 添加到Milvus
                if milvus_op.add_memory_unit(unit, space_names):
                    success_count += 1
                else:
                    logging.warning(f"导出内存单元 '{unit_id}' 到Milvus失败")
            
            logging.info(f"成功导出 {success_count}/{len(self.memory_units)} 个内存单元到Milvus")
            milvus_op.close()
            
            return success_count > 0
            
        except Exception as e:
            logging.error(f"导出到Milvus失败: {e}")
            return False

# --- 语义图谱 ---

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

    def add_memory_unit(self,
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
        self.semantic_map.add_memory_unit(
            unit,
            explicit_content_for_embedding,
            content_type_for_embedding,
            space_names,
            rebuild_index_immediately=rebuild_semantic_map_index_immediately # 注意这里传递的是否立即重建map索引
        )
        
        # 2. 将单元ID作为节点添加到 NetworkX 图中 (如果尚不存在)
        if not self.nx_graph.has_node(unit.id):
            # 可以在节点上存储来自 unit.value 的一些属性，如果需要的话
            self.nx_graph.add_node(unit.id, **unit.value) 
            logging.debug(f"节点 '{unit.id}' 已添加到 NetworkX 图。")
        else: # 如果节点已存在，可以选择更新其属性
            nx.set_node_attributes(self.nx_graph, {unit.id: unit.value})
            logging.debug(f"节点 '{unit.id}' 的属性已在 NetworkX 图中更新。")


    def add_relationship(self,
                         source_unit_id: str,
                         target_unit_id: str,
                         relationship_name: str,
                         bidirectional: bool = False,
                         **kwargs: Any): # 允许添加其他关系属性
        """
        在两个已存在的内存单元 (节点) 之间添加一条显式关系 (边)。
        参数:
            source_unit_id (str): 源节点的ID。
            target_unit_id (str): 目标节点的ID。
            relationship_name (str): 关系的名称 (例如 "连接到", "依赖于", "父子")。
            bidirectional (bool): 如果为 True，则添加一条从 target 到 source 的具有相同名称的反向关系。
            **kwargs: 任何其他要存储为边属性的键值对。
        """
        if not self.semantic_map.get_memory_unit(source_unit_id):
            logging.error(f"源节点 '{source_unit_id}' 不存在于 SemanticMap 中。无法添加关系。")
            return
        if not self.semantic_map.get_memory_unit(target_unit_id):
            logging.error(f"目标节点 '{target_unit_id}' 不存在于 SemanticMap 中。无法添加关系。")
            return
        
        # 确保节点也存在于nx_graph中 (通常 add_memory_unit 会处理)
        if not self.nx_graph.has_node(source_unit_id): self.nx_graph.add_node(source_unit_id)
        if not self.nx_graph.has_node(target_unit_id): self.nx_graph.add_node(target_unit_id)

        # 使用 relationship_name 作为边的 'type' 或 'label' 属性
        edge_attributes = {"type": relationship_name, **kwargs}
        self.nx_graph.add_edge(source_unit_id, target_unit_id, **edge_attributes)
        logging.info(f"已添加从 '{source_unit_id}' 到 '{target_unit_id}' 的关系 '{relationship_name}'。")

        if bidirectional:
            self.nx_graph.add_edge(target_unit_id, source_unit_id, **edge_attributes) # 注意：如果关系有方向性，反向关系可能需要不同名称/属性
            logging.info(f"已添加从 '{target_unit_id}' 到 '{source_unit_id}' 的双向关系 '{relationship_name}'。")

    def delete_memory_unit(self, unit_id: str, rebuild_semantic_map_index_immediately: bool = False):
        """从图谱和底层的 SemanticMap 中删除一个内存单元及其所有相关关系。"""
        # 1. 从 SemanticMap 删除
        self.semantic_map.delete_memory_unit(unit_id, rebuild_index_immediately=rebuild_semantic_map_index_immediately)
        
        # 2. 从 NetworkX 图中删除节点 (这会自动删除所有相关的边)
        if self.nx_graph.has_node(unit_id):
            self.nx_graph.remove_node(unit_id)
            logging.info(f"节点 '{unit_id}' 及其关系已从 NetworkX 图中删除。")
        else:
            logging.warning(f"尝试从 NetworkX 图中删除不存在的节点 '{unit_id}'。")

    def delete_relationship(self, source_unit_id: str, target_unit_id: str, relationship_name: Optional[str] = None):
        """
        删除两个节点之间的特定关系或所有关系。
        参数:
            source_unit_id (str): 源节点ID。
            target_unit_id (str): Target 节点ID。
            relationship_name (Optional[str]): 如果提供，则只删除具有此名称 (作为'type'属性) 的关系。
                                               否则，删除这两个节点之间的所有直接关系。
        """
        if not self.nx_graph.has_edge(source_unit_id, target_unit_id):
            logging.warning(f"节点 '{source_unit_id}' 和 '{target_unit_id}' 之间没有直接边。")
            return

        if relationship_name:
            # NetworkX DiGraph 可以有平行边，但 add_edge 通常会替换。
            # 如果允许多个同名关系，则需要更复杂的删除逻辑。
            # 假设每个 (source, target) 对之间特定类型的关系是唯一的。
            edge_data = self.nx_graph.get_edge_data(source_unit_id, target_unit_id)
            # 对于有向图，通常只有一个直接边。如果有多条边（MultiDiGraph），则需要迭代。
            if edge_data and edge_data.get("type") == relationship_name:
                self.nx_graph.remove_edge(source_unit_id, target_unit_id)
                logging.info(f"已删除从 '{source_unit_id}' 到 '{target_unit_id}' 的关系 '{relationship_name}'。")
            else:
                logging.warning(f"未找到从 '{source_unit_id}' 到 '{target_unit_id}' 的名为 '{relationship_name}' 的关系。")
        else: # 删除所有直接边
            self.nx_graph.remove_edge(source_unit_id, target_unit_id)
            logging.info(f"已删除从 '{source_unit_id}' 到 '{target_unit_id}' 的所有直接关系。")


    def get_memory_unit_data(self, unit_id: str) -> Optional[MemoryUnit]:
        """从底层的 SemanticMap 检索内存单元对象。"""
        return self.semantic_map.get_memory_unit(unit_id)

    def build_semantic_map_index(self):
        """构建底层 SemanticMap 的 FAISS 索引。"""
        self.semantic_map.build_index()

    # --- 查询API ---
    def search_similarity_in_graph(self,
                                   query_text: Optional[str] = None,
                                   query_vector: Optional[np.ndarray] = None,
                                   query_image_path: Optional[str] = None,
                                   k: int = 5,
                                   space_name: Optional[str] = None,
                                   filter_unit_ids: Optional[Set[str]] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        在图谱中执行语义相似性搜索 (委托给 SemanticMap)。
        参数:
            query_text (Optional[str]): 查询文本。
            query_vector (Optional[np.ndarray]): 查询向量。
            query_image_path (Optional[str]): 查询图像的路径。
            k (int): 返回结果数量。
            space_name (Optional[str]): 限制在 SemanticMap 中的特定内存空间内搜索。
            filter_unit_ids (Optional[Set[str]]): 进一步限制搜索范围的单元ID集合。
        返回:
            List[Tuple[MemoryUnit, float]]: (MemoryUnit, 相似度得分) 列表。
        """
        if query_vector is not None:
            return self.semantic_map.search_similarity_by_vector(query_vector, k, space_name, filter_unit_ids)
        elif query_text is not None:
            return self.semantic_map.search_similarity_by_text(query_text, k, space_name, filter_unit_ids)
        elif query_image_path is not None:
            return self.semantic_map.search_similarity_by_image(query_image_path, k, space_name, filter_unit_ids)
        else:
            logging.warning("必须提供 query_text, query_vector 或 query_image_path 之一进行相似性搜索。")
            return []

    def traverse_explicit_nodes(self,
                                unit_id: str,
                                relationship_type: Optional[str] = None, # 对应于边的 'type' 属性
                                direction: str = "successors", # "successors", "predecessors", or "all"
                                space_name: Optional[str] = None) -> List[MemoryUnit]:
        """
        遍历与给定节点通过显式关系连接的邻居节点。
        参数:
            unit_id (str): 起始节点的ID。
            relationship_type (Optional[str]): 要筛选的关系类型 (边属性 'type')。如果为 None，则不按类型筛选。
            direction (str): 遍历方向:
                             "successors" (默认): 查找 unit_id 指向的节点 (子节点/出边)。
                             "predecessors": 查找指向 unit_id 的节点 (父节点/入边)。
                             "all": 查找双向的邻居。
            space_name (Optional[str]): 如果提供，则仅返回那些也存在于 SemanticMap 中指定内存空间的邻居。
        返回:
            List[MemoryUnit]: 符合条件的邻居 MemoryUnit 对象列表。
        """
        if not self.nx_graph.has_node(unit_id):
            logging.warning(f"节点 '{unit_id}' 不在图中，无法遍历。")
            return []

        neighbor_ids: Set[str] = set()
        if direction == "successors":
            for successor in self.nx_graph.successors(unit_id):
                if relationship_type:
                    edge_data = self.nx_graph.get_edge_data(unit_id, successor)
                    # 对于有向图，通常只有一个直接边。如果是MultiDiGraph，需要检查所有边。
                    # 假设默认的DiGraph，get_edge_data返回第一个找到的边的属性。
                    # 如果一个 (u,v) 对有多条不同类型的边，这个逻辑需要调整为检查所有边。
                    # 对于简单的DiGraph，如果 (u,v) 存在，则只有一条边。
                    if edge_data and edge_data.get("type") == relationship_type:
                        neighbor_ids.add(successor)
                else:
                    neighbor_ids.add(successor)
        elif direction == "predecessors":
            for predecessor in self.nx_graph.predecessors(unit_id):
                if relationship_type:
                    edge_data = self.nx_graph.get_edge_data(predecessor, unit_id)
                    if edge_data and edge_data.get("type") == relationship_type:
                        neighbor_ids.add(predecessor)
                else:
                    neighbor_ids.add(predecessor)
        elif direction == "all":
            # 获取所有邻居 (包括前驱和后继)
            all_neighbors_temp = set(self.nx_graph.successors(unit_id))
            all_neighbors_temp.update(self.nx_graph.predecessors(unit_id))
            
            for neighbor in all_neighbors_temp:
                # 检查 (unit_id, neighbor) 或 (neighbor, unit_id) 的边
                passes_filter = False
                if not relationship_type:
                    passes_filter = True
                else:
                    if self.nx_graph.has_edge(unit_id, neighbor) and self.nx_graph.get_edge_data(unit_id, neighbor).get("type") == relationship_type:
                        passes_filter = True
                    elif self.nx_graph.has_edge(neighbor, unit_id) and self.nx_graph.get_edge_data(neighbor, unit_id).get("type") == relationship_type:
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
                space_unit_ids = space.get_memory_unit_ids()
                neighbor_ids.intersection_update(space_unit_ids) # 只保留也在空间内的ID
            else:
                logging.warning(f"内存空间 '{space_name}' 未找到，无法按空间过滤邻居。")
                return [] # 如果指定了空间但空间不存在，则不返回任何结果

        # 获取 MemoryUnit 对象
        results: List[MemoryUnit] = []
        for nid in neighbor_ids:
            unit = self.semantic_map.get_memory_unit(nid)
            if unit:
                results.append(unit)
        return results

    def traverse_implicit_nodes(self,
                                unit_id: str,
                                k: int = 5,
                                space_name: Optional[str] = None) -> List[Tuple[MemoryUnit, float]]:
        """
        基于语义相似性查找与给定节点隐式相关的节点。
        参数:
            unit_id (str): 起始节点的ID。
            k (int): 要查找的相似邻居数量。
            space_name (Optional[str]): 如果提供，则在 SemanticMap 中的指定内存空间内限制搜索。
        返回:
            List[Tuple[MemoryUnit, float]]: (MemoryUnit, 相似度得分) 列表，不包括 unit_id 本身。
        """
        start_unit = self.semantic_map.get_memory_unit(unit_id)
        if not start_unit or start_unit.vector is None:
            logging.warning(f"节点 '{unit_id}' 不存在或没有向量，无法进行隐式遍历。")
            return []

        # 搜索时排除自身 (如果 SemanticMap 的搜索结果可能包含查询项本身)
        # k+1 然后过滤，或者让 SemanticMap 的搜索处理过滤（如果它支持的话）
        # 这里假设 search_similarity_by_vector 返回的结果不包含查询向量本身（除非它在数据集中且非常相似）
        # 通常，我们会获取 k+1 个结果，然后手动排除 unit_id。
        
        # 创建一个过滤器，排除 unit_id 本身
        filter_ids_to_exclude_self = {uid for uid in self.semantic_map.memory_units.keys() if uid != unit_id}
        
        # 如果指定了空间，则 filter_ids_to_exclude_self 会被 search_similarity_by_vector 内部的 space_name 逻辑覆盖或合并。
        # 我们需要确保 unit_id 本身被排除。
        # 一个更简单的方法是获取k+1个结果，然后从结果中移除unit_id。

        similar_units_with_scores = self.semantic_map.search_similarity_by_vector(
            start_unit.vector,
            k=k + 1, # 获取稍多一些，以防 unit_id 是最相似的
            space_name=space_name
        )
        
        results: List[Tuple[MemoryUnit, float]] = []
        for unit, score in similar_units_with_scores:
            if unit.id != unit_id: # 排除起始节点本身
                results.append((unit, score))
            if len(results) >= k: # 如果已达到k个结果
                break
        return results

    # --- MemorySpace 相关 (通过 SemanticMap 操作) ---
    def create_memory_space_in_map(self, space_name: str) -> MemorySpace:
        """在底层的 SemanticMap 中创建或获取一个内存空间。"""
        return self.semantic_map.create_memory_space(space_name)

    def add_unit_to_space_in_map(self, unit_id: str, space_name: str):
        """将一个内存单元添加到 SemanticMap 中的指定内存空间。"""
        self.semantic_map.add_unit_to_space(unit_id, space_name)

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
        """
        将SemanticGraph中的节点和关系导出到Neo4j数据库
        
        参数:
            uri: Neo4j服务器URI
            user: 用户名
            password: 密码
            database: 数据库名称
        
        返回:
            导出是否成功
        """
        try:
            # 延迟导入，避免强制依赖
            from neo4j_operator import Neo4jOperator
        except ImportError:
            logging.error("未找到neo4j_operator模块，请确保已安装neo4j并创建了neo4j_operator.py")
            return False
        
        try:
            # 创建Neo4j操作类
            neo4j_op = Neo4jOperator(
                uri=uri, 
                user=user, 
                password=password,
                database=database
            )
            
            if not neo4j_op.is_connected:
                logging.error("连接Neo4j失败，无法导出数据")
                return False
            
            # 导出所有内存单元
            unit_success_count = 0
            for unit_id, unit in self.semantic_map.memory_units.items():
                if neo4j_op.add_memory_unit(unit):
                    unit_success_count += 1
                else:
                    logging.warning(f"导出内存单元 '{unit_id}' 到Neo4j失败")
            
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


if __name__ == '__main__':
    # --- 示例用法 ---
    logging.info("开始 Hippo.py 示例用法...")

    # 1. 创建 SemanticMap
    # 使用较小的模型进行快速测试 (如果可用) 或默认模型
    # text_model_for_test = "paraphrase-MiniLM-L6-v2" # 384 dim
    # image_model_for_test = "clip-ViT-B-32" # 512 dim, 但这里为了统一，假设都用一个维度
    # 如果模型维度不同，需要更复杂的处理或使用能输出相同维度的模型对
    
    # 为简单起见，假设使用默认模型和维度
    semantic_map = SemanticMap(embedding_dim=512)
    semantic_map = SemanticMap(
        image_embedding_model_name="/home/zyh/model/clip-ViT-B-32",
        text_embedding_model_name="/home/zyh/model/clip-ViT-B-32-multilingual-v1",
        embedding_dim=512
        )

    # 2. 创建和添加 MemoryUnit
    unit1_text = "这是一个关于人工智能的文档。"
    unit1_val = {"description": "AI介绍", "type": "文档", "text_content": unit1_text, "author": "系统"}
    unit1 = MemoryUnit(unit_id="doc_ai_intro", value=unit1_val)
    
    unit2_text = "机器学习是人工智能的一个分支。"
    unit2_val = {"description": "ML定义", "type": "概念", "text_content": unit2_text, "field": "AI"}
    unit2 = MemoryUnit(unit_id="concept_ml", value=unit2_val)

    unit3_text = "深度学习推动了自然语言处理的进步。"
    unit3_val = {"description": "DL对NLP的影响", "type": "观察", "text_content": unit3_text}
    unit3 = MemoryUnit(unit_id="obs_dl_nlp", value=unit3_val)
    
    # 假设我们有一个图像单元 (需要一个真实图像路径)
    # dummy_image_path = "path/to/your/dummy_image.jpg" # 替换为真实图像路径
    # if not os.path.exists(dummy_image_path):
    #     try: # 创建一个虚拟图像用于测试
    #         Image.new('RGB', (60, 30), color = 'red').save("dummy_image.jpg")
    #         dummy_image_path = "dummy_image.jpg"
    #     except ImportError:
    #         dummy_image_path = None # PIL可能未安装
    #         logging.warning("PIL/Pillow 未安装，无法创建虚拟图像。跳过图像单元示例。")

    # unit_img1 = None
    # if dummy_image_path and os.path.exists(dummy_image_path):
    #     unit_img1_val = {"description": "红色矩形图片", "type": "图像", "image_path": dummy_image_path}
    #     unit_img1 = MemoryUnit(unit_id="img_red_rect", value=unit_img1_val)
    #     semantic_map.add_memory_unit(unit_img1, space_names=["图像空间"])


    # 添加文本单元到 map，并指定内容进行嵌入
    semantic_map.add_memory_unit(unit1, explicit_content_for_embedding=unit1_text, content_type_for_embedding="text", space_names=["AI文档", "通用知识"])
    semantic_map.add_memory_unit(unit2, explicit_content_for_embedding=unit2_text, content_type_for_embedding="text", space_names=["AI概念"])
    semantic_map.add_memory_unit(unit3, space_names=["AI观察", "NLP相关"]) # 让map从unit.value推断嵌入内容

    # 3. 构建 SemanticMap 索引
    semantic_map.build_index()

    # 4. 在 SemanticMap 中进行相似性搜索
    logging.info("\n--- SemanticMap 相似性搜索 (全局) ---")
    query1 = "什么是机器学习？"
    similar_results_map = semantic_map.search_similarity_by_text(query_text=query1, k=2)
    for res_unit, score in similar_results_map:
        logging.info(f"找到单元: {res_unit.id}, 值: {res_unit.value.get('description', res_unit.id)}, 得分: {score:.4f}")

    logging.info("\n--- SemanticMap 相似性搜索 (在 'AI文档' 空间) ---")
    similar_results_map_space = semantic_map.search_similarity_by_text(query_text="AI", k=1, space_name="AI文档")
    for res_unit, score in similar_results_map_space:
        logging.info(f"找到单元: {res_unit.id}, 值: {res_unit.value.get('description', res_unit.id)}, 得分: {score:.4f}")

    # 5. 创建 SemanticGraph (使用已有的 semantic_map)
    semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)

    # 6. 在 SemanticGraph 中添加单元 (它们已在map中，这里主要是为了在图中创建节点)
    # 通常，如果单元是新的，会通过 graph.add_memory_unit 添加
    # 如果单元已在map中，我们只需确保它们作为节点存在于图中
    if not semantic_graph.nx_graph.has_node(unit1.id): semantic_graph.nx_graph.add_node(unit1.id, **unit1.value)
    if not semantic_graph.nx_graph.has_node(unit2.id): semantic_graph.nx_graph.add_node(unit2.id, **unit2.value)
    if not semantic_graph.nx_graph.has_node(unit3.id): semantic_graph.nx_graph.add_node(unit3.id, **unit3.value)
    # if unit_img1 and not semantic_graph.nx_graph.has_node(unit_img1.id): semantic_graph.nx_graph.add_node(unit_img1.id, **unit_img1.value)


    # 7. 在 SemanticGraph 中添加关系
    semantic_graph.add_relationship(unit1.id, unit2.id, relationship_name="包含主题", relevance=0.9)
    semantic_graph.add_relationship(unit2.id, unit3.id, relationship_name="相关概念", bidirectional=True)
    
    # 8. 在 SemanticGraph 中进行查询
    logging.info("\n--- SemanticGraph 显式遍历 ('包含主题' 的子节点) ---")
    explicit_neighbors = semantic_graph.traverse_explicit_nodes(unit_id=unit1.id, relationship_type="包含主题", direction="successors")
    for neighbor_unit in explicit_neighbors:
        logging.info(f"'{unit1.id}' 的 '{'包含主题'}' 子节点: {neighbor_unit.id} - {neighbor_unit.value.get('description')}")

    logging.info("\n--- SemanticGraph 隐式遍历 (与 unit1 语义相似的节点) ---")
    implicit_neighbors = semantic_graph.traverse_implicit_nodes(unit_id=unit1.id, k=2)
    for neighbor_unit, score in implicit_neighbors:
        logging.info(f"与 '{unit1.id}' 语义相似的节点: {neighbor_unit.id} - {neighbor_unit.value.get('description')}, 得分: {score:.4f}")

    logging.info("\n--- SemanticGraph 隐式遍历 (与 unit1 语义相似的节点, 在 'AI概念' 空间) ---")
    implicit_neighbors_space = semantic_graph.traverse_implicit_nodes(unit_id=unit1.id, k=1, space_name="AI概念")
    for neighbor_unit, score in implicit_neighbors_space:
        logging.info(f"与 '{unit1.id}' 语义相似 (在 'AI概念' 空间): {neighbor_unit.id} - {neighbor_unit.value.get('description')}, 得分: {score:.4f}")

    # 9. 显示图谱摘要
    semantic_graph.display_graph_summary()

    # 示例1: 将SemanticMap导出到Milvus
    semantic_map.export_to_milvus(
        host="localhost",
        port="19530",
        collection_name="my_memory_units"
    )

    # 示例2: 将SemanticGraph导出到Neo4j
    semantic_graph.export_to_neo4j(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="20031117",
        database="academicgraph"
    )

    # 10. 持久化和加载 (示例)
    save_dir = "hippo_save_data"
    logging.info(f"\n--- 保存 SemanticGraph 到 '{save_dir}' ---")
    semantic_graph.save_graph(save_dir)
    
    logging.info(f"\n--- 从 '{save_dir}' 加载 SemanticGraph ---")
    try:
        loaded_graph = SemanticGraph.load_graph(save_dir)
        loaded_graph.display_graph_summary()
        
        # 测试加载后的图谱
        logging.info("\n--- 测试加载后的图谱: 隐式遍历 (与 unit1 语义相似的节点) ---")
        loaded_implicit_neighbors = loaded_graph.traverse_implicit_nodes(unit_id=unit1.id, k=1)
        for neighbor_unit, score in loaded_implicit_neighbors:
            logging.info(f"与 '{unit1.id}' 语义相似的节点: {neighbor_unit.id} - {neighbor_unit.value.get('description')}, 得分: {score:.4f}")

    except FileNotFoundError as e:
        logging.error(f"加载失败，因为保存目录或文件未完全创建/找到: {e}")
    except Exception as e:
        logging.error(f"加载或测试加载的图谱时发生错误: {e}", exc_info=True)


    # 清理虚拟图像
    # if dummy_image_path and dummy_image_path == "dummy_image.jpg" and os.path.exists("dummy_image.jpg"):
    #     os.remove("dummy_image.jpg")

    logging.info("\nHippo.py 示例用法结束。")

