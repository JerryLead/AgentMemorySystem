import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image

from .memory_unit import MemoryUnit
from .memory_space import MemorySpace
from .milvus_operator import MilvusOperator

class SemanticMap:
    """
    语义地图 (SemanticMap) 负责存储内存单元及其向量嵌入，并支持基于相似度的搜索。
    它还管理记忆空间，允许在特定上下文中进行操作。
    类似于一个向量数据库。
    """

    DEFAULT_TEXT_EMBEDDING_KEY = (
        "text_content"  # MemoryUnit.raw_data 中用于文本嵌入的默认键
    )
    DEFAULT_IMAGE_EMBEDDING_KEY = (
        "image_path"  # MemoryUnit.raw_data 中用于图像嵌入的默认键
    )

    def __init__(
        self,
        image_embedding_model_name: str = "clip-ViT-B-32",
        text_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        embedding_dim: int = 512,
        faiss_index_type: str = "IDMap,Flat",
    ):  # 使用IDMap来支持按ID删除
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

        self.memory_units: Dict[str, MemoryUnit] = {}  # 存储 MemoryUnit 对象，键为 uid
        self.memory_spaces: Dict[str, MemorySpace] = (
            {}
        )  # 存储 MemorySpace 对象，键为 space_name

        self.faiss_index: Optional[faiss.Index] = None  # FAISS 索引

        # FAISS ID 到 MemoryUnit ID 的映射 (以及反向映射)
        self._uid_to_internal_faiss_id: Dict[str, int] = {}
        self._internal_faiss_id_counter: int = 0  # 用于生成唯一的内部 FAISS ID

        # 添加以下跟踪变量
        self._modified_units = set()  # 被修改的单元ID集合
        self._deleted_units = set()  # 已删除但尚未同步到磁盘的单元ID
        self._last_sync_time = None  # 上次与外部存储同步的时间
        self._external_storage: Optional[MilvusOperator] = (
            None  # 指向外部存储的连接（Milvus）
        )
        self._max_memory_units = 10000  # 内存中最大单元数，超过则触发换页
        self._access_counts = {}  # 记录每个单元的访问次数，用于LRU策略

        try:
            # 初始化图像和文本嵌入模型
            self.image_model = SentenceTransformer(image_embedding_model_name)
            self.text_model = SentenceTransformer(text_embedding_model_name)
            logging.info(
                f"已初始化嵌入模型：图像模型='{image_embedding_model_name}'，文本模型='{text_embedding_model_name}'"
            )
        except Exception as e:
            logging.error(f"无法初始化嵌入模型: {e}")
            # 使用更简单的默认模型作为回退
            self.image_model = None
            self.text_model = None

        self._init_faiss_index()
        logging.info(
            f"SemanticMap 已初始化。嵌入维度: {self.embedding_dim}, FAISS索引类型: '{self.faiss_index_type}'。"
        )

    # TODO: 索引构建或者初始化时，同时应该为SemanticMap中所有MemorySpace进行递归索引构建/初始化。
    def _init_faiss_index(self):
        """初始化或重新初始化 FAISS 索引。"""
        try:
            if self.faiss_index_type.startswith("IDMap,"):
                if not hasattr(faiss, "IndexIDMap"):
                    raise ImportError("FAISS 版本不支持 IndexIDMap")
                base_index_type = self.faiss_index_type.split("IDMap,", 1)[1]
                base_index = faiss.index_factory(self.embedding_dim, base_index_type)
                self.faiss_index = faiss.IndexIDMap(base_index)
            else:
                self.faiss_index = faiss.index_factory(
                    self.embedding_dim, self.faiss_index_type
                )

            logging.info(
                f"FAISS 索引 '{self.faiss_index_type}' 已初始化。总向量数: {self.faiss_index.ntotal if self.faiss_index else 0}"
            )
        except Exception as e:
            logging.error(f"初始化 FAISS 索引 '{self.faiss_index_type}' 失败: {e}")
            self.faiss_index = None  # 确保索引在失败时为None
            raise

    ### 嵌入生成函数

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """为给定文本生成嵌入向量。"""
        if not text or not isinstance(text, str):
            logging.warning("尝试为无效文本生成嵌入。")
            return None
        try:
            if self.text_model is None:
                logging.error("文本嵌入模型未初始化")
                return None
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
            if self.image_model is None:
                logging.error("图像嵌入模型未初始化")
                return None
            img = Image.open(image_path)
            emb = self.image_model.encode(img)
            return emb.astype(np.float32)
        except Exception as e:
            logging.error(f"生成图像嵌入失败: '{image_path}' - {e}")
            return None

    def _generate_embedding_for_unit(
        self,
        unit: MemoryUnit,
        explicit_content: Optional[Any] = None,
        content_type: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        为内存单元生成嵌入向量。
        """
        embedding = None
        if explicit_content and content_type:
            if content_type == "text":
                embedding = self._get_text_embedding(str(explicit_content))
            elif content_type == "image_path":
                embedding = self._get_image_embedding(str(explicit_content))
            else:
                logging.warning(f"未知的内容类型 '{content_type}'。使用默认推断方法。")
        else:  # 尝试从 unit.raw_data 推断
            image_path = unit.raw_data.get(self.DEFAULT_IMAGE_EMBEDDING_KEY)
            text_content = unit.raw_data.get(self.DEFAULT_TEXT_EMBEDDING_KEY)

            if image_path and isinstance(image_path, str):
                embedding = self._get_image_embedding(image_path)

            if embedding is None and text_content and isinstance(text_content, str):
                embedding = self._get_text_embedding(text_content)

            if embedding is None:
                logging.debug(
                    f"为单元 '{unit.uid}' 无法从raw_data生成嵌入。没有找到有效的文本或图像内容。"
                )

        if embedding is not None and embedding.shape[0] != self.embedding_dim:
            logging.error(
                f"为单元 '{unit.uid}' 生成的嵌入维度 ({embedding.shape[0]}) 与期望维度 ({self.embedding_dim}) 不符。"
            )
            return None
        return embedding

    # ==============================
    # 新的MemorySpace管理方法（兼容新架构）
    # ==============================

    def create_memory_space(self, space_name: str) -> MemorySpace:
        """创建或获取一个记忆空间"""
        if space_name not in self.memory_spaces:
            space = MemorySpace(space_name)
            space._set_semantic_map_ref(self)  # 设置引用
            self.memory_spaces[space_name] = space
            logging.info(f"记忆空间 '{space_name}' 已创建")
        return self.memory_spaces[space_name]

    def get_memory_space(self, space_name: str) -> Optional[MemorySpace]:
        """通过名称获取记忆空间"""
        return self.memory_spaces.get(space_name)

    def add_unit_to_space(self, unit_or_uid: Union[str, MemoryUnit], space_name: str):
        """
        智能将内存单元添加到记忆空间：自动判断输入是UID字符串还是MemoryUnit对象
        """
        if isinstance(unit_or_uid, str):
            uid = unit_or_uid
        elif hasattr(unit_or_uid, "uid"):
            uid = unit_or_uid.uid
        else:
            raise TypeError(
                f"add_unit_to_space() 期望 str(UID) 或 MemoryUnit 对象，得到 {type(unit_or_uid)}"
            )

        if uid not in self.memory_units:
            logging.warning(f"尝试将不存在的内存单元 '{uid}' 添加到空间 '{space_name}'")
            return

        space = self.create_memory_space(space_name)  # 如果空间不存在则创建
        space.add_unit(uid)  # 使用智能方法

    def remove_unit_from_space(
        self, unit_or_uid: Union[str, MemoryUnit], space_name: str
    ):
        """
        智能从记忆空间移除内存单元：自动判断输入类型
        """
        if isinstance(unit_or_uid, str):
            uid = unit_or_uid
        elif hasattr(unit_or_uid, "uid"):
            uid = unit_or_uid.uid
        else:
            raise TypeError(
                f"remove_unit_from_space() 期望 str(UID) 或 MemoryUnit 对象，得到 {type(unit_or_uid)}"
            )

        space = self.get_memory_space(space_name)
        if space:
            space.remove_unit(uid)  # 使用智能方法
        else:
            logging.warning(f"尝试从不存在的记忆空间 '{space_name}' 移除单元 '{uid}'")

    def add_space_to_space(
        self, child_space_or_name: Union[str, MemorySpace], parent_space_name: str
    ):
        """
        智能将子空间添加到父空间：自动判断输入类型
        """
        if isinstance(child_space_or_name, str):
            child_space_name = child_space_or_name
        elif hasattr(child_space_or_name, "name"):
            child_space_name = child_space_or_name.name
        else:
            raise TypeError(
                f"add_space_to_space() 期望 str(名称) 或 MemorySpace 对象，得到 {type(child_space_or_name)}"
            )

        if child_space_name not in self.memory_spaces:
            logging.warning(f"子空间 '{child_space_name}' 不存在")
            return

        parent_space = self.create_memory_space(parent_space_name)
        parent_space.add_child_space(child_space_name)  # 使用智能方法

    def get_units_in_memory_space(
        self, ms_names: Union[str, List[str]], recursive: bool = True
    ) -> List[MemoryUnit]:
        """
        获取指定ms_names下所有unit，支持递归，支持多ms，自动去重
        """
        # 确保ms_names是列表
        if isinstance(ms_names, str):
            ms_names = [ms_names]

        all_uids = set()

        for space_name in ms_names:
            space = self.memory_spaces.get(space_name)
            if space:
                if recursive:
                    all_uids.update(space.get_all_unit_uids(recursive=True))
                else:
                    all_uids.update(space.get_unit_uids())
            else:
                logging.warning(f"记忆空间 '{space_name}' 不存在")

        # 转换UID为MemoryUnit对象
        units = []
        for uid in all_uids:
            unit = self.memory_units.get(uid)
            if unit:
                units.append(unit)
            else:
                logging.warning(f"MemoryUnit '{uid}' 在内存中不存在")

        return units

    # ==============================
    # 数据库操作函数（保持兼容）
    # ==============================

    def connect_external_storage(
        self,
        storage_type: str = "milvus",
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        collection_name: str = "hippo_memory_units",
    ):
        """
        连接到外部存储系统
        """
        if storage_type.lower() == "milvus":
            try:
                self._external_storage = MilvusOperator(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    collection_name=collection_name,
                    embedding_dim=self.embedding_dim,
                )
                if self._external_storage.is_connected:
                    logging.info(f"成功连接到外部存储 (Milvus): {host}:{port}")
                    return True
                else:
                    logging.error(f"连接到外部存储失败")
                    self._external_storage = None
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
                if space.contains_unit(uid):
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
        智能同步数据到外部存储
        """
        if not self._external_storage:
            logging.error("未连接到外部存储，无法同步")
            return 0, 0

        success_count = 0
        fail_count = 0

        # 确定同步模式
        sync_mode = "full" if force_full_sync else "incremental"

        # 自动检测首次同步
        if not force_full_sync and (
            not hasattr(self, "_last_sync_time") or self._last_sync_time is None
        ):
            # 检查外部存储是否为空来判断是否为首次同步
            try:
                # 尝试获取少量数据来检测是否为空
                sample_units = self._external_storage.get_units_batch([])
                if len(sample_units) == 0:
                    force_full_sync = True
                    sync_mode = "auto_full"
                    logging.info("检测到首次同步，将执行全量同步")
            except Exception as e:
                logging.warning(f"检测外部存储状态失败: {e}，使用增量同步")

        if force_full_sync:
            logging.info(f"开始全量同步到外部存储... (模式: {sync_mode})")
        else:
            logging.info("开始增量同步到外部存储...")

        # 处理修改过的单元
        units_to_sync = (
            list(self.memory_units.keys())
            if force_full_sync
            else list(self._modified_units)
        )

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

        # 更新同步状态
        if success_count > 0:
            self._last_sync_time = datetime.now()
            # 如果是全量同步，清理所有修改标记
            if force_full_sync:
                self._modified_units.clear()
                self._deleted_units.clear()

        logging.info(
            f"同步完成 ({sync_mode})。成功: {success_count}, 失败: {fail_count}"
        )
        return success_count, fail_count

    # 保留原有方法作为便利函数
    def incremental_sync(self):
        """增量同步修改过的单元到外部存储"""
        return self.sync_to_external(force_full_sync=False)

    def full_sync(self):
        """完整同步所有单元到外部存储"""
        return self.sync_to_external(force_full_sync=True)

    def load_from_external(
        self,
        filter_space: Optional[str] = None,
        limit: int = 1000,
        replace_existing: bool = False,
    ):
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
                logging.warning(
                    "从外部存储加载所有单元可能消耗大量内存，建议指定空间过滤"
                )
                return 0

            load_count = 0
            # 将加载的单元添加到内存中
            for unit in units:
                if unit.uid not in self.memory_units or replace_existing:
                    # 检查内存限制
                    if len(self.memory_units) >= self._max_memory_units:
                        self.swap_out(count=int(self._max_memory_units * 0.1))

                    self.memory_units[unit.uid] = unit
                    self._access_counts[unit.uid] = 0  # 初始化访问计数
                    load_count += 1

                    # 处理单元的空间归属
                    if hasattr(unit, "metadata") and isinstance(unit.metadata, dict):
                        space_info = unit.metadata.get("spaces", [])
                        for space_name in space_info:
                            self.add_unit_to_space(unit.uid, space_name)

            # 如果加载了单元，重建索引
            if load_count > 0:
                logging.info(f"从外部存储加载了 {load_count} 个单元")
                self.build_index()

            return load_count

        except Exception as e:
            logging.error(f"从外部存储加载单元失败: {e}")
            return 0

    def swap_out(
        self,
        count: int = 100,
        strategy: str = "LRU",
        query_context: Optional[str] = None,
    ):
        """
        将最少使用的单元从内存移出到外部存储
        """
        if not self._external_storage:
            logging.warning("没有配置外部存储，无法执行换页操作")
            return

        if len(self.memory_units) == 0:
            logging.debug("内存中没有单元可以换出")
            return

        # 根据策略确定要换出的单元
        if strategy == "LRU":
            # 按最后访问时间排序，最少最近使用的优先换出
            sorted_units = sorted(
                [
                    (uid, getattr(self, "_last_accessed", {}).get(uid, 0))
                    for uid in self.memory_units.keys()
                ],
                key=lambda x: x[1],  # 按最后访问时间排序
            )
        elif strategy == "LFU":
            # 按访问频率排序，访问次数最少的优先换出
            sorted_units = sorted(
                [
                    (uid, self._access_counts.get(uid, 0))
                    for uid in self.memory_units.keys()
                ],
                key=lambda x: x[1],  # 按访问次数排序
            )
        else:  # 默认使用LRU
            logging.warning(f"不支持的换出策略: {strategy}，使用LRU")
            sorted_units = sorted(
                [
                    (uid, getattr(self, "_last_accessed", {}).get(uid, 0))
                    for uid in self.memory_units.keys()
                ],
                key=lambda x: x[1],
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
                if space.contains_unit(uid):
                    space_names.append(space_name)

            # 同步到外部存储
            if self._external_storage.add_unit(unit, space_names):
                synced_count += 1
                self._modified_units.discard(uid)

            # 从FAISS索引中移除
            if uid in self._uid_to_internal_faiss_id and self.faiss_index:
                try:
                    internal_id = self._uid_to_internal_faiss_id[uid]
                    if hasattr(self.faiss_index, "remove_ids"):
                        self.faiss_index.remove_ids(
                            np.array([internal_id], dtype=np.int64)
                        )
                    del self._uid_to_internal_faiss_id[uid]
                except Exception as e:
                    logging.error(f"从FAISS索引移除单元 '{uid}' 失败: {e}")

            # 从内存中移除
            if uid in self.memory_units:
                del self.memory_units[uid]
                removed_count += 1

            # 清理访问计数和时间记录
            self._access_counts.pop(uid, None)
            if hasattr(self, "_last_accessed"):
                self._last_accessed.pop(uid, None)

            logging.debug(f"单元 '{uid}' 已从内存换出（策略: {strategy}）")

        logging.info(
            f"使用{strategy}策略已将 {removed_count} 个单元从内存换出，其中 {synced_count} 个已同步到外部存储"
        )

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

    # ==============================
    # 内存单元操作函数（更新兼容性）
    # ==============================

    def add_unit(
        self,
        unit: MemoryUnit,
        explicit_content_for_embedding: Optional[Any] = None,
        content_type_for_embedding: Optional[str] = None,
        space_names: Optional[List[str]] = None,
        rebuild_index_immediately: bool = False,
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
                explicit_content_for_embedding = unit.raw_data[
                    self.DEFAULT_TEXT_EMBEDDING_KEY
                ]
                content_type_for_embedding = "text"
            elif self.DEFAULT_IMAGE_EMBEDDING_KEY in unit.raw_data:
                explicit_content_for_embedding = unit.raw_data[
                    self.DEFAULT_IMAGE_EMBEDDING_KEY
                ]
                content_type_for_embedding = "image_path"
            else:
                # 如果没有标准键，使用整个raw_data的字符串表示
                explicit_content_for_embedding = str(unit.raw_data)
                content_type_for_embedding = "text"

        # 生成或更新嵌入
        new_embedding = self._generate_embedding_for_unit(
            unit, explicit_content_for_embedding, content_type_for_embedding
        )

        if new_embedding is None:
            logging.warning(
                f"未能为内存单元 '{unit.uid}' 生成嵌入。该单元将被添加，但无法用于相似性搜索。"
            )
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
        """从语义地图中删除一个内存单元"""
        if uid in self.memory_units:
            del self.memory_units[uid]

            # 从所有记忆空间中移除此单元的引用
            for space_name, space_obj in self.memory_spaces.items():
                if space_obj.contains_unit(uid):
                    space_obj.remove_unit(uid)

            # 从FAISS索引中移除
            if self.faiss_index and uid in self._uid_to_internal_faiss_id:
                internal_id_to_remove = self._uid_to_internal_faiss_id[uid]
                try:
                    if hasattr(self.faiss_index, "remove_ids"):
                        self.faiss_index.remove_ids(
                            np.array([internal_id_to_remove], dtype=np.int64)
                        )
                        logging.debug(f"内存单元 '{uid}' 已从FAISS索引中移除")
                    del self._uid_to_internal_faiss_id[uid]
                except Exception as e:
                    logging.error(f"从FAISS索引中移除单元 '{uid}' 失败: {e}")

            logging.info(f"内存单元 '{uid}' 已从 SemanticMap 删除")
            if rebuild_index_immediately:
                self.build_index()
        else:
            logging.warning(f"尝试删除不存在的内存单元ID '{uid}'")

        # 添加到删除跟踪
        self._deleted_units.add(uid)
        if uid in self._modified_units:
            self._modified_units.remove(uid)
        if uid in self._access_counts:
            del self._access_counts[uid]

    def get_all_units(self) -> List[MemoryUnit]:
        """获取所有MemoryUnit对象"""
        return list(self.memory_units.values())

    # ==============================
    # 与Hippo保持一致的方法
    # ==============================

    def get_all_memory_space_names(self) -> List[str]:
        """
        获取所有MemorySpace名称（递归，去重）。
        """
        result = set()

        def _collect_names(space: MemorySpace):
            result.add(space.name)
            for child_space_name in space.get_child_space_names():
                child_space = self.memory_spaces.get(child_space_name)
                if child_space:
                    _collect_names(child_space)

        for space in self.memory_spaces.values():
            _collect_names(space)
        return list(result)

    def get_memory_space_structures(self) -> List[dict]:
        """
        递归导出所有MemorySpace嵌套结构（树/嵌套dict），
        每个ms展示：名称、unit uid列表、所有unit的raw_data字段全集、子空间。
        返回列表，每个元素为一个ms的结构。
        """

        def _struct(space: MemorySpace):
            # 收集本space下所有unit的uid和raw_data字段全集
            unit_uids = list(space.get_unit_uids())
            unit_fields = set()
            for uid in unit_uids:
                unit = self.memory_units.get(uid)
                if unit:
                    unit_fields.update(unit.raw_data.keys())

            # 获取子空间
            child_space_names = list(space.get_child_space_names())
            children = []
            for child_name in child_space_names:
                child_space = self.memory_spaces.get(child_name)
                if child_space:
                    children.append(_struct(child_space))

            d = {
                "name": space.name,
                "unit_uids": unit_uids,
                "unit_fields": sorted(list(unit_fields)),
            }
            if children:
                d["children"] = children
            return d

        # 找到根空间（没有被其他空间包含的空间）
        all_child_names = set()
        for space in self.memory_spaces.values():
            all_child_names.update(space.get_child_space_names())

        root_spaces = [
            space
            for space in self.memory_spaces.values()
            if space.name not in all_child_names
        ]

        return [_struct(space) for space in root_spaces]

    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        """去重单元列表"""
        seen = set()
        result = []
        for u in units:
            if u.uid not in seen:
                seen.add(u.uid)
                result.append(u)
        return result

    def units_union(self, *args) -> List[MemoryUnit]:
        """
        支持多个MemorySpace、MemoryUnit列表、UID列表的并集，返回去重后的MemoryUnit列表。
        """
        seen = set()
        result = []
        for arg in args:
            units = self._expand_to_units(arg)
            for u in units:
                if u.uid not in seen:
                    seen.add(u.uid)
                    result.append(u)
        return result

    def units_intersection(self, *args) -> List[MemoryUnit]:
        """
        支持多个MemorySpace、MemoryUnit列表、UID列表的交集，返回去重后的MemoryUnit列表。
        """
        if not args:
            return []
        sets = [set(u.uid for u in self._expand_to_units(arg)) for arg in args]
        common_uids = set.intersection(*sets) if sets else set()
        # 取第一个参数的unit对象（可选：可合并属性）
        first_units = {u.uid: u for u in self._expand_to_units(args[0])}
        return [first_units[uid] for uid in common_uids if uid in first_units]

    def units_difference(self, arg1, arg2) -> List[MemoryUnit]:
        """
        返回arg1中有而arg2中没有的unit（按uid）。支持MemorySpace、MemoryUnit列表、UID列表。
        """
        uids2 = set(u.uid for u in self._expand_to_units(arg2))
        return [u for u in self._expand_to_units(arg1) if u.uid not in uids2]

    def _expand_to_units(self, obj) -> List[MemoryUnit]:
        """
        辅助方法：将MemorySpace、MemoryUnit列表、UID列表等统一展开为MemoryUnit列表。
        """
        result = []
        if obj is None:
            return result
        if isinstance(obj, MemoryUnit):
            result.append(obj)
        elif isinstance(obj, str):
            # 视为UID
            u = self.memory_units.get(obj)
            if u:
                result.append(u)
        elif hasattr(obj, "get_all_units"):
            # MemorySpace
            result.extend(obj.get_all_units())
        elif isinstance(obj, (list, set, tuple)):
            for item in obj:
                result.extend(self._expand_to_units(item))
        return result

    # ==============================
    # FAISS索引构建和搜索（保持原有功能）
    # ==============================

    def build_index(self):
        """
        根据当前所有具有有效嵌入的内存单元构建（或重建）FAISS索引。
        并递归为所有MemorySpace及其子空间建立索引。
        """
        if not self.memory_units:
            logging.info("没有内存单元可用于构建索引。")
            if self.faiss_index:
                self.faiss_index.reset()  # 清空现有索引
            self._uid_to_internal_faiss_id.clear()
            self._internal_faiss_id_counter = 0
            return

        valid_embeddings: List[np.ndarray] = []
        internal_faiss_ids_for_index: List[int] = []

        # 重置映射和计数器，因为我们要重建
        self._uid_to_internal_faiss_id.clear()

        current_internal_id = 0
        for uid, unit in self.memory_units.items():
            if (
                unit.embedding is not None
                and unit.embedding.shape[0] == self.embedding_dim
            ):
                valid_embeddings.append(unit.embedding)
                # 分配一个新的内部FAISS ID
                self._uid_to_internal_faiss_id[uid] = current_internal_id
                internal_faiss_ids_for_index.append(current_internal_id)
                current_internal_id += 1
            else:
                logging.debug(
                    f"内存单元 '{uid}' 没有有效向量，将不被包含在FAISS索引中。"
                )

        if not valid_embeddings:
            logging.info("没有有效的嵌入可用于构建FAISS索引。")
            if self.faiss_index:
                self.faiss_index.reset()
            return

        embeddings_np = np.array(valid_embeddings).astype(np.float32)
        ids_np = np.array(internal_faiss_ids_for_index, dtype=np.int64)

        # 重新初始化FAISS索引以确保它是干净的
        self._init_faiss_index()
        if not self.faiss_index:  # 如果初始化失败
            logging.error("FAISS索引未初始化，无法构建。")
            return

        # 如果索引类型需要训练 (例如 IVF 系列)
        if "IVF" in self.faiss_index_type and not self.faiss_index.is_trained:
            logging.info(f"正在训练FAISS索引 ('{self.faiss_index_type}')...")
            if embeddings_np.shape[0] < getattr(self.faiss_index, "nlist", 1):
                logging.warning(
                    f"训练数据太少 ({embeddings_np.shape[0]} 个向量) 对于 IVF 索引。可能导致错误或性能不佳。"
                )
            if embeddings_np.shape[0] > 0:
                self.faiss_index.train(embeddings_np)
                logging.info("FAISS索引训练完成。")
            else:
                logging.error("没有数据用于训练FAISS索引。")
                return

        self.faiss_index.add_with_ids(embeddings_np, ids_np)
        logging.info(
            f"FAISS索引已成功构建/重建。包含 {self.faiss_index.ntotal} 个向量。"
        )
        # 更新内部ID计数器
        self._internal_faiss_id_counter = current_internal_id

        # 递归为所有MemorySpace及其子空间建立索引
        def _recursive_build_index(space: MemorySpace):
            space.build_index(embedding_dim=self.embedding_dim)
            try:
                semantic_map = space._get_semantic_map()
                for child_space_name in space.get_child_space_names():
                    child_space = semantic_map.memory_spaces.get(child_space_name)
                    if child_space:
                        _recursive_build_index(child_space)
            except Exception as e:
                logging.warning(f"递归为MemorySpace '{space.name}' 建立索引时出错: {e}")

        for space in self.memory_spaces.values():
            _recursive_build_index(space)

    def search_similarity_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        candidate_uids: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        通过查询向量在语义地图中搜索相似的内存单元。

        参数:
            query_embedding (np.ndarray): 查询向量
            k (int): 返回的相似单元数量
            ms_names (Optional[List[str]]): 指定搜索的记忆空间名称列表，支持多个空间
            candidate_uids (Optional[List[str]]): 候选单元UID列表，限制搜索范围

        返回:
            List[Tuple[MemoryUnit, float]]: 相似单元及其相似度分数的列表
        """
        # 输入验证
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logging.warning("FAISS索引未构建或为空。无法执行搜索。")
            return []
        if query_embedding is None or query_embedding.shape[0] != self.embedding_dim:
            logging.error(f"查询向量无效或维度不匹配 (期望 {self.embedding_dim})。")
            return []
        if k <= 0:
            logging.warning("搜索数量k必须大于0")
            return []

        query_embedding_np = query_embedding.reshape(1, -1).astype(np.float32)

        # 确定搜索策略和候选单元
        search_strategy = self._determine_search_strategy(ms_names, candidate_uids)
        candidate_uids_set = self._get_candidate_uids_set(ms_names, candidate_uids)

        if not candidate_uids_set:
            logging.info("没有候选单元可供搜索。")
            return []

        logging.debug(
            f"搜索策略: {search_strategy}, 候选单元数: {len(candidate_uids_set)}"
        )

        # 根据搜索策略执行搜索
        if search_strategy == "global_index":
            return self._search_with_global_index(
                query_embedding_np, k, candidate_uids_set
            )
        elif search_strategy == "single_space_index":
            return self._search_with_single_space_index(
                query_embedding_np, k, ms_names[0]
            )
        else:  # "brute_force"
            return self._search_with_brute_force(
                query_embedding_np, k, candidate_uids_set
            )

    def _determine_search_strategy(
        self, ms_names: Optional[List[str]], candidate_uids: Optional[List[str]]
    ) -> str:
        """
        确定搜索策略：
        - global_index: 使用全局索引（无空间限制或单空间但无局部索引）
        - single_space_index: 使用单个空间的局部索引
        - brute_force: 暴力搜索（多空间或指定候选单元）
        """
        # 如果指定了候选单元，使用暴力搜索
        if candidate_uids is not None:
            return "brute_force"

        # 如果没有指定空间，使用全局索引
        if not ms_names:
            return "global_index"

        # 如果只指定了一个空间，尝试使用局部索引
        if len(ms_names) == 1:
            space_name = ms_names[0]
            space = self.get_memory_space(space_name)
            if space and hasattr(space, "_emb_index") and space._emb_index is not None:
                return "single_space_index"
            else:
                return "global_index"

        # 多个空间，使用暴力搜索
        return "brute_force"

    def _get_candidate_uids_set(
        self, ms_names: Optional[List[str]], candidate_uids: Optional[List[str]]
    ) -> Set[str]:
        """
        获取候选单元UID集合，支持空间、UID列表的并集/交集，利用集合操作接口简化逻辑。
        """
        # 同时指定ms_names和candidate_uids，取交集
        if ms_names and candidate_uids:
            units = self.units_intersection(ms_names, candidate_uids)
            return set(u.uid for u in units)
        # 只指定ms_names，取并集
        elif ms_names:
            units = self.units_union(*ms_names)
            return set(u.uid for u in units)
        # 只指定candidate_uids
        elif candidate_uids:
            units = self.units_union(candidate_uids)
            return set(u.uid for u in units)
        # 都未指定，返回全体
        else:
            return set(self.memory_units.keys())

    def _search_with_global_index(
        self, query_embedding_np: np.ndarray, k: int, candidate_uids_set: Set[str]
    ) -> List[Tuple[MemoryUnit, float]]:
        """使用全局索引进行搜索"""
        logging.debug("使用全局FAISS索引进行搜索")

        # 构建候选单元的内部ID映射
        target_internal_faiss_ids = [
            self._uid_to_internal_faiss_id[uid]
            for uid in candidate_uids_set
            if uid in self._uid_to_internal_faiss_id
        ]

        if not target_internal_faiss_ids:
            logging.info("候选单元在FAISS索引中没有对应的内部ID。")
            return []

        # 使用ID选择器进行过滤搜索
        try:
            import faiss

            final_target_internal_ids_np = np.array(
                target_internal_faiss_ids, dtype=np.int64
            )
            id_selector = faiss.IDSelectorArray(final_target_internal_ids_np)
            search_params = (
                faiss.SearchParametersIVF()
                if "IVF" in self.faiss_index_type
                else faiss.SearchParameters()
            )
            search_params.sel = id_selector
            k = min(k, len(final_target_internal_ids_np))

            distances, internal_faiss_indices = self.faiss_index.search(
                query_embedding_np, k, params=search_params
            )

        except RuntimeError as e:
            if "search params not supported for this index" in str(e):
                logging.warning(
                    f"当前FAISS索引类型 '{self.faiss_index_type}' 不支持搜索参数。执行不带过滤器的搜索。"
                )
                distances, internal_faiss_indices = self.faiss_index.search(
                    query_embedding_np, k
                )
            else:
                raise

        return self._process_search_results(
            distances, internal_faiss_indices, candidate_uids_set
        )

    def _search_with_single_space_index(
        self, query_embedding_np: np.ndarray, k: int, space_name: str
    ) -> List[Tuple[MemoryUnit, float]]:
        """使用单个空间的局部索引进行搜索"""
        logging.debug(f"使用记忆空间 '{space_name}' 的局部索引进行搜索")

        space = self.get_memory_space(space_name)
        if not space or not hasattr(space, "_emb_index") or space._emb_index is None:
            logging.warning(f"记忆空间 '{space_name}' 没有局部索引，回退到全局索引")
            return self._search_with_global_index(
                query_embedding_np, k, space.get_all_unit_uids(recursive=True)
            )

        # 使用空间的局部索引搜索
        try:
            distances, indices = space._emb_index.search(query_embedding_np, k)
            results = []

            for i in range(len(indices[0])):
                idx = int(indices[0][i])
                if indices[0][i] == -1:
                    continue

                unit = space._index_to_uid.get(idx)
                if unit:
                    memory_unit = self.get_unit(unit)
                    if memory_unit:
                        results.append((memory_unit, float(distances[0][i])))

            return results

        except Exception as e:
            logging.error(f"使用局部索引搜索失败: {e}，回退到全局索引")
            return self._search_with_global_index(
                query_embedding_np, k, space.get_all_unit_uids(recursive=True)
            )

    def _search_with_brute_force(
        self, query_embedding_np: np.ndarray, k: int, candidate_uids_set: Set[str]
    ) -> List[Tuple[MemoryUnit, float]]:
        """使用暴力搜索（适用于多空间或指定候选单元）"""
        logging.debug("使用暴力搜索（多空间或指定候选单元）")

        # 获取候选单元
        candidate_units = []
        for uid in candidate_uids_set:
            unit = self.get_unit(uid)
            if unit and unit.embedding is not None:
                candidate_units.append(unit)

        if not candidate_units:
            logging.info("没有有效的候选单元可供搜索。")
            return []

        # 构建向量数组
        embeddings = np.array(
            [unit.embedding for unit in candidate_units], dtype=np.float32
        )

        # 使用FAISS进行暴力搜索
        try:
            import faiss

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            distances, indices = index.search(
                query_embedding_np, min(k, len(candidate_units))
            )

            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])
                if indices[0][i] == -1:
                    continue
                results.append((candidate_units[idx], float(distances[0][i])))

            return results

        except ImportError:
            logging.error("FAISS不可用，无法进行暴力搜索")
            return []
        except Exception as e:
            logging.error(f"暴力搜索失败: {e}")
            return []

    def _process_search_results(
        self,
        distances: np.ndarray,
        internal_faiss_indices: np.ndarray,
        candidate_uids_set: Set[str],
    ) -> List[Tuple[MemoryUnit, float]]:
        """处理搜索结果，过滤并返回有效的MemoryUnit"""
        results: List[Tuple[MemoryUnit, float]] = []
        internal_id_to_uid_map = {
            v: k for k, v in self._uid_to_internal_faiss_id.items()
        }

        for i in range(internal_faiss_indices.shape[1]):
            internal_id = internal_faiss_indices[0, i]
            if internal_id == -1:
                continue

            uid = internal_id_to_uid_map.get(internal_id)
            if uid and uid in candidate_uids_set:
                unit = self.get_unit(uid)
                if unit:
                    results.append((unit, float(distances[0, i])))
                else:
                    logging.warning(
                        f"在FAISS搜索结果中找到内部ID {internal_id}，但在内存单元字典中找不到对应的单元ID '{uid}'。"
                    )
            else:
                logging.debug(
                    f"在FAISS搜索结果中找到无法映射回单元ID的内部ID {internal_id}。"
                )

        return results

    def search_similarity_by_text(
        self,
        query_text: str,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        candidate_uids: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """通过查询文本进行相似性搜索。"""
        query_embedding = self._get_text_embedding(query_text)
        if query_embedding is None:
            return []
        return self.search_similarity_by_vector(
            query_embedding, k, ms_names, candidate_uids
        )

    def search_similarity_by_image(
        self,
        image_path: str,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        candidate_uids: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """通过查询图像路径进行相似性搜索。"""
        query_embedding = self._get_image_embedding(image_path)
        if query_embedding is None:
            return []
        return self.search_similarity_by_vector(
            query_embedding, k, ms_names, candidate_uids
        )

    # ==============================
    # 持久化方法（更新兼容性）
    # ==============================
    # 修改 dev/semantic_map.py 中的 save_map 方法

    def save_map(self, directory_path: str):
        """
        将 SemanticMap 的状态保存到指定目录。
        """
        os.makedirs(directory_path, exist_ok=True)

        # 1. 准备保存数据，排除不可序列化的对象
        memory_spaces_for_save = {}
        for space_name, space in self.memory_spaces.items():
            # 创建一个可序列化的MemorySpace副本
            space_data = {
                "name": space.name,
                "unit_uids": space._unit_uids.copy(),
                "child_space_names": space._child_space_names.copy(),
                # 不保存 _semantic_map_ref 弱引用
            }
            memory_spaces_for_save[space_name] = space_data

        # 2. 保存 MemoryUnit 和处理过的 MemorySpace 数据
        with open(os.path.join(directory_path, "semantic_map_data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "memory_units": self.memory_units,
                    "memory_spaces_data": memory_spaces_for_save,  # 保存处理过的数据
                    "_uid_to_internal_faiss_id": self._uid_to_internal_faiss_id,
                    "_internal_faiss_id_counter": self._internal_faiss_id_counter,
                    "embedding_dim": self.embedding_dim,
                    "faiss_index_type": self.faiss_index_type,
                    # 保存其他状态
                    "_modified_units": list(self._modified_units),
                    "_deleted_units": list(self._deleted_units),
                    "_last_sync_time": (
                        self._last_sync_time.isoformat()
                        if self._last_sync_time
                        else None
                    ),
                    "_max_memory_units": self._max_memory_units,
                    "_access_counts": self._access_counts.copy(),
                },
                f,
            )

        # 3. 保存 FAISS 索引
        if self.faiss_index:
            try:
                import faiss

                faiss.write_index(
                    self.faiss_index,
                    os.path.join(directory_path, "semantic_map.faissindex"),
                )
            except Exception as e:
                logging.warning(f"保存FAISS索引失败: {e}")

        logging.info(f"SemanticMap 已保存到目录: '{directory_path}'")

    @classmethod
    def load_map(
        cls,
        directory_path: str,
        image_embedding_model_name: Optional[str] = None,
        text_embedding_model_name: Optional[str] = None,
    ) -> "SemanticMap":
        """
        从指定目录加载 SemanticMap 的状态。
        """
        data_file = os.path.join(directory_path, "semantic_map_data.pkl")
        index_file = os.path.join(directory_path, "semantic_map.faissindex")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"SemanticMap 数据文件未找到: {data_file}")

        with open(data_file, "rb") as f:
            saved_state = pickle.load(f)

        # 使用保存的参数或加载时提供的参数初始化实例
        img_model = (
            image_embedding_model_name
            if image_embedding_model_name
            else "clip-ViT-B-32"
        )
        txt_model = (
            text_embedding_model_name
            if text_embedding_model_name
            else "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        )

        embedding_dim = saved_state.get("embedding_dim", 512)
        faiss_index_type = saved_state.get("faiss_index_type", "IDMap,Flat")

        instance = cls(
            image_embedding_model_name=img_model,
            text_embedding_model_name=txt_model,
            embedding_dim=embedding_dim,
            faiss_index_type=faiss_index_type,
        )

        # 恢复数据
        instance.memory_units = saved_state["memory_units"]

        # 重建 MemorySpace 对象
        memory_spaces_data = saved_state.get(
            "memory_spaces_data", saved_state.get("memory_spaces", {})
        )
        instance.memory_spaces = {}

        for space_name, space_data in memory_spaces_data.items():
            if isinstance(space_data, dict):
                # 新格式：从字典数据重建MemorySpace
                space = MemorySpace(space_data["name"])
                space._unit_uids = space_data.get("unit_uids", set())
                space._child_space_names = space_data.get("child_space_names", set())
            else:
                # 旧格式：直接使用MemorySpace对象
                space = space_data

            # 设置SemanticMap引用
            space._set_semantic_map_ref(instance)
            instance.memory_spaces[space_name] = space

        # 恢复其他状态
        instance._uid_to_internal_faiss_id = saved_state.get(
            "_uid_to_internal_faiss_id", {}
        )
        instance._internal_faiss_id_counter = saved_state.get(
            "_internal_faiss_id_counter", 0
        )
        instance._modified_units = set(saved_state.get("_modified_units", []))
        instance._deleted_units = set(saved_state.get("_deleted_units", []))
        instance._max_memory_units = saved_state.get("_max_memory_units", 10000)
        instance._access_counts = saved_state.get("_access_counts", {})

        # 恢复时间戳
        last_sync_time_str = saved_state.get("_last_sync_time")
        if last_sync_time_str:
            try:
                from datetime import datetime

                instance._last_sync_time = datetime.fromisoformat(last_sync_time_str)
            except Exception as e:
                logging.warning(f"恢复同步时间戳失败: {e}")
                instance._last_sync_time = None

        # 加载FAISS索引
        if os.path.exists(index_file):
            try:
                import faiss

                instance.faiss_index = faiss.read_index(index_file)
                logging.info(
                    f"FAISS 索引已从 '{index_file}' 加载，包含 {instance.faiss_index.ntotal} 个向量"
                )
            except Exception as e:
                logging.error(f"加载 FAISS 索引失败: {e}")
                instance.faiss_index = None
                instance._init_faiss_index()
        else:
            logging.warning(f"FAISS 索引文件 '{index_file}' 未找到")
            instance._init_faiss_index()

        logging.info(f"SemanticMap 已从目录 '{directory_path}' 加载")
        return instance

    # ==============================
    # 兼容旧接口的方法
    # ==============================

    def export_to_milvus(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        collection_name: str = "hippo_memory_units",
    ) -> bool:
        """
        将SemanticMap中的内存单元导出到Milvus数据库
        注意：此方法已废弃，推荐使用 sync_to_external() 方法
        """
        logging.warning(
            "export_to_milvus() 方法已废弃，推荐使用 sync_to_external() 方法"
        )

        try:
            # 创建Milvus操作类
            milvus_op = MilvusOperator(
                host=host,
                port=port,
                user=user,
                password=password,
                collection_name=collection_name,
                embedding_dim=self.embedding_dim,
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
                    if space.contains_unit(uid):
                        space_names.append(space_name)

                # 添加到Milvus
                if milvus_op.add_unit(unit, space_names):
                    success_count += 1
                else:
                    logging.warning(f"导出内存单元 '{uid}' 到Milvus失败")

            logging.info(
                f"成功导出 {success_count}/{len(self.memory_units)} 个内存单元到Milvus"
            )
            milvus_op.close()

            return success_count > 0

        except Exception as e:
            logging.error(f"导出到Milvus失败: {e}")
            return False

    def filter_memory_units(
        self,
        candidate_units: Optional[List[MemoryUnit]] = None,
        filter_condition: Optional[dict] = None,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[MemoryUnit]:
        """
        过滤内存单元，支持多种过滤条件和空间集合操作。
        """
        # 优先用集合操作获取候选单元
        if candidate_units is not None:
            units = self.units_union(candidate_units)
        elif ms_names:
            units = self.units_union(*ms_names)
        else:
            units = list(self.memory_units.values())

        def match(unit):
            if not filter_condition:
                return True
            for field, cond in filter_condition.items():
                val = getattr(unit, field, None)
                if val is None:
                    val = unit.raw_data.get(field)
                for op, op_val in cond.items():
                    if op == "eq" and not (val == op_val):
                        return False
                    if op == "ne" and not (val != op_val):
                        return False
                    if op == "in" and not (val in op_val):
                        return False
                    if op == "nin" and not (val not in op_val):
                        return False
                    if op == "gt" and not (val > op_val):
                        return False
                    if op == "gte" and not (val >= op_val):
                        return False
                    if op == "lt" and not (val < op_val):
                        return False
                    if op == "lte" and not (val <= op_val):
                        return False
                    if op == "contain" and not (op_val in str(val)):
                        return False
                    if op == "not_contain" and not (op_val not in str(val)):
                        return False
            return True

        return [u for u in units if match(u)]