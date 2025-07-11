import logging
import pickle
import os
from datetime import datetime

import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx
from collections import Counter


class MemoryUnit:
    """
    记忆单元 (MemoryUnit) 代表系统中的一个基本信息片段。
    它包含一个唯一的ID、一个存储具体数据的字典以及一个可选的向量表示。
    """

    def __init__(
        self,
        uid: str,
        raw_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """
        初始化一个记忆单元。
        参数:
            unit_id (str): 记忆单元的唯一标识符。
            value (Dict[str, Any]): 包含记忆单元具体内容的字典。例如: {"text": "一些描述", "image_path": "路径/到/图片.jpg", "type": "文档"}
            embedding (Optional[np.ndarray]): 该单元的向量表示。如果为None，则可以由SemanticMap生成。
        """
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("记忆单元UID不能为空")
        if not isinstance(raw_data, dict):
            raise ValueError("记忆单元Raw Data必须是一个字典")
        # 修改逻辑，允许metadata为None
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"记忆单元Metadata必须是一个字典")

        self.uid: str = uid
        self.raw_data: Dict[str, Any] = raw_data
        self.metadata = metadata or {
            "created": str(datetime.now()),
            "updated": str(datetime.now()),
        }
        self.embedding: Optional[np.ndarray] = embedding

    def __str__(self) -> str:
        embedding_shape = self.embedding.shape if self.embedding is not None else None
        return f"MemoryUnit(uid='{self.uid}', raw_data={self.raw_data}, metadata={self.metadata}, embedding_shape={embedding_shape})"

    def __repr__(self) -> str:
        return f"MemoryUnit({self.uid})"

    def __eq__(self, value):
        if not isinstance(value, MemoryUnit):
            return False
        return (
            self.uid == value.uid
            and self.raw_data == value.raw_data
            and self.metadata == value.metadata
            # and np.array_equal(self.embedding,value.embedding)
        )

    def __hash__(self):
        return hash(self.uid)


class MemorySpace:
    """
    记忆空间 (MemorySpace) 支持嵌套结构，使用引用管理 MemoryUnit 和其他 MemorySpace。
    存储 UID/名称引用而非完整对象，确保数据一致性和节省内存。
    支持局部faiss索引。
    """

    def __init__(self, ms_name: str, faiss_index_type: Optional[str] = None):
        if not isinstance(ms_name, str) or not ms_name.strip():
            raise ValueError("记忆空间名称不能为空")
        self.name: str = ms_name

        # 使用引用存储，而非完整对象
        self._unit_uids: Set[str] = set()  # 存储 MemoryUnit 的 UID 引用
        self._child_space_names: Set[str] = set()  # 存储子 MemorySpace 的名称引用

        # 索引相关
        self._emb_index = None
        self._index_to_uid: Dict[int, str] = {}  # 索引位置到UID的映射
        self._faiss_index_type: Optional[str] = faiss_index_type  # 支持自定义索引类型

        # 引用到父级SemanticMap的弱引用（用于获取实际对象）
        self._semantic_map_ref = None

    def __str__(self) -> str:
        total_members = len(self._unit_uids) + len(self._child_space_names)
        return f"MemorySpace(name={self.name}, units={len(self._unit_uids)}, child_spaces={len(self._child_space_names)})"

    def __repr__(self):
        return f"MemorySpace({self.name})"

    def _set_semantic_map_ref(self, semantic_map):
        """设置对SemanticMap的引用（由SemanticMap调用）"""
        import weakref

        self._semantic_map_ref = weakref.ref(semantic_map)

    def _get_semantic_map(self):
        """获取SemanticMap实例"""
        if self._semantic_map_ref is None:
            raise RuntimeError(f"MemorySpace '{self.name}' 没有关联的SemanticMap引用")
        semantic_map = self._semantic_map_ref()
        if semantic_map is None:
            raise RuntimeError(f"MemorySpace '{self.name}' 关联的SemanticMap已被回收")
        return semantic_map

    # ===============================
    # 主要方法：智能判断类型的统一接口
    # ===============================

    def add_unit(self, unit_or_uid: Union[str, "MemoryUnit"]):
        """
        智能添加MemoryUnit：自动判断输入是UID字符串还是MemoryUnit对象
        """
        if isinstance(unit_or_uid, str):
            # 输入是UID字符串
            self._add_unit_by_uid(unit_or_uid)
        elif hasattr(unit_or_uid, "uid"):
            # 输入是MemoryUnit对象
            self._add_unit_by_uid(unit_or_uid.uid)
        else:
            raise TypeError(
                f"add_unit() 期望 str(UID) 或 MemoryUnit 对象，得到 {type(unit_or_uid)}"
            )

    def add_child_space(self, space_or_name: Union[str, "MemorySpace"]):
        """
        智能添加子MemorySpace：自动判断输入是名称字符串还是MemorySpace对象
        """
        if isinstance(space_or_name, str):
            # 输入是空间名称字符串
            self._add_child_space_by_name(space_or_name)
        elif hasattr(space_or_name, "name"):
            # 输入是MemorySpace对象
            self._add_child_space_by_name(space_or_name.name)
        else:
            raise TypeError(
                f"add_child_space() 期望 str(名称) 或 MemorySpace 对象，得到 {type(space_or_name)}"
            )

    def remove_unit(self, unit_or_uid: Union[str, "MemoryUnit"]):
        """
        智能移除MemoryUnit：自动判断输入是UID字符串还是MemoryUnit对象
        """
        if isinstance(unit_or_uid, str):
            # 输入是UID字符串
            self._remove_unit_by_uid(unit_or_uid)
        elif hasattr(unit_or_uid, "uid"):
            # 输入是MemoryUnit对象
            self._remove_unit_by_uid(unit_or_uid.uid)
        else:
            raise TypeError(
                f"remove_unit() 期望 str(UID) 或 MemoryUnit 对象，得到 {type(unit_or_uid)}"
            )

    def remove_child_space(self, space_or_name: Union[str, "MemorySpace"]):
        """
        智能移除子MemorySpace：自动判断输入是名称字符串还是MemorySpace对象
        """
        if isinstance(space_or_name, str):
            # 输入是空间名称字符串
            self._remove_child_space_by_name(space_or_name)
        elif hasattr(space_or_name, "name"):
            # 输入是MemorySpace对象
            self._remove_child_space_by_name(space_or_name.name)
        else:
            raise TypeError(
                f"remove_child_space() 期望 str(名称) 或 MemorySpace 对象，得到 {type(space_or_name)}"
            )

    def contains_unit(
        self, unit_or_uid: Union[str, "MemoryUnit"], recursive: bool = False
    ) -> bool:
        """
        智能检查是否包含MemoryUnit：自动判断输入类型
        """
        if isinstance(unit_or_uid, str):
            uid = unit_or_uid
        elif hasattr(unit_or_uid, "uid"):
            uid = unit_or_uid.uid
        else:
            raise TypeError(
                f"contains_unit() 期望 str(UID) 或 MemoryUnit 对象，得到 {type(unit_or_uid)}"
            )

        if uid in self._unit_uids:
            return True

        if recursive:
            child_uids = self.get_all_unit_uids(recursive=True)
            return uid in child_uids

        return False

    def contains_space(
        self, space_or_name: Union[str, "MemorySpace"], recursive: bool = False
    ) -> bool:
        """
        智能检查是否包含子MemorySpace：自动判断输入类型
        """
        if isinstance(space_or_name, str):
            space_name = space_or_name
        elif hasattr(space_or_name, "name"):
            space_name = space_or_name.name
        else:
            raise TypeError(
                f"contains_space() 期望 str(名称) 或 MemorySpace 对象，得到 {type(space_or_name)}"
            )

        if space_name in self._child_space_names:
            return True

        if recursive:
            child_names = self.get_all_child_space_names(recursive=True)
            return space_name in child_names

        return False

    # ===============================
    # 内部实现方法（基于引用操作）
    # ===============================

    def _add_unit_by_uid(self, uid: str):
        """内部方法：通过UID添加MemoryUnit引用"""
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("MemoryUnit UID不能为空")

        # 验证UID是否存在（如果有SemanticMap引用）
        try:
            semantic_map = self._get_semantic_map()
            if uid not in semantic_map.memory_units:
                logging.warning(
                    f"MemoryUnit '{uid}' 在SemanticMap中不存在，仍将添加引用"
                )
        except RuntimeError:
            # 没有SemanticMap引用时跳过验证
            pass

        self._unit_uids.add(uid)
        logging.debug(f"MemoryUnit引用 '{uid}' 已添加到MemorySpace '{self.name}'")

    def _add_child_space_by_name(self, space_name: str):
        """内部方法：通过名称添加子MemorySpace引用"""
        if not isinstance(space_name, str) or not space_name.strip():
            raise ValueError("MemorySpace名称不能为空")

        if space_name == self.name:
            raise ValueError("不能将自身作为子空间")

        # 验证空间名称是否存在（如果有SemanticMap引用）
        try:
            semantic_map = self._get_semantic_map()
            if space_name not in semantic_map.memory_spaces:
                logging.warning(
                    f"MemorySpace '{space_name}' 在SemanticMap中不存在，仍将添加引用"
                )
        except RuntimeError:
            # 没有SemanticMap引用时跳过验证
            pass

        self._child_space_names.add(space_name)
        logging.debug(
            f"MemorySpace引用 '{space_name}' 已添加到MemorySpace '{self.name}'"
        )

    def _remove_unit_by_uid(self, uid: str):
        """内部方法：通过UID移除MemoryUnit引用"""
        if uid in self._unit_uids:
            self._unit_uids.remove(uid)
            logging.debug(f"MemoryUnit引用 '{uid}' 已从MemorySpace '{self.name}' 移除")
        else:
            logging.warning(
                f"MemoryUnit引用 '{uid}' 在MemorySpace '{self.name}' 中不存在"
            )

    def _remove_child_space_by_name(self, space_name: str):
        """内部方法：通过名称移除子MemorySpace引用"""
        if space_name in self._child_space_names:
            self._child_space_names.remove(space_name)
            logging.debug(
                f"MemorySpace引用 '{space_name}' 已从MemorySpace '{self.name}' 移除"
            )
        else:
            logging.warning(
                f"MemorySpace引用 '{space_name}' 在MemorySpace '{self.name}' 中不存在"
            )

    # ===============================
    # 查询和访问方法
    # ===============================

    def get_unit_uids(self) -> Set[str]:
        """获取所有MemoryUnit的UID集合"""
        return self._unit_uids.copy()

    def get_child_space_names(self) -> Set[str]:
        """获取所有子MemorySpace的名称集合"""
        return self._child_space_names.copy()

    def get_all_unit_uids(self, recursive: bool = True) -> Set[str]:
        """递归获取此空间及所有子空间中的所有MemoryUnit UID"""
        result = self._unit_uids.copy()

        if recursive:
            try:
                semantic_map = self._get_semantic_map()
                for child_space_name in self._child_space_names:
                    child_space = semantic_map.memory_spaces.get(child_space_name)
                    if child_space:
                        result.update(child_space.get_all_unit_uids(recursive=True))
                    else:
                        logging.warning(
                            f"子空间 '{child_space_name}' 在SemanticMap中不存在"
                        )
            except RuntimeError:
                logging.warning("无法访问SemanticMap，跳过递归获取")

        return result

    def get_all_child_space_names(self, recursive: bool = True) -> Set[str]:
        """递归获取所有子MemorySpace名称（不含自身）"""
        result = self._child_space_names.copy()

        if recursive:
            try:
                semantic_map = self._get_semantic_map()
                for child_space_name in self._child_space_names:
                    child_space = semantic_map.memory_spaces.get(child_space_name)
                    if child_space:
                        result.update(
                            child_space.get_all_child_space_names(recursive=True)
                        )
                    else:
                        logging.warning(
                            f"子空间 '{child_space_name}' 在SemanticMap中不存在"
                        )
            except RuntimeError:
                logging.warning("无法访问SemanticMap，跳过递归获取")

        return result

    def get_all_units(self) -> List["MemoryUnit"]:
        """递归获取此空间及所有子空间中的所有MemoryUnit对象"""
        try:
            semantic_map = self._get_semantic_map()
            unit_uids = self.get_all_unit_uids(recursive=True)

            units = []
            for uid in unit_uids:
                unit = semantic_map.memory_units.get(uid)
                if unit:
                    units.append(unit)
                else:
                    logging.warning(f"MemoryUnit '{uid}' 在SemanticMap中不存在")

            return units
        except RuntimeError:
            logging.error("无法访问SemanticMap，无法获取MemoryUnit对象")
            return []

    def get_all_spaces(self) -> List["MemorySpace"]:
        """递归获取所有子MemorySpace对象（不含自身）"""
        try:
            semantic_map = self._get_semantic_map()
            space_names = self.get_all_child_space_names(recursive=True)

            spaces = []
            for space_name in space_names:
                space = semantic_map.memory_spaces.get(space_name)
                if space:
                    spaces.append(space)
                else:
                    logging.warning(f"MemorySpace '{space_name}' 在SemanticMap中不存在")

            return spaces
        except RuntimeError:
            logging.error("无法访问SemanticMap，无法获取MemorySpace对象")
            return []

    # ===============================
    # 索引和搜索功能
    # ===============================

    # TODO: MemorySpace的索引构建应该也可以支持自定义索引类型，可以和所属SemanticMap的索引类型相同，也可以由参数主动指定。
    def build_index(self, embedding_dim: int = 512, min_unit_threshold: int = 100):
        """
        递归收集所有unit并构建局部faiss索引。
        如果内部unit数量小于阈值（如100），则不建立索引。
        支持自定义faiss_index_type。
        """
        try:
            semantic_map = self._get_semantic_map()
            units = self.get_all_units()
            if len(units) < min_unit_threshold:
                self._emb_index = None
                self._index_to_uid = {}
                logging.info(
                    f"MemorySpace {self.name} 单元数{len(units)}，未建立索引。"
                )
                return

            import faiss

            # 优先用自身的faiss_index_type，否则继承父级SemanticMap，否则用默认
            faiss_index_type = (
                self._faiss_index_type
                or getattr(semantic_map, "faiss_index_type", None)
                or "IDMap,Flat"
            )
            self._emb_index = faiss.index_factory(embedding_dim, faiss_index_type)
            self._index_to_uid = {}
            embeddings = []
            count = 0
            for unit in units:
                if unit.embedding is not None:
                    embeddings.append(unit.embedding)
                    self._index_to_uid[count] = unit.uid
                    count += 1
            if not embeddings:
                logging.warning(
                    f"MemorySpace '{self.name}' 没有有效的embeddings来构建索引"
                )
                return
            self._emb_index.add(np.array(embeddings, dtype=np.float32))
            logging.info(
                f"MemorySpace '{self.name}' 索引构建完成，包含 {count} 个向量 (faiss_index_type={faiss_index_type})"
            )
        except ImportError:
            logging.error("FAISS不可用，无法构建本地索引")
        except Exception as e:
            logging.error(f"构建索引时出错: {e}")

    def search_similarity_units_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple["MemoryUnit", float]]:
        """
        在局部索引中搜索相似单元。如果没有索引则暴力搜索。
        """
        semantic_map = self._get_semantic_map()
        # 优先用索引
        if self._emb_index and getattr(self._emb_index, "ntotal", 0) > 0:
            try:
                query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
                D, I = self._emb_index.search(query_vector_np, top_k)
                results = []
                for i in range(len(I[0])):
                    idx = int(I[0][i])
                    if I[0][i] == -1:
                        continue
                    uid = self._index_to_uid.get(idx)
                    if uid:
                        unit = semantic_map.memory_units.get(uid)
                        if unit:
                            results.append((unit, float(D[0][i])))
                        else:
                            logging.warning(
                                f"索引中的UID '{uid}' 在SemanticMap中不存在"
                            )
                return results
            except Exception as e:
                logging.error(f"索引搜索出错: {e}")
                # 回退暴力搜索
        # 暴力搜索
        units = self.get_all_units()
        if not units:
            return []
        sims = []
        for u in units:
            if u.embedding is not None:
                a = query_vector.astype(np.float32)
                b = u.embedding.astype(np.float32)
                sim = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                )
                sims.append((u, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    # ===============================
    # 持久化方法
    # ===============================

    def save(self, file_path: str):
        """保存MemorySpace（不包含SemanticMap引用）"""
        save_data = {
            "name": self.name,
            "unit_uids": self._unit_uids,
            "child_space_names": self._child_space_names,
            # 不保存索引和SemanticMap引用
        }

        with open(file_path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, file_path: str) -> "MemorySpace":
        """从文件加载MemorySpace"""
        with open(file_path, "rb") as f:
            save_data = pickle.load(f)

        instance = cls(save_data["name"])
        instance._unit_uids = save_data.get("unit_uids", set())
        instance._child_space_names = save_data.get("child_space_names", set())

        return instance


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
        self.semantic_map: SemanticMap = (
            semantic_map_instance if semantic_map_instance else SemanticMap()
        )
        self.nx_graph: nx.DiGraph = (
            nx.DiGraph()
        )  # 使用 NetworkX有向图存储节点和显式关系
        logging.info("SemanticGraph 已初始化。")

        # 添加Neo4j连接跟踪
        self._neo4j_connection: Optional[Neo4jOperator] = None  # 指向Neo4j的连接
        self._modified_relationships = (
            set()
        )  # 修改过的关系 (source_id, target_id, rel_type)
        self._deleted_relationships = set()  # 删除的关系
        self._modified_units = set()  # 修改过的内存单元
        self._deleted_units = set()  # 删除的内存单元

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
        password: str = "password",
        database: str = "neo4j",
        # Milvus连接参数
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_user: str = "neo4j",
        milvus_password: str = "k4s9k4s9",
        milvus_collection: str = "hippo_memory_units",
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
        向图谱添加一个内存单元 (节点)。
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
        在两个已存在的内存单元 (节点) 或 MemorySpace 之间添加一条显式关系 (边)。
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
        edge_attributes = {
            "type": relationship_name,
            "created": str(datetime.now()),
            **kwargs,
        }
        self.nx_graph.add_edge(src_id, tgt_id, **edge_attributes)

        # 缓存关系属性
        self.swap_in_relationship(src_id, tgt_id, relationship_name, kwargs)

        # 记录修改
        self._modified_relationships.add((src_id, tgt_id, relationship_name))

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

    def delete_unit(
        self, uid: str, rebuild_semantic_map_index_immediately: bool = False
    ):
        """从图谱和底层的 SemanticMap 中删除一个内存单元及其所有相关关系。"""
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
        """将一个内存单元添加到 SemanticMap 中的指定记忆空间。"""
        self.semantic_map.add_unit_to_space(unit_or_uid, space_name)

    def remove_unit_from_space_in_map(self, unit_or_uid, space_name: str):
        """从 SemanticMap 中的指定记忆空间移除一个内存单元。"""
        self.semantic_map.remove_unit_from_space(unit_or_uid, space_name)

    def get_units_in_memory_space(
        self, ms_names, recursive: bool = True
    ) -> List[MemoryUnit]:
        """获取指定记忆空间中的所有单元"""
        return self.semantic_map.get_units_in_memory_space(ms_names, recursive)

    def search_similarity_in_graph(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        query_image_path: Optional[str] = None,
        k: int = 5,
        ms_names: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        在图谱中执行语义相似性搜索，兼容新的 MemorySpace 架构
        """
        # 如果指定了 ms_names，使用新的搜索方法
        if ms_names:
            if query_text:
                return self.semantic_map.search_similarity_units(
                    query_text, k, ms_names, recursive
                )
            else:
                # 对于非文本查询，先在指定空间获取单元，然后进行搜索
                units = self.get_units_in_memory_space(ms_names, recursive)
                if not units:
                    return []

                # 构建候选单元UID列表
                candidate_uids = [unit.uid for unit in units]

                # 执行搜索
                if query_embedding is not None:
                    return self.semantic_map.search_similarity_by_vector(
                        query_embedding, k, ms_names, candidate_uids
                    )
                elif query_image_path is not None:
                    return self.semantic_map.search_similarity_by_image(
                        query_image_path, k, ms_names, candidate_uids
                    )
        else:
            # 使用原有的搜索方法
            if query_embedding is not None:
                return self.semantic_map.search_similarity_by_vector(
                    query_embedding, k, ms_names
                )
            elif query_text is not None:
                return self.semantic_map.search_similarity_by_text(
                    query_text, k, ms_names
                )
            elif query_image_path is not None:
                return self.semantic_map.search_similarity_by_image(
                    query_image_path, k, ms_names
                )

        logging.warning(
            "必须提供 query_text, query_embedding 或 query_image_path 之一进行相似性搜索。"
        )
        return []

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
                if self._neo4j_connection.add_relationship(
                    source_uid, target_uid, rel_type, properties
                ):
                    synced_count += 1
                    logging.debug(
                        f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 已同步到Neo4j"
                    )
                else:
                    logging.warning(
                        f"关系 ({source_uid} -[{rel_type}]-> {target_uid}) 同步到Neo4j失败"
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

                # 确保节点存在
                if not self._neo4j_connection.ensure_node_exists(source_uid):
                    logging.warning(f"确保源节点 '{source_uid}' 存在失败")
                if not self._neo4j_connection.ensure_node_exists(target_uid):
                    logging.warning(f"确保目标节点 '{target_uid}' 存在失败")

                # 同步到Neo4j
                if self._neo4j_connection.add_relationship(
                    source_uid, target_uid, rel_type, properties
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
                try:
                    if self._neo4j_connection.delete_relationship(
                        source_uid, target_uid, rel_type
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
        """从底层的 SemanticMap 检索内存单元对象。"""
        return self.get_unit(uid)

    def build_semantic_map_index(self):
        """构建底层 SemanticMap 的 FAISS 索引。"""
        self.semantic_map.build_index()

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
            for space_name in ms_names:
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

        similar_units_with_scores = self.semantic_map.search_similarity_by_vector(
            start_unit.embedding,
            k=k + 1,  # 获取稍多一些，以防 uid 是最相似的
            ms_names=ms_names,
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
            f"  - 内存单元总数: {num_map_units}\n"
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
        return self.semantic_map.filter_memory_units(
            candidate_units=candidate_units,
            filter_condition=filter_condition,
            ms_names=ms_names,
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
                    logging.warning(
                        f"导出从 '{source}' 到 '{target}' 的 '{rel_type}' 关系到Neo4j失败"
                    )

            logging.info(
                f"成功导出 {unit_success_count}/{len(self.semantic_map.memory_units)} 个内存单元和 {rel_success_count}/{self.nx_graph.number_of_edges()} 个关系到Neo4j"
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
