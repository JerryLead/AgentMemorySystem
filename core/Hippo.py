import os
import pickle
import logging

from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx
from collections import Counter

# 配置日志记录器
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


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
            raise ValueError("MemoryUnit uid cannot be empty")
        if not isinstance(raw_data, dict):
            raise ValueError("MemoryUnit raw_data must be a dict")

        self.uid: str = uid
        self.raw_data: Dict[str, Any] = raw_data
        self.metadata = metadata or {
            "created": datetime.now(),
            "updated": datetime.now(),
        }
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"记忆单元Metadata必须是一个字典")

        self.embedding: Optional[np.ndarray] = embedding

    def __str__(self) -> str:
        embedding_shape = self.embedding.shape if self.embedding is not None else None
        return f"MemoryUnit(uid='{self.uid}', raw_data={self.raw_data}, metadata={self.metadata}, embed_shape={embedding_shape})"

    def __repr__(self) -> str:
        return f"MemoryUnit({self.uid})"

    def __eq__(self, value):
        if not isinstance(value, MemoryUnit):
            return False
        return (
            self.uid == value.uid
            and self.raw_data == value.raw_data
            and self.metadata == value.metadata
        )

    def __hash__(self):
        return hash(self.uid)


class MemorySpace:
    """
    记忆空间 (MemorySpace) 用于组织和管理一组相关的记忆单元，一个记忆空间内的记忆单元拥有相同的类型。
    一个记忆单元可以属于多个记忆空间。
    """

    def __init__(
        self,
        ms_name: str,
        index: faiss.Index = None,
    ):
        """
        初始化一个记忆空间。
        参数:
            ms_name (str): 记忆空间的名称，应唯一。
            index (faiss.Index): 用户指定的Faiss索引类型。
        """
        if not isinstance(ms_name, str) or not ms_name.strip():
            raise ValueError("记忆空间名称不能为空")
        self.name: str = ms_name
        self._units: Dict[str, MemoryUnit] = {}  # 存储记忆单元字典

        self._emb_index = index
        self.index_to_unit: Dict[int, MemoryUnit] = {}  # 额外维护索引到uid的映射

    def __str__(self) -> str:
        return f"MemorySpace(name={self.name}, unit_count={len(self._units)})"

    def __repr__(self):
        return f"MemorySpace({self.name})"

    def add_unit(self, unit: MemoryUnit):
        """向此记忆空间添加一个记忆单元"""
        if not isinstance(unit, MemoryUnit):
            raise TypeError(f"{unit} is not a MemoryUnit")
        if unit in self._units:
            raise ValueError(f"Duplicate unit {unit} in MemorySpace {self.name}")

        self._units[unit.uid] = unit
        logging.debug(f"记忆单元 '{unit}' 已添加到记忆空间 '{self.name}'。")

    def add_units(self, units: List[MemoryUnit]):
        for unit in units:
            self.add_unit(unit)

    def remove_unit(self, uid: str):
        """从此记忆空间移除一个记忆单元的ID。"""
        if uid not in self._units:
            raise ValueError(f"MemoryUnit {uid} not in MemorySpace {self.name}")

        del self._units[uid]
        logging.debug(f"记忆单元ID '{uid}' 已从记忆空间 '{self.name}' 移除。")

    def remove_units(self, uids: List[str]):
        for uid in uids:
            self.remove_unit(uid)

    def update_unit(self, unit: MemoryUnit):
        if unit.uid not in self._units:
            raise ValueError(f"MemoryUnit {unit.uid} not in MemorySpace {self.name}")

        self._units[unit.uid] = unit
        logging.debug(f"记忆单元{unit}已更新到记忆空间{self.name}")

    def update_units(self, units: List[MemoryUnit]):
        for unit in units:
            self.update_unit(unit)

    def get_unit(self, uid: str):
        unit = self._units.get(uid, None)
        if not unit:
            raise ValueError(f"MemoryUnit {unit.uid} not in MemorySpace {self.name}")
        return unit

    def get_units(self, uids: List[str]):
        units = [self.get_unit(uid) for uid in uids]
        return units

    def get_all_units(self) -> List[MemoryUnit]:
        """获取此记忆空间中所有记忆单元的ID集合。"""
        return list(self._units.values())

    def build_index(self):
        self._emb_index.reset()

        self.index_to_unit = {}
        count = 0
        embeddings = []

        for unit in self.get_all_units():
            if unit.embedding is not None:
                embeddings.append(unit.embedding)
                self.index_to_unit[count] = unit
                count += 1

        should_train_index = [faiss.IndexIVFFlat, faiss.IndexIVFPQ]

        if any([isinstance(self._emb_index, x) for x in should_train_index]):
            self._emb_index.train(np.array(embeddings))

        self._emb_index.add(np.array(embeddings))


class SemanticMap:
    """
    语义地图 (SemanticMap) 负责存储记忆空间及其向量嵌入，并支持基于相似度的搜索。
    它还管理记忆空间，允许在特定上下文中进行操作。
    类似于一个向量数据库。
    """

    DEFAULT_TEXT_EMBEDDING_KEY = (
        "text_content"  # MemoryUnit.value 中用于文本嵌入的默认键
    )
    DEFAULT_IMAGE_EMBEDDING_KEY = (
        "image_path"  # MemoryUnit.value 中用于图像嵌入的默认键
    )

    def __init__(
        self,
        image_embedding_model_name: str = "clip-ViT-B-32",
        text_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        index: Optional[faiss.Index] = None,
        update_interval: Optional[int] = 100,
    ):
        """
        初始化语义地图。
        参数:
            image_embedding_model_name (str): 用于图像嵌入的 SentenceTransformer 模型名称。
            text_embedding_model_name (str): 用于文本嵌入的 SentenceTransformer 模型名称。
            index (faiss.Index): 用户指定的Faiss索引类型。
            update_interval (int): 每隔多少次操作骤更新一次索引。
        """
        self._memory_spaces: Dict[str, MemorySpace] = (
            {}
        )  # 存储 MemorySpace 对象，键为 ms_name
        self._reverse_index: Dict[str, List[str]] = {}  # uid -> namems_name 快速查找

        if index:
            if not isinstance(index, faiss.Index):
                raise ValueError("index must be a faiss.Index.")
        else:
            index = faiss.IndexFlatL2(512)

        self._emb_index = index  # 全局索引
        self._index_to_unit: Dict[int, MemoryUnit] = {}

        if update_interval:
            self._update_interval = update_interval
            self._operation_counter = 0

        try:
            self.text_model = SentenceTransformer(text_embedding_model_name)
            self.image_model = SentenceTransformer(image_embedding_model_name)
        except FileNotFoundError or ValueError as e:
            raise e

        logging.info(f"SemanticMap 已初始化。索引类型：{self._emb_index}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """为给定文本生成嵌入向量。"""
        if not text or not isinstance(text, str):
            logging.warning("尝试为无效文本生成嵌入。")
            raise ValueError("Invalid text for embedding.")
        try:
            emb = self.text_model.encode(text)
            return emb.astype(np.float32)
        except Exception as e:
            logging.error(f"生成文本嵌入失败: '{text[:50]}...' - {e}")
            raise e

    def _get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """为给定图像路径生成嵌入向量。"""
        if not image_path or not isinstance(image_path, str):
            logging.warning("尝试为无效图像路径生成嵌入。")
            raise ValueError("无效的图像路径")
        if not os.path.isfile(image_path):
            logging.error(f"图像文件未找到: {image_path}")
            raise FileNotFoundError(f"图像文件未找到：{image_path}")
        try:
            img = Image.open(image_path)
            emb = self.image_model.encode(img)
            return emb.astype(np.float32)
        except Exception as e:
            logging.error(f"生成图像嵌入失败: '{image_path}' - {e}")
            raise e

    def add_unit(
        self,
        unit: MemoryUnit,
        ms_names: Optional[List[str]] = None,
        rebuild_index_immediately: bool = False,
    ):
        """
        添加一个MemoryUnit到SemanticMap指定的MemorySpace中。

        :param unit: 要添加的MemoryUnit。
        :param ms_names: 指定的MemorySpace名称列表。
        :param rebuild_index_immediately: 是否立即重建索引。
        """
        if not isinstance(unit, MemoryUnit):
            raise TypeError(f"{unit} is not a MemoryUnit.")

        if not ms_names:
            raise ValueError("Should provide at least one MemorySpace name.")

        # 生成或更新嵌入
        text = unit.raw_data.get("text_content")
        img = unit.raw_data.get("image_path")
        if text and img:
            raise ValueError(
                f"MemoryUnit {unit} cannot have both text_content and image_path."
            )
        if text:
            unit.embedding = self._get_text_embedding(text)
        elif img:
            unit.embedding = self._get_image_embedding(img)
        # 如果没有文本或图像，则不生成嵌入，且不进行索引

        for ms_name in ms_names:
            ms = self._memory_spaces.get(ms_name, None)
            if not ms:
                raise ValueError(f"MemorySpace {ms_name} does not exist.")
            ms.add_unit(unit)

            if not self._reverse_index.get(unit.uid):
                self._reverse_index[unit.uid] = []
            self._reverse_index[unit.uid].append(ms_name)

        if rebuild_index_immediately:
            self.build_index()  # 立即重建索引 (可能效率不高，除非是单个添加)
            return

        if hasattr(self, "_update_interval"):
            self._operation_counter += 1
            if self._operation_counter >= self._update_interval:
                self.build_index()
                self._operation_counter = 0

    def remove_unit(
        self,
        uid: str,
        rebuild_index_immediately: bool = False,
    ):
        """
        从SemanticMap中删除指定MemoryUnit。

        :param uid: 要删除的MemoryUnit的ID。
        :param rebuild_index_immediately: 是否立即重建索引。
        """
        if uid not in self._reverse_index:
            raise ValueError(f"MemoryUnit {uid} does not exist.")

        ms_names = self._reverse_index[uid]
        for ms_name in ms_names:
            self._memory_spaces[ms_name].remove_unit(uid)
        self._reverse_index.pop(uid)

        if rebuild_index_immediately:
            self.build_index()
            return

        if hasattr(self, "_update_interval"):
            self._operation_counter += 1
            if self._operation_counter >= self._update_interval:
                self.build_index()
                self._operation_counter = 0

    def update_unit(self, unit: MemoryUnit):
        if unit.uid not in self._reverse_index:
            raise ValueError(f"MemoryUnit {unit.uid} does not exist.")

        ms_names = self._reverse_index[unit.uid]
        for ms_name in ms_names:
            self._memory_spaces[ms_name].update_unit(unit)

        if hasattr(self, "_update_interval"):
            self._operation_counter += 1
            if self._operation_counter >= self._update_interval:
                self.build_index()
                self._operation_counter = 0

    def get_unit(self, uid: str) -> MemoryUnit:
        """通过ID检索记忆单元。"""
        if uid not in self._reverse_index:
            raise ValueError(f"MemoryUnit {uid} does not exist.")

        ms_name = self._reverse_index[uid][0]
        return self._memory_spaces[ms_name].get_unit(uid)

    def build_index(self):
        """
        根据当前所有具有有效嵌入的记忆单元构建（或重建）FAISS索引。
        """
        self._emb_index.reset()
        self._index_to_unit = {}
        count = 0
        embeddings = []

        for ms in self._memory_spaces.values():
            ms.build_index()
            for unit in ms.get_all_units():
                if unit.embedding is not None:
                    embeddings.append(unit.embedding)
                    self._index_to_unit[count] = unit
                    count += 1

        should_train_index = [faiss.IndexIVFFlat, faiss.IndexIVFPQ]

        if any([isinstance(self._emb_index, x) for x in should_train_index]):
            self._emb_index.train(np.array(embeddings))

        self._emb_index.add(np.array(embeddings))

    def search_similarity_units_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        ms_name: Optional[str] = None,
    ) -> List[MemoryUnit]:
        """
        通过查询向量在语义地图中搜索相似的记忆单元。
        参数:
            query_vector (np.ndarray): 用于查询的嵌入向量。
            k (int): 要返回的最相似单元的数量。
            ms_name (Optional[str]): 如果提供，则仅在指定的记忆空间内搜索。
            filter_unit_ids (Optional[Set[str]]): 一个可选的单元ID集合，用于进一步限制搜索范围（仅搜索这些ID对应的单元）。
                                                如果同时提供了 ms_name 和 filter_unit_ids，则取它们的交集。
        返回:
            List[Tuple[MemoryUnit, float]]: 一个元组列表，每个元组包含 (相似的MemoryUnit, 相似度得分/距离)。
                                            列表按相似度降序排列 (距离越小越相似)。
        """
        index = self._emb_index
        index_map = self._index_to_unit

        if ms_name:
            ms: MemorySpace = self.get_memory_space(ms_name)
            if not ms._emb_index:
                raise ValueError(
                    f"MemorySpace {ms_name} does not have an index. Please build index first."
                )
            index = ms._emb_index
            index_map = ms.index_to_unit
        else:
            if not self._emb_index or self._emb_index.ntotal == 0:
                raise ValueError("Index is empty. Please build index first.")

        query_vector_np = query_vector.reshape(1, -1).astype(np.float32)

        results = []
        D, I = index.search(query_vector_np, top_k)
        print(f"Search result D: {D}, I: {I}")
        for i in range(len(I[0])):
            print(f"i: {i}, I[0][i]: {I[0][i]}, int(I[0][i]) - 1: {int(I[0][i]) - 1}")
            if I[0][i] == -1:
                continue
            unit = index_map[int(I[0][i]) - 1]
            if not unit.metadata:
                unit.metadata = {}
            results.append((unit, D[0][i]))

        return results

    def search_similarity_units_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        ms_name: Optional[str] = None,
    ) -> List[MemoryUnit]:
        """通过查询文本进行相似性搜索。"""
        query_vector = self._get_text_embedding(query_text)
        if query_vector is None:
            return []
        return self.search_similarity_units_by_vector(query_vector, top_k, ms_name)

    def search_similarity_units_by_image(
        self,
        image_path: str,
        top_k: int = 5,
        ms_name: Optional[str] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """通过查询图像路径进行相似性搜索。"""
        query_vector = self._get_image_embedding(image_path)
        if query_vector is None:
            return []
        return self.search_similarity_units_by_vector(query_vector, top_k, ms_name)

    # --- MemorySpace 管理方法 ---
    def create_memory_space(self, ms_name: str):
        """创建一个记忆空间。"""
        if ms_name in self._memory_spaces:
            raise ValueError(f"MemorySpace {ms_name} already exists.")

        self._memory_spaces[ms_name] = MemorySpace(
            ms_name=ms_name, index=self._emb_index
        )
        logging.info(f"记忆空间 '{ms_name}' 已创建。")

    def remove_memory_space(self, ms_name: str):
        """删除一个记忆空间"""
        if ms_name not in self._memory_spaces:
            raise ValueError(f"MemorySpace {ms_name} does not exist.")

        self._memory_spaces.pop(ms_name)
        logging.info(f"MemorySpace {ms_name} has been removed.")
        # TODO: 直接删除会不会导致内存泄漏？

    def update_memory_space(self, ms_name: str):
        raise NotImplementedError

    def get_memory_space(self, ms_name: str) -> MemorySpace:
        """通过名称获取记忆空间。"""
        if ms_name not in self._memory_spaces:
            raise ValueError(f"MemorySpace {ms_name} does not exist.")

        return self._memory_spaces[ms_name]

    def get_all_memory_spaces(self) -> List[MemorySpace]:
        """获取所有记忆空间"""
        return list(self._memory_spaces.values())

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
            pickle.dump(
                {
                    "memory_units": self.memory_units,
                    "memory_spaces": self._memory_spaces,
                    "_unit_id_to_internal_faiss_id": self._unit_id_to_internal_faiss_id,
                    "_internal_faiss_id_counter": self._internal_faiss_id_counter,
                    "embedding_dim": self.embedding_dim,
                    "faiss_index_type": self.faiss_index_type,
                },
                f,
            )

        # 2. 保存 FAISS 索引
        if self.faiss_index:
            faiss.write_index(
                self.faiss_index,
                os.path.join(directory_path, "semantic_map.faissindex"),
            )

        logging.info(f"SemanticMap 已保存到目录: '{directory_path}'")

    @classmethod
    def load_map(
        cls,
        directory_path: str,
        image_embedding_model_name: Optional[str] = None,  # 加载时可以覆盖模型名称
        text_embedding_model_name: Optional[str] = None,
    ) -> "SemanticMap":
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
        img_model = (
            image_embedding_model_name
            if image_embedding_model_name
            else "clip-ViT-B-32"
        )  # 默认值
        txt_model = (
            text_embedding_model_name
            if text_embedding_model_name
            else "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        )  # 默认值

        # 从保存的状态中获取参数，如果存在的话
        embedding_dim = saved_state.get("embedding_dim", 512)
        faiss_index_type = saved_state.get(
            "faiss_index_type", "IDMap,Flat"
        )  # 确保有默认值

        # 如果加载时提供了模型名称，则使用它们
        # (注意：如果模型名称与保存时的不一致，嵌入维度也可能需要调整，这里简化处理)
        # 实际应用中，模型和维度应保持一致或有兼容性策略

        instance = cls(
            image_embedding_model_name=img_model,  # 实际使用时应确保与保存时一致或兼容
            text_embedding_model_name=txt_model,  # 同上
            embedding_dim=embedding_dim,
            faiss_index_type=faiss_index_type,
        )

        instance.memory_units = saved_state["memory_units"]
        instance._memory_spaces = saved_state["memory_spaces"]
        instance._unit_id_to_internal_faiss_id = saved_state.get(
            "_unit_id_to_internal_faiss_id", {}
        )
        instance._internal_faiss_id_counter = saved_state.get(
            "_internal_faiss_id_counter", 0
        )

        if os.path.exists(index_file):
            try:
                instance.faiss_index = faiss.read_index(index_file)
                logging.info(
                    f"FAISS 索引已从 '{index_file}' 加载。包含 {instance.faiss_index.ntotal} 个向量。"
                )
                # 验证维度
                if instance.faiss_index.d != instance.embedding_dim:
                    logging.warning(
                        f"加载的FAISS索引维度 ({instance.faiss_index.d}) 与 SemanticMap 期望维度 ({instance.embedding_dim}) 不符。可能需要重建索引。"
                    )
            except Exception as e:
                logging.error(f"加载 FAISS 索引失败: {e}。索引将为空，可能需要重建。")
                instance.faiss_index = None  # 确保失败时索引为空
                instance._init_faiss_index()  # 尝试重新初始化一个空索引结构
        else:
            logging.warning(
                f"FAISS 索引文件 '{index_file}' 未找到。索引将为空，需要重建。"
            )
            instance._init_faiss_index()  # 初始化一个空索引结构

        logging.info(f"SemanticMap 已从目录 '{directory_path}' 加载。")
        return instance


# --- 语义图谱 ---


class SemanticGraph:
    """
    语义图谱 (SemanticGraph) 结合了 SemanticMap 的向量存储/搜索能力和 NetworkX 的图结构管理能力。
    它存储记忆单元作为节点，并允许在它们之间定义显式的命名关系。
    查询可以利用显式图遍历和隐式语义相似性。
    """

    def __init__(self, semantic_map: SemanticMap = None):
        """
        初始化语义图谱。
        参数:
            semantic_map_instance (Optional[SemanticMap]): 一个 SemanticMap 实例。
                                                          如果为 None，将创建一个新的默认 SemanticMap。
        """
        self.semantic_map: SemanticMap = semantic_map if semantic_map else SemanticMap()
        self.nx_graph: nx.DiGraph = (
            nx.DiGraph()
        )  # 使用 NetworkX有向图存储节点和显式关系
        logging.info("SemanticGraph 已初始化。")

    def add_unit(
        self,
        unit: MemoryUnit,
        ms_names: Optional[List[str]] = None,
        rebuild_semantic_map_index_immediately: bool = False,
    ):
        """
        向图谱添加一个记忆单元 (节点)。
        单元也会被添加到内部的 SemanticMap 中。
        参数:
            unit (MemoryUnit): 要添加的记忆单元。
            explicit_content_for_embedding, content_type_for_embedding: 传递给 SemanticMap 用于嵌入生成。
            ms_names (Optional[List[str]]): 要将此单元添加到的 SemanticMap 中的记忆空间名称。
            rebuild_semantic_map_index_immediately (bool): 是否在添加后立即重建 SemanticMap 的 FAISS 索引。
        """
        # 1. 将单元添加到 SemanticMap
        self.semantic_map.add_unit(
            unit,
            ms_names,
            rebuild_index_immediately=rebuild_semantic_map_index_immediately,  # 注意这里传递的是否立即重建map索引
        )

        # 2. 将单元ID作为节点添加到 NetworkX 图中 (如果尚不存在)
        if self.nx_graph.has_node(unit.uid):
            raise ValueError(f"MemoryUnit {unit.uid} already exists in the graph.")
        # 可以在节点上存储来自 unit.value 的一些属性，如果需要的话
        self.nx_graph.add_node(
            unit.uid, {"raw_data": unit.raw_data, "metadata": unit.metadata}
        )
        logging.debug(f"节点 '{unit.uid}' 已添加到 NetworkX 图。")

    def remove_unit(self, uid: str, rebuild_index_immediately: bool = False):
        if self.nx_graph.has_node(uid):
            self.nx_graph.remove_node(uid)
            logging.debug(f"MemoryUnit {uid} has been removed from NetworkX graph.")
        else:
            logging.warning(f"MemoryUnit {uid} does not exist in NetworkX graph.")

        self.semantic_map.remove_unit(uid, rebuild_index_immediately)

    def update_unit(self, unit: MemoryUnit):
        if self.nx_graph.has_node(unit.uid):
            self.nx_graph.update(unit.uid)  # TODO: 未查验
            logging.debug(
                f"MemoryUnit {unit.uid} has been removed from NetworkX graph."
            )
        else:
            logging.warning(f"MemoryUnit {unit.uid} does not exist in NetworkX graph.")

        self.semantic_map.update_unit(unit)

    def get_unit(self, uid: str):
        if not self.nx_graph.has_node(uid):
            raise ValueError(f"MemoryUnit {uid} does not exist in NetworkX graph.")

        return self.semantic_map.get_unit(uid)

    def add_explicit_edge(
        self,
        src_uid: str,
        tgt_uid: str,
        rel_type: str,
        bidirectional: bool = False,
        metadata: Dict = None,
    ):  # 允许添加其他关系属性
        """
        在两个已存在的记忆单元 (节点) 之间添加一条显式关系 (边)。
        参数:
            source_unit_id (str): 源节点的ID。
            target_unit_id (str): 目标节点的ID。
            relationship_name (str): 关系的名称 (例如 "连接到", "依赖于", "父子")。
            bidirectional (bool): 如果为 True，则添加一条从 target 到 source 的具有相同名称的反向关系。
            **kwargs: 任何其他要存储为边属性的键值对。
        """
        if not self.semantic_map.get_unit(src_uid):
            raise ValueError(f"Source MemoryUnit {src_uid} does not exist.")
        if not self.semantic_map.get_unit(tgt_uid):
            raise ValueError(f"Target MemoryUnit {tgt_uid} does not exist.")

        # 确保节点也存在于nx_graph中 (通常 add_memory_unit 会处理)
        if not self.nx_graph.has_node(src_uid):
            self.nx_graph.add_node(src_uid)
        if not self.nx_graph.has_node(tgt_uid):
            self.nx_graph.add_node(tgt_uid)

        if not metadata:
            metadata = {}
        metadata["created"] = datetime.now()

        # 使用 relationship_name 作为边的 'type' 或 'label' 属性
        edge_attributes = {"type": rel_type, **metadata}
        self.nx_graph.add_edge(src_uid, tgt_uid, **edge_attributes)
        logging.info(f"已添加从 '{src_uid}' 到 '{tgt_uid}' 的关系 '{rel_type}'。")

        if bidirectional:
            self.nx_graph.add_edge(
                tgt_uid, src_uid, **edge_attributes
            )  # 注意：如果关系有方向性，反向关系可能需要不同名称/属性
            logging.info(
                f"已添加从 '{tgt_uid}' 到 '{src_uid}' 的双向关系 '{rel_type}'。"
            )

    def delete_relationship(
        self,
        source_unit_id: str,
        target_unit_id: str,
        relationship_name: Optional[str] = None,
    ):
        """
        删除两个节点之间的特定关系或所有关系。
        参数:
            source_unit_id (str): 源节点ID。
            target_unit_id (str): Target 节点ID。
            relationship_name (Optional[str]): 如果提供，则只删除具有此名称 (作为'type'属性) 的关系。
                                               否则，删除这两个节点之间的所有直接关系。
        """
        if not self.nx_graph.has_edge(source_unit_id, target_unit_id):
            logging.warning(
                f"节点 '{source_unit_id}' 和 '{target_unit_id}' 之间没有直接边。"
            )
            return

        if relationship_name:
            # NetworkX DiGraph 可以有平行边，但 add_edge 通常会替换。
            # 如果允许多个同名关系，则需要更复杂的删除逻辑。
            # 假设每个 (source, target) 对之间特定类型的关系是唯一的。
            edge_data = self.nx_graph.get_edge_data(source_unit_id, target_unit_id)
            # 对于有向图，通常只有一个直接边。如果有多条边（MultiDiGraph），则需要迭代。
            if edge_data and edge_data.get("type") == relationship_name:
                self.nx_graph.remove_edge(source_unit_id, target_unit_id)
                logging.info(
                    f"已删除从 '{source_unit_id}' 到 '{target_unit_id}' 的关系 '{relationship_name}'。"
                )
            else:
                logging.warning(
                    f"未找到从 '{source_unit_id}' 到 '{target_unit_id}' 的名为 '{relationship_name}' 的关系。"
                )
        else:  # 删除所有直接边
            self.nx_graph.remove_edge(source_unit_id, target_unit_id)
            logging.info(
                f"已删除从 '{source_unit_id}' 到 '{target_unit_id}' 的所有直接关系。"
            )

    def get_memory_unit_data(self, unit_id: str) -> Optional[MemoryUnit]:
        """从底层的 SemanticMap 检索记忆单元对象。"""
        return self.semantic_map.get_unit(unit_id)

    def build_index(self):
        """构建底层 SemanticMap 的 FAISS 索引。"""
        self.semantic_map.build_index()

    # --- 查询API ---
    def search_similarity_in_graph(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 5,
        ms_name: Optional[str] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        在图谱中执行语义相似性搜索 (委托给 SemanticMap)。
        参数:
            query_text (Optional[str]): 查询文本。
            query_vector (Optional[np.ndarray]): 查询向量。
            query_image_path (Optional[str]): 查询图像的路径。
            k (int): 返回结果数量。
            ms_name (Optional[str]): 限制在 SemanticMap 中的特定记忆空间内搜索。
            filter_unit_ids (Optional[Set[str]]): 进一步限制搜索范围的单元ID集合。
        返回:
            List[Tuple[MemoryUnit, float]]: (MemoryUnit, 相似度得分) 列表。
        """
        if (
            sum([x is not None for x in [query_text, query_vector, query_image_path]])
            != 1
        ):
            raise ValueError(
                "query_text,query_vector, query_image_path can only be one."
            )

        if query_vector is not None:
            return self.semantic_map.search_similarity_units_by_vector(
                query_vector,
                top_k,
                ms_name,
            )
        elif query_text is not None:
            return self.semantic_map.search_similarity_units_by_text(
                query_text,
                top_k,
                ms_name,
            )
        elif query_image_path is not None:
            return self.semantic_map.search_similarity_units_by_image(
                query_image_path,
                top_k,
                ms_name,
            )
        else:
            logging.warning(
                "必须提供 query_text, query_vector 或 query_image_path 之一进行相似性搜索。"
            )
            return []

    def get_explicit_neighbors(
        self,
        uid: str,
        rel_type: Optional[str] = None,  # 对应于边的 'type' 属性
        direction: str = "successors",  # "successors", "predecessors", or "all"
        ms_name: Optional[str] = None,
    ) -> List[MemoryUnit]:
        """
        遍历与给定节点通过显式关系连接的邻居节点。
        参数:
            unit_id (str): 起始节点的ID。
            relationship_type (Optional[str]): 要筛选的关系类型 (边属性 'type')。如果为 None，则不按类型筛选。
            direction (str): 遍历方向:
                             "successors" (默认): 查找 unit_id 指向的节点 (子节点/出边)。
                             "predecessors": 查找指向 unit_id 的节点 (父节点/入边)。
                             "all": 查找双向的邻居。
            ms_name (Optional[str]): 如果提供，则仅返回那些也存在于 SemanticMap 中指定记忆空间的邻居。
        返回:
            List[MemoryUnit]: 符合条件的邻居 MemoryUnit 对象列表。
        """
        if not self.nx_graph.has_node(uid):
            raise ValueError(f"MemoryUnit {uid} does not exist in the graph.")

        neighbor_ids: Set[str] = set()
        if direction == "successors":
            for successor in self.nx_graph.successors(uid):
                if rel_type:
                    edge_data = self.nx_graph.get_edge_data(uid, successor)
                    # 对于有向图，通常只有一个直接边。如果是MultiDiGraph，需要检查所有边。
                    # 假设默认的DiGraph，get_edge_data返回第一个找到的边的属性。
                    # 如果一个 (u,v) 对有多条不同类型的边，这个逻辑需要调整为检查所有边。
                    # 对于简单的DiGraph，如果 (u,v) 存在，则只有一条边。
                    if edge_data and edge_data.get("type") == rel_type:
                        neighbor_ids.add(successor)
                else:
                    neighbor_ids.add(successor)
        elif direction == "predecessors":
            for predecessor in self.nx_graph.predecessors(uid):
                if rel_type:
                    edge_data = self.nx_graph.get_edge_data(predecessor, uid)
                    if edge_data and edge_data.get("type") == rel_type:
                        neighbor_ids.add(predecessor)
                else:
                    neighbor_ids.add(predecessor)
        elif direction == "all":
            # 获取所有邻居 (包括前驱和后继)
            all_neighbors_temp = set(self.nx_graph.successors(uid))
            all_neighbors_temp.update(self.nx_graph.predecessors(uid))

            for neighbor in all_neighbors_temp:
                # 检查 (unit_id, neighbor) 或 (neighbor, unit_id) 的边
                passes_filter = False
                if not rel_type:
                    passes_filter = True
                else:
                    if (
                        self.nx_graph.has_edge(uid, neighbor)
                        and self.nx_graph.get_edge_data(uid, neighbor).get("type")
                        == rel_type
                    ):
                        passes_filter = True
                    elif (
                        self.nx_graph.has_edge(neighbor, uid)
                        and self.nx_graph.get_edge_data(neighbor, uid).get("type")
                        == rel_type
                    ):
                        passes_filter = True
                if passes_filter:
                    neighbor_ids.add(neighbor)
        else:
            logging.warning(
                f"无效的遍历方向: '{direction}'。应为 'successors', 'predecessors', 或 'all'。"
            )
            return []

        # 根据 ms_name 过滤 (如果提供)
        if ms_name:
            space = self.semantic_map.get_memory_space(ms_name)
            if space:
                space_unit_ids = space.get_all_units()
                neighbor_ids.intersection_update(space_unit_ids)  # 只保留也在空间内的ID
            else:
                logging.warning(f"记忆空间 '{ms_name}' 未找到，无法按空间过滤邻居。")
                return []  # 如果指定了空间但空间不存在，则不返回任何结果

        # 获取 MemoryUnit 对象
        results: List[MemoryUnit] = []
        for nid in neighbor_ids:
            uid = self.semantic_map.get_unit(nid)
            if uid:
                results.append(uid)
        return results

    def get_implicit_neighbors(
        self,
        uid: str,
        top_k: int = 5,
        ms_name: Optional[str] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        基于语义相似性查找与给定节点隐式相关的节点。
        参数:
            unit_id (str): 起始节点的ID。
            k (int): 要查找的相似邻居数量。
            ms_name (Optional[str]): 如果提供，则在 SemanticMap 中的指定记忆空间内限制搜索。
        返回:
            List[Tuple[MemoryUnit, float]]: (MemoryUnit, 相似度得分) 列表，不包括 unit_id 本身。
        """
        start_unit = self.semantic_map.get_unit(uid)
        if not start_unit or start_unit.embedding is None:
            raise ValueError("Unit not found or has no embedding")

        # 搜索时排除自身 (如果 SemanticMap 的搜索结果可能包含查询项本身)
        # k+1 然后过滤，或者让 SemanticMap 的搜索处理过滤（如果它支持的话）
        # 这里假设 search_similarity_by_vector 返回的结果不包含查询向量本身（除非它在数据集中且非常相似）
        # 通常，我们会获取 k+1 个结果，然后手动排除 unit_id。

        # 创建一个过滤器，排除 unit_id 本身
        # filter_ids_to_exclude_self = {
        #     uid for uid in self.semantic_map.memory_units.keys() if uid != uid
        # }

        # 如果指定了空间，则 filter_ids_to_exclude_self 会被 search_similarity_by_vector 内部的 ms_name 逻辑覆盖或合并。
        # 我们需要确保 unit_id 本身被排除。
        # 一个更简单的方法是获取k+1个结果，然后从结果中移除unit_id。

        similar_units_with_scores = self.semantic_map.search_similarity_units_by_vector(
            start_unit.embedding,
            top_k=top_k + 1,  # 获取稍多一些，以防 unit_id 是最相似的
            ms_name=ms_name,
        )

        results: List[Tuple[MemoryUnit, float]] = []
        for unit, score in similar_units_with_scores:
            if unit.uid != uid:  # 排除起始节点本身
                results.append((unit, score))
            if len(results) >= top_k:  # 如果已达到k个结果
                break
        return results

    def filter_memory_units(
        self,
        filter_condition: Dict,
        ms_names: List[str] = None,
    ) -> List[MemoryUnit]:
        """
        根据过滤器条件和记忆空间名称列表筛选记忆单元。
        :param filter_condition: 字典形式的过滤条件，key为字段名，value为过滤值。仅当raw_data中该字段值不等于过滤值时保留该单元。
        :param ms_names: 可选，指定只从这些记忆空间中检索。
        :return: 满足条件的MemoryUnit列表。
        """
        if not isinstance(filter_condition, dict):
            raise ValueError("filter_condition 必须为字典类型")

        # 获取候选单元
        candidate_units = set()
        if ms_names:
            for ms_name in ms_names:
                ms = self.semantic_map.get_memory_space(ms_name)
                candidate_units.update(ms.get_all_units())
        else:
            # 所有空间的所有单元
            for ms in self.semantic_map.get_all_memory_spaces():
                candidate_units.update(ms.get_all_units())
        # 过滤
        filtered_units = []
        for unit in candidate_units:
            keep = True
            for k, v in filter_condition.items():
                if unit.raw_data.get(k) == v:
                    keep = False
                    break
                if keep:
                    filtered_units.append(unit)
        return filtered_units

    def aggregate_results(
        self, memory_units: List[MemoryUnit]
    ) -> Dict[MemoryUnit, int]:
        counter = Counter(memory_units)
        return dict(counter)

    def path_search(
        self,
        start_uid: str,
        end_uid: str,
        max_depth: int = 3,
        rel_type: Optional[str] = None,
    ) -> List[List[str]]:
        """
        搜索起始节点和目标节点之间的所有路径（不超过最大深度），可选按关系类型过滤。
        参数:
            start_uid (str): 起始节点ID。
            end_uid (str): 目标节点ID。
            max_depth (int): 路径最大深度（包含起点和终点）。
            rel_type (Optional[str]): 只考虑指定类型的关系（边）。
        返回:
            List[List[str]]: 所有满足条件的路径，每条路径为节点ID列表。
        """
        if not self.nx_graph.has_node(start_uid):
            raise ValueError(f"起始节点 {start_uid} 不存在于图中。")
        if not self.nx_graph.has_node(end_uid):
            raise ValueError(f"目标节点 {end_uid} 不存在于图中。")
        if max_depth < 1:
            return []

        def edge_type_filter(path):
            if rel_type is None:
                return True
            for i in range(len(path) - 1):
                edge_data = self.nx_graph.get_edge_data(path[i], path[i + 1])
                if not edge_data or edge_data.get("type") != rel_type:
                    return False
            return True

        # 使用networkx的所有简单路径生成器
        all_paths = []
        try:
            for path in nx.all_simple_paths(
                self.nx_graph, source=start_uid, target=end_uid, cutoff=max_depth
            ):
                if edge_type_filter(path):
                    all_paths.append(path)
        except nx.NetworkXNoPath:
            return []
        return all_paths

    # --- MemorySpace 相关 (通过 SemanticMap 操作) ---
    def create_memory_space(self, ms_name: str) -> MemorySpace:
        """在底层的 SemanticMap 中创建或获取一个记忆空间。"""
        return self.semantic_map.create_memory_space(ms_name)

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
        except Exception as e:  # GML 可能不支持所有数据类型作为属性，pickle 更通用
            logging.warning(f"以 GML 格式保存 NetworkX 图失败: {e}。尝试使用 pickle。")
            nx_graph_file_pkl = os.path.join(directory_path, "semantic_graph.pkl")
            with open(nx_graph_file_pkl, "wb") as f:
                pickle.dump(self.nx_graph, f)
            logging.info(f"NetworkX 图已使用 pickle 保存到 '{nx_graph_file_pkl}'。")

        logging.info(f"SemanticGraph 已保存到目录: '{directory_path}'")

    @classmethod
    def load_graph(
        cls,
        directory_path: str,
        image_embedding_model_name: Optional[str] = None,
        text_embedding_model_name: Optional[str] = None,
    ) -> "SemanticGraph":
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
                    instance.nx_graph = nx.DiGraph()  # 创建一个空图
        elif os.path.exists(nx_graph_file_pkl):
            with open(nx_graph_file_pkl, "rb") as f:
                instance.nx_graph = pickle.load(f)
            logging.info(f"NetworkX 图已从 pickle 文件 '{nx_graph_file_pkl}' 加载。")
        else:
            logging.warning(
                f"NetworkX 图文件 (GML 或 pickle) 在 '{directory_path}' 中未找到。图将为空。"
            )
            instance.nx_graph = nx.DiGraph()

        logging.info(f"SemanticGraph 已从目录 '{directory_path}' 加载。")
        return instance

    def display_graph_summary(self):
        """打印图谱的摘要信息。"""
        num_map_units = len(self.semantic_map.memory_units)
        num_map_indexed = (
            self.semantic_map.faiss_index.ntotal if self.semantic_map.faiss_index else 0
        )
        num_map_spaces = len(self.semantic_map._memory_spaces)

        num_graph_nodes = self.nx_graph.number_of_nodes()
        num_graph_edges = self.nx_graph.number_of_edges()

        summary = (
            f"--- SemanticGraph 摘要 ---\n"
            f"SemanticMap:\n"
            f"  - 记忆单元总数: {num_map_units}\n"
            f"  - 已索引向量数: {num_map_indexed}\n"
            f"  - 记忆空间数: {num_map_spaces} ({list(self.semantic_map._memory_spaces.keys())})\n"
            f"NetworkX Graph:\n"
            f"  - 节点数: {num_graph_nodes}\n"
            f"  - 边数 (关系数): {num_graph_edges}\n"
            f"---------------------------\n"
        )
        print(summary)
        logging.info(summary.replace("\n", " | "))


if __name__ == "__main__":
    # --- 示例用法 ---
    logging.info("开始 Hippo.py 示例用法...")

    # 1. 创建 SemanticMap
    # 使用较小的模型进行快速测试 (如果可用) 或默认模型
    # text_model_for_test = "paraphrase-MiniLM-L6-v2" # 384 dim
    # image_model_for_test = "clip-ViT-B-32" # 512 dim, 但这里为了统一，假设都用一个维度
    # 如果模型维度不同，需要更复杂的处理或使用能输出相同维度的模型对

    # 为简单起见，假设使用默认模型和维度
    smap = SemanticMap()

    for ms_name in ["AI文档", "通用知识", "AI概念", "AI观察", "NLP相关"]:
        smap.create_memory_space(ms_name)

    # 2. 创建和添加 MemoryUnit
    unit1_text = "这是一个关于人工智能的文档。"
    unit1_val = {
        "description": "AI介绍",
        "type": "文档",
        "text_content": unit1_text,
        "author": "系统",
    }
    unit1 = MemoryUnit(uid="doc_ai_intro", raw_data=unit1_val)

    unit2_text = "机器学习是人工智能的一个分支。"
    unit2_val = {
        "description": "ML定义",
        "type": "概念",
        "text_content": unit2_text,
        "field": "AI",
    }
    unit2 = MemoryUnit(uid="concept_ml", raw_data=unit2_val)

    unit3_text = "深度学习推动了自然语言处理的进步。"
    unit3_val = {
        "description": "DL对NLP的影响",
        "type": "观察",
        "text_content": unit3_text,
    }
    unit3 = MemoryUnit(uid="obs_dl_nlp", raw_data=unit3_val)

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
    #     semantic_map.add_memory_unit(unit_img1, ms_names=["图像空间"])

    # 添加文本单元到 map，并指定内容进行嵌入
    smap.add_unit(
        unit=unit1,
        ms_names=["AI文档", "通用知识"],
    )
    smap.add_unit(
        unit=unit2,
        ms_names=["AI概念"],
    )
    smap.add_unit(
        unit3, ms_names=["AI观察", "NLP相关"]
    )  # 让map从unit.value推断嵌入内容

    # 3. 构建 SemanticMap 索引
    smap.build_index()

    # 4. 在 SemanticMap 中进行相似性搜索
    logging.info("\n--- SemanticMap 相似性搜索 (全局) ---")
    query1 = "什么是机器学习？"
    similar_results_map = smap.search_similarity_units_by_text(
        query_text=query1, top_k=2
    )
    for res_unit, score in similar_results_map:
        logging.info(
            f"找到单元: {res_unit.uid}, 值: {res_unit.raw_data}, 得分: {score:.4f}"
        )

    logging.info("\n--- SemanticMap 相似性搜索 (在 'AI文档' 空间) ---")
    similar_results_map_space = smap.search_similarity_units_by_text(
        query_text="AI", top_k=1, ms_name="AI文档"
    )
    for res_unit, score in similar_results_map:
        logging.info(
            f"找到单元: {res_unit.uid}, 值: {res_unit.raw_data}, 得分: {score:.4f}"
        )

    # 5. 创建 SemanticGraph (使用已有的 semantic_map)
    semantic_graph = SemanticGraph(semantic_map=smap)

    # 6. 在 SemanticGraph 中添加单元 (它们已在map中，这里主要是为了在图中创建节点)
    # 通常，如果单元是新的，会通过 graph.add_memory_unit 添加
    # 如果单元已在map中，我们只需确保它们作为节点存在于图中
    if not semantic_graph.nx_graph.has_node(unit1.uid):
        semantic_graph.nx_graph.add_node(unit1.uid, **unit1.raw_data)
    if not semantic_graph.nx_graph.has_node(unit2.uid):
        semantic_graph.nx_graph.add_node(unit2.uid, **unit2.raw_data)
    if not semantic_graph.nx_graph.has_node(unit3.uid):
        semantic_graph.nx_graph.add_node(unit3.uid, **unit3.raw_data)
    # if unit_img1 and not semantic_graph.nx_graph.has_node(unit_img1.id): semantic_graph.nx_graph.add_node(unit_img1.id, **unit_img1.value)

    # 7. 在 SemanticGraph 中添加关系
    semantic_graph.add_explicit_edge(
        unit1.uid, unit2.uid, rel_type="包含主题", metadata={"relevance": 0.9}
    )
    semantic_graph.add_explicit_edge(
        unit2.uid, unit3.uid, rel_type="相关概念", bidirectional=True
    )

    # 8. 在 SemanticGraph 中进行查询
    logging.info("\n--- SemanticGraph 显式遍历 ('包含主题' 的子节点) ---")
    explicit_neighbors = semantic_graph.get_explicit_neighbors(
        uid=unit1.uid, rel_type="包含主题", direction="successors"
    )
    for neighbor_unit in explicit_neighbors:
        logging.info(
            f"'{unit1.uid}' 的 '{'包含主题'}' 子节点: {neighbor_unit.uid} - {neighbor_unit.raw_data.get('description')}"
        )

    logging.info("\n--- SemanticGraph 隐式遍历 (与 unit1 语义相似的节点) ---")
    implicit_neighbors = semantic_graph.get_implicit_neighbors(uid=unit1.uid, top_k=2)
    for neighbor_unit, score in implicit_neighbors:
        logging.info(
            f"与 '{unit1.uid}' 语义相似的节点: {neighbor_unit.uid} - {neighbor_unit.raw_data.get('description')}, 得分: {score:.4f}"
        )

    logging.info(
        "\n--- SemanticGraph 隐式遍历 (与 unit1 语义相似的节点, 在 'AI概念' 空间) ---"
    )
    implicit_neighbors_space = semantic_graph.get_implicit_neighbors(
        uid=unit1.uid, top_k=1, ms_name="AI概念"
    )
    for neighbor_unit, score in implicit_neighbors_space:
        logging.info(
            f"与 '{unit1.uid}' 语义相似 (在 'AI概念' 空间): {neighbor_unit.uid} - {neighbor_unit.raw_data.get('description')}, 得分: {score:.4f}"
        )

    # 9. 显示图谱摘要
    semantic_graph.display_graph_summary()

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
        loaded_implicit_neighbors = loaded_graph.get_implicit_neighbors(
            uid=unit1.uid, top_k=1
        )
        for neighbor_unit, score in loaded_implicit_neighbors:
            logging.info(
                f"与 '{unit1.uid}' 语义相似的节点: {neighbor_unit.uid} - {neighbor_unit.raw_data.get('description')}, 得分: {score:.4f}"
            )

    except FileNotFoundError as e:
        logging.error(f"加载失败，因为保存目录或文件未完全创建/找到: {e}")
    except Exception as e:
        logging.error(f"加载或测试加载的图谱时发生错误: {e}", exc_info=True)

    # 清理虚拟图像
    # if dummy_image_path and dummy_image_path == "dummy_image.jpg" and os.path.exists("dummy_image.jpg"):
    #     os.remove("dummy_image.jpg")

    logging.info("\nHippo.py 示例用法结束。")
