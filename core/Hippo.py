import os
import pickle
import logging

from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx
from collections import Counter
import random

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
    记忆空间 (MemorySpace) 支持嵌套结构，可包含 MemoryUnit 或其他 MemorySpace。
    用户可直接操作 MemorySpace，实现树状/嵌套组织。
    支持局部faiss索引。
    """

    def __init__(self, ms_name: str):
        if not isinstance(ms_name, str) or not ms_name.strip():
            raise ValueError("记忆空间名称不能为空")
        self.name: str = ms_name
        self._members: Dict[str, Union[MemoryUnit, "MemorySpace"]] = {}
        self._emb_index = None
        self._index_to_unit: Dict[int, MemoryUnit] = {}

    def __str__(self) -> str:
        return f"MemorySpace(name={self.name}, member_count={len(self._members)})"

    def __repr__(self):
        return f"MemorySpace({self.name})"

    def add(self, member: Union[MemoryUnit, "MemorySpace"]):
        if isinstance(member, MemoryUnit):
            if member.uid in self._members:
                raise ValueError(
                    f"Duplicate MemoryUnit {member.uid} in MemorySpace {self.name}"
                )
            self._members[member.uid] = member
        elif isinstance(member, MemorySpace):
            if member.name in self._members:
                raise ValueError(
                    f"Duplicate MemorySpace {member.name} in MemorySpace {self.name}"
                )
            if member is self:
                raise ValueError("不能将自身作为子空间加入")
            self._members[member.name] = member
        else:
            raise TypeError("Only MemoryUnit or MemorySpace can be added.")

    def remove(self, key: str):
        if key not in self._members:
            raise ValueError(f"{key} not found in MemorySpace {self.name}")
        self._members.pop(key)

    def get(self, key: str) -> Union[MemoryUnit, "MemorySpace", None]:
        return self._members.get(key, None)

    def get_all_units(self) -> List[MemoryUnit]:
        """递归获取此空间及所有子空间中的所有 MemoryUnit。"""
        result = []
        for m in self._members.values():
            if isinstance(m, MemoryUnit):
                result.append(m)
            elif isinstance(m, MemorySpace):
                result.extend(m.get_all_units())
        return result

    def get_all_spaces(self) -> List["MemorySpace"]:
        """递归获取所有子 MemorySpace（不含自身）。"""
        spaces = []
        for m in self._members.values():
            if isinstance(m, MemorySpace):
                spaces.append(m)
                spaces.extend(m.get_all_spaces())
        return spaces

    def list_members(self) -> List[str]:
        """返回所有成员（unit或space）的key列表。"""
        return list(self._members.keys())

    def build_index(self, embedding_dim: int = 512):
        """递归收集所有unit并构建局部faiss索引。"""
        from faiss import IndexFlatL2

        self._emb_index = IndexFlatL2(embedding_dim)
        self._index_to_unit = {}
        embeddings = []
        count = 0
        for unit in self.get_all_units():
            if unit.embedding is not None:
                embeddings.append(unit.embedding)
                self._index_to_unit[count] = unit
                count += 1
        if not embeddings:
            logging.warning(
                f"MemorySpace {self.name} has no valid embeddings to build index."
            )
            return
        import numpy as np

        self._emb_index.add(np.array(embeddings, dtype=np.float32))

    def search_similarity_units_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[MemoryUnit, float]]:
        if not self._emb_index or self._emb_index.ntotal == 0:
            raise ValueError(
                f"Index is empty in MemorySpace {self.name}. Please build index first."
            )
        query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
        D, I = self._emb_index.search(query_vector_np, top_k)
        results = []
        for i in range(len(I[0])):
            idx = int(I[0][i])
            if I[0][i] == -1:
                continue
            unit = self._index_to_unit[idx]
            results.append((unit, D[0][i]))
        return results

    # 可根据需要扩展更多接口，如递归查找、移动成员等

    def save(self, file_path: str):
        """将当前MemorySpace对象保存到指定文件"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "MemorySpace":
        """从指定文件加载MemorySpace对象"""
        with open(file_path, "rb") as f:
            return pickle.load(f)


class SemanticMap:
    """
    语义地图 (SemanticMap) 只负责存储所有不重复的 MemoryUnit，并支持基于相似度的搜索。
    MemorySpace 由用户独立管理，支持嵌套。
    """

    DEFAULT_TEXT_EMBEDDING_KEY = "text_content"
    DEFAULT_IMAGE_EMBEDDING_KEY = "image_path"

    def __init__(
        self,
        image_embedding_model_name: str = "clip-ViT-B-32",
        text_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        update_interval: Optional[int] = 100,
    ):
        self._units: Dict[str, MemoryUnit] = {}  # 全局唯一unit
        self._emb_index = None
        self._index_to_unit: Dict[int, MemoryUnit] = {}
        if update_interval:
            self._update_interval = update_interval
            self._operation_counter = 0
        try:
            self.text_model = SentenceTransformer(text_embedding_model_name)
            self.image_model = SentenceTransformer(image_embedding_model_name)
        except FileNotFoundError or ValueError as e:
            raise e
        logging.info(f"SemanticMap 已初始化。")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            logging.warning("尝试为无效文本生成嵌入。")
            raise ValueError("Invalid text for embedding.")
        try:
            emb = self.text_model.encode(text)
            if hasattr(emb, "cpu") and hasattr(emb, "numpy"):
                emb = emb.cpu().numpy()
            return emb.astype(np.float32)
        except Exception as e:
            logging.error(f"生成文本嵌入失败: '{text[:50]}...' - {e}")
            raise e

    def _get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
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

    def register_unit(self, unit: MemoryUnit, update_embedding: bool = True):
        if not isinstance(unit, MemoryUnit):
            raise TypeError(f"{unit} is not a MemoryUnit.")
        if update_embedding:
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
        self._units[unit.uid] = unit
        if hasattr(self, "_update_interval"):
            self._operation_counter += 1
            if self._operation_counter >= self._update_interval:
                self.build_index()
                self._operation_counter = 0

    def unregister_unit(self, uid: str):
        if uid in self._units:
            self._units.pop(uid)
            if hasattr(self, "_update_interval"):
                self._operation_counter += 1
                if self._operation_counter >= self._update_interval:
                    self.build_index()
                    self._operation_counter = 0

    def register_units_from_space(
        self, ms: "MemorySpace", update_embedding: bool = True
    ):
        for unit in ms.get_all_units():
            self.register_unit(unit, update_embedding=update_embedding)

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        return self._units.get(uid, None)

    def get_all_units(self) -> List[MemoryUnit]:
        return list(self._units.values())

    def build_index(self):
        """根据所有unit的embedding重建索引。"""
        from faiss import IndexFlatL2

        self._emb_index = IndexFlatL2(512)
        self._index_to_unit = {}
        embeddings = []
        count = 0
        for unit in self._units.values():
            if unit.embedding is not None:
                embeddings.append(unit.embedding)
                self._index_to_unit[count] = unit
                count += 1
        if not embeddings:
            logging.warning("SemanticMap has no valid embeddings to build index.")
            return
        import numpy as np

        self._emb_index.add(np.array(embeddings, dtype=np.float32))

    def search_similarity_units_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[MemoryUnit, float]]:
        if not self._emb_index or self._emb_index.ntotal == 0:
            raise ValueError("Index is empty. Please build index first.")
        query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
        D, I = self._emb_index.search(query_vector_np, top_k)
        results = []
        for i in range(len(I[0])):
            idx = int(I[0][i])
            if I[0][i] == -1:
                continue
            unit = self._index_to_unit[idx]
            results.append((unit, D[0][i]))
        return results

    def search_similarity_units_by_text(
        self, query_text: str, top_k: int = 5
    ) -> List[Tuple[MemoryUnit, float]]:
        if not isinstance(query_text, str):
            try:
                query_text = str(query_text)
            except Exception as e:
                logging.error(f"Invalid query_text: {query_text} - {e}")
                raise ValueError("Invalid query_text for embedding.")
        query_vector = self._get_text_embedding(query_text)
        if query_vector is None:
            return []
        return self.search_similarity_units_by_vector(query_vector, top_k)

    def search_similarity_units_by_image(
        self, image_path: str, top_k: int = 5
    ) -> List[Tuple[MemoryUnit, float]]:
        query_vector = self._get_image_embedding(image_path)
        if query_vector is None:
            return []
        return self.search_similarity_units_by_vector(query_vector, top_k)

    # --- 持久化 ---
    def save_map(self, dir_path: str):
        with open(os.path.join(dir_path, "semantic_map.pkl"), "wb") as f:
            pickle.dump(self, f)
        logging.info(f"SemanticMap 已保存到 {dir_path}。")

    @classmethod
    def load_map(cls, file_path: str) -> "SemanticMap":
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "rb") as f:
            loaded_map = pickle.load(f)
        logging.info(f"SemanticMap 已从 {file_path} 加载。")
        # 加载后自动重建索引（如果有unit）
        if hasattr(loaded_map, "build_index"):
            loaded_map.build_index()
        return loaded_map


class SemanticGraph:
    """
    语义图谱 (SemanticGraph) 结合了 SemanticMap 的向量存储/搜索能力和 NetworkX 的图结构管理能力。
    它存储记忆单元作为节点，并允许在它们之间定义显式的命名关系。
    查询可以利用显式图遍历和隐式语义相似性。
    """

    def __init__(
        self,
        semantic_map: SemanticMap = None,
        memory_spaces: Optional[List[MemorySpace]] = None,
    ):
        self.semantic_map: SemanticMap = semantic_map if semantic_map else SemanticMap()
        self.nx_graph: nx.DiGraph = nx.DiGraph()
        self.rel_types: Set[str] = set()
        self.memory_spaces: List[MemorySpace] = (
            memory_spaces or []
        )  # 用户主动维护的所有ms列表
        logging.info("SemanticGraph 已初始化。")

    def add_unit(
        self, unit: MemoryUnit, rebuild_semantic_map_index_immediately: bool = False
    ):
        """
        向图谱添加一个记忆单元 (节点)，并注册到 SemanticMap。
        """
        self.semantic_map.register_unit(unit)
        if rebuild_semantic_map_index_immediately:
            self.semantic_map.build_index()
        if self.nx_graph.has_node(unit.uid):
            logging.warning(
                f"MemoryUnit {unit.uid} already exists in the graph, skip add."
            )
            return
        self.nx_graph.add_node(unit.uid, raw_data=unit.raw_data, metadata=unit.metadata)
        logging.debug(f"节点 '{unit.uid}' 已添加到 NetworkX 图。")

    def remove_unit(self, uid: str, rebuild_index_immediately: bool = False):
        if self.nx_graph.has_node(uid):
            self.nx_graph.remove_node(uid)
            logging.debug(f"MemoryUnit {uid} has been removed from NetworkX graph.")
        else:
            logging.warning(f"MemoryUnit {uid} does not exist in NetworkX graph.")
        self.semantic_map.unregister_unit(uid)
        if rebuild_index_immediately:
            self.semantic_map.build_index()

    def update_unit(self, unit: MemoryUnit):
        if self.nx_graph.has_node(unit.uid):
            # 可根据需要更新节点属性
            self.nx_graph.nodes[unit.uid]["raw_data"] = unit.raw_data
            self.nx_graph.nodes[unit.uid]["metadata"] = unit.metadata
            logging.debug(f"MemoryUnit {unit.uid} 节点属性已更新。")
        else:
            logging.warning(f"MemoryUnit {unit.uid} does not exist in NetworkX graph.")
        self.semantic_map.register_unit(unit)  # 重新注册会覆盖

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        """
        获取指定uid的MemoryUnit。如果nx_graph中有该节点，则返回semantic_map中的unit，否则返回None。
        """
        if self.nx_graph.has_node(uid):
            return self.semantic_map.get_unit(uid)
        return None

    def add_explicit_edge(
        self,
        src_uid: str,
        tgt_uid: str,
        rel_type: str,
        bidirectional: bool = False,
        metadata: Dict = None,
    ):
        if not self.semantic_map.get_unit(src_uid):
            raise ValueError(f"Source MemoryUnit {src_uid} does not exist.")
        if not self.semantic_map.get_unit(tgt_uid):
            raise ValueError(f"Target MemoryUnit {tgt_uid} does not exist.")
        if not self.nx_graph.has_node(src_uid):
            self.nx_graph.add_node(src_uid)
        if not self.nx_graph.has_node(tgt_uid):
            self.nx_graph.add_node(tgt_uid)
        if not metadata:
            metadata = {}
        metadata["created"] = datetime.now()
        edge_attributes = {"type": rel_type, **metadata}
        self.nx_graph.add_edge(src_uid, tgt_uid, **edge_attributes)
        logging.info(f"已添加从 '{src_uid}' 到 '{tgt_uid}' 的关系 '{rel_type}'。")
        if bidirectional:
            self.nx_graph.add_edge(tgt_uid, src_uid, **edge_attributes)
            logging.info(
                f"已添加从 '{tgt_uid}' 到 '{src_uid}' 的双向关系 '{rel_type}'。"
            )
        self.rel_types.add(rel_type)

    def delete_explicit_edge(
        self,
        source_unit_id: str,
        target_unit_id: str,
        relationship_name: Optional[str] = None,
    ):
        if not self.nx_graph.has_edge(source_unit_id, target_unit_id):
            logging.warning(
                f"节点 '{source_unit_id}' 和 '{target_unit_id}' 之间没有直接边。"
            )
            return
        if relationship_name:
            edge_data = self.nx_graph.get_edge_data(source_unit_id, target_unit_id)
            if edge_data and edge_data.get("type") == relationship_name:
                self.nx_graph.remove_edge(source_unit_id, target_unit_id)
                logging.info(
                    f"已删除从 '{source_unit_id}' 到 '{target_unit_id}' 的关系 '{relationship_name}'。"
                )
            else:
                logging.warning(
                    f"未找到从 '{source_unit_id}' 到 '{target_unit_id}' 的名为 '{relationship_name}' 的关系。"
                )
        else:
            self.nx_graph.remove_edge(source_unit_id, target_unit_id)
            logging.info(
                f"已删除从 '{source_unit_id}' 到 '{target_unit_id}' 的所有直接关系。"
            )

    def get_all_units(self) -> List[MemoryUnit]:
        return self.semantic_map.get_all_units()

    def build_index(self):
        """重建语义图谱的全局向量索引（调用SemanticMap的build_index）。"""
        self.semantic_map.build_index()
        logging.info(
            "SemanticGraph: 全局向量索引已重建（调用SemanticMap.build_index）。"
        )

    def get_all_relations(self) -> list:
        """
        返回所有已注册的显式关系类型列表。
        """
        return list(self.rel_types)

    # --- 持久化 ---
    def save_graph(self, directory_path: str, ms_root: "MemorySpace" = None):
        """
        将 SemanticGraph 的状态保存到指定目录。
        包括 SemanticMap 的数据、NetworkX 图结构、关系类型和可选的MemorySpace根节点。
        """
        os.makedirs(directory_path, exist_ok=True)
        self.semantic_map.save_map(directory_path)
        with open(os.path.join(directory_path, "semantic_graph.pkl"), "wb") as f:
            pickle.dump(self.nx_graph, f)
        with open(os.path.join(directory_path, "rel_types.pkl"), "wb") as f:
            pickle.dump(self.rel_types, f)
        if ms_root is not None:
            with open(os.path.join(directory_path, "memory_space.pkl"), "wb") as f:
                pickle.dump(ms_root, f)
        logging.info(f"SemanticGraph 已保存到 {directory_path}")

    @classmethod
    def load_graph(cls, directory_path: str):
        """
        从指定目录加载 SemanticGraph 的状态。
        包括 SemanticMap 的数据、NetworkX 图结构、关系类型和可选的MemorySpace根节点。
        返回: (SemanticGraph实例, ms_root)
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist.")
        semantic_map = SemanticMap.load_map(
            os.path.join(directory_path, "semantic_map.pkl")
        )
        graph_file = os.path.join(directory_path, "semantic_graph.pkl")
        if not os.path.isfile(graph_file):
            raise ValueError(f"Graph file {graph_file} does not exist.")
        with open(graph_file, "rb") as f:
            nx_graph = pickle.load(f)
        rel_types_file = os.path.join(directory_path, "rel_types.pkl")
        if not os.path.isfile(rel_types_file):
            raise ValueError(f"Rel types file {rel_types_file} does not exist.")
        with open(rel_types_file, "rb") as f:
            rel_types = pickle.load(f)
        ms_root = None
        ms_path = os.path.join(directory_path, "memory_space.pkl")
        if os.path.isfile(ms_path):
            with open(ms_path, "rb") as f:
                ms_root = pickle.load(f)
            # 递归注册所有unit到semantic_map
            if ms_root is not None:
                semantic_map.register_units_from_space(ms_root, update_embedding=False)
                semantic_map.build_index()
        # 补全nx_graph节点（如果有unit但节点缺失）
        for unit in semantic_map.get_all_units():
            if not nx_graph.has_node(unit.uid):
                nx_graph.add_node(
                    unit.uid, raw_data=unit.raw_data, metadata=unit.metadata
                )
        graph_instance = cls(semantic_map)
        graph_instance.nx_graph = nx_graph
        graph_instance.rel_types = rel_types
        logging.info(
            f"SemanticGraph 已从 '{directory_path}' 加载，并已注册所有unit与重建索引。"
        )
        return graph_instance, ms_root

    def display_graph_summary(self):
        """打印图谱的摘要信息。"""
        num_map_units = len(self.semantic_map.get_all_units())
        num_map_indexed = 0
        emb_index = getattr(self.semantic_map, "_emb_index", None)
        if emb_index is not None and hasattr(emb_index, "ntotal"):
            num_map_indexed = emb_index.ntotal
        # 记忆空间数无法直接统计，略去
        num_graph_nodes = self.nx_graph.number_of_nodes()
        num_graph_edges = self.nx_graph.number_of_edges()

        summary = (
            f"--- SemanticGraph 摘要 ---\n"
            f"SemanticMap:\n"
            f"  - 记忆单元总数: {num_map_units}\n"
            f"  - 已索引向量数: {num_map_indexed}\n"
            f"NetworkX Graph:\n"
            f"  - 节点数: {num_graph_nodes}\n"
            f"  - 边数 (关系数): {num_graph_edges}\n"
            f"---------------------------\n"
        )
        print(summary)
        logging.info(summary.replace("\n", " | "))

    def get_all_memory_space_names(self) -> List[str]:
        """
        获取所有MemorySpace名称（递归，去重）。
        """
        result = set()

        def _collect_names(ms):
            result.add(ms.name)
            for m in ms._members.values():
                if isinstance(m, MemorySpace):
                    _collect_names(m)

        for ms in self.memory_spaces:
            _collect_names(ms)
        return list(result)

    def get_memory_space_structures(self) -> List[dict]:
        """
        递归导出所有MemorySpace嵌套结构（树/嵌套dict），
        每个ms展示：名称、unit uid列表、所有unit的raw_data字段全集、子空间。
        返回列表，每个元素为一个ms的结构。
        """

        def _struct(ms):
            # 收集本ms下所有unit的uid和raw_data字段全集
            unit_uids = []
            unit_fields = set()
            for k, v in ms._members.items():
                if isinstance(v, MemoryUnit):
                    unit_uids.append(k)
                    unit_fields.update(v.raw_data.keys())
            children = [v for v in ms._members.values() if isinstance(v, MemorySpace)]
            d = {
                "name": ms.name,
                "unit_uids": unit_uids,
                "unit_fields": sorted(list(unit_fields)),
            }
            if children:
                d["children"] = [_struct(child) for child in children]
            return d

        return [_struct(ms) for ms in self.memory_spaces]

    def get_units_in_memory_space(self, ms_names, recursive=True) -> List[MemoryUnit]:
        """
        获取指定ms_names下所有unit，支持递归，支持多ms，自动去重。
        """
        found = []

        def _find_ms(ms, name):
            if ms.name == name:
                return ms
            for m in ms._members.values():
                if isinstance(m, MemorySpace):
                    res = _find_ms(m, name)
                    if res:
                        return res
            return None

        for name in ms_names:
            for ms in self.memory_spaces:
                target = _find_ms(ms, name)
                if target:
                    if recursive:
                        found.extend(target.get_all_units())
                    else:
                        found.extend(
                            [
                                v
                                for v in target._members.values()
                                if isinstance(v, MemoryUnit)
                            ]
                        )
        # 去重
        seen = set()
        result = []
        for u in found:
            if u.uid not in seen:
                seen.add(u.uid)
                result.append(u)
        return result

    # TODO: 可以理解为并集，那么MemorySpace是否需要其他操作，例如：交集、差集等？
    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        seen = set()
        result = []
        for u in units:
            if u.uid not in seen:
                seen.add(u.uid)
                result.append(u)
        return result

    # 现有API递归参数适配示例
    def filter_memory_units(
        self, candidate_units=None, filter_condition=None, ms_names=None, recursive=True
    ):
        if candidate_units is None:
            if ms_names:
                candidate_units = self.get_units_in_memory_space(
                    ms_names, recursive=recursive
                )
            else:
                candidate_units = self.semantic_map.get_all_units()

        def match(unit):
            if not filter_condition:
                return True
            for field, cond in filter_condition.items():
                # 优先从unit对象本身取字段（如uid），否则从raw_data取
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
                    if op == "contain" and not (op_val in val):
                        return False
                    if op == "not_contain" and not (op_val not in val):
                        return False
            return True

        return [u for u in candidate_units if match(u)]

    def search_similarity_in_graph(
        self, query_text, top_k=5, ms_names=None, recursive=True
    ):
        if ms_names:
            units = self.get_units_in_memory_space(ms_names, recursive=recursive)
        else:
            units = self.semantic_map.get_all_units()
        if not units:
            return []
        query_vec = self.semantic_map._get_text_embedding(query_text)
        arr = np.array(
            [u.embedding for u in units if u.embedding is not None], dtype=np.float32
        )
        if arr.shape[0] == 0:
            return []
        from faiss import IndexFlatL2

        index = IndexFlatL2(arr.shape[1])
        index.add(arr)
        D, I = index.search(
            query_vec.reshape(1, -1).astype(np.float32), min(top_k, arr.shape[0])
        )
        results = []
        for i in range(len(I[0])):
            idx = int(I[0][i])
            if I[0][i] == -1:
                continue
            results.append((units[idx], D[0][i]))
        return results

    def get_explicit_neighbors(
        self, uids, rel_type=None, ms_names=None, direction="successors", recursive=True
    ):
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

    def get_implicit_neighbors(self, uid, top_k=5, ms_names=None, recursive=True):
        unit = self.semantic_map.get_unit(uid)
        if not unit or unit.embedding is None:
            return []
        if ms_names:
            units = self.get_units_in_memory_space(ms_names, recursive=recursive)
        else:
            units = self.semantic_map.get_all_units()
        arr = np.array(
            [u.embedding for u in units if u.embedding is not None], dtype=np.float32
        )
        if arr.shape[0] == 0:
            return []
        from faiss import IndexFlatL2

        index = IndexFlatL2(arr.shape[1])
        index.add(arr)
        D, I = index.search(
            unit.embedding.reshape(1, -1).astype(np.float32), min(top_k, arr.shape[0])
        )
        results = []
        for i in range(len(I[0])):
            idx = int(I[0][i])
            if I[0][i] == -1:
                continue
            results.append(units[idx])
        return results

    # ...existing code...


if __name__ == "__main__":
    logging.info("开始 Hippo.py 示例用法（MemorySpace 独立与嵌套演示）...")

    # 1. 构建嵌套 MemorySpace 结构
    ms_root = MemorySpace("知识库")
    ms_ai = MemorySpace("AI文档")
    ms_nlp = MemorySpace("NLP相关")
    ms_concept = MemorySpace("AI概念")
    ms_paper = MemorySpace("论文集")
    ms_root.add(ms_ai)
    ms_root.add(ms_nlp)
    ms_ai.add(ms_concept)
    ms_ai.add(ms_paper)

    # 2. 创建更多 MemoryUnit 并主动添加到不同 MemorySpace
    unit1 = MemoryUnit(
        uid="doc_ai_intro",
        raw_data={
            "description": "AI介绍",
            "type": "文档",
            "text_content": "这是一个关于人工智能的文档。",
            "author": "系统",
        },
    )
    unit2 = MemoryUnit(
        uid="concept_ml",
        raw_data={
            "description": "ML定义",
            "type": "概念",
            "text_content": "机器学习是人工智能的一个分支。",
            "field": "AI",
        },
    )
    unit3 = MemoryUnit(
        uid="obs_dl_nlp",
        raw_data={
            "description": "DL对NLP的影响",
            "type": "观察",
            "text_content": "深度学习推动了自然语言处理的进步。",
        },
    )
    unit4 = MemoryUnit(
        uid="doc_nlp_intro",
        raw_data={
            "description": "NLP介绍",
            "type": "文档",
            "text_content": "自然语言处理是AI的重要领域。",
        },
    )
    unit5 = MemoryUnit(
        uid="concept_dl",
        raw_data={
            "description": "DL定义",
            "type": "概念",
            "text_content": "深度学习是机器学习的一个分支。",
        },
    )
    unit6 = MemoryUnit(
        uid="paper_transformer",
        raw_data={
            "description": "Transformer论文",
            "type": "论文",
            "text_content": "Attention is All You Need.",
        },
    )
    unit7 = MemoryUnit(
        uid="paper_bert",
        raw_data={
            "description": "BERT论文",
            "type": "论文",
            "text_content": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.",
        },
    )
    unit8 = MemoryUnit(
        uid="obs_ai_trend",
        raw_data={
            "description": "AI发展趋势",
            "type": "观察",
            "text_content": "AI正在快速发展，应用领域不断扩展。",
        },
    )

    ms_ai.add(unit1)
    ms_concept.add(unit2)
    ms_nlp.add(unit3)
    ms_nlp.add(unit4)
    ms_concept.add(unit5)
    ms_paper.add(unit6)
    ms_paper.add(unit7)
    ms_root.add(unit8)

    # 3. 递归收集所有unit
    all_units = ms_root.get_all_units()
    logging.info(f"ms_root递归收集到的所有unit: {[u.uid for u in all_units]}")

    # 输出各个记忆空间包含的unit（含有关系）
    def print_space_units(space: MemorySpace, indent: int = 0):
        prefix = "  " * indent
        units = [k for k, v in space._members.items() if isinstance(v, MemoryUnit)]
        spaces = [v for v in space._members.values() if isinstance(v, MemorySpace)]
        logging.info(f"{prefix}MemorySpace '{space.name}' 包含 MemoryUnit: {units}")
        for subspace in spaces:
            print_space_units(subspace, indent + 1)

    print_space_units(ms_root)

    # 4. 构建局部索引并做局部相似性检索
    # 先为所有unit生成embedding（用SemanticMap的接口）
    smap = SemanticMap(
        image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
        text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
    )

    for u in all_units:
        smap.register_unit(u)  # 生成embedding
    # 再构建ms_root的局部索引
    ms_root.build_index()
    # 用文本做一次局部相似性检索
    query = "人工智能的分支有哪些？"
    query_vec = smap._get_text_embedding(query)
    local_results = ms_root.search_similarity_units_by_vector(query_vec, top_k=3)
    logging.info("ms_root 局部相似性检索结果：")
    for unit, score in local_results:
        logging.info(
            f"unit: {unit.uid}, desc: {unit.raw_data.get('description')}, 距离: {score:.4f}"
        )

    # 5. 用户可自由组合/嵌套ms，灵活组织知识
    ms_sub = MemorySpace("AI子空间")
    ms_sub.add(unit1)
    ms_sub.add(unit5)
    ms_root.add(ms_sub)
    logging.info(f"ms_root成员: {ms_root.list_members()}")
    logging.info(f"ms_sub成员: {ms_sub.list_members()}")

    # 6. 展示 MemoryUnit、MemorySpace、SemanticMap、SemanticGraph 之间的关系
    logging.info("------ 结构关系展示 ------")
    logging.info(
        f"MemorySpace（知识库）包含的所有子空间: {[s.name for s in ms_root.get_all_spaces()]}"
    )
    logging.info(
        f"MemorySpace（知识库）递归包含的所有unit: {[u.uid for u in ms_root.get_all_units()]}"
    )
    logging.info(f"SemanticMap 全局unit数: {len(smap.get_all_units())}")

    # 构建语义图谱，将所有unit注册为节点，并添加部分关系
    sgraph = SemanticGraph(smap)
    for u in all_units:
        try:
            sgraph.add_unit(u)
        except ValueError:
            pass  # 已存在则跳过
    # 添加显式关系
    sgraph.add_explicit_edge("concept_ml", "concept_dl", "子概念")
    sgraph.add_explicit_edge("doc_ai_intro", "concept_ml", "介绍")
    sgraph.add_explicit_edge("paper_transformer", "paper_bert", "引用")
    sgraph.add_explicit_edge("obs_dl_nlp", "doc_nlp_intro", "相关")
    logging.info(
        f"SemanticGraph 节点数: {sgraph.nx_graph.number_of_nodes()}，边数: {sgraph.nx_graph.number_of_edges()}"
    )

    logging.info("------ 结构说明 ------")
    logging.info("MemoryUnit：最小知识单元，存储具体内容。")
    logging.info("MemorySpace：可嵌套的知识空间，用户主动组织和收集unit。")
    logging.info("SemanticMap：全局唯一unit集合，支持embedding和向量检索。")
    logging.info(
        "SemanticGraph：在SemanticMap基础上，支持unit间的显式关系和图结构分析。"
    )

    logging.info("\nHippo.py 示例用法结束。")
