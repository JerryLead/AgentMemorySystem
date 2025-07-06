import os
import pickle
import logging

from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx
import faiss

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

    参数说明：
        ms_name (str): 记忆空间名称。
        index_factory (可选): 用于自定义faiss索引类型的工厂。支持以下类型：
            - None（默认）：使用faiss.IndexFlatL2。
            - 可调用对象（如lambda dim: faiss.IndexFlatIP(dim)）：传入维度，返回faiss索引实例。
            - 已实例化的faiss索引对象（如faiss.IndexFlatL2(512)）。
        示例：
            import faiss
            ms = MemorySpace("myspace")  # 默认L2索引
            ms2 = MemorySpace("ipspace", index_factory=lambda dim: faiss.IndexFlatIP(dim))
            ms3 = MemorySpace("custom", index_factory=faiss.IndexFlatL2(256))
    """

    def __init__(self, ms_name: str, index_factory=None):
        if not isinstance(ms_name, str) or not ms_name.strip():
            raise ValueError("记忆空间名称不能为空")
        self.name: str = ms_name
        self._members: Dict[str, Union[MemoryUnit, "MemorySpace"]] = {}
        self._emb_index = None
        self._index_to_unit: Dict[int, MemoryUnit] = {}
        self._index_factory = index_factory  # 新增：可自定义faiss索引工厂

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

    def build_index(self, embedding_dim: int = 512, min_unit_threshold: int = 100):
        """
        递归收集所有unit并构建局部faiss索引。
        仅当unit数量大于等于min_unit_threshold时才建立索引，否则不建立。
        """
        units = self.get_all_units()
        if len(units) < min_unit_threshold:
            self._emb_index = None
            self._index_to_unit = {}
            logging.info(f"MemorySpace {self.name} 单元数{len(units)}，未建立索引。")
            return

        if self._index_factory is None:
            self._emb_index = faiss.IndexFlatL2(embedding_dim)
        elif callable(self._index_factory):
            self._emb_index = self._index_factory(embedding_dim)
        elif hasattr(self._index_factory, "add"):
            self._emb_index = self._index_factory
        else:
            raise ValueError("index_factory必须为None、可调用工厂或faiss索引实例")
        self._index_to_unit = {}
        embeddings = []
        count = 0
        for unit in units:
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
        logging.info(f"MemorySpace {self.name} 建立了索引，单元数{len(units)}。")

    def search_similarity_units_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[MemoryUnit, float]]:
        # 如果有索引且索引非空，优先用索引
        if self._emb_index and getattr(self._emb_index, "ntotal", 0) > 0:
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
        # 否则直接暴力遍历所有unit
        units = self.get_all_units()
        if not units:
            return []
        sims = []
        for u in units:
            if u.embedding is not None:
                # 余弦相似度
                a = query_vector.astype(np.float32)
                b = u.embedding.astype(np.float32)
                sim = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                )
                sims.append((u, sim))
        # 取top_k
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    # 可根据需要扩展更多接口，如递归查找、移动成员等

    def save(self, file_path: str):
        """将当前MemorySpace对象保存到指定文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "MemorySpace":
        """从指定文件加载MemorySpace对象"""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def union(*spaces: "MemorySpace") -> List[MemoryUnit]:
        """
        返回多个MemorySpace的unit并集（去重，按uid）。
        用法：MemorySpace.union(ms1, ms2, ...)
        """
        seen = set()
        result = []
        for ms in spaces:
            for u in ms.get_all_units():
                if u.uid not in seen:
                    seen.add(u.uid)
                    result.append(u)
        return result

    @staticmethod
    def intersection(*spaces: "MemorySpace") -> List[MemoryUnit]:
        """
        返回多个MemorySpace的unit交集（按uid）。
        用法：MemorySpace.intersection(ms1, ms2, ...)
        """
        if not spaces:
            return []
        sets = [set(u.uid for u in ms.get_all_units()) for ms in spaces]
        common_uids = set.intersection(*sets)
        # 取第一个空间的unit对象（可选：可合并属性）
        first_units = {u.uid: u for u in spaces[0].get_all_units()}
        return [first_units[uid] for uid in common_uids]

    @staticmethod
    def difference(ms1: "MemorySpace", ms2: "MemorySpace") -> List[MemoryUnit]:
        """
        返回ms1中有而ms2中没有的unit（按uid）。
        用法：MemorySpace.difference(ms1, ms2)
        """
        uids2 = set(u.uid for u in ms2.get_all_units())
        return [u for u in ms1.get_all_units() if u.uid not in uids2]


class SemanticMap:
    """
    语义地图 (SemanticMap) 只负责存储所有不重复的 MemoryUnit，并支持基于相似度的搜索。
    统一管理所有MemorySpace（树/嵌套结构），支持全局和局部索引。

    参数说明：
        image_embedding_model_name (str): 图像嵌入模型名。
        text_embedding_model_name (str): 文本嵌入模型名。
        update_interval (int): 自动重建索引的操作计数阈值。
        index_factory (可选): 用于自定义faiss索引类型的工厂。支持以下类型：
            - None（默认）：使用faiss.IndexFlatL2。
            - 可调用对象（如lambda dim: faiss.IndexFlatIP(dim)）：传入维度，返回faiss索引实例。
            - 已实例化的faiss索引对象（如faiss.IndexFlatL2(512)）。
        memory_spaces (可选): 传入初始MemorySpace列表。
        示例：
            import faiss
            sm = SemanticMap()  # 默认L2索引
            sm2 = SemanticMap(index_factory=lambda dim: faiss.IndexFlatIP(dim))
            sm3 = SemanticMap(index_factory=faiss.IndexFlatL2(256))
    """

    DEFAULT_TEXT_EMBEDDING_KEY = "text_content"
    DEFAULT_IMAGE_EMBEDDING_KEY = "image_path"

    def __init__(
        self,
        image_embedding_model_name: str = "clip-ViT-B-32",
        text_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        update_interval: Optional[int] = 100,
        index_factory=None,
        memory_spaces: Optional[List["MemorySpace"]] = None,  # 新增
    ):
        self._units: Dict[str, MemoryUnit] = {}  # 全局唯一unit
        self._emb_index = None
        self._index_to_unit: Dict[int, MemoryUnit] = {}
        self._index_factory = index_factory
        self.memory_spaces: List["MemorySpace"] = memory_spaces or []  # 新增
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

    def build_index(self, min_unit_threshold: int = 100):
        """
        根据所有unit的embedding重建全局索引，并递归为所有MemorySpace建立索引。
        小于阈值的空间不建立索引。
        """
        # 构建全局索引
        if self._index_factory is None:
            self._emb_index = faiss.IndexFlatL2(512)
        elif callable(self._index_factory):
            self._emb_index = self._index_factory(512)
        elif hasattr(self._index_factory, "add"):
            self._emb_index = self._index_factory
        else:
            raise ValueError("index_factory必须为None、可调用工厂或faiss索引实例")
        self._index_to_unit = {}
        embeddings = []
        count = 0
        for unit in self._units.values():
            if unit.embedding is not None:
                embeddings.append(unit.embedding)
                self._index_to_unit[count] = unit
                count += 1
        if embeddings:
            import numpy as np

            self._emb_index.add(np.array(embeddings, dtype=np.float32))
        else:
            logging.warning("SemanticMap has no valid embeddings to build index.")
        # 递归为所有memory_spaces及其子空间建立索引
        for ms in self.memory_spaces:
            self._recursive_build_index(ms, min_unit_threshold)
        logging.info("SemanticMap: 全局及所有MemorySpace索引已重建（含阈值过滤）。")

    def _recursive_build_index(self, ms: "MemorySpace", min_unit_threshold: int):
        ms.build_index(min_unit_threshold=min_unit_threshold)
        for child in ms.get_all_spaces():
            self._recursive_build_index(child, min_unit_threshold)

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

    # --- MemorySpace 相关接口迁移 ---
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

    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        seen = set()
        result = []
        for u in units:
            if u.uid not in seen:
                seen.add(u.uid)
                result.append(u)
        return result

    # --- 持久化 ---
    def save_map(self, dir_path: str):
        # 保存memory_spaces结构
        with open(os.path.join(dir_path, "memory_spaces.pkl"), "wb") as f:
            pickle.dump(self.memory_spaces, f)
        with open(os.path.join(dir_path, "semantic_map.pkl"), "wb") as f:
            pickle.dump(self, f)
        logging.info(f"SemanticMap 已保存到 {dir_path}。")

    @classmethod
    def load_map(cls, file_path: str) -> "SemanticMap":
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "rb") as f:
            loaded_map = pickle.load(f)
        # 加载memory_spaces
        dir_path = os.path.dirname(file_path)
        ms_file = os.path.join(dir_path, "memory_spaces.pkl")
        if os.path.isfile(ms_file):
            with open(ms_file, "rb") as fms:
                loaded_map.memory_spaces = pickle.load(fms)
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
        semantic_map: Optional[SemanticMap] = None,
        memory_spaces: Optional[List[MemorySpace]] = None,
    ):
        # memory_spaces参数兼容旧用法，优先传给semantic_map
        if semantic_map is None:
            self.semantic_map: SemanticMap = SemanticMap(memory_spaces=memory_spaces)
        else:
            self.semantic_map: SemanticMap = semantic_map
            if memory_spaces:
                self.semantic_map.memory_spaces = memory_spaces
        self.nx_graph: nx.DiGraph = nx.DiGraph()
        self.rel_types: Set[str] = set()
        logging.info("SemanticGraph 已初始化。")

    # --- MemorySpace相关接口直通 ---
    def get_all_memory_space_names(self) -> List[str]:
        return self.semantic_map.get_all_memory_space_names()

    def get_memory_space_structures(self) -> List[dict]:
        return self.semantic_map.get_memory_space_structures()

    def get_units_in_memory_space(self, ms_names, recursive=True) -> List[MemoryUnit]:
        return self.semantic_map.get_units_in_memory_space(
            ms_names, recursive=recursive
        )

    def deduplicate_units(self, units: List[MemoryUnit]) -> List[MemoryUnit]:
        return self.semantic_map.deduplicate_units(units)

    # --- 辅助方法 ---
    def _get_node_id(self, obj):
        if isinstance(obj, MemoryUnit):
            return obj.uid
        elif isinstance(obj, MemorySpace):
            return f"ms:{obj.name}"
        elif isinstance(obj, str):
            return obj
        else:
            raise ValueError("不支持的节点类型")

    def _is_memory_space_id(self, node_id):
        return isinstance(node_id, str) and node_id.startswith("ms:")

    def _get_memory_space_by_id(self, ms_id):
        # ms_id: ms:{name}
        if not self._is_memory_space_id(ms_id):
            return None
        ms_name = ms_id[3:]

        def _find(ms, name):
            if ms.name == name:
                return ms
            for m in ms._members.values():
                if isinstance(m, MemorySpace):
                    res = _find(m, name)
                    if res:
                        return res
            return None

        for ms in self.semantic_map.memory_spaces:
            res = _find(ms, ms_name)
            if res:
                return res
        return None

    # 允许直接操作memory_spaces（如增删空间），实际操作smap，并同步nx_graph
    def add_memory_space(self, ms: MemorySpace):
        self.semantic_map.memory_spaces.append(ms)
        ms_id = self._get_node_id(ms)
        if not self.nx_graph.has_node(ms_id):
            self.nx_graph.add_node(ms_id, type="memory_space", name=ms.name)

    def remove_memory_space(self, ms_name: str):
        self.semantic_map.memory_spaces = [
            ms for ms in self.semantic_map.memory_spaces if ms.name != ms_name
        ]
        ms_id = f"ms:{ms_name}"
        if self.nx_graph.has_node(ms_id):
            self.nx_graph.remove_node(ms_id)

    def add_unit(
        self, unit: MemoryUnit, rebuild_semantic_map_index_immediately: bool = False
    ):
        """
        向图谱添加一个记忆单元 (节点)，并注册到 SemanticMap。
        """
        self.semantic_map.register_unit(unit)
        if rebuild_semantic_map_index_immediately:
            self.semantic_map.build_index()
        node_id = self._get_node_id(unit)
        if self.nx_graph.has_node(node_id):
            logging.warning(
                f"MemoryUnit {unit.uid} already exists in the graph, skip add."
            )
            return
        self.nx_graph.add_node(
            node_id, raw_data=unit.raw_data, metadata=unit.metadata, type="memory_unit"
        )
        logging.debug(f"节点 '{node_id}' 已添加到 NetworkX 图。")

    def remove_unit(self, uid: str, rebuild_index_immediately: bool = False):
        node_id = self._get_node_id(uid)
        if self.nx_graph.has_node(node_id):
            self.nx_graph.remove_node(node_id)
            logging.debug(f"MemoryUnit {uid} has been removed from NetworkX graph.")
        else:
            logging.warning(f"MemoryUnit {uid} does not exist in NetworkX graph.")
        self.semantic_map.unregister_unit(uid)
        if rebuild_index_immediately:
            self.semantic_map.build_index()

    def update_unit(self, unit: MemoryUnit):
        node_id = self._get_node_id(unit)
        if self.nx_graph.has_node(node_id):
            # 可根据需要更新节点属性
            self.nx_graph.nodes[node_id]["raw_data"] = unit.raw_data
            self.nx_graph.nodes[node_id]["metadata"] = unit.metadata
            logging.debug(f"MemoryUnit {unit.uid} 节点属性已更新。")
        else:
            logging.warning(f"MemoryUnit {unit.uid} does not exist in NetworkX graph.")
        self.semantic_map.register_unit(unit)  # 重新注册会覆盖

    def get_unit(self, uid: str) -> Optional[MemoryUnit]:
        node_id = self._get_node_id(uid)
        if self.nx_graph.has_node(node_id):
            return self.semantic_map.get_unit(uid)
        return None

    def add_explicit_edge(
        self,
        src_uid,
        tgt_uid,
        rel_type: str,
        bidirectional: bool = False,
        metadata: Optional[Dict] = None,
    ):
        src_id = self._get_node_id(src_uid)
        tgt_id = self._get_node_id(tgt_uid)
        # 节点存在性检查：MemoryUnit 或 MemorySpace
        src_is_unit = self.semantic_map.get_unit(src_id) is not None
        tgt_is_unit = self.semantic_map.get_unit(tgt_id) is not None
        src_is_space = (
            self._is_memory_space_id(src_id)
            and self._get_memory_space_by_id(src_id) is not None
        )
        tgt_is_space = (
            self._is_memory_space_id(tgt_id)
            and self._get_memory_space_by_id(tgt_id) is not None
        )
        if not (src_is_unit or src_is_space):
            raise ValueError(f"Source node {src_id} does not exist.")
        if not (tgt_is_unit or tgt_is_space):
            raise ValueError(f"Target node {tgt_id} does not exist.")
        if not self.nx_graph.has_node(src_id):
            if src_is_unit:
                unit = self.semantic_map.get_unit(src_id)
                if unit is not None:
                    self.nx_graph.add_node(
                        src_id,
                        raw_data=unit.raw_data,
                        metadata=unit.metadata,
                        type="memory_unit",
                    )
                else:
                    self.nx_graph.add_node(src_id, type="memory_unit")
            elif src_is_space:
                ms = self._get_memory_space_by_id(src_id)
                if ms is not None:
                    self.nx_graph.add_node(src_id, type="memory_space", name=ms.name)
                else:
                    self.nx_graph.add_node(src_id, type="memory_space")
        if not self.nx_graph.has_node(tgt_id):
            if tgt_is_unit:
                unit = self.semantic_map.get_unit(tgt_id)
                if unit is not None:
                    self.nx_graph.add_node(
                        tgt_id,
                        raw_data=unit.raw_data,
                        metadata=unit.metadata,
                        type="memory_unit",
                    )
                else:
                    self.nx_graph.add_node(tgt_id, type="memory_unit")
            elif tgt_is_space:
                ms = self._get_memory_space_by_id(tgt_id)
                if ms is not None:
                    self.nx_graph.add_node(tgt_id, type="memory_space", name=ms.name)
                else:
                    self.nx_graph.add_node(tgt_id, type="memory_space")
        if not metadata:
            metadata = {}
        metadata["created"] = datetime.now()
        edge_attributes = {"type": rel_type, **metadata}
        self.nx_graph.add_edge(src_id, tgt_id, **edge_attributes)
        logging.info(f"已添加从 '{src_id}' 到 '{tgt_id}' 的关系 '{rel_type}'。")
        if bidirectional:
            self.nx_graph.add_edge(tgt_id, src_id, **edge_attributes)
            logging.info(f"已添加从 '{tgt_id}' 到 '{src_id}' 的双向关系 '{rel_type}'。")
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

    def build_index(self, min_unit_threshold: int = 100):
        """
        重建语义图谱的全局向量索引（调用SemanticMap的build_index）。
        """
        self.semantic_map.build_index(min_unit_threshold=min_unit_threshold)
        logging.info("SemanticGraph: 全局及所有MemorySpace索引已重建（含阈值过滤）。")

    def get_all_relations(self) -> list:
        """
        返回所有已注册的显式关系类型列表。
        """
        return list(self.rel_types)

    # --- 持久化 ---
    def save_graph(self, directory_path: str):
        """
        将 SemanticGraph 的状态保存到指定目录。
        包括 SemanticMap 的数据、NetworkX 图结构、关系类型、memory_spaces结构。
        """
        os.makedirs(directory_path, exist_ok=True)
        self.semantic_map.save_map(directory_path)
        with open(os.path.join(directory_path, "semantic_graph.pkl"), "wb") as f:
            pickle.dump(self.nx_graph, f)
        with open(os.path.join(directory_path, "rel_types.pkl"), "wb") as f:
            pickle.dump(self.rel_types, f)
        logging.info(f"SemanticGraph 已保存到 {directory_path}")

    @classmethod
    def load_graph(cls, directory_path: str):
        """
        从指定目录加载 SemanticGraph 的状态。
        包括 SemanticMap 的数据、NetworkX 图结构、关系类型、memory_spaces结构。
        返回: SemanticGraph实例
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

        # 加载memory_spaces结构
        memory_spaces_file = os.path.join(directory_path, "memory_spaces.pkl")
        if not os.path.isfile(memory_spaces_file):
            raise ValueError(f"MemorySpaces file {memory_spaces_file} does not exist.")
        with open(memory_spaces_file, "rb") as f:
            memory_spaces = pickle.load(f)

        # 递归注册所有memory_spaces下的unit到semantic_map
        for ms in memory_spaces:
            semantic_map.register_units_from_space(ms, update_embedding=False)

        # 确保所有unit都在nx_graph节点中
        for unit in semantic_map.get_all_units():
            if not nx_graph.has_node(unit.uid):
                nx_graph.add_node(
                    unit.uid, raw_data=unit.raw_data, metadata=unit.metadata
                )

        graph_instance = cls(semantic_map, memory_spaces)
        graph_instance.nx_graph = nx_graph
        graph_instance.rel_types = rel_types
        logging.info(
            f"SemanticGraph 已从 '{directory_path}' 加载，并已注册所有unit与重建索引。"
        )
        return graph_instance

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
        # 优化：ms_names为None或单空间时用索引
        if ms_names is None:
            # 全局索引
            try:
                return [
                    u
                    for u, _ in self.semantic_map.search_similarity_units_by_vector(
                        unit.embedding, top_k
                    )
                ]
            except Exception:
                pass  # 回退暴力查找
        elif isinstance(ms_names, (list, tuple)) and len(ms_names) == 1:
            ms = None
            for m in self.semantic_map.memory_spaces:
                if m.name == ms_names[0]:
                    ms = m
                    break
            if ms:
                try:
                    return [
                        u
                        for u, _ in ms.search_similarity_units_by_vector(
                            unit.embedding, top_k
                        )
                    ]
                except Exception:
                    pass  # 回退暴力查找
        # 多空间或找不到索引时，暴力查找
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
        # 优化：ms_names为None或单空间时用索引
        if ms_names is None:
            # 全局索引
            try:
                return self.semantic_map.search_similarity_units_by_text(
                    query_text, top_k
                )
            except Exception:
                pass  # 回退暴力查找
        elif isinstance(ms_names, (list, tuple)) and len(ms_names) == 1:
            ms = None
            for m in self.semantic_map.memory_spaces:
                if m.name == ms_names[0]:
                    ms = m
                    break
            if ms:
                try:
                    return ms.search_similarity_units_by_vector(
                        self.semantic_map._get_text_embedding(query_text), top_k
                    )
                except Exception:
                    pass  # 回退暴力查找
        # 多空间或找不到索引时，暴力查找
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
