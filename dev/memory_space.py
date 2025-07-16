import logging
import pickle
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import numpy as np
import faiss
from .memory_unit import MemoryUnit

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

    # TODO:有改动，建议同步到Hippo
    def build_index(self, embedding_dim: int = 512, min_unit_threshold: int = 100):
        """
        递归收集所有unit并构建局部faiss索引。
        如果内部unit数量小于阈值（如100），则不建立索引。
        支持自定义faiss_index_type。
        修复IndexIDMap的add_with_ids问题。
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

            # 优先用自身的faiss_index_type，否则继承父级SemanticMap，否则用默认
            faiss_index_type = (
                self._faiss_index_type
                or getattr(semantic_map, "faiss_index_type", None)
                or "IDMap,Flat"
            )
            
            # 初始化FAISS索引
            if faiss_index_type.startswith("IDMap,"):
                # 对于IDMap类型，需要特殊处理
                base_index_type = faiss_index_type.split("IDMap,", 1)[1]
                base_index = faiss.index_factory(embedding_dim, base_index_type)
                self._emb_index = faiss.IndexIDMap(base_index)
            else:
                self._emb_index = faiss.index_factory(embedding_dim, faiss_index_type)
            
            self._index_to_uid = {}
            embeddings = []
            ids = []  # 用于IndexIDMap
            count = 0
            
            for unit in units:
                if unit.embedding is not None and unit.embedding.shape[0] == embedding_dim:
                    embeddings.append(unit.embedding)
                    self._index_to_uid[count] = unit.uid
                    ids.append(count)  # 使用计数作为ID
                    count += 1
            
            if not embeddings:
                logging.warning(
                    f"MemorySpace '{self.name}' 没有有效的embeddings来构建索引"
                )
                return
            
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # 根据索引类型选择添加方法
            if faiss_index_type.startswith("IDMap,"):
                # IndexIDMap类型必须使用add_with_ids
                ids_np = np.array(ids, dtype=np.int64)
                
                # 检查是否需要训练
                if hasattr(self._emb_index, 'is_trained') and not self._emb_index.is_trained:
                    if "IVF" in faiss_index_type:
                        logging.info(f"训练MemorySpace '{self.name}' 的索引...")
                        self._emb_index.train(embeddings_np)
                
                self._emb_index.add_with_ids(embeddings_np, ids_np)
            else:
                # 普通索引类型使用add
                if hasattr(self._emb_index, 'is_trained') and not self._emb_index.is_trained:
                    if "IVF" in faiss_index_type:
                        logging.info(f"训练MemorySpace '{self.name}' 的索引...")
                        self._emb_index.train(embeddings_np)
                
                self._emb_index.add(embeddings_np)
            
            logging.info(
                f"MemorySpace '{self.name}' 索引构建完成，包含 {count} 个向量 (faiss_index_type={faiss_index_type})"
            )
            
        except ImportError:
            logging.error("FAISS不可用，无法构建本地索引")
        except Exception as e:
            logging.error(f"构建索引时出错: {e}")
            # 提供更详细的错误信息
            import traceback
            logging.debug(f"详细错误信息: {traceback.format_exc()}")

    def search_similarity_units_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple["MemoryUnit", float]]:
        """
        在局部索引中搜索相似单元。如果没有索引则暴力搜索。
        修复IndexIDMap的搜索逻辑。
        """
        semantic_map = self._get_semantic_map()
        
        # 优先用索引
        if self._emb_index and getattr(self._emb_index, "ntotal", 0) > 0:
            try:
                query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
                
                # 检查索引类型
                faiss_index_type = (
                    self._faiss_index_type
                    or getattr(semantic_map, "faiss_index_type", None)
                    or "IDMap,Flat"
                )
                
                if faiss_index_type.startswith("IDMap,"):
                    # IndexIDMap返回的是实际的ID，不是索引位置
                    D, I = self._emb_index.search(query_vector_np, top_k)
                    results = []
                    for i in range(len(I[0])):
                        internal_id = int(I[0][i])
                        if I[0][i] == -1:
                            continue
                        uid = self._index_to_uid.get(internal_id)
                        if uid:
                            unit = semantic_map.memory_units.get(uid)
                            if unit:
                                results.append((unit, float(D[0][i])))
                            else:
                                logging.warning(
                                    f"索引中的UID '{uid}' 在SemanticMap中不存在"
                                )
                    return results
                else:
                    # 普通索引返回的是索引位置
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
                try:
                    a = query_vector.astype(np.float32)
                    b = u.embedding.astype(np.float32)
                    # 使用余弦相似度，但返回距离（越小越相似）
                    cosine_sim = float(
                        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    )
                    # 转换为距离（1 - 相似度）
                    distance = 1.0 - cosine_sim
                    sims.append((u, distance))
                except Exception as e:
                    logging.warning(f"计算相似度时出错: {e}")
                    continue
        
        # 按距离排序（越小越相似）
        sims.sort(key=lambda x: x[1])
        return sims[:top_k]

    # def build_index(self, embedding_dim: int = 512, min_unit_threshold: int = 100):
    #     """
    #     递归收集所有unit并构建局部faiss索引。
    #     如果内部unit数量小于阈值（如100），则不建立索引。
    #     支持自定义faiss_index_type。
    #     """
    #     try:
    #         semantic_map = self._get_semantic_map()
    #         units = self.get_all_units()
    #         if len(units) < min_unit_threshold:
    #             self._emb_index = None
    #             self._index_to_uid = {}
    #             logging.info(
    #                 f"MemorySpace {self.name} 单元数{len(units)}，未建立索引。"
    #             )
    #             return

    #         import faiss

    #         # 优先用自身的faiss_index_type，否则继承父级SemanticMap，否则用默认
    #         faiss_index_type = (
    #             self._faiss_index_type
    #             or getattr(semantic_map, "faiss_index_type", None)
    #             or "IDMap,Flat"
    #         )
    #         self._emb_index = faiss.index_factory(embedding_dim, faiss_index_type)
    #         self._index_to_uid = {}
    #         embeddings = []
    #         count = 0
    #         for unit in units:
    #             if unit.embedding is not None:
    #                 embeddings.append(unit.embedding)
    #                 self._index_to_uid[count] = unit.uid
    #                 count += 1
    #         if not embeddings:
    #             logging.warning(
    #                 f"MemorySpace '{self.name}' 没有有效的embeddings来构建索引"
    #             )
    #             return
    #         self._emb_index.add(np.array(embeddings, dtype=np.float32))
    #         logging.info(
    #             f"MemorySpace '{self.name}' 索引构建完成，包含 {count} 个向量 (faiss_index_type={faiss_index_type})"
    #         )
    #     except ImportError:
    #         logging.error("FAISS不可用，无法构建本地索引")
    #     except Exception as e:
    #         logging.error(f"构建索引时出错: {e}")

    # def search_similarity_units_by_vector(
    #     self, query_vector: np.ndarray, top_k: int = 5
    # ) -> List[Tuple["MemoryUnit", float]]:
    #     """
    #     在局部索引中搜索相似单元。如果没有索引则暴力搜索。
    #     """
    #     semantic_map = self._get_semantic_map()
    #     # 优先用索引
    #     if self._emb_index and getattr(self._emb_index, "ntotal", 0) > 0:
    #         try:
    #             query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
    #             D, I = self._emb_index.search(query_vector_np, top_k)
    #             results = []
    #             for i in range(len(I[0])):
    #                 idx = int(I[0][i])
    #                 if I[0][i] == -1:
    #                     continue
    #                 uid = self._index_to_uid.get(idx)
    #                 if uid:
    #                     unit = semantic_map.memory_units.get(uid)
    #                     if unit:
    #                         results.append((unit, float(D[0][i])))
    #                     else:
    #                         logging.warning(
    #                             f"索引中的UID '{uid}' 在SemanticMap中不存在"
    #                         )
    #             return results
    #         except Exception as e:
    #             logging.error(f"索引搜索出错: {e}")
    #             # 回退暴力搜索
    #     # 暴力搜索
    #     units = self.get_all_units()
    #     if not units:
    #         return []
    #     sims = []
    #     for u in units:
    #         if u.embedding is not None:
    #             a = query_vector.astype(np.float32)
    #             b = u.embedding.astype(np.float32)
    #             sim = float(
    #                 np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    #             )
    #             sims.append((u, sim))
    #     sims.sort(key=lambda x: x[1], reverse=True)
    #     return sims[:top_k]

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

