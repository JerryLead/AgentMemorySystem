import logging
import pickle

from semantic_data_structure.semantic_map import BaseSemanticMap

from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from functools import lru_cache
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import ast
import faiss

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class MemoryUnit:
    def __init__(
        self,
        uid: str,
        raw_data: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ):
        """
        :param uid: 唯一标识符 (如"Issue_456")
        :param raw_data: 原始业务数据
        :param metadata: 扩展元数据 (创建者、标签等)
        :param embedding: 语义向量（延迟生成）
        """
        self.uid = uid
        self.raw_data = raw_data
        self.metadata = metadata or {
            "created": datetime.now(),
            "updated": datetime.now(),
        }
        self.embedding: Optional[List[float]] = None  # 可存储多模态向量

        self.versions = []  # 支持版本演变记录
        # 自动化生成数据指纹
        self.data_fingerprint = self._generate_fingerprint(raw_data)

    def __str__(self):
        raw_data = {
            k: v for k, v in self.raw_data.items() if k not in ["body", "content"]
        }
        for k, v in raw_data.items():
            if k == "files_changed":
                raw_data[k] = v[:1]  # 仅显示第一个文件名
        return f"MemoryUnit(Uid: {self.uid}, Raw Data: {raw_data}, Metadata: {self.metadata}, Versions: {self.versions}, Fingerprint: {self.data_fingerprint})"

    def __repr__(self):
        return f"MemoryUnit({self.uid})"

    def __eq__(self, value):
        if not isinstance(value, MemoryUnit):
            return False
        return self.data_fingerprint == value.data_fingerprint

    def __hash__(self):
        return hash(self.data_fingerprint)

    def _generate_fingerprint(self, data: Dict) -> str:
        """生成数据唯一标识哈希"""
        data_str = str(sorted(data.items())).encode()
        fingerprint = hashlib.sha256(data_str).hexdigest()
        return fingerprint

    def add_version(self, new_data: Dict, update_meta: Dict):
        """记录新版本并更新元数据"""
        self.versions.append(
            {
                "timestamp": datetime.now(),
                "data": self.raw_data.copy(),
                "metadata": self.metadata.copy(),
            }
        )
        self.raw_data = new_data
        self.metadata.update(update_meta)
        self.updated_at = datetime.now()

    def get_latest_version(self) -> Dict:
        """获取当前版本完整状态"""
        return {
            "id": self.uid,
            "type": self.type,
            "data": self.raw_data,
            "metadata": self.metadata,
            "fingerprint": self.data_fingerprint,
        }


class MemorySpace:
    """按类型划分的语义命名空间（如Issue/PR/人员）"""

    def __init__(
        self,
        ms_name: str,
        ms_type: str,
        embedding_fields: List[str],
    ):
        """
        :param name: 命名空间名称（如"github_issues"）
        :param ms_type: 类型标签（如"github/issue"）
        """
        self.name = ms_name
        self.type = ms_type
        self.embedding_fields = embedding_fields
        self.units: Dict[str, MemoryUnit] = {}
        self.schema = None  # 可扩展Schema验证器
        self.emb_index = faiss.IndexFlatL2(512)

    def __str__(self):
        return f"MemorySpace Name: {self.name}, Type: {self.type}, Units Num: {len(self.units)}"

    def add_unit(self, unit: MemoryUnit) -> str:
        """添加单元并返回操作结果状态码"""
        # 状态码定义：200=成功；400=ID重复；403=数据类型违反约束
        if unit.uid in self.units:
            return "400: Duplicate ID"
        # if not self._is_unit_type_compatible(unit):
        #     return "403: Type incompatibility"

        self.units[unit.uid] = unit
        return "200: Success"

    def get_unit_by_uid(self, uid: str) -> MemoryUnit:
        unit = self.units.get(uid, None)
        return unit

    def _build_index(self):
        embs = [self.units.values()]


class IssueManagerSemanticMap(BaseSemanticMap):
    def __init__(
        self,
        image_embedding_model="clip-ViT-B-32",
        text_embedding_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        enable_embedding_model: bool = True,
    ):
        self.memoryspaces: Dict[str, MemorySpace] = {}
        if enable_embedding_model:
            self.text_model = SentenceTransformer(text_embedding_model)
            self.image_model = SentenceTransformer(image_embedding_model)
        self.reverse_index = {}  # uid -> namespace_name 快速查找

        self.embedding_strategies: Dict[str, Callable[[Dict], str]] = {}
        self._strategy_cache = lru_cache(maxsize=128)

    def __str__(self):
        return f"SemanticMap MemorySpaces: {list(self.memoryspaces.keys())}"

    def _generate_embedding_text(
        self,
        unit: MemoryUnit,
        ms: MemorySpace,
    ) -> Optional[str]:
        """
        生成用于嵌入编码的文本内容，优先使用自定义策略
        返回None时应跳过嵌入生成
        """
        # 1. 查找自定义策略
        strategy = self.embedding_strategies.get(ms.type)
        if strategy is not None:
            try:
                return strategy(unit.raw_data)
            except Exception as e:
                logging.warning(f"Strategy failed for {unit.uid}: {str(e)}")
                # 回退到默认策略

        # 2. 默认字段拼接策略
        text_parts = [
            str(unit.raw_data.get(field, "")) for field in ms.embedding_fields
        ]
        combined_text = "\n".join(text_parts).strip()
        return combined_text if combined_text else None

    def create_namespace(
        self,
        ms_name: str,
        ms_type: str,
        embedding_fields: List[str],
    ):
        """创建并配置新命名空间"""
        if ms_name in self.memoryspaces:
            raise ValueError(f"Memoryspace {ms_name} already exists")

        self.memoryspaces[ms_name] = MemorySpace(ms_name, ms_type, embedding_fields)

    def insert_unit(
        self,
        ms_name: str,
        unit: MemoryUnit,
        auto_embed: bool = True,
    ):
        """插入记忆单元并触发嵌入编码（若需要）"""
        # 校验命名空间存在性
        if ms_name not in self.memoryspaces:
            raise KeyError(f"Namespace {ms_name} not found")

        ms = self.memoryspaces[ms_name]

        # 语义嵌入生成（按需）
        if auto_embed and not unit.embedding:
            # 使用策略生成文本（可能返回None）
            embedding_text = self._generate_embedding_text(unit, ms)

            if embedding_text is not None:  # 这里会有打印日志吗？
                unit.embedding = self.text_model.encode(
                    embedding_text, convert_to_numpy=True
                )

        # 更新索引
        if ms.add_unit(unit) != "200: Success":
            logging.warning(f"Unit insertion failed: {unit.uid} already exists")
            return
        self.reverse_index[unit.uid] = ms_name

    def find_similar_units(
        self,
        query_text: str,
        ms_names: List[str] = None,
        top_k: int = 5,
    ) -> List[MemoryUnit]:
        """跨命名空间语义检索"""
        # 确定目标命名空间范围
        target_memoryspaces = [
            ms for mn, ms in self.memoryspaces.items() if mn in ms_names
        ]

        # 生成查询向量
        query_vec = self.text_model.encode(query_text)

        # 遍历所有候选单元计算相似度
        similarities = []
        for ms in target_memoryspaces:
            for unit in ms.units.values():
                if unit.embedding is not None:
                    sim = cosine_similarity([query_vec], [unit.embedding])[0][0]
                    similarities.append((sim, unit))

        # 返回Top-K结果
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
        res = []
        for sim, unit in similarities:
            unit.metadata["similarity_score"] = sim
            res.append(unit)

        return res

    def set_embedding_strategy(
        self,
        ms_type: str,
        strategy: Callable[[Dict], str],
        enable_cache: bool = True,
    ) -> None:
        """
        为特定命名空间类型注册自定义嵌入策略

        :param ms_type: 要绑定的命名空间类型(与Memoryspace的ms_type对应)
        :param strategy: 输入raw_data字典，返回用于编码的文本字符串
        :param enable_cache: 是否启用策略结果的LRU缓存
        """
        if not callable(strategy):
            raise TypeError("Strategy must be a callable function")

        # 动态创建带缓存版本
        cached_strategy = self._strategy_cache(strategy) if enable_cache else strategy
        self.embedding_strategies[ms_type] = cached_strategy

    def get_unit_by_uid(self, uid: str) -> MemoryUnit:
        ms = self.reverse_index.get(uid)

        if not ms:
            p_ms = uid.split("_")[1]
            if p_ms in ["issue", "pr", "commit", "contributor"]:
                ms = f"github_{p_ms}s"
            elif p_ms == "code":
                ms = "github_code"

        if ms is not None:
            return self.memoryspaces[ms].get_unit_by_uid(uid)
        return None

    def save_data(self, data_path: str):
        with open(data_path, "wb") as f:
            pickle.dump(self.memoryspaces, f)
        logging.info(f"Map data saved to {data_path}")

    def load_data(self, data_path: str):
        with open(data_path, "rb") as f:
            self.memoryspaces = pickle.load(f)
        logging.info(f"Map data loaded from {data_path}")


class CodeSemanticStrategy:
    """可扩展的策略类替代纯函数"""

    def __init__(self):
        self.ast_parser = ast

    def __call__(self, raw_data: Dict) -> str:
        code_content = raw_data.get("code_content", "")
        print(code_content)
        if not code_content:
            return ""

        try:
            ast_tree = self.ast_parser.parse(code_content)
            print(ast_tree)
            return self._ast_to_text(ast_tree)
        except SyntaxError as e:
            logging.warning(f"AST解析失败: {str(e)}")
            return code_content  # 退化到原始代码

    def _ast_to_text(self, ast_node) -> str:
        # 实现AST的特征化转换
        return " ".join(
            [
                f"FuncDef:{node.name}"
                for node in ast.walk(ast_node)
                if isinstance(node, ast.FunctionDef)
            ]
        )


if __name__ == "__main__":
    # 初始化全局语义地图
    semantic_map = IssueManagerSemanticMap()

    # 注册代码类策略
    code_strategy = CodeSemanticStrategy()
    semantic_map.set_embedding_strategy(
        ms_type="code", strategy=code_strategy, enable_cache=True
    )

    # 创建对应类型的命名空间
    semantic_map.create_namespace(
        ms_name="python_src",
        ms_type="code",
        embedding_fields=["code_content"],  # 即使字段存在也会被策略覆盖
    )

    # 插入代码单元
    code_unit = MemoryUnit(
        "file_utils_1",
        {
            "code_content": """
        def read_file(path):
            with open(path) as f:
                return f.read()
        """
        },
    )
    semantic_map.insert_unit("python_src", code_unit)

    # 检查生成的嵌入
    print(f"生成的嵌入维度: {code_unit.embedding.shape}")
