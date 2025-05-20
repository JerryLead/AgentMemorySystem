from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

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