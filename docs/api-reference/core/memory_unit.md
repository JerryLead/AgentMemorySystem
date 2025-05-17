# MemoryUnit API参考

`MemoryUnit`是Hippo框架的基本记忆单元，用于存储原始数据和元数据。

## 类定义

```python
class MemoryUnit:
    """
    最小记忆单元，存储原始数据和元数据
    """
    def __init__(
        self, 
        uid: str, 
        raw_data: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ):
        """
        初始化一个记忆单元
        
        参数:
            uid (str): 唯一标识符
            raw_data (Dict[str, Any]): 原始数据，例如文本或图像路径
            metadata (Optional[Dict[str, Any]]): 可选元数据，如标签、类别等
            embedding (Optional[np.ndarray]): 可选的预计算嵌入向量
        """
        self.uid = uid
        self.raw_data = raw_data
        self.metadata = metadata or {}
        self.embedding = embedding
```

## 属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `uid` | str | 记忆单元的唯一标识符 |
| `raw_data` | Dict[str, Any] | 存储原始数据，如文本内容或图像路径 |
| `metadata` | Dict[str, Any] | 存储附加元数据，如标签、类别等 |
| `embedding` | np.ndarray | 单元的嵌入向量表示 |

## 使用示例

```python
# 创建一个文本记忆单元
text_unit = MemoryUnit(
    uid="text_1",
    raw_data={"text_content": "这是一段示例文本"},
    metadata={"category": "文本", "tags": ["示例", "文档"]}
)

# 创建一个图像记忆单元
image_unit = MemoryUnit(
    uid="image_1",
    raw_data={"image_path": "/path/to/image.jpg"},
    metadata={"category": "图像", "tags": ["示例", "照片"]}
)

# 创建带有预计算嵌入的记忆单元
import numpy as np
embedding = np.random.rand(768)  # 假设使用768维的嵌入向量
embedded_unit = MemoryUnit(
    uid="embedded_1",
    raw_data={"text": "带有预计算嵌入的文本"},
    metadata={"category": "文本"},
    embedding=embedding
)
```

## 注意事项

- `uid`应确保在整个系统中唯一，以避免冲突
- `raw_data`和`metadata`都是字典类型，可以灵活存储各种数据
- 如果不提供`embedding`，当添加到`SemanticMap`时会自动计算嵌入向量
