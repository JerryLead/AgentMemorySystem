# SemanticMap API参考

`SemanticMap`是全局语义地图，用于聚合多个`MemorySpace`，支持跨空间语义检索。

## 类定义

```python
class SemanticMap:
    """
    全局语义地图，聚合多个MemorySpace，支持跨空间语义检索
    """
    def __init__(
        self,
        image_embedding_model_name: str = "clip-ViT-B-32",
        text_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        embedding_dim: int = 512,
        index: faiss.Index = faiss.IndexFlatL2,
        update_interval: Optional[int] = 100,
    ):
        """
        初始化语义地图
        
        参数:
            image_embedding_model_name (str): 图像嵌入模型名称
            text_embedding_model_name (str): 文本嵌入模型名称
            embedding_dim (int): 嵌入向量维度
            index (faiss.Index): 全局索引
            update_interval (Optional[int]): 每隔多少次操作更新一次索引。
        """
        self.text_model = SentenceTransformer("...")  # 文本嵌入模型
        self.image_model = SentenceTransformer("...")  # 图像嵌入模型
        self.embedding_dim = embedding_dim
        self._memory_spaces = {}  # 记忆空间字典
        self._emb_index = index  # 全局向量索引
```

## 属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `text_model` | SentenceTransformer | 文本嵌入模型 |
| `image_model` | SentenceTransformer | 图像嵌入模型 |
| `embedding_dim` | int | 嵌入向量维度 |
| `index` | faiss.Index | 记忆空间的局部索引 |

## 方法

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|-------|------|
| `add_unit` | `unit: MemoryUnit, ms_name: str` | `None` | 向指定空间添加记忆单元 |
| `remove_unit` | `uid: str` | `None` | 删除指定UID的记忆单元 |
| `update_unit` | `unit: MemoryUnit` | `None` | 更新指定的记忆单元 |
| `get_unit` | `uid: str` | `MemoryUnit` | 获取指定UID的记忆单元 |
| `create_memory_space` | `ms_name: str` | `MemorySpace` | 创建并添加新的记忆空间 |
| `get_memory_space` | `ms_name: str` | `None` | 获取指定名称的记忆空间 |
| `search_similarity_units_by_text` | `query_text: str, top_k: int, ms_name: str` | `List[MemoryUnit]` | 根据文本查询语义相似的记忆单元 |
| `search_similarity_units_by_image` | `query_image: str, top_k: int, ms_name: str` | `List[MemoryUnit]` | 根据图像查询语义相似的记忆单元 |
| `build_index` | `embedding_dim: int` | `None` | 重建全局向量索引 |

## 使用示例

```python
from hippo import SemanticMap, MemoryUnit

# 初始化语义地图
semantic_map = SemanticMap(embedding_dim=768)

# 创建记忆空间
space1 = semantic_map.create_memory_space("文档")
space2 = semantic_map.create_memory_space("图像")

# 创建并添加记忆单元
text_unit = MemoryUnit(
    uid="text1",
    raw_data={"text": "这是一个示例文本"}
)
semantic_map.add_unit(text_unit, ms_name="文档")

image_unit = MemoryUnit(
    uid="img1",
    raw_data={"image_path": "/path/to/image.jpg"}
)
semantic_map.add_unit(image_unit, ms_name="图像")

# 构建索引
semantic_map.build_index()

# 文本语义检索
similar_to_text = semantic_map.search_similarity_units_by_text(
    query_text="查找相关示例", 
    top_k=3
)

# 图像语义检索
similar_to_image = semantic_map.search_similarity_units_by_image(
    query_image="/path/to/query.jpg",
    top_k=3
)

# 在指定空间中检索
docs_only = semantic_map.search_similarity_units_by_text(
    query_text="查找相关示例",
    top_k=3,
    ms_name="文档"
)
```

## 注意事项

- 首次使用时会下载并缓存嵌入模型，需要确保网络连接
- 大规模数据管理时，请考虑使用更高效的索引结构
- 添加或更新记忆单元后，需要重建索引才能反映更改
- 如果未指定`ms_name`，则在全局范围内搜索