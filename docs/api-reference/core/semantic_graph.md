
# SemanticGraph API参考

`SemanticGraph`是基于`SemanticMap`的图结构，支持显式结构关系和隐式语义关联检索。

## 类定义

```python
class SemanticGraph:
    """
    基于语义地图的图结构，支持显式结构关系和隐式语义关联检索
    """
    def __init__(self, semantic_map: SemanticMap):
        """
        初始化语义图
        
        参数:
            semantic_map (SemanticMap): 关联的语义地图
        """
        self.semantic_map = semantic_map
        self.nx_graph = nx.DiGraph()  # 有向图结构
```

## 属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `semantic_map` | SemanticMap | 关联的语义地图 |
| `nx_graph` | nx.DiGraph | 有向图结构 |

## 方法

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|-------|------|
| `add_unit` | `unit: MemoryUnit, ms_name: str` | `None` | 向指定空间添加记忆单元 |
| `remove_unit` | `uid: str` | `None` | 删除指定UID的记忆单元 |
| `update_unit` | `unit: MemoryUnit` | `None` | 更新指定的记忆单元 |
| `get_unit` | `uid: str` | `MemoryUnit` | 获取指定UID的记忆单元 |
| `create_memory_space` | `ms_name: str` | `MemorySpace` | 创建并添加新的记忆空间 |
| `get_memory_space` | `ms_name: str` | `None` | 获取指定名称的记忆空间 |
| `add_explicit_edge` | `from_uid: str, to_uid: str` | `None` | 添加显式边 |
| `remove_explicit_edge` | `from_uid: str, to_uid: str` | `None` | 删除显式边 |
| `build_index` | `embedding_dim: int` | `None` | 构建语义图索引 |
| `search_similarity_units_by_text` | `query_text: str, top_k: int` | `List[MemoryUnit]` | 基于图结构的文本语义相似性检索 |
| `search_similarity_units_by_image` | `query_image: str, top_k: int` | `List[MemoryUnit]` | 基于图结构的图像语义相似性检索 |
| `get_explicit_neighbors` | `uid: str` | `List[MemoryUnit]` | 获取显式连接的邻近节点 |
| `get_implicit_neighbors` | `uid: str, top_k: int` | `List[MemoryUnit]` | 获取隐式语义关联的邻近节点 |

## 使用示例

```python
from hippo import SemanticMap, SemanticGraph, MemoryUnit

# 初始化语义地图与图结构
semantic_map = SemanticMap(embedding_dim=768)
semantic_graph = SemanticGraph(semantic_map)

# 创建记忆空间
space = semantic_graph.create_memory_space("文档")

# 创建记忆单元
unit1 = MemoryUnit(uid="doc1", raw_data={"text": "Python编程语言"})
unit2 = MemoryUnit(uid="doc2", raw_data={"text": "机器学习算法"})
unit3 = MemoryUnit(uid="doc3", raw_data={"text": "深度学习框架"})

# 添加记忆单元
semantic_graph.add_unit(unit1, ms_name="文档")
semantic_graph.add_unit(unit2, ms_name="文档")
semantic_graph.add_unit(unit3, ms_name="文档")

# 添加显式关系
semantic_graph.add_explicit_edge("doc1", "doc2", rel_type="相关")
semantic_graph.add_explicit_edge("doc2", "doc3", rel_type="前提")

# 构建索引
semantic_graph.build_index()

# 检索显式关联的节点
neighbors = semantic_graph.get_explicit_neighbors("doc2")

# 检索隐式语义相似的节点
similar = semantic_graph.get_implicit_neighbors("doc1", top_k=2)

# 基于文本的语义相似性检索
results = semantic_graph.search_similarity_units_by_text(
    "机器学习入门",
    top_k=3
)
```

## 注意事项

- 显式边需要手动添加，而隐式关系基于语义相似性自动计算
- 复杂图结构可能导致检索性能下降，请适当优化图的结构
- 对于大规模图，考虑使用图数据库作为后端存储
- 添加或更新节点后，需要重建索引才能反映更改
