# 快速入门

本指南将帮助您快速了解如何使用Hippo框架进行记忆管理和语义检索。

## 基本概念

Hippo框架由四个核心类组成：

1. **MemoryUnit**：最小记忆单元，存储原始数据和元数据
2. **MemorySpace**：管理一组记忆单元，支持索引构建和查询
3. **SemanticMap**：全局语义地图，聚合多个MemorySpace，支持跨空间语义检索
4. **SemanticGraph**：基于语义地图的图结构，支持显式结构关系和隐式语义关联检索

## 两种使用模式

### 1. 纯向量模式 (SemanticMap)

适用于仅需基于语义相似度的检索场景。

```python
from Hippo import MemoryUnit, SemanticMap
import numpy as np

# 创建记忆单元
unit = MemoryUnit(
    uid="u1",
    raw_data={"text": "示例文本"},
    metadata={"category": "example"}
)

# 构建语义地图
semantic_map = SemanticMap(embedding_dim=768)

# 添加记忆空间
semantic_map.create_memory_space("example_space")

# 添加记忆单元
semantic_map.add_unit(unit, ms_name="example_space")

# 检索前需手动建立索引
semantic_map.build_index()

# 检索相似单元
similar_units = semantic_map.search_similarity_units_by_text("查询文本", top_k=5)
```

### 2. 向量+图结构模式 (SemanticMap + SemanticGraph)
适用于同时需要语义相似性和显式关系的复杂场景。

```python
from hippo import MemoryUnit, SemanticMap, SemanticGraph
import numpy as np

# 创建记忆单元
unit1 = MemoryUnit(
    uid="u1",
    raw_data={"text": "示例文本1"},
    metadata={"category": "example"}
)

unit2 = MemoryUnit(
    uid="u2",
    raw_data={"text": "示例文本2"},
    metadata={"category": "example"}
)

# 构建语义地图与图结构
semantic_map = SemanticMap(embedding_dim=768)
semantic_graph = SemanticGraph(semantic_map)

# 添加记忆空间
semantic_graph.create_memory_space("example_space")

# 添加记忆单元
semantic_graph.add_unit(unit1, ms_name="example_space")
semantic_graph.add_unit(unit2, ms_name="example_space")

# 添加显式关系
semantic_graph.add_explicit_edge("u1", "u2", rel_type="related_to")

# 检索前需手动建立索引
semantic_graph.build_index()

# 检索相似单元
similar_units = semantic_graph.search_similarity_units_by_text("查询文本", top_k=5)

# 图中检索显式关联
explicit_neighbors = semantic_graph.get_explicit_neighbors("u1")

# 图中检索隐式关联
implicit_neighbors = semantic_graph.get_implicit_neighbors("u1", top_k=3)
```

## 下一步
- 查看[API文档](./api-reference/core/)了解详细的函数接口
- 浏览[示例代码](./examples/Hippo/)了解更多使用场景
- 阅读[常见问题](./faq/)解决可能遇到的问题