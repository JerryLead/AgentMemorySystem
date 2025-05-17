# 纯向量模型示例

本文档展示如何使用Hippo框架的纯向量模式（SemanticMap）进行记忆管理和检索。

## 基本使用流程

1. 创建记忆单元
2. 构建语义地图
3. 添加记忆空间
4. 添加记忆单元到空间
5. 构建索引
6. 执行检索

## 多模态记忆示例

```python
from hippo import MemoryUnit, SemanticMap
import numpy as np

# 创建记忆单元 - 文本
text_unit = MemoryUnit(
    uid="text1",
    raw_data={"text": "这是一只可爱的猫"},
    metadata={"modality": "text", "subject": "cat"}
)

# 创建记忆单元 - 图像
image_unit = MemoryUnit(
    uid="img1",
    raw_data={"image_path": "/path/to/cat.jpg"},
    metadata={"modality": "image", "subject": "cat"}
)

# 构建语义地图
semantic_map = SemanticMap(embedding_dim=768)

# 添加记忆空间
semantic_map.create_memory_space("multimodal")

# 添加记忆单元
semantic_map.add_unit(text_unit, ms_name="multimodal")
semantic_map.add_unit(image_unit, ms_name="multimodal")

# 构建索引
semantic_map.build_index()

# 文本查询图像
text_to_image = semantic_map.search_similarity_units_by_text(
    "猫咪照片",
    top_k=1
)

# 图像查询文本
image_to_text = semantic_map.search_similarity_units_by_image(
    "/path/to/query_cat.jpg",
    top_k=1
)
```

## 多空间组织示例

```python
from hippo import MemoryUnit, SemanticMap

# 构建语义地图
semantic_map = SemanticMap(embedding_dim=768)

# 创建多个记忆空间
semantic_map.create_memory_space("texts")
semantic_map.create_memory_space("images")
semantic_map.create_memory_space("videos")

# 添加文本记忆单元
text_unit = MemoryUnit(
    uid="text1",
    raw_data={"text": "这是一段关于AI的文本"}
)
semantic_map.add_unit(text_unit, ms_name="texts")

# 添加图像记忆单元
image_unit = MemoryUnit(
    uid="img1",
    raw_data={"image_path": "/path/to/ai_image.jpg"}
)
semantic_map.add_unit(image_unit, ms_name="images")

# 构建索引
semantic_map.build_index()

# 跨空间检索 - 在所有空间中搜索
all_results = semantic_map.search_similarity_units_by_text(
    "人工智能",
    top_k=5
)

# 限定空间检索 - 只在texts空间中搜索
text_results = semantic_map.search_similarity_units_by_text(
    "人工智能",
    top_k=3,
    ms_name="texts"
)
```
