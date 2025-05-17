# MemorySpace API参考

`MemorySpace`用于管理一组相关的记忆单元，提供索引构建和查询功能。

## 类定义

```python
class MemorySpace:
    """
    管理一组记忆单元，支持索引构建和查询
    """
    def __init__(
        self,
        ms_name: str,
        index: faiss.Index = None,
    ):
        """
        初始化一个记忆空间
        
        参数:
            ms_name (str): 空间名称
        """
        self.name = ms_name
        self._units = {}  # 存储记忆单元的字典
        self._index = index  # faiss向量索引
```

## 属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `name` | str | 记忆空间的名称 |
| `index` | faiss.Index | 记忆空间的局部索引 |

## 方法

| 方法名 | 参数 | 返回值 | 描述 |
|--------|------|-------|------|
| `add_unit` | `unit: MemoryUnit` | `None` | 添加记忆单元 |
| `remove_unit` | `uid: str` | `None` | 通过UID删除记忆单元 |
| `update_unit` | `unit: MemoryUnit` | `None` | 更新记忆单元 |
| `get_unit` | `uid: str` | `MemoryUnit` | 获取指定UID的记忆单元 |
| `add_units` | `units: List[MemoryUnit]` | `None` | 批量添加记忆单元 |
| `remove_units` | `uids: List[str]` | `None` | 批量通过UID删除记忆单元 |
| `update_units` | `units: List[MemoryUnit]` | `None` | 批量更新记忆单元 |
| `get_units` | `uids: List[str]` | `List[MemoryUnit]` | 批量获取指定UID的记忆单元 |
| `get_all_units` | `None` | `List[MemoryUnit]` | 获取所有的记忆单元 |
| `build_index` | `embedding_dim: int` | `None` | 重建向量索引 |

## 使用示例

```python
from hippo import MemorySpace, MemoryUnit

# 创建记忆空间
space = MemorySpace(name="documents")

# 创建记忆单元
unit1 = MemoryUnit(uid="doc1", raw_data={"text": "第一个文档"})
unit2 = MemoryUnit(uid="doc2", raw_data={"text": "第二个文档"})

# 添加记忆单元
space.add_unit(unit1)
space.add_unit(unit2)

# 批量添加记忆单元
units = [
    MemoryUnit(uid="doc3", raw_data={"text": "第三个文档"}),
    MemoryUnit(uid="doc4", raw_data={"text": "第四个文档"})
]
space.add_units(units)

# 获取指定记忆单元
unit = space.get_unit("doc1")

# 更新记忆单元
unit1.metadata = {"importance": "high"}
space.update_unit(unit1)

# 删除记忆单元
space.remove_unit("doc2")

# 构建索引
space.build_index()

# 获取所有记忆单元
all_units = space.get_all_units()
```

## 注意事项

- 在执行向量检索前，需要先调用`build_index`构建索引
- 索引构建需要消耗资源，建议在批量添加或更新记忆单元后再构建索引
- 记忆单元的`uid`在同一记忆空间内必须唯一