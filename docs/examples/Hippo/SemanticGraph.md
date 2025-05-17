# SemanticGraph 使用方法（向量+标量模式）

`SemanticGraph` 是一种结合向量与标量信息的知识表示与检索结构，适用于复杂语义关系建模和高效检索。其核心思想是将节点的语义信息通过向量（embedding）表示，同时节点之间的关系可附加标量属性，实现更丰富的语义表达。

## 1. 初始化 SemanticGraph

```python
from agent_memory.semantic_graph import SemanticGraph

graph = SemanticGraph()
```

<!-- - `embedding_dim`：指定向量维度，需与所用 embedding 模型一致。 -->

## 2. 添加节点

每个节点包含文本内容及其向量表示，可选附加标量属性。

```python
node_id = graph.add_node(
   text="量子力学基础",
   embedding=embedding_vector,  # 由外部 embedding 模型生成
   attributes={"重要性": 0.9}
)
```

- `text`：节点描述文本。
- `embedding`：节点的向量表示。
- `attributes`：可选，节点的标量属性（如权重、重要性等）。

## 3. 添加边

边用于表达节点间的语义关系，可附加标量权重。

```python
graph.add_edge(
   source_id=node_id1,
   target_id=node_id2,
   relation="相关",
   weight=0.7
)
```

- `relation`：关系类型描述。
- `weight`：可选，边的权重或强度。

## 4. 检索与查询

### 4.1 向量相似度检索

根据输入文本，检索语义最相关的节点。

```python
results = graph.search_by_text(
   query="量子理论",
   top_k=5,
   embedding_model=your_embedding_model
)
```

### 4.2 结合标量属性筛选

可根据节点或边的标量属性进一步筛选结果。

```python
filtered = [n for n in results if n.attributes["重要性"] > 0.8]
```

## 5. 可视化与导出

支持将语义图导出为常见格式（如 NetworkX、Graphviz）进行可视化。

```python
graph.export_to_networkx()
```

## 6. 应用场景

- 复杂知识图谱构建
- 多模态语义检索
- 关系推理与可解释性分析

---

如需详细 API 说明，请参考项目文档或源码注释。