import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import requests

from ..core.Hippo import SemanticGraph

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class StructuredQuery:
    def __init__(self, step, target_types, inputs, filters, semantic_query):
        self.step = step
        self.target_types = target_types
        self.inputs = inputs
        self.filters = filters
        self.semantic_query = semantic_query


class QueryPlanner:
    def __init__(self, semantic_graph: SemanticGraph):
        self.semantic_graph = semantic_graph
        self.cache = {}

    def generate_prompt(self, query:str,ms_names:List[str],ms_structures:str,rel_types:List[str],example_query=None,example_plan=None,):
        prompt = f"""# SemanticGraph查询计划生成任务

你是一个专门处理SemanticGraph数据结构查询的AI助手。你的任务是将用户的自然语言查询转换为结构化的查询计划。

## 背景信息

SemanticGraph是一种新型Python数据结构，它具有以下层次结构：
- **MemoryUnit**：最基本的信息存储单元，作为图中的节点
- **MemorySpace**：根据类别属性组织的MemoryUnit集合
- **SemanticGraph**：包含多个MemorySpace的完整图结构

MemoryUnit之间存在两种连接方式：
1. **显式结构边**：有向边，具有明确定义的关系类型
2. **隐式语义边**：基于MemoryUnit内容之间的语义相似度建立的连接

## 当前查询环境

现在，用户需要通过自然语言对这个SemanticGraph进行查询。你需要：
1. 分析用户的自然语言查询
2. 根据提供的图结构信息和可用API
3. 生成一个JSON格式的查询计划

## 提供的信息

### 1. 用户查询
```
{query}
```

### 2. 图结构信息

#### (a) 记忆空间名称列表
```
{ms_names}
```

#### (b) 记忆空间结构描述
```
{ms_structures}
```
注意：同一记忆空间中的所有记忆单元具有相同的字段结构。

#### (c) 显式关系边的类型
```
{rel_types}
```

## 可用的查询API

1. **search_similarity_in_graph(query_text, top_k=5, ms_name)**
   - 功能：根据输入的自然语言查询，返回图中语义最相似的Top-K个记忆单元
   - 参数：
     - query_text: 自然语言查询文本
     - top_k: 返回结果数量，默认为5
     - ms_name: 指定记忆空间名称，若不指定则在全图范围内查找
   - 返回：记忆单元列表

2. **get_implicit_neighbors(uid, top_k=5, ms_name)**
   - 功能：获取指定记忆单元的隐式关系邻居节点
   - 参数：
     - uid: 记忆单元ID
     - top_k: 返回的邻居数量，默认为5
     - ms_name: 指定记忆空间名称，若不指定则在全图范围内查找
   - 返回：记忆单元列表

3. **get_explicit_neighbors(src_uid=None, tgt_uid=None, rel_type=None)**
   - 功能：获取记忆单元的显式关系邻居节点
   - 参数：
     - src_uid: 可选，源节点ID
     - tgt_uid: 可选，目标节点ID
     - rel_type: 可选，关系类型筛选
   - 返回：记忆单元列表

4. **filter_memory_units(filter_condition, ms_names)**
   - 功能：通过特定字段的值过滤记忆单元
   - 参数：
     - filter_condition: 字典形式的过滤键值对
     - ms_names: 指定记忆空间名称列表，若不指定则在全图范围内查找
   - 返回：过滤后的记忆单元列表

5. **aggregate_results(memory_units)**
   - 功能：对查询结果进行聚合操作，统计记忆单元的出现频率
   - 参数：
     - memory_units: 记忆单元列表
   - 返回：聚合结果，记忆单元出现频率

## 查询计划格式

你需要生成一个JSON格式的查询计划，包含以下结构：
```json
{
  "plan": [
    {
      "step": 1,
      "operation": "API调用名称",
      "parameters": {
        "参数名1": "参数值1",
        "参数名2": "参数值2"
      },
      "result_var": "存储结果的变量名"
    },
    {
      "step": 2,
      "operation": "API调用名称",
      "parameters": {
        "参数名": "参数值或之前步骤的变量名"
      },
      "result_var": "存储结果的变量名"
    }
  ]
}
```"""

        if example_query and example_plan:

          prompt +=  f"""
          
## 示例

以下是一个自然语言查询及其对应的查询计划示例：

### 用户查询
```
{example_query}
```

### 查询计划
```json
{example_plan}
        ```"""

        prompt+=f"""
        
## 你的任务

    请根据用户的自然语言查询，分析查询意图，并使用提供的API生成一个能有效检索所需信息的查询计划。

    只需输出JSON格式的查询计划，不需要解释或说明过程。确保JSON格式正确无误，可以被程序直接解析。"""

#         prompt = f"""您是一个专门为SemanticGraph数据结构设计查询计划的AI助手。SemanticGraph是一种新型Python数据结构，以节点为粒度存储信息单元(MemoryUnit)，并以图的形式组织这些单元。这个数据结构具有以下特点：

# 1. 层次结构：MemoryUnit -> MemorySpace -> SemanticGraph
#    - MemoryUnit：基本信息单元，存储具体内容
#    - MemorySpace：按照标量类别组织MemoryUnit，同一MemorySpace中的所有MemoryUnit具有相同的类别属性和字段结构
#    - SemanticGraph：最顶层结构，包含多个MemorySpace

# 2. 两种连接方式：
#    - 显式结构边：有向边，明确存储节点间的关系类型
#    - 隐式语义边：根据MemoryUnit内容间的语义相似度自动建立

# 我将向您提供以下关键信息：
# 1. 用户的自然语言查询描述
# 2. 图结构信息：
#    - 所有记忆空间名称(ms_names)
#    - 每个记忆空间中记忆单元的所有字段及示例值
#    - 显式关系边的关系类型(rel_type)
# 3. 可用的查询API：
#    - search_similarity_in_graph(query, top_k)：根据自然语言查询返回图中相似的Top-K个记忆单元
#    - get_implicit_neighbors(memory_unit_id, top_k)：获取记忆单元的Top-K个隐式关系(语义相似)邻居节点
#    - get_explicit_neighbors(source_id=None, target_id=None, rel_type=None)：获取记忆单元的显式关系邻居节点，可按源节点、目标节点或关系类型筛选
#    - filter_memory_units(memory_units, field, value)：通过特定字段的值过滤记忆单元
#    - aggregate_results(memory_units, field)：对查询结果进行聚合操作(当前仅支持频率统计)

# 您的任务是将用户的自然语言查询转换为结构化的查询计划。查询计划应由一系列步骤组成，每一步可以：
# 1. 调用API
# 2. 暂时存储中间结果
# 3. 使用先前步骤产生的结果

# 请以JSON格式输出查询计划，格式示例：
# {{
#   "steps": [
#     {{
#       "step_id": 1,
#       "operation": "API_NAME",
#       "parameters": {{
#         "param1": "value1",
#         "param2": "value2"
#       }},
#       "output_var": "result_1"
#     }},
#     {{
#       "step_id": 2,
#       "operation": "API_NAME",
#       "parameters": {{
#         "param1": "$result_1",
#         "param2": "value2"
#       }},
#       "output_var": "result_2"
#     }}
#   ]
# }}

# 在参数值中，可以使用"$变量名"引用之前步骤的输出结果。

# 请基于我提供的信息，仅输出符合要求的JSON格式查询计划。不需要解释或其他额外内容。
# """

        return prompt

    def query_llm(self, query):
        prompt = self.generate_prompt(query)
        print(f"prompt: {prompt}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:32b", "prompt": prompt, "stream": False},
        )
        response_json = response.json()
        return response_json.get("response", "")

    def generate_query_plan(
        self,
        natural_language_query,
    ):
        result = self.query_llm(natural_language_query)

        try:
            structured_query = json.loads(result)
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")
            return None
        return structured_query

  

    def semantic_search(self, query, filtered_datas: list[tuple]):
        if not filtered_datas:
            return []

        # 提取过滤数据中的嵌入向量
        filtered_embeddings = [data[3] for data in filtered_datas]

        # 获取查询的嵌入向量
        query_emb = self.semantic_graph.semantic_map._get_text_embedding(query).reshape(
            1, -1
        )

        top_k = min(50, len(filtered_embeddings))

        # 如果过滤后的嵌入向量数量较少，直接计算距离而不构建索引
        if len(filtered_embeddings) < 10000:
            distances = np.linalg.norm(
                np.array(filtered_embeddings) - query_emb, ord=2, axis=1
            )
            results = sorted(zip(filtered_datas, distances), key=lambda x: x[1])
            return [result[0] for result in results[:top_k]]

        # 否则，使用FAISS构建索引并执行搜索
        index = faiss.IndexFlatL2(query_emb.shape[1])
        index.add(np.array(filtered_embeddings, dtype=np.float32))
        distances, indices = index.search(query_emb, len(filtered_embeddings))

        # 根据距离排序（升序）
        results = sorted(zip(filtered_datas, distances[0]), key=lambda x: x[1])
        return [result[0] for result in results[:top_k]]


def yelp_init(
    category_map_disk_path: str,
    attribute_map_disk_path: str,
    sgraph_disk_path: str,
    sgraph_semantic_map_disk_path: str,
    db_name="yelp_sds",
):
    global category_map, attribute_map, sgraph

    # category_map = SemanticMap()
    # category_map.load_data(category_map_disk_path)
    # category_map.build_index()

    # attribute_map = SemanticMap()
    # attribute_map.load_data(attribute_map_disk_path)
    # attribute_map.build_index()

    sgraph = SemanticGraph()
    sgraph.load_data(sgraph_disk_path)
    sgraph.semantic_map.load_data(sgraph_semantic_map_disk_path)
    sgraph.build_index()

    # sgraph.save_graph(
    #     "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/persist_data/graph/yelp_big.pkl"
    # )
    # sgraph.semantic_map.save_data(
    #     "/mnt/data1/home/guozy/gzy/SDS/multimodal_semantic_map_dev/persist_data/map/yelp_big.pkl"
    # )


if __name__ == "__main__":
    # LLM-based query parsing and execution
    planner = QueryPlanner(sgraph)

    natural_language_query = "Recommend some restaurants in California with good environment and spicy Chinese food."

    """
    查找我朋友们最喜欢的海鲜餐厅
    找到我朋友们在洛杉矶拍过的餐厅照片
    查找我朋友们在海鲜餐厅发布的评论和小费
    找到在纽约写过很多评论并且喜欢意大利菜的用户
    找到我朋友们评论过的海鲜餐厅
    """

    structured_query = planner.generate_query_plan(None, None, natural_language_query)
    # structured_query = planner.parse_query(
    #     category_map, attribute_map, natural_language_query
    # )

    if structured_query:
        # plan = planner.generate_plan(structured_query)
        results = planner.execute_plan(structured_query)

        print(f"\n\nQuery: {natural_language_query}, Results =>")
        for r in results:
            print(r)
    else:
        print("Failed to parse the query.")
