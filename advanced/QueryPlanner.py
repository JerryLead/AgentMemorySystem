import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import requests
import re

from core.Hippo import SemanticGraph, MemoryUnit

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


class QueryPlanner:
    def __init__(self, semantic_graph: SemanticGraph):
        self.semantic_graph = semantic_graph

    def _generate_prompt(
        self,
        query: str,
    ):
        ms_names = self.semantic_graph.get_all_memory_space_names()
        ms_structures = self.semantic_graph.get_memory_space_structures()
        rel_types = self.semantic_graph.get_all_relations()

        prompt = f"""
# SemanticGraph查询计划生成任务

你是一个专门处理SemanticGraph数据结构查询的AI助手。你的任务是将用户的自然语言查询转换为结构化的查询计划。

## 背景信息

SemanticGraph是一种新型Python数据结构，具有如下特点：
- **MemoryUnit**：最基本的信息存储单元，作为图中的节点。
- **MemorySpace**（ms）：用户主动组织的知识空间，支持多层嵌套和unit重叠。每个ms可包含unit和子ms，unit可属于多个ms，实现跨领域、跨主题的灵活组织。ms之间没有强制的树状限制，支持任意嵌套和重叠。
- **SemanticGraph**：包含多个MemorySpace的完整图结构，支持unit间的显式结构边和隐式语义边。

MemoryUnit之间存在两种连接方式：
1. **显式结构边**：有向边，具有明确定义的关系类型
2. **隐式语义边**：基于MemoryUnit内容之间的语义相似度建立的连接

## 图结构信息

### 记忆空间名称列表
```
{str(ms_names)}
```

### 记忆空间嵌套结构（unit可重叠）
```
{str(ms_structures)}
```
- 注意：同一个unit可能出现在多个ms中，查询时可指定ms范围或合并结果。

### 显式关系边的类型
```
{str(rel_types)}
```

## 可用的查询API

1. **search_similarity_in_graph(query_text, top_k=5, ms_names=None, recursive=True)**
   - 功能：根据输入的自然语言查询，返回指定ms及其子ms中语义最相似的Top-K个记忆单元。
   - 参数：
     - `query_text` (str): 自然语言查询文本
     - `top_k` (int): 返回结果数量，默认为5
     - `ms_names` (Optional[List[str]]): 指定记忆空间名称列表，若不指定则在全图范围内查找
     - `recursive` (bool): 是否递归包含子ms，默认为True
   - 返回：记忆单元列表

2. **get_explicit_neighbors(uids, rel_type=None, ms_names=None, direction="successors", recursive=True)**
   - 功能：获取指定记忆单元的显式关系邻居节点，可限定ms范围。
   - 参数：
     - `uids` (List[str]): 源节点ID列表
     - `rel_type` (Optional[str]): 关系类型筛选
     - `ms_names` (Optional[List[str]]): 指定记忆空间名称列表，若不指定则在全图范围内查找
     - `direction` (str): 邻居方向，可选："successors"(出边邻居)，"predecessors"(入边邻居)，"all"(双向邻居)，默认为"successors"
     - `recursive` (bool): 是否递归包含子ms，默认为True
   - 返回：记忆单元列表

3. **filter_memory_units(candidate_units=None, filter_condition=None, ms_names=None, recursive=True)**
   - 功能：通过特定字段的值和操作符过滤记忆单元，支持ms范围限定。
   - 参数：
     - `candidate_units` (Optional[List[MemoryUnit]]): 候选记忆单元列表，若为None则使用全图记忆单元
     - `filter_condition` (Optional[Dict]): 过滤条件，**每个字段必须为操作符字典**，如：
       `{{ "field": {{"eq": value}} }}`、`{{"field": {{"in": [v1, v2]}}}}`、`{{"field": {{"gte": 2020, "lte": 2024}}}}`
       支持的操作符有：
         - `eq`（等于）
         - `ne`（不等于）
         - `in`（属于列表/集合）
         - `nin`（不属于列表/集合）
         - `gt`（大于）
         - `gte`（大于等于）
         - `lt`（小于）
         - `lte`（小于等于）
         - `contain`（op_val in val，适用于val为list/set/str等可迭代对象）
         - `not_contain`（op_val not in val，适用于val为list/set/str等可迭代对象）
     - `ms_names` (Optional[List[str]]): 指定记忆空间名称列表，若不指定则在全图范围内查找
     - `recursive` (bool): 是否递归包含子ms，默认为True
   - 返回：过滤后的记忆单元列表

4. **get_implicit_neighbors(uids, top_k=5, ms_names=None, recursive=True)**
   - 功能：获取指定记忆单元（可为多个起点）的隐式关系邻居节点，可限定ms范围。
   - 参数：
     - uids: 记忆单元ID列表（支持多个起点）
     - top_k: 返回的邻居数量，默认为5
     - ms_names: 指定记忆空间名称列表，若不指定则在全图范围内查找
     - recursive: 是否递归包含子ms，默认为True
   - 返回：记忆单元列表

5. **get_units_in_memory_space(ms_names, recursive=True)**
   - 功能：获取指定ms及其子ms下所有unit。
   - 参数：
     - ms_names: 记忆空间名称列表
     - recursive: 是否递归包含子ms，默认为True
   - 返回：记忆单元列表

6. **deduplicate_units(units)**
   - 功能：对unit列表去重，避免重叠带来的重复。
   - 参数：
     - units: 记忆单元列表
   - 返回：去重后的记忆单元列表

7. **aggregate_results(memory_units)**
   - 功能：对查询结果进行聚合操作，统计记忆单元的出现频率
   - 参数：
     - memory_units: 记忆单元列表
   - 返回：聚合结果，记忆单元出现频率

8. **units_union(*args)**
   - 功能：支持多个MemorySpace、MemoryUnit列表、UID列表的并集，返回去重后的MemoryUnit列表
   - 参数：
     - *args: 支持多种类型的参数：
       - MemoryUnit对象
       - UID字符串（单个或列表）
       - MemorySpace对象
       - MemoryUnit列表
       - 上述类型的混合列表/元组/集合
   - 返回：去重后的MemoryUnit列表

9. **units_intersection(*args)**
   - 功能：支持多个MemorySpace、MemoryUnit列表、UID列表的交集，返回去重后的MemoryUnit列表
   - 参数：
     - *args: 支持多种类型的参数（同units_union）
   - 返回：去重后的MemoryUnit列表（取第一个参数的unit对象）

10. **units_difference(arg1, arg2)**
    - 功能：返回arg1中有而arg2中没有的unit（按uid），支持MemorySpace、MemoryUnit列表、UID列表
    - 参数：
      - arg1: 第一个参数，支持多种类型（同units_union）
      - arg2: 第二个参数，支持多种类型（同units_union）
    - 返回：arg1中有而arg2中没有的MemoryUnit列表

## 查询计划格式与变量引用规则

查询计划采用JSON格式，支持步骤间的变量引用机制：

### 变量引用规则
- 使用 `$变量名` 引用之前步骤的完整结果
- 使用 `$变量名.属性名` 引用结果中的特定属性
- 支持嵌套属性访问：`$变量名.属性1.属性2`
- 支持跨ms、合并多个ms结果

### 格式示例
```json
{{
  "plan": [
    {{
      "step": 1,
      "operation": "get_units_in_memory_space",
      "parameters": {{
        "ms_names": ["AI文档", "NLP相关"],
        "recursive": true
      }},
      "result_var": "candidate_units"
    }},
    {{
      "step": 2,
      "operation": "filter_memory_units",
      "parameters": {{
        "candidate_units": "$candidate_units",
        "filter_condition": {{"type": {{"eq": "论文"}}}}
      }},
      "result_var": "filtered_papers"
    }},
    {{
      "step": 3,
      "operation": "deduplicate_units",
      "parameters": {{
        "units": "$filtered_papers"
      }},
      "result_var": "unique_papers"
    }}
  ]
}}
```

## 任务要求

请根据以下用户自然语言查询，分析查询意图，并使用提供的API生成一个能有效检索所需信息的查询计划。

### 用户查询
```
{query}
```

## 输出要求

- 只输出JSON格式的查询计划，不需要解释或说明过程
- 确保JSON格式正确无误，可以被程序直接解析
- 合理使用变量引用机制，提高查询效率
- 确保查询计划能够准确回答用户的问题
"""
        return prompt

    def query_llm(self, prompt):
        prompt = self._generate_prompt(prompt)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:32b", "prompt": prompt, "stream": False},
        )
        response_json = response.json()
        return response_json.get("response", "")

    def generate_query_plan(
        self,
        natural_language_query,
    ) -> Optional[Dict]:
        result = self.query_llm(natural_language_query)
        match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
        if not match:
            logging.error("Failed to extract JSON query plan from LLM response.")
            return None
        structured_plan = match.group(1).strip()

        try:
            plan = json.loads(structured_plan)
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")
            return None
        return plan

    def _resolve_variable(self, value, temp_results):
        """
        支持 $var, $var.field, $var.field1.field2 形式的变量解析，支持对象属性提取。
        针对 MemoryUnit，uid 用属性，其它字段用 raw_data。
        """
        if not (isinstance(value, str) and value.startswith("$")):
            return value

        parts = value[1:].split(".")
        var = temp_results.get(parts[0], None)
        if var is None:
            return None

        def extract(obj, keys):
            if not keys:
                return obj
            key = keys[0]
            rest_keys = keys[1:]
            if isinstance(obj, list):
                results = [extract(item, keys) for item in obj if item is not None]
                results = [r for r in results if r is not None]
                if not results:
                    return None
                if len(results) == 1:
                    return results[0]
                return results
            elif isinstance(obj, dict):
                return extract(obj.get(key, None), rest_keys)
            else:
                # 针对 MemoryUnit 特殊处理
                if obj.__class__.__name__ == "MemoryUnit":
                    if key == "uid":
                        attr = getattr(obj, "uid", None)
                        return extract(attr, rest_keys)
                    else:
                        raw_data = getattr(obj, "raw_data", {})
                        attr = raw_data.get(key, None)
                        return extract(attr, rest_keys)
                # 其它对象按原有逻辑
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    return extract(attr, rest_keys)
                else:
                    return None

        return extract(var, parts[1:])

    def execute_query_plan(self, structured_query: Dict) -> List[MemoryUnit]:
        plan = structured_query.get("plan", None)
        if not plan:
            raise ValueError("Structured query doesnot have plan.")

        temp_results = {}
        for step in plan:
            oper = step.get("operation", None)
            params: Dict[str, Any] | None = step.get("parameters", None)
            res = step.get("result_var", None)

            # 递归解析参数中的变量
            def resolve_params(obj):
                if isinstance(obj, dict):
                    return {k: resolve_params(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [resolve_params(v) for v in obj]
                elif isinstance(obj, str) and obj.startswith("$"):
                    return self._resolve_variable(obj, temp_results)
                else:
                    return obj

            processed_params = resolve_params(params) if params else {}

            print(
                f"执行操作：{oper}，参数：{processed_params}，暂存为：{res}，已有结果：{temp_results}"
            )

            if hasattr(self.semantic_graph, oper):
                # 确保 processed_params 是字典类型
                if isinstance(processed_params, dict):
                    temp_results[res] = getattr(self.semantic_graph, oper)(
                        **processed_params
                    )
                else:
                    temp_results[res] = getattr(self.semantic_graph, oper)(
                        processed_params
                    )
            else:
                raise ValueError(f"Operation {oper} not found in SemanticGraph.")

        return list(temp_results.values())[-1] if temp_results else []


if __name__ == "__main__":
    sgraph = SemanticGraph.load_graph("path/to/your/semantic_graph dir")

    planner = QueryPlanner(sgraph)

    natural_language_query = "Recommend some restaurants in California with good environment and spicy Chinese food."

    """
    查找我朋友们最喜欢的海鲜餐厅
    找到我朋友们在洛杉矶拍过的餐厅照片
    查找我朋友们在海鲜餐厅发布的评论和小费
    找到在纽约写过很多评论并且喜欢意大利菜的用户
    找到我朋友们评论过的海鲜餐厅
    """

    # planner.generate_prompt(natural_language_query, "")
    plan = planner.generate_query_plan(natural_language_query)

    if plan:
        results = planner.execute_query_plan(plan)

        print(f"\n\nQuery: {natural_language_query}, Results =>")
        for r in results:
            print(r.__repr__())
    else:
        print("Failed to parse the query.")
