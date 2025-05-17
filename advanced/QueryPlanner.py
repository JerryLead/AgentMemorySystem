import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import logging
import uuid
import pickle

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
    def __init__(self, semantic_graph: SemanticSimpleGraph):
        self.semantic_graph = semantic_graph
        self.cache = {}

    def generate_prompt(self, query):
        #     arxiv_graph_structure = """
        # ArxivGraph 的结构如下：
        # - 根节点：以 "{paper_id}_title_authors" 命名，包含论文的标题（title）和作者（authors）信息。
        # - 摘要节点：以 "{paper_id}_abstract" 命名，包含论文的摘要（abstract），与根节点通过 "has_abstract" 关系相连。
        # - 章节节点：以 "{paper_id}_chapter_{idx}" 命名，包含章节标题（title），与根节点通过 "has_chapter" 关系相连。每个章节节点下可能包含段落（paragraphs）和图片（images）子节点。
        # - 表格行节点：以 "{paper_id}_table_{table_idx}_row_{row_idx}" 命名，包含表格行的数据，与根节点通过 "has_table_row" 关系相连。

        # ArxivGraph 可以有多个实例，每个实例代表一篇独立的 arXiv 论文，通过不同的 paper_id 进行区分。
        # """

        #     paper_examples = """
        # 以下是几个 paper 的样例：
        # - paper: {
        #     "paper_id": "2304.01234",
        #     "title": "Advances in Machine Learning Techniques",
        #     "authors": ["Alice Smith", "Bob Johnson"],
        #     "abstract": "This paper explores the latest advancements in machine learning techniques and their applications.",
        #     "chapters": [
        #         {
        #             "title": "Introduction",
        #             "paragraphs": ["Machine learning has become an important field in recent years.", "It has various applications in different industries."]
        #         },
        #         {
        #             "title": "Related Work",
        #             "paragraphs": ["Many researchers have contributed to the development of machine learning algorithms.", "This section reviews some of the key works."]
        #         }
        #     ],
        #     "references": ["Smith, A. (2022). Machine Learning Basics. Journal of AI Research.", "Johnson, B. (2023). Advanced Machine Learning Models. AI Today."],
        #     "tables": [
        #         {
        #             "Column1": "Model Type",
        #             "Column2": "Accuracy"
        #         },
        #         {
        #             "Column1": "Neural Network",
        #             "Column2": "0.9"
        #         }
        #     ]
        # }
        # - paper: {
        #     "paper_id": "2305.05678",
        #     "title": "Natural Language Processing in Healthcare",
        #     "authors": ["Charlie Brown", "David Davis"],
        #     "abstract": "This research focuses on the application of natural language processing in the healthcare industry.",
        #     "chapters": [
        #         {
        #             "title": "Overview",
        #             "paragraphs": ["Natural language processing can be used to analyze medical records.", "It has the potential to improve patient care."]
        #         },
        #         {
        #             "title": "Case Studies",
        #             "paragraphs": ["Several case studies are presented to demonstrate the effectiveness of NLP in healthcare.", "These studies show promising results."]
        #         }
        #     ],
        #     "references": ["Brown, C. (2023). NLP for Medical Text Analysis. Healthcare Informatics Journal.", "Davis, D. (2023). NLP Applications in Patient Diagnosis. Medical AI Review."],
        #     "tables": [
        #         {
        #             "Column1": "Task",
        #             "Column2": "Performance"
        #         },
        #         {
        #             "Column1": "Medical Text Classification",
        #             "Column2": "0.85"
        #         }
        #     ]
        # }
        # """

        #     prompt = f"""
        # {query}

        # Please convert the above natural language query into a JSON format query plan. The query plan should follow this structure:
        # [
        #     {{
        #         "step": 1,
        #         "target": ["paper", "abstract", ...],
        #         "constraints": {{
        #             "semantic_query": "your semantic query here",
        #             "filter": {{"attribute": "value"}}
        #         }},
        #         "input": null
        #     }},
        #     ...
        # ]

        # Use the following data structure information for reference:
        # - paper: {{
        #     "paper_id": "xxxx.xxxxx",
        #     "title": "Paper Title",
        #     "authors": ["Author1", "Author2"],
        #     "abstract": "Abstract text",
        #     "chapters": [{{"title": "Chapter 1", "paragraphs": ["Para 1", "Para 2"]}}],
        #     "references": ["Reference 1", "Reference 2"],
        #     "tables": [{{"Column1": "Value1", "Column2": "Value2"}}]
        # }}

        # {arxiv_graph_structure}
        # {paper_examples}
        # """
        #     print(f"生成的提示信息: {prompt}")

        prompt = f"""
### Task Description:
You will receive a natural language query, and your goal is to break it down into a sequence of steps that correspond to a query plan for a multimodal dataset, such as Yelp. The query plan should be structured as a sequence of steps, where each step contains:

- "step": The step number in the query sequence.
- "target": The list of target data structures that will be queried ("review", "tip", "photo", "business", "user").
- "constraints": The filtering or querying constraints, which can include:
  - "semantic_query": A textual semantic query for matching data (e.g., "seafood restaurant", only "review", "tip", "photo" can have this constraints).
  - "filter": Structured filters based on specific attributes.
- "input": A list of results from previous steps (if any), which will be used as input for the next step.

### Data Structures:
Below are the structures of the data in the Yelp dataset, including the relationship between different entities (e.g., a user writes a review for a business, a business has tips and photos):

#### business:
{{
  "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
  "name": "Garaje",
  "address": "475 3rd St",
  "city": "San Francisco",
  "state": "CA",
  "stars": 4.5,
  "review_count": 1198,
}}

#### review:
{{
  "review_id": "zdSx_SD6obEhz9VrW9uAWA",
  "user_id": "Ha3iJu77CxlrFm-vQRs_8g",
  "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
  "stars": 4,
  "useful": 0,
  "funny": 0,
  "cool": 0
}}

#### user:
{{
  "user_id": "Ha3iJu77CxlrFm-vQRs_8g",
  "name": "Sebastien",
  "review_count": 56,
  "friends": ["wqoXYLWmpkEH0YvTmHBsJQ", "KUXLLiJGrjtSsapmxmpvTA", "6e9rJKQC3n0RSKyHLViL-Q"],
  "useful": 21,
  "funny": 88,
  "cool": 15,
  "fans": 1032,
  "average_stars": 4.31,
}}

#### tip:
{{
  "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
  "user_id": "49JhAJh8vSQ-vM4Aourl0g"
}}

#### photo:
{{
  "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
  "label": "food"
}}

### Example Query:
** Natural language query: ** "Find my friends' favorite seafood restaurants."

### Output Format:
The query plan should be in the following JSON format:
[
  {{
    "step": 1,
    "target": ["review", "tip", "photo"],
    "constraints": {{
      "semantic_query": "seafood restaurant"
    }},
    "input": null
  }},
  {{
    "step": 2,
    "target": ["business"],
    "constraints": {{
      "filter": {{"business_id": {{"$in": ["step1"]}}}}
    }},
    "input": [1]
  }},
  {{
    "step": 3,
    "target": ["review", "tip"],
    "constraints": {{
      "filter": {{"business_id": {{"$in": ["step2"]}}}}
    }},
    "input": [2]
  }},
  {{
    "step": 4,
    "target": ["user"],
    "constraints": {{
      "filter": {{"user_id": {{"$in": ["friends list", "step3"]}}}}
    }},
    "input": [3]
  }},
  {{
    "step": 5,
    "target": ["review", "tip"],
    "constraints": {{
      "filter": {{"user_id": {{"$in": ["step4"]}}, "review_id": {{"$in": ["step3"]}}, "tip_id": {{"$in": ["step3"]}}}}
    }},
    "input": [3, 4]
  }},
  {{
    "step": 6,
    "target": ["business"],
    "constraints": {{
      "filter": {{"business_id": {{"$in": ["step5"]}}}}
    }},
    "input": [5]
  }}
]

### Instructions:
+ Break down the natural language query into multiple steps, each corresponding to a specific data structure.
+ Use the provided data structure formats to guide your query steps and constraints.
+ Use relationships between data structures to guide your step sequence (e.g., a user writes reviews, a business has reviews, etc.).
Please convert the following natural language query into a json format query plan: ** Natural language query: ** "{query}"
"""
        return prompt

    def query_llm(self, query):
        prompt = self.generate_prompt(query)
        # response = requests.post(
        #     "http://localhost:11434/api/generate",
        #     json={"model": "deepseek-r1:32b", "prompt": prompt, "stream": False},
        # )
        # response_json = response.json()
        # return response_json.get("response", "")

        return """
[
  {
    "step": 1,
    "target": ["review", "tip", "photo"],
    "constraints": {
      "semantic_query": "spicy Chinese food"
    },
    "input": null
  },
  {
    "step": 2,
    "target": ["business"],
    "constraints": {
      "filter": {
        "business_id": {"$in": ["step1"]},
        "state": "CA"
      }
    },
    "input": [1]
  },
  {
    "step": 3,
    "target": ["review", "tip", "photo"],
    "constraints": {
      "semantic_query": "good environment",
      "filter": {
        "business_id": {"$in": ["step2"]}
      }
    },
    "input": [2]
  },
  {
    "step": 4,
    "target": ["business"],
    "constraints": {
      "filter": {
        "business_id": {"$in": ["step3"]},
        "stars": {"$gte": 4.0}
      }
    },
    "input": [3]
  }
]
"""

    def parse_query(
        self,
        category_map: SemanticMap,
        attribute_map: SemanticMap,
        natural_language_query,
    ):
        result = self.query_llm(natural_language_query)

        try:
            structured_query = json.loads(result)
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")
            return None

        target_type_map = {
            "business": DataType.BUSINESS,
            "user": DataType.USER,
            "review": DataType.REVIEW,
            "photo": DataType.PHOTO,
            "tip": DataType.TIP,
        }

        for step in structured_query:
            targets = step.get("target")
            step["target"] = []
            for target in targets:
                target = target_type_map.get(target)
                step["target"].append(target)
                if target is None:
                    logging.error(f"Unsupported target type: {target}")
                    return None

            # # Handle categories and attributes for business targets with filters
            # if DataType.BUSINESS in step["target"] and step.get("constraints", {}).get("filter"):
            #     query_emb = self.semantic_graph.semantic_map._get_text_embedding(
            #         natural_language_query
            #     ).reshape(1, -1)

            #     # Get top-3 categories
            #     distances, indices = category_map.index.search(query_emb, 3)
            #     categories = [category_map.data[i][1]["text"] for i in indices[0]]

            #     # Get top-3 attributes
            #     distances, indices = attribute_map.index.search(query_emb, 3)
            #     attributes = [attribute_map.data[i][1]["text"] for i in indices[0]]

            #     # Add categories and attributes to filters
            #     step["constraints"]["filter"]["categories"] = {"$in": categories}
            #     step["constraints"]["filter"]["attributes"] = {"$in": attributes}

        res = [
            StructuredQuery(
                step=query["step"],
                target_types=query["target"],
                inputs=query["input"],
                filters=query["constraints"].get("filter"),
                semantic_query=query["constraints"].get("semantic_query"),
            )
            for query in structured_query
        ]
        return res

    def execute_plan(self, plan: list[StructuredQuery]):
        for query_step in plan:
            step_results = []
            step_number = query_step.step  # this step number
            print("step: ", step_number)
            self.cache[f"step{step_number}"] = []

            step_inputs = []
            if query_step.inputs:
                for input_num in query_step.inputs:
                    for cache in self.cache[f"step{input_num}"]:
                        step_inputs.append(cache)
            if step_inputs:
                print("input_step: ", step_inputs[0][:-1])

            print("target types: ", query_step.target_types)
            candidates = [
                data
                for data in self.semantic_graph.semantic_map.data
                if data[2] in query_step.target_types
            ]
            if candidates:
                print("candidates: ", candidates[0][:-1])

            if query_step.filters:
                # Apply filters to the semantic graph
                filtered_results = []
                for candidate in candidates:
                    key, value, datatype, emb = candidate
                    match = True
                    for filter_key, filter_value in query_step.filters.items():
                        if isinstance(filter_value, dict) and "$in" in filter_value:
                            step_num = filter_value["$in"][0]
                            if not if_key_in_map(key, self.cache[f"{step_num}"]):
                                match = False
                                break
                        elif isinstance(filter_value, dict) and "$gte" in filter_value:
                            if value.get(filter_key, 0) < filter_value["$gte"]:
                                match = False
                                break
                        elif value.get(filter_key) != filter_value:
                            match = False
                            break
                    if match:
                        filtered_results.append(candidate)
                step_results = filtered_results
            else:
                step_results = candidates

            if query_step.semantic_query:
                # Perform semantic search on the filtered results
                step_results = self.semantic_search(
                    query_step.semantic_query, step_results
                )

            if step_results:
                print("results: ", step_results[0][:-1])
            self.cache[f"step{step_number}"] = step_results
            print("caches: ", self.cache.keys())

        # If the final target is business, find the most popular reviews for each business
        if DataType.BUSINESS in query_step.target_types:
            final_results = []
            for business in step_results:
                business_id = business[0]
                reviews = [
                    data
                    for data in self.semantic_graph.semantic_map.data
                    if data[2] == DataType.REVIEW
                    and data[1]["business_id"] == business_id
                ]
                # Sort reviews by some popularity metric, e.g., 'useful' votes
                sorted_reviews = sorted(
                    reviews,
                    key=lambda x: x[1].get("useful", 0)
                    + x[1].get("funny", 0)
                    + x[1].get("cool", 0),
                    reverse=True,
                )
                top_reviews = sorted_reviews[:3]  # Get top 3 reviews
                final_results.append({"business": business, "top_reviews": top_reviews})
            return final_results

        return step_results

    def apply_filters(self, filters):
        # Apply filters to the semantic graph and return filtered results
        filtered_keys = []
        for key, value, _, _ in self.semantic_graph.semantic_map.data:
            match = all(value.get(k) == v for k, v in filters.items())
            if match:
                filtered_keys.append(key)
        return filtered_keys

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

    sgraph = SemanticSimpleGraph()
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

    structured_query = planner.parse_query(None, None, natural_language_query)
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
