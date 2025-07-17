from core.semantic_graph import SemanticGraph
from advanced.query_planner import QueryPlanner


class YelpQueryPlanner(QueryPlanner):
    def query_llm(self, query):
        return """
```json
{
  "plan": [
    {
      "step": 1,
      "operation": "filter_memory_units",
      "parameters": {
        "ms_names": ["business"],
        "filter_condition": {
          "city": {"eq": "New York"},
          "categories": {"contain": "Italian"}
        }
      },
      "result_var": "ny_italian_businesses"
    },
    {
      "step": 2,
      "operation": "get_explicit_neighbors",
      "parameters": {
        "uids": "$ny_italian_businesses.uid",
        "rel_type": "review_for",
        "direction": "predecessors"
      },
      "result_var": "business_reviews"
    },
    {
      "step": 3,
      "operation": "get_explicit_neighbors",
      "parameters": {
        "uids": "$business_reviews.uid",
        "rel_type": "write_review",
        "direction": "predecessors"
      },
      "result_var": "review_authors"
    },
    {
      "step": 4,
      "operation": "filter_memory_units",
      "parameters": {
        "candidate_units": "$review_authors",
        "filter_condition": {"review_count": {"gt": 50}}
      },
      "result_var": "active_users"
    },
    {
      "step": 5,
      "operation": "deduplicate_units",
      "parameters": {
        "units": "$active_users"
      },
      "result_var": "final_users"
    }
  ]
}
```
"""


if __name__ == "__main__":
    sgraph = SemanticGraph.load_graph(
        "data/yelp",
        image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
        text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
    )
    sgraph.build_semantic_map_index()

    planner = YelpQueryPlanner(sgraph)

    natural_language_query = "找到在纽约写过很多评论并且喜欢意大利菜的用户。"

    """
    查找我朋友们最喜欢的海鲜餐厅
    找到我朋友们在洛杉矶拍过的餐厅照片
    查找我朋友们在海鲜餐厅发布的评论和小费
    找到在纽约写过很多评论并且喜欢意大利菜的用户
    找到我朋友们评论过的海鲜餐厅
    """

    # 可选：打印生成的prompt
    # print(planner._generate_prompt(natural_language_query))
    # exit(0)
    plan = planner.generate_query_plan(natural_language_query)

    if plan:
        results = planner.execute_query_plan(plan)
        print(f"\n\nQuery: {natural_language_query}, Results =>")
        for r in results:
            print(r.__repr__())
    else:
        print("Failed to parse the query.")

# python -m example.yelp.query_plan
