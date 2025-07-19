import argparse
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
          "state": {"eq": "CA"},
          "categories": {"contain": "Chinese"}
        }
      },
      "result_var": "ca_chinese_restaurants"
    },
    {
      "step": 2,
      "operation": "get_explicit_neighbors",
      "parameters": {
        "uids": "$ca_chinese_restaurants.uid",
        "rel_type": "review_for",
        "direction": "predecessors",
        "ms_names": ["review"]
      },
      "result_var": "restaurant_reviews"
    },
    {
      "step": 3,
      "operation": "search_similarity_in_graph",
      "parameters": {
        "candidate_units": "$restaurant_reviews",
        "query_text": "good environment and spicy Chinese food",
        "top_k": 10
      },
      "result_var": "similar_reviews"
    },
    {
      "step": 4,
      "operation": "get_explicit_neighbors",
      "parameters": {
        "uids": "$similar_reviews.uid",
        "rel_type": "review_for",
        "direction": "successors"
      },
      "result_var": "recommended_restaurants_temp"
    },
    {
      "step": 5,
      "operation": "deduplicate_units",
      "parameters": {
        "units": "$recommended_restaurants_temp"
      },
      "result_var": "final_recommendations"
    }
  ]
}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Yelp Query Planner CLI")
    parser.add_argument(
        "-p", "--print-prompt", action="store_true", help="只输出生成的 prompt"
    )
    parser.add_argument(
        "-e",
        "--execute",
        type=str,
        default=None,
        help="只执行前 N 步或指定步（如 3 或 1,2,4）",
    )
    parser.add_argument("-q", "--query", type=str, default=None, help="自然语言查询")
    args = parser.parse_args()

    sgraph = SemanticGraph.load_graph(
        "data/yelp",
        image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
        text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
    )
    sgraph.build_semantic_map_index()

    planner = YelpQueryPlanner(sgraph)

    natural_language_query = (
        args.query
        or "Recommend some restaurants in California with good environment and spicy Chinese food."
    )

    """
    查找我朋友们最喜欢的海鲜餐厅
    找到我朋友们在洛杉矶拍过的餐厅照片
    查找我朋友们在海鲜餐厅发布的评论和小费
    找到在纽约写过很多评论并且喜欢意大利菜的用户
    找到我朋友们评论过的海鲜餐厅
    """
    if args.print_prompt:
        prompt = planner._generate_prompt(natural_language_query)
        print(prompt)
        return

    plan = planner.generate_query_plan(natural_language_query)
    if not plan:
        print("Failed to parse the query.")
        return

    if args.execute:
        # 支持 --execute 3 或 --execute 1,2,4
        steps = []
        if "," in args.execute:
            steps = [int(x) for x in args.execute.split(",") if x.strip().isdigit()]
        else:
            try:
                n = int(args.execute)
                steps = list(range(1, n + 1))
            except ValueError:
                print("--execute 参数格式错误，应为整数或逗号分隔的步号")
                return
        # 只保留指定步
        if isinstance(plan, dict) and "plan" in plan:
            plan["plan"] = [step for step in plan["plan"] if step.get("step") in steps]
        else:
            print("计划格式不支持")
            return
        results = planner.execute_query_plan(plan)
        print(f"\n\nQuery: {natural_language_query}, Results =>")
        for r in results:
            print(r.__repr__())
        return

    # 默认完整执行
    results = planner.execute_query_plan(plan)
    print(f"\n\nQuery: {natural_language_query}, Results =>")
    for r in results:
        print(r.__repr__())


# python -m example.yelp.query_plan
if __name__ == "__main__":
    main()
