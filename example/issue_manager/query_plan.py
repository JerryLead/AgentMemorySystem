from core.semantic_graph import SemanticGraph
from advanced.query_planner import QueryPlanner


class IssueManagerQueryPlanner(QueryPlanner):
    def query_llm(self, query):
        return """
```json
{
  "plan": [
    {
      "step": 1,
      "operation": "search_similarity_in_graph",
      "parameters": {
        "query_text": "Issue#1968",
        "top_k": 5,
        "ms_names": ["github_issues"],
        "recursive": true
      },
      "result_var": "similar_issues"
    },
    {
      "step": 2,
      "operation": "get_explicit_neighbors",
      "parameters": {
        "uids": "$similar_issues.uid",
        "rel_type": "corresponding_pr",
        "ms_names": ["github_issues", "github_prs"],
        "direction": "successors",
        "recursive": true
      },
      "result_var": "related_prs"
    },
    {
      "step": 3,
      "operation": "filter_memory_units",
      "parameters": {
        "candidate_units": "$related_prs",
        "filter_condition": {
          "state": {
            "in": ["closed", "merged"]
          }
        }
      },
      "result_var": "closed_prs"
    },
    {
      "step": 4,
      "operation": "deduplicate_units",
      "parameters": {
        "units": "$closed_prs"
      },
      "result_var": "final_prs"
    }
  ]
}
```
"""


if __name__ == "__main__":
    sgraph = SemanticGraph.load_graph(
        "data/issue_manager",
        image_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32",
        text_embedding_model_name="/mnt/data1/home/guozy/gzy/models/clip-ViT-B-32-multilingual-v1",
    )
    sgraph.build_semantic_map_index()

    planner = IssueManagerQueryPlanner(sgraph)

    natural_language_query = "历史上有没有修复过与 Issue#1968 类似问题的 PR？"

    """
    谁是这个项目中在处理与“GraphQL 查询性能优化”相关问题时经验最丰富的开发者？
    这个特定的 Bug 报告 (Issue #123) 最终是由哪个开发者提交的哪次代码修改 (Commit) 修复的？修改了哪些文件？
    上周开发者“Alice”合并的那个大型重构 PR (#456) 都影响了哪些文件？这些文件之前关联过哪些 Issue？这些 Issue 中有没有提到过潜在的性能问题？
    我正在处理一个关于“身份验证令牌过期”的新 Issue (#789)。历史上有没有修复过类似问题的 PR？这些 PR 是由谁处理的？它们都修改了哪些核心文件？
    经常一起 Review 对方 PR 的开发者有哪些组合？他们合作修复的主要是哪类问题（基于关联 Issue 的语义主题）？
    文件 src/auth/service.py 历史上被哪些 Commit 频繁修改？这些修改关联的 PR 主要是为了解决哪些类型的问题（从关联 Issue 中提取主题）？有没有反复出现的同类问题？
    这个即将合并的 PR (#101)，它修改的文件，历史上有没有被其他在处理“用户会话管理”问题的 PR 修改过？那些 PR 有没有引入过难以解决的 Bug？
    """

    # 可选：打印生成的prompt
    print(planner._generate_prompt(natural_language_query))
    exit(0)
    plan = planner.generate_query_plan(natural_language_query)

    if plan:
        results = planner.execute_query_plan(plan)
        print(f"\n\nQuery: {natural_language_query}, Results =>")
        for r in results:
            print(r.__repr__())
    else:
        print("Failed to parse the query.")

# python -m example.issue_manager.query_plan
