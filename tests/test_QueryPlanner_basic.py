import unittest
from unittest.mock import MagicMock, Mock
from advanced.QueryPlanner import QueryPlanner


class TestQueryPlannerBasic(unittest.TestCase):
    def setUp(self):
        # 用Mock对象模拟SemanticGraph
        mock_graph = Mock()
        mock_graph.get_all_memory_space_names.return_value = ["美食"]
        mock_graph.get_memory_space_structures.return_value = [
            {"name": "美食", "unit_uids": ["u1", "u2"], "unit_fields": ["text", "type"]}
        ]
        mock_graph.get_all_relations.return_value = ["friend", "colleague"]

        class DummyUnit:
            def __init__(self, uid):
                self.uid = uid

        mock_graph.get_units_in_memory_space.return_value = [
            DummyUnit("u1"),
            DummyUnit("u2"),
        ]
        self.planner = QueryPlanner(mock_graph)

    def test_generate_query_plan(self):
        # mock query_llm 只返回一个固定结构化计划
        self.planner.query_llm = MagicMock(
            return_value="""```json\n{"plan": [{"step": 1, "operation": "get_units_in_memory_space", "parameters": {"ms_names": ["美食"]}, "result_var": "units"}]}\n```"""
        )
        plan = self.planner.generate_query_plan("查找美食空间下的所有餐厅")
        self.assertIsInstance(plan, dict)
        self.assertIn("plan", plan)
        self.assertEqual(plan["plan"][0]["operation"], "get_units_in_memory_space")

    def test_execute_query_plan(self):
        # 直接用mock的plan
        plan = {
            "plan": [
                {
                    "step": 1,
                    "operation": "get_units_in_memory_space",
                    "parameters": {"ms_names": ["美食"]},
                    "result_var": "units",
                }
            ]
        }
        result = self.planner.execute_query_plan(plan)
        # 应该返回DummyUnit列表
        self.assertIsInstance(result, list)
        self.assertTrue(any(u.uid == "u1" for u in result))
        self.assertTrue(any(u.uid == "u2" for u in result))


if __name__ == "__main__":
    unittest.main()

# 运行方法：
# python -m pytest tests/test_QueryPlanner_basic.py
