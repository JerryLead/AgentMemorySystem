import unittest
from unittest.mock import MagicMock
from advanced.query_planner import QueryPlanner
from core.memory_unit import MemoryUnit
from core.semantic_graph import SemanticGraph

class TestEnd2EndIntegration(unittest.TestCase):
    def setUp(self):
        # 构建完整的SemanticGraph
        self.graph = SemanticGraph()
        self.unit1 = MemoryUnit(uid="u1", raw_data={"text": "加州川菜馆", "type": "餐厅", "city": "加州", "flavor": "辣"})
        self.unit2 = MemoryUnit(uid="u2", raw_data={"text": "北京烤鸭店", "type": "餐厅", "city": "北京", "flavor": "咸"})
        self.unit3 = MemoryUnit(uid="u3", raw_data={"text": "四川火锅", "type": "餐厅", "city": "加州", "flavor": "辣"})
        self.unit4 = MemoryUnit(uid="u4", raw_data={"text": "洛杉矶意大利餐厅", "type": "餐厅", "city": "洛杉矶", "flavor": "鲜"})
        for u in [self.unit1, self.unit2, self.unit3, self.unit4]:
            self.graph.add_unit(u)
        self.graph.semantic_map.create_memory_space("加州美食")
        self.graph.semantic_map.add_unit_to_space(self.unit1, "加州美食")
        self.graph.semantic_map.add_unit_to_space(self.unit3, "加州美食")
        self.graph.semantic_map.create_memory_space("北京美食")
        self.graph.semantic_map.add_unit_to_space(self.unit2, "北京美食")
        self.graph.semantic_map.create_memory_space("洛杉矶美食")
        self.graph.semantic_map.add_unit_to_space(self.unit4, "洛杉矶美食")
        # 显式关系
        self.graph.add_relationship("u1", "u3", "同城推荐")
        self.graph.add_relationship("u2", "u1", "跨城推荐")
        self.planner = QueryPlanner(self.graph)

    def test_end2end_query_california_spicy(self):
        # mock LLM返回一个合理的结构化查询计划
        self.planner.query_llm = MagicMock(return_value='''```json\n{"plan": [
    {"step": 1, "operation": "get_units_in_memory_space", "parameters": {"ms_names": ["加州美食"]}, "result_var": "units_in_ca"},
    {"step": 2, "operation": "filter_memory_units", "parameters": {"candidate_units": "$units_in_ca", "filter_condition": {"flavor": {"eq": "辣"}}}, "result_var": "spicy_ca"}
]}\n```')
        plan = self.planner.generate_query_plan("查找加州美食空间下所有辣味餐厅")
        self.assertIsInstance(plan, dict)
        result = self.planner.execute_query_plan(plan)
        self.assertIsInstance(result, list)
        uids = [u.uid for u in result]
        self.assertIn("u1", uids)
        self.assertIn("u3", uids)
        self.assertNotIn("u2", uids)
        self.assertNotIn("u4", uids)

    def test_end2end_query_explicit_relation(self):
        # mock LLM返回显式关系查询
        self.planner.query_llm = MagicMock(return_value='''```json\n{"plan": [
    {"step": 1, "operation": "get_explicit_neighbors", "parameters": {"uids": ["u1"], "rel_type": "同城推荐"}, "result_var": "neighbors"}
]}\n```')
        plan = self.planner.generate_query_plan("查找与u1有同城推荐关系的餐厅")
        self.assertIsInstance(plan, dict)
        result = self.planner.execute_query_plan(plan)
        self.assertIsInstance(result, list)
        uids = [u.uid for u in result]
        self.assertIn("u3", uids)
        self.assertNotIn("u1", uids)

    def test_end2end_query_type_filter(self):
        # mock LLM返回类型过滤
        self.planner.query_llm = MagicMock(return_value='''```json\n{"plan": [
    {"step": 1, "operation": "filter_memory_units", "parameters": {"filter_condition": {"type": {"eq": "餐厅"}}}, "result_var": "all_restaurants"}
]}\n```''')
        plan = self.planner.generate_query_plan("查找所有类型为餐厅的单元")
        self.assertIsInstance(plan, dict)
        result = self.planner.execute_query_plan(plan)
        self.assertIsInstance(result, list)
        uids = [u.uid for u in result]
        self.assertIn("u1", uids)
        self.assertIn("u2", uids)
        self.assertIn("u3", uids)
        self.assertIn("u4", uids)

if __name__ == "__main__":
    unittest.main()

# 运行方法：
# python -m pytest tests/test_end2end_integration.py 