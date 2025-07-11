import unittest
from core.stdHippo import SemanticGraph, MemoryUnit


class TestSemanticGraphBasic(unittest.TestCase):
    def setUp(self):
        self.graph = SemanticGraph()
        self.unit1 = MemoryUnit(uid="u1", raw_data={"text": "A"})
        self.unit2 = MemoryUnit(uid="u2", raw_data={"text": "B"})
        self.unit3 = MemoryUnit(uid="u3", raw_data={"text": "C"})
        self.graph.add_unit(self.unit1)
        self.graph.add_unit(self.unit2)
        self.graph.add_unit(self.unit3)

    def test_add_and_traverse_explicit_relationship(self):
        # 添加关系
        self.graph.add_relationship("u1", "u2", "friend")
        self.graph.add_relationship("u2", "u3", "colleague")
        # 显式遍历
        neighbors = self.graph.get_explicit_neighbors(["u1"], rel_type="friend")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].uid, "u2")
        # 方向测试
        neighbors_pred = self.graph.get_explicit_neighbors(
            ["u2"], rel_type="friend", direction="predecessors"
        )
        self.assertEqual(len(neighbors_pred), 1)
        self.assertEqual(neighbors_pred[0].uid, "u1")

    def test_delete_unit_and_relationship(self):
        self.graph.add_relationship("u1", "u2", "friend")
        self.graph.delete_unit("u2")
        # u2被删除后，关系也应消失
        neighbors = self.graph.get_explicit_neighbors(["u1"], rel_type="friend")
        self.assertEqual(len(neighbors), 0)

    def test_implicit_neighbors(self):
        # 由于没有embedding模型，隐式邻居只测试接口可调用
        try:
            result = self.graph.get_implicit_neighbors(["u1", "u2"], top_k=2)
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"get_implicit_neighbors接口调用异常: {e}")

    def test_memory_space_in_graph(self):
        # 测试空间相关接口
        self.graph.semantic_map.create_memory_space("spaceA")
        self.graph.semantic_map.add_unit_to_space(self.unit1, "spaceA")
        units = self.graph.get_units_in_memory_space(["spaceA"])
        self.assertIn(self.unit1, units)


if __name__ == "__main__":
    unittest.main()

# 运行方法：
# python -m pytest tests/test_SemanticGraph_basic.py
