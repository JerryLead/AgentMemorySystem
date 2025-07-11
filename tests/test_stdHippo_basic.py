import unittest
import numpy as np
from core.stdHippo import MemoryUnit, MemorySpace, SemanticMap


class TestMemoryUnitAndSpace(unittest.TestCase):
    def test_memory_unit_basic(self):
        # 创建MemoryUnit
        mu = MemoryUnit(uid="u1", raw_data={"text": "测试内容", "type": "文档"})
        self.assertEqual(mu.uid, "u1")
        self.assertEqual(mu.raw_data["text"], "测试内容")
        self.assertIsInstance(mu.metadata, dict)
        self.assertIsNone(mu.embedding)
        # __str__ 和 __repr__
        self.assertIn("u1", str(mu))
        self.assertIn("u1", repr(mu))
        # __eq__ 和 __hash__
        mu2 = MemoryUnit(
            uid="u1",
            raw_data={"text": "测试内容", "type": "文档"},
            metadata=mu.metadata,
        )
        self.assertEqual(mu, mu2)
        self.assertEqual(hash(mu), hash(mu2))

    def test_memory_space_basic(self):
        ms = MemorySpace("space1")
        mu1 = MemoryUnit(uid="u1", raw_data={"text": "A"})
        mu2 = MemoryUnit(uid="u2", raw_data={"text": "B"})
        # 添加单元
        ms.add_unit(mu1)
        ms.add_unit(mu2.uid)
        self.assertIn("u1", ms.get_unit_uids())
        self.assertIn("u2", ms.get_unit_uids())
        # 移除单元
        ms.remove_unit(mu1)
        self.assertNotIn("u1", ms.get_unit_uids())
        # 嵌套空间
        ms2 = MemorySpace("space2")
        ms.add_child_space(ms2)
        self.assertIn("space2", ms.get_child_space_names())
        ms.remove_child_space("space2")
        self.assertNotIn("space2", ms.get_child_space_names())

    def test_memory_space_recursive(self):
        from core.stdHippo import SemanticMap

        smap = SemanticMap()
        ms1 = smap.create_memory_space("root")
        ms2 = smap.create_memory_space("child1")
        ms3 = smap.create_memory_space("child2")
        mu1 = MemoryUnit(uid="u1", raw_data={})
        mu2 = MemoryUnit(uid="u2", raw_data={})
        ms1.add_unit(mu1)
        ms2.add_unit(mu2)
        ms1.add_child_space(ms2)
        ms2.add_child_space(ms3)
        all_uids = ms1.get_all_unit_uids(recursive=True)
        self.assertIn("u1", all_uids)
        self.assertIn("u2", all_uids)


class TestSemanticMapBasic(unittest.TestCase):
    def test_semantic_map_add_and_get(self):
        smap = SemanticMap()
        mu = MemoryUnit(uid="u1", raw_data={"text": "hello"})
        smap.add_unit(mu)
        self.assertIn("u1", smap.memory_units)
        self.assertEqual(smap.get_unit("u1"), mu)

    def test_semantic_map_space(self):
        smap = SemanticMap()
        mu = MemoryUnit(uid="u1", raw_data={"text": "A"})
        smap.add_unit(mu)
        smap.add_unit_to_space(mu, "spaceA")
        units = smap.get_units_in_memory_space("spaceA")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].uid, "u1")
        # 多空间
        mu2 = MemoryUnit(uid="u2", raw_data={"text": "B"})
        smap.add_unit(mu2)
        smap.add_unit_to_space(mu2, "spaceA")
        smap.add_unit_to_space(mu2, "spaceB")
        unitsA = smap.get_units_in_memory_space(["spaceA"])
        unitsB = smap.get_units_in_memory_space(["spaceB"])
        self.assertIn(mu2, unitsA)
        self.assertIn(mu2, unitsB)


if __name__ == "__main__":
    unittest.main()

# 运行方法：
# python -m pytest tests/test_stdHippo_basic.py
