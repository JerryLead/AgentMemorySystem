import unittest
from core.stdHippo import SemanticMap, MemoryUnit

class TestSemanticMapDBConn(unittest.TestCase):
    def setUp(self):
        self.smap = SemanticMap()
        self.unit = MemoryUnit(uid="u1", raw_data={"text": "db test"})
        self.smap.add_unit(self.unit)

    def test_connect_external_storage(self):
        # 这里只测试接口可调用，不要求真实连接
        try:
            result = self.smap.connect_external_storage(
                storage_type="milvus", host="localhost", port="19530"
            )
            self.assertIn(result, [True, False])
        except Exception as e:
            self.fail(f"connect_external_storage接口调用异常: {e}")

    def test_sync_to_external(self):
        # 测试同步接口可调用
        try:
            count = self.smap.sync_to_external(force_full_sync=False)
            self.assertIsInstance(count, (tuple, list, dict))
        except Exception as e:
            self.fail(f"sync_to_external接口调用异常: {e}")

    def test_swap_out(self):
        # 测试换页接口可调用
        try:
            self.smap._max_memory_units = 1
            self.smap.swap_out(count=1)
        except Exception as e:
            self.fail(f"swap_out接口调用异常: {e}")

    def test_load_from_external(self):
        # 测试加载接口可调用
        try:
            count = self.smap.load_from_external(filter_space=None, limit=1)
            self.assertIsInstance(count, int)
        except Exception as e:
            self.fail(f"load_from_external接口调用异常: {e}")

if __name__ == "__main__":
    unittest.main()

# 运行方法：
# python -m pytest tests/test_stdHippo_dbconn.py 