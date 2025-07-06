import pytest
from core.Hippo import MemoryUnit, MemorySpace, SemanticGraph

@pytest.fixture
def simple_graph():
    # 创建两个unit和两个memoryspace
    u1 = MemoryUnit("u1", {"text_content": "A"})
    u2 = MemoryUnit("u2", {"text_content": "B"})
    ms1 = MemorySpace("ms1")
    ms2 = MemorySpace("ms2")
    ms1.add(u1)
    ms2.add(u2)
    sg = SemanticGraph()
    sg.add_memory_space(ms1)
    sg.add_memory_space(ms2)
    sg.add_unit(u1)
    sg.add_unit(u2)
    return sg, u1, u2, ms1, ms2

def test_unit_to_unit_edge(simple_graph):
    sgraph, u1, u2, ms1, ms2 = simple_graph
    sgraph.add_explicit_edge(u1, u2, rel_type="refers")
    assert sgraph.nx_graph.has_edge(u1.uid, u2.uid)
    edge = sgraph.nx_graph.get_edge_data(u1.uid, u2.uid)
    assert edge["type"] == "refers"

def test_space_to_unit_edge(simple_graph):
    sgraph, u1, u2, ms1, ms2 = simple_graph
    sgraph.add_explicit_edge(ms1, u2, rel_type="contains")
    ms1_id = f"ms:{ms1.name}"
    assert sgraph.nx_graph.has_edge(ms1_id, u2.uid)
    edge = sgraph.nx_graph.get_edge_data(ms1_id, u2.uid)
    assert edge["type"] == "contains"

def test_unit_to_space_edge(simple_graph):
    sgraph, u1, u2, ms1, ms2 = simple_graph
    sgraph.add_explicit_edge(u1, ms2, rel_type="belongs")
    ms2_id = f"ms:{ms2.name}"
    assert sgraph.nx_graph.has_edge(u1.uid, ms2_id)
    edge = sgraph.nx_graph.get_edge_data(u1.uid, ms2_id)
    assert edge["type"] == "belongs"

def test_space_to_space_edge(simple_graph):
    sgraph, u1, u2, ms1, ms2 = simple_graph
    sgraph.add_explicit_edge(ms1, ms2, rel_type="related")
    ms1_id = f"ms:{ms1.name}"
    ms2_id = f"ms:{ms2.name}"
    assert sgraph.nx_graph.has_edge(ms1_id, ms2_id)
    edge = sgraph.nx_graph.get_edge_data(ms1_id, ms2_id)
    assert edge["type"] == "related"

# 运行方法：
# python -m pytest tests/test_semanticgraph_edges.py
