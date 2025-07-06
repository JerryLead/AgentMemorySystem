import numpy as np
import faiss
import pytest
import time
from core.Hippo import MemoryUnit, MemorySpace, SemanticMap


def make_units(dim=64, n=30):
    return [
        MemoryUnit(
            f"u{i}",
            {"text_content": f"test {i}"},
            embedding=np.random.rand(dim).astype(np.float32),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize(
    "index_factory",
    [
        None,  # 默认L2
        lambda dim: faiss.IndexFlatIP(dim),
        lambda dim: faiss.IndexHNSWFlat(dim, 16),
    ],
)
def test_memoryspace_index_factory(index_factory):
    dim = 64
    units = make_units(dim, 30)
    print(f"\n[test_memoryspace_index_factory] index_factory={index_factory}")
    ms = MemorySpace("test", index_factory=index_factory)
    for u in units:
        ms.add(u)
    t0 = time.time()
    ms.build_index(embedding_dim=dim, min_unit_threshold=10)
    print(f"build_index耗时: {time.time()-t0:.3f}s")
    # 检查索引类型
    if index_factory is None:
        assert isinstance(ms._emb_index, faiss.IndexFlatL2)
    else:
        assert ms._emb_index is not None
    query = units[0].embedding
    t1 = time.time()
    results = ms.search_similarity_units_by_vector(query, top_k=5)
    print(
        f"search_similarity_units_by_vector耗时: {time.time()-t1:.3f}s, 结果数: {len(results)}"
    )
    assert len(results) > 0


@pytest.mark.parametrize(
    "index_factory",
    [
        None,
        lambda dim: faiss.IndexFlatIP(dim),
        lambda dim: faiss.IndexHNSWFlat(dim, 16),
    ],
)
def test_semanticmap_index_factory(index_factory):
    dim = 64
    units = make_units(dim, 30)
    print(f"\n[test_semanticmap_index_factory] index_factory={index_factory}")
    sm = SemanticMap(index_factory=index_factory)
    for u in units:
        sm.register_unit(u)
    t0 = time.time()
    sm.build_index()
    print(f"build_index耗时: {time.time()-t0:.3f}s")
    if index_factory is None:
        assert isinstance(sm._emb_index, faiss.IndexFlatL2)
    else:
        assert sm._emb_index is not None
    query = units[0].embedding
    t1 = time.time()
    results = sm.search_similarity_units_by_vector(query, top_k=5)
    print(
        f"search_similarity_units_by_vector耗时: {time.time()-t1:.3f}s, 结果数: {len(results)}"
    )
    assert len(results) > 0


# 运行方法：
# 在项目根目录下执行 python -m pytest -s tests/test_faiss_index_types.py