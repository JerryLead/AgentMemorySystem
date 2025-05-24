"""
AgentMemorySystem - 高效的智能体记忆系统
包的主入口，提供标准的Python包接口
"""

__version__ = "0.1.0"

# 导出核心类 - 这是包的标准接口
from .semantic_map import SemanticMap
from .semantic_graph import SemanticGraph
from .memory_unit import MemoryUnit
from .memory_space import MemorySpace
from .milvus_operator import MilvusOperator
from .neo4j_operator import Neo4jOperator

__all__ = [
    # 核心类
    'SemanticMap', 
    'SemanticGraph', 
    'MemoryUnit', 
    'MemorySpace',
    # 操作类
    'Neo4jOperator', 
    'MilvusOperator',
    # 工厂函数
    'create_memory_system'
]

# 设置默认日志配置
import logging
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# 提供一个便捷的工厂函数
def create_memory_system(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j", 
    neo4j_password="20031117",
    neo4j_database="academicgraph",
    milvus_host="localhost",
    milvus_port="19530",
    embedding_dim=512,
    image_model_path=None,
    text_model_path=None
):
    """
    工厂函数：创建并初始化一个完整配置的记忆系统
    这是推荐的创建记忆系统的方式
    """
    semantic_map = SemanticMap(
        image_embedding_model_name=image_model_path or "clip-ViT-B-32",
        text_embedding_model_name=text_model_path or "clip-ViT-B-32-multilingual-v1",
        embedding_dim=embedding_dim
    )
    
    semantic_map.connect_external_storage(
        storage_type="milvus",
        host=milvus_host,
        port=milvus_port,
        collection_name="hippo_memory_units"
    )
    
    semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
    
    semantic_graph.connect_to_neo4j(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database
    )
    
    return semantic_graph