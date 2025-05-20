"""
AgentMemorySystem - 高效的智能体记忆系统

该包提供了用于构建和管理智能体记忆的核心组件，包括:
- 语义记忆表示和存储 (MemoryUnit)
- 向量和关系双重索引 (SemanticMap, SemanticGraph)
- 数据库集成 (Neo4j, Milvus)
- 记忆检索和管理工具
"""

__version__ = "0.1.0"

# 导出核心类，使它们可以直接从包导入
from .Hippo import (
    # MemoryUnit,
    MemorySpace,
    SemanticMap,
    SemanticGraph
)

# 导出数据库连接器
from .milvus_operator import MilvusOperator
from .neo4j_operator import Neo4jOperator
from .memory_unit import MemoryUnit

# 可选：设置默认日志配置
import logging
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# 提供一个便捷函数来创建完整的记忆系统
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
    创建并初始化一个完整配置的记忆系统
    
    参数:
        neo4j_uri: Neo4j服务器URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        neo4j_database: Neo4j数据库名
        milvus_host: Milvus服务器地址
        milvus_port: Milvus服务器端口
        embedding_dim: 嵌入向量维度
        image_model_path: 可选的图像模型本地路径
        text_model_path: 可选的文本模型本地路径
        
    返回:
        配置好的SemanticGraph实例
    """
    # 创建SemanticMap
    semantic_map = SemanticMap(
        image_embedding_model_name=image_model_path or "clip-ViT-B-32",
        text_embedding_model_name=text_model_path or "clip-ViT-B-32-multilingual-v1",
        embedding_dim=embedding_dim
    )
    
    # 连接到Milvus
    semantic_map.connect_external_storage(
        storage_type="milvus",
        host=milvus_host,
        port=milvus_port,
        collection_name="hippo_memory_units"
    )
    
    # 创建SemanticGraph
    semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
    
    # 连接到Neo4j
    semantic_graph.connect_to_neo4j(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database
    )
    
    return semantic_graph

# 提供实用工具函数
def clear_memory_system():
    """清除所有记忆系统数据"""
    from .clear import clear_neo4j, clear_milvus, clear_local_files
    clear_neo4j()
    clear_milvus()
    clear_local_files()