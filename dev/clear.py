import logging
import os
import shutil
from neo4j import GraphDatabase
from pymilvus import connections, utility, Collection

# 配置日志记录器
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def clear_neo4j(uri="bolt://localhost:7687", user="neo4j", password="20031117", database="academicgraph"):
    """
    清除Neo4j数据库中与MemoryUnit相关的所有节点和关系
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            # 删除所有MemoryUnit节点及其关系
            result = session.run("MATCH (n:MemoryUnit) DETACH DELETE n")
            summary = result.consume()
            
            logging.info(f"从Neo4j清除了 {summary.counters.nodes_deleted} 个节点和 {summary.counters.relationships_deleted} 个关系")
            
            # 删除其他可能存在的测试节点和关系(根据需要添加)
            
        driver.close()
        return True
    except Exception as e:
        logging.error(f"清除Neo4j数据失败: {e}")
        return False

def clear_milvus(host="localhost", port="19530", collection_names=["hippo_memory_units", "my_memory_units"]):
    """
    清除Milvus数据库中的集合
    """
    try:
        # 连接到Milvus
        connections.connect("default", host=host, port=port)
        
        for collection_name in collection_names:
            if utility.has_collection(collection_name):
                # 方法1: 删除集合
                utility.drop_collection(collection_name)
                logging.info(f"已删除Milvus集合: {collection_name}")
                
                # 方法2: 仅清空集合内容(保留结构)
                # collection = Collection(collection_name)
                # collection.drop()
                # logging.info(f"已清空Milvus集合: {collection_name}")
                
        connections.disconnect("default")
        return True
    except Exception as e:
        logging.error(f"清除Milvus数据失败: {e}")
        return False

def clear_local_files(save_dir="hippo_save_data"):
    """
    删除本地保存的数据文件
    """
    try:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            logging.info(f"已删除本地数据目录: {save_dir}")
        else:
            logging.info(f"本地数据目录不存在: {save_dir}")
        return True
    except Exception as e:
        logging.error(f"删除本地数据文件失败: {e}")
        return False

if __name__ == "__main__":
    logging.info("开始清除测试数据...")
    
    # 清除Neo4j数据
    neo4j_success = clear_neo4j()
    
    # 清除Milvus数据
    milvus_success = clear_milvus()
    
    # 清除本地文件
    files_success = clear_local_files()
    
    if neo4j_success and milvus_success and files_success:
        logging.info("所有测试数据已成功清除")
    else:
        logging.warning("部分测试数据清除失败，请检查日志")