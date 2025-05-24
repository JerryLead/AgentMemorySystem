"""
Hippo - Agent Memory System Demo
示例脚本，展示如何使用记忆系统的各种功能
适合学习和测试使用
"""

import logging
from .semantic_map import SemanticMap
from .semantic_graph import SemanticGraph  
from .memory_unit import MemoryUnit

def run_basic_demo():
    """基础功能演示"""
    logging.info("=== 基础功能演示 ===")
    
    # 创建基础组件
    semantic_map = SemanticMap(embedding_dim=512)
    semantic_graph = SemanticGraph(semantic_map_instance=semantic_map)
    
    # 创建示例数据
    unit1 = MemoryUnit(
        uid="demo_ai", 
        raw_data={
            "text_content": "这是一个关于人工智能的文档",
            "description": "AI介绍"
        }
    )
    
    # 添加到图谱
    semantic_graph.add_unit(unit1, space_names=["demo"])
    semantic_graph.build_semantic_map_index()
    
    # 搜索演示
    results = semantic_map.search_similarity_by_text("机器学习", k=1)
    for unit, score in results:
        logging.info(f"找到相似内容: {unit.uid}, 得分: {score:.4f}")

def run_database_demo():
    """数据库集成演示"""
    logging.info("=== 数据库集成演示 ===")
    
    # 创建连接到数据库的记忆系统
    from . import create_memory_system
    
    try:
        system = create_memory_system()
        logging.info("数据库连接成功")
        
        # 添加测试数据
        test_unit = MemoryUnit(
            uid="db_test",
            raw_data={"text_content": "数据库测试内容"}
        )
        system.add_unit(test_unit)
        
        # 同步到数据库
        stats = system.sync_to_external()
        logging.info(f"同步结果: {stats}")
        
    except Exception as e:
        logging.error(f"数据库演示失败: {e}")

def run_advanced_demo():
    """高级功能演示"""
    logging.info("=== 高级功能演示 ===")
    
    # 演示关系管理、换页等高级功能
    pass

if __name__ == '__main__':
    """当直接运行此文件时执行演示"""
    logging.info("开始 Hippo 记忆系统演示...")
    
    # 运行各种演示
    run_basic_demo()
    run_database_demo() 
    run_advanced_demo()
    
    logging.info("演示完成")