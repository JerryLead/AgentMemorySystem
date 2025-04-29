import os
import sys
import json
import time
from datetime import datetime
import traceback

# 确保导入路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from academic_group_meeting_graph import (
    AcademicDataType, 
    NamespaceType, 
    AcademicGroupMeetingMap, 
    AcademicGroupMeetingGraph
)
from semantic_map.neo4j_interface import Neo4jInterface
from milvus_enabled_academic_meeting_system import MilvusEnabledAcademicMeetingSystem

# 设置日志
import logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neo4j_test")

def safe_execute_query(neo4j, query, params=None):
    """安全执行Neo4j查询，增加错误处理"""
    try:
        logger.debug(f"执行查询: {query}")
        if params:
            logger.debug(f"参数: {params}")
        
        result = neo4j.execute_query(query, params)
        
        logger.debug(f"查询结果类型: {type(result)}")
        if result:
            logger.debug(f"查询结果内容: {result[:100] if len(str(result)) > 100 else result}")
        else:
            logger.debug("查询结果为空")
        
        return result
    except Exception as e:
        logger.error(f"执行查询出错: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def create_mock_academic_system():
    """创建模拟的学术组会系统，包含基本的测试数据"""
    
    # 初始化系统
    system = MilvusEnabledAcademicMeetingSystem(
        use_remote_llm=False,
        use_local_embeddings=True,
        milvus_host="localhost",
        milvus_port="19530",
        milvus_collection="test_academic_dialogues",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="20031117",
        neo4j_database="academicgraph"
    )
    
    # 创建一个测试场景
    scene_id = "test_scene_001"
    system.backend.add_research_topic(
        scene_id, 
        {
            "Title": "测试学术讨论主题",
            "Description": "这是一个用于测试Neo4j导出功能的模拟学术讨论",
            "Keywords": ["测试", "学术", "Neo4j"],
            "CreatedAt": datetime.now().isoformat()
        }
    )
    
    # 创建用户
    professor_id = "professor_test_001"
    system.backend.create_agent(
        professor_id, 
        {
            "Nickname": "张教授",
            "Role": "Professor",
            "Specialty": "人工智能",
            "Background": "资深AI研究员"
        }
    )
    
    phd_id = "phd_test_001"
    system.backend.create_agent(
        phd_id, 
        {
            "Nickname": "李博士",
            "Role": "PhD",
            "Specialty": "机器学习",
            "Background": "博士研究生"
        }
    )
    
    # 添加消息
    message1_id = system.add_message(
        scene_id, 
        professor_id, 
        "今天我们讨论一下语义图在学术研究中的应用。"
    )
    
    message2_id = system.add_message(
        scene_id, 
        phd_id, 
        "我认为语义图可以帮助我们更好地组织知识结构。"
    )
    
    # 添加总结
    summary_id = system.add_summary(
        scene_id,
        professor_id,
        "今天的讨论主要围绕语义图的应用展开，涉及知识组织和检索方面的优势。",
        summary_type="round",
        round_num=1,
        related_message_ids=[message1_id, message2_id]
    )
    
    # 添加任务
    try:
        task_ids = system.generate_tasks_from_summary(summary_id)
        print(f"生成的任务ID: {task_ids}")
    except Exception as e:
        logger.error(f"生成任务时出错: {str(e)}")
        task_ids = []
    
    return system, scene_id

def print_semantic_graph_structure(system:MilvusEnabledAcademicMeetingSystem):
    """打印语义图的详细结构"""
    logger.info("======== 语义图结构详情 ========")
    
    # 打印命名空间信息
    logger.info("--- 命名空间信息 ---")
    for namespace in NamespaceType:
        logger.info(f"命名空间: {namespace.name}")
        
        # 获取此命名空间下的节点
        nodes = [
            node_id for node_id in system.backend.semantic_graph.namespace_nodes.get(namespace, [])
        ]
        logger.info(f"  包含节点数量: {len(nodes)}")
        
        # 打印前5个节点示例
        if nodes:
            logger.info(f"  节点示例: {', '.join(nodes[:5])}")
    
    # 打印节点类型分布
    logger.info("--- 节点类型分布 ---")
    node_types = {}
    for _, _, datatype, _ in system.backend.semantic_map.data:
        if datatype not in node_types:
            node_types[datatype] = 0
        node_types[datatype] += 1
    
    for datatype, count in node_types.items():
        logger.info(f"  {datatype}: {count}个节点")
    
    # 打印关系统计
    logger.info("--- 关系统计 ---")
    relation_count = 0
    relation_types = {}
    
    for src in system.backend.semantic_graph.graph_relations:
        for rel_type, targets in system.backend.semantic_graph.graph_relations[src].items():
            if rel_type == "children":
                for target in targets:
                    rel = system.backend.semantic_graph.get_relation_type(src, target)
                    relation_count += 1
                    if rel not in relation_types:
                        relation_types[rel] = 0
                    relation_types[rel] += 1
    
    logger.info(f"总关系数量: {relation_count}")
    for rel_type, count in relation_types.items():
        logger.info(f"  {rel_type}: {count}个关系")
    
    logger.info("======== 语义图结构详情结束 ========")

def test_namespace_insertion(system, neo4j_interface):
    """测试每个命名空间的单独插入"""
    logger.info("======== 测试每个命名空间的单独插入 ========")
    
    # 清理数据库
    logger.info("清理数据库...")
    safe_execute_query(neo4j_interface, "MATCH (n) DETACH DELETE n")
    
    # 遍历每个命名空间，单独创建
    for namespace in NamespaceType:
        namespace_id = f"namespace_{namespace.name}"
        logger.info(f"\n插入命名空间: {namespace.name} (ID: {namespace_id})")
        
        # 创建命名空间节点
        query = """
        CREATE (n:Namespace {id: $id, name: $name, description: $description})
        RETURN n
        """
        params = {
            "id": namespace_id,
            "name": namespace.name,
            "description": f"{namespace.name} namespace"
        }
        
        result = safe_execute_query(neo4j_interface, query, params)
        if result and len(result) > 0:
            logger.info(f"✓ 命名空间 {namespace.name} 创建成功")
        else:
            logger.error(f"✗ 命名空间 {namespace.name} 创建失败")
        
        # 验证命名空间创建
        verify_query = "MATCH (n:Namespace {id: $id}) RETURN n"
        verify_result = safe_execute_query(neo4j_interface, verify_query, {"id": namespace_id})
        
        if verify_result and len(verify_result) > 0:
            logger.info(f"✓ 命名空间 {namespace.name} 验证成功")
            if 'n' in verify_result[0]:
                logger.info(f"  节点属性: {verify_result[0]['n']}")
        else:
            logger.error(f"✗ 命名空间 {namespace.name} 验证失败")
    
    # 验证所有命名空间
    all_namespaces_query = "MATCH (n:Namespace) RETURN n.name as name"
    all_result = safe_execute_query(neo4j_interface, all_namespaces_query)
    
    if all_result:
        inserted_namespaces = [r.get('name', '') for r in all_result if 'name' in r]
        logger.info(f"\n所有已插入的命名空间: {inserted_namespaces}")
        
        # 验证是否所有命名空间都存在
        all_namespace_names = [ns.name for ns in NamespaceType]
        missing = set(all_namespace_names) - set(inserted_namespaces)
        
        if missing:
            logger.error(f"⚠ 缺少命名空间: {missing}")
        else:
            logger.info("✓ 所有命名空间均已正确插入")
    else:
        logger.error("✗ 无法检索命名空间节点")
    
    logger.info("======== 命名空间插入测试结束 ========")

def diagnose_neo4j_interface(system):
    """诊断Neo4j接口功能"""
    logger.info("======== Neo4j接口诊断 ========")
    
    try:
        # 创建接口实例
        neo4j = Neo4jInterface(
            uri=system.neo4j_uri,
            user=system.neo4j_user,
            password=system.neo4j_password,
            database=system.neo4j_database
        )
        
        # 测试连接
        logger.info("测试基本连接...")
        test_query = "RETURN 1 as test"
        result = safe_execute_query(neo4j, test_query)
        
        if result and len(result) > 0 and 'test' in result[0]:
            logger.info(f"✓ 基本连接测试成功: {result[0]['test']}")
        else:
            logger.error("✗ 基本连接测试失败")
            return False, neo4j
        
        # 测试创建节点
        logger.info("测试创建节点...")
        create_query = """
        CREATE (n:TestNode {id: 'test123', name: 'test node'})
        RETURN n
        """
        create_result = safe_execute_query(neo4j, create_query)
        
        if create_result and len(create_result) > 0:
            logger.info("✓ 节点创建测试成功")
        else:
            logger.error("✗ 节点创建测试失败")
        
        # 测试创建关系
        logger.info("测试创建关系...")
        rel_query = """
        CREATE (a:TestNode {id: 'source123', name: 'source'})
        CREATE (b:TestNode {id: 'target123', name: 'target'})
        CREATE (a)-[r:TEST_RELATION]->(b)
        RETURN a, b
        """
        rel_result = safe_execute_query(neo4j, rel_query)
        
        if rel_result and len(rel_result) > 0:
            logger.info("✓ 关系创建测试成功")
        else:
            logger.error("✗ 关系创建测试失败")
        
        # 测试关系查询
        rel_check_query = """
        MATCH (a:TestNode {id: 'source123'})-[r:TEST_RELATION]->(b:TestNode {id: 'target123'})
        RETURN type(r) as rel_type
        """
        rel_check_result = safe_execute_query(neo4j, rel_check_query)
        
        if rel_check_result and len(rel_check_result) > 0:
            logger.info(f"✓ 关系查询测试成功: {rel_check_result[0].get('rel_type', 'unknown')}")
        else:
            logger.error("✗ 关系查询测试失败")
        
        # 清理测试数据
        logger.info("清理测试数据...")
        safe_execute_query(neo4j, "MATCH (n:TestNode) DETACH DELETE n")
        
        logger.info("======== Neo4j接口诊断完成 ========")
        return True, neo4j
        
    except Exception as e:
        logger.error(f"Neo4j接口诊断出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None

def inspect_semantic_map_data(system:MilvusEnabledAcademicMeetingSystem):
    """检查语义图数据结构，帮助调试"""
    logger.info("======== 检查语义图数据结构 ========")
    
    # 获取图数据
    data_count = len(system.backend.semantic_map.data)
    logger.info(f"语义图中的数据项数量: {data_count}")
    
    # 输出前5个数据项
    logger.info("语义图数据样本:")
    for i, (key, value, datatype, _) in enumerate(system.backend.semantic_map.data[:5]):
        logger.info(f"数据项 {i+1}:")
        logger.info(f"  ID: {key}")
        logger.info(f"  类型: {datatype}")
        
        # 截取值的部分内容输出
        if isinstance(value, dict):
            value_preview = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool))}
            logger.info(f"  值: {json.dumps(value_preview, ensure_ascii=False)[:100]}...")
    
    # 打印节点和关系的数量
    node_count = len(system.backend.semantic_graph.graph_relations)
    
    relation_count = 0
    for src in system.backend.semantic_graph.graph_relations:
        if "children" in system.backend.semantic_graph.graph_relations[src]:
            relation_count += len(system.backend.semantic_graph.graph_relations[src]["children"])
    
    logger.info(f"图中的节点数量: {node_count}")
    logger.info(f"图中的关系数量: {relation_count}")
    
    logger.info("======== 语义图数据结构检查结束 ========")

def detailed_error_check(system: MilvusEnabledAcademicMeetingSystem):
    """详细检查可能的错误点"""
    logger.info("======== 详细错误检查 ========")
    
    # 检查非字符串键
    non_string_keys = []
    for key, _, _, _ in system.backend.semantic_map.data:
        if not isinstance(key, str):
            non_string_keys.append((key, type(key)))
    
    if non_string_keys:
        logger.error(f"发现非字符串键: {non_string_keys}")
    else:
        logger.info("✓ 所有键均为字符串类型")
    
    # 检查空属性
    empty_props = []
    for key, value, _, _ in system.backend.semantic_map.data:
        if not value:
            empty_props.append(key)
    
    if empty_props:
        logger.warning(f"发现空属性节点: {empty_props}")
    else:
        logger.info("✓ 所有节点均有属性")
    
    # 检查复杂嵌套结构
    complex_props = []
    for key, value, _, _ in system.backend.semantic_map.data:
        for k, v in value.items() if isinstance(value, dict) else {}:
            if isinstance(v, (list, dict)) and v:
                complex_props.append((key, k, type(v)))
    
    if complex_props:
        logger.info(f"发现复杂嵌套属性 (可能需要序列化): {complex_props[:10]}")
    
    # 检查图的关系一致性
    for src in system.backend.semantic_graph.graph_relations:
        # 检查源节点是否存在
        src_exists = any(key == src for key, _, _, _ in system.backend.semantic_map.data)
        if not src_exists:
            logger.error(f"关系中的源节点不存在于数据中: {src}")
        
        # 检查目标节点
        if "children" in system.backend.semantic_graph.graph_relations[src]:
            for dst in system.backend.semantic_graph.graph_relations[src]["children"]:
                dst_exists = any(key == dst for key, _, _, _ in system.backend.semantic_map.data)
                if not dst_exists:
                    logger.error(f"关系中的目标节点不存在于数据中: {src} -> {dst}")
    
    logger.info("======== 详细错误检查结束 ========")

def test_full_export(system: MilvusEnabledAcademicMeetingSystem):
    """测试完整的Neo4j导出功能"""
    logger.info("======== 测试完整Neo4j导出 ========")
    
    try:
        # 导出到Neo4j
        logger.info("开始导出数据到Neo4j...")
        stats = system.export_to_neo4j(
            create_constraints=True
        )
        logger.info(f"导出结果: {stats}")
        
        # 创建连接以验证导出结果
        neo4j = Neo4jInterface(
            uri=system.neo4j_uri,
            user=system.neo4j_user,
            password=system.neo4j_password,
            database=system.neo4j_database
        )
        
        # 验证命名空间节点
        logger.info("验证导出的命名空间节点...")
        namespace_query = "MATCH (n:Namespace) RETURN count(n) as count"
        namespace_result = safe_execute_query(neo4j, namespace_query)
        
        if namespace_result and len(namespace_result) > 0 and 'count' in namespace_result[0]:
            namespace_count = namespace_result[0]['count']
            logger.info(f"命名空间节点数量: {namespace_count}")
        else:
            logger.error("无法获取命名空间节点数量")
        
        # 验证各类型节点
        logger.info("验证导出的节点...")
        node_query = "MATCH (n) RETURN labels(n) as labels, count(*) as count"
        node_result = safe_execute_query(neo4j, node_query)
        
        if node_result:
            for row in node_result:
                if 'labels' in row and 'count' in row:
                    logger.info(f"节点类型 {row['labels']}: {row['count']}个")
        else:
            logger.error("无法获取节点分布信息")
        
        # 验证关系数量
        logger.info("验证导出的关系...")
        rel_query = "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
        rel_result = safe_execute_query(neo4j, rel_query)
        
        if rel_result:
            for row in rel_result:
                if 'type' in row and 'count' in row:
                    logger.info(f"关系类型 {row['type']}: {row['count']}个")
        else:
            logger.error("无法获取关系分布信息")
        
        # 总结验证结果
        total_nodes_query = "MATCH (n) RETURN count(n) as count"
        total_nodes_result = safe_execute_query(neo4j, total_nodes_query)
        
        total_rels_query = "MATCH ()-[r]->() RETURN count(r) as count"
        total_rels_result = safe_execute_query(neo4j, total_rels_query)
        
        total_nodes = total_nodes_result[0]['count'] if total_nodes_result and len(total_nodes_result) > 0 and 'count' in total_nodes_result[0] else 0
        total_rels = total_rels_result[0]['count'] if total_rels_result and len(total_rels_result) > 0 and 'count' in total_rels_result[0] else 0
        
        logger.info(f"总结: 导出了 {total_nodes} 个节点和 {total_rels} 个关系")
        
        # 检查与预期是否一致
        expected_nodes = len(system.backend.semantic_map.data) + len(NamespaceType)  # 数据节点加命名空间节点
        expected_rels = sum(len(system.backend.semantic_graph.graph_relations[src].get("children", {})) 
                           for src in system.backend.semantic_graph.graph_relations)
        
        if total_nodes >= expected_nodes * 0.9:  # 允许90%的容差
            logger.info(f"✓ 节点导出基本符合预期: {total_nodes}/{expected_nodes}")
        else:
            logger.warning(f"⚠ 节点导出数量不符预期: {total_nodes}/{expected_nodes}")
        
        if total_rels >= expected_rels * 0.9:  # 允许90%的容差
            logger.info(f"✓ 关系导出基本符合预期: {total_rels}/{expected_rels}")
        else:
            logger.warning(f"⚠ 关系导出数量不符预期: {total_rels}/{expected_rels}")
        
        logger.info("======== 完整Neo4j导出测试结束 ========")
        return True, {"nodes": total_nodes, "relationships": total_rels}
    
    except Exception as e:
        logger.error(f"完整Neo4j导出测试出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False, str(e)

def main():
    """主测试函数"""
    start_time = time.time()
    logger.info("=== 开始Neo4j插入功能测试 ===")
    
    # 创建模拟数据
    system, scene_id = create_mock_academic_system()
    logger.info(f"创建的测试场景ID: {scene_id}")
    
    # 打印语义图结构
    print_semantic_graph_structure(system)
    
    # 检查语义图数据
    inspect_semantic_map_data(system)
    
    # 详细错误检查
    detailed_error_check(system)
    
    # 诊断Neo4j接口
    neo4j_ok, neo4j = diagnose_neo4j_interface(system)
    if not neo4j_ok:
        logger.error("Neo4j接口诊断失败，终止测试")
        return
    
    # 测试命名空间插入
    test_namespace_insertion(system, neo4j)
    
    # 测试完整Neo4j导出
    export_success, export_result = test_full_export(system)

    # 测试语义图打印
    logger.info("打印语义图结构...")
    data = system.backend.semantic_graph.print_str_graph(show=False)
    print(data)
    
    if export_success:
        logger.info(f"Neo4j导出测试成功! 导出了 {export_result['nodes']} 个节点和 {export_result['relationships']} 个关系")
    else:
        logger.error(f"Neo4j导出测试失败: {export_result}")
    
    end_time = time.time()
    logger.info(f"=== 测试完成，耗时 {end_time - start_time:.2f} 秒 ===")

if __name__ == "__main__":
    main()