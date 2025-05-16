import sys
import os
import unittest
import traceback
from datetime import datetime
import numpy as np

# 确保能找到相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from milvus_enabled_academic_meeting_system import MilvusEnabledAcademicMeetingSystem
from academic_group_meeting_graph import AcademicDataType

class TestMilvusTasks(unittest.TestCase):
    """测试Milvus任务存储功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # 使用本地嵌入模型，启用Milvus
        print("初始化测试环境...")
        self.system = MilvusEnabledAcademicMeetingSystem(
            use_remote_llm=False,  # 使用本地LLM避免API调用
            use_local_embeddings=True,
            milvus_host="localhost",
            milvus_port="19530",
            milvus_collection="test_academic_dialogues"
        )
        
        # 创建测试角色
        self.professor_id = "prof_test"
        self.phd_id = "phd_test"
        self.master_id = "master_test"
        
        # 创建教授
        self.system.create_academic_agent(
            agent_id=self.professor_id,
            nickname="测试教授",
            age=50,
            role_type="教授",
            specialty=["人工智能", "机器学习"],
            personality="严谨认真"
        )
        
        # 创建博士生
        self.system.create_academic_agent(
            agent_id=self.phd_id,
            nickname="测试博士",
            age=28,
            role_type="博士生",
            specialty=["深度学习"],
            personality="勤奋好学"
        )
        
        # 创建硕士生
        self.system.create_academic_agent(
            agent_id=self.master_id,
            nickname="测试硕士",
            age=25,
            role_type="硕士生",
            specialty=["计算机视觉"],
            personality="创新思考"
        )
        
        # 创建测试场景
        self.scene_id = self.system.create_scene(
            name="智能体记忆测试",
            description="测试智能体的记忆存储与检索功能"
        )
        
        # 添加一些基础消息
        self.system.add_message(self.scene_id, self.professor_id, "今天我们讨论智能体记忆模型的设计。")
        self.system.add_message(self.scene_id, self.phd_id, "我认为向量化表示是一个好的起点。")
        self.system.add_message(self.scene_id, self.master_id, "我们应该考虑分层记忆结构。")
        
        print("测试环境初始化完成")
    
    def test_add_single_task(self):
        """测试添加单个任务到Milvus"""
        print("\n=== 测试添加单个任务到Milvus ===")
        
        # 创建一个测试任务
        task_title = "智能体记忆模型研究"
        task_description = """
        调研当前主流智能体记忆表示方法，比较向量化和符号化表示的优缺点，提出改进方案。
        需要考察现有的记忆存储结构和检索机制，分析性能瓶颈，设计新型的记忆表示模型。
        """
        
        # 调用add_task方法
        task_id = self.system.add_task(
            scene_id=self.scene_id,
            title=task_title,
            description=task_description,
            assignees=[self.phd_id],
            priority="高",
            due_date="2024-05-15",
            metadata={"workload": "10", "methods": "文献调研、模型设计、原型实现"}
        )
        
        self.assertIsNotNone(task_id, "创建任务失败")
        print(f"成功添加任务，ID: {task_id}")
        
        # 验证任务是否添加到语义图
        task_found = False
        task_data = None
        for key, value, datatype, _ in self.system.backend.semantic_map.data:
            if key == task_id and datatype == AcademicDataType.Task:
                task_found = True
                task_data = value
                print(f"验证成功: 在语义图中找到任务 '{value.get('Title', '')}'")
                break
        
        self.assertTrue(task_found, "任务未正确添加到语义图")
        self.assertEqual(task_data.get("Title"), task_title, "任务标题不匹配")
        self.assertEqual(task_data.get("Assignees"), [self.phd_id], "任务执行者不匹配")
        
        # 尝试通过语义搜索查找任务
        if self.system.use_milvus:
            try:
                # 使用任务描述中的关键词搜索
                search_results = self.system.semantic_search(
                    query_text="智能体记忆模型 向量化表示",
                    entity_type="task", 
                    top_k=5
                )
                
                print(f"搜索结果: {len(search_results)} 条记录")
                for result in search_results:
                    result_id = result.get('id', 'unknown')
                    result_title = result.get('title', 'unknown')
                    result_score = result.get('score', 0)
                    print(f"  ID: {result_id}, 标题: {result_title}, 相似度: {result_score}")
                    
                    # 如果找到匹配任务，验证其属性
                    if result_id == task_id:
                        print(f"验证成功: 在Milvus搜索结果中找到任务 {task_id}")
                        self.assertEqual(result.get('title'), task_title, "搜索结果中的任务标题不匹配")
                
            except Exception as e:
                print(f"搜索任务时出错: {str(e)}")
                traceback.print_exc()
    
    def test_add_multiple_tasks(self):
        """测试添加多个相互依赖的任务"""
        print("\n=== 测试添加多个相互依赖的任务 ===")
        
        # 创建第一个任务
        task1_title = "记忆模型设计"
        task1_id = self.system.add_task(
            scene_id=self.scene_id,
            title=task1_title,
            description="设计智能体记忆模型的整体架构，包括数据结构和存储方式",
            assignees=[self.professor_id],
            priority="高"
        )
        self.assertIsNotNone(task1_id, "创建第一个任务失败")
        
        # 创建第二个任务，依赖于第一个任务
        task2_title = "记忆检索算法实现"
        task2_id = self.system.add_task(
            scene_id=self.scene_id,
            title=task2_title,
            description="基于设计好的模型架构，实现高效的记忆检索算法",
            assignees=[self.phd_id],
            priority="中",
            source_id=task1_id  # 设置依赖关系
        )
        self.assertIsNotNone(task2_id, "创建第二个任务失败")
        
        # 创建第三个任务，同样依赖于第一个任务
        task3_title = "原型系统开发" 
        task3_id = self.system.add_task(
            scene_id=self.scene_id,
            title=task3_title,
            description="开发一个验证记忆模型的原型系统，实现基本功能",
            assignees=[self.master_id],
            priority="中",
            source_id=task1_id  # 设置依赖关系
        )
        self.assertIsNotNone(task3_id, "创建第三个任务失败")
        
        print(f"成功创建3个任务: {task1_id}, {task2_id}, {task3_id}")
        
        # 检查语义图中是否建立了正确的依赖关系
        dependencies_correct = False
        for start_node, relations in self.system.backend.semantic_graph.graph_relations.items():
            if start_node == task1_id:
                children = relations.get("children", {})
                if task2_id in children and task3_id in children:
                    dependencies_correct = True
                    print("验证成功: 依赖关系正确建立")
                    break
        
        self.assertTrue(dependencies_correct, "任务依赖关系未正确建立")
        
        # 验证任务是否存储在Milvus中
        if self.system.use_milvus:
            try:
                # 搜索所有任务
                search_results = self.system.semantic_search(
                    query_text="记忆模型",
                    entity_type="task",
                    top_k=10
                )
                
                print(f"搜索到 {len(search_results)} 个相关任务")
                
                # 检查是否能找到所有创建的任务
                found_tasks = set()
                for result in search_results:
                    result_id = result.get('id')
                    if result_id in [task1_id, task2_id, task3_id]:
                        found_tasks.add(result_id)
                        print(f"在搜索结果中找到任务: {result.get('title')}")
                
                print(f"共找到 {len(found_tasks)}/{3} 个创建的任务")
                self.assertTrue(len(found_tasks) > 0, "在Milvus中没有找到任何创建的任务")
                
            except Exception as e:
                print(f"搜索任务时出错: {str(e)}")
                traceback.print_exc()
    
    def test_add_summary_and_generate_tasks(self):
        """测试添加总结并从总结生成任务"""
        print("\n=== 测试添加总结并从总结生成任务 ===")
        
        # 定义模拟方法以避免实际调用LLM
        def mock_structured_summary(self, content, topic, summary_type, round_num=None):
            return {
                "key_findings": "记忆模型需要同时支持向量化和符号化表示；分层记忆架构能提高检索效率",
                "challenges": "大规模记忆的高效检索；记忆一致性管理",
                "future_directions": "探索混合记忆架构；开发自适应遗忘机制",
                "research_gaps": "缺乏统一评估基准；跨模态记忆整合不足",
                "action_items": "设计记忆模型；实现检索算法；开发原型系统"
            }
        
        def mock_llm_response(messages):
            return """
            任务1：设计混合记忆表示模型
            描述：设计一种同时支持向量化和符号化表示的混合记忆模型，能够高效存储和检索多类型知识
            优先级：高
            工作量：10人天
            研究方法：文献调研、模型设计、原型实现
            适合角色：博士生
            依赖任务：无
            
            任务2：开发分层记忆检索算法
            描述：实现基于分层架构的记忆检索算法，平衡检索速度和准确性
            优先级：中
            工作量：8人天
            研究方法：算法设计、性能测试、参数调优
            适合角色：博士生
            依赖任务：1
            
            任务3：构建智能体记忆评估基准
            描述：开发一套用于评估记忆模型性能的标准测试集和评价指标
            优先级：中
            工作量：7人天
            研究方法：数据集构建、指标设计、基准测试
            适合角色：硕士生
            依赖任务：无
            """
        
        # 添加辅助方法
        def find_suitable_assignees(self, role_type):
            if "博士" in role_type or "phd" in role_type.lower():
                return [self.phd_id]
            elif "硕士" in role_type or "master" in role_type.lower():
                return [self.master_id]
            else:
                return [self.professor_id]
        
        def find_random_assignees(self, count=1):
            return [self.phd_id]
        
        # 保存原始方法
        import types
        original_summary_method = None
        if hasattr(self.system, '_generate_structured_summary'):
            original_summary_method = self.system._generate_structured_summary
            self.system._generate_structured_summary = types.MethodType(mock_structured_summary, self.system)
        
        # 添加辅助方法
        if not hasattr(self.system, '_find_suitable_assignees'):
            self.system._find_suitable_assignees = types.MethodType(find_suitable_assignees, self.system)
        
        if not hasattr(self.system, '_find_random_assignees'):
            self.system._find_random_assignees = types.MethodType(find_random_assignees, self.system)
        
        # 保存并替换LLM响应方法
        original_llm_response = None
        if hasattr(self.system.backend.llm, 'get_response'):
            original_llm_response = self.system.backend.llm.get_response
            self.system.backend.llm.get_response = mock_llm_response
        
        try:
            # 添加一个总结
            summary_content = """
            本次讨论主要围绕智能体记忆模型进行。我们达成以下共识：
            1. 记忆模型应同时支持向量化和符号化表示
            2. 分层记忆架构能够提高检索效率
            3. 需要开发评估记忆系统性能的标准基准
            4. 记忆的一致性管理是关键挑战
            
            下一步我们应该着手设计混合记忆表示模型并开发原型系统进行验证。
            """
            
            summary_id = self.system.add_summary(
                scene_id=self.scene_id,
                speaker_id=self.professor_id,
                content=summary_content,
                summary_type="final"
            )
            
            self.assertIsNotNone(summary_id, "创建总结失败")
            print(f"成功添加总结，ID: {summary_id}")
            
            # 从总结生成任务
            task_ids = self.system.generate_tasks_from_summary(summary_id)
            
            self.assertTrue(len(task_ids) > 0, "从总结生成任务失败")
            print(f"从总结生成了 {len(task_ids)} 个任务")
            
            # 检查生成的任务
            for i, task_id in enumerate(task_ids):
                task_data = None
                for key, value, datatype, _ in self.system.backend.semantic_map.data:
                    if key == task_id and datatype == AcademicDataType.Task:
                        task_data = value
                        break
                
                if task_data:
                    print(f"任务{i+1}: {task_data.get('Title', '')}")
                    print(f"  描述: {task_data.get('Description', '')[:50]}...")
                    print(f"  优先级: {task_data.get('Priority', '')}")
                    print(f"  执行者: {task_data.get('Assignees', [])}")
                    
                    # 检查Milvus中是否存储了该任务
                    if self.system.use_milvus:
                        search_results = self.system.semantic_search(
                            query_text=task_data.get('Title', ''),
                            entity_type="task",
                            top_k=3
                        )
                        
                        task_found = False
                        for result in search_results:
                            if result.get('id') == task_id:
                                task_found = True
                                print(f"  √ 在Milvus中找到该任务")
                                break
                        
                        if not task_found:
                            print(f"  × 在Milvus中未找到该任务")
            
            # 检查任务依赖关系
            print("\n任务依赖关系:")
            dependency_count = 0
            for task_id in task_ids:
                for key, value, datatype, _ in self.system.backend.semantic_map.data:
                    if key == task_id and datatype == AcademicDataType.Task:
                        deps = value.get("Dependencies", [])
                        if deps:
                            dependency_count += len(deps)
                            dep_titles = []
                            for dep_id in deps:
                                for k, v, dt, _ in self.system.backend.semantic_map.data:
                                    if k == dep_id and dt == AcademicDataType.Task:
                                        dep_titles.append(v.get("Title", "未知任务"))
                                        break
                            print(f"任务 '{value.get('Title', '')}' 依赖于: {', '.join(dep_titles)}")
            
            print(f"共检测到 {dependency_count} 个任务依赖关系")
            
        except Exception as e:
            print(f"测试过程中出错: {str(e)}")
            traceback.print_exc()
            self.fail(f"测试失败: {str(e)}")
        finally:
            # 恢复原始方法
            if original_summary_method:
                self.system._generate_structured_summary = original_summary_method
            if original_llm_response:
                self.system.backend.llm.get_response = original_llm_response


    def test_export_to_neo4j(self):
        """测试导出到Neo4j并验证命名空间结构"""
        print("\n=== 测试导出到Neo4j并验证命名空间结构 ===")
        
        # 1. 准备测试数据：创建总结和任务
        print("准备测试数据...")
        
        # 创建总结
        summary_content = """
        本次讨论聚焦于智能体记忆系统设计。关键发现：
        1. 分层记忆架构可以平衡检索效率和存储成本
        2. 需要设计统一的知识表示形式
        3. 记忆管理策略对长期任务至关重要
        """
        
        summary_id = self.system.add_summary(
            scene_id=self.scene_id,
            speaker_id=self.professor_id,
            content=summary_content,
            summary_type="final"
        )
        self.assertIsNotNone(summary_id)
        print(f"创建总结: {summary_id}")
        
        # 创建几个任务
        task_ids = []
        
        # 任务1
        task1_id = self.system.add_task(
            scene_id=self.scene_id,
            title="设计分层记忆架构",
            description="设计智能体的分层记忆架构，包括短期、工作和长期记忆",
            assignees=[self.professor_id],
            priority="高",
            metadata={"workload": "10", "methods": "架构设计、流程图绘制"}
        )
        task_ids.append(task1_id)
        
        # 任务2，依赖于任务1
        task2_id = self.system.add_task(
            scene_id=self.scene_id,
            title="开发记忆检索算法",
            description="基于分层架构，实现高效的记忆检索算法",
            assignees=[self.phd_id],
            priority="中",
            source_id=task1_id
        )
        task_ids.append(task2_id)
        
        # 任务3，从总结生成
        task3_id = self.system.add_task(
            scene_id=self.scene_id,
            title="研究记忆管理策略",
            description="探索适合长期任务的记忆管理策略，包括遗忘机制和重要性评估",
            assignees=[self.master_id],
            priority="中",
            source_id=summary_id,
            metadata={"source_type": "summary"}
        )
        task_ids.append(task3_id)
        
        print(f"创建了 {len(task_ids)} 个任务")
        
        # 2. 导出到Neo4j
        try:
            # 获取Neo4j连接参数
            neo4j_uri = "bolt://localhost:7687"  # 默认值，可以根据实际情况修改
            neo4j_user = "neo4j"
            neo4j_password = "20031117"  # 使用您的实际密码
            neo4j_database = "academicgraph"
            
            print(f"尝试导出到Neo4j: {neo4j_uri}, 数据库: {neo4j_database}")
            
            # 调用导出方法
            stats = self.system.export_to_neo4j(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                create_constraints=True
            )
            
            print("导出完成，结果统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
            # 3. 验证导出的数据（直接查询Neo4j）
            from semantic_map import Neo4jInterface
            
            # 创建Neo4j接口实例
            neo4j = Neo4jInterface(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database
            )
            
            # 3.1 验证命名空间是否创建
            namespace_query = """
            MATCH (n:Namespace)
            RETURN n.id as id, n.name as name
            """
            namespace_results = neo4j.execute_query(namespace_query)
            
            print("\n命名空间节点:")
            namespaces = {}
            for record in namespace_results:
                namespace_id = record.get("id")
                namespace_name = record.get("name")
                namespaces[namespace_id] = namespace_name
                print(f"  {namespace_id}: {namespace_name}")
            
            # 必要的命名空间
            required_namespaces = ["namespace_TASK", "namespace_SUMMARY"]
            for ns in required_namespaces:
                self.assertIn(ns, namespaces, f"未找到必要的命名空间: {ns}")
            
            # 3.2 验证任务节点是否创建
            task_query = """
            MATCH (t:Task)
            RETURN t.id as id, t.title as title
            """
            task_results = neo4j.execute_query(task_query)
            
            print("\n任务节点:")
            exported_tasks = set()
            for record in task_results:
                task_id = record.get("id")
                task_title = record.get("title")
                exported_tasks.add(task_id)
                print(f"  {task_id}: {task_title}")
            
            # 验证所有创建的任务都已导出
            for task_id in task_ids:
                self.assertIn(task_id, exported_tasks, f"任务 {task_id} 未导出到Neo4j")
            
            # 3.3 验证命名空间和任务之间的关系
            relation_query = """
            MATCH (n:Namespace)-[r:CONTAINS]->(t:Task)
            RETURN n.id as namespace_id, t.id as task_id, type(r) as relation_type
            """
            relation_results = neo4j.execute_query(relation_query)
            
            print("\n命名空间与任务的关系:")
            task_namespace_relations = {}
            for record in relation_results:
                namespace_id = record.get("namespace_id")
                task_id = record.get("task_id")
                relation_type = record.get("relation_type")
                
                if task_id not in task_namespace_relations:
                    task_namespace_relations[task_id] = []
                task_namespace_relations[task_id].append((namespace_id, relation_type))
                
                print(f"  {namespace_id} -{relation_type}-> {task_id}")
            
            # 验证每个任务都与TASK命名空间相关联
            for task_id in task_ids:
                self.assertIn(task_id, task_namespace_relations, f"任务 {task_id} 没有与任何命名空间关联")
                
                # 检查是否与TASK命名空间关联
                has_task_namespace = False
                for ns_id, rel_type in task_namespace_relations.get(task_id, []):
                    if ns_id == "namespace_TASK":
                        has_task_namespace = True
                        break
                        
                self.assertTrue(has_task_namespace, f"任务 {task_id} 没有与TASK命名空间关联")
            
            # 3.4 验证任务之间的依赖关系
            dependency_query = """
            MATCH (t1:Task)-[r:DEPENDS_ON]->(t2:Task)
            RETURN t1.id as source_id, t2.id as target_id
            """
            dependency_results = neo4j.execute_query(dependency_query)
            
            print("\n任务依赖关系:")
            dependencies = {}
            for record in dependency_results:
                source_id = record.get("source_id")
                target_id = record.get("target_id")
                
                if source_id not in dependencies:
                    dependencies[source_id] = []
                dependencies[source_id].append(target_id)
                
                print(f"  {source_id} 依赖于 {target_id}")
            
            # 验证任务2依赖于任务1
            self.assertIn(task2_id, dependencies, f"任务 {task2_id} 没有依赖关系")
            self.assertIn(task1_id, dependencies.get(task2_id, []), f"任务 {task2_id} 没有依赖于任务 {task1_id}")
            
            # 关闭Neo4j连接
            neo4j.close()
            
        except Exception as e:
            print(f"测试导出到Neo4j时出错: {str(e)}")
            traceback.print_exc()
            self.fail(f"Neo4j导出测试失败: {str(e)}")

if __name__ == "__main__":
    unittest.main()
