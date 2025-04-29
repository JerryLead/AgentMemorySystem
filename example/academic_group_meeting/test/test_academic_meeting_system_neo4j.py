import os
import sys
import unittest
from datetime import datetime
import random

# 确保能找到模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from academic_group_meeting_backend import AcademicMeetingSystem
from academic_group_meeting_graph import AcademicDataType

class TestAcademicMeetingSystem(unittest.TestCase):
    """测试学术会议系统的关键功能"""
    
    def setUp(self):
        """测试前的初始化"""
        # 使用本地嵌入模型
        self.system = AcademicMeetingSystem(
            use_remote_llm=False,  # 使用本地LLM避免API调用
            use_local_embeddings=True
        )
        
        # 创建测试角色
        self.professor_id = "prof_1"
        self.phd_id = "phd_1"
        self.master_id = "master_1"
        
        # 创建教授
        self.system.create_academic_agent(
            agent_id=self.professor_id,
            nickname="张教授",
            age=45,
            role_type="教授",
            specialty=["人工智能", "机器学习"],
            personality="严谨认真"
        )
        
        # 创建博士生
        self.system.create_academic_agent(
            agent_id=self.phd_id,
            nickname="李博士",
            age=28,
            role_type="博士生",
            specialty=["深度学习", "自然语言处理"],
            personality="勤奋好学"
        )
        
        # 创建硕士生
        self.system.create_academic_agent(
            agent_id=self.master_id,
            nickname="王硕士",
            age=24,
            role_type="硕士生",
            specialty=["计算机视觉"],
            personality="创新思考"
        )
        
        # 创建测试场景
        self.scene_id = self.system.create_scene(
            name="智能体记忆模型研究",
            description="探讨智能体的记忆模型与实现方法"
        )
        
        # 添加一些基础消息
        self.system.add_message(self.scene_id, self.professor_id, "今天我们讨论智能体记忆模型的研究现状和挑战。")
        self.system.add_message(self.scene_id, self.phd_id, "我认为记忆检索机制是当前的一个难点。")
        self.system.add_message(self.scene_id, self.master_id, "是否可以考虑向量数据库来存储记忆？")
        self.system.add_message(self.scene_id, self.professor_id, "这是个好想法，我们需要更深入地探讨记忆表示方法。")
    
    def test_add_summary(self):
        """测试添加总结功能"""
        print("\n=== 测试添加总结 ===")
        
        # 1. 添加轮次总结
        round_summary_content = """
        本轮讨论主要围绕智能体记忆模型的表示方法和存储结构展开。
        主要观点包括：1) 向量化表示是当前主流方法；2) 分层次的记忆组织更有效；
        3) 记忆检索机制需要平衡速度和准确性。
        还需进一步探讨记忆的时效性管理和冗余控制策略。
        """
        
        round_summary_id = self.system.add_summary(
            scene_id=self.scene_id,
            speaker_id=self.professor_id,
            content=round_summary_content,
            summary_type="round",
            round_num=1
        )
        
        self.assertIsNotNone(round_summary_id)
        print(f"成功添加轮次总结，ID: {round_summary_id}")
        
        # 验证总结是否正确添加到语义图
        summary_found = False
        for key, value, datatype, _ in self.system.backend.semantic_map.data:
            if key == round_summary_id and datatype == AcademicDataType.Summary:
                summary_found = True
                print(f"验证成功: 在语义图中找到总结 '{value.get('Title', '')}'")
                break
        
        self.assertTrue(summary_found, "总结未正确添加到语义图")
        
        # 2. 添加最终总结
        final_summary_content = """
        本次讨论总结了智能体记忆模型的多个关键方面。我们认为，高效的智能体记忆系统应该具备：
        1. 灵活的记忆表示方式，支持多模态数据
        2. 高效的存储和检索机制，可以使用向量数据库
        3. 智能的记忆管理策略，包括遗忘机制和重要性评估
        4. 上下文感知能力，能根据当前任务动态调整记忆检索优先级
        
        未来研究方向包括记忆压缩技术、记忆冲突解决、跨模态记忆整合等。
        """
        
        final_summary_id = self.system.add_summary(
            scene_id=self.scene_id,
            speaker_id=self.professor_id,
            content=final_summary_content,
            summary_type="final"
        )
        
        self.assertIsNotNone(final_summary_id)
        print(f"成功添加最终总结，ID: {final_summary_id}")
        
        # 验证最终总结是否添加到语义图
        final_summary_found = False
        for key, value, datatype, _ in self.system.backend.semantic_map.data:
            if key == final_summary_id and datatype == AcademicDataType.Summary:
                final_summary_found = True
                print(f"验证成功: 在语义图中找到最终总结 '{value.get('Title', '')}'")
                break
        
        self.assertTrue(final_summary_found, "最终总结未正确添加到语义图")
        
        return final_summary_id  # 返回最终总结ID，用于后续测试
    
    def test_add_task(self):
        """测试添加任务功能"""
        print("\n=== 测试添加任务 ===")
        
        # 手动添加一个任务
        task_id = self.system.add_task(
            scene_id=self.scene_id,
            title="智能体记忆表示方法研究",
            description="调研当前主流智能体记忆表示方法，比较向量化和符号化表示的优缺点，提出改进方案。",
            assignees=[self.phd_id],
            priority="高",
            due_date="2023-12-31"
        )
        
        self.assertIsNotNone(task_id)
        print(f"成功添加任务，ID: {task_id}")
        
        # 验证任务是否正确添加到语义图
        task_found = False
        for key, value, datatype, _ in self.system.backend.semantic_map.data:
            if key == task_id and datatype == AcademicDataType.Task:
                task_found = True
                print(f"验证成功: 在语义图中找到任务 '{value.get('Title', '')}'")
                # 检查任务属性
                self.assertEqual(value.get("Title"), "智能体记忆表示方法研究")
                self.assertEqual(value.get("Priority"), "高")
                self.assertEqual(value.get("Assignees"), [self.phd_id])
                break
        
        self.assertTrue(task_found, "任务未正确添加到语义图")
        
        # 添加另一个任务，分配给硕士生
        task2_id = self.system.add_task(
            scene_id=self.scene_id,
            title="向量数据库在智能体记忆中的应用",
            description="研究Milvus、FAISS等向量数据库在智能体记忆存储中的应用，进行性能测试和比较。",
            assignees=[self.master_id],
            priority="中"
        )
        
        self.assertIsNotNone(task2_id)
        print(f"成功添加第二个任务，ID: {task2_id}")
        
        # 验证第二个任务
        task2_found = False
        for key, value, datatype, _ in self.system.backend.semantic_map.data:
            if key == task2_id and datatype == AcademicDataType.Task:
                task2_found = True
                print(f"验证成功: 在语义图中找到任务 '{value.get('Title', '')}'")
                break
        
        self.assertTrue(task2_found, "第二个任务未正确添加到语义图")
        
        # 检查语义图中的任务总数
        task_count = sum(1 for _, _, datatype, _ in self.system.backend.semantic_map.data 
                        if datatype == AcademicDataType.Task)
        print(f"语义图中的任务总数: {task_count}")
        
        return [task_id, task2_id]  # 返回任务ID列表，用于后续测试
    
    def test_generate_tasks_from_summary(self):
        """测试从总结生成任务功能"""
        print("\n=== 测试从总结生成任务 ===")
        
        # 先添加一个新总结
        summary_content = """
        记忆管理是智能体系统的核心组件，我们需要重点关注以下方面：
        1. 记忆表示：需要设计高效的向量化表示方法，支持语义检索
        2. 存储结构：探索分层存储架构，区分短期记忆和长期记忆
        3. 检索机制：实现基于相关性和重要性的高效检索算法
        4. 记忆更新：设计动态更新机制，包括记忆强化和遗忘
        5. 跨模态整合：研究文本、图像等多模态数据的统一记忆表示
        
        建议接下来分别针对这几个方向展开深入研究，并进行原型系统实现与评估。
        """
        
        mock_generate_structured_summary = lambda content, topic, summary_type, round_num: {
            "key_findings": "向量化表示是主流方法；分层存储可区分短期和长期记忆；检索需考虑相关性和重要性",
            "challenges": "记忆表示效率；大规模记忆管理；跨模态整合",
            "future_directions": "压缩记忆表示；自适应遗忘机制；多智能体记忆共享",
            "research_gaps": "缺乏统一的记忆评估基准；跨模态记忆整合不足",
            "action_items": "开发记忆表示模型；设计检索算法；实现原型系统；进行性能评估"
        }
        
        # 暂时替换_generate_structured_summary方法避免LLM调用
        original_method = self.system._generate_structured_summary
        self.system._generate_structured_summary = mock_generate_structured_summary
        
        summary_id = self.system.add_summary(
            scene_id=self.scene_id,
            speaker_id=self.professor_id,
            content=summary_content,
            summary_type="final"
        )
        
        self.assertIsNotNone(summary_id)
        print(f"已添加用于生成任务的总结，ID: {summary_id}")
        
        # 模拟LLM任务生成响应
        mock_task_response = """
        任务1：设计智能体记忆表示模型
        描述：研究并设计适合智能体系统的记忆表示模型，支持向量化表示和语义检索，能够有效捕捉记忆的关键特征
        优先级：高
        工作量：10人天
        研究方法：文献调研、模型设计、原型实现
        适合角色：博士生
        依赖任务：无
        
        任务2：实现分层记忆存储结构
        描述：设计并实现区分短期记忆和长期记忆的分层存储结构，研究记忆从短期到长期的转换机制
        优先级：中
        工作量：7人天
        研究方法：存储架构设计、数据结构实现、性能测试
        适合角色：硕士生
        依赖任务：1
        
        任务3：开发记忆检索算法
        描述：研发基于相关性和重要性的记忆检索算法，优化检索效率和准确率
        优先级：高
        工作量：8人天
        研究方法：算法设计、向量索引、性能评估
        适合角色：博士生
        依赖任务：1
        
        任务4：设计记忆更新与遗忘机制
        描述：实现智能体记忆的动态更新机制，包括记忆强化和自适应遗忘策略
        优先级：中
        工作量：6人天
        研究方法：策略设计、算法实现、对比实验
        适合角色：硕士生
        依赖任务：2, 3
        """
        
        # 暂时替换LLM响应以避免真实调用
        original_llm_response = self.system.backend.llm.get_response
        self.system.backend.llm.get_response = lambda messages: mock_task_response
        
        # 从总结生成任务
        try:
            task_ids = self.system.generate_tasks_from_summary(summary_id)
            
            # 恢复原始方法
            self.system._generate_structured_summary = original_method
            self.system.backend.llm.get_response = original_llm_response
            
            self.assertIsNotNone(task_ids)
            self.assertTrue(len(task_ids) > 0)
            print(f"成功从总结生成了 {len(task_ids)} 个任务")
            
            # 检查生成的任务
            for task_id in task_ids:
                task_found = False
                for key, value, datatype, _ in self.system.backend.semantic_map.data:
                    if key == task_id and datatype == AcademicDataType.Task:
                        task_found = True
                        print(f"验证任务: '{value.get('Title', '')}', 优先级: {value.get('Priority', '')}")
                        
                        # 检查任务是否包含必要的字段
                        self.assertIn('Title', value)
                        self.assertIn('Description', value)
                        self.assertIn('Priority', value)
                        self.assertIn('Status', value)
                        self.assertIn('Assignees', value)
                        break
                        
                self.assertTrue(task_found, f"任务 {task_id} 未找到")
            
            # 检查任务依赖关系
            dependency_count = 0
            for start_node, relations in self.system.backend.semantic_graph.graph_relations.items():
                for end_node, relation_type in relations.get("children", {}).items():
                    if relation_type == "依赖于" and end_node in task_ids:
                        dependency_count += 1
            
            print(f"检测到 {dependency_count} 个任务依赖关系")
            
        except Exception as e:
            # 恢复原始方法
            self.system._generate_structured_summary = original_method
            self.system.backend.llm.get_response = original_llm_response
            self.fail(f"生成任务测试失败: {str(e)}")
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        print("\n=== 测试完整工作流程 ===")
        
        # 1. 添加更多讨论内容
        self.system.add_message(self.scene_id, self.phd_id, "我认为我们应该设计一个实验来比较不同记忆表示方法的效率。")
        self.system.add_message(self.scene_id, self.master_id, "我可以开发一个原型系统来测试向量数据库的性能。")
        self.system.add_message(self.scene_id, self.professor_id, "这些都是很好的想法，我们需要制定具体的研究计划。")
        
        # 2. 添加总结
        summary_id = self.test_add_summary()
        self.assertIsNotNone(summary_id)
        
        # 3. 手动添加任务
        task_ids = self.test_add_task()
        self.assertTrue(len(task_ids) > 0)
        
        # 4. 从总结生成任务
        self.test_generate_tasks_from_summary()
        
        # 5. 验证最终的语义图
        node_counts = {}
        for _, _, datatype, _ in self.system.backend.semantic_map.data:
            if datatype not in node_counts:
                node_counts[datatype] = 0
            node_counts[datatype] += 1
        
        print("\n语义图节点统计:")
        for datatype, count in node_counts.items():
            print(f"  {datatype.name}: {count} 个节点")
        
        # 6. 检查所有任务和总结
        all_summaries = self.system.get_all_summaries(self.scene_id)
        print(f"\n获取到 {len(all_summaries)} 个总结")
        
        all_tasks = self.system.get_all_tasks(self.scene_id)
        print(f"获取到 {len(all_tasks)} 个任务")
        
        # 7. 更新一个任务的状态
        if all_tasks:
            random_task = random.choice(all_tasks)
            task_id = random_task["id"]
            old_status = random_task["status"]
            new_status = "进行中"
            
            success = self.system.update_task_status(task_id, new_status)
            self.assertTrue(success)
            print(f"成功更新任务 '{random_task['title']}' 状态从 '{old_status}' 到 '{new_status}'")
            
            # 验证状态更新
            updated_tasks = self.system.get_all_tasks(self.scene_id)
            for task in updated_tasks:
                if task["id"] == task_id:
                    self.assertEqual(task["status"], new_status)
                    break
        
        print("\n完整工作流程测试成功!")
        data = self.system.backend.semantic_graph.print_str_graph()
        print(data)

if __name__ == "__main__":
    unittest.main()