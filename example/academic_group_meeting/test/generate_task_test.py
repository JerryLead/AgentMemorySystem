import sys
import os
import unittest
import traceback
from datetime import datetime
import json
import re
import uuid

# 确保能够找到相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from academic_group_meeting_backend import AcademicMeetingSystem
from academic_group_meeting_graph import AcademicDataType

class TestResearchTasksGeneration(unittest.TestCase):
    """测试从综述报告生成研究任务的功能"""
    
    def setUp(self):
        """初始化测试环境"""
        # 使用本地LLM避免API调用
        self.system = AcademicMeetingSystem(use_remote_llm=False)
        
        # 创建测试角色
        self.professor_id = "prof_test"
        self.system.create_academic_agent(
            agent_id=self.professor_id,
            nickname="测试教授",
            age=50,
            role_type="教授",
            specialty=["人工智能", "智能体系统"],
            personality="严谨认真"
        )
        
        # 创建博士生
        self.phd_id = "phd_test"
        self.system.create_academic_agent(
            agent_id=self.phd_id,
            nickname="测试博士",
            age=28,
            role_type="博士生",
            specialty=["深度学习"],
            personality="勤奋好学"
        )
        
        # 创建测试场景
        self.topic = "面向智能体的记忆管理系统"
        self.scene_id = self.system.create_scene(
            name=f"{self.topic}研究",
            description=f"测试{self.topic}相关研究任务生成"
        )
        
        # 模拟综述报告内容
        self.review_content = """
        # 面向智能体的记忆管理系统综述报告

        ## 摘要
        本文对面向智能体的记忆管理系统进行了全面的综述，涵盖了基础概念、关键技术和未来趋势。

        ## 1. 记忆模型与表示方法
        记忆模型是智能体系统的核心组件，支持知识的存储和检索。主要包括向量化表示和符号化表示两种方式。
        向量化表示通过嵌入空间捕获语义关系，而符号化表示则保留了结构化知识的逻辑关系。

        ## 2. 记忆存储与检索机制
        高效的存储和检索机制对智能体至关重要。索引结构如HNSW和IVF-PQ能够加速大规模记忆的检索过程。
        混合检索策略结合了密集检索和稀疏检索的优势，提高了召回率和精度。

        ## 3. 记忆更新与遗忘策略
        记忆管理还需要考虑更新策略，包括增量学习和定期重训练。
        基于重要性的遗忘机制可以优化记忆利用，确保关键信息的保留。

        ## 未来研究方向
        1. 多模态记忆整合：研究如何在统一框架下管理文本、图像和视频等多种模态的记忆。
        2. 记忆一致性管理：探索如何解决大规模记忆中的冲突和矛盾问题。
        3. 自适应记忆架构：根据任务需求动态调整记忆结构。
        4. 跨智能体记忆共享：研究多智能体之间安全高效的记忆交换机制。
        5. 记忆隐私与安全：解决记忆系统中的隐私保护和安全挑战。

        ## 结论
        面向智能体的记忆管理是提升智能体系统性能的关键领域，融合多学科知识，未来发展前景广阔。
        """
        
        # 创建模拟的研究规划内容
        self.research_plan_content = """
        基于综述报告分析，以下是未来研究任务：

        任务1：多模态记忆表示模型设计
        描述：设计一种统一的表示框架，能够同时处理文本、图像、视频等多种模态信息，并保持它们之间的语义关联。
        预期成果：一种新型多模态记忆表示模型及其原型实现。
        优先级：高

        任务2：记忆一致性检测与冲突解决算法
        描述：开发算法自动识别记忆库中的矛盾信息，并提供冲突解决策略，确保智能体基于一致的知识做出决策。
        预期成果：一套完整的记忆一致性管理框架和评估基准。
        优先级：中

        任务3：自适应记忆架构实现
        描述：实现能够根据任务需求和资源限制动态调整记忆组织结构的自适应系统。
        预期成果：自适应记忆架构原型及其在不同任务场景下的性能评估。
        优先级：中

        任务4：跨智能体记忆共享协议
        描述：设计安全、高效的协议允许多个智能体在保护隐私的前提下共享和交换记忆内容。
        预期成果：跨智能体记忆共享协议规范和参考实现。
        优先级：低
        """
        
        print("测试环境初始化完成")
    
    def test_task_generation_from_research_plan(self):
        """测试从研究规划中生成任务的功能"""
        print("\n=== 测试从研究规划生成任务 ===")
        
        # 1. 创建研究规划总结节点
        summary_id = f"summary_{uuid.uuid4().hex[:8]}"
        summary_info = {
            "Title": f"{self.topic}研究规划",
            "Content": self.research_plan_content,
            "Author": self.professor_id,
            "Type": "research_plan",
            "Topic": self.topic,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加总结节点到语义图
        self.system.backend.add_summary(summary_id, summary_info, [self.professor_id])
        print(f"已创建研究规划总结节点: {summary_id}")
        
        # 2. 测试原有任务生成方法
        print("\n测试原有的任务生成方法:")
        try:
            task_ids = self.system.generate_tasks_from_summary(summary_id)
            print(f"原方法生成了 {len(task_ids)} 个任务")
            
            # 分析问题
            if not task_ids:
                print("问题：没有生成任何任务，检查任务解析逻辑")
                # 调试任务文本解析部分
                self._debug_task_parsing(self.research_plan_content)
            else:
                # 输出生成的任务详情
                for i, task_id in enumerate(task_ids):
                    # 获取任务数据
                    task_data = None
                    for key, value, datatype, _ in self.system.backend.semantic_map.data:
                        if key == task_id and datatype == AcademicDataType.Task:
                            task_data = value
                            break
                    
                    if task_data:
                        print(f"任务 {i+1}: {task_data.get('Title')}")
                        print(f"  描述: {task_data.get('Description')[:50]}...")
                        print(f"  优先级: {task_data.get('Priority')}")
                        print(f"  负责人: {task_data.get('Assignees')}")
        except Exception as e:
            print(f"原始任务生成方法出错: {str(e)}")
            traceback.print_exc()
        
        # 3. 测试改进的任务生成方法
        print("\n测试改进的任务生成方法:")
        new_task_ids = self._improved_generate_tasks(summary_id)
        print(f"改进方法生成了 {len(new_task_ids)} 个任务")
        
        # 输出改进方法生成的任务
        for i, task_id in enumerate(new_task_ids):
            task_data = None
            for key, value, datatype, _ in self.system.backend.semantic_map.data:
                if key == task_id and datatype == AcademicDataType.Task:
                    task_data = value
                    break
            
            if task_data:
                print(f"任务 {i+1}: {task_data.get('Title')}")
                print(f"  描述: {task_data.get('Description')[:50]}...")
                print(f"  优先级: {task_data.get('Priority')}")
    
    def test_end_to_end_review_to_tasks(self):
        """测试从综述报告到任务生成的完整流程"""
        print("\n=== 测试从综述报告到任务生成的完整流程 ===")
        
        # 保存综述报告到本地文件
        review_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "article", f"{self.topic.replace(' ', '_')}_test.md")
        try:
            os.makedirs(os.path.dirname(review_file), exist_ok=True)
            with open(review_file, "w", encoding="utf-8") as f:
                f.write(self.review_content)
            print(f"已保存综述报告到: {review_file}")
        except Exception as e:
            print(f"保存综述报告时出错: {str(e)}")
        
        # 手动模拟综述报告生成后的操作
        report_id = f"review_{uuid.uuid4().hex[:8]}"
        
        # 1. 从综述报告生成研究规划
        research_plan = self._generate_research_plan_from_review(self.review_content, self.topic)
        
        # 2. 保存研究规划
        summary_id = f"summary_{uuid.uuid4().hex[:8]}"
        summary_info = {
            "Title": f"{self.topic}研究规划",
            "Content": research_plan,
            "Author": self.professor_id,
            "Type": "research_plan",
            "Topic": self.topic,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.system.backend.add_summary(summary_id, summary_info, [self.professor_id, report_id])
        print(f"从综述报告生成并保存了研究规划: {summary_id}")
        
        # 3. 从研究规划生成任务
        task_ids = self._improved_generate_tasks(summary_id)
        print(f"从研究规划生成了 {len(task_ids)} 个任务")
        
        # 验证生成的任务
        self.assertTrue(len(task_ids) > 0, "应该至少生成一个研究任务")
        
        # 输出任务详情
        for i, task_id in enumerate(task_ids):
            task_data = None
            for key, value, datatype, _ in self.system.backend.semantic_map.data:
                if key == task_id and datatype == AcademicDataType.Task:
                    task_data = value
                    break
            
            if task_data:
                print(f"任务 {i+1}: {task_data.get('Title')}")
                print(f"  描述: {task_data.get('Description')[:50]}...")
                print(f"  优先级: {task_data.get('Priority')}")
    
    def _debug_task_parsing(self, tasks_text):
        """调试任务解析逻辑"""
        print("\n调试任务解析:")
        
        # 尝试使用不同的正则表达式分割任务
        print("尝试正则表达式: r'任务\\d+[：:][^\\n]*'")
        pattern1 = r'任务\d+[：:][^\n]*'
        matches1 = re.findall(pattern1, tasks_text)
        print(f"找到 {len(matches1)} 个匹配: {matches1}\n")
        
        print("尝试正则表达式: r'任务\\d+[：:]'")
        pattern2 = r'任务\d+[：:]'
        matches2 = re.findall(pattern2, tasks_text)
        print(f"找到 {len(matches2)} 个匹配: {matches2}\n")
        
        print("尝试使用re.split分割:")
        task_sections = re.split(r'任务\d+[：:]', tasks_text)
        filtered_sections = [s.strip() for s in task_sections if s.strip()]
        print(f"分割得到 {len(filtered_sections)} 个部分")
        for i, section in enumerate(filtered_sections):
            print(f"部分 {i+1} 开头: {section[:50]}...\n")
        
        # 手动提取任务
        tasks = []
        for i, section in enumerate(filtered_sections):
            if i == 0 and not section.startswith("描述"):  # 跳过引言部分
                continue
                
            lines = section.split('\n')
            task = {"title": "", "description": "", "priority": "中"}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("描述"):
                    task["description"] = line.replace("描述：", "").replace("描述:", "").strip()
                elif line.startswith("预期"):
                    task["expected"] = line.replace("预期成果：", "").replace("预期成果:", "").strip()
                elif line.startswith("优先级"):
                    priority = line.replace("优先级：", "").replace("优先级:", "").strip()
                    task["priority"] = priority
            
            # 尝试从前一个匹配获取标题
            if i < len(matches1):
                title_match = matches1[i]
                task["title"] = title_match.replace(f"任务{i+1}：", "").replace(f"任务{i+1}:", "").strip()
            
            if task["description"]:
                tasks.append(task)
        
        print(f"\n手动提取得到 {len(tasks)} 个任务:")
        for i, task in enumerate(tasks):
            print(f"任务 {i+1}:")
            print(f"  标题: {task.get('title', '无标题')}")
            print(f"  描述: {task.get('description', '无描述')}")
            print(f"  优先级: {task.get('priority', '无优先级')}")
    
    def _improved_generate_tasks(self, summary_id):
        """改进的从总结生成任务方法"""
        # 获取总结节点数据
        summary_data = None
        for key, value, datatype, _ in self.system.backend.semantic_map.data:
            if key == summary_id and datatype == AcademicDataType.Summary:
                summary_data = value
                break
        
        if not summary_data:
            print(f"未找到ID为 {summary_id} 的总结节点")
            return []
        
        # 获取总结内容和主题
        content = summary_data.get("Content", "")
        topic = summary_data.get("Topic", "")
        scene_id = None
        
        # 查找对应的场景
        for scene_id, scene in self.system.scenes.items():
            if scene.name and topic and topic in scene.name:
                break
        
        if not scene_id:
            scene_id = self.scene_id  # 使用当前测试场景
        
        # 直接解析任务内容
        task_ids = []
        tasks = self._parse_tasks_from_text(content)
        
        # 创建任务节点
        for task in tasks:
            # 生成任务ID
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            # 为任务分配执行者
            assignees = []
            if "角色" in task and "博士" in task["角色"]:
                assignees = [self.phd_id]
            else:
                # 随机选择执行者
                assignees = [self.phd_id]  # 在测试中始终使用博士生
            
            # 创建任务信息
            task_info = {
                "Title": task.get("title", "未命名任务"),
                "Description": task.get("description", ""),
                "Assignees": assignees,
                "Priority": task.get("priority", "中"),
                "Status": "待开始",
                "Topic": topic,
                "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 添加额外元数据
            if "expected" in task:
                task_info["ExpectedOutcome"] = task["expected"]
            
            # 确定父节点
            parent_keys = [summary_id, f"topic_{scene_id}"]
            if assignees:
                parent_keys.extend(assignees)
            
            # 添加到语义图
            self.system.backend.add_task(task_id, task_info, parent_keys)
            task_ids.append(task_id)
        
        return task_ids
    
    def _parse_tasks_from_text(self, text):
        """从文本中解析任务信息"""
        tasks = []
        
        # 寻找任务标记
        matches = re.finditer(r'任务(\d+)[：:]([^\n]*)', text)
        task_starters = []
        for match in matches:
            task_num = int(match.group(1))
            task_title = match.group(2).strip()
            start_pos = match.start()
            task_starters.append((task_num, task_title, start_pos))
        
        # 按照出现位置排序
        task_starters.sort(key=lambda x: x[2])
        
        # 提取每个任务的完整内容
        for i, (task_num, task_title, start_pos) in enumerate(task_starters):
            # 确定当前任务的内容结束位置
            end_pos = len(text)
            if i < len(task_starters) - 1:
                end_pos = task_starters[i+1][2]
            
            # 提取任务内容
            task_content = text[start_pos:end_pos].strip()
            
            # 解析任务属性
            task = {"title": task_title}
            
            # 提取描述
            desc_match = re.search(r'描述[：:](.*?)(?=\n\w+[：:]|$)', task_content, re.DOTALL)
            if desc_match:
                task["description"] = desc_match.group(1).strip()
            
            # 提取预期成果
            expected_match = re.search(r'预期成果[：:](.*?)(?=\n\w+[：:]|$)', task_content, re.DOTALL)
            if expected_match:
                task["expected"] = expected_match.group(1).strip()
            
            # 提取优先级
            priority_match = re.search(r'优先级[：:](.*?)(?=\n|$)', task_content)
            if priority_match:
                task["priority"] = priority_match.group(1).strip()
            
            # 提取适合角色
            role_match = re.search(r'适合角色[：:](.*?)(?=\n|$)', task_content)
            if role_match:
                task["角色"] = role_match.group(1).strip()
            
            tasks.append(task)
        
        # 如果无法使用任务标记解析，尝试使用描述标记
        if not tasks:
            # 按描述分割
            desc_sections = re.split(r'\n描述[：:]', text)
            for i, section in enumerate(desc_sections):
                if i == 0:  # 跳过第一部分(通常是引言)
                    continue
                
                # 创建任务
                task = {"description": section.strip().split('\n')[0].strip()}
                
                # 提取优先级
                priority_match = re.search(r'优先级[：:](.*?)(?=\n|$)', section)
                if priority_match:
                    task["priority"] = priority_match.group(1).strip()
                else:
                    task["priority"] = "中"  # 默认优先级
                
                # 使用描述作为标题
                if task["description"]:
                    words = task["description"].split()
                    task["title"] = " ".join(words[:min(8, len(words))]) + "..."
                else:
                    task["title"] = f"任务 {i}"
                
                tasks.append(task)
        
        return tasks
    
    def _generate_research_plan_from_review(self, review_content, topic):
        """从综述报告内容生成研究规划"""
        # 提取未来研究方向部分
        future_research_section = ""
        section_match = re.search(r'## 未来研究方向(.*?)(?=##|$)', review_content, re.DOTALL)
        if section_match:
            future_research_section = section_match.group(1).strip()
        
        # 如果没有找到未来研究方向部分，使用全文
        if not future_research_section:
            future_research_section = review_content
        
        # 生成研究规划
        research_plan = f"""基于{topic}综述报告分析，以下是未来研究任务：\n\n"""
        
        # 解析出未来研究方向中的数字列表项
        directions = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', future_research_section, re.DOTALL)
        directions = [d.strip() for d in directions if d.strip()]
        
        # 如果找不到列表项，尝试按行分割
        if not directions:
            lines = future_research_section.split('\n')
            directions = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        # 为每个方向创建任务
        for i, direction in enumerate(directions):
            if i >= 4:  # 限制最多4个任务
                break
                
            # 设置优先级
            priority = "高" if i == 0 else "中" if i < 2 else "低"
            
            # 从方向生成标题和描述
            title = direction
            if ':' in direction:
                title = direction.split(':')[0].strip()
            elif '：' in direction:
                title = direction.split('：')[0].strip()
                
            # 确保标题不要太长
            if len(title) > 40:
                title = title[:40] + "..."
                
            description = f"研究{direction}，探索其在智能体记忆管理系统中的应用和实现方法。"
            
            research_plan += f"任务{i+1}：{title}\n"
            research_plan += f"描述：{description}\n"
            research_plan += f"预期成果：一套完整的{title}解决方案和性能评估。\n"
            research_plan += f"优先级：{priority}\n\n"
        
        return research_plan

if __name__ == "__main__":
    unittest.main()