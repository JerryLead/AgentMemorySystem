import uuid
import json
import re
import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import traceback

# 导入大模型客户端
from semantic_map.deepseek_client import deepseek_remote, deepseek_local

# 任务状态枚举
class TaskStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# 角色类型枚举
class AcademicRole:
    PROFESSOR = "professor"
    PHD_STUDENT = "phd_student"
    MSC_STUDENT = "msc_student"
    RESEARCH_ASSISTANT = "research_assistant"

# 角色特长映射
ROLE_EXPERTISE = {
    AcademicRole.PROFESSOR: ["研究规划", "问题分析", "理论基础", "批判性思考", "学术指导", "研究方向把握"],
    AcademicRole.PHD_STUDENT: ["文献调研", "方法讨论", "实验设计", "数据分析", "论文写作"],
    AcademicRole.MSC_STUDENT: ["实验执行", "数据收集", "技术实现", "论文检索", "可视化展示"],
    AcademicRole.RESEARCH_ASSISTANT: ["资料整理", "附件分析", "会议记录", "图谱生成", "文献管理"]
}

# 任务角色最佳匹配
TASK_ROLE_MATCH = {
    "创建会议": AcademicRole.PROFESSOR,
    "组会主持": AcademicRole.PROFESSOR,
    "会议总结": AcademicRole.RESEARCH_ASSISTANT,
    "文献调研": AcademicRole.PHD_STUDENT,
    "问题分析": AcademicRole.PROFESSOR,
    "方法讨论": AcademicRole.PHD_STUDENT,
    "实验设计": AcademicRole.PHD_STUDENT,
    "结果分析": AcademicRole.PROFESSOR,
    "论文检索": AcademicRole.MSC_STUDENT,
    "附件分析": AcademicRole.RESEARCH_ASSISTANT,
    "生成图谱": AcademicRole.MSC_STUDENT
}

class Task:
    """表示一个任务的类"""
    def __init__(self, 
                task_id: str = None, 
                name: str = "", 
                description: str = "", 
                assignee: str = None,
                dependencies: List[str] = None,
                priority: int = 1,
                status: str = TaskStatus.PENDING,
                result: Any = None):
        """初始化任务
        
        Args:
            task_id: 任务ID，如果不指定则自动生成
            name: 任务名称
            description: 任务描述
            assignee: 任务执行者ID
            dependencies: 依赖的任务ID列表
            priority: 优先级(1-5)，数字越大优先级越高
            status: 任务状态
            result: 任务结果
        """
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = description
        self.assignee = assignee
        self.dependencies = dependencies or []
        self.priority = priority
        self.status = status
        self.result = result
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.subtasks = []  # 子任务ID列表
        self.parent_task = None  # 父任务ID
        self.tags = []  # 标签列表
        self.metadata = {}  # 元数据字典
        self.role_expertise = None  # 执行任务的角色专长
    
    def to_dict(self):
        """将任务转换为字典表示"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "assignee": self.assignee,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "subtasks": self.subtasks,
            "parent_task": self.parent_task,
            "tags": self.tags,
            "metadata": self.metadata,
            "role_expertise": self.role_expertise
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建任务对象"""
        task = cls(
            task_id=data.get("task_id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            assignee=data.get("assignee"),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", 1),
            status=data.get("status", TaskStatus.PENDING),
            result=data.get("result")
        )
        
        # 处理日期时间字段
        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
            
        task.subtasks = data.get("subtasks", [])
        task.parent_task = data.get("parent_task")
        task.tags = data.get("tags", [])
        task.metadata = data.get("metadata", {})
        task.role_expertise = data.get("role_expertise")
        
        return task
    
    def start(self):
        """开始执行任务"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete(self, result=None):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        if result is not None:
            self.result = result
    
    def fail(self, error=None):
        """标记任务失败"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        if error is not None:
            self.result = error
    
    def add_subtask(self, subtask_id):
        """添加子任务"""
        if subtask_id not in self.subtasks:
            self.subtasks.append(subtask_id)
    
    def __str__(self):
        return f"Task({self.task_id}: {self.name}, status={self.status})"

class TaskManager:
    """任务管理器，负责任务的创建、分配、执行和监控"""
    # 在TaskManager类的__init__方法中修改：
    def __init__(self, use_remote_llm=False):
        """初始化任务管理器"""
        self.tasks = {}  # task_id -> Task
        self.agent_tasks = {}  # agent_id -> [task_ids]
        self.task_handlers = {}  # task_name -> handler_function
        self.role_to_agent = {}  # role_type -> [agent_ids]
        
        # 初始化模型客户端 - 修复本地LLM支持
        print(f"初始化LLM客户端: {'远程' if use_remote_llm else '本地'}")
        if use_remote_llm:
            self.llm_client = deepseek_remote()
        else:
            # 修复：启用思考过程，正确传递参数
            self.llm_client = deepseek_local()
            print("已启用本地LLM：DeepSeek Local与思考过程")

    # 在execute_task方法中添加更多日志输出
    def execute_task(self, task_id: str) -> Any:
        """
        执行指定任务，添加更多日志输出
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")
        
        task = self.tasks[task_id]
        print(f"开始执行任务: {task.name}")
        
        # 获取执行者信息
        agent_nickname = "未知执行者"
        role_name = "未知角色"
        for role, agents in self.role_to_agent.items():
            if task.assignee in agents:
                if role == AcademicRole.PROFESSOR:
                    agent_nickname = "许教授"
                    role_name = "教授"
                elif role == AcademicRole.PHD_STUDENT:
                    agent_nickname = "李同学"
                    role_name = "博士生"
                elif role == AcademicRole.MSC_STUDENT:
                    if "msc_1" in agents and task.assignee == "msc_1":
                        agent_nickname = "郭同学" 
                    else:
                        agent_nickname = "吴同学"
                    role_name = "硕士生"
                elif role == AcademicRole.RESEARCH_ASSISTANT:
                    agent_nickname = "陈助理"
                    role_name = "研究助理"
                
                # 修改输出格式以匹配前端期望格式
                print(f"{agent_nickname}：正在执行任务'{task.name}'，作为{role_name}，我将负责这项工作。")
                break
        
        # 根据任务类型和角色选择处理方法
        role_expertise = task.role_expertise or self.get_role_for_agent(task.assignee)
        handler = self.get_role_based_handler(task.name, role_expertise)
        
        if not handler:
            print(f"系统：错误: 未找到任务'{task.name}'的处理函数")
            task.fail(f"No handler registered for task type: {task.name}")
            return None
        
        try:
            # 标记任务为进行中
            task.start()
            # 使用更清晰的状态输出
            print(f"系统：任务'{task.name}'已开始")
            
            # 执行任务
            print(f"{agent_nickname}：我开始分析{task.name}相关问题...")
            result = handler(task)
            
            # 标记任务为完成，使用清晰的格式输出结果
            task.complete(result)
            print(f"{agent_nickname}：我已完成'{task.name}'任务，这是我的分析结果。")
            
            return result
        except Exception as e:
            # 记录异常信息
            print(f"系统：执行任务'{task.name}'时出错: {str(e)}")
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            task.fail(error_info)
            print(f"系统：任务'{task.name}'执行失败")
            return None
    
    def register_task_handler(self, task_name: str, handler: Callable):
        """注册任务处理函数
        
        Args:
            task_name: 任务名称
            handler: 处理函数，接收任务对象并返回结果
        """
        self.task_handlers[task_name] = handler
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """注册智能体及其信息
        
        Args:
            agent_id: 智能体ID
            agent_info: 智能体信息，包含角色类型等
        """
        role_type = agent_info.get("role_type", "").lower()
        
        # 确定角色类型
        if "教授" in role_type or "professor" in role_type:
            if AcademicRole.PROFESSOR not in self.role_to_agent:
                self.role_to_agent[AcademicRole.PROFESSOR] = []
            self.role_to_agent[AcademicRole.PROFESSOR].append(agent_id)
        elif "博士" in role_type or "phd" in role_type:
            if AcademicRole.PHD_STUDENT not in self.role_to_agent:
                self.role_to_agent[AcademicRole.PHD_STUDENT] = []
            self.role_to_agent[AcademicRole.PHD_STUDENT].append(agent_id)
        elif "硕士" in role_type or "msc" in role_type or "master" in role_type:
            if AcademicRole.MSC_STUDENT not in self.role_to_agent:
                self.role_to_agent[AcademicRole.MSC_STUDENT] = []
            self.role_to_agent[AcademicRole.MSC_STUDENT].append(agent_id)
        else:
            # 默认为研究助理
            if AcademicRole.RESEARCH_ASSISTANT not in self.role_to_agent:
                self.role_to_agent[AcademicRole.RESEARCH_ASSISTANT] = []
            self.role_to_agent[AcademicRole.RESEARCH_ASSISTANT].append(agent_id)
    
    def assign_role_to_task(self, task: Task) -> str:
        """根据任务类型分配最合适的角色
        
        Args:
            task: 要分配的任务
            
        Returns:
            最合适的智能体ID
        """
        # 获取任务最匹配的角色类型
        task_type = task.name
        best_role = TASK_ROLE_MATCH.get(task_type, AcademicRole.PROFESSOR)
        
        # 找到该角色的所有智能体
        agents = self.role_to_agent.get(best_role, [])
        
        # 如果该角色没有可用智能体，尝试其他角色
        if not agents:
            # 按优先级顺序尝试其他角色
            fallback_roles = [
                AcademicRole.PROFESSOR, 
                AcademicRole.PHD_STUDENT, 
                AcademicRole.MSC_STUDENT,
                AcademicRole.RESEARCH_ASSISTANT
            ]
            
            for role in fallback_roles:
                if role in self.role_to_agent and self.role_to_agent[role]:
                    agents = self.role_to_agent[role]
                    break
        
        # 如果仍然没有找到可用智能体，返回None
        if not agents:
            return None
        
        # 选择工作量最少的智能体
        agent_workloads = {}
        for agent_id in agents:
            agent_tasks = self.agent_tasks.get(agent_id, [])
            active_tasks = sum(1 for task_id in agent_tasks 
                              if task_id in self.tasks 
                              and self.tasks[task_id].status != TaskStatus.COMPLETED)
            agent_workloads[agent_id] = active_tasks
        
        # 按工作量排序，选择工作量最少的智能体
        sorted_agents = sorted(agent_workloads.items(), key=lambda x: x[1])
        if sorted_agents:
            chosen_agent = sorted_agents[0][0]
            # 设置任务的角色专长
            task.role_expertise = best_role
            return chosen_agent
        
        return None
    
    def create_task(self, name: str, description: str = "", assignee: str = None,
                  dependencies: List[str] = None, priority: int = 1, 
                  parent_task_id: str = None, tags: List[str] = None,
                  metadata: Dict[str, Any] = None) -> Task:
        """创建新任务
        
        Args:
            name: 任务名称
            description: 任务描述
            assignee: 任务执行者ID
            dependencies: 依赖的任务ID列表
            priority: 优先级(1-5)
            parent_task_id: 父任务ID
            tags: 标签列表
            metadata: 元数据字典
            
        Returns:
            创建的任务对象
        """
        task = Task(
            name=name,
            description=description,
            assignee=assignee,
            dependencies=dependencies or [],
            priority=priority
        )
        
        # 设置标签和元数据
        task.tags = tags or []
        task.metadata = metadata or {}
        
        # 处理父子任务关系
        if parent_task_id and parent_task_id in self.tasks:
            task.parent_task = parent_task_id
            self.tasks[parent_task_id].add_subtask(task.task_id)
        
        # 如果没有指定执行者，自动分配
        if not assignee:
            assignee = self.assign_role_to_task(task)
            task.assignee = assignee
        
        # 保存任务
        self.tasks[task.task_id] = task
        
        # 分配给执行者
        if assignee:
            if assignee not in self.agent_tasks:
                self.agent_tasks[assignee] = []
            self.agent_tasks[assignee].append(task.task_id)
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务对象"""
        return self.tasks.get(task_id)
    
    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """获取指定智能体的所有任务"""
        task_ids = self.agent_tasks.get(agent_id, [])
        return [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks]
    
    def get_pending_tasks(self) -> List[Task]:
        """获取所有待处理的任务"""
        return [task for task in self.tasks.values() 
                if task.status == TaskStatus.PENDING]
    
    def get_ready_tasks(self) -> List[Task]:
        """获取准备就绪的任务（所有依赖已完成）"""
        ready_tasks = []
        for task in self.get_pending_tasks():
            all_deps_complete = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks or self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    all_deps_complete = False
                    break
            if all_deps_complete:
                ready_tasks.append(task)
        
        # 按优先级排序
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks
    
    def execute_task(self, task_id: str) -> Any:
        """执行指定任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务执行结果
            
        Raises:
            ValueError: 如果任务不存在或者没有注册处理函数
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")
        
        task = self.tasks[task_id]
        
        # 根据任务类型和角色选择处理方法
        role_expertise = task.role_expertise or self.get_role_for_agent(task.assignee)
        handler = self.get_role_based_handler(task.name, role_expertise)
        
        if not handler:
            task.fail(f"No handler registered for task type: {task.name}")
            return None
        
        try:
            # 标记任务为进行中
            task.start()
            
            # 执行任务
            result = handler(task)
            
            # 标记任务为完成
            task.complete(result)
            
            return result
        except Exception as e:
            # 记录异常信息
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            task.fail(error_info)
            return None
    
    def get_role_for_agent(self, agent_id: str) -> str:
        """获取智能体的角色类型"""
        for role, agents in self.role_to_agent.items():
            if agent_id in agents:
                return role
        return AcademicRole.RESEARCH_ASSISTANT  # 默认角色
    
    def get_role_based_handler(self, task_name: str, role_type: str) -> Callable:
        """根据任务类型和角色类型获取适当的处理方法"""
        # 首先检查是否有针对特定任务的处理器
        if task_name in self.task_handlers:
            return self.task_handlers[task_name]
        
        # 根据角色类型和任务类型返回适当的LLM处理方法
        return lambda task: self.process_task_with_llm(task, role_type)
    
    def process_task_with_llm(self, task: Task, role_type: str) -> Dict:
        """使用LLM处理任务，根据角色类型生成不同的提示词
        
        Args:
            task: 要处理的任务
            role_type: 角色类型
            
        Returns:
            任务处理结果
        """
        task_type = task.name
        topic = task.metadata.get("topic", "")
        description = task.description
        
        # 获取父任务结果（如果有）
        parent_context = ""
        if task.parent_task and task.parent_task in self.tasks:
            parent_task = self.tasks[task.parent_task]
            if parent_task.status == TaskStatus.COMPLETED and parent_task.result:
                parent_context = f"父任务'{parent_task.name}'的结果: {json.dumps(parent_task.result, ensure_ascii=False)}\n"
        
        # 获取依赖任务结果
        dependencies_context = ""
        for dep_id in task.dependencies:
            if dep_id in self.tasks and self.tasks[dep_id].status == TaskStatus.COMPLETED:
                dep_task = self.tasks[dep_id]
                dependencies_context += f"依赖任务'{dep_task.name}'的结果: {json.dumps(dep_task.result, ensure_ascii=False)}\n"
        
        # 角色专长
        expertise = ROLE_EXPERTISE.get(role_type, [])
        
        # 根据角色类型和任务类型生成提示词
        system_prompt = self.get_role_system_prompt(role_type, expertise)
        user_prompt = self.get_task_prompt(task_type, topic, description, parent_context, dependencies_context)
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用LLM
        response = self.llm_client.get_response(messages)
        
        # 解析返回结果
        result = self.parse_llm_response(response, task_type)
        
        return result
    
    def get_role_system_prompt(self, role_type: str, expertise: List[str]) -> str:
        """根据角色类型生成系统提示词"""
        expertise_str = "、".join(expertise)
        
        role_prompts = {
            AcademicRole.PROFESSOR: f"""你是一位资深教授，专长于{expertise_str}。
            你有丰富的学术经验，思维严谨，擅长批判性思考和理论分析。
            在回答问题时，你会全面考虑问题的各个方面，提供深入而系统的见解。
            你的表达方式学术化、专业化，但同时也能清晰易懂。
            你会关注研究的理论基础、学术意义和创新点，帮助引导研究方向。""",
            
            AcademicRole.PHD_STUDENT: f"""你是一位博士研究生，专长于{expertise_str}。
            你有扎实的研究基础，思维活跃，擅长文献综述和实验设计。
            在回答问题时，你会关注研究方法的选择和实验的可行性，提供具体而实用的建议。
            你的表达方式既有学术性又有创新性，经常提出新的研究思路。
            你会特别关注研究的技术细节和方法论，确保研究的科学性和严谨性。""",
            
            AcademicRole.MSC_STUDENT: f"""你是一位硕士研究生，专长于{expertise_str}。
            你有良好的学术训练，勤奋努力，擅长实验执行和数据处理。
            在回答问题时，你会从实践角度出发，关注实验的可行性和技术实现。
            你的表达方式直接明了，注重问题的解决方案。
            你会特别关注研究的应用价值和技术实现，提供具体的操作建议。""",
            
            AcademicRole.RESEARCH_ASSISTANT: f"""你是一位研究助理，专长于{expertise_str}。
            你有扎实的基础知识，工作细致，擅长资料整理和数据收集。
            在回答问题时，你会注重细节和实用性，提供具体而详尽的支持。
            你的表达方式清晰有条理，善于将信息系统化。
            你会特别关注研究的基础工作和支持性任务，确保研究工作顺利进行。"""
        }
        
        return role_prompts.get(role_type, role_prompts[AcademicRole.RESEARCH_ASSISTANT])
    
    def get_task_prompt(self, task_type: str, topic: str, description: str, 
                        parent_context: str, dependencies_context: str) -> str:
        """根据任务类型生成任务提示词"""
        # 基础上下文
        context = f"研究主题: {topic}\n"
        if description:
            context += f"任务描述: {description}\n"
        if parent_context:
            context += f"{parent_context}\n"
        if dependencies_context:
            context += f"{dependencies_context}\n"
        
        # 特定任务提示词
        task_prompts = {
            "创建会议": f"""请帮我组织一个关于"{topic}"的学术研讨会。
            {context}
            请提供：
            1. 会议主题的详细描述和定义范围
            2. 会议的主要目标和预期成果
            3. 应该讨论的关键问题和研究方向
            4. 会议的结构和组织方式

            请以JSON格式返回，包含以下字段：
            - meeting_title: 会议标题
            - description: 会议描述
            - key_topics: 关键讨论主题列表
            - expected_outcomes: 预期成果列表
            """,
                        
                        "组会主持": f"""请帮我主持关于"{topic}"的学术讨论。
            {context}
            作为主持人，请：
            1. 介绍讨论主题及其重要性
            2. 提出讨论的主要问题和方向
            3. 引导讨论的进行，确保讨论有条理
            4. 总结讨论的主要观点

            请以JSON格式返回，包含以下字段：
            - introduction: 主题介绍
            - key_questions: 关键问题列表
            - discussion_guidance: 讨论引导建议
            - conclusion_points: 总结要点
            """,
                        
                        "会议总结": f"""请帮我总结关于"{topic}"的学术讨论。
            {context}
            请提供：
            1. 讨论的主要内容和观点
            2. 达成的共识或结论
            3. 遗留的问题和未来研究方向
            4. 会议的主要贡献和收获

            请以JSON格式返回，包含以下字段：
            - main_points: 主要讨论点列表
            - conclusions: 结论列表
            - open_questions: 未解决问题列表
            - contributions: 会议贡献列表
            """,
                        
                        "文献调研": f"""请帮我进行关于"{topic}"的文献调研。
            {context}
            请提供：
            1. 该领域的主要研究趋势和方向
            2. 关键相关文献及其主要观点
            3. 现有研究的优缺点和局限性
            4. 可能的研究空白和机会

            请以JSON格式返回，包含以下字段：
            - research_trends: 研究趋势列表
            - key_papers: 关键文献列表（包含标题、作者、年份、主要贡献）
            - limitations: 现有研究局限性列表
            - research_gaps: 研究空白列表
            """,
                        
                        "问题分析": f"""请帮我分析"{topic}"研究中的关键问题。
            {context}
            请提供：
            1. 该研究主题中的核心问题和挑战
            2. 问题的复杂性和多维度分析
            3. 问题的理论基础和学术背景
            4. 解决问题的潜在途径和方法

            请以JSON格式返回，包含以下字段：
            - core_problems: 核心问题列表
            - complexity_analysis: 复杂性分析
            - theoretical_background: 理论背景
            - potential_approaches: 潜在解决方法列表
            """,
                        
                        "方法讨论": f"""请帮我讨论解决"{topic}"研究问题的方法。
            {context}
            请提供：
            1. 适合该问题的主要研究方法
            2. 每种方法的优势、劣势和适用条件
            3. 方法选择的依据和考虑因素
            4. 方法创新的可能性和建议

            请以JSON格式返回，包含以下字段：
            - methods: 研究方法列表（每个包含名称、描述、优势、劣势）
            - selection_criteria: 方法选择标准
            - combination_possibilities: 方法组合可能性
            - innovation_suggestions: 创新建议
            """,
                        
                        "实验设计": f"""请帮我设计验证"{topic}"的实验。
            {context}
            请提供：
            1. 实验目的和要验证的假设
            2. 实验设计（包括样本、变量、控制等）
            3. 数据收集方法和工具
            4. 数据分析方法和预期结果

            请以JSON格式返回，包含以下字段：
            - experiment_purpose: 实验目的
            - hypotheses: 实验假设列表
            - design: 实验设计详情（包含样本、变量、控制）
            - data_collection: 数据收集方法
            - analysis_methods: 数据分析方法
            - expected_results: 预期结果
            """,
                        
                        "结果分析": f"""请帮我分析"{topic}"的研究结果。
            {context}
            请提供：
            1. 主要研究发现和结果解释
            2. 结果的统计和学术意义
            3. 结果与现有理论和文献的关系
            4. 研究局限性和未来改进方向

            请以JSON格式返回，包含以下字段：
            - key_findings: 主要发现列表
            - significance: 结果意义分析
            - relation_to_literature: 与文献关系
            - limitations: 局限性列表
            - future_improvements: 未来改进方向
            """,
                        
                        "论文检索": f"""请帮我检索与"{topic}"相关的论文。
            {context}
            请提供：
            1. 最相关的5-10篇重要论文
            2. 每篇论文的基本信息（标题、作者、发表年份、期刊/会议）
            3. 每篇论文的主要贡献和研究方法
            4. 这些论文对当前研究的启示

            请以JSON格式返回，包含以下字段：
            - papers: 论文列表（每个包含标题、作者、年份、期刊、主要贡献、方法）
            - key_insights: 关键启示列表
            """,
                        
                        "附件分析": f"""请帮我分析与"{topic}"相关的研究附件。
            {context}
            请提供：
            1. 附件的主要内容和类型
            2. 附件中的关键信息和数据
            3. 附件对研究的价值和作用
            4. 附件的质量评估和改进建议

            请以JSON格式返回，包含以下字段：
            - attachment_summary: 附件概述
            - key_information: 关键信息列表
            - research_value: 研究价值分析
            - quality_assessment: 质量评估
            - improvement_suggestions: 改进建议
            """,
                        
                        "生成图谱": f"""请帮我设计"{topic}"的研究关系图谱。
            {context}
            请提供：
            1. 图谱的主要节点和关系
            2. 图谱的结构和组织方式
            3. 图谱中的关键概念和联系
            4. 图谱的解读和使用建议

            请以JSON格式返回，包含以下字段：
            - nodes: 节点列表（包含名称、类型、描述）
            - relationships: 关系列表（包含源节点、目标节点、关系类型、描述）
            - structure: 图谱结构描述
            - interpretation_guide: 解读指南
            """,
            }
                    
            # 返回匹配的提示词，如果没有则使用通用提示词
        default_prompt = f"""请帮我完成关于"{topic}"的{task_type}任务。
            {context}
            请详细分析任务需求并提供专业的解决方案。
            请以JSON格式返回你的分析结果和建议。
            """
        
        return task_prompts.get(task_type, default_prompt)
    
    def parse_llm_response(self, response: str, task_type: str) -> Dict:
        """解析LLM的回复，提取有效信息"""
        # 尝试提取JSON部分
        json_content = response
        
        # 如果返回内容包含JSON代码块，提取它
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        json_matches = re.findall(json_pattern, response)
        if json_matches:
            json_content = json_matches[0].strip()
        
        # 尝试解析JSON
        try:
            result = json.loads(json_content)
            # 添加原始响应
            result["raw_response"] = response
            return result
        except json.JSONDecodeError:
            # 如果无法解析为JSON，返回原始回复
            return {
                "type": task_type,
                "content": response,
                "raw_response": response
            }
    
    def execute_ready_tasks(self) -> List[Tuple[str, Any]]:
        """执行所有准备就绪的任务
        
        Returns:
            List of (task_id, result) tuples
        """
        results = []
        for task in self.get_ready_tasks():
            result = self.execute_task(task.task_id)
            results.append((task.task_id, result))
        return results
    
    def decompose_task(self, task_id: str, subtasks: List[Dict[str, Any]]) -> List[str]:
        """将任务分解为多个子任务
        
        Args:
            task_id: 父任务ID
            subtasks: 子任务配置列表，每个子任务是包含任务配置的字典
            
        Returns:
            List of subtask IDs
        """
        if task_id not in self.tasks:
            raise ValueError(f"Parent task {task_id} does not exist")
        
        parent_task = self.tasks[task_id]
        subtask_ids = []
        
        # 创建子任务
        for subtask_config in subtasks:
            # 确保子任务配置中包含基本信息
            if "name" not in subtask_config:
                continue
                
            # 创建子任务，继承父任务的一些属性
            subtask = self.create_task(
                name=subtask_config["name"],
                description=subtask_config.get("description", ""),
                assignee=subtask_config.get("assignee"),  # 允许自动分配
                dependencies=subtask_config.get("dependencies", []),
                priority=subtask_config.get("priority", parent_task.priority),
                parent_task_id=task_id,
                tags=subtask_config.get("tags", parent_task.tags.copy()),
                metadata=subtask_config.get("metadata", parent_task.metadata.copy())  # 继承父任务元数据
            )
            
            subtask_ids.append(subtask.task_id)
        
        return subtask_ids
    
    def save_to_file(self, filename: str):
        """保存任务状态到文件"""
        data = {
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "agent_tasks": self.agent_tasks,
            "role_to_agent": self.role_to_agent
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filename: str):
        """从文件加载任务状态"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 加载任务
            self.tasks = {
                tid: Task.from_dict(task_data) 
                for tid, task_data in data.get("tasks", {}).items()
            }
            
            # 加载智能体-任务映射
            self.agent_tasks = data.get("agent_tasks", {})
            
            # 加载角色-智能体映射
            self.role_to_agent = data.get("role_to_agent", {})
            
            return True
        except Exception as e:
            print(f"Error loading task data: {e}")
            return False

class AcademicTaskDecomposer:
    """学术任务分解器，用于将高级学术任务分解为子任务"""
    
    def __init__(self, task_manager: TaskManager):
        """初始化分解器
        
        Args:
            task_manager: 任务管理器实例
        """
        self.task_manager = task_manager
    
    def decompose_research_task(self, task_id: str) -> List[str]:
        """分解研究任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            List of subtask IDs
        """
        task = self.task_manager.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} does not exist")
        
        # 构建提示词，要求LLM分解任务
        prompt = f"""
        你是一个学术研究任务规划专家。请将以下研究任务分解为3-7个子任务：
        
        任务名称: {task.name}
        任务描述: {task.description}
        研究主题: {task.metadata.get('topic', '未指定主题')}
        
        请考虑学术研究的完整流程，包括但不限于：文献调研、问题分析、方法讨论、实验设计、结果分析等。
        
        请返回JSON格式的子任务列表，每个子任务包含以下字段:
        - name: 子任务名称（从以下选项中选择：文献调研、问题分析、方法讨论、实验设计、结果分析、论文检索、附件分析、组会主持、会议总结、生成图谱）
        - description: 子任务详细描述
        - priority: 优先级(1-5)，数字越大优先级越高
        - dependencies: 依赖的其他子任务索引列表(如果有)
        
        返回格式示例:
        ```json
        [
          {{
            "name": "文献调研",
            "description": "收集和分析关于人工智能在医疗领域应用的相关文献",
            "priority": 5,
            "dependencies": []
          }},
          {{
            "name": "问题分析",
            "description": "分析AI医疗诊断面临的主要挑战和问题",
            "priority": 4,
            "dependencies": [0]
          }}
        ]
        ```
        
        请确保子任务是合理的，并且覆盖了完成主任务所需的全部工作。每个子任务应该有明确的目标和可衡量的成果。
        子任务之间的依赖关系应该合理，避免循环依赖。
        """
        
        # 调用LLM获取任务分解
        messages = [
            {"role": "system", "content": "你是一个学术研究任务规划专家，擅长将复杂研究任务分解为可管理的步骤。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.task_manager.llm_client.get_response(messages)
            
            # 解析JSON响应
            # 提取JSON部分
            json_content = response
            if "```json" in response:
                json_content = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_content = response.split("```")[1].split("```")[0].strip()
            
            # 解析JSON
            subtasks_config = json.loads(json_content)
            
            # 转换子任务索引依赖为实际任务ID依赖
            subtask_index_to_id = {}
            processed_subtasks = []
            
            # 第一遍：创建所有子任务并记录索引到ID的映射
            for i, subtask in enumerate(subtasks_config):
                # 暂时移除依赖
                dependencies = subtask.pop("dependencies", [])
                subtask["dependencies"] = []
                
                # 继承父任务的元数据
                subtask["metadata"] = task.metadata.copy()
                
                processed_subtasks.append(subtask)
            
            # 分解任务
            subtask_ids = self.task_manager.decompose_task(task_id, processed_subtasks)
            
            # 记录索引到ID的映射
            for i, subtask_id in enumerate(subtask_ids):
                subtask_index_to_id[i] = subtask_id
            
            # 第二遍：更新依赖关系
            for i, subtask_config in enumerate(subtasks_config):
                raw_dependencies = subtask_config.get("dependencies", [])
                if not raw_dependencies:
                    continue
                
                # 获取子任务ID
                subtask_id = subtask_ids[i]
                subtask = self.task_manager.get_task(subtask_id)
                
                # 更新依赖
                for dep_index in raw_dependencies:
                    if isinstance(dep_index, int) and dep_index in subtask_index_to_id:
                        dep_id = subtask_index_to_id[dep_index]
                        subtask.dependencies.append(dep_id)
            
            return subtask_ids
        except Exception as e:
            print(f"Error decomposing task: {e}")
            return []

class AcademicGroupMeetingOrchestrator:
    """学术组会编排器，整合任务系统与学术组会系统"""
    
    def __init__(self, academic_meeting_system, use_remote_llm=False):
        """初始化编排器
        
        Args:
            academic_meeting_system: 学术组会系统实例
            use_remote_llm: 是否使用远程LLM
        """
        self.academic_system = academic_meeting_system
        self.task_manager = TaskManager(use_remote_llm=use_remote_llm)
        self.task_decomposer = AcademicTaskDecomposer(self.task_manager)
        
        # 跟踪当前活动场景
        self.active_scene_id = None
        self.meeting_metadata = {}
    
    def schedule_meeting(self, topic: str, moderator_id: str = None, rounds: int = 3, 
                        deep_search: bool = False) -> str:
        """安排和规划学术组会讨论
        
        Args:
            topic: 讨论主题
            moderator_id: 主持人ID
            rounds: 讨论轮次
            deep_search: 是否使用深度搜索
            
        Returns:
            会议任务ID
        """
        # 注册智能体
        self._register_agents()
        
        # 创建会议主任务
        meeting_task = self.task_manager.create_task(
            name="创建学术组会",
            description=f"组织关于'{topic}'的学术讨论",
            assignee=moderator_id,  # 如果为None，会自动分配
            priority=5,
            metadata={
                "topic": topic,
                "rounds": rounds,
                "deep_search": deep_search
            }
        )
        
        # 使用任务分解器分解会议任务
        try:
            subtask_ids = self.task_decomposer.decompose_research_task(meeting_task.task_id)
            print(f"会议任务已分解为{len(subtask_ids)}个子任务")
        except Exception as e:
            print(f"分解会议任务失败: {e}")
            
        # 记录会议任务ID
        self.meeting_metadata["current_meeting_task_id"] = meeting_task.task_id
        
        return meeting_task.task_id
    
    def _register_agents(self):
        """注册系统中的智能体到任务管理器"""
        for agent_id, agent_info in self.academic_system.backend.agents.items():
            self.task_manager.register_agent(agent_id, agent_info)
    
    def execute_meeting_plan(self, meeting_task_id: str):
        """执行会议计划
        
        Args:
            meeting_task_id: 会议任务ID
        """
        try:
            # 执行主任务
            print(f"开始执行主任务: {meeting_task_id}")
            main_result = self.task_manager.execute_task(meeting_task_id)
            
            # 获取场景信息
            task = self.task_manager.get_task(meeting_task_id)
            if task and task.result:
                # 如果主任务返回的是包含scene_id的结果
                scene_id = task.result.get("scene_id")
                if scene_id:
                    self.active_scene_id = scene_id
                    print(f"设置当前场景ID: {scene_id}")
            
            # 创建会议场景（如果还没有）
            if not self.active_scene_id:
                self._create_initial_scene(task)
            
            # 添加会议初始消息
            if self.active_scene_id and task:
                topic = task.metadata.get("topic", "未指定主题")
                moderator_id = task.assignee
                
                if moderator_id:
                    initial_message = f"我们今天要讨论的主题是：{topic}"
                    self.academic_system.add_message(self.active_scene_id, moderator_id, initial_message)
            
            # 持续执行就绪的子任务
            executed_count = 0
            max_iterations = 20  # 防止无限循环
            
            while executed_count < max_iterations:
                ready_tasks = self.task_manager.get_ready_tasks()
                if not ready_tasks:
                    print("没有更多就绪的任务")
                    break
                
                print(f"发现{len(ready_tasks)}个就绪任务")
                results = self.task_manager.execute_ready_tasks()
                executed_count += len(results)
                
                if results:
                    print(f"执行了{len(results)}个任务")
                    
                    # 处理任务结果
                    for task_id, result in results:
                        task = self.task_manager.get_task(task_id)
                        if task and result:
                            self._process_task_result(task, result)
                else:
                    print("没有任务被执行")
                    break
            
            print(f"总共执行了{executed_count}个任务")
            
            # 检查主任务状态
            meeting_task = self.task_manager.get_task(meeting_task_id)
            if meeting_task and meeting_task.status == TaskStatus.COMPLETED:
                print(f"会议'{meeting_task.metadata.get('topic', '未知主题')}'已成功完成")
                return True
            else:
                print("会议执行未完成或发生错误")
                return False
                
        except Exception as e:
            print(f"执行会议计划时出错: {e}")
            traceback.print_exc()
            return False
    
    def _create_initial_scene(self, task):
        """创建初始会议场景"""
        if not task:
            print("无法创建场景：任务不存在")
            return
            
        topic = task.metadata.get("topic", "未指定主题")
        scene_name = f"{topic}学术研讨会"
        scene_description = f"讨论主题：{topic}，探讨研究现状、方法技术与未来方向"
        
        scene_id = self.academic_system.create_scene(scene_name, scene_description)
        self.active_scene_id = scene_id
        task.metadata["scene_id"] = scene_id
        
        print(f"创建了新的场景: {scene_id}")
        return scene_id
    
    def _process_task_result(self, task, result):
        """处理任务执行结果"""
        scene_id = self.active_scene_id or task.metadata.get("scene_id")
        if not scene_id:
            print(f"无法处理结果：未找到场景ID (任务: {task.name})")
            return
            
        assignee = task.assignee
        if not assignee:
            print(f"无法处理结果：未找到任务执行者 (任务: {task.name})")
            return
        
        # 将结果添加到对话
        content = self._format_result_message(task.name, result)
        if content:
            try:
                self.academic_system.add_message(scene_id, assignee, content)
                print(f"添加了任务结果消息: {task.name}")
            except Exception as e:
                print(f"添加消息时出错: {e}")
        
        # 处理特殊任务结果
        if task.name == "生成图谱" and scene_id:
            try:
                self.academic_system.visualize_conversation_graph(scene_id)
                print("生成了会话图谱")
            except Exception as e:
                print(f"生成图谱时出错: {e}")
    
    def _format_result_message(self, task_type, result):
        """格式化任务结果为可读消息"""
        if not result:
            return None
            
        # 如果有原始回复，直接使用
        if "raw_response" in result:
            # 移除原始回复中可能的JSON格式代码块
            response = result["raw_response"]
            cleaned_response = re.sub(r'```(?:json)?\s*[\s\S]*?```', '', response)
            cleaned_response = cleaned_response.strip()
            
            # 如果清理后内容非空，使用清理后的内容
            if cleaned_response:
                return cleaned_response
        
        # 否则，根据任务类型构建结构化消息
        formatted_messages = {
            "创建会议": lambda r: f"我们将组织一个关于「{r.get('meeting_title', '研究主题')}」的学术讨论。\n\n{r.get('description', '')}\n\n我们将讨论以下关键主题：\n" + "\n".join([f"- {topic}" for topic in r.get('key_topics', [])]),
            
            "组会主持": lambda r: f"{r.get('introduction', '')}\n\n我想提出以下关键问题供大家讨论：\n" + "\n".join([f"- {q}" for q in r.get('key_questions', [])]),
            
            "会议总结": lambda r: f"关于我们的讨论，我想做个总结：\n\n主要观点：\n" + "\n".join([f"- {point}" for point in r.get('main_points', [])]) + f"\n\n结论：\n" + "\n".join([f"- {conc}" for conc in r.get('conclusions', [])]),
            
            "文献调研": lambda r: f"经过文献调研，我发现这个领域的主要研究趋势是：\n" + "\n".join([f"- {trend}" for trend in r.get('research_trends', [])]) + f"\n\n关键文献包括：\n" + "\n".join([f"- {paper.get('title', '')}" for paper in r.get('key_papers', [])]),
            
            "问题分析": lambda r: f"我认为这个研究主题的核心问题包括：\n" + "\n".join([f"- {problem}" for problem in r.get('core_problems', [])]) + f"\n\n理论背景：{r.get('theoretical_background', '')}",
            
            "方法讨论": lambda r: f"针对这个问题，我们可以考虑以下研究方法：\n" + "\n".join([f"- {method.get('name', '')}: {method.get('description', '')}" for method in r.get('methods', [])]),
            
            "实验设计": lambda r: f"我设计了以下实验来验证我们的假设：\n\n实验目的：{r.get('experiment_purpose', '')}\n\n假设：\n" + "\n".join([f"- {hyp}" for hyp in r.get('hypotheses', [])]) + f"\n\n实验设计：{r.get('design', '')}",
            
            "结果分析": lambda r: f"从分析来看，主要发现包括：\n" + "\n".join([f"- {finding}" for finding in r.get('key_findings', [])]) + f"\n\n这些发现的意义在于：{r.get('significance', '')}",
            
            "论文检索": lambda r: f"我找到了几篇与我们研究相关的重要论文：\n" + "\n".join([f"- {paper.get('title', '')} ({paper.get('year', '')}): {paper.get('main_contribution', '')}" for paper in r.get('papers', [])]),
            
            "附件分析": lambda r: f"我分析了研究附件，主要内容是：{r.get('attachment_summary', '')}\n\n关键信息包括：\n" + "\n".join([f"- {info}" for info in r.get('key_information', [])]),
            
            "生成图谱": lambda r: f"我设计了研究关系图谱，包含以下主要概念：\n" + "\n".join([f"- {node.get('name', '')}: {node.get('description', '')}" for node in r.get('nodes', [])])
        }
        
        # 使用匹配的格式化函数，如果没有则返回原始JSON
        formatter = formatted_messages.get(task_type)
        if formatter:
            try:
                return formatter(result)
            except Exception as e:
                print(f"格式化消息时出错: {e}")
        
        # 默认返回JSON字符串
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except:
            return str(result)
