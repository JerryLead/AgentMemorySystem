from semantic_map_case.academic_group_meeting.memory_manager import MemoryManager
from semantic_map_case.academic_group_meeting.meeting_utils import MeetingUtils

# 创建全局的记忆管理器实例
memory_manager = MemoryManager()

def generate_comprehensive_review(student_data, meeting_data, phase_name=None):
    """
    生成综合评审报告，包含阶段记忆、上下文记忆，并生成会议纪要和任务分配
    
    参数:
        student_data: 学生数据
        meeting_data: 会议数据
        phase_name: 当前阶段名称(可选)
    
    返回:
        包含评审信息、会议纪要和任务的字典
    """
    # 从记忆中获取上下文
    context_memory = memory_manager.get_context_memory()
    
    # 将上下文记忆嵌入到当前处理中
    enhanced_data = {
        "student_data": student_data,
        "meeting_data": meeting_data,
        "context_memory": context_memory
    }
    
    # 如果有指定阶段，添加阶段记忆
    if phase_name:
        phase_memory = memory_manager.get_phase_memory(phase_name)
        enhanced_data["phase_memory"] = phase_memory
    
    # 分析数据并生成评审内容
    review_content = analyze_data(enhanced_data)
    
    # 提取本次会议的讨论要点和决定
    discussion_points = extract_discussion_points(meeting_data, review_content)
    decisions = extract_decisions(meeting_data, review_content)
    
    # 生成会议纪要
    meeting_summary = MeetingUtils.generate_meeting_summary(meeting_data, discussion_points, decisions)
    memory_manager.add_meeting_record(meeting_summary)
    
    # 打印会议纪要
    MeetingUtils.print_meeting_summary(meeting_summary)
    
    # 生成并分配任务
    students = [student["name"] for student in student_data]
    tasks = MeetingUtils.generate_tasks(students, meeting_summary)
    
    # 保存任务到记忆管理器
    for task in tasks:
        memory_manager.add_task(task)
    
    # 打印任务
    MeetingUtils.print_tasks(tasks)
    
    # 更新记忆
    update_memories(review_content, meeting_data, phase_name)
    
    # 返回最终结果
    result = {
        "review": review_content,
        "meeting_summary": meeting_summary,
        "tasks": tasks
    }
    
    return result

def analyze_data(enhanced_data):
    """分析增强数据生成评审内容"""
    # 这里是分析学生数据、会议数据以及记忆数据的具体实现
    # ...实际分析逻辑...
    
    review_content = {
        "strengths": ["..."],
        "weaknesses": ["..."],
        "recommendations": ["..."]
    }
    return review_content

def extract_discussion_points(meeting_data, review_content):
    """从会议数据和评审内容中提取讨论要点"""
    # 这里应该有提取讨论要点的具体逻辑
    points = meeting_data.get("topics", [])
    # 加入从评审内容提取的要点
    points.extend(review_content.get("recommendations", []))
    return points

def extract_decisions(meeting_data, review_content):
    """从会议数据和评审内容中提取决定"""
    # 这里应该有提取决定的具体逻辑
    decisions = meeting_data.get("decisions", [])
    return decisions

def update_memories(review_content, meeting_data, phase_name):
    """更新记忆系统"""
    # 更新上下文记忆
    memory_item = {
        "timestamp": meeting_data.get("timestamp", ""),
        "summary": review_content
    }
    memory_manager.add_context_memory(memory_item)
    
    # 如果有指定阶段，更新阶段记忆
    if phase_name:
        phase_memory_content = {
            "timestamp": meeting_data.get("timestamp", ""),
            "key_points": review_content.get("recommendations", [])
        }
        memory_manager.add_phase_memory(phase_name, phase_memory_content)
