import sys
import os
from typing import List, Dict, Tuple, Any, Optional
import uuid
import time
from datetime import datetime

# 确保导入路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from academic_group_meeting_backend import AcademicMeetingSystem
from milvus_enabled_academic_meeting_system import MilvusEnabledAcademicMeetingSystem
from task_manager import AcademicGroupMeetingOrchestrator, AcademicRole
from semantic_map.deepseek_client import deepseek_local, deepseek_remote

# 定义输出目录常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

# 确保输出目录存在
os.makedirs(JSON_DIR, exist_ok=True)


def create_default_agents(meeting_system: AcademicMeetingSystem) -> Dict[str, Dict[str, Any]]:
    """创建默认学术智能体
    
    Args:
        meeting_system: 学术组会系统实例
        
    Returns:
        Dict: agent_id -> agent_info 的映射
    """
    agents = {}
    
    # 创建教授
    professor_id = "professor_1"
    meeting_system.create_academic_agent(
        agent_id=professor_id,
        nickname="许教授",
        age=45,
        role_type="教授",
        specialty=["人工智能", "机器学习", "知识图谱"],
        personality="严谨认真，关注理论基础，注重学术严谨性"
    )
    agents[professor_id] = {
        "nickname": "许教授",
        "role_type": "教授",
        "academic_role": AcademicRole.PROFESSOR
    }
    
    # 创建博士生
    phd_id = "phd_1"
    meeting_system.create_academic_agent(
        agent_id=phd_id,
        nickname="李同学",
        age=28,
        role_type="博士生",
        specialty=["深度学习", "自然语言处理"],
        personality="思维活跃，善于提出创新观点，但有时缺乏系统性"
    )
    agents[phd_id] = {
        "nickname": "李同学",
        "role_type": "博士生",
        "academic_role": AcademicRole.PHD_STUDENT
    }
    
    # 创建硕士生1
    msc1_id = "msc_1"
    meeting_system.create_academic_agent(
        agent_id=msc1_id,
        nickname="郭同学",
        age=25,
        role_type="硕士生",
        specialty=["计算机视觉", "多模态学习"],
        personality="勤奋努力，注重实验细节，对新技术有很强的学习能力"
    )
    agents[msc1_id] = {
        "nickname": "郭同学",
        "role_type": "硕士生",
        "academic_role": AcademicRole.MSC_STUDENT
    }
    
    # 创建硕士生2
    msc2_id = "msc_2"
    meeting_system.create_academic_agent(
        agent_id=msc2_id,
        nickname="吴同学",
        age=24,
        role_type="硕士生",
        specialty=["知识图谱", "图神经网络"],
        personality="性格沉稳，逻辑思维强，善于思考问题的本质"
    )
    agents[msc2_id] = {
        "nickname": "吴同学",
        "role_type": "硕士生",
        "academic_role": AcademicRole.MSC_STUDENT
    }
    
    # 创建研究助理（可选）
    ra_id = "ra_1"
    meeting_system.create_academic_agent(
        agent_id=ra_id,
        nickname="陈助理",
        age=26,
        role_type="研究助理",
        specialty=["文献管理", "数据分析", "实验记录"],
        personality="细致认真，擅长整理资料和记录，保持高效的研究支持工作"
    )
    agents[ra_id] = {
        "nickname": "陈助理",
        "role_type": "研究助理",
        "academic_role": AcademicRole.RESEARCH_ASSISTANT
    }
    
    return agents

def run_academic_meeting_autogen(
        topic: str,
        professor_id: str = "professor_1",
        rounds: int = 3,
        deep_search: bool = False,
        use_milvus: bool = False,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "20031117",
        neo4j_database: str = "academicgraph",
        use_remote_llm: bool = True,
        use_local_embeddings: bool = True,
        local_text_model_path: str = "/home/zyh/model/clip-ViT-B-32-multilingual-v1",
        local_image_model_path: str = "/home/zyh/model/clip-ViT-B-32",
        clear_neo4j: bool = False,
        custom_output=None  # 可以传入自定义输出对象
    ) -> str:
    """使用AutoGen任务系统运行学术组会"""
    # 重定向标准输出（如果提供了自定义输出）
    original_stdout = sys.stdout
    if custom_output:
        sys.stdout = custom_output
    
    try:
        # 初始化学术组会系统
        print("系统：初始化学术组会系统...")
        
        # 根据配置选择合适的系统实现
        if use_milvus:
            meeting_system = MilvusEnabledAcademicMeetingSystem(
                use_remote_llm=use_remote_llm,
                use_local_embeddings=use_local_embeddings,
                local_text_model_path=local_text_model_path,
                local_image_model_path=local_image_model_path,
                milvus_host=milvus_host,
                milvus_port=milvus_port,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database
            )
        else:
            meeting_system = AcademicMeetingSystem(
                use_remote_llm=use_remote_llm,
                use_local_embeddings=use_local_embeddings,
                local_text_model_path=local_text_model_path,
                local_image_model_path=local_image_model_path
            )
        
        # 创建默认智能体
        print("系统：创建学术智能体...")
        agents = create_default_agents(meeting_system)
        
        # 初始化学术组会编排器
        orchestrator = AcademicGroupMeetingOrchestrator(meeting_system, use_remote_llm=use_remote_llm)
        
        # 安排学术组会讨论
        print(f"\n=== 开始规划学术组会：{topic} ===\n")
        meeting_task_id = orchestrator.schedule_meeting(
            topic=topic,
            moderator_id=professor_id,
            rounds=rounds,
            deep_search=deep_search
        )
        
        # 执行会议计划前先输出明确的开始信息
        print(f"许教授：作为主持人，我将开始组织关于'{topic}'的学术讨论。")
        print("\n=== 开始执行学术组会计划 ===\n")
        
        # 明确告知用户任务系统正在启动
        print(f"系统：正在启动AutoGen任务系统，初始化协作智能体...")
        
        # 确保所有输出内容被刷新显示
        if custom_output:
            custom_output.flush()
            
        try:
            # 执行会议计划
            print(f"系统：开始执行学术组会任务流，主题：'{topic}'")
            orchestrator.execute_meeting_plan(meeting_task_id)
            print(f"系统：学术组会任务流执行完成")
        except Exception as e:
            print(f"系统：执行任务时发生错误: {str(e)}")
        
        # 获取会议场景ID
        scene_id = orchestrator.active_scene_id
        
        # 导出到Neo4j（如果启用）
        if use_milvus and clear_neo4j:
            try:
                print("\n=== 导出会议数据到Neo4j ===\n")
                meeting_system.export_to_neo4j()
            except Exception as e:
                print(f"系统：导出到Neo4j失败: {e}")
        
        # 保存任务状态数据供前端使用
        try:
            tasks_file = os.path.join(JSON_DIR, f"meeting_tasks_{scene_id}.json")
            orchestrator.task_manager.save_to_file(tasks_file)
            print(f"系统：任务数据已保存到 {tasks_file}")
        except Exception as e:
            print(f"系统：保存任务数据失败: {e}")
        
        print("\n=== 学术组会完成 ===\n")
        print("许教授：我们的学术组会讨论已经结束，感谢各位的参与和贡献。")
        
        # 最终刷新确保所有输出都被处理
        if custom_output:
            custom_output.flush()
            
        return scene_id
    
    finally:
        # 恢复原始标准输出
        if custom_output:
            sys.stdout = original_stdout

# 如果直接运行，执行测试会议
if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="运行学术组会讨论")
    parser.add_argument("--topic", type=str, default="大语言模型的应用与挑战", help="讨论主题")
    parser.add_argument("--rounds", type=int, default=3, help="讨论轮次数")
    parser.add_argument("--deep-search", action="store_true", help="是否使用深度搜索")
    parser.add_argument("--use-milvus", action="store_true", help="是否使用Milvus向量存储")
    parser.add_argument("--use-local-llm", action="store_true", help="是否使用本地LLM")
    args = parser.parse_args()
    
    # 运行会议
    scene_id = run_academic_meeting_autogen(
        topic=args.topic,
        rounds=args.rounds,
        deep_search=args.deep_search,
        use_milvus=args.use_milvus,
        use_remote_llm=not args.use_local_llm
    )
    
    print(f"会议场景ID: {scene_id}")