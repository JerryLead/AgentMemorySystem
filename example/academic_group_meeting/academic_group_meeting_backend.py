import random
import sys
import os
import faiss
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import re
import json
import requests
from semantic_map.deepseek_client import deepseek_remote, deepseek_local
from sentence_transformers import SentenceTransformer
import uuid
import time
from datetime import datetime
from typing import List, Dict , Set, Tuple, Any
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from semantic_map.data_type import BaseDataType
from semantic_map.semantic_map import BaseSemanticMap
from semantic_map.semantic_simple_graph import BaseSemanticSimpleGraph
from enum import auto
import numpy as np
# from academic_group_meeting_backend import AcademicMeetingSystem
from semantic_map import MilvusDialogueStorage
from semantic_map import Neo4jInterface
from matplotlib import pyplot as plt
from academic_group_meeting_graph import *
from AttachmentSearchEngine import AttachmentSearchEngine
from LocalAttachmentProcessor import LocalAttachmentProcessor

# 在文件开头添加输出目录定义
import os
from pathlib import Path

from memory_manager import DialogueMemory, MemoryType, SummaryMemory

# 定义输出目录常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PICTURE_DIR = os.path.join(OUTPUT_DIR, "picture")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
ARTICLE_DIR = os.path.join(OUTPUT_DIR, "article")
ATTACHMENT_DIR = os.path.join(BASE_DIR, "attachment")

# 确保输出目录存在
os.makedirs(PICTURE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(ARTICLE_DIR, exist_ok=True)
os.makedirs(ATTACHMENT_DIR, exist_ok=True)

# plt.rc('SimHei') # 设置中文显示
try:
    plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：可能无法正确显示中文，请安装相应字体")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入部分保持不变...
class MessageStream:
    """结构化消息输出系统，替代简单的print语句"""
    
    # def __init__(self, stream=sys.stdout):
    #     self.stream = stream
    def __init__(self):
        pass
    
    def system_message(self, content):
        """输出系统消息"""
        self._send_message({
            "type": "system", 
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def round_info(self, round_num, topic=None):
        """输出讨论轮次信息"""
        content = f"第{round_num}轮讨论"
        if topic:
            content += f": {topic}"
            
        self._send_message({
            "type": "round_info",
            "round": round_num,
            "topic": topic,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def agent_message(self, agent_id, agent_name, content, message_type="normal"):
        """输出代理消息（教授或学生）"""
        self._send_message({
            "type": "agent",
            "agent_id": agent_id,
            "agent_name": agent_name,
            "content": content,
            "message_type": message_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def summary(self, agent_id, agent_name, content, summary_type, round_num=None):
        """输出总结消息"""
        self._send_message({
            "type": "summary",
            "agent_id": agent_id,
            "agent_name": agent_name,
            "content": content,
            "summary_type": summary_type,
            "round": round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def _send_message(self, message_dict):
        """发送JSON格式消息"""
        # 添加特殊标记，使前端更容易识别消息边界
        print("<MSG_START>" + json.dumps(message_dict, ensure_ascii=False) + "<MSG_END>")\
        # print("<MSG_START>"+str(message_dict)+"<MSG_END>")
        # self.stream.write("<MSG_START>")=以
        # json.dump(message_dict, self.stream, ensure_ascii=False)
        # self.stream.write("<MSG_END>\n")
        # self.stream.flush()

class AcademicGroupMeetingDataSystem:
    def __init__(self, 
            use_remote_llm: bool = True,  # 将 use_remote 改为 use_remote_llm
            use_local_embeddings: bool = True,
            local_text_model_path: str = "/home/zyh/model/clip-ViT-B-32-multilingual-v1",
            local_image_model_path: str = "/home/zyh/model/clip-ViT-B-32",
            local_model: str = "deepseek-r1:1.5b"):
        """
        初始化学术组会数据系统
        
        Args:
            use_remote_llm: 是否使用远程大语言模型 
            use_local_embeddings: 是否使用本地嵌入模型
            local_text_model_path: 本地文本模型路径
            local_image_model_path: 本地图像模型路径
        """
        # 初始化语义地图，传递本地模型参数
        self.semantic_map = AcademicGroupMeetingMap(
            use_local_model=use_local_embeddings,
            local_text_model_path=local_text_model_path,
            local_image_model_path=local_image_model_path
        )
        self.semantic_graph = AcademicGroupMeetingGraph(self.semantic_map)
        self.agents = {}  # {agent_id: agent_info}
        # 根据参数使用远程或本地 deepseek 客户端
        if use_remote_llm:
            self.llm = deepseek_remote() 
        else:
            self.llm = deepseek_local(memory_type="buffer",
                                      max_token_limit=2000,
                                      window_size=10,
                                      system_prompt="You are a helpful assistant.",
                                      local_model=local_model)
 
    def create_agent(self, agent_id: str, agent_info: dict):
        """
        创建智能体，将其添加到 SemanticMap 和 SemanticSimpleGraph 中
        根据角色类型选择合适的数据类型
        """
        self.agents[agent_id] = agent_info
        
        # 根据智能体的角色选择合适的数据类型
        occupation = agent_info.get("Occupation", "").lower()
        if "教授" in occupation or "professor" in occupation:
            agent_type = AcademicDataType.Professor
        elif "博士" in occupation or "phd" in occupation:
            agent_type = AcademicDataType.PhD
        elif "硕士" in occupation or "master" in occupation:
            agent_type = AcademicDataType.Master
        else:
            # 默认为PhD
            agent_type = AcademicDataType.PhD
            
        self.semantic_graph.add_node(agent_id, agent_info, agent_type)
        print(f"创建了智能体 {agent_info.get('Nickname', agent_id)}，类型：{agent_type.name}")

    def add_event(self, event_id: str, event_info: dict, parent_keys: List[str] = None):
        """
        添加事件节点，学术讨论类型
        """
        self.semantic_graph.add_node(event_id, event_info, AcademicDataType.Discussion, parent_keys)

    def add_conversation(self, conversation_id: str, conversation_info: dict, parent_keys: List[str] = None):
        """
        添加对话节点，根据内容可能是问题或讨论
        """
        # 判断是问题还是普通讨论
        content = conversation_info.get("Content", "").lower()
        if "?" in content or "？" in content or "问题" in content or "question" in content:
            conv_type = AcademicDataType.Question
        else:
            conv_type = AcademicDataType.Discussion
            
        self.semantic_graph.add_node(conversation_id, conversation_info, conv_type, parent_keys)

    def add_research_topic(self, topic_id: str, topic_info: dict, parent_keys: List[str] = None):
        """
        添加研究主题节点
        """
        self.semantic_graph.add_node(topic_id, topic_info, AcademicDataType.ResearchTopic, parent_keys)

    def add_paper(self, paper_id: str, paper_info: dict, parent_keys: List[str] = None):
        """
        添加论文节点
        """
        self.semantic_graph.add_node(paper_id, paper_info, AcademicDataType.Paper, parent_keys)

    def add_conclusion(self, conclusion_id: str, conclusion_info: dict, parent_keys: List[str] = None):
        """
        添加结论节点
        """
        self.semantic_graph.add_node(conclusion_id, conclusion_info, AcademicDataType.Conclusion, parent_keys)

    def add_attachment(self, attachment_id: str, attachment_info: dict, parent_keys: List[str] = None):
        """
        添加附件节点
        """
        self.semantic_graph.add_node(attachment_id, attachment_info, AcademicDataType.Attachment, parent_keys)

    def query(self, query_text: str, k: int = 5):
        """
        查询 SemanticMap 返回最相似的节点
        """
        return self.semantic_graph.retrieve_similar_nodes(query_text, k)

    def visualize_graph(self, filename=None, fontpath=None):
        """可视化学术组会图
        
        Args:
            filename: 输出文件名，如果为None则使用默认名称
            fontpath: 字体路径，用于解决中文显示问题
        """
        if filename is None:
            filename = os.path.join(PICTURE_DIR, "academic_meeting_graph.png")
        if fontpath is None:
            self.semantic_graph.visualize_academic_meeting("学术组会关系图")
        else:
            self.semantic_graph.visualize_academic_meeting("学术组会关系图", fontpath=fontpath)

    def build_implicit_graph(self, M=16, ef_construction=200, ef_search=50):
        self.semantic_graph.add_implicit_edges(M, ef_construction, ef_search)

    def query_implicit_neighbors(self, query_text: str, k=5) -> list:
        query_emb = self.semantic_map._get_text_embedding(query_text)
        return self.semantic_graph.query_implicit_neighbors(query_emb, k)

    def auto_dialogue(self, speaker_id: str, message: str) -> str:
        """
        实现自动对话：调用 LLM（deepseek_client）基于当前对话历史生成回复，
        对话历史可以结合语义图中的相关节点上下文进行拼接。
        """
        speaker = self.agents.get(speaker_id, {}).get("Nickname", speaker_id)
        prompt = f"{speaker} says: {message}\nReply:"
        # 将 prompt 包装为消息列表：系统消息+用户消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.get_response(messages)
        if response:
            # 将自动生成的回复作为对话节点插入
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            conv_info = {
                "Content": response,
                "Speaker": speaker_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Participants": [speaker_id]
            }
            # 判断是问题还是普通讨论
            if "?" in response or "？" in response or "question" in response.lower():
                self.semantic_graph.add_node(conversation_id, conv_info, AcademicDataType.Question, [speaker_id])
            else:
                self.semantic_graph.add_node(conversation_id, conv_info, AcademicDataType.Discussion, [speaker_id])
                
            return response
        else:
            return "No response generated."

    def generate_event(self, description: str) -> str:
        """
        基于输入描述调用 LLM API 自动生成事件信息，并作为学术讨论节点插入语义图。
        """
        prompt = f"Generate a detailed academic discussion description based on the following input: {description}\nDiscussion:"
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant specialized in academic discussions."},
            {"role": "user", "content": prompt}
        ]
        event_response = self.llm.get_response(messages)
        if event_response:
            event_id = f"discussion_{uuid.uuid4().hex[:8]}"
            event_info = {
                "Content": event_response,
                "Topic": description,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Participants": []
            }
            self.semantic_graph.add_node(key=event_id, value=event_info, datatype=AcademicDataType.Discussion)
            print(f"生成的讨论 [{event_id}]: {event_response}")
            return event_id
        else:
            print("生成讨论失败。")
            return None

    def add_summary(self, summary_id: str, summary_info: dict, parent_keys: List[str] = None):
        """
        添加总结节点，用于存储教授的讨论总结
        
        Args:
            summary_id: 总结节点的唯一标识
            summary_info: 总结信息字典
            parent_keys: 父节点列表，如相关讨论、教授等
        """
        self.semantic_graph.add_node(key=summary_id, value=summary_info, 
                                     datatype=AcademicDataType.Summary, parent_keys=parent_keys)
        print(f"添加总结: {summary_info.get('Title', 'Untitled')}")

    def add_task(self, task_id: str, task_info: dict, parent_keys: List[str] = None):
        """
        添加任务节点，用于存储研究过程中产生的任务
        
        Args:
            task_id: 任务节点的唯一标识
            task_info: 任务信息字典
            parent_keys: 父节点列表，如相关讨论、总结、教授等
        """
        self.semantic_graph.add_node(key=task_id, value=task_info, 
                                     datatype=AcademicDataType.Task, parent_keys=parent_keys)
        print(f"添加任务: {task_info.get('Title', 'Untitled')}")

    def _generate_structured_summary(self, content: str, topic: str, summary_type: str, round_num=None) -> dict:
        """使用LLM生成结构化总结内容
        
        Args:
            content: 原始总结内容
            topic: 主题
            summary_type: 总结类型
            round_num: 轮次编号
            
        Returns:
            dict: 结构化总结内容
        """
        type_desc = "阶段性总结" if summary_type == "round" else "最终总结"
        round_text = f"第{round_num}轮" if round_num is not None else ""
        
        prompt = f"""
        请将以下关于"{topic}"的{round_text}{type_desc}内容结构化为更有条理的格式。

        原始总结:
        {content}

        请提取并组织为以下结构（以JSON格式返回）:
        1. key_findings: 主要发现和讨论要点（列出3-5点）
        2. challenges: 当前面临的挑战（列出2-3点）
        3. future_directions: 未来研究方向（列出2-3点）
        4. research_gaps: 研究空白和机会（列出1-2点）
        5. action_items: 具体行动项目（列出3-4点）

        确保内容简洁明了，每点不超过50字。返回格式应为有效的JSON。
        """
        
        messages = [
            {"role": "system", "content": "你是一位擅长总结学术讨论并提取结构化信息的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm.get_response(messages)
            # 尝试解析JSON响应
            structured_content = json.loads(response)
            return structured_content
        except Exception as e:
            print(f"结构化总结生成失败: {str(e)}")
            # 返回默认结构
            return {
                "key_findings": "未能成功提取主要发现",
                "challenges": "未能成功提取挑战",
                "future_directions": "未能成功提取未来方向",
                "research_gaps": "未能成功提取研究空白",
                "action_items": "未能成功提取行动项目"
            }

# 确保导入所需模块

class AcademicMeetingSystem:
    """学术组会系统，独立实现，使用AcademicGroupMeetingDataSystem"""
    def __init__(self, 
                use_remote_llm: bool = True,
                use_local_embeddings: bool = True,
                local_text_model_path: str = "/home/zyh/model/clip-ViT-B-32-multilingual-v1",
                local_image_model_path: str = "/home/zyh/model/clip-ViT-B-32",
                local_model: str = "deepseek-r1:1.5b"):
        """
        初始化学术组会系统
        
        Args:
            use_remote_llm: 是否使用远程大语言模型服务
            use_local_embeddings: 是否使用本地嵌入模型
            local_text_model_path: 本地文本嵌入模型路径
            local_image_model_path: 本地图像嵌入模型路径
            local_model: 本地模型名称，仅在use_remote_llm为False时有效
        """
        # 使用学术组会专用的数据系统，传递本地模型参数
        self.backend = AcademicGroupMeetingDataSystem(
            use_remote_llm=use_remote_llm,  # 改为统一的参数名
            use_local_embeddings=use_local_embeddings,
            local_text_model_path=local_text_model_path,
            local_image_model_path=local_image_model_path
        )
        self.search_engine = AcademicSearchEngine()
        self.academic_roles = {}  # agent_id -> AcademicRole
        self.scenes = {}  # 所有对话场景
        self.active_scene = None  # 当前活跃场景
        self.agent_personalities = {}  # 存储智能体个性设置
        self.agents = {}  # {agent_id: agent_info}
        # 根据参数使用远程或本地 deepseek 客户端
        self.message_stream = MessageStream()
        if use_remote_llm:
            self.llm = deepseek_remote() 
        else:
            self.llm = deepseek_local(memory_type="buffer",
                                      max_token_limit=2000,
                                      window_size=10,
                                      system_prompt="You are a helpful assistant.",
                                      local_model=local_model)
        
    def create_academic_agent(self, agent_id: str, nickname: str, age: int, 
                             role_type: str, specialty: List[str], personality: str):
        """创建学术智能体，包括教授、博士生、硕士生等角色"""
        agent_info = {
            "Nickname": nickname,
            "Age": age,
            "Occupation": role_type,
            "Specialty": ", ".join(specialty),
            "Personality": personality,
            "Background": f"{role_type}，专精于{', '.join(specialty)}，{personality}"
        }
        
        # 创建学术角色
        academic_role = AcademicRole(role_type, specialty, personality)
        self.academic_roles[agent_id] = academic_role
        
        # 创建智能体并设置个性
        self.create_agent(agent_id, agent_info, academic_role.get_role_prompt())
        print(f"创建了学术智能体 {nickname}（{role_type}），专业领域：{', '.join(specialty)}")
    
    def create_agent(self, agent_id: str, agent_info: dict, personality: str = None):
        """创建一个智能体，可以指定其个性特征"""
        # 创建后端智能体
        self.backend.create_agent(agent_id, agent_info)
        # 记录个性设置
        if personality:
            self.agent_personalities[agent_id] = personality
        print(f"创建了智能体 {agent_id}（{agent_info.get('Nickname', '无名')}），个性：{personality or '未定义'}")
    
    def create_scene(self, name: str, description: str) -> str:
        """创建一个新的对话场景"""
        scene = ConversationScene(name, description)
        self.scenes[scene.id] = scene
        if self.active_scene is None:
            self.active_scene = scene.id
        
        # 将场景作为研究主题节点添加到语义图中
        topic_info = {
            "Title": name, 
            "Description": description, 
            "Keywords": description,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.backend.add_research_topic(f"topic_{scene.id}", topic_info)
        print(f"创建了研究主题：{name} - {description}")
        return scene.id
    
    def add_message(self, scene_id: str, speaker_id: str, content: str):
        """在指定场景中添加一条消息"""
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return None
            
        scene = self.scenes[scene_id]
        message_id = scene.add_message(speaker_id, content)
        
        # 添加到语义图中
        conversation_info = {
            "Content": content,
            "Speaker": speaker_id,
            "Topic": scene.name,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Participants": [speaker_id]
        }
        
        parent_keys = [speaker_id, f"topic_{scene_id}"]
        # 如果有前一条消息，也将其作为父节点
        if len(scene.messages) > 1:
            prev_msg_id = scene.messages[-2]["id"]
            parent_keys.append(prev_msg_id)
            
        self.backend.add_conversation(message_id, conversation_info, parent_keys)
        return message_id
    
    def add_attachment(self, scene_id: str, attachment_id: str, attachment_info: dict):
        """在指定场景中添加一个附件"""
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return None
            
        scene = self.scenes[scene_id]
        attachment_info["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 添加到语义图中
        self.backend.add_attachment(attachment_id, attachment_info, [f"topic_{scene_id}"])
        print(f"添加了附件 {attachment_info.get('Title', '无标题')} 到场景 {scene.name}")
    # 其余方法保持不变...
    
    def generate_event_from_conversation(self, scene_id: str = None):
        """从对话历史生成一个结论"""
        if not scene_id:
            scene_id = self.active_scene
            
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return None
            
        scene = self.scenes[scene_id]
        history = scene.get_history()
        
        if not history:
            print(f"场景 {scene_id} 没有对话历史")
            return None
            
        # 构建提示，包含场景描述和对话历史
        formatted_history = self.format_dialogue_history(history)
        
        prompt = f"""
        基于以下学术讨论历史，生成一个研究结论:

        研究主题: {scene.name}
        描述: {scene.description}

        讨论历史:
        {formatted_history}

        请提供一段简洁的学术结论，包括主要发现、共识和未来研究方向。
        """
        
        messages = [
            {"role": "system", "content": "你是一个能够从学术讨论中总结研究结论的AI。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用 LLM 生成研究结论
        conclusion_text = self.backend.llm.get_response(messages)
        
        if not conclusion_text:
            print("无法生成结论")
            return None
            
        # 创建结论
        conclusion_id = f"conclusion_{uuid.uuid4().hex[:8]}"
        participants = list(scene.participants)
        
        conclusion_info = {
            "Summary": conclusion_text,
            "Implications": "讨论结论的应用意义",
            "FutureDirections": "未来研究方向",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Topic": scene.name,
            "Participants": participants
        }
        
        # 添加结论到语义图，并与场景和参与者建立关联
        parent_keys = [f"topic_{scene_id}"] + participants
        self.backend.add_conclusion(conclusion_id, conclusion_info, parent_keys)
        
        print(f"从讨论生成的结论: {conclusion_text}")
        return conclusion_id
    
    def _generate_structured_summary(self, content: str, topic: str, summary_type: str, round_num=None) -> dict:
        """使用LLM生成结构化总结内容
        
        Args:
            content: 原始总结内容
            topic: 主题
            summary_type: 总结类型
            round_num: 轮次编号
            
        Returns:
            dict: 结构化总结内容
        """
        type_desc = "阶段性总结" if summary_type == "round" else "最终总结"
        round_text = f"第{round_num}轮" if round_num is not None else ""
        
        prompt = f"""
        请将以下关于"{topic}"的{round_text}{type_desc}内容结构化为更有条理的格式。

        原始总结:
        {content}

        请提取并组织为以下结构（以JSON格式返回）:
        1. key_findings: 主要发现和讨论要点（列出3-5点）
        2. challenges: 当前面临的挑战（列出2-3点）
        3. future_directions: 未来研究方向（列出2-3点）
        4. research_gaps: 研究空白和机会（列出1-2点）
        5. action_items: 具体行动项目（列出3-4点）

        确保内容简洁明了，每点不超过50字。返回格式应为有效的JSON。
        """
        
        messages = [
            {"role": "system", "content": "你是一位擅长总结学术讨论并提取结构化信息的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.backend.llm.get_response(messages)
            # 用于测试是否能正确返回响应
            # print(f"{response}")
            # 尝试解析JSON响应
            structured_content = json.loads(response)
            return structured_content
        except Exception as e:
            print(f"结构化总结生成失败: {str(e)}")
            # 返回默认结构
            return {
                "key_findings": "未能成功提取主要发现",
                "challenges": "未能成功提取挑战",
                "future_directions": "未能成功提取未来方向",
                "research_gaps": "未能成功提取研究空白",
                "action_items": "未能成功提取行动项目"
            }
    
    def add_summary(self, scene_id: str, speaker_id: str, content: str, 
              summary_type: str = "round", round_num: int = None, 
              related_message_ids: List[str] = None) -> str:
        """添加教授总结到语义图，优化版本添加更丰富的结构化数据
        
        Args:
            scene_id: 场景ID
            speaker_id: 发言者ID（通常是教授）
            content: 总结内容
            summary_type: 总结类型，'round'表示轮次总结，'final'表示最终总结
            round_num: 轮次编号，仅当summary_type为'round'时有效
            related_message_ids: 相关消息ID列表
            
        Returns:
            str: 总结节点ID
        """
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return None
            
        scene = self.scenes[scene_id]
        
        # 生成总结ID
        summary_id = f"summary_{uuid.uuid4().hex[:8]}"
        
        # 使用LLM生成结构化总结信息
        structured_content = self._generate_structured_summary(content, scene.name, summary_type, round_num)
        
        # 创建总结信息，包含更丰富的结构
        summary_info = {
            "Content": content,  # 原始总结内容
            "Author": speaker_id,
            "Type": summary_type,
            "Round": round_num,
            "Topic": scene.name,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 新增结构化字段
            "KeyFindings": structured_content.get("key_findings", ""),
            "Challenges": structured_content.get("challenges", ""),
            "FutureDirections": structured_content.get("future_directions", ""),
            "ResearchGaps": structured_content.get("research_gaps", ""),
            "ActionItems": structured_content.get("action_items", "")
        }
        
        if summary_type == "round" and round_num is not None:
            summary_info["Title"] = f"第{round_num}轮讨论总结"
        else:
            summary_info["Title"] = f"{scene.name}最终总结"
        
        # 确定父节点
        parent_keys = [speaker_id, f"topic_{scene_id}"]
        if related_message_ids:
            parent_keys.extend(related_message_ids)
        
        # 添加到语义图 - 使用backend的add_summary方法
        self.backend.add_summary(summary_id, summary_info, parent_keys)
        
        print(f"已添加结构化总结: {summary_info['Title']}")
        
        # 后续思考是否应该添加任务
        # # 自动从总结生成任务
        # if summary_type == "final" or (summary_type == "round" and round_num and round_num > 0):
        #     task_ids = self.generate_tasks_from_summary(summary_id)
        #     if task_ids:
        #         print(f"从总结中生成了 {len(task_ids)} 个研究任务")
        
        return summary_id
    
    def add_task(self, scene_id: str, title: str, description: str, 
               assignees: List[str] = None, due_date: str = None,
               priority: str = "中", source_id: str = None,
               metadata:str =None) -> str:
        """添加研究任务到语义图
        
        Args:
            scene_id: 场景ID
            title: 任务标题
            description: 任务描述
            assignees: 任务负责人ID列表
            due_date: 截止日期
            priority: 优先级：'高', '中', '低'
            source_id: 任务来源ID（如总结ID或讨论ID）
            
        Returns:
            str: 任务节点ID
        """
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return None
            
        scene = self.scenes[scene_id]
        
        # 生成任务ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # 创建任务信息
        task_info = {
            "Title": title,
            "Description": description,
            "Assignees": assignees or [],
            "DueDate": due_date,
            "Priority": priority,
            "Status": "待开始",
            "Topic": scene.name,
            "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata
        }
        
        # 确定父节点
        parent_keys = [f"topic_{scene_id}"]
        if source_id:
            parent_keys.append(source_id)
        if assignees:
            parent_keys.extend(assignees)
        
        # 添加到语义图
        self.backend.add_task(task_id, task_info, parent_keys)
        
        return task_id
    
    def generate_tasks_from_summary(self, summary_id: str) -> List[str]:
        """从总结内容中提取并生成任务 - 优化版本支持更细化的任务和依赖关系
        
        Args:
            summary_id: 总结节点ID
            
        Returns:
            List[str]: 生成的任务ID列表
        """
        # 获取总结节点数据
        summary_data = None
        for key, value, datatype, _ in self.backend.semantic_map.data:
            if key == summary_id and datatype == AcademicDataType.Summary:
                summary_data = value
                break
        
        if not summary_data:
            print(f"总结 {summary_id} 不存在或类型错误")
            return []
        
        # 获取总结内容和场景
        content = summary_data.get("Content", "")
        key_findings = summary_data.get("KeyFindings", "")
        future_directions = summary_data.get("FutureDirections", "")
        action_items = summary_data.get("ActionItems", "")
        topic = summary_data.get("Topic", "")
        scene_id = None
        
        # 查找对应的场景
        for scene_id, scene in self.scenes.items():
            if scene.name == topic:
                break
        
        if not scene_id:
            print(f"找不到与总结相关的场景: {topic}")
            return []
        
        # 使用结构化内容构建更好的提示
        structured_content = f"""
        主要发现与讨论要点:
        {key_findings}
        
        未来研究方向:
        {future_directions}
        
        具体行动项目:
        {action_items}
        
        原始总结内容:
        {content}
        """
        
        # 使用LLM从总结中提取任务 - 改进的提示词
        prompt = f"""
        请分析以下学术讨论总结，提取出需要执行的具体研究任务。这些任务将用于组织和推进"{topic}"的研究工作。
        注意：请只提供字符串，不要有多余的符号或者markdown格式。
        注意，所有的生成格式都要严格遵循下方提供的示例格式，不要有多余的字符。
        总结内容：
        {structured_content}
        
        请列出4-6个具体、可执行的研究任务，这些任务应：
        1. 涵盖不同研究阶段（调研、设计、实现、评估等）
        2. 适合不同学术角色（教授、博士生、硕士生）执行
        3. 有明确的时间范围和交付物
        4. 考虑任务之间的依赖关系和执行顺序
        
        每个任务必须包括的描述如下：
        1. 任务标题（简短明确）
        2. 任务描述（详细说明研究方法和预期结果）
        3. 任务优先级（高/中/低）
        4. 预计工作量（人天）
        5. 研究方法说明（具体采用什么方法完成任务）
        6. 适合的执行角色类型（教授/博士生/硕士生）
        7. 依赖任务（如有，写任务编号；无依赖则写"无"）
        
        下面是参考格式，
        请严格按以下格式返回：
        任务1：[标题]
        描述：[详细描述]
        优先级：[优先级]
        工作量：[预计人天]
        研究方法：[方法说明]
        适合角色：[角色类型]
        依赖任务：[依赖的任务编号，如无则填"无"]
        
        任务2：[标题]
        描述：[详细描述]
        优先级：[优先级]
        工作量：[预计人天]
        研究方法：[方法说明]
        适合角色：[角色类型]
        依赖任务：[依赖的任务编号，如无则填"无"]

        请确保任务具体可行、涵盖不同研究阶段，并考虑任务间的依赖关系。
        """
        
        messages = [
            {"role": "system", "content": "你是一个擅长从学术讨论中提取研究任务并进行任务规划的助手，具有丰富的学术研究和项目管理经验。"},
            {"role": "user", "content": prompt}
        ]
        
        tasks_text = self.backend.llm.get_response(messages)
        if not tasks_text:
            print("无法从总结中提取任务")
            return []
        print(f"{tasks_text}")
        # 解析LLM返回的任务文本 - 改进解析逻辑，支持更多属性
        # # 修改前
        # task_sections = re.split(r'任务\d+：', tasks_text)

        #!!!
        #!!!主要bug点，大模型的生成结构不一定可以正常解析，需要密切观察
        #!!!

        # 修改后 - 支持多种格式的任务标记
        task_sections = re.split(r'(?:###\s*)?任务\d+[：:]', tasks_text)
        task_sections = [section for section in task_sections if section.strip()]
        
        created_task_ids = []
        task_dependency_map = {}  # 存储任务依赖关系
        
        # 先创建所有任务，记录ID
        for i, section in enumerate(task_sections):
            lines = section.strip().split('\n')
            
            # 初始化任务属性
            title = lines[0].strip() if lines else ""
            description = ""
            priority = "中"  # 默认中优先级
            workload = "5"  # 默认工作量
            methods = ""
            role_type = "博士生"  # 默认角色
            dependencies = "无"
            
            # 解析各行内容
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("描述："):
                    description = line[3:].strip()
                elif line.startswith("优先级："):
                    priority = line[4:].strip()
                elif line.startswith("工作量："):
                    workload = line[4:].strip()
                elif line.startswith("研究方法："):
                    methods = line[5:].strip()
                elif line.startswith("适合角色："):
                    role_type = line[5:].strip()
                elif line.startswith("依赖任务："):
                    dependencies = line[5:].strip()
            
            # 如果成功解析出标题和描述，则创建任务
            if title and description:
                # 丰富任务描述
                full_description = f"{description}\n\n研究方法：{methods}\n预计工作量：{workload}人天"
                
                # 根据角色类型选择合适的执行者
                assignees = self._find_suitable_assignees(role_type)
                if assignees and len(assignees) > 0:
                    # 从匹配角色中选一个，并确保任务均匀分配
                    assignee = [assignees[i % len(assignees)]]
                else:
                    # 如果没有找到匹配角色，随机选择
                    assignee = self._find_random_assignees(1)
                
                # 创建任务
                task_id = self.add_task(
                    scene_id=scene_id,
                    title=title,
                    description=full_description,
                    assignees=assignee,
                    priority=priority,
                    source_id=summary_id,
                    metadata={
                        "workload": workload,
                        "methods": methods,
                        "role_type": role_type,
                        "raw_dependencies": dependencies,
                        "task_index": i + 1
                    }
                )
                
                if task_id:
                    created_task_ids.append(task_id)
                    # 记录任务索引与ID的映射关系，用于后续处理依赖
                    task_dependency_map[str(i + 1)] = task_id
                    print(f"从总结中创建任务 '{title}' (执行者: {assignee})")
        
        # 处理任务依赖关系
        for task_id in created_task_ids:
            # 获取任务数据
            task_data = None
            for key, value, datatype, _ in self.backend.semantic_map.data:
                if key == task_id and datatype == AcademicDataType.Task:
                    task_data = value
                    break
            
            if task_data and "metadata" in task_data:
                # 获取原始依赖描述
                raw_deps = task_data["metadata"].get("raw_dependencies", "无")
                if raw_deps and raw_deps != "无":
                    # 解析依赖任务索引
                    dep_indices = re.findall(r'\d+', raw_deps)
                    dep_task_ids = []
                    
                    # 添加依赖关系到语义图
                    for dep_idx in dep_indices:
                        if dep_idx in task_dependency_map:
                            dep_task_id = task_dependency_map[dep_idx]
                            dep_task_ids.append(dep_task_id)
                            # 在语义图中添加任务间依赖关系
                            self.backend.semantic_graph.add_edge(dep_task_id, task_id, "依赖于")
                    
                    # 更新任务数据中的依赖关系
                    task_data["Dependencies"] = dep_task_ids
                    # 更新语义图中的节点
                    for key, value, datatype, embedding in self.backend.semantic_map.data:
                        if key == task_id and datatype == AcademicDataType.Task:
                            self.backend.semantic_map.data.remove((key, value, datatype, embedding))
                            self.backend.semantic_map.insert(key, task_data, datatype)
                            break
        
        # 打印任务依赖关系图
        if created_task_ids:
            print("\n任务依赖关系:")
            for task_id in created_task_ids:
                task_data = None
                for key, value, datatype, _ in self.backend.semantic_map.data:
                    if key == task_id and datatype == AcademicDataType.Task:
                        task_data = value
                        break
                
                if task_data:
                    title = task_data.get("Title", "Unknown")
                    deps = task_data.get("Dependencies", [])
                    if deps:
                        dep_titles = []
                        for dep_id in deps:
                            for key, value, datatype, _ in self.backend.semantic_map.data:
                                if key == dep_id and datatype == AcademicDataType.Task:
                                    dep_titles.append(value.get("Title", "Unknown"))
                                    break
                        print(f"  {title} 依赖于: {', '.join(dep_titles)}")
                    else:
                        print(f"  {title}: 无依赖")
        
        return created_task_ids

    def _find_suitable_assignees(self, role_type: str) -> List[str]:
        """根据角色类型找到合适的任务执行者
        
        Args:
            role_type: 角色类型描述
            
        Returns:
            List[str]: 符合角色的智能体ID列表
        """
        suitable_agents = []
        
        for agent_id, agent_info in self.backend.agents.items():
            occupation = agent_info.get("Occupation", "").lower()
            
            if "教授" in role_type.lower() or "professor" in role_type.lower():
                if "教授" in occupation or "professor" in occupation:
                    suitable_agents.append(agent_id)
            elif "博士" in role_type.lower() or "phd" in role_type.lower():
                if "博士" in occupation or "phd" in occupation:
                    suitable_agents.append(agent_id)
            elif "硕士" in role_type.lower() or "master" in role_type.lower() or "msc" in role_type.lower():
                if "硕士" in occupation or "master" in occupation or "msc" in occupation:
                    suitable_agents.append(agent_id)
        
        return suitable_agents

    def _find_random_assignees(self, count: int = 1) -> List[str]:
        """随机选择指定数量的智能体作为任务执行者
        
        Args:
            count: 需要的执行者数量
            
        Returns:
            List[str]: 随机选择的智能体ID列表
        """
        all_agents = list(self.backend.agents.keys())
        if not all_agents:
            return []
        
        # 确保不超过可用的智能体数量
        count = min(count, len(all_agents))
        
        # 随机选择智能体
        return random.sample(all_agents, count)

    def visualize_conversation_graph(self, scene_id: str = None, fontpath=None, output_filename=None):
        """使用AcademicGroupMeetingGraph的可视化功能
        
        Args:
            scene_id: 场景ID，None则使用当前活跃场景
            fontpath: 字体路径，用于解决中文显示问题
            output_filename: 输出文件名，如果为None则使用默认名称
        """
        if not scene_id:
            scene_id = self.active_scene
                
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return
            
        # 确保在生成图形前所有节点已正确添加到图中
        scene = self.scenes[scene_id]
        
        # 重新构建关系图以确保数据完整性
        topic_node = f"topic_{scene_id}"
        
        # 确保添加所有消息关系
        prev_msg_id = None
        for msg in scene.messages:
            msg_id = msg["id"]
            speaker_id = msg["speaker_id"]
            
            # 确保消息节点存在且连接了适当的父节点
            parent_keys = [speaker_id, topic_node]
            if prev_msg_id:
                parent_keys.append(prev_msg_id)
                
            # 添加到图中（如果已存在不会重复添加）
            conversation_info = {
                "Content": msg["content"],
                "Speaker": speaker_id,
                "Topic": scene.name,
                "Timestamp": msg["timestamp"],
                "Participants": [speaker_id]
            }
            
            # 检查节点是否已存在
            self.backend.semantic_graph._ensure_node(msg_id)
            
            # 重新构建连接
            for parent_key in parent_keys:
                if parent_key in self.backend.semantic_graph.graph_relations:
                    self.backend.semantic_graph._ensure_node(parent_key)
                    self.backend.semantic_graph.graph_relations[parent_key]["children"][msg_id] = "发言"
                    self.backend.semantic_graph.graph_relations[msg_id]["parents"][parent_key] = "属于"
            
            prev_msg_id = msg_id
        
        # 更新子图结构
        self.backend.semantic_graph.auto_generate_subgraphs()
        
        # 如果未指定输出文件名，则使用默认名称
        if output_filename is None:
            output_filename = os.path.join(PICTURE_DIR, f"academic_meeting_{scene_id}.png")
        
        # 使用学术组会专用的可视化方法，传递字体参数
        if fontpath:
            self.backend.visualize_graph(
                filename=output_filename, 
                fontpath=fontpath 
                # else '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            )
        else:
            self.backend.visualize_graph(
                filename=output_filename
            )
        
        print(f"学术组会图已保存为 {output_filename}")

    # 在AcademicMeetingSystem类中添加以下方法

    def format_dialogue_history(self, history: List[Dict[str, Any]]) -> str:
        """格式化对话历史，用于创建提示"""
        formatted = []
        for msg in history:
            speaker_id = msg["speaker_id"]
            speaker_name = self.backend.agents.get(speaker_id, {}).get("Nickname", speaker_id)
            formatted.append(f"{speaker_name}: {msg['content']}")
        return "\n".join(formatted)

    def agent_search_and_reply(self, scene_id: str, speaker_id: str, 
                            search_query: str, context_size: int = 5) -> str:
        """让智能体进行网络检索并基于检索结果回复"""
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return None
        
        scene = self.scenes[scene_id]
        history = scene.get_history(context_size)
        
        if not history:
            print(f"场景 {scene_id} 没有对话历史")
            return None
        
        # 获取说话者信息与学术角色
        speaker = self.backend.agents.get(speaker_id, {}).get("Nickname", speaker_id)
        academic_role = self.academic_roles.get(speaker_id)
        
        # 进行网络检索
        search_results = self.search_engine.search(search_query)
        
        # 格式化搜索结果
        formatted_results = ""
        if "organic_results" in search_results:
            for i, result in enumerate(search_results["organic_results"][:3], 1):
                formatted_results += f"{i}. {result.get('title', 'No Title')}\n"
                formatted_results += f"   摘要: {result.get('snippet', 'No snippet available')}\n"
                formatted_results += f"   链接: {result.get('link', 'No link')}\n\n"
        else:
            formatted_results = "未找到相关搜索结果。\n"
        
        # 构建提示，包含对话历史、学术角色信息和检索结果
        formatted_history = self.format_dialogue_history(history)
        
        prompt = f"""这是一段学术讨论的对话历史:
        {formatted_history}

        以下是关于"{search_query}"的检索结果:
        {formatted_results}

        现在请你以{speaker}的身份回复，基于以上检索结果参与学术讨论。
        """
        
        if academic_role:
            prompt += f"\n你是一名{academic_role.get_role_prompt()}"
        
        # 包装成消息列表
        messages = [
            {"role": "system", "content": "你是一个学术助手，能够模拟不同学术角色参与专业讨论。请基于检索结果提供专业、有深度的回复。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用 LLM 生成回复
        response = self.backend.llm.get_response(messages)
        
        if not response:
            return "无法生成回复。"
            
        # 添加到场景历史和语义图
        self.add_message(scene_id, speaker_id, response)
        
        return response

    def start_academic_meeting(self, scene_id: str, topic: str, 
                          moderator_id: str, rounds: int = 3, deep_search: bool = False):
        """启动学术组会讨论，由主持人引导讨论流程
        
        Args:
            scene_id: 场景ID
            topic: 讨论主题
            moderator_id: 主持人ID
            rounds: 讨论轮次
            deep_search: 是否使用深度搜索
        """
        if scene_id not in self.scenes:
            print(f"场景 {scene_id} 不存在")
            return
            
        scene = self.scenes[scene_id]
        
        # 确保主持人存在，如果不存在则使用第一个参与者
        if moderator_id not in self.backend.agents:
            if len(self.backend.agents) > 0:
                moderator_id = list(self.backend.agents.keys())[0]
                print(f"主持人不存在，使用 {moderator_id} 作为主持人")
            else:
                print("没有可用的智能体作为主持人")
                return
                
        moderator_name = self.backend.agents.get(moderator_id, {}).get("Nickname", moderator_id)
        
        print(f"\n--- 开始学术组会（主题：{topic}，主持人：{moderator_name}）---\n")
        
        # 主持人开场白
        opening_prompt = f"""
        作为{moderator_name}，你是这次学术组会的主持人。请提供一个组会开场白，介绍今天讨论的主题:
        主题: {topic}
        请简明扼要地介绍主题的背景、重要性，并鼓励大家积极参与讨论。
        """
        
        messages = [
            {"role": "system", "content": "你是一名学术组会的主持人，负责引导讨论。"},
            {"role": "user", "content": opening_prompt}
        ]
        
        opening = self.backend.llm.get_response(messages)
        if opening:
            self.add_message(scene_id, moderator_id, opening)
            print(f"{moderator_name}: {opening}\n")
        
        # 添加所有组会参与者到场景
        for agent_id in self.backend.agents:
            if agent_id not in scene.participants:
                # 让每个人简单介绍一下自己
                agent_name = self.backend.agents.get(agent_id, {}).get("Nickname", agent_id)
                intro_prompt = f"""
                作为{agent_name}，请简短介绍一下你自己以及你对"{topic}"的初步看法。
                不要超过3句话。
                """
                messages = [
                    {"role": "system", "content": "你是参加学术讨论的成员，需要简短介绍自己。"},
                    {"role": "user", "content": intro_prompt}
                ]
                intro = self.backend.llm.get_response(messages)
                if intro:
                    self.add_message(scene_id, agent_id, intro)
                    print(f"{agent_name}: {intro}\n")
        
        # 进行讨论轮次
        for round_num in range(1, rounds + 1):
            print(f"\n--- 第 {round_num} 轮讨论 ---\n")
            
            # 针对当前轮次的讨论点
            if round_num == 1:
                discussion_point = f"{topic}的研究现状和挑战"
            elif round_num == 2:
                discussion_point = f"解决{topic}问题的方法和技术"
            else:
                discussion_point = f"{topic}的未来发展方向和应用前景"
                
            # 主持人引导当前轮次讨论
            guide_prompt = f"""
            作为{moderator_name}，请引导组会进入第{round_num}轮讨论。
            当前讨论点: {discussion_point}
            请简要介绍这个讨论点，并邀请组内成员发表看法。
            """
            
            messages = [
                {"role": "system", "content": "你是一名学术组会的主持人，负责引导讨论。"},
                {"role": "user", "content": guide_prompt}
            ]
            
            guide = self.backend.llm.get_response(messages)
            if guide:
                self.add_message(scene_id, moderator_id, guide)
                print(f"{moderator_name}: {guide}\n")
            
            # 其他参与者依次发言，并进行网络检索
        for agent_id in self.backend.agents:
            if agent_id != moderator_id:
                agent_name = self.backend.agents.get(agent_id, {}).get("Nickname", agent_id)
                print(f"\n{agent_name}正在查询相关资料..." + 
                    ("(使用深度网页内容)" if deep_search else ""))
                
                # 根据参与者角色和当前讨论点生成检索关键词
                academic_role = self.academic_roles.get(agent_id)
                if academic_role and academic_role.specialty:
                    specialty = academic_role.specialty[0]  # 使用第一个专业领域
                    search_query = f"{topic} {discussion_point} {specialty}"
                else:
                    search_query = f"{topic} {discussion_point}"
                    
                # 使用改进的搜索功能
                if deep_search:
                    # 使用网页内容增强的搜索
                    search_result = self.search_engine.search_and_analyze(
                        search_query, 
                        include_full_content=True
                    )
                    
                    # 提取分析结果
                    if "analysis" in search_result:
                        analysis = search_result["analysis"]
                        # 基于分析结果生成发言
                        content_prompt = f"""
                        作为{agent_name}，以下是你搜索到的关于"{search_query}"的研究分析:
                        
                        {analysis}
                        
                        请基于此分析，以你的专业角色发表学术见解。
                        """
                        
                        messages = [
                            {"role": "system", "content": "你是一名学术会议的参与者，基于搜索分析结果发表专业见解。"},
                            {"role": "user", "content": content_prompt}
                        ]
                        
                        response = self.backend.llm.get_response(messages)
                        if response:
                            self.add_message(scene_id, agent_id, response)
                            print(f"{agent_name}: {response}\n")
                else:
                    # 使用原有的检索方法
                    response = self.agent_search_and_reply(scene_id, agent_id, search_query)
                    print(f"{agent_name}: {response}\n")
            
            # 主持人小结本轮讨论
            if round_num < rounds:
                summary_prompt = f"""
                作为{moderator_name}，请对第{round_num}轮关于"{discussion_point}"的讨论进行小结，
                并自然地过渡到下一轮讨论。
                """
                
                messages = [
                    {"role": "system", "content": "你是一名学术组会的主持人，负责引导和总结讨论。"},
                    {"role": "user", "content": summary_prompt}
                ]
                
                summary = self.backend.llm.get_response(messages)
                if summary:
                    self.add_message(scene_id, moderator_id, summary)
                    print(f"{moderator_name}: {summary}\n")
        
        # 主持人总结组会
        conclusion_prompt = f"""
        作为{moderator_name}，请对整个关于"{topic}"的学术讨论进行总结。
        包括讨论的主要内容、达成的共识、未来研究方向等。
        最后，感谢大家的参与并结束组会。
        """
        
        messages = [
            {"role": "system", "content": "你是一名学术组会的主持人，负责总结讨论成果。"},
            {"role": "user", "content": conclusion_prompt}
        ]
        
        conclusion = self.backend.llm.get_response(messages)
        if conclusion:
            self.add_message(scene_id, moderator_id, conclusion)
            print(f"{moderator_name}: {conclusion}\n")
            
        print("\n--- 学术组会结束 ---\n")
        
        # 生成组会总结
        self.generate_event_from_conversation(scene_id)

    def run_custom_meeting(self, topic: str, professor_id: str = None, rounds: int = 3, deep_search: bool = False):
        """执行用户自定义主题的学术组会"""
        # 1. 创建场景
        scene_name = f"{topic}学术研讨会"
        scene_description = f"讨论主题：{topic}，探讨研究现状、方法技术与未来方向"
        scene_id = self.create_scene(scene_name, scene_description)
        
        # 处理本地附件文件
        print("\n处理本地附件...")
        local_attachments = self.process_local_attachments(topic_key=f"topic_{scene_id}")
        if local_attachments["pdfs"] or local_attachments["images"]:
            print(f"处理了 {len(local_attachments['pdfs'])} 篇本地论文和 {len(local_attachments['images'])} 张本地图片")

        # 搜索相关附件
        print("\n搜索与主题相关的论文和图片...")
        attachments = self.search_and_add_attachments(
            topic=topic, 
            topic_key=f"topic_{scene_id}", 
            download_files=True
        )
        
        if attachments["papers"]:
            # 可以在讨论中引用搜索到的论文
            print(f"搜索到 {len(attachments['papers'])} 篇相关论文，可在讨论中引用")

        # 2. 识别教授角色和学生角色
        professors = []
        students = []
        
        for agent_id, agent_info in self.backend.agents.items():
            role = agent_info.get("Occupation", "").lower()
            if "教授" in role or "professor" in role:
                professors.append(agent_id)
            else:
                students.append(agent_id)
        
        # 如果未指定教授或指定的教授不存在
        if not professor_id or professor_id not in professors:
            if professors:
                professor_id = professors[0]
            else:
                print("错误：未找到教授角色，无法进行学术组会")
                return None
        
        professor_name = self.backend.agents[professor_id]["Nickname"]
        print(f"\n=== 开始自定义学术组会：{topic} ===\n")
        print(f"主持人：{professor_name}")
        
        # 3. 教授搜索背景资料并介绍主题
        print(f"\n{professor_name}正在搜索主题背景资料...")
        search_results = self.search_engine.search(f"{topic} research background latest developments")
        
        # 提取搜索结果
        formatted_results = self._format_search_results(search_results)
        
        # 教授基于搜索结果介绍主题
        intro_prompt = f"""
        作为{professor_name}，你是一位经验丰富的教授，需要为学术组会开场并介绍研究主题。
        
        研究主题: {topic}
        
        以下是关于该主题的最新研究背景资料:
        {formatted_results}
        
        请提供一个全面的主题介绍，包括:
        1. 该研究领域的背景和重要性
        2. 目前研究现状和主要挑战
        3. 为什么这个主题值得讨论和研究
        4. 本次组会希望讨论的主要问题
        
        以学术演讲的风格展开，富有洞见但不过于冗长。
        """
        
        messages = [
            {"role": "system", "content": "你是一位资深学者，正在主持一场学术讨论。"},
            {"role": "user", "content": intro_prompt}
        ]
        
        introduction = self.backend.llm.get_response(messages)
        if introduction:
            self.add_message(scene_id, professor_id, introduction)

            ###
            print(f"\n{professor_name}:\n{introduction}\n")
            self.message_stream.agent_message(professor_id, professor_name, introduction)
        
        # 4. 按轮次讨论
        for round_num in range(1, rounds + 1):
            print(f"\n--- 第 {round_num} 轮讨论 ---\n")
            self.message_stream.round_info(round_num=round_num, topic=topic)
            
            # 确定本轮讨论的焦点
            if round_num == 1:
                focus = f"{topic}的研究现状分析"
                search_suffix = "current research status challenges"
            elif round_num == 2:
                focus = f"{topic}的研究方法与技术"
                search_suffix = "research methods techniques implementation"
            else:
                focus = f"{topic}的应用前景与未来方向"
                search_suffix = "future directions applications potential impact"
            
            # 教授引导本轮讨论
            guide_prompt = f"""
            作为{professor_name}，你需要引导学术组会进入第{round_num}轮讨论。
            
            本轮讨论主题: {focus}
            
            请简要介绍这个讨论点，提出1-2个思考问题，并邀请同学们发表见解。
            请保持简洁，控制在150字以内。
            """
            
            messages = [
                {"role": "system", "content": "你是一位引导学术讨论的教授。"},
                {"role": "user", "content": guide_prompt}
            ]
            
            guidance = self.backend.llm.get_response(messages)
            if guidance:
                self.add_message(scene_id, professor_id, guidance)
                print(f"{professor_name}:\n{guidance}\n")
            
            # 学生依次发言 - 确保每个轮次中所有学生都能发言
            for student_id in students:
                student_name = self.backend.agents[student_id]["Nickname"]
                specialty = self.backend.agents[student_id].get("Specialty", "")
                print(f"\n{student_name}正在搜索相关资料..." + 
                    ("(使用深度网页内容)" if deep_search else ""))
                
                # 学生搜索信息
                search_query = f"{topic} {search_suffix} {specialty}"
                
                student_results = ""
                if deep_search:
                    try:
                        # 使用深度搜索并分析
                        search_analysis = self.search_engine.search_and_analyze(
                            search_query,
                            include_full_content=True
                        )
                        
                        if "analysis" in search_analysis:
                            student_results = search_analysis["analysis"]
                            print(f"获取了深度分析结果 ({len(student_results)} 字符)")
                        else:
                            # 回退到普通搜索
                            student_search = self.search_engine.search(search_query)
                            student_results = self._format_search_results(student_search)
                            print("深度分析失败，回退到普通搜索")
                    except Exception as e:
                        print(f"深度搜索出错: {str(e)}，回退到普通搜索")
                        student_search = self.search_engine.search(search_query)
                        student_results = self._format_search_results(student_search)
                else:
                    # 普通搜索
                    student_search = self.search_engine.search(search_query)
                    student_results = self._format_search_results(student_search)
                    
                # 获取学术角色信息
                academic_role = self.academic_roles.get(student_id)
                role_info = academic_role.get_role_prompt() if academic_role else ""
                
                # 学生发言提示
                student_prompt = f"""
                作为{student_name}，你是一名{role_info}。

                当前讨论的主题是: {topic}
                本轮讨论的焦点是: {focus}
                
                以下是你搜索到的资料:
                {student_results}
                
                请根据你的专业背景和搜索结果，对当前讨论点发表你的见解，可以包括:
                1. 对研究现状或方法的评价
                2. 相关领域的经验和观点
                3. 提出问题或解决方案
                4. 与其他参与者观点的连接或讨论
                
                以学术研讨的口吻发言，专业但不晦涩，观点鲜明有深度。
                """
                
                messages = [
                    {"role": "system", "content": "你是参与学术讨论的研究生，有自己的专业见解。"},
                    {"role": "user", "content": student_prompt}
                ]
                
                response = self.backend.llm.get_response(messages)
                if response:
                    self.add_message(scene_id, student_id, response)
                    print(f"{student_name}:\n{response}\n")
                    
                    # 教授简短回应每个学生（非必需，但更真实）
                    if round_num < rounds or students.index(student_id) < len(students) - 1:
                        feedback_prompt = f"""
                        作为{professor_name}，请对{student_name}的发言给予简短的回应和点评。
                        
                        {student_name}的发言:
                        {response}
                        
                        请给予学术性的建设性反馈，提出1-2个思考点或补充，控制在100字以内。
                        """
                        
                        messages = [
                            {"role": "system", "content": "你是指导学生讨论的教授，需要给予建设性的简短反馈。"},
                            {"role": "user", "content": feedback_prompt}
                        ]
                        
                        feedback = self.backend.llm.get_response(messages)
                        if feedback:
                            self.add_message(scene_id, professor_id, feedback)
                            print(f"{professor_name}:\n{feedback}\n")
            
            # 轮次结束，教授总结本轮（除最后一轮）
            if round_num < rounds:
                round_summary_prompt = f"""
                作为{professor_name}，请对第{round_num}轮关于"{focus}"的讨论进行小结，并自然地引导到下一轮讨论。
                请控制在150字左右，保持简洁明了。
                """
                
                messages = [
                    {"role": "system", "content": "你是总结学术讨论的教授。"},
                    {"role": "user", "content": round_summary_prompt}
                ]
                
                round_summary = self.backend.llm.get_response(messages)
                if round_summary:
                    # 添加常规对话消息
                    self.add_message(scene_id, professor_id, round_summary)
                    print(f"{professor_name} (本轮总结):\n{round_summary}\n")
                    
                    # 添加到summary空间作为轮次总结
                    round_message_ids = []
                    # 获取本轮的最后几条消息作为相关消息
                    for msg in self.scenes[scene_id].messages[-5:]:
                        round_message_ids.append(msg["id"])
                    
                    summary_id = self.add_summary(
                        scene_id=scene_id,
                        speaker_id=professor_id,
                        content=round_summary,
                        summary_type="round",
                        round_num=round_num,
                        related_message_ids=round_message_ids
                    )
                    
                    # 从总结中提取研究任务
                    if summary_id:
                        self.generate_tasks_from_summary(summary_id)
        
        # 5. 教授汇总整个讨论并提出解决方案
        print("\n--- 教授总结与方案提出 ---\n")
        
        # 获取历史讨论内容
        history = self.scenes[scene_id].get_history()
        formatted_history = self.format_dialogue_history(history[-20:])  # 最后20条消息
        
        # 搜索解决方案和最佳实践
        solution_search = self.search_engine.search(f"{topic} solutions best practices implementation")
        solution_results = self._format_search_results(solution_search)
        
        summary_prompt = f"""
        作为{professor_name}，请对整个关于"{topic}"的学术讨论进行全面总结，并提出具体解决方案。
        
        讨论历史摘要:
        {formatted_history}
        
        关于解决方案的搜索结果:
        {solution_results}
        
        请提供:
        1. 对整个讨论的系统性总结，包括各方观点和达成的共识
        2. 明确的研究路径或解决方案，分步骤阐述
        3. 潜在的应用场景和实现方法
        4. 未来研究方向的具体建议
        
        以资深学者的专业视角，提出既有理论深度又有实践指导意义的方案。
        """
        
        messages = [
            {"role": "system", "content": "你是一位资深教授，正在为学术组会提供最终总结和解决方案。"},
            {"role": "user", "content": summary_prompt}
        ]
        
        final_summary = self.backend.llm.get_response(messages)
        if final_summary:
            # 添加常规对话消息
            self.add_message(scene_id, professor_id, final_summary)
            print(f"{professor_name}:\n{final_summary}\n")
            
            # 添加到summary空间作为最终总结
            all_message_ids = []
            for msg in self.scenes[scene_id].messages[-10:]:  # 取最后10条消息作为相关消息
                all_message_ids.append(msg["id"])
                
            final_summary_id = self.add_summary(
                scene_id=scene_id,
                speaker_id=professor_id,
                content=final_summary,
                summary_type="final",
                related_message_ids=all_message_ids
            )
            
            # 从最终总结中提取研究任务
            if final_summary_id:
                task_ids = self.generate_tasks_from_summary(final_summary_id)
                print(f"从最终总结中创建了 {len(task_ids)} 个研究任务")
        
        # 6. 生成会议结论
        conclusion = self.generate_event_from_conversation(scene_id)
        
        print("\n=== 自定义学术组会结束 ===\n")
        
        # 7. 可视化对话图
        try:
            self.visualize_conversation_graph(scene_id)
        except Exception as e:
            print(f"无法生成可视化: {str(e)}")
        
        return scene_id

    def _format_search_results(self, search_results):
        """格式化Google搜索结果为易读文本"""
        formatted_text = ""
        
        if not search_results:
            return "未找到相关搜索结果。"
        
        if "organic_results" in search_results and search_results["organic_results"]:
            formatted_text += "搜索结果:\n\n"
            for i, result in enumerate(search_results["organic_results"][:3], 1):
                title = result.get("title", "无标题")
                snippet = result.get("snippet", "无摘要")
                link = result.get("link", "无链接")
                
                formatted_text += f"{i}. {title}\n"
                formatted_text += f"   摘要: {snippet}\n"
                formatted_text += f"   来源: {link}\n\n"
        
        # 处理知识图谱结果(如果有)
        if "knowledge_graph" in search_results:
            kg = search_results["knowledge_graph"]
            if "title" in kg:
                formatted_text += f"知识图谱: {kg.get('title')}\n"
            if "description" in kg:
                formatted_text += f"描述: {kg.get('description')}\n\n"
        
        if not formatted_text:
            formatted_text = "未能提取有效的搜索结果。"
        
        return formatted_text
    
    def search_and_add_attachments(self, topic: str, topic_key: str = None, download_files: bool = True):
        """搜索与主题相关的附件并添加到语义图
        
        Args:
            topic: 搜索主题
            topic_key: 主题节点ID，为None则使用当前场景的主题
            download_files: 是否下载文件到本地
            
        Returns:
            Dict: 包含添加的附件信息
        """
        # 如果未指定topic_key，则使用当前激活场景的主题
        if not topic_key and self.active_scene:
            scene_id = self.active_scene
            topic_key = f"topic_{scene_id}"
        
        # 初始化附件检索引擎
        attachment_engine = AttachmentSearchEngine()
        
        # 执行搜索并添加到语义图
        results = attachment_engine.search_and_add_attachments(
            backend=self.backend,
            topic=topic,
            topic_key=topic_key,
            download_files=download_files
        )
        
        # 如果添加了任何附件，重新生成图表
        if results["papers"] or results["images"]:
            # 更新语义图
            self.backend.semantic_graph.auto_generate_subgraphs()
            
            # 可选，在搜索完成后生成新的可视化图
            if self.active_scene:
                self.visualize_conversation_graph(self.active_scene)
        
        return results
    
    def process_local_attachments(self, topic_key: str = None):
        """处理本地附件目录中的文件并添加到语义图
        
        Args:
            topic_key: 主题节点ID，为None则使用当前场景的主题
            
        Returns:
            Dict: 包含添加的附件信息
        """
        # 如果未指定topic_key，则使用当前激活场景的主题
        if not topic_key and self.active_scene:
            scene_id = self.active_scene
            topic_key = f"topic_{scene_id}"
        
        # 初始化本地附件处理器
        attachment_processor = LocalAttachmentProcessor()
        
        # 执行处理并添加到语义图
        results = attachment_processor.process_all_attachments(
            backend=self.backend,
            topic_key=topic_key
        )
        
        # 如果添加了任何附件，重新生成图表
        if results["pdfs"] or results["images"]:
            # 更新语义图
            self.backend.semantic_graph.auto_generate_subgraphs()
            
            # 可选，在处理完成后生成新的可视化图
            if self.active_scene:
                self.visualize_conversation_graph(self.active_scene)
        
        return results
    
    def generate_comprehensive_review(self, topic: str, professor_id: str = None, 
                               subtopics: List[str] = None, rounds: int = 3, 
                               include_literature: bool = True, 
                               include_open_source: bool = True,
                               output_dir: str = None):
        """生成智能体相关主题的综述报告
        
        Args:
            topic: 主题，如"面向智能体的记忆管理系统"
            professor_id: 主持讨论的教授ID
            subtopics: 子话题列表，如为None则自动生成
            rounds: 讨论轮次，影响讨论深度和广度，最少为1
            include_literature: 是否包含文献引用
            include_open_source: 是否包含开源系统
            output_dir: 输出目录，如果为None则使用默认值
            
        Returns:
            dict: 包含综述报告及其元数据的字典
        """
        # 规范化参数
        rounds = max(1, rounds)  # 确保至少有1轮讨论
        
        # 任务管理结构：学生ID -> 任务列表[(轮次, 任务描述, 状态)]
        # 添加统一的任务管理数据结构
        task_manager = {
            'all_tasks': [],  # 存储所有任务，格式：(轮次, 学生ID, 任务描述, 状态)
            'by_student': {},  # 按学生ID组织，格式：{student_id: [(轮次, 任务描述, 状态)]}
            'by_round': {}     # 按轮次组织，格式：{round_num: [(学生ID, 任务描述, 状态)]}
        }
        
        # 1. 创建场景
        scene_name = f"{topic}综述研讨会"
        scene_description = f"讨论主题：{topic}，撰写综合性学术综述报告"
        scene_id = self.create_scene(scene_name, scene_description)
        
        # 2. 识别教授和学生角色
        professors, students = self._identify_professors_and_students()
        
        # 如果未指定教授或指定的教授不存在，选择一个可用的教授
        professor_id = self._ensure_valid_professor(professor_id, professors)
        if not professor_id:
            self.message_stream.system_message("错误：未找到教授角色，无法生成综述报告")
            return None
        
        professor_name = self.backend.agents[professor_id]["Nickname"]
        
        # 发送初始消息
        self._send_start_messages(topic, professor_name)
        
        # 3. 处理本地附件和搜索相关资料
        papers = self._process_attachments(topic, scene_id)
        
        # 4. 搜索开源系统
        open_source_systems = self._search_open_source_systems(topic, include_open_source)
        
        # 5. 准备子话题
        subtopics, effective_rounds = self._prepare_subtopics(topic, subtopics, rounds)
        round_topics = [[subtopic] for subtopic in subtopics[:effective_rounds]]
        
        # 6. 教授介绍综述主题和写作计划
        introduction = self._professor_introduces_topic(scene_id, professor_id, professor_name, topic, subtopics, effective_rounds)
        
        # 7. 初始学生反馈与问题
        subtopic_discussion = self._get_student_initial_feedback(scene_id, students, professor_name, topic, subtopics, effective_rounds)
        
        # 8. 教授回应与框架调整
        self._professor_adjusts_framework(
            scene_id, professor_id, professor_name, topic, students, task_manager
        )
        
        # 9. 正式讨论轮次 - 深入每个子话题
        subsection_contents, literature_by_topic = self._conduct_discussion_rounds(
            scene_id, topic, subtopics, effective_rounds, professor_id, professor_name, 
            students, include_literature, papers, task_manager
        )
        
        # 10. 生成最终综述报告
        review_result = self._generate_final_review(
            scene_id, topic, professor_id, professor_name, subtopics, effective_rounds,
            subsection_contents, literature_by_topic, open_source_systems,
            all_literature=self._merge_literature(literature_by_topic),
            output_dir=output_dir
        )
        
        # 在函数结尾添加研究任务生成
        if review_result and "error" not in review_result:
            self._generate_research_tasks_from_review(topic, professor_id, review_result)
        
        self.message_stream.system_message(f"综述报告生成完成！")
        return review_result

    def _identify_professors_and_students(self):
        """识别系统中的教授和学生角色"""
        professors = []
        students = []
        
        for agent_id, agent_info in self.backend.agents.items():
            role = agent_info.get("Occupation", "").lower()
            if "教授" in role or "professor" in role:
                professors.append(agent_id)
            else:
                students.append(agent_id)
        
        return professors, students

    def _ensure_valid_professor(self, professor_id, professors):
        """确保使用有效的教授ID"""
        if not professor_id or professor_id not in professors:
            if professors:
                return professors[0]
            else:
                print("错误：未找到教授角色，无法生成综述报告")
                return None
        return professor_id

    def _send_start_messages(self, topic, professor_name):
        """发送开始会议的消息"""
        print(f"\n=== 开始{topic}综述报告研讨会 ===\n")
        self.message_stream.system_message(f"开始{topic}综述报告研讨会")
        self.message_stream.system_message(f"主持人：{professor_name}")

    def _process_attachments(self, topic, scene_id):
        """处理本地附件和搜索相关资料"""
        self.message_stream.system_message("处理本地附件与搜索相关文献...")
        self.process_local_attachments(topic_key=f"topic_{scene_id}")
        attachments = self.search_and_add_attachments(
            topic=topic, 
            topic_key=f"topic_{scene_id}", 
            download_files=True
        )
        
        papers = []
        if attachments.get("papers"):
            papers = attachments["papers"]
            self.message_stream.system_message(f"搜索到 {len(papers)} 篇相关论文，将作为综述参考文献")
        
        return papers

    def _search_open_source_systems(self, topic, include_open_source):
        """搜索相关开源系统"""
        open_source_systems = []
        if include_open_source:
            self.message_stream.system_message("搜索相关开源系统...")
            try:
                search_results = self.search_engine.search(f"{topic} open source github implementation")
                if "organic_results" in search_results:
                    for result in search_results["organic_results"][:5]:
                        if "github" in result.get("link", "").lower():
                            open_source_systems.append({
                                "name": result.get("title", "未命名项目"),
                                "description": result.get("snippet", "无描述"),
                                "url": result.get("link", "")
                            })
                self.message_stream.system_message(f"找到 {len(open_source_systems)} 个相关开源系统")
            except Exception as e:
                error_msg = f"搜索开源系统时出错: {str(e)}"
                print(error_msg)
        
        return open_source_systems

    def _prepare_subtopics(self, topic, subtopics, rounds):
        """准备要讨论的子话题"""
        # 首先检查是否为记忆管理系统相关主题
        memory_keywords = ["记忆管理", "记忆系统", "智能体记忆", "面向智能体的记忆管理系统","agent memory", "memory management"]
        if any(keyword.lower() in topic.lower() for keyword in memory_keywords):
            # 如果是记忆管理相关主题，直接使用预设子话题
            memory_subtopics = [
                "记忆的理论模型",
                "记忆如何存储", 
                "记忆如何查询",
                "典型应用"
            ]
            # 调整子话题数量以匹配轮次
            if len(memory_subtopics) > rounds:
                subtopics = memory_subtopics[:rounds]
            else:
                subtopics = memory_subtopics
                # 如需要更多轮次但子话题不足，可以补充
                while len(subtopics) < rounds:
                    subtopics.append(f"{topic}的拓展研究方向{len(subtopics)-len(memory_subtopics)+1}")
            
            # 显示使用的子话题
            subtopics_display = "\n".join([f"- {s}" for s in subtopics])
            self.message_stream.system_message(f"检测到记忆管理系统主题，自动使用预设子话题:\n{subtopics_display}")
            print(f"检测到记忆管理系统主题，自动使用预设子话题:\n{subtopics_display}")
            
            return subtopics, min(rounds, len(subtopics))  # 返回子话题列表和有效轮次数
        
        # 如果不是记忆管理相关主题，走原有逻辑
        if not subtopics:
            # 使用LLM生成与轮次匹配的子话题
            self.message_stream.system_message(f"自动生成{rounds}个子话题结构")
            print(f"\n自动生成{rounds}个子话题结构...")
            
            # 搜索主题相关信息
            topic_search = self.search_engine.search(f"{topic} key aspects components survey review")
            topic_info = self._format_search_results(topic_search)
            
            # 构建提示词
            # 3. 子话题应覆盖该领域的概念框架、理论基础、关键技术、应用实践和未来趋势
            # 4. 子话题之间应具有逻辑连贯性和层次性，从基础到应用，从现状到未来
            subtopic_prompt = f"""
            作为一位研究"{topic}"领域多年的学术专家，请分析这一研究领域并提出准确的{rounds}个最核心的子话题，用于组织一篇全面、系统的学术综述。

            分析要求：
            1. 必须提供恰好{rounds}个子话题，不多不少
            2. 每个子话题必须直接关联"{topic}"的核心内容，不能偏离主题
            3. 子话题之间应具有逻辑连贯性和层次性
            4. 每个子话题需要足够具体且可研究，便于深入讨论

            示例：
            对于"智能体的记忆管理系统"，可以提出以下子话题：
            记忆的理论模型
            记忆如何存储
            记忆如何查询
            典型应用

            如果有类似的topic如"智能体的记忆管理系统"，可以直接使用上述的子话题（根据rounds调整数量）

            以下是搜索到的相关信息:
            {topic_info}

            请直接返回{rounds}个子话题列表，每行一个子话题，无需编号或其他说明。
            """
            
            messages = [
                {"role": "system", "content": "你是一位擅长组织学术综述的研究者，能够确定研究领域的关键子话题。"},
                {"role": "user", "content": subtopic_prompt}
            ]
            
            subtopics_text = self.backend.llm.get_response(messages)
            if subtopics_text:
                # 处理返回的文本，提取子话题
                raw_subtopics = [s.strip() for s in subtopics_text.strip().split('\n') if s.strip()]
                subtopics = []
                for topic_text in raw_subtopics:
                    # 移除可能的编号、符号等
                    clean_topic = re.sub(r'^[\d\.\-\*]+\s*', '', topic_text)
                    if clean_topic:
                        subtopics.append(clean_topic)
                
                # 确保子话题数量与轮次匹配
                subtopics = self._adjust_subtopics_count(subtopics, rounds, topic)
                
                # 显示生成的子话题
                subtopics_display = "\n".join([f"- {s}" for s in subtopics])
                self.message_stream.system_message(f"已生成子话题结构:\n{subtopics_display}")
                print(f"自动生成的子话题:\n{subtopics_display}")
            else:
                # 使用默认子话题
                subtopics = self._get_default_subtopics(rounds, topic)
        else:
            # 调整提供的子话题数量
            if len(subtopics) != rounds:
                message = f"提供的子话题数量({len(subtopics)})与轮次({rounds})不匹配，将调整"
                self.message_stream.system_message(message)
                print(message)
                
            if len(subtopics) < rounds:
                rounds = len(subtopics)
            elif len(subtopics) > rounds:
                subtopics = subtopics[:rounds]
            
            # 显示使用的子话题
            subtopics_display = "\n".join([f"- {s}" for s in subtopics])
            self.message_stream.system_message(f"使用提供的子话题:\n{subtopics_display}")
        
        return subtopics, min(rounds, len(subtopics))  # 返回子话题列表和有效轮次数

    def _adjust_subtopics_count(self, subtopics, rounds, topic):
        """调整子话题数量以匹配轮次"""
        if len(subtopics) < rounds:
            print(f"警告：生成的子话题数量({len(subtopics)})少于轮次({rounds})，将自动补充")
            # 不足则生成通用补充子话题
            for i in range(rounds - len(subtopics)):
                subtopics.append(f"{topic}的拓展应用与实践{i+1}")
        elif len(subtopics) > rounds:
            print(f"警告：生成的子话题数量({len(subtopics)})多于轮次({rounds})，将取前{rounds}个")
            subtopics = subtopics[:rounds]
        
        return subtopics

    def _get_default_subtopics(self, rounds, topic):
        """获取默认子话题"""
        default_topics = [
            "记忆的理论模型",
            "记忆如何存储", 
            "记忆如何查询",
            "典型应用"
        ]
        # 确保有足够的默认主题
        while len(default_topics) < rounds:
            default_topics.append(f"{topic}研究方向{len(default_topics)+1}")
            
        subtopics = default_topics[:rounds]  # 只取需要的轮次数量
        self.message_stream.system_message(f"自动生成子话题失败，使用默认{rounds}个子话题")
        subtopics_display = "\n".join([f"- {s}" for s in subtopics])
        self.message_stream.system_message(subtopics_display)
        print(f"自动生成子话题失败，使用默认{rounds}个子话题")
        print(subtopics_display)
        
        return subtopics

    def _professor_introduces_topic(self, scene_id, professor_id, professor_name, topic, subtopics, rounds):
        """教授介绍综述主题和写作计划"""
        print(f"\n--- 第0轮：主题介绍与写作计划 ---")
        self.message_stream.round_info(round_num=0, topic=f"第0轮：主题介绍与写作计划")
        print(f"\n{professor_name}正在准备综述写作计划...")
        self.message_stream.system_message(f"{professor_name}正在准备综述写作计划...")
        
        # 搜索相关研究方法论
        formatted_results = ""
        try:
            intro_search = self.search_engine.search(f"{topic} latest research review methodology")
            formatted_results = self._format_search_results(intro_search)
        except Exception as e:
            error_msg = f"为教授介绍搜索信息时出错: {str(e)}"
            print(error_msg)
        
        # 构建提示词
        intro_prompt = f"""
        作为{professor_name}，你是一位资深教授，需要组织一次关于"{topic}"的综述报告研讨会。

        我们将围绕以下{len(subtopics)}个子话题进行{rounds}轮深入讨论，每轮讨论一个子话题:
        {', '.join([f"{i+1}.{subtopic}" for i, subtopic in enumerate(subtopics)])}
        
        请提供:
        1. 该主题的重要性、背景和当前研究状况
        2. 综述报告的组织框架和理论基础
        3. 各轮讨论的具体安排和预期目标
        4. 综述写作的计划和方法论
        
        请给出一个全面、专业的学术介绍，以引导后续的系统讨论。
        """
        
        messages = [
            {"role": "system", "content": "你是一位组织综述报告撰写的资深学者。"},
            {"role": "user", "content": intro_prompt}
        ]
        
        introduction = self.backend.llm.get_response(messages)
        if introduction:
            self.add_message(scene_id, professor_id, introduction)
            print(f"\n{professor_name}:\n{introduction}\n")
            self.message_stream.agent_message(professor_id, professor_name, introduction)
        
        return introduction

    def _get_student_initial_feedback(self, scene_id, students, professor_name, topic, subtopics, rounds):
        """获取学生对研究框架的初步反馈"""
        print("\n--- 学生对研究框架的初步反应 ---")
        self.message_stream.system_message("学生们正在对研究框架提供初步反馈...")
        
        subtopic_discussion = []
        for student_id in students:
            student_name = self.backend.agents[student_id]["Nickname"]
            specialty = self.backend.agents[student_id].get("Specialty", "")
            
            # 构建提示词
            response_prompt = f"""
            作为{student_name}，你刚刚听取了{professor_name}关于"{topic}"综述研究框架的介绍。
            
            研究计划将分{rounds}轮讨论以下子话题:
            {', '.join([f"{i+1}.{subtopic}" for i, subtopic in enumerate(subtopics)])}
            
            请提供你的初步反应，包括:
            1. 对这些子话题划分的评价（是否全面、合理）
            2. 基于你的专业背景({specialty})，你认为还有哪些方面值得关注
            3. 提出1-2个你希望在综述中探讨的具体问题
            
            请简明扼要地表达你的看法，控制在150字以内。
            """
            
            messages = [
                {"role": "system", "content": "你是一名参与综述研究的研究生，正在对研究框架提供初步反馈。"},
                {"role": "user", "content": response_prompt}
            ]
            
            response = self.backend.llm.get_response(messages)
            if response:
                message_id = self.add_message(scene_id, student_id, response)
                print(f"{student_name}:\n{response}\n")
                
                # 将ID添加到字典中
                subtopic_discussion.append({
                    "id": message_id, 
                    "speaker": student_name, 
                    "content": response
                })
                
                self.message_stream.agent_message(student_id, student_name, response)
        
        return subtopic_discussion

    def _professor_adjusts_framework(self, scene_id, professor_id, professor_name, topic, students, task_manager):
        """教授回应学生反馈并调整框架"""
        # 获取所有学生名字
        student_names = [self.backend.agents[s_id]["Nickname"] for s_id in students]
        
        self.message_stream.system_message(f"{professor_name}正在根据学生反馈调整研究框架...")
        
        # 构建提示词
        adj_prompt = f"""
        作为{professor_name}，请回应学生们的初步反馈，调整和完善"{topic}"综述的研究框架。
        
        请包括:
        1. 对学生提出的关键问题的回应
        2. 框架的必要调整或补充说明
        3. 各子话题之间的联系和整体结构优化
        4. 为第一轮讨论的每位学生分配一项具体的研究任务，学生的名字为{student_names}
        共{len(students)}个任务，这些任务应在下一轮讨论前完成。

        对于4中的分配任务，具体要求如下：请严格按以下格式为每位学生分配一项任务,中间不要有换行或空格：
        1. **学生姓名1**: [任务具体描述]
        2. **学生姓名2**: [任务具体描述]
        ...（依此类推）
        
        任务描述应在50-100字左右。
        
        请以清晰、客观的方式呈现，便于会后查阅和跟进。
        """
        # 任务描述应在50-100字左右，清晰表述研究目标和期望的成果形式
        messages = [
            {"role": "system", "content": "你是一位指导综述研究的教授，正在完善研究框架。"},
            {"role": "user", "content": adj_prompt}
        ]
        
        adjustment = self.backend.llm.get_response(messages)
        if adjustment:
            self.add_message(scene_id, professor_id, adjustment)
            print(f"{professor_name}:\n{adjustment}\n")
            self.message_stream.agent_message(professor_id, professor_name, adjustment)
            
            # 从调整内容中提取任务
            self._extract_and_assign_tasks(adjustment, students, 0, task_manager)

    def _extract_and_assign_tasks(self, text, students, round_num, task_manager):
        """从文本中提取任务并分配给学生"""
        # # 提取 [学生姓名]: [任务描述] 格式的任务
        # task_pattern = r"\d+\.\s*\[([^\]]+)\]:\s*(.+?)(?=\d+\.\s*\[|$)"
        # task_matches = re.findall(task_pattern, text + "999. [End]", re.DOTALL)
        # 修改正则表达式以匹配 **学生姓名**: [任务描述] 格式
        # 更加灵活的正则表达式 - 不要求方括号，增加多种格式支持
        task_pattern = r"\d+\.\s*\*\*([^*]+?)\*\*\s*:?\s*(?:\[)?(.*?)(?:\])?(?=\s*\d+\.\s*\*\*|$|999\. \[End\])"
        task_matches = re.findall(task_pattern, text + "999. [End]", re.DOTALL)
        
        # 打印匹配结果
        # print(f"找到 {len(task_matches)} 个任务匹配")
        # for i, (name, desc) in enumerate(task_matches):
        #     print(f"任务 {i+1}: 学生='{name.strip()}', 描述='{desc.strip()}'")

        task_count = 0
        # 处理每个提取的任务
        for student_name, task_desc in task_matches:
            task_desc = task_desc.strip()
            
            # 查找对应的student_id
            target_student_id = None
            for s_id in students:
                if self.backend.agents[s_id]["Nickname"].strip() == student_name.strip():
                    target_student_id = s_id
                    break
            
            if target_student_id:
                # 添加到按学生组织的任务列表
                if target_student_id not in task_manager['by_student']:
                    task_manager['by_student'][target_student_id] = []
                task_manager['by_student'][target_student_id].append((round_num, task_desc, "待完成"))
                
                # 添加到按轮次组织的任务列表
                if round_num not in task_manager['by_round']:
                    task_manager['by_round'][round_num] = []
                task_manager['by_round'][round_num].append((target_student_id, task_desc, "待完成"))
                
                # 添加到所有任务列表
                task_manager['all_tasks'].append((round_num, target_student_id, task_desc, "待完成"))
                
                task_count += 1
                print(f"在第{round_num}轮为学生 {student_name} 分配任务: {task_desc}")
                
                # 直接向前端发送任务通知
                self.message_stream.system_message(
                    f"📋 第{round_num}轮任务分配 - {student_name}:\n{task_desc}"
                )
        
        if task_count > 0:
            self.message_stream.system_message(f"在第{round_num}轮讨论中分配了 {task_count} 个任务")
            
        return task_count

    def _conduct_discussion_rounds(self, scene_id, topic, subtopics, effective_rounds, professor_id, professor_name, students, include_literature, papers, task_manager):
        """进行多轮子话题讨论"""
        subsection_contents = {}  # 存储每个子话题的内容
        literature_by_topic = {}  # 按子话题存储文献
        
        # 处理每一轮讨论
        for round_num in range(1, effective_rounds + 1):
            current_subtopic = subtopics[round_num-1]
            print(f"\n--- 第{round_num}轮讨论: {current_subtopic} ---")
            self.message_stream.round_info(round_num, current_subtopic)
            
            # 确定本轮讨论的特点和搜索焦点
            round_focus, search_focus = self._determine_round_focus(round_num, effective_rounds)
            
            # 教授引导本轮讨论
            round_guide = self._professor_guides_round(scene_id, professor_id, professor_name, current_subtopic, round_focus, round_num)
            
            # 搜索子话题相关文献
            subtopic_papers = self._search_subtopic_literature(topic, current_subtopic, search_focus, include_literature, papers)
            literature_by_topic[current_subtopic] = subtopic_papers
            
            # 学生讨论子话题
            # 学生讨论子话题
            subtopic_discussion = self._students_discuss_subtopic(
                scene_id, students, professor_id, professor_name, 
                topic, current_subtopic, round_num, round_focus, search_focus,
                subtopic_papers, task_manager, subtopics  # 添加subtopics参数
            )
            
            # 教授总结子话题讨论
            subsection_summary = self._professor_summarizes_subtopic(
                scene_id, professor_id, professor_name, current_subtopic, 
                round_num, round_focus, subtopic_discussion
            )
            
            if subsection_summary:
                subsection_contents[current_subtopic] = subsection_summary
            
            # 处理轮次结束逻辑
            if round_num < effective_rounds:
                next_subtopic = subtopics[round_num]
                
                # 教授总结本轮并过渡到下一轮
                round_conclusion = self._professor_concludes_round(
                    scene_id, professor_id, professor_name, current_subtopic, 
                    next_subtopic, round_num
                )
                
                # 生成会议纪要和分配任务
                if round_conclusion:
                    self._generate_round_minutes_and_tasks(
                        scene_id, round_num, current_subtopic, round_focus,
                        round_conclusion, professor_name, students, task_manager
                    )
            
            elif round_num == effective_rounds:
                # 生成最终会议纪要
                self._generate_final_meeting_minutes(topic, current_subtopic, effective_rounds)
        
        return subsection_contents, literature_by_topic

    def _determine_round_focus(self, round_num, effective_rounds):
        """确定当前轮次的讨论焦点和搜索关键词"""
        if effective_rounds == 1:
            return "综合视角：从基础概念到未来趋势", "comprehensive overview fundamentals applications future"
        elif round_num == 1:
            return "基础概念与研究现状", "fundamental concepts current status"
        elif round_num == effective_rounds:
            return "未来发展方向与应用前景", "future trends applications challenges"
        else:
            return "关键技术与方法分析", "key techniques methods comparison"

    def _professor_guides_round(self, scene_id, professor_id, professor_name, current_subtopic, round_focus, round_num):
        """教授引导当前轮次讨论"""
        self.message_stream.system_message(f"{professor_name}准备引导第{round_num}轮讨论...")
        
        guide_prompt = f"""
        作为{professor_name}，你需要引导第{round_num}轮讨论，主题是"{current_subtopic}"。
        
        本轮讨论重点是: {round_focus}
        
        请引导讨论，包括:
        1. 简要介绍本轮将讨论的子话题及其重要性
        2. 强调本轮讨论应关注的{round_focus}方面
        3. 提出2-3个关键问题引导学生思考
        
        请控制在200字以内，简明扼要地引导讨论。
        """
        
        messages = [
            {"role": "system", "content": "你是组织综述讨论的教授，正在引导特定轮次的讨论。"},
            {"role": "user", "content": guide_prompt}
        ]
        
        round_guide = self.backend.llm.get_response(messages)
        if round_guide:
            self.add_message(scene_id, professor_id, round_guide)
            print(f"{professor_name}:\n{round_guide}\n")
            self.message_stream.agent_message(professor_id, professor_name, round_guide)
        
        return round_guide

    def _search_subtopic_literature(self, topic, current_subtopic, search_focus, include_literature, papers):
        """搜索子话题相关文献"""
        subtopic_papers = []
        
        if include_literature:
            print(f"搜索{current_subtopic}相关文献...")
            self.message_stream.system_message(f"搜索{current_subtopic}相关文献...")
            
            try:
                search_results = self.search_engine.search(f"{topic} {current_subtopic} {search_focus} research papers")
                if "organic_results" in search_results:
                    for result in search_results["organic_results"][:3]:
                        if "pdf" in result.get("link", "").lower() or "research" in result.get("link", "").lower():
                            subtopic_papers.append({
                                "title": result.get("title", "未命名论文"),
                                "abstract": result.get("snippet", "无摘要"),
                                "url": result.get("link", ""),
                                "authors": "N/A",
                                "year": "N/A"
                            })
                self.message_stream.system_message(f"为{current_subtopic}找到 {len(subtopic_papers)} 篇相关文献")
            except Exception as e:
                print(f"搜索文献时出错: {str(e)}")
                self.message_stream.system_message(f"搜索文献时出错: {str(e)}")
        
        # 将已有论文与子话题关联
        for paper in papers:
            paper_title = paper.get("Title", "").lower()
            paper_abstract = paper.get("Abstract", "").lower()
            if any(keyword in paper_title or keyword in paper_abstract 
                for keyword in current_subtopic.lower().split()):
                subtopic_papers.append(paper)
        
        return subtopic_papers
    
    def _students_discuss_subtopic(self, scene_id, students, professor_id, professor_name, topic, current_subtopic, round_num, round_focus, search_focus, subtopic_papers, task_manager, subtopics):
        """学生讨论子话题，教授给予反馈"""
        subtopic_discussion = []
        
        for student_id in students:
            student_name = self.backend.agents[student_id]["Nickname"]
            specialty = self.backend.agents[student_id].get("Specialty", "")
            
            print(f"\n{student_name}正在分析{current_subtopic}...")
            self.message_stream.system_message(f"{student_name}正在分析{current_subtopic}...")
            
            # 学生搜索信息
            search_query = f"{topic} {current_subtopic} {search_focus} {specialty}"
            student_search = self.search_engine.search(search_query)
            student_results = self._format_search_results(student_search)
            
            # 检查学生是否有上一轮分配的任务需要完成
            previous_task = self._get_student_previous_task(student_id, round_num-1, task_manager)
            
            # 学生分析文献
            literature_info = self._format_literature_info(subtopic_papers)
            
            # 使用传入的subtopics参数获取轮次总数
            effective_rounds = len(subtopics) if subtopics else round_num
            
            # 学生发言提示，根据轮次侧重不同
            focus_instructions = self._get_focus_instructions(round_num, effective_rounds)
            
            # 修改学生提示，增加任务报告部分
            task_report_section = ""
            
            # print(f"学生{student_name}的上一轮任务: {previous_task}")
            # 如果有上一轮任务，添加任务报告部分
            if previous_task:
                print(f"学生{student_name}的上一轮任务: {previous_task}")
                
                task_report_section = f"""
                在讨论本轮主题前，请先简要汇报你完成的上一轮任务：
                
                任务内容：{previous_task}
                
                请用1-2段文字总结你的研究发现和完成情况，然后再进入本轮主题的讨论。
                """
                
                # 在学生回复前，将任务标记为已完成
                self._mark_task_as_completed(student_id, round_num-1, previous_task, task_manager)
            
            # print(f"学生{student_name}的上一轮任务: {previous_task}")

            # 构建学生发言提示
            # student_prompt = f"""
            # 在讨论本轮主题前，请先简要汇报你完成的上一轮任务：
                
            # 任务内容：{previous_task}
            
            # 请用1-2段文字总结你的研究发现和完成情况，然后再进入本轮主题的讨论。
            student_prompt = f"""
            作为{student_name}，你需要针对"{current_subtopic}"这一"{topic}"综述的子话题提供深入分析。
            
            本轮（第{round_num}轮）讨论重点: {round_focus}
            
            {task_report_section}
            
            {focus_instructions}
            
            搜索资料:
            {student_results}
            
            {literature_info}
            
            请提供对该子话题的专业分析，包括:
            1. {'' if not previous_task else '首先简要报告你完成的上一轮任务情况'}
            2. 关键概念、挑战和研究问题
            3. 主要方法和技术路线
            4. 当前研究水平与存在的不足
            5. 基于本轮讨论重点的专业见解
            
            请从你的专业角度({specialty})出发，提供深入的学术分析，可以引用上述文献支持你的观点。
            """
            
            messages = [
                {"role": "system", "content": "你是一名参与综述撰写的研究生，需要提供针对特定轮次讨论重点的学术分析。"},
                {"role": "user", "content": student_prompt}
            ]
            
            response = self.backend.llm.get_response(messages)
            if response:
                self._mark_task_as_completed(student_id, round_num-1, previous_task, task_manager)
                message_id = self.add_message(scene_id, student_id, response)
                print(f"{student_name}:\n{response}\n")
                subtopic_discussion.append({"id": message_id, "speaker": student_name, "content": response})
                self.message_stream.agent_message(student_id, student_name, response)
                
                # 教授给予反馈
                self._professor_gives_feedback(scene_id, professor_id, professor_name, student_name, current_subtopic, round_focus)
        
        return subtopic_discussion

    def _get_student_previous_task(self, student_id, prev_round, task_manager):
        """获取学生上一轮的任务"""
        if student_id in task_manager['by_student']:
            for task_round, task_desc, status in task_manager['by_student'][student_id]:
                if task_round == prev_round and status == "待完成":
                    return task_desc
        return None

    def _mark_task_as_completed(self, student_id, task_round, task_desc, task_manager):
        """将任务标记为已完成"""
        # 更新按学生组织的任务列表
        if student_id in task_manager['by_student']:
            for i, (round_num, desc, status) in enumerate(task_manager['by_student'][student_id]):
                if round_num == task_round and desc == task_desc and status == "待完成":
                    task_manager['by_student'][student_id][i] = (round_num, desc, "已完成")
        
        # 更新按轮次组织的任务列表
        if task_round in task_manager['by_round']:
            for i, (s_id, desc, status) in enumerate(task_manager['by_round'][task_round]):
                if s_id == student_id and desc == task_desc and status == "待完成":
                    task_manager['by_round'][task_round][i] = (s_id, desc, "已完成")
        
        # 更新所有任务列表
        for i, (round_num, s_id, desc, status) in enumerate(task_manager['all_tasks']):
            if round_num == task_round and s_id == student_id and desc == task_desc and status == "待完成":
                task_manager['all_tasks'][i] = (round_num, s_id, desc, "已完成")
        
        # 通知前端任务已完成
        student_name = self.backend.agents[student_id]["Nickname"]
        self.message_stream.system_message(f"✅ {student_name}完成了第{task_round}轮任务：{task_desc[:30]}...")

    def _format_literature_info(self, papers):
        """格式化文献信息"""
        if not papers:
            return ""
            
        literature_info = "相关文献:\n"
        for idx, paper in enumerate(papers[:3], 1):
            title = paper.get("title", paper.get("Title", "未命名论文"))
            abstract = paper.get("abstract", paper.get("Abstract", "无摘要"))
            literature_info += f"{idx}. {title}\n   摘要: {abstract[:150]}...\n\n"
        
        return literature_info

    def _get_focus_instructions(self, round_num, effective_rounds):
        """获取不同轮次的讨论焦点说明"""
        if effective_rounds == 1:
            # 只有一轮讨论的情况，需要全面覆盖
            return """
            请全面分析该子话题，包括基础概念定义、研究现状、关键技术方法、应用场景以及未来发展趋势。
            你的分析应该涵盖该子话题的各个重要方面，既有理论深度又有实践指导意义。
            """
        elif round_num == 1:
            return """
            重点分析该子话题的基础概念、定义和研究现状。讨论这一领域的基础理论、发展历程和当前关注的主要问题。
            """
        elif round_num == effective_rounds:
            return """
            重点分析该子话题的未来发展趋势、面临的挑战和潜在的应用场景。讨论该方向的创新点和可能的突破口。
            """
        else:
            return """
            重点分析该子话题的关键技术和方法，比较不同方法的优缺点，讨论当前研究中的技术难点和解决思路。
            """

    def _professor_gives_feedback(self, scene_id, professor_id, professor_name, student_name, current_subtopic, round_focus):
        """教授对学生发言给予反馈"""
        self.message_stream.system_message(f"{professor_name}正在对{student_name}的分析给予反馈...")
        
        feedback_prompt = f"""
        作为{professor_name}，请针对{student_name}关于"{current_subtopic}"的分析给予专业反馈，
        特别是基于本轮讨论重点({round_focus})的角度。
        
        请指出其分析的优点、可以改进的地方，并添加你的专业见解或补充。
        请控制在150字以内。
        """
        
        messages = [
            {"role": "system", "content": "你是一位指导综述讨论的教授，需要给予有针对性的建设性反馈。"},
            {"role": "user", "content": feedback_prompt}
        ]
        
        feedback = self.backend.llm.get_response(messages)
        if feedback:
            self.add_message(scene_id, professor_id, feedback)
            print(f"{professor_name}:\n{feedback}\n")
            self.message_stream.agent_message(professor_id, professor_name, feedback)
        
        return feedback

    def _professor_summarizes_subtopic(self, scene_id, professor_id, professor_name, current_subtopic, round_num, round_focus, subtopic_discussion):
        """教授总结子话题讨论"""
        self.message_stream.system_message(f"{professor_name}正在总结第{round_num}轮子话题讨论...")
        
        summary_prompt = f"""
        作为{professor_name}，请对"{current_subtopic}"的第{round_num}轮讨论({round_focus})进行全面总结。
        
        请提供一个系统的学术总结，包括:
        1. 该子话题在本轮讨论中的关键发现和共识
        2. 主要争议点或不同观点
        3. 与其他子话题的联系和影响
        4. 下一步研究方向的建议
        
        这将作为综述报告中该部分的基础内容，请提供结构清晰、内容全面的学术总结。
        """
        
        messages = [
            {"role": "system", "content": "你是一位撰写综述报告的资深学者，需要对特定轮次的子话题讨论进行专业总结。"},
            {"role": "user", "content": summary_prompt}
        ]
        
        subsection_summary = self.backend.llm.get_response(messages)
        if subsection_summary:
            self.add_message(scene_id, professor_id, subsection_summary)
            print(f"\n{professor_name} (子话题总结):\n{subsection_summary}\n")
            
            self.message_stream.agent_message(
                professor_id, 
                professor_name, 
                subsection_summary, 
                message_type="subtopic_summary"
            )
            
            # 将子话题总结添加到语义图中
            subtopic_summary_id = self.add_summary(
                scene_id=scene_id,
                speaker_id=professor_id,
                content=subsection_summary,
                summary_type="subtopic",
                round_num=round_num,
                related_message_ids=[msg["id"] for msg in subtopic_discussion if "id" in msg]
            )
            
            print(f"已将子话题'{current_subtopic}'的总结添加到语义图(ID: {subtopic_summary_id})")
            self.message_stream.system_message(f"已将子话题'{current_subtopic}'的总结添加到语义图")
        
        return subsection_summary

    def _professor_concludes_round(self, scene_id, professor_id, professor_name, current_subtopic, next_subtopic, round_num):
        """教授总结本轮讨论并过渡到下一轮"""
        self.message_stream.system_message(f"正在总结第{round_num}轮讨论并过渡到下一轮...")
        
        round_summary_prompt = f"""
        作为{professor_name}，请对第{round_num}轮关于"{current_subtopic}"的讨论进行总结，
        并自然地引导到第{round_num+1}轮将要讨论的内容"{next_subtopic}"。
        
        请提供一个连贯的过渡，控制在200字以内。
        """
        
        messages = [
            {"role": "system", "content": "你是一位综述研讨会的主持人，需要进行轮次总结与过渡。"},
            {"role": "user", "content": round_summary_prompt}
        ]
        
        round_conclusion = self.backend.llm.get_response(messages)
        if round_conclusion:
            self.add_message(scene_id, professor_id, round_conclusion)
            print(f"\n{professor_name} (第{round_num}轮总结):\n{round_conclusion}\n")
            
            # 获取相关消息ID
            scene = None
            if hasattr(self, 'scenes') and scene_id in self.scenes:
                scene = self.scenes.get(scene_id)

            if scene:
                related_message_ids = [msg.get("id") for msg in scene.messages[-20:] if "id" in msg]
            else:
                print(f"警告：未找到场景 {scene_id}，无法访问消息历史")
                self.message_stream.system_message(f"警告：未找到场景 {scene_id}，无法访问消息历史")
                related_message_ids = []

            # 将轮次总结添加到语义图中
            round_summary_id = self.add_summary(
                scene_id=scene_id,
                speaker_id=professor_id,
                content=round_conclusion,
                summary_type="round",
                round_num=round_num,
                related_message_ids=related_message_ids
            )
            
            print(f"已将第{round_num}轮总结添加到语义图(ID: {round_summary_id})")
            self.message_stream.system_message(f"已将第{round_num}轮总结添加到语义图")
            
            self.message_stream.agent_message(
                professor_id, 
                professor_name, 
                round_conclusion,
                message_type="round_summary"
            )
        
        return round_conclusion

    def _generate_round_minutes_and_tasks(self, scene_id, round_num, current_subtopic, round_focus, round_conclusion, professor_name, students, task_manager):
        """生成轮次会议纪要并分配任务"""
        meeting_minutes_prompt = f"""
        请根据以下第{round_num}轮关于"{current_subtopic}"的讨论内容，生成一份简明扼要的会议纪要。
        
        讨论重点: {round_focus}
        教授总结: {round_conclusion}
        
        纪要应包含：
        1. 本轮主要讨论的要点（3-5点）
        2. 达成的共识或结论
        3. 待解决的问题或分歧
        4. 为每位学生分配一项具体的研究任务，共{len(students)}个任务，这些任务应在下一轮讨论前完成。

        对于4中的分配任务，具体要求如下：请严格按以下格式为每位学生分配一项任务,中间不要有换行或空格：
        1. **学生姓名1**: [任务具体描述]
        2. **学生姓名2**: [任务具体描述]
        ...（依此类推）
        
        任务描述应在50-100字左右。
        
        请以清晰、客观的方式呈现，便于会后查阅和跟进。
        """
        # 任务描述应在50-100字左右，清晰表述研究目标和期望的成果形式
        minutes_messages = [
            {"role": "system", "content": f"你是一位专业的学术会议纪要记录员，需要提供简洁明了的会议纪要，同时为下一轮的讨论生成{len(students)}个任务。"},
            {"role": "user", "content": meeting_minutes_prompt}
        ]
        
        meeting_minutes = self.backend.llm.get_response(minutes_messages)
        if meeting_minutes:
            # 使用系统消息形式发送会议纪要
            minutes_title = f"📝 第{round_num}轮「{current_subtopic}」会议纪要"
            formatted_minutes = f"{minutes_title}\n\n{meeting_minutes}"
            self.message_stream.system_message(formatted_minutes)
            print(f"\n{minutes_title}\n{meeting_minutes}\n")
            
            # 从会议纪要中提取和分配任务
            self._extract_and_assign_tasks(meeting_minutes, students, round_num, task_manager)
            
            # 显示每个学生的任务列表
            self._display_student_tasks(students, round_num, task_manager)

    def _display_student_tasks(self, students, current_round, task_manager):
        """显示每个学生的任务列表"""
        for student_id in students:
            student_name = self.backend.agents[student_id]["Nickname"]
            # 获取当前轮次新分配的任务
            new_tasks = []
            if student_id in task_manager['by_student']:
                new_tasks = [
                    task_desc for round_num, task_desc, status in task_manager['by_student'][student_id]
                    if round_num == current_round and status == "待完成"
                ]
            
            if new_tasks:
                task_display = "\n".join([f"- {task}" for task in new_tasks])
                self.message_stream.system_message(f"📋 第{current_round}轮后 {student_name}的新任务：\n{task_display}")

    def _generate_final_meeting_minutes(self, topic, current_subtopic, effective_rounds):
        """生成最终会议纪要"""
        final_minutes_prompt = f"""
        请根据关于"{topic}"的全部{effective_rounds}轮讨论，生成一份简明扼要的最终会议纪要。
        
        最后讨论的子话题: {current_subtopic}
        
        纪要应包含：
        1. 会议总体概况和目标达成情况
        2. 各轮次讨论的核心要点
        3. 与会者达成的主要共识
        4. 需要后续跟进的研究方向和行动点
        
        请以清晰、客观的方式呈现整体会议成果。
        """
        
        final_minutes_messages = [
            {"role": "system", "content": "你是一位专业的学术会议纪要记录员，需要提供综合性的会议纪要总结。"},
            {"role": "user", "content": final_minutes_prompt}
        ]
        
        final_meeting_minutes = self.backend.llm.get_response(final_minutes_messages)
        if final_meeting_minutes:
            # 使用系统消息形式发送最终会议纪要
            final_minutes_title = f"📝 {topic}研讨会议总纪要"
            formatted_final_minutes = f"{final_minutes_title}\n\n{final_meeting_minutes}"
            self.message_stream.system_message(formatted_final_minutes)
            print(f"\n{final_minutes_title}\n{final_meeting_minutes}\n")

    def _merge_literature(self, literature_by_topic):
        """合并所有文献信息，去重"""
        all_literature = []
        for subtopic, papers in literature_by_topic.items():
            for paper in papers:
                paper_title = paper.get("title", paper.get("Title", ""))
                if not any(p.get("title", p.get("Title", "")) == paper_title for p in all_literature):
                    all_literature.append(paper)
        return all_literature

    def _generate_final_review(self, scene_id, topic, professor_id, professor_name, subtopics, effective_rounds, subsection_contents, literature_by_topic, open_source_systems, all_literature, output_dir=None):
        """生成最终综述报告"""
        print("\n--- 整合讨论成果，生成最终综述报告 ---\n")
        self.message_stream.system_message("整合讨论成果，生成最终综述报告...")
        
        # 获取场景对象
        scene = self.scenes.get(scene_id)
        if not scene:
            print(f"警告：未找到场景 {scene_id}")
            self.message_stream.system_message(f"警告：未找到场景 {scene_id}")
            return {"error": "未找到场景", "scene_id": scene_id}
        
        # 准备综述报告各部分内容
        review_sections = self._prepare_review_sections(subsection_contents)
        literature_section = self._prepare_literature_section(all_literature)
        open_source_section = self._prepare_open_source_section(open_source_systems)
        
        # 构建最终综述报告生成提示并生成报告
        final_review = self._generate_review_content(
            professor_name, topic, subtopics, effective_rounds, review_sections
        )
        
        # 处理并保存最终报告
        if final_review:
            # 完善报告内容
            final_review = self._enhance_review_content(
                final_review, topic, literature_section, open_source_section
            )
            
            # 保存报告到文件
            report_info = self._save_review_to_file(
                topic, professor_name, final_review, output_dir
            )
            
            # 添加到语义图
            self._add_review_to_semantic_graph(
                scene_id, professor_id, report_info, final_review, topic, subtopics,
                all_literature, open_source_systems, effective_rounds, scene
            )
            
            # 可视化对话图
            try:
                self.visualize_conversation_graph(scene_id)
                self.message_stream.system_message("已生成对话可视化图表")
            except Exception as e:
                print(f"无法生成可视化: {str(e)}")
                self.message_stream.system_message(f"无法生成可视化: {str(e)}")
            
            # 返回综述报告信息
            return {
                "report_id": report_info["report_id"],
                "scene_id": scene_id,
                "filename": report_info["filepath"],
                "topic": topic,
                "subtopics": subtopics,
                "professor": professor_name,
                "content": final_review,
                "references": all_literature,
                "open_source_systems": open_source_systems,
                "rounds": effective_rounds
            }
        else:
            # 如果最终无法生成报告，返回基本信息
            print("警告：无法生成完整的综述报告内容")
            self.message_stream.system_message("警告：无法生成完整的综述报告内容")
            return {
                "error": "无法生成综述报告内容",
                "scene_id": scene_id,
                "topic": topic
            }

    def _prepare_review_sections(self, subsection_contents):
        """准备综述报告的各个章节"""
        review_sections = []
        for subtopic, content in subsection_contents.items():
            review_sections.append(f"## {subtopic}\n\n{content}")
        return review_sections

    def _prepare_literature_section(self, all_literature):
        """准备综述报告的文献引用部分"""
        if not all_literature:
            return ""
            
        literature_section = "## 参考文献\n\n"
        for i, paper in enumerate(all_literature, 1):
            title = paper.get("title", paper.get("Title", "未命名论文"))
            authors = paper.get("authors", paper.get("Authors", "N/A"))
            year = paper.get("year", paper.get("Year", "N/A"))
            url = paper.get("url", paper.get("URL", ""))
            literature_section += f"[{i}] {authors}. ({year}). {title}. {url}\n\n"
        
        return literature_section

    def _prepare_open_source_section(self, open_source_systems):
        """准备综述报告的开源系统部分"""
        if not open_source_systems:
            return ""
            
        open_source_section = "## 相关开源系统\n\n"
        for i, system in enumerate(open_source_systems, 1):
            name = system.get("name", "未命名项目")
            description = system.get("description", "无描述")
            url = system.get("url", "")
            open_source_section += f"### {i}. {name}\n\n{description}\n\n链接: {url}\n\n"
        
        return open_source_section

    def _generate_review_content(self, professor_name, topic, subtopics, effective_rounds, review_sections):
        """生成综述报告内容"""
        # 构建最终综述报告生成提示
        if effective_rounds == 1:
            final_prompt = self._build_single_round_review_prompt(
                professor_name, topic, subtopics, review_sections
            )
        else:
            final_prompt = self._build_multi_round_review_prompt(
                professor_name, topic, subtopics, effective_rounds, review_sections
            )
        
        messages = [
            {"role": "system", "content": "你是一位撰写高质量学术综述报告的资深教授，专门研究该领域多年。创建一个结构完整、内容翔实的学术综述，确保内容全面且深度专业。"},
            {"role": "user", "content": final_prompt}
        ]
        
        # 增加重试机制确保内容生成
        max_retries = 3
        final_review = None
        self.message_stream.system_message("正在撰写最终综述报告，可能需要一点时间...")
        
        for attempt in range(max_retries):
            try:
                generated_content = self.backend.llm.get_response(messages)
                if generated_content and len(generated_content.strip()) > 100:
                    final_review = generated_content
                    print(f"成功生成综述报告，长度：{len(final_review)}字符")
                    self.message_stream.system_message(f"成功生成综述报告，长度：{len(final_review)}字符")
                    break
                else:
                    print(f"生成的内容太短或为空，尝试第{attempt + 2}次...")
            except:
                print(f"生成内容时出错，尝试第{attempt + 2}次...")
                self.message_stream.system_message(f"生成内容时出错，尝试第{attempt + 2}次...")

    def _enhance_review_content(self, final_review, topic, literature_section, open_source_section):
        """完善综述报告内容，添加结论和参考资料
        
        Args:
            final_review: 原始综述内容
            topic: 主题
            literature_section: 参考文献部分
            open_source_section: 开源系统部分
            
        Returns:
            str: 增强后的综述报告内容
        """
        # 如果缺少结论部分，自动添加
        if "# 结论" not in final_review and "## 结论" not in final_review:
            self.message_stream.system_message("补充结论部分...")
            conclusion_prompt = f"""
            为"{topic}"的综述报告生成一个简洁而全面的结论部分。
            结论应总结主要发现，强调研究意义，并指出未来发展方向。
            请使用学术语言，控制在300字左右。
            """
            
            messages = [
                {"role": "system", "content": "你是一位撰写学术综述结论的专家。"},
                {"role": "user", "content": conclusion_prompt}
            ]
            
            try:
                conclusion = self.backend.llm.get_response(messages)
                if conclusion and len(conclusion.strip()) > 50:
                    final_review += f"\n\n## 结论\n\n{conclusion}"
            except Exception as e:
                print(f"生成结论时出错: {str(e)}")
                self.message_stream.system_message(f"生成结论时出错: {str(e)}")
                # 添加基本结论确保不为空
                final_review += f"\n\n## 结论\n\n本综述报告全面分析了{topic}的研究现状、关键技术和未来发展方向。通过系统梳理该领域的重要文献和研究成果，为后续研究提供了参考框架。"
        
        # 添加参考文献和开源系统部分
        if literature_section:
            final_review += f"\n\n{literature_section}"
        
        if open_source_section:
            final_review += f"\n\n{open_source_section}"
        
        return final_review
    
    def _save_review_to_file(self, topic, professor_name, final_review, output_dir=None):
        """保存综述报告到文件
        
        Args:
            topic: 主题
            professor_name: 教授姓名
            final_review: 综述内容
            output_dir: 输出目录
            
        Returns:
            dict: 包含报告ID和文件路径的信息
        """
        # 生成报告ID和文件名
        report_id = f"review_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{topic.replace(' ', '_')}_{timestamp}.md"
        
        # 设置输出目录
        if output_dir is None:
            output_dir = ARTICLE_DIR
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建完整文件路径
        filepath = os.path.join(output_dir, filename)
        
        # 尝试保存文件
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {topic}综述报告\n\n")
                f.write(f"作者: {professor_name}\n\n")
                f.write(f"日期: {datetime.now().strftime('%Y-%m-%d')}\n\n")
                f.write(final_review)
            
            print(f"\n综述报告已保存至: {filepath}")
            self.message_stream.system_message(f"综述报告已保存至: {filepath}")
            
            return {
                "report_id": report_id,
                "filepath": filepath
            }
        except Exception as e:
            print(f"保存报告时出错: {str(e)}")
            self.message_stream.system_message(f"保存报告时出错: {str(e)}")
            
            return {
                "report_id": report_id,
                "filepath": None,
                "error": str(e)
            }
        
    def _add_review_to_semantic_graph(self, scene_id, professor_id, report_info, final_review, topic, subtopics, all_literature, open_source_systems, effective_rounds, scene):
        """将综述报告添加到语义图
        
        Args:
            scene_id: 场景ID
            professor_id: 教授ID 
            report_info: 报告信息字典
            final_review: 综述内容
            topic: 主题
            subtopics: 子话题列表
            all_literature: 所有参考文献
            open_source_systems: 所有开源系统
            effective_rounds: 有效讨论轮次
            scene: 场景对象
        """
        # 准备报告基本信息
        report_id = report_info["report_id"]
        review_info = {
            "Title": f"{topic}综述报告" if topic else "学术综述报告",
            "Author": self.backend.agents[professor_id]["Nickname"] if professor_id in self.backend.agents else "学术教授",
            "Date": datetime.now().strftime('%Y-%m-%d'),
            "Content": final_review if final_review else f"{topic}相关研究综述",
            "Topics": ", ".join(subtopics) if subtopics else topic,
            "References": len(all_literature),
            "OpenSourceSystems": len(open_source_systems),
            "Rounds": effective_rounds
        }
        
        # 确定父节点列表
        valid_parent_keys = []
        if f"topic_{scene_id}" in self.backend.semantic_graph.graph_relations:
            valid_parent_keys.append(f"topic_{scene_id}")
        if professor_id in self.backend.semantic_graph.graph_relations:
            valid_parent_keys.append(professor_id)
        
        # 添加到语义图
        try:
            if valid_parent_keys:
                self.backend.add_conclusion(report_id, review_info, parent_keys=valid_parent_keys)
                print(f"综述报告已添加到语义图，ID: {report_id}")
                self.message_stream.system_message(f"综述报告已添加到语义图，ID: {report_id}")
            else:
                print("警告：找不到有效的父节点，将直接添加报告节点")
                self.message_stream.system_message("警告：找不到有效的父节点，将直接添加报告节点")
                self.backend.add_conclusion(report_id, review_info)
            
            # 将最终报告添加为总结节点
            try:
                related_message_ids = [msg["id"] for msg in scene.messages[-20:]] if scene else []
                
                final_summary_id = self.add_summary(
                    scene_id=scene_id,
                    speaker_id=professor_id,
                    content=final_review,
                    summary_type="final",
                    related_message_ids=related_message_ids
                )
                print(f"已将最终综述报告添加到语义图(ID: {final_summary_id})")
                self.message_stream.system_message(f"已将最终综述报告添加到语义图")
            except Exception as e:
                print(f"添加最终总结节点时出错: {str(e)}")
                self.message_stream.system_message(f"添加最终总结节点时出错: {str(e)}")
        
        except Exception as e:
            print(f"添加报告到语义图时出错: {str(e)}")
            self.message_stream.system_message(f"添加报告到语义图时出错: {str(e)}")

    def _build_single_round_review_prompt(self, professor_name, topic, subtopics, review_sections):
        """构建单轮讨论的综述提示词
        
        Args:
            professor_name: 教授姓名
            topic: 主题
            subtopics: 子话题列表
            review_sections: 各部分综述内容
            
        Returns:
            str: 完整的提示词
        """
        prompt = f"""
        作为{professor_name}，请基于我们的单轮深入讨论，撰写一份关于"{topic}"的完整学术综述报告。
        
        虽然我们只进行了一轮讨论，但已经深入探讨了核心子话题"{subtopics[0]}"。请确保综述全面且有深度。

        综述主题: {topic}

        请严格按照以下结构组织内容：
        1. 摘要 (200字左右)：概述研究目的、方法、主要发现和结论
        2. 引言 (500字左右)：阐述研究背景、意义、范围和综述组织结构
        3. 主体内容：详细展开对"{subtopics[0]}"的全面分析，包括基础概念、研究现状、核心技术方法、挑战与解决方案
        4. 未来研究方向 (400字左右)：明确指出3-5个具体的未来研究方向
        5. 结论 (300字左右)：总结主要发现和贡献

        基于我们的讨论内容：
        {"".join(review_sections)}

        请确保综述质量不受轮次限制影响，内容全面且深入。
        """
        return prompt
    
    def _build_multi_round_review_prompt(self, professor_name, topic, subtopics, effective_rounds, review_sections):
        """构建多轮讨论的综述提示词
        
        Args:
            professor_name: 教授姓名
        ic: 主题
            subtopics: 子话题列表
            effective_rounds: 有效讨论轮次
            review_sections: 各部分综述内容
            
        Returns:
            str: 完整的提示词
        """
        prompt = f"""
        作为{professor_name}，请整合我们经过{effective_rounds}轮讨论的成果，撰写一份关于"{topic}"的完整学术综述报告。

        综述主题: {topic}

        您需要严格按照以下结构组织内容：
        1. 摘要 (200字左右)：概述研究目的、方法、主要发现和结论
        2. 引言 (500字左右)：阐述研究背景、意义、范围和综述组织结构
        3. 子话题详细内容：每个子话题需包含关键概念、研究现状、核心技术和方法、挑战与解决方案
        {", ".join([f"- {subtopic}" for subtopic in subtopics])}
        4. 未来研究方向 (400字左右)：明确指出3-5个具体的未来研究方向
        5. 结论 (300字左右)：总结主要发现和贡献

        基于我们的讨论内容：
        {"".join(review_sections)}

        请确保：
        - 内容深度：每个子话题分析要深入具体，包含实际研究成果和技术方法
        - 逻辑结构：各部分之间有清晰的逻辑关系和过渡
        - 学术严谨：准确引用文献，使用学术语言
        - 焦点聚焦：所有内容必须严格围绕"{topic}"，避免离题讨论

        下面是一个示例,对于不同的实例,可以参考这个结构：
        对于面向智能体的记忆管理系统，可以分为四个主要的子话题：
        记忆的理论模型、记忆存储技术、记忆查询与检索方法、典型应用案例 

        具体的要求可以是：
        记忆的理论模型：
        - 比较认知科学中的记忆模型与智能体中的实现方式
        - 分析短期、工作、长期记忆的区别与联系
        - 评述注意力机制与记忆检索的关系
    
        记忆存储技术 ：
        - 分析向量数据库、图数据库等存储技术的优缺点
        - 讨论记忆表征方法(嵌入向量、符号化等)
        - 评估不同存储技术的扩展性和效率
        
        记忆查询与检索方法：
        - 分析语义相似度搜索、关键词匹配等检索方法
        - 讨论记忆的压缩与摘要技术
        - 评述上下文窗口管理与记忆检索的协同方式
        
        典型应用案例：
        - 分析至少3个使用记忆管理系统的智能体实例
        - 评估不同应用场景中记忆系统的关键设计差异
        - 讨论记忆系统在实际应用中的挑战与解决方案

        最终报告应当是高质量学术综述，展现该领域的系统性研究进展。
        """
        return prompt 
    
    def _generate_research_tasks_from_review(self, topic, professor_id, review_result):
        """从综述报告生成研究任务
        
        Args:
            topic: 主题
            professor_id: 教授ID
            review_result: 综述报告结果字典
        """
        self.message_stream.system_message("从综述报告生成研究任务...")
        
        future_research_prompt = f"""
        请分析以下"{topic}"综述报告的未来研究方向，提取出3-5个具体的研究任务。
        
        综述报告内容:
        {review_result.get('content', '')}
        
        请列出每个任务的:
        1. 任务标题
        2. 详细描述
        3. 预期成果
        4. 优先级(高/中/低)
        
        以任务列表形式返回。
        """
        
        messages = [
            {"role": "system", "content": "你是一名能从学术综述中提取研究任务的专家。"},
            {"role": "user", "content": future_research_prompt}
        ]
        
        tasks_text = self.backend.llm.get_response(messages)
        
        if tasks_text:
            report_id = review_result.get("report_id")
            
            # 将综述报告的未来研究方向作为总结节点添加
            summary_info = {
                "Title": f"{topic}研究规划",
                "Content": tasks_text,
                "Author": professor_id,
                "Type": "research_plan",
                "Topic": topic,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            parent_keys = [professor_id]
            if report_id:
                parent_keys.append(report_id)
            
            summary_id = f"summary_{uuid.uuid4().hex[:8]}"
            self.backend.add_summary(summary_id, summary_info, parent_keys)
            self.message_stream.system_message(f"已创建研究规划: {summary_info['Title']}")
            
            # 从研究规划中提取任务
            task_ids = self.generate_tasks_from_summary(summary_id)
            print(f"从综述报告生成了研究规划和{len(task_ids)}个研究任务")
            self.message_stream.system_message(f"从综述报告生成了研究规划和{len(task_ids)}个研究任务")

if __name__ == "__main__":
    # 测试 _extract_and_assign_tasks 函数
    print("开始测试任务提取和分配功能...")
    
    # 创建会议系统实例
    system = AcademicMeetingSystem(use_remote_llm=True)
    
    # 创建测试用的教授和学生
    professor_id = "professor_1"
    system.create_academic_agent(
        agent_id=professor_id,
        nickname="张教授",
        age=45,
        role_type="教授",
        specialty=["人工智能", "机器学习"],
        personality="专业、严谨"
    )
    
    student_ids = []
    for i in range(3):
        student_id = f"student_{i+1}"
        student_ids.append(student_id)
        system.create_academic_agent(
            agent_id=student_id,
            nickname=f"学生{i+1}",
            age=25 + i,
            role_type="博士生" if i < 2 else "硕士生",
            specialty=["深度学习", "强化学习"] if i < 2 else ["计算机视觉"],
            personality="求知好学"
        )
    
    # 创建测试场景
    topic = "人工智能在教育领域的应用"
    scene_id = system.create_scene(f"{topic}研讨会", f"讨论{topic}相关研究")
    
    # 初始化任务管理器
    task_manager = {
        'all_tasks': [],  # 存储所有任务，格式：(轮次, 学生ID, 任务描述, 状态)
        'by_student': {},  # 按学生ID组织，格式：{student_id: [(轮次, 任务描述, 状态)]}
        'by_round': {}     # 按轮次组织，格式：{round_num: [(学生ID, 任务描述, 状态)]}
    }
    
    # 测试场景1：_professor_adjusts_framework + _extract_and_assign_tasks
    print("\n测试场景1: _professor_adjusts_framework + _extract_and_assign_tasks")
    
    # 模拟教授调整框架的文本，包含任务分配
    framework_text = """
    根据同学们的反馈，我们对研究框架做如下调整：
    
    1. 更加关注教育场景下的实际应用案例分析
    2. 增加对伦理和隐私问题的讨论
    3. 加强技术实现与教育理论的结合
    
    针对第一轮讨论，我为大家分配以下任务：
    
    1. **学生1**: 调研AI在个性化学习中的应用案例，找出3-5个成功案例，分析其实施方法、技术选型和实际效果，形成一份2-3页的简报。
    2. **学生2**: 分析当前AI教育应用面临的技术挑战，包括模型精度、实时性、可解释性等方面，并结合具体教育场景进行说明，总结目前主流的解决思路。
    3. **学生3**: 收集并整理国内外教育机构采用AI技术的现状数据，对比不同地区、不同教育阶段的应用差异，探讨背后的原因和启示。
    
    请大家在下次讨论前完成任务并准备分享。
    """
    
    # 调用函数进行任务提取和分配
    tasks_count1 = system._extract_and_assign_tasks(framework_text, student_ids, 0, task_manager)
    
    print(f"场景1提取到 {tasks_count1} 个任务")
    print("按学生组织的任务:")
    for student_id, tasks in task_manager['by_student'].items():
        student_name = system.backend.agents[student_id]["Nickname"]
        print(f"  {student_name}:")
        for round_num, task_desc, status in tasks:
            print(f"    - 轮次 {round_num}: {task_desc[:50]}... ({status})")
    
    # 测试场景2：_generate_round_minutes_and_tasks + _extract_and_assign_tasks
    print("\n测试场景2: _generate_round_minutes_and_tasks + _extract_and_assign_tasks")
    
    # 模拟会议纪要，包含任务分配
    minutes_text = """
    第1轮"AI教育个性化学习"讨论要点：
    
    1. 个性化学习是AI在教育中最具潜力的应用方向
    2. 当前技术已经可以根据学生学习行为进行实时调整
    3. 存在数据隐私和算法公平性的挑战
    4. 需要更多跨学科合作开发适合教育场景的AI模型
    
    达成共识：
    个性化学习将是未来教育的主要发展方向，但技术实现需要更加注重教育理论指导。
    
    待解决问题：
    如何平衡技术可能性与教育实际需求，减少过度技术化倾向。
    
    下一轮任务分配：
    
    1. **学生1**: 调研国际上领先的AI教育平台（如Knewton、ALEKS等）的技术架构和算法选择，分析其个性化推荐机制的工作原理和效果评估方法。
    2. **学生2**: 综合分析AI教育应用中的数据隐私保护措施，对比不同国家和地区的相关法规要求，提出适合教育场景的数据安全框架建议。
    3. **学生3**: 探索AI与教育心理学理论的结合点，特别是认知负荷理论、建构主义等如何指导AI教育产品设计，提出理论驱动的技术实现思路。
    """
    
    # 调用函数进行任务提取和分配
    tasks_count2 = system._extract_and_assign_tasks(minutes_text, student_ids, 1, task_manager)
    
    print(f"场景2提取到 {tasks_count2} 个任务")
    print("按轮次组织的任务:")
    for round_num, tasks in task_manager['by_round'].items():
        print(f"  第{round_num}轮:")
        for student_id, task_desc, status in tasks:
            student_name = system.backend.agents[student_id]["Nickname"]
            print(f"    - {student_name}: {task_desc[:50]}... ({status})")
    
    # 显示所有任务
    print("\n所有任务列表:")
    for round_num, student_id, task_desc, status in task_manager['all_tasks']:
        student_name = system.backend.agents[student_id]["Nickname"]
        print(f"  轮次{round_num} - {student_name}: {task_desc[:50]}... ({status})")
    
    # 测试任务完成标记
    print("\n测试任务完成标记:")
    if task_manager['all_tasks']:
        # 选择第一个任务进行完成标记测试
        test_round, test_student_id, test_task, _ = task_manager['all_tasks'][0]
        print(f"将标记任务为已完成: 轮次{test_round} - 学生{system.backend.agents[test_student_id]['Nickname']}")
        
        # 标记任务完成
        system._mark_task_as_completed(test_student_id, test_round, test_task, task_manager)
        
        # 验证任务状态
        for round_num, student_id, task_desc, status in task_manager['all_tasks']:
            if round_num == test_round and student_id == test_student_id and task_desc == test_task:
                print(f"  任务状态: {status}")
    
    print("\n测试完成")