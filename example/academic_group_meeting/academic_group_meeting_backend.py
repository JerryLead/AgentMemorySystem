import faiss
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import re
import requests
from deepseek_client import deepseek_remote, deepseek_local
from sentence_transformers import SentenceTransformer
import uuid
import time
from datetime import datetime
from typing import List, Dict, Set, Tuple, Any
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from semantic_data_structure.data_type import BaseDataType
from semantic_data_structure.semantic_map import BaseSemanticMap
from semantic_data_structure.semantic_simple_graph import BaseSemanticSimpleGraph
from enum import auto
import numpy as np

# from academic_group_meeting_backend import AcademicMeetingSystem
from semantic_data_structure.semantic_map import MilvusDialogueStorage
from semantic_data_structure.semantic_map import Neo4jInterface
from matplotlib import pyplot as plt
from academic_group_meeting_graph import *
from AttachmentSearchEngine import AttachmentSearchEngine
from LocalAttachmentProcessor import LocalAttachmentProcessor


# plt.rc('SimHei') # 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入部分保持不变...


class AcademicGroupMeetingDataSystem:
    def __init__(
        self,
        use_remote_llm: bool = True,  # 将 use_remote 改为 use_remote_llm
        use_local_embeddings: bool = True,
        local_text_model_path: str = "/home/zyh/model/clip-ViT-B-32-multilingual-v1",
        local_image_model_path: str = "/home/zyh/model/clip-ViT-B-32",
    ):
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
            local_image_model_path=local_image_model_path,
        )
        self.semantic_graph = AcademicGroupMeetingGraph(self.semantic_map)
        self.agents = {}  # {agent_id: agent_info}
        # 根据参数使用远程或本地 deepseek 客户端
        self.llm = deepseek_remote() if use_remote_llm else deepseek_local()

    # def __init__(self, use_remote: bool = True):
    #     self.semantic_map = AcademicGroupMeetingMap()
    #     self.semantic_graph = AcademicGroupMeetingGraph(self.semantic_map)
    #     self.agents = {}  # {agent_id: agent_info}
    #     # 根据参数使用远程或本地 deepseek 客户端
    #     self.llm = deepseek_remote() if use_remote else deepseek_local()

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
        print(
            f"创建了智能体 {agent_info.get('Nickname', agent_id)}，类型：{agent_type.name}"
        )

    def add_event(self, event_id: str, event_info: dict, parent_keys: List[str] = None):
        """
        添加事件节点，学术讨论类型
        """
        self.semantic_graph.add_node(
            event_id, event_info, AcademicDataType.Discussion, parent_keys
        )

    def add_conversation(
        self,
        conversation_id: str,
        conversation_info: dict,
        parent_keys: List[str] = None,
    ):
        """
        添加对话节点，根据内容可能是问题或讨论
        """
        # 判断是问题还是普通讨论
        content = conversation_info.get("Content", "").lower()
        if (
            "?" in content
            or "？" in content
            or "问题" in content
            or "question" in content
        ):
            conv_type = AcademicDataType.Question
        else:
            conv_type = AcademicDataType.Discussion

        self.semantic_graph.add_node(
            conversation_id, conversation_info, conv_type, parent_keys
        )

    def add_research_topic(
        self, topic_id: str, topic_info: dict, parent_keys: List[str] = None
    ):
        """
        添加研究主题节点
        """
        self.semantic_graph.add_node(
            topic_id, topic_info, AcademicDataType.ResearchTopic, parent_keys
        )

    def add_paper(self, paper_id: str, paper_info: dict, parent_keys: List[str] = None):
        """
        添加论文节点
        """
        self.semantic_graph.add_node(
            paper_id, paper_info, AcademicDataType.Paper, parent_keys
        )

    def add_conclusion(
        self, conclusion_id: str, conclusion_info: dict, parent_keys: List[str] = None
    ):
        """
        添加结论节点
        """
        self.semantic_graph.add_node(
            conclusion_id, conclusion_info, AcademicDataType.Conclusion, parent_keys
        )

    def query(self, query_text: str, k: int = 5):
        """
        查询 SemanticMap 返回最相似的节点
        """
        return self.semantic_graph.retrieve_similar_nodes(query_text, k)

    def visualize_graph(self, filename="academic_meeting_graph.png", fontpath=None):
        """可视化学术组会图

        Args:
            filename: 输出文件名
            fontpath: 字体路径，用于解决中文显示问题
        """
        self.semantic_graph.visualize_academic_meeting(
            "学术组会关系图", fontpath=fontpath
        )

    # def visualize_graph(self, filename="academic_meeting_graph.png", fontpath=None):
    #     """可视化学术组会图"""
    #     self.semantic_graph.visualize_academic_meeting("学术组会关系图", fontpath)

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
            {"role": "user", "content": prompt},
        ]
        response = self.llm.get_response(messages)
        if response:
            # 将自动生成的回复作为对话节点插入
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            conv_info = {
                "Content": response,
                "Speaker": speaker_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Participants": [speaker_id],
            }
            # 判断是问题还是普通讨论
            if "?" in response or "？" in response or "question" in response.lower():
                self.semantic_graph.add_node(
                    conversation_id, conv_info, AcademicDataType.Question, [speaker_id]
                )
            else:
                self.semantic_graph.add_node(
                    conversation_id,
                    conv_info,
                    AcademicDataType.Discussion,
                    [speaker_id],
                )

            return response
        else:
            return "No response generated."

    def generate_event(self, description: str) -> str:
        """
        基于输入描述调用 LLM API 自动生成事件信息，并作为学术讨论节点插入语义图。
        """
        prompt = f"Generate a detailed academic discussion description based on the following input: {description}\nDiscussion:"
        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant specialized in academic discussions.",
            },
            {"role": "user", "content": prompt},
        ]
        event_response = self.llm.get_response(messages)
        if event_response:
            event_id = f"discussion_{uuid.uuid4().hex[:8]}"
            event_info = {
                "Content": event_response,
                "Topic": description,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Participants": [],
            }
            self.semantic_graph.add_node(
                event_id, event_info, AcademicDataType.Discussion
            )
            print(f"生成的讨论 [{event_id}]: {event_response}")
            return event_id
        else:
            print("生成讨论失败。")
            return None


# 确保导入所需模块


class AcademicMeetingSystem:
    """学术组会系统，独立实现，使用AcademicGroupMeetingDataSystem"""

    def __init__(
        self,
        use_remote_llm: bool = True,
        use_local_embeddings: bool = True,
        local_text_model_path: str = "/home/zyh/model/clip-ViT-B-32-multilingual-v1",
        local_image_model_path: str = "/home/zyh/model/clip-ViT-B-32",
    ):
        """
        初始化学术组会系统

        Args:
            use_remote_llm: 是否使用远程大语言模型服务
            use_local_embeddings: 是否使用本地嵌入模型
            local_text_model_path: 本地文本嵌入模型路径
            local_image_model_path: 本地图像嵌入模型路径
        """
        # 使用学术组会专用的数据系统，传递本地模型参数
        self.backend = AcademicGroupMeetingDataSystem(
            use_remote_llm=use_remote_llm,  # 改为统一的参数名
            use_local_embeddings=use_local_embeddings,
            local_text_model_path=local_text_model_path,
            local_image_model_path=local_image_model_path,
        )
        self.search_engine = AcademicSearchEngine()
        self.academic_roles = {}  # agent_id -> AcademicRole
        self.scenes = {}  # 所有对话场景
        self.active_scene = None  # 当前活跃场景
        self.agent_personalities = {}  # 存储智能体个性设置

    # def __init__(self, use_remote: bool = True):
    #     # 使用学术组会专用的数据系统
    #     self.backend = AcademicGroupMeetingDataSystem(use_remote)
    #     self.search_engine = AcademicSearchEngine()
    #     self.academic_roles = {}  # agent_id -> AcademicRole
    #     self.scenes = {}  # 所有对话场景
    #     self.active_scene = None  # 当前活跃场景
    #     self.agent_personalities = {}  # 存储智能体个性设置

    def create_academic_agent(
        self,
        agent_id: str,
        nickname: str,
        age: int,
        role_type: str,
        specialty: List[str],
        personality: str,
    ):
        """创建学术智能体，包括教授、博士生、硕士生等角色"""
        agent_info = {
            "Nickname": nickname,
            "Age": age,
            "Occupation": role_type,
            "Specialty": ", ".join(specialty),
            "Personality": personality,
            "Background": f"{role_type}，专精于{', '.join(specialty)}，{personality}",
        }

        # 创建学术角色
        academic_role = AcademicRole(role_type, specialty, personality)
        self.academic_roles[agent_id] = academic_role

        # 创建智能体并设置个性
        self.create_agent(agent_id, agent_info, academic_role.get_role_prompt())
        print(
            f"创建了学术智能体 {nickname}（{role_type}），专业领域：{', '.join(specialty)}"
        )

    def create_agent(self, agent_id: str, agent_info: dict, personality: str = None):
        """创建一个智能体，可以指定其个性特征"""
        # 创建后端智能体
        self.backend.create_agent(agent_id, agent_info)
        # 记录个性设置
        if personality:
            self.agent_personalities[agent_id] = personality
        print(
            f"创建了智能体 {agent_id}（{agent_info.get('Nickname', '无名')}），个性：{personality or '未定义'}"
        )

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
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "Participants": [speaker_id],
        }

        parent_keys = [speaker_id, f"topic_{scene_id}"]
        # 如果有前一条消息，也将其作为父节点
        if len(scene.messages) > 1:
            prev_msg_id = scene.messages[-2]["id"]
            parent_keys.append(prev_msg_id)

        self.backend.add_conversation(message_id, conversation_info, parent_keys)
        return message_id

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
            {"role": "user", "content": prompt},
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
            "Participants": participants,
        }

        # 添加结论到语义图，并与场景和参与者建立关联
        parent_keys = [f"topic_{scene_id}"] + participants
        self.backend.add_conclusion(conclusion_id, conclusion_info, parent_keys)

        print(f"从讨论生成的结论: {conclusion_text}")
        return conclusion_id

    def visualize_conversation_graph(self, scene_id: str = None, fontpath=None):
        """使用AcademicGroupMeetingGraph的可视化功能

        Args:
            scene_id: 场景ID，None则使用当前活跃场景
            fontpath: 字体路径，用于解决中文显示问题
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
                "Participants": [speaker_id],
            }

            # 检查节点是否已存在
            self.backend.semantic_graph._ensure_node(msg_id)

            # 重新构建连接
            for parent_key in parent_keys:
                if parent_key in self.backend.semantic_graph.graph_relations:
                    self.backend.semantic_graph._ensure_node(parent_key)
                    self.backend.semantic_graph.graph_relations[parent_key]["children"][
                        msg_id
                    ] = "发言"
                    self.backend.semantic_graph.graph_relations[msg_id]["parents"][
                        parent_key
                    ] = "属于"

            prev_msg_id = msg_id

        # 更新子图结构
        self.backend.semantic_graph.auto_generate_subgraphs()

        # 使用学术组会专用的可视化方法，传递字体参数
        self.backend.visualize_graph(
            filename=f"academic_meeting_{scene_id}.png",
            fontpath=(
                fontpath
                if fontpath
                else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ),
        )

        print(f"学术组会图已保存为 academic_meeting_{scene_id}.png")

    # def visualize_conversation_graph(self, scene_id: str = None):
    #     """使用AcademicGroupMeetingGraph的可视化功能"""
    #     if not scene_id:
    #         scene_id = self.active_scene

    #     if scene_id not in self.scenes:
    #         print(f"场景 {scene_id} 不存在")
    #         return

    #     # 确保在生成图形前所有节点已正确添加到图中
    #     scene = self.scenes[scene_id]

    #     # 重新构建关系图以确保数据完整性
    #     topic_node = f"topic_{scene_id}"

    #     # 确保添加所有消息关系
    #     prev_msg_id = None
    #     for msg in scene.messages:
    #         msg_id = msg["id"]
    #         speaker_id = msg["speaker_id"]

    #         # 确保消息节点存在且连接了适当的父节点
    #         parent_keys = [speaker_id, topic_node]
    #         if prev_msg_id:
    #             parent_keys.append(prev_msg_id)

    #         # 添加到图中（如果已存在不会重复添加）
    #         conversation_info = {
    #             "Content": msg["content"],
    #             "Speaker": speaker_id,
    #             "Topic": scene.name,
    #             "Timestamp": msg["timestamp"],
    #             "Participants": [speaker_id]
    #         }

    #         # 检查节点是否已存在
    #         self.backend.semantic_graph._ensure_node(msg_id)

    #         # 重新构建连接
    #         for parent_key in parent_keys:
    #             if parent_key in self.backend.semantic_graph.graph_relations:
    #                 self.backend.semantic_graph._ensure_node(parent_key)
    #                 self.backend.semantic_graph.graph_relations[parent_key]["children"][msg_id] = "发言"
    #                 self.backend.semantic_graph.graph_relations[msg_id]["parents"][parent_key] = "属于"

    #         prev_msg_id = msg_id

    #     # 更新子图结构
    #     self.backend.semantic_graph.auto_generate_subgraphs()

    #     # 使用学术组会专用的可视化方法
    #     self.backend.visualize_graph(
    #         filename=f"academic_meeting_{scene_id}.png",
    #         fontpath='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'  # 使用更通用的字体
    #     )

    #     print(f"学术组会图已保存为 academic_meeting_{scene_id}.png")

    # 在AcademicMeetingSystem类中添加以下方法

    def format_dialogue_history(self, history: List[Dict[str, Any]]) -> str:
        """格式化对话历史，用于创建提示"""
        formatted = []
        for msg in history:
            speaker_id = msg["speaker_id"]
            speaker_name = self.backend.agents.get(speaker_id, {}).get(
                "Nickname", speaker_id
            )
            formatted.append(f"{speaker_name}: {msg['content']}")
        return "\n".join(formatted)

    def agent_search_and_reply(
        self, scene_id: str, speaker_id: str, search_query: str, context_size: int = 5
    ) -> str:
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
                formatted_results += (
                    f"   摘要: {result.get('snippet', 'No snippet available')}\n"
                )
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
            {
                "role": "system",
                "content": "你是一个学术助手，能够模拟不同学术角色参与专业讨论。请基于检索结果提供专业、有深度的回复。",
            },
            {"role": "user", "content": prompt},
        ]

        # 调用 LLM 生成回复
        response = self.backend.llm.get_response(messages)

        if not response:
            return "无法生成回复。"

        # 添加到场景历史和语义图
        self.add_message(scene_id, speaker_id, response)

        return response

    def start_academic_meeting(
        self,
        scene_id: str,
        topic: str,
        moderator_id: str,
        rounds: int = 3,
        deep_search: bool = False,
    ):
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

        moderator_name = self.backend.agents.get(moderator_id, {}).get(
            "Nickname", moderator_id
        )

        print(f"\n--- 开始学术组会（主题：{topic}，主持人：{moderator_name}）---\n")

        # 主持人开场白
        opening_prompt = f"""
        作为{moderator_name}，你是这次学术组会的主持人。请提供一个组会开场白，介绍今天讨论的主题:
        主题: {topic}
        请简明扼要地介绍主题的背景、重要性，并鼓励大家积极参与讨论。
        """

        messages = [
            {"role": "system", "content": "你是一名学术组会的主持人，负责引导讨论。"},
            {"role": "user", "content": opening_prompt},
        ]

        opening = self.backend.llm.get_response(messages)
        if opening:
            self.add_message(scene_id, moderator_id, opening)
            print(f"{moderator_name}: {opening}\n")

        # 添加所有组会参与者到场景
        for agent_id in self.backend.agents:
            if agent_id not in scene.participants:
                # 让每个人简单介绍一下自己
                agent_name = self.backend.agents.get(agent_id, {}).get(
                    "Nickname", agent_id
                )
                intro_prompt = f"""
                作为{agent_name}，请简短介绍一下你自己以及你对"{topic}"的初步看法。
                不要超过3句话。
                """
                messages = [
                    {
                        "role": "system",
                        "content": "你是参加学术讨论的成员，需要简短介绍自己。",
                    },
                    {"role": "user", "content": intro_prompt},
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
                {
                    "role": "system",
                    "content": "你是一名学术组会的主持人，负责引导讨论。",
                },
                {"role": "user", "content": guide_prompt},
            ]

            guide = self.backend.llm.get_response(messages)
            if guide:
                self.add_message(scene_id, moderator_id, guide)
                print(f"{moderator_name}: {guide}\n")

            # 其他参与者依次发言，并进行网络检索
        for agent_id in self.backend.agents:
            if agent_id != moderator_id:
                agent_name = self.backend.agents.get(agent_id, {}).get(
                    "Nickname", agent_id
                )
                print(
                    f"\n{agent_name}正在查询相关资料..."
                    + ("(使用深度网页内容)" if deep_search else "")
                )

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
                        search_query, include_full_content=True
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
                            {
                                "role": "system",
                                "content": "你是一名学术会议的参与者，基于搜索分析结果发表专业见解。",
                            },
                            {"role": "user", "content": content_prompt},
                        ]

                        response = self.backend.llm.get_response(messages)
                        if response:
                            self.add_message(scene_id, agent_id, response)
                            print(f"{agent_name}: {response}\n")
                else:
                    # 使用原有的检索方法
                    response = self.agent_search_and_reply(
                        scene_id, agent_id, search_query
                    )
                    print(f"{agent_name}: {response}\n")
            # for agent_id in self.backend.agents:
            #     if agent_id != moderator_id:
            #         agent_name = self.backend.agents.get(agent_id, {}).get("Nickname", agent_id)
            #         print(f"\n{agent_name}正在查询相关资料...")

            #         # 根据参与者角色和当前讨论点生成检索关键词
            #         academic_role = self.academic_roles.get(agent_id)
            #         if academic_role and academic_role.specialty:
            #             specialty = academic_role.specialty[0]  # 使用第一个专业领域
            #             search_query = f"{topic} {discussion_point} {specialty}"
            #         else:
            #             search_query = f"{topic} {discussion_point}"

            #         # 检索并回复
            #         response = self.agent_search_and_reply(scene_id, agent_id, search_query)
            #         print(f"{agent_name}: {response}\n")

            # 主持人小结本轮讨论
            if round_num < rounds:
                summary_prompt = f"""
                作为{moderator_name}，请对第{round_num}轮关于"{discussion_point}"的讨论进行小结，
                并自然地过渡到下一轮讨论。
                """

                messages = [
                    {
                        "role": "system",
                        "content": "你是一名学术组会的主持人，负责引导和总结讨论。",
                    },
                    {"role": "user", "content": summary_prompt},
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
            {
                "role": "system",
                "content": "你是一名学术组会的主持人，负责总结讨论成果。",
            },
            {"role": "user", "content": conclusion_prompt},
        ]

        conclusion = self.backend.llm.get_response(messages)
        if conclusion:
            self.add_message(scene_id, moderator_id, conclusion)
            print(f"{moderator_name}: {conclusion}\n")

        print("\n--- 学术组会结束 ---\n")

        # 生成组会总结
        self.generate_event_from_conversation(scene_id)

    def run_custom_meeting(
        self,
        topic: str,
        professor_id: str = None,
        rounds: int = 3,
        deep_search: bool = False,
    ):
        """执行用户自定义主题的学术组会"""
        # 1. 创建场景
        scene_name = f"{topic}学术研讨会"
        scene_description = f"讨论主题：{topic}，探讨研究现状、方法技术与未来方向"
        scene_id = self.create_scene(scene_name, scene_description)

        # 处理本地附件文件
        print("\n处理本地附件...")
        local_attachments = self.process_local_attachments(
            topic_key=f"topic_{scene_id}"
        )
        if local_attachments["pdfs"] or local_attachments["images"]:
            print(
                f"处理了 {len(local_attachments['pdfs'])} 篇本地论文和 {len(local_attachments['images'])} 张本地图片"
            )

        # 搜索相关附件
        print("\n搜索与主题相关的论文和图片...")
        attachments = self.search_and_add_attachments(
            topic=topic, topic_key=f"topic_{scene_id}", download_files=True
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
        search_results = self.search_engine.search(
            f"{topic} research background latest developments"
        )

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
            {"role": "user", "content": intro_prompt},
        ]

        introduction = self.backend.llm.get_response(messages)
        if introduction:
            self.add_message(scene_id, professor_id, introduction)
            print(f"\n{professor_name}:\n{introduction}\n")

        # 4. 按轮次讨论
        for round_num in range(1, rounds + 1):
            print(f"\n--- 第 {round_num} 轮讨论 ---\n")

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
                {"role": "user", "content": guide_prompt},
            ]

            guidance = self.backend.llm.get_response(messages)
            if guidance:
                self.add_message(scene_id, professor_id, guidance)
                print(f"{professor_name}:\n{guidance}\n")

            # 学生依次发言 - 确保每个轮次中所有学生都能发言
            for student_id in students:
                student_name = self.backend.agents[student_id]["Nickname"]
                specialty = self.backend.agents[student_id].get("Specialty", "")
                print(
                    f"\n{student_name}正在搜索相关资料..."
                    + ("(使用深度网页内容)" if deep_search else "")
                )

                # 学生搜索信息
                search_query = f"{topic} {search_suffix} {specialty}"

                student_results = ""
                if deep_search:
                    try:
                        # 使用深度搜索并分析
                        search_analysis = self.search_engine.search_and_analyze(
                            search_query, include_full_content=True
                        )

                        if "analysis" in search_analysis:
                            student_results = search_analysis["analysis"]
                            print(f"获取了深度分析结果 ({len(student_results)} 字符)")
                        else:
                            # 回退到普通搜索
                            student_search = self.search_engine.search(search_query)
                            student_results = self._format_search_results(
                                student_search
                            )
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
                    {
                        "role": "system",
                        "content": "你是参与学术讨论的研究生，有自己的专业见解。",
                    },
                    {"role": "user", "content": student_prompt},
                ]

                response = self.backend.llm.get_response(messages)
                if response:
                    self.add_message(scene_id, student_id, response)
                    print(f"{student_name}:\n{response}\n")

                    # 教授简短回应每个学生（非必需，但更真实）
                    if (
                        round_num < rounds
                        or students.index(student_id) < len(students) - 1
                    ):
                        feedback_prompt = f"""
                        作为{professor_name}，请对{student_name}的发言给予简短的回应和点评。
                        
                        {student_name}的发言:
                        {response}
                        
                        请给予学术性的建设性反馈，提出1-2个思考点或补充，控制在100字以内。
                        """

                        messages = [
                            {
                                "role": "system",
                                "content": "你是指导学生讨论的教授，需要给予建设性的简短反馈。",
                            },
                            {"role": "user", "content": feedback_prompt},
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
                    {"role": "user", "content": round_summary_prompt},
                ]

                round_summary = self.backend.llm.get_response(messages)
                if round_summary:
                    self.add_message(scene_id, professor_id, round_summary)
                    print(f"\n{professor_name} (本轮总结):\n{round_summary}\n")

        # 5. 教授汇总整个讨论并提出解决方案
        print("\n--- 教授总结与方案提出 ---\n")

        # 获取历史讨论内容
        history = self.scenes[scene_id].get_history()
        formatted_history = self.format_dialogue_history(history[-20:])  # 最后20条消息

        # 搜索解决方案和最佳实践
        solution_search = self.search_engine.search(
            f"{topic} solutions best practices implementation"
        )
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
            {
                "role": "system",
                "content": "你是一位资深教授，正在为学术组会提供最终总结和解决方案。",
            },
            {"role": "user", "content": summary_prompt},
        ]

        final_summary = self.backend.llm.get_response(messages)
        if final_summary:
            self.add_message(scene_id, professor_id, final_summary)
            print(f"{professor_name}:\n{final_summary}\n")

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

    def search_and_add_attachments(
        self, topic: str, topic_key: str = None, download_files: bool = True
    ):
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
            download_files=download_files,
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
            backend=self.backend, topic_key=topic_key
        )

        # 如果添加了任何附件，重新生成图表
        if results["pdfs"] or results["images"]:
            # 更新语义图
            self.backend.semantic_graph.auto_generate_subgraphs()

            # 可选，在处理完成后生成新的可视化图
            if self.active_scene:
                self.visualize_conversation_graph(self.active_scene)

        return results


class MilvusEnabledAcademicMeetingSystem(AcademicMeetingSystem):
    """扩展学术组会系统，支持Milvus存储"""

    def __init__(
        self,
        use_remote_llm: bool = True,
        use_local_embeddings: bool = True,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_collection: str = "academic_dialogues",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "20031117",
        neo4j_database: str = "academicgraph",
        **kwargs,
    ):
        """初始化支持Milvus的学术组会系统

        Args:
            use_remote_llm: 是否使用远程大语言模型
            use_local_embeddings: 是否使用本地嵌入模型
            milvus_host: Milvus服务器地址
            milvus_port: Milvus服务器端口
            milvus_collection: Milvus集合名称
        """
        # 初始化原始学术组会系统
        super().__init__(
            use_remote_llm=use_remote_llm,
            use_local_embeddings=use_local_embeddings,
            **kwargs,
        )
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        # 初始化Milvus存储
        try:
            self.milvus_storage = MilvusDialogueStorage(
                host=milvus_host,
                port=milvus_port,
                collection_name=milvus_collection,
                embedding_dim=self.backend.semantic_map.embedding_dim,
            )
            self.use_milvus = True
            print(f"已启用Milvus对话存储: {milvus_collection}")
        except Exception as e:
            print(f"初始化Milvus存储失败: {str(e)}")
            print("将仅使用原始存储方式")
            self.use_milvus = False

        # 当前讨论轮次
        self.current_round = 0

    def add_message(self, scene_id: str, speaker_id: str, content: str):
        """重写添加消息方法，增加Milvus存储功能

        Args:
            scene_id: 场景ID
            speaker_id: 发言者ID
            content: 发言内容

        Returns:
            str: 消息ID
        """
        # 调用原始方法添加消息
        message_id = super().add_message(scene_id, speaker_id, content)

        # 如果启用了Milvus且添加消息成功
        if self.use_milvus and message_id:
            try:
                # 获取场景信息
                scene = self.scenes.get(scene_id)
                if not scene:
                    print(f"场景不存在: {scene_id}")
                    return message_id

                # 获取发言者信息
                speaker_info = self.backend.agents.get(speaker_id, {})
                speaker_nickname = speaker_info.get("Nickname", speaker_id)

                # 获取角色信息
                academic_role = self.academic_roles.get(speaker_id)
                role_type = academic_role.role_type if academic_role else ""
                specialty = (
                    ", ".join(academic_role.specialty)
                    if academic_role and hasattr(academic_role, "specialty")
                    else ""
                )

                # 确定消息类型
                message_type = (
                    "Question" if "?" in content or "？" in content else "Statement"
                )
                if "总结" in content or "结论" in content:
                    message_type = "Conclusion"

                # 获取时间戳
                timestamp = ""
                for msg in scene.messages:
                    if msg["id"] == message_id:
                        timestamp = msg["timestamp"]
                        break

                # 生成嵌入向量
                embedding = self.backend.semantic_map._get_text_embedding(content)

                # 插入到Milvus
                self.milvus_storage.insert_dialogue(
                    message_id=message_id,
                    scene_id=scene_id,
                    speaker_id=speaker_id,
                    speaker_nickname=speaker_nickname,
                    role_type=role_type,
                    specialty=specialty,
                    content=content,
                    topic=scene.name,
                    timestamp=timestamp,
                    round_num=self.current_round,
                    message_type=message_type,
                    embedding=embedding,
                )
            except Exception as e:
                print(f"将对话存储到Milvus时出错: {str(e)}")

        return message_id

    def start_academic_meeting(
        self,
        scene_id: str,
        topic: str,
        moderator_id: str,
        rounds: int = 3,
        deep_search: bool = False,
    ):
        """重写会议开始方法，增加轮次跟踪"""
        # 重置轮次计数
        self.current_round = 0
        # 调用原方法
        super().start_academic_meeting(
            scene_id, topic, moderator_id, rounds, deep_search
        )

    def run_custom_meeting(
        self,
        topic: str,
        professor_id: str = None,
        rounds: int = 3,
        deep_search: bool = False,
    ):
        """重写自定义会议方法，增加轮次跟踪"""
        # 重置轮次计数
        self.current_round = 0
        # 调用原方法
        return super().run_custom_meeting(topic, professor_id, rounds, deep_search)

    def search_similar_dialogues(self, query_text: str, top_k: int = 5):
        """搜索与查询文本相似的对话

        Args:
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 相似对话列表
        """
        if not self.use_milvus:
            print("Milvus存储未启用，无法搜索")
            return []

        try:
            # 生成查询文本的嵌入向量
            query_embedding = self.backend.semantic_map._get_text_embedding(query_text)

            # 搜索相似对话
            return self.milvus_storage.search_similar_dialogues(query_embedding, top_k)
        except Exception as e:
            print(f"搜索相似对话失败: {str(e)}")
            return []

    def search_dialogues_by_topic(self, topic: str, top_k: int = 10):
        """按主题搜索对话

        Args:
            topic: 主题关键词
            top_k: 返回结果数量

        Returns:
            List[Dict]: 对话列表
        """
        if not self.use_milvus:
            print("Milvus存储未启用，无法搜索")
            return []

        return self.milvus_storage.search_by_topic(topic, top_k)

    def search_dialogues_by_speaker(self, speaker_id: str, top_k: int = 10):
        """按发言者搜索对话

        Args:
            speaker_id: 发言者ID
            top_k: 返回结果数量

        Returns:
            List[Dict]: 对话列表
        """
        if not self.use_milvus:
            print("Milvus存储未启用，无法搜索")
            return []

        return self.milvus_storage.search_by_speaker(speaker_id, top_k)

    def close_milvus(self):
        """关闭Milvus连接"""
        if self.use_milvus:
            self.milvus_storage.close()
            print("已关闭Milvus连接")

    # def export_to_neo4j(self,
    #                 # neo4j_uri: str = "bolt://localhost:7687",
    #                 # neo4j_user: str = "neo4j",
    #                 # neo4j_password: str = "password",
    #                 # neo4j_database: str = "academicgraph",
    #                 create_constraints: bool = True):
    #     """将语义图中的节点和关系导出到Neo4j图数据库

    #     Args:
    #         neo4j_uri: Neo4j数据库URI
    #         neo4j_user: Neo4j用户名
    #         neo4j_password: Neo4j密码
    #         create_constraints: 是否创建唯一性约束

    #     Returns:
    #         Dict: 包含导出统计信息的字典
    #     """
    #     try:
    #         # 初始化Neo4j图存储
    #         neo4j_graph = Neo4jInterface(
    #             uri=self.neo4j_uri,
    #             user=self.neo4j_user,
    #             password=self.neo4j_password,
    #             database=self.neo4j_database
    #         )

    #         print(f"已连接Neo4j数据库: {self.neo4j_uri}")

    #         # 创建唯一性约束以提高性能
    #         if create_constraints:
    #             constraints = [
    #                 "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
    #                 "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    #                 "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
    #                 "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conclusion) REQUIRE c.id IS UNIQUE"
    #             ]

    #             for constraint in constraints:
    #                 neo4j_graph.execute_query(constraint)
    #             print("已创建Neo4j唯一性约束")

    #         # 获取语义图中的所有节点
    #         nodes = {}
    #         edges = []
    #         node_types_map = {}
    #         node_data = {}

    #         # 统计信息
    #         stats = {
    #             "nodes_total": 0,
    #             "nodes_by_type": {},
    #             "relations_total": 0,
    #             "relations_by_type": {}
    #         }

    #         # 从语义图中收集节点和边信息
    #         for key in self.backend.semantic_graph.graph_relations:
    #             # 获取节点数据和类型
    #             node_info = None
    #             for k, value, datatype, _ in self.backend.semantic_map.data:
    #                 if k == key:
    #                     node_info = {
    #                         "id": key,
    #                         "value": value,
    #                         "datatype": datatype
    #                     }
    #                     node_types_map[key] = datatype
    #                     node_data[key] = value
    #                     break

    #             if node_info:
    #                 nodes[key] = node_info

    #             # 获取边信息
    #             for child_key, relation_type in self.backend.semantic_graph.graph_relations[key]["children"].items():
    #                 edges.append({
    #                     "source": key,
    #                     "target": child_key,
    #                     "relation": relation_type
    #                 })

    #         # 转换节点类型为Neo4j标签
    #         for node_id, node_info in nodes.items():
    #             datatype = node_info.get("datatype")
    #             value = node_info.get("value", {})

    #             if datatype == AcademicDataType.Professor:
    #                 node_label = "Professor"
    #                 label_group = "Person"
    #             elif datatype == AcademicDataType.PhD:
    #                 node_label = "PhD"
    #                 label_group = "Person"
    #             elif datatype == AcademicDataType.Master:
    #                 node_label = "Master"
    #                 label_group = "Person"
    #             elif datatype == AcademicDataType.ResearchTopic:
    #                 node_label = "ResearchTopic"
    #                 label_group = "Topic"
    #             elif datatype == AcademicDataType.Question:
    #                 node_label = "Question"
    #                 label_group = "Message"
    #             elif datatype == AcademicDataType.Discussion:
    #                 node_label = "Discussion"
    #                 label_group = "Message"
    #             elif datatype == AcademicDataType.Conclusion:
    #                 node_label = "Conclusion"
    #                 label_group = "Conclusion"
    #             elif datatype == AcademicDataType.Paper:
    #                 node_label = "Paper"
    #                 label_group = "Document"
    #             else:
    #                 node_label = "Unknown"
    #                 label_group = "Other"

    #             # 准备节点属性
    #             node_props = {
    #                 "id": node_id,
    #                 "label_type": node_label
    #             }

    #             # 添加特有属性
    #             if isinstance(value, dict):
    #                 for k, v in value.items():
    #                     if isinstance(v, (str, int, float, bool)) or v is None:
    #                         node_props[k] = v
    #                     elif isinstance(v, list):
    #                         try:
    #                             # 尝试将列表转换为字符串
    #                             node_props[k] = ", ".join(str(item) for item in v)
    #                         except:
    #                             # 如果失败，忽略该属性
    #                             pass

    #             # 添加可读标签
    #             if "Nickname" in value:
    #                 node_props["name"] = value["Nickname"]
    #             elif "Title" in value:
    #                 node_props["name"] = value["Title"]
    #             elif "Content" in value:
    #                 content = value["Content"]
    #                 node_props["name"] = content[:50] + "..." if len(content) > 50 else content

    #             # 创建Neo4j节点
    #             neo4j_graph.create_node(node_id, [label_group, node_label], node_props)

    #             # 更新统计信息
    #             stats["nodes_total"] += 1
    #             if node_label not in stats["nodes_by_type"]:
    #                 stats["nodes_by_type"][node_label] = 0
    #             stats["nodes_by_type"][node_label] += 1

    #         # 创建Neo4j关系
    #         for edge in edges:
    #             source_id = edge["source"]
    #             target_id = edge["target"]
    #             relation = edge["relation"]

    #             # 处理关系类型，确保符合Neo4j标准
    #             relation_type = relation.replace(" ", "_").upper()

    #             # 创建Neo4j关系
    #             neo4j_graph.create_relationship(
    #                 source_id=source_id,
    #                 target_id=target_id,
    #                 relationship_type=relation_type,
    #                 properties={"weight": 1.0}  # 可以根据需求添加权重或其他属性
    #             )

    #             # 更新统计信息
    #             stats["relations_total"] += 1
    #             if relation_type not in stats["relations_by_type"]:
    #                 stats["relations_by_type"][relation_type] = 0
    #             stats["relations_by_type"][relation_type] += 1

    #         # 输出导入结果摘要
    #         print(f"Neo4j导入完成 - 总节点: {stats['nodes_total']}, 总关系: {stats['relations_total']}")
    #         print("节点类型统计:")
    #         for node_type, count in stats["nodes_by_type"].items():
    #             print(f"  - {node_type}: {count}")

    #         print("关系类型统计:")
    #         for rel_type, count in stats["relations_by_type"].items():
    #             print(f"  - {rel_type}: {count}")

    #         # 示例查询
    #         example_queries = [
    #             "MATCH (p:Professor)-[r]->(m:Message) RETURN p.name AS Professor, TYPE(r) as Relation, m.name AS Message LIMIT 5",
    #             "MATCH (p:Person)-[:发言]->(d:Discussion) RETURN p.name AS Person, d.name AS Discussion LIMIT 5",
    #             "MATCH path = (t:Topic)-[*1..3]->(c:Conclusion) RETURN path LIMIT 3"
    #         ]

    #         print("\nNeo4j示例查询:")
    #         for i, query in enumerate(example_queries, 1):
    #             print(f"查询{i}: {query}")

    #         return stats

    #     except Exception as e:
    #         print(f"导出到Neo4j失败: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         return {"error": str(e)}

    def export_to_neo4j(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        create_constraints: bool = True,
    ):
        """将语义图中的节点和关系导出到Neo4j图数据库，支持命名空间结构

        Args:
            neo4j_uri: Neo4j数据库URI，默认使用实例化时设置的值
            neo4j_user: Neo4j用户名，默认使用实例化时设置的值
            neo4j_password: Neo4j密码，默认使用实例化时设置的值
            neo4j_database: Neo4j数据库名称，默认使用实例化时设置的值
            create_constraints: 是否创建唯一性约束

        Returns:
            Dict: 包含导出统计信息的字典
        """
        try:
            # 使用传入的参数，如果没有则使用实例化时的默认值
            uri = neo4j_uri if neo4j_uri is not None else self.neo4j_uri
            user = neo4j_user if neo4j_user is not None else self.neo4j_user
            password = (
                neo4j_password if neo4j_password is not None else self.neo4j_password
            )
            database = (
                neo4j_database if neo4j_database is not None else self.neo4j_database
            )

            # 初始化Neo4j图存储
            neo4j_graph = Neo4jInterface(
                uri=uri, user=user, password=password, database=database
            )

            print(f"已连接Neo4j数据库: {uri}")

            # 创建唯一性约束以提高性能
            if create_constraints:
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conclusion) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Namespace) REQUIRE n.id IS UNIQUE",
                ]

                for constraint in constraints:
                    neo4j_graph.execute_query(constraint)
                print("已创建Neo4j唯一性约束")

            # 更新命名空间子图确保数据最新
            self.backend.semantic_graph.auto_generate_subgraphs()

            # 统计信息
            stats = {
                "nodes_total": 0,
                "nodes_by_type": {},
                "nodes_by_namespace": {},
                "relations_total": 0,
                "relations_by_type": {},
                "cross_namespace_relations": 0,
            }

            # 先创建命名空间节点
            for namespace in NamespaceType:
                namespace_id = f"namespace_{namespace.name}"
                neo4j_graph.create_node(
                    namespace_id,
                    ["Namespace"],
                    {
                        "id": namespace_id,
                        "name": namespace.name,
                        "description": f"{namespace.name} namespace",
                    },
                )
                print(f"创建命名空间节点: {namespace.name}")

                # 添加命名空间节点到统计
                stats["nodes_total"] += 1
                if "Namespace" not in stats["nodes_by_type"]:
                    stats["nodes_by_type"]["Namespace"] = 0
                stats["nodes_by_type"]["Namespace"] += 1

            # 从语义图中获取所有节点并创建
            for key in self.backend.semantic_graph.graph_relations:
                # 获取节点数据和类型
                node_info = None
                for k, value, datatype, _ in self.backend.semantic_map.data:
                    if k == key:
                        node_info = {"id": key, "value": value, "datatype": datatype}
                        break

                if not node_info:
                    continue

                datatype = node_info.get("datatype")
                value = node_info.get("value", {})

                # 获取命名空间
                namespace = self.backend.semantic_graph.datatype_namespace_map.get(
                    datatype
                )
                namespace_name = namespace.name if namespace else "UNKNOWN"

                # 确定节点标签
                if datatype == AcademicDataType.Professor:
                    node_label = "Professor"
                    label_group = "Person"
                elif datatype == AcademicDataType.PhD:
                    node_label = "PhD"
                    label_group = "Person"
                elif datatype == AcademicDataType.Master:
                    node_label = "Master"
                    label_group = "Person"
                elif datatype == AcademicDataType.ResearchTopic:
                    node_label = "ResearchTopic"
                    label_group = "Topic"
                elif datatype == AcademicDataType.Question:
                    node_label = "Question"
                    label_group = "Message"
                elif datatype == AcademicDataType.Discussion:
                    node_label = "Discussion"
                    label_group = "Message"
                elif datatype == AcademicDataType.Conclusion:
                    node_label = "Conclusion"
                    label_group = "Conclusion"
                elif datatype == AcademicDataType.Paper:
                    node_label = "Paper"
                    label_group = "Document"
                else:
                    node_label = "Unknown"
                    label_group = "Other"

                # 准备节点属性
                node_props = {
                    "id": key,
                    "label_type": node_label,
                    "namespace": namespace_name,
                }

                # 添加特有属性
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            node_props[k] = v
                        elif isinstance(v, list):
                            try:
                                # 尝试将列表转换为字符串
                                node_props[k] = ", ".join(str(item) for item in v)
                            except:
                                # 如果失败，忽略该属性
                                pass

                # 添加可读标签
                if "Nickname" in value:
                    node_props["name"] = value["Nickname"]
                elif "Title" in value:
                    node_props["name"] = value["Title"]
                elif "Content" in value:
                    content = value["Content"]
                    node_props["name"] = (
                        content[:50] + "..." if len(content) > 50 else content
                    )

                # 创建Neo4j节点
                neo4j_graph.create_node(key, [label_group, node_label], node_props)

                # 创建节点与命名空间的关系
                if namespace:
                    neo4j_graph.create_relationship(
                        source_id=f"namespace_{namespace.name}",
                        target_id=key,
                        relationship_type="CONTAINS",
                        properties={"type": "namespace_membership"},
                    )

                    # 增加关系统计
                    stats["relations_total"] += 1
                    rel_type = "CONTAINS"
                    if rel_type not in stats["relations_by_type"]:
                        stats["relations_by_type"][rel_type] = 0
                    stats["relations_by_type"][rel_type] += 1

                # 更新统计信息
                stats["nodes_total"] += 1
                if node_label not in stats["nodes_by_type"]:
                    stats["nodes_by_type"][node_label] = 0
                stats["nodes_by_type"][node_label] += 1

                if namespace_name not in stats["nodes_by_namespace"]:
                    stats["nodes_by_namespace"][namespace_name] = 0
                stats["nodes_by_namespace"][namespace_name] += 1

            # # 创建节点之间的关系
            # for src in self.backend.semantic_graph.graph_relations:
            #     for dst, relation in self.backend.semantic_graph.graph_relations[src]["children"].items():
            #         # 处理关系类型，确保符合Neo4j标准
            #         relation_type = relation.replace(" ", "_").upper()

            #         # 创建Neo4j关系
            #         neo4j_graph.create_relationship(
            #             source_id=src,
            #             target_id=dst,
            #             relationship_type=relation_type,
            #             properties={"weight": 1.0}
            #         )

            #         # 判断是否跨命名空间
            #         src_namespace = None
            #         dst_namespace = None

            #         for k, value, datatype, _ in self.backend.semantic_map.data:
            #             if k == src:
            #                 src_namespace = self.backend.semantic_graph.datatype_namespace_map.get(datatype)
            #             elif k == dst:
            #                 dst_namespace = self.backend.semantic_graph.datatype_namespace_map.get(datatype)

            #         is_cross_namespace = src_namespace and dst_namespace and src_namespace != dst_namespace

            #         # 更新统计信息
            #         stats["relations_total"] += 1
            #         if relation_type not in stats["relations_by_type"]:
            #             stats["relations_by_type"][relation_type] = 0
            #         stats["relations_by_type"][relation_type] += 1

            #         if is_cross_namespace:
            #             stats["cross_namespace_relations"] += 1
            #             # 如果需要，可以额外标记跨命名空间的关系
            #             neo4j_graph.execute_query(
            #                 "MATCH (a)-[r:$relation_type]->(b) WHERE a.id = $src_id AND b.id = $dst_id "
            #                 "SET r.is_cross_namespace = true, r.src_namespace = $src_ns, r.dst_namespace = $dst_ns",
            #                 {"relation_type": relation_type, "src_id": src, "dst_id": dst,
            #                 "src_ns": src_namespace.name if src_namespace else "UNKNOWN",
            #                 "dst_ns": dst_namespace.name if dst_namespace else "UNKNOWN"}
            #             )

            # 创建节点之间的关系
            for src in self.backend.semantic_graph.graph_relations:
                for dst, relation in self.backend.semantic_graph.graph_relations[src][
                    "children"
                ].items():
                    # 处理关系类型，确保符合Neo4j标准
                    relation_type = relation.replace(" ", "_").upper()

                    # 创建Neo4j关系
                    neo4j_graph.create_relationship(
                        source_id=src,
                        target_id=dst,
                        relationship_type=relation_type,
                        properties={"weight": 1.0},
                    )

                    # 判断是否跨命名空间
                    src_namespace = None
                    dst_namespace = None

                    for k, value, datatype, _ in self.backend.semantic_map.data:
                        if k == src:
                            src_namespace = (
                                self.backend.semantic_graph.datatype_namespace_map.get(
                                    datatype
                                )
                            )
                        elif k == dst:
                            dst_namespace = (
                                self.backend.semantic_graph.datatype_namespace_map.get(
                                    datatype
                                )
                            )

                    is_cross_namespace = (
                        src_namespace
                        and dst_namespace
                        and src_namespace != dst_namespace
                    )

                    # 更新统计信息
                    stats["relations_total"] += 1
                    if relation_type not in stats["relations_by_type"]:
                        stats["relations_by_type"][relation_type] = 0
                    stats["relations_by_type"][relation_type] += 1

                    if is_cross_namespace:
                        stats["cross_namespace_relations"] += 1
                        # 修复：生成动态查询，不再使用参数化的关系类型
                        src_ns_name = src_namespace.name if src_namespace else "UNKNOWN"
                        dst_ns_name = dst_namespace.name if dst_namespace else "UNKNOWN"

                        # 构建直接包含关系类型的查询
                        query = f"""
                        MATCH (a)-[r:{relation_type}]->(b) 
                        WHERE a.id = $src_id AND b.id = $dst_id 
                        SET r.is_cross_namespace = true, 
                            r.src_namespace = $src_ns, 
                            r.dst_namespace = $dst_ns
                        """

                        try:
                            neo4j_graph.execute_query(
                                query,
                                {
                                    "src_id": src,
                                    "dst_id": dst,
                                    "src_ns": src_ns_name,
                                    "dst_ns": dst_ns_name,
                                },
                            )
                        except Exception as e:
                            print(
                                f"标记跨命名空间关系时出错 ({src} -> {dst}): {str(e)}"
                            )
                            # 继续处理其他关系，不中断整个过程

            # 输出导入结果摘要
            print(
                f"\nNeo4j导入完成 - 总节点: {stats['nodes_total']}, 总关系: {stats['relations_total']}"
            )
            print("节点类型统计:")
            for node_type, count in stats["nodes_by_type"].items():
                print(f"  - {node_type}: {count}")

            print("\n命名空间节点统计:")
            for ns_name, count in stats["nodes_by_namespace"].items():
                print(f"  - {ns_name}: {count}")

            print("\n关系类型统计:")
            for rel_type, count in stats["relations_by_type"].items():
                print(f"  - {rel_type}: {count}")

            print(f"跨命名空间关系: {stats['cross_namespace_relations']}")

            # 示例查询
            print("\nNeo4j命名空间查询示例:")
            examples = [
                "// 查询所有命名空间节点\nMATCH (n:Namespace) RETURN n.name",
                "// 查询用户命名空间中的所有节点\nMATCH (ns:Namespace {name:'USER'})-[:CONTAINS]->(n) RETURN n.name",
                "// 查找跨命名空间的关系\nMATCH (a)-[r {is_cross_namespace:true}]->(b) RETURN a.name, type(r), b.name, r.src_namespace, r.dst_namespace",
                "// 按命名空间统计节点数\nMATCH (ns:Namespace)-[:CONTAINS]->(n) RETURN ns.name AS Namespace, count(n) AS NodeCount",
            ]

            for i, example in enumerate(examples, 1):
                print(f"\n查询{i}:\n{example}")

            return stats

        except Exception as e:
            print(f"导出到Neo4j失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}
