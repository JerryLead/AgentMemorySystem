import sys
import os
import traceback
import faiss
from pymilvus import DataType
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import re
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
from academic_group_meeting_backend import AcademicMeetingSystem

# 在文件开头添加输出目录定义
import os
from pathlib import Path

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


class MilvusEnabledAcademicMeetingSystem(AcademicMeetingSystem):
    """扩展学术组会系统，支持Milvus存储"""
    
    def __init__(
        self, 
        use_remote_llm: bool = True,
        use_local_embeddings: bool = True,
        local_model:str ="deepseek-r1:1.5b",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_collection: str = "academic_dialogues",
        neo4j_uri: str = "bolt://localhost:7687", 
        neo4j_user: str = "neo4j",
        neo4j_password: str = "20031117",
        neo4j_database: str = "academicgraph",
        **kwargs
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
        super().__init__(use_remote_llm=use_remote_llm, 
                         use_local_embeddings=use_local_embeddings,
                         local_model=local_model,
                         **kwargs)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.neo4j_graph = None
        self.use_neo4j = False
        
        # 初始化Milvus存储
        try:
            self.milvus_storage = MilvusDialogueStorage(
                host=milvus_host,
                port=milvus_port,
                collection_name=milvus_collection,
                embedding_dim=self.backend.semantic_map.embedding_dim
            )
            self.use_milvus = True
            print(f"已启用Milvus对话存储: {milvus_collection}")
        except Exception as e:
            print(f"初始化Milvus存储失败: {str(e)}")
            print("将仅使用原始存储方式")
            self.use_milvus = False
            self.milvus_storage = None

        # 初始化Neo4j连接
        try:
            self.neo4j_graph = Neo4jInterface(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database
            )
            # 测试连接
            test_result = self.neo4j_graph.execute_query("RETURN 1 AS test")
            if test_result and test_result[0].get("test") == 1:
                self.use_neo4j = True
                print(f"成功连接到Neo4j数据库: {self.neo4j_uri}, 数据库: {self.neo4j_database}")
            else:
                print(f"Neo4j连接测试失败，将不使用Neo4j功能")
        except Exception as e:
            print(f"Neo4j连接初始化失败: {str(e)}")
            self.use_neo4j = False
            self.neo4j_graph = None

        # 当前讨论轮次
        self.current_round = 0
    
    def add_message(self, scene_id: str, speaker_id: str, content: str):
        """重写添加消息方法，增加Milvus存储功能"""
        # 调用原始方法添加消息
        message_id = super().add_message(scene_id, speaker_id, content)
        
        # 如果启用了Milvus且添加消息成功
        if self.use_milvus and message_id and self.milvus_storage:
            try:
                # 获取场景信息
                scene = self.scenes.get(scene_id)
                if not scene:
                    print(f"无法找到场景: {scene_id}")
                    return message_id
                
                # 获取发言者信息
                speaker_info = self.backend.agents.get(speaker_id, {})
                speaker_nickname = speaker_info.get("Nickname", speaker_id)
                
                # 获取角色信息
                academic_role = self.academic_roles.get(speaker_id)
                role_type = academic_role.role_type if academic_role else ""
                specialty = ", ".join(academic_role.specialty) if academic_role and hasattr(academic_role, "specialty") else ""
                
                # 确定消息类型
                message_type = "Question" if "?" in content or "？" in content else "Statement"
                if "总结" in content or "结论" in content:
                    message_type = "Summary"
                
                # 获取时间戳
                timestamp = ""
                for msg in scene.messages:
                    if msg["id"] == message_id:
                        timestamp = msg.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
                    embedding=embedding
                )
            except Exception as e:
                print(f"将对话存储到Milvus时出错: {str(e)}")
                traceback.print_exc()
        
        return message_id

    def add_summary(self, scene_id: str, speaker_id: str, content: str, 
               summary_type: str = "round", round_num: int = None, 
               related_message_ids: List[str] = None) -> str:
        """添加教授总结到语义图，确保放入SUMMARY命名空间
        
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
        # 调用父类方法
        summary_id = super().add_summary(scene_id, speaker_id, content, 
                                    summary_type, round_num, related_message_ids)
        
        # 添加对milvus_storage是否为None的检查
        if self.use_milvus and self.milvus_storage is not None:
            try:
                self.milvus_storage.insert_summary(
                    summary_id=summary_id,
                    title=f"{summary_type}总结",
                    content=content,
                    author=speaker_id,
                    summary_type=summary_type,
                    # round_num=round_num if round_num is not None else self.current_round,
                    round_num = round_num if round_num is not None else 0,
                    topic=self.scenes[scene_id].name,
                    key_findings="",
                    challenges="",
                    future_directions="",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    embedding=self.backend.semantic_map._get_text_embedding(content)
                )     
            except Exception as e:
                print(f"向Milvus添加总结时出错: {str(e)}")
                # 错误不应影响整体功能，继续执行
    
        return summary_id

    # 从这里开始修复任务相关的Milvus存储方法
    def generate_tasks_from_summary(self, summary_id: str) -> List[str]:
        """重写从总结生成任务的方法，增加Milvus存储支持
        
        Args:
            summary_id: 总结节点ID
            
        Returns:
            List[str]: 生成的任务ID列表
        """
        # 调用父类方法生成任务
        task_ids = super().generate_tasks_from_summary(summary_id)
        
        # 如果启用了Milvus，将任务添加到Milvus索引中
        if self.use_milvus and task_ids and self.milvus_storage:
            try:
                for task_id in task_ids:
                    # 获取任务数据
                    task_data = None
                    for key, value, datatype, _ in self.backend.semantic_graph.semantic_map.data:
                        if key == task_id and datatype == AcademicDataType.Task:
                            task_data = value
                            break
                    
                    if task_data:
                        # 提取任务信息
                        title = task_data.get("Title", "")
                        description = task_data.get("Description", "")
                        assignees = task_data.get("Assignees", [])
                        priority = task_data.get("Priority", "中")
                        status = task_data.get("Status", "待开始")
                        topic = task_data.get("Topic", "")
                        
                        # 合并标题和描述作为向量搜索内容
                        content = f"{title} {description}"
                        embedding = self.backend.semantic_map._get_text_embedding(content)
                        
                        # 添加到Milvus - 使用正确的方法
                        self.milvus_storage.insert_task(
                            task_id=task_id,
                            title=title,
                            description=description,
                            assignees=",".join(assignees) if isinstance(assignees, list) else str(assignees),
                            priority=priority,
                            status=status,
                            topic=topic,
                            embedding=embedding
                        )
                
                print(f"已将 {len(task_ids)} 个任务添加到Milvus存储")
            except Exception as e:
                print(f"将任务添加到Milvus时出错: {str(e)}")
                traceback.print_exc()
        
        return task_ids
    
    # 清理和重构Neo4j相关方法
    def export_to_neo4j(self, 
                neo4j_uri: str = None, 
                neo4j_user: str = None, 
                neo4j_password: str = None,
                neo4j_database: str = None,
                create_constraints: bool = True):
        """将语义图导出到Neo4j数据库
        
        Args:
            neo4j_uri: Neo4j数据库URI，如果为None则使用实例化时的URI
            neo4j_user: Neo4j用户名，如果为None则使用实例化时的用户名
            neo4j_password: Neo4j密码，如果为None则使用实例化时的密码
            neo4j_database: Neo4j数据库名，如果为None则使用实例化时的数据库名
            create_constraints: 是否创建约束
            
        Returns:
            Dict: 包含导出统计信息的字典
        """
        try:
            # 使用传入的参数，如果没有则使用实例化时的默认值
            uri = neo4j_uri if neo4j_uri is not None else self.neo4j_uri
            user = neo4j_user if neo4j_user is not None else self.neo4j_user
            password = neo4j_password if neo4j_password is not None else self.neo4j_password
            database = neo4j_database if neo4j_database is not None else self.neo4j_database
            
            # 初始化Neo4j图存储
            neo4j_interface = Neo4jInterface(
                uri=uri,
                user=user,
                password=password,
                database=database
            )
            
            # 保存Neo4j连接以供后续使用
            self.neo4j_graph = neo4j_interface
            self.use_neo4j = True
            
            print(f"已连接Neo4j数据库: {uri}, 数据库: {database}")
            
            # 创建唯一性约束以提高性能
            if create_constraints:
                self._prepare_neo4j_constraints(neo4j_interface)
            
            # 更新命名空间子图确保数据最新
            self.backend.semantic_graph.auto_generate_subgraphs()
            
            # 统计信息
            stats = {
                "nodes_total": 0,
                "nodes_by_type": {},
                "nodes_by_namespace": {},
                "relations_total": 0, 
                "relations_by_type": {},
                "cross_namespace_relations": 0
            }
            
            # 分步骤导出数据，便于维护和追踪问题
            stats = self._export_namespaces_to_neo4j(neo4j_interface, stats)
            node_namespaces, stats = self._export_nodes_to_neo4j(neo4j_interface, stats)
            stats = self._export_relationships_to_neo4j(neo4j_interface, node_namespaces, stats)

            # 输出导入结果摘要
            print(f"\nNeo4j导入完成 - 总节点: {stats['nodes_total']}, 总关系: {stats['relations_total']}")
            print("节点类型统计:")
            for node_type, count in stats["nodes_by_type"].items():
                print(f"  {node_type}: {count}个节点")
            
            print("\n命名空间节点统计:")
            for ns_name, count in stats["nodes_by_namespace"].items():
                print(f"  {ns_name}: {count}个节点")
            
            print("\n关系类型统计:")
            for rel_type, count in stats["relations_by_type"].items():
                print(f"  {rel_type}: {count}个关系")
                
            print(f"跨命名空间关系: {stats['cross_namespace_relations']}")
            
            # 示例查询
            print("\nNeo4j查询示例:")
            # 基本命名空间查询示例
            namespace_examples = [
                "// 查询所有命名空间节点\nMATCH (n:Namespace) RETURN n.name",
                "// 查询用户命名空间中的所有节点\nMATCH (ns:Namespace {name:'USER'})-[:CONTAINS]->(n) RETURN n.name",
                "// 查找跨命名空间的关系\nMATCH (a)-[r {is_cross_namespace:true}]->(b) RETURN a.name, type(r), b.name, r.src_namespace, r.dst_namespace",
                "// 按命名空间统计节点数\nMATCH (ns:Namespace)-[:CONTAINS]->(n) RETURN ns.name AS Namespace, count(n) AS NodeCount"
            ]
            
            # 示例查询显示
            for i, example in enumerate(namespace_examples[:5], 1):
                print(f"示例{i}:\n{example}\n")
            
            return stats
            
        except Exception as e:
            print(f"导出到Neo4j失败: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    # 辅助方法 - 重构为更模块化的形式
    def _prepare_neo4j_constraints(self, neo4j_interface: Neo4jInterface):
        """准备Neo4j约束 - 为实体创建唯一ID约束"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conclusion) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Namespace) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subtopic) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Reference) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:OpenSourceSystem) REQUIRE o.id IS UNIQUE"
        ]
        
        success_count = 0
        for constraint in constraints:
            try:
                neo4j_interface.execute_query(constraint)
                success_count += 1
            except Exception as e:
                print(f"创建约束失败: {str(e)}")
        
        print(f"已创建 {success_count}/{len(constraints)} 个Neo4j约束")
        return success_count == len(constraints)

    def _export_namespaces_to_neo4j(self, neo4j_interface: Neo4jInterface, stats):
        """导出命名空间到Neo4j"""
        for namespace in NamespaceType:
            namespace_id = f"namespace_{namespace.name}"
            success = self._create_neo4j_node(
                neo4j_interface,
                namespace_id,
                ["Namespace"],
                {
                    "id": namespace_id,
                    "name": namespace.name,
                    "description": f"{namespace.name} namespace"
                }
            )
            
            if success:
                # 添加命名空间节点到统计
                stats["nodes_total"] += 1
                if "Namespace" not in stats["nodes_by_type"]:
                    stats["nodes_by_type"]["Namespace"] = 0
                stats["nodes_by_type"]["Namespace"] += 1
        
        return stats
    
    def _export_nodes_to_neo4j(self, neo4j_interface: Neo4jInterface, stats):
        """导出所有节点到Neo4j - 修复版本"""
        try:
            node_namespaces = {}  # 节点ID到命名空间的映射，用于后续创建边
            
            # 从语义图中获取所有节点并创建
            for key in self.backend.semantic_graph.graph_relations:
                # 获取节点数据和类型
                node_info = None
                for k, value, datatype, _ in self.backend.semantic_map.data:
                    if k == key:
                        node_info = {"value": value, "datatype": datatype}
                        break
                        
                if not node_info:
                    print(f"警告：找不到节点 {key} 的数据")
                    continue
                    
                datatype = node_info.get("datatype")
                value = node_info.get("value", {})
                
                # 获取命名空间
                namespace = self.backend.semantic_graph.datatype_namespace_map.get(datatype)
                namespace_name = namespace.name if namespace else "UNKNOWN"
                node_namespaces[key] = namespace

                # 确定节点标签
                node_label = "Unknown"
                label_group = "Academic"
                
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
                elif datatype == AcademicDataType.Summary:
                    node_label = "Summary"
                    label_group = "Conclusion"
                elif datatype == AcademicDataType.Task:
                    node_label = "Task"
                    label_group = "Action"
                else:
                    node_label = "Unknown"
                    label_group = "Other"
                
                # 准备节点属性 - 始终包含基本字段
                node_props = {
                    "id": key,
                    "label_type": node_label,
                    "namespace": namespace_name
                }
                
                # 处理可能的数据类型特殊情况 - 确保所有关键属性被保留
                if datatype == AcademicDataType.Task:
                    self._prepare_task_properties(node_props, value)
                elif datatype == AcademicDataType.Summary:
                    self._prepare_summary_properties(node_props, value)
                else:
                    # 通用属性处理
                    self._prepare_general_properties(node_props, value)
                
                # 确保有可读的名称属性
                if "name" not in node_props:
                    node_props["name"] = value.get("Title", value.get("Name", key))
                
                # 创建Neo4j节点
                success = self._create_neo4j_node(
                    neo4j_interface, 
                    key, 
                    [label_group, node_label], 
                    node_props
                )
                
                if success:
                    # 创建节点与命名空间的关系
                    if namespace:
                        success = self._create_neo4j_relationship(
                            neo4j_interface,
                            f"namespace_{namespace.name}", 
                            key,
                            "CONTAINS",
                            {"type": "namespace_membership"}
                        )
                        
                        if success:
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
                
            return node_namespaces, stats
        except Exception as e:
            print(f"导出节点到Neo4j失败: {str(e)}")
            traceback.print_exc()
            return {}, stats
    
    def _prepare_task_properties(self, node_props, value: Dict[str, Any]):
        """准备任务节点属性"""
        # 任务节点特殊处理
        node_props["title"] = value.get("Title", "Untitled Task")
        node_props["description"] = value.get("Description", "")
        node_props["status"] = value.get("Status", "待开始")
        node_props["priority"] = value.get("Priority", "中")
        node_props["topic"] = value.get("Topic", "")
        node_props["created_at"] = value.get("CreatedAt", "")
        
        # 处理执行者列表
        assignees = value.get("Assignees", [])
        if assignees and isinstance(assignees, list):
            node_props["assignees"] = ", ".join(assignees)
        else:
            node_props["assignees"] = str(assignees)
            
        # 处理元数据
        metadata = value.get("metadata", {})
        if metadata and isinstance(metadata, dict):
            for meta_key, meta_value in metadata.items():
                if isinstance(meta_value, (str, int, float, bool)):
                    node_props[f"meta_{meta_key}"] = meta_value
    
    def _prepare_summary_properties(self, node_props, value: Dict[str, Any]):
        """准备总结节点属性"""
        # 总结节点特殊处理
        node_props["title"] = value.get("Title", "Untitled Summary")
        node_props["content"] = value.get("Content", "")
        node_props["type"] = value.get("Type", "")
        node_props["round"] = value.get("Round", "")
        node_props["topic"] = value.get("Topic", "")
        node_props["timestamp"] = value.get("Timestamp", "")
        node_props["author"] = value.get("Author", "")
        
        # 添加结构化内容
        node_props["key_findings"] = value.get("KeyFindings", "")
        node_props["challenges"] = value.get("Challenges", "")
        node_props["future_directions"] = value.get("FutureDirections", "")
        node_props["research_gaps"] = value.get("ResearchGaps", "")
        node_props["action_items"] = value.get("ActionItems", "")
    
    def _prepare_general_properties(self, node_props, value: Dict[str, Any]):
        """准备通用节点属性"""
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)):
                node_props[k.lower()] = v
            elif isinstance(v, list):
                try:
                    node_props[k.lower()] = ", ".join(str(item) for item in v)
                except:
                    node_props[k.lower()] = str(v)
            elif v is None:
                node_props[k.lower()] = ""
            else:
                try:
                    node_props[k.lower()] = str(v)
                except:
                    node_props[k.lower()] = "无法转换的数据"
    
    def _export_relationships_to_neo4j(self, neo4j_interface:Neo4jInterface, node_namespaces, stats):
        """导出所有关系到Neo4j"""
        # 创建节点之间的关系
        for src in self.backend.semantic_graph.graph_relations:
            for dst, relation in self.backend.semantic_graph.graph_relations[src]["children"].items():
                # 确保源节点和目标节点都存在
                src_namespace = node_namespaces.get(src)
                dst_namespace = node_namespaces.get(dst)
                
                if not src_namespace or not dst_namespace:
                    continue
                
                # 确定关系类型
                relation_type = relation or "RELATED_TO"
                
                # 检查是否跨命名空间
                is_cross_namespace = src_namespace != dst_namespace
                
                # 关系属性
                rel_props = {
                    "is_cross_namespace": is_cross_namespace
                }
                
                if is_cross_namespace:
                    rel_props["src_namespace"] = src_namespace.name
                    rel_props["dst_namespace"] = dst_namespace.name
                
                # 创建关系
                success = self._create_neo4j_relationship(
                    neo4j_interface, 
                    src, 
                    dst, 
                    relation_type, 
                    rel_props
                )
                
                if success:
                    stats["relations_total"] += 1
                    if relation_type not in stats["relations_by_type"]:
                        stats["relations_by_type"][relation_type] = 0
                    stats["relations_by_type"][relation_type] += 1
                    
                    if is_cross_namespace:
                        stats["cross_namespace_relations"] += 1
        
        # 添加任务特定的依赖关系 - 从元数据中提取
        tasks = {}
        for key, value, datatype, _ in self.backend.semantic_map.data:
            if datatype == AcademicDataType.Task:
                tasks[key] = value
        
        # 查找并添加任务依赖关系
        for task_id, task_data in tasks.items():
            if "Dependencies" in task_data and isinstance(task_data["Dependencies"], list):
                for dep_id in task_data["Dependencies"]:
                    if dep_id in tasks:
                        rel_props = {"relationship_type": "DEPENDS_ON"}
                        success = self._create_neo4j_relationship(
                            neo4j_interface, 
                            task_id, 
                            dep_id, 
                            "DEPENDS_ON", 
                            rel_props
                        )
                        
                        if success:
                            stats["relations_total"] += 1
                            if "DEPENDS_ON" not in stats["relations_by_type"]:
                                stats["relations_by_type"]["DEPENDS_ON"] = 0
                            stats["relations_by_type"]["DEPENDS_ON"] += 1
        
        return stats
    
    # 创建节点和关系的辅助方法
    def _create_neo4j_node(self, neo4j_interface, node_id, labels, properties):
        """创建Neo4j节点的辅助方法，带错误处理"""
        try:
            # 确保属性中没有None值，Neo4j不接受None值
            clean_props = {k: ('' if v is None else v) for k, v in properties.items()}
            neo4j_interface.create_node(node_id, labels, clean_props)
            return True
        except Exception as e:
            print(f"创建节点失败 [{node_id}]: {str(e)}")
            return False
    
    def _create_neo4j_relationship(self, neo4j_interface:Neo4jInterface, source_id, target_id, rel_type, properties=None):
        """创建Neo4j关系的辅助方法，带错误处理"""
        try:
            props = properties or {}
            # 确保属性中没有None值
            clean_props = {k: ('' if v is None else v) for k, v in props.items()}
            neo4j_interface.create_relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                properties=clean_props
            )
            return True
        except Exception as e:
            print(f"创建关系失败 [{source_id}]-[{rel_type}]->[{target_id}]: {str(e)}")
            return False

    # 其他方法保持不变...
    # ...existing code...
    
    def close_milvus(self):
        """关闭Milvus连接"""
        if self.use_milvus and self.milvus_storage:
            try:
                self.milvus_storage.close()
                print("已关闭Milvus连接")
            except Exception as e:
                print(f"关闭Milvus连接时出错: {str(e)}")
                
    def __del__(self):
        """析构函数，确保关闭所有连接"""
        self.close_milvus()

    def start_academic_meeting(self, scene_id: str, topic: str, 
                          moderator_id: str, rounds: int = 3, deep_search: bool = False):
        """重写会议开始方法，增加轮次跟踪"""
        # 重置轮次计数
        self.current_round = 0
        # 调用原方法
        super().start_academic_meeting(scene_id, topic, moderator_id, rounds, deep_search)
    
    def run_custom_meeting(self, topic: str, professor_id: str = None, rounds: int = 3, deep_search: bool = False):
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
    
    def generate_comprehensive_review(self, topic: str, professor_id: str = None, 
                                     subtopics: List[str] = None, rounds: int = 3, 
                                     include_literature: bool = True, 
                                     include_open_source: bool = True,
                                     output_dir: str = None):
        """重写综述报告生成方法，增加Milvus存储支持
        
        Args:
            topic: 主题，如"面向智能体的记忆管理系统"
            professor_id: 主持讨论的教授ID
            subtopics: 子话题列表，如记忆模型、记忆存储等
            rounds: 讨论轮次
            include_literature: 是否包含文献引用
            include_open_source: 是否包含开源系统
            output_dir: 输出目录，如果为None则使用默认值
            
        Returns:
            dict: 包含综述报告及其元数据的字典
        """
        # 重置轮次计数
        self.current_round = 0
        
        # 调用父类方法生成综述报告
        review_result = super().generate_comprehensive_review(
            topic=topic,
            professor_id=professor_id,
            subtopics=subtopics,
            rounds=rounds,
            include_literature=include_literature,
            include_open_source=include_open_source,
            output_dir=output_dir
        )

        # 添加到语义图前进行命名空间设置
        # if report_info and report_id:
        #     # 添加到语义图
        #     try:
        #         # 确保加入SUMMARY命名空间
        #         self.backend.semantic_graph.add_node(
        #             report_id, 
        #             report_info, 
        #             AcademicDataType.Summary,  # 明确指定为Summary类型
        #             valid_parent_keys
        #         )
        #         print(f"已将综述报告添加到语义图，ID: {report_id}")
        #     except Exception as e:
        #         print(f"添加报告到语义图时出错: {str(e)}")

        # 如果启用了Milvus，将综述报告添加到Milvus索引中
        if self.use_milvus and review_result:
            try:
                report_id = review_result.get("report_id")
                content = review_result.get("content", "")
                
                # 获取教授信息
                professor_name = review_result.get("professor", "")
                
                # 生成嵌入向量
                embedding = self.backend.semantic_map._get_text_embedding(content[:5000])  # 使用前5000字符生成嵌入
                
                # 为Neo4j图结构添加报告节点和关系
                if hasattr(self, 'neo4j_graph') and self.neo4j_graph:
                    self._add_review_to_neo4j(review_result)
                
                # 添加到Milvus存储
                if report_id and content:
                    self.milvus_storage.insert_dialogue(
                        message_id=report_id,
                        scene_id="review",
                        speaker_id=professor_id,
                        speaker_nickname=professor_name,
                        role_type="Professor",
                        specialty="Research Review",
                        content=content[:5000],  # Milvus可能有字段长度限制
                        topic=topic,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        round_num=100,  # 使用特殊轮次编号表示这是一个综述报告
                        message_type="LiteratureReview",
                        embedding=embedding
                    )
                    print(f"综述报告已添加到Milvus存储: {report_id}")
            except Exception as e:
                print(f"将综述报告添加到Milvus存储时出错: {str(e)}")

        # if self.use_neo4j and review_result:
        #     try:
        #         self.export_to_neo4j(self.neo4j_uri, self.neo4j_user, self.neo4j_password, self.neo4j_database)
        #     except Exception as e:
        #         print(f"将综述报告添加到Neo4j时出错: {str(e)}")
        
        return review_result
    
    def search_summary(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """搜索与查询文本相似的总结内容
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相似总结列表
        """
        summaries = []
        # 获取所有总结节点
        for key, value, datatype, embedding in self.backend.semantic_map.data:
            if datatype == AcademicDataType.Summary:
                summaries.append((key, value, embedding))
        
        if not summaries:
            return []
        
        # 生成查询文本的嵌入向量
        query_embedding = self.backend.semantic_map._get_text_embedding(query_text)
        
        # 计算相似度并排序
        similar_summaries = []
        for key, value, embedding in summaries:
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similar_summaries.append((key, value, similarity))
        
        similar_summaries.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前top_k个结果
        results = []
        for key, value, similarity in similar_summaries[:top_k]:
            results.append({
                "id": key,
                "title": value.get("Title", ""),
                "content": value.get("Content", ""),
                "type": value.get("Type", ""),
                "score": float(similarity),
                "timestamp": value.get("Timestamp", "")
            })
        
        return results
    
    def search_tasks(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """搜索与查询文本相似的任务
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相似任务列表
        """
        tasks = []
        # 获取所有任务节点
        for key, value, datatype, embedding in self.backend.semantic_map.data:
            if datatype == AcademicDataType.Task:
                tasks.append((key, value, embedding))
        
        if not tasks:
            return []
        
        # 生成查询文本的嵌入向量
        query_embedding = self.backend.semantic_map._get_text_embedding(query_text)
        
        # 计算相似度并排序
        similar_tasks = []
        for key, value, embedding in tasks:
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similar_tasks.append((key, value, similarity))
        
        similar_tasks.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前top_k个结果
        results = []
        for key, value, similarity in similar_tasks[:top_k]:
            results.append({
                "id": key,
                "title": value.get("Title", ""),
                "description": value.get("Description", ""),
                "priority": value.get("Priority", "中"),
                "status": value.get("Status", "待开始"),
                "score": float(similarity)
            })
        
        return results

    def _add_review_to_neo4j(self, review_result):
        """将综述报告相关数据添加到Neo4j图数据库
        
        Args:
            review_result: 综述报告结果字典
        """
        if not self.use_neo4j or not self.neo4j_graph:
            print("Neo4j未启用，无法添加综述报告到图数据库")
            return
            
        try:
            # 提取报告信息
            report_id = review_result.get("report_id")
            topic = review_result.get("topic")
            professor = review_result.get("professor")
            subtopics = review_result.get("subtopics", [])
            filename = review_result.get("filename")
            references = review_result.get("references", [])
            open_source_systems = review_result.get("open_source_systems", [])
            
            # 创建综述报告节点
            report_props = {
                "id": report_id,
                "name": f"{topic}综述报告",
                "topic": topic,
                "author": professor,
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "file_path": filename,
                "reference_count": len(references),
                "system_count": len(open_source_systems),
                "subtopic_count": len(subtopics)
            }
            
            success = self._create_neo4j_node(
                self.neo4j_graph, 
                report_id, 
                ["Review", "Document"], 
                report_props
            )
            
            if not success:
                print(f"创建综述报告节点失败: {report_id}")
                return
                
            # 找到教授节点ID
            professor_id = None
            for agent_id, agent_info in self.backend.agents.items():
                if agent_info.get("Nickname") == professor:
                    professor_id = agent_id
                    break
            
            # 建立教授与报告的关系
            if professor_id:
                self._create_neo4j_relationship(
                    self.neo4j_graph,
                    source_id=professor_id,
                    target_id=report_id,
                    rel_type="AUTHORED",
                    properties={"date": datetime.now().strftime("%Y-%m-%d")}
                )
            
            # 为每个子话题创建节点并与报告关联
            for i, subtopic in enumerate(subtopics):
                subtopic_id = f"subtopic_{report_id}_{i}"
                subtopic_props = {
                    "id": subtopic_id,
                    "name": subtopic,
                    "order": i + 1,
                    "parent_topic": topic
                }
                
                # 创建子话题节点
                success = self._create_neo4j_node(
                    self.neo4j_graph,
                    subtopic_id,
                    ["Subtopic", "ResearchTopic"],
                    subtopic_props
                )
                
                if success:
                    # 与报告建立关系
                    self._create_neo4j_relationship(
                        self.neo4j_graph,
                        source_id=report_id,
                        target_id=subtopic_id,
                        rel_type="CONTAINS_SECTION",
                        properties={"order": i + 1}
                    )
            
            # 为参考文献创建节点
            for i, ref in enumerate(references):
                ref_id = f"ref_{report_id}_{i}"
                ref_title = ref.get("title", ref.get("Title", "未命名文献"))
                
                ref_props = {
                    "id": ref_id,
                    "name": ref_title,
                    "authors": ref.get("authors", ref.get("Authors", "N/A")),
                    "year": ref.get("year", ref.get("Year", "N/A")),
                    "url": ref.get("url", ref.get("URL", "")),
                    "citation_index": i + 1
                }
                
                # 创建参考文献节点
                success = self._create_neo4j_node(
                    self.neo4j_graph,
                    ref_id,
                    ["Reference", "Paper"],
                    ref_props
                )
                
                if success:
                    # 与报告建立关系
                    self._create_neo4j_relationship(
                        self.neo4j_graph,
                        source_id=report_id,
                        target_id=ref_id,
                        rel_type="CITES",
                        properties={"citation_index": i + 1}
                    )
            
            # 为开源系统创建节点
            for i, system in enumerate(open_source_systems):
                system_id = f"system_{report_id}_{i}"
                system_name = system.get("name", "未命名系统")
                
                system_props = {
                    "id": system_id,
                    "name": system_name,
                    "description": system.get("description", ""),
                    "url": system.get("url", ""),
                    "order": i + 1
                }
                
                # 创建开源系统节点
                success = self._create_neo4j_node(
                    self.neo4j_graph,
                    system_id,
                    ["OpenSourceSystem", "Implementation"],
                    system_props
                )
                
                if success:
                    # 与报告建立关系
                    self._create_neo4j_relationship(
                        self.neo4j_graph,
                        source_id=report_id,
                        target_id=system_id,
                        rel_type="MENTIONS_SYSTEM",
                        properties={"order": i + 1}
                    )
            
            print(f"综述报告成功添加到Neo4j图数据库")
            
        except Exception as e:
            print(f"将综述报告添加到Neo4j时出错: {str(e)}")
            traceback.print_exc()
    
    def _increment_discussion_round(self):
        """递增讨论轮次计数器，在每轮讨论结束时调用
        
        Returns:
            int: 当前轮次号
        """
        self.current_round += 1
        # 修改输出格式，使其更容易被前端识别和处理
        print(f"\n--- 第 {self.current_round} 轮讨论 ---\n")
        return self.current_round
    
    # 删除重复的函数 _export_nodes_to_neo4j，之前有两个同名函数

    # 添加用于搜索的通用方法
    def semantic_search(self, query_text: str, entity_type: str = "all", top_k: int = 5) -> List[Dict]:
        """进行语义搜索，查找与查询文本相似的实体
        
        Args:
            query_text: 查询文本
            entity_type: 实体类型，可选值为 "all", "dialogue", "task", "summary", "paper"
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相似实体列表
        """
        # 如果启用了Milvus，优先使用Milvus进行搜索
        if self.use_milvus and self.milvus_storage:
            try:
                # 生成查询文本的嵌入向量
                query_embedding = self.backend.semantic_map._get_text_embedding(query_text)
                
                if entity_type == "dialogue" or entity_type == "all":
                    return self.milvus_storage.search_similar_dialogues(query_embedding, top_k)
                elif entity_type == "task":
                    return self.milvus_storage.search_similar_tasks(query_embedding, top_k)
                else:
                    # 对于未实现专门存储的类型，使用本地搜索
                    pass
            except Exception as e:
                print(f"Milvus搜索失败: {str(e)}")
                # 如果Milvus搜索失败，回退到本地搜索
        
        # 本地搜索实现
        results = []
        
        # 获取所有符合条件的实体
        entities = []
        for key, value, datatype, embedding in self.backend.semantic_map.data:
            if entity_type == "all":
                entities.append((key, value, datatype, embedding))
            elif entity_type == "dialogue" and datatype in [AcademicDataType.Question, AcademicDataType.Discussion]:
                entities.append((key, value, datatype, embedding))
            elif entity_type == "task" and datatype == AcademicDataType.Task:
                entities.append((key, value, datatype, embedding))
            elif entity_type == "summary" and datatype == AcademicDataType.Summary:
                entities.append((key, value, datatype, embedding))
            elif entity_type == "paper" and datatype == AcademicDataType.Paper:
                entities.append((key, value, datatype, embedding))
        
        if not entities:
            return []
        
        # 生成查询文本的嵌入向量
        query_embedding = self.backend.semantic_map._get_text_embedding(query_text)
        
        # 计算相似度并排序
        similar_entities = []
        for key, value, datatype, embedding in entities:
            # 计算余弦相似度
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similar_entities.append((key, value, datatype, similarity))
        
        similar_entities.sort(key=lambda x: x[3], reverse=True)
        
        # 返回前top_k个结果
        for key, value, datatype, similarity in similar_entities[:top_k]:
            if datatype == AcademicDataType.Task:
                results.append({
                    "id": key,
                    "type": "task",
                    "title": value.get("Title", ""),
                    "description": value.get("Description", ""),
                    "priority": value.get("Priority", "中"),
                    "status": value.get("Status", "待开始"),
                    "score": float(similarity)
                })
            elif datatype == AcademicDataType.Summary:
                results.append({
                    "id": key,
                    "type": "summary",
                    "title": value.get("Title", ""),
                    "content": value.get("Content", ""),
                    "score": float(similarity)
                })
            elif datatype in [AcademicDataType.Question, AcademicDataType.Discussion]:
                results.append({
                    "id": key,
                    "type": "dialogue",
                    "content": value.get("Content", ""),
                    "speaker": value.get("Speaker", ""),
                    "score": float(similarity)
                })
            elif datatype == AcademicDataType.Paper:
                results.append({
                    "id": key,
                    "type": "paper",
                    "title": value.get("Title", ""),
                    "authors": value.get("Authors", ""),
                    "year": value.get("Year", ""),
                    "score": float(similarity)
                })
            else:
                results.append({
                    "id": key,
                    "type": str(datatype),
                    "content": str(value),
                    "score": float(similarity)
                })
        
        return results