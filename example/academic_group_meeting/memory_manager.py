import uuid
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import numpy as np
from collections import defaultdict
import re
from heapq import nlargest
import os
import pandas as pd
from semantic_map.deepseek_client import deepseek_remote, deepseek_local

class MemoryType:
    """记忆类型枚举"""
    DIALOGUE = "dialogue"       # 对话记忆
    SUMMARY = "summary"         # 总结记忆
    TASK = "task"               # 任务记忆  
    LITERATURE = "literature"   # 文献记忆
    CONCEPT = "concept"         # 概念记忆
    BELIEF = "belief"           # 信念记忆

class MemoryNode:
    """记忆节点基类"""
    def __init__(
        self,
        content: str,
        memory_type: str,
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        embedding: np.ndarray = None,
        timestamp: str = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.importance = importance
        self.embedding = embedding
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0
    
    def update_access(self):
        """更新访问时间和访问计数"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count
        }
        # 不包含嵌入向量，因为它不能直接JSON序列化
        return result

class DialogueMemory(MemoryNode):
    """对话记忆节点类"""
    def __init__(
        self,
        content: str,
        speaker_id: str,
        speaker_name: str,
        scene_id: str = None,
        round_num: int = None,
        response_to: str = None,
        embedding: np.ndarray = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(
            content=content,
            memory_type=MemoryType.DIALOGUE,
            metadata=metadata or {},
            embedding=embedding
        )
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.scene_id = scene_id
        self.round_num = round_num
        self.response_to = response_to  # 回复的消息ID
        
        # 更新元数据
        self.metadata.update({
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "scene_id": scene_id,
            "round_num": round_num,
            "response_to": response_to
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result.update({
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "scene_id": self.scene_id,
            "round_num": self.round_num,
            "response_to": self.response_to
        })
        return result

class SummaryMemory(MemoryNode):
    """总结记忆节点类"""
    def __init__(
        self,
        content: str,
        summary_type: str,
        author_id: str = None,
        round_num: int = None,
        related_dialogue_ids: List[str] = None,
        key_points: List[str] = None,
        embedding: np.ndarray = None,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(
            content=content,
            memory_type=MemoryType.SUMMARY,
            metadata=metadata or {},
            importance=0.8,  # 总结默认重要性较高
            embedding=embedding
        )
        self.summary_type = summary_type  # 'round', 'subtopic', 'final'
        self.author_id = author_id
        self.round_num = round_num
        self.related_dialogue_ids = related_dialogue_ids or []
        self.key_points = key_points or []
        
        # 更新元数据
        self.metadata.update({
            "summary_type": summary_type,
            "author_id": author_id,
            "round_num": round_num
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result.update({
            "summary_type": self.summary_type,
            "author_id": self.author_id,
            "round_num": self.round_num,
            "related_dialogue_ids": self.related_dialogue_ids,
            "key_points": self.key_points
        })
        return result

class TaskMemory(MemoryNode):
    """任务记忆节点类"""
    def __init__(
        self,
        title: str,
        description: str,
        assignees: List[str] = None,
        priority: str = "中",
        status: str = "待开始",
        due_date: str = None,
        dependencies: List[str] = None,
        source_id: str = None,
        embedding: np.ndarray = None,
        metadata: Dict[str, Any] = None
    ):
        content = f"{title}: {description}"
        super().__init__(
            content=content,
            memory_type=MemoryType.TASK,
            metadata=metadata or {},
            importance=0.7,  # 任务默认重要性较高
            embedding=embedding
        )
        self.title = title
        self.description = description
        self.assignees = assignees or []
        self.priority = priority
        self.status = status
        self.due_date = due_date
        self.dependencies = dependencies or []
        self.source_id = source_id  # 来源ID，可能是总结或讨论
        
        # 更新元数据
        self.metadata.update({
            "title": title,
            "priority": priority,
            "status": status,
            "due_date": due_date,
            "source_id": source_id
        })
    
    def update_status(self, status: str):
        """更新任务状态"""
        self.status = status
        self.metadata["status"] = status
        self.last_access_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result.update({
            "title": self.title,
            "description": self.description,
            "assignees": self.assignees,
            "priority": self.priority,
            "status": self.status,
            "due_date": self.due_date,
            "dependencies": self.dependencies,
            "source_id": self.source_id
        })
        return result

class LiteratureMemory(MemoryNode):
    """文献记忆节点类"""
    def __init__(
        self,
        title: str,
        authors: List[str] = None,
        year: str = None,
        abstract: str = None,
        url: str = None,
        key_findings: List[str] = None,
        related_topics: List[str] = None,
        embedding: np.ndarray = None,
        metadata: Dict[str, Any] = None
    ):
        content = f"{title}"
        if abstract:
            content += f"\n{abstract}"
        
        super().__init__(
            content=content,
            memory_type=MemoryType.LITERATURE,
            metadata=metadata or {},
            importance=0.6,  # 文献默认重要性适中
            embedding=embedding
        )
        self.title = title
        self.authors = authors or []
        self.year = year
        self.abstract = abstract
        self.url = url
        self.key_findings = key_findings or []
        self.related_topics = related_topics or []
        
        # 更新元数据
        self.metadata.update({
            "title": title,
            "authors": ", ".join(authors) if authors else "",
            "year": year,
            "url": url
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result.update({
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "key_findings": self.key_findings,
            "related_topics": self.related_topics
        })
        return result

class ConceptMemory(MemoryNode):
    """概念记忆节点类"""
    def __init__(
        self,
        name: str,
        definition: str,
        related_concepts: List[str] = None,
        examples: List[str] = None,
        source_ids: List[str] = None,
        embedding: np.ndarray = None,
        metadata: Dict[str, Any] = None
    ):
        content = f"{name}: {definition}"
        super().__init__(
            content=content,
            memory_type=MemoryType.CONCEPT,
            metadata=metadata or {},
            importance=0.65,  # 概念默认重要性适中偏高
            embedding=embedding
        )
        self.name = name
        self.definition = definition
        self.related_concepts = related_concepts or []
        self.examples = examples or []
        self.source_ids = source_ids or []  # 来源ID列表
        
        # 更新元数据
        self.metadata.update({
            "name": name
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result.update({
            "name": self.name,
            "definition": self.definition,
            "related_concepts": self.related_concepts,
            "examples": self.examples,
            "source_ids": self.source_ids
        })
        return result

class MemoryManager:
    """记忆管理器类，负责维护、检索和管理不同类型的记忆"""
    
    def __init__(
        self,
        use_remote_llm: bool = True,
        embedding_model: Any = None,
        memory_decay_rate: float = 0.05,
        max_dialogue_memories: int = 200,
        relevance_threshold: float = 0.6,
        persistence_dir: str = "./memories",
        local_model: str = "deepseek-r1:1.5b"
    ):
        """
        初始化记忆管理器
        
        Args:
            use_remote_llm: 是否使用远程LLM
            embedding_model: 嵌入模型，用于生成向量表示
            memory_decay_rate: 记忆衰减率，控制记忆重要性随时间降低的速度
            max_dialogue_memories: 最大对话记忆数量
            relevance_threshold: 相关性阈值，用于筛选检索结果
            persistence_dir: 记忆持久化目录
            local_model: 本地模型名称
        """
        self.use_remote_llm = use_remote_llm
        self.embedding_model = embedding_model
        self.memory_decay_rate = memory_decay_rate
        self.max_dialogue_memories = max_dialogue_memories
        self.relevance_threshold = relevance_threshold
        self.persistence_dir = persistence_dir
        
        # 初始化LLM客户端
        self.llm = deepseek_remote() if use_remote_llm else deepseek_local(model=local_model)
        
        # 初始化记忆存储
        self.memories: Dict[str, MemoryNode] = {}
        
        # 按类型索引记忆
        self.memory_by_type: Dict[str, Dict[str, MemoryNode]] = {
            MemoryType.DIALOGUE: {},
            MemoryType.SUMMARY: {},
            MemoryType.TASK: {},
            MemoryType.LITERATURE: {},
            MemoryType.CONCEPT: {},
            MemoryType.BELIEF: {}
        }
        
        # 按场景索引对话记忆
        self.dialogue_by_scene: Dict[str, List[str]] = defaultdict(list)
        
        # 按轮次索引对话和总结记忆
        self.memory_by_round: Dict[int, List[str]] = defaultdict(list)
        
        # 创建记忆持久化目录
        os.makedirs(self.persistence_dir, exist_ok=True)
    
    def add_memory(self, memory: MemoryNode) -> str:
        """
        添加一个记忆节点到记忆管理器
        
        Args:
            memory: 记忆节点对象
            
        Returns:
            str: 记忆节点ID
        """
        # 1. 如果没有嵌入向量，尝试生成
        if memory.embedding is None and self.embedding_model is not None:
            try:
                memory.embedding = self._get_embedding(memory.content)
            except Exception as e:
                print(f"生成嵌入向量失败: {str(e)}")
        
        # 2. 保存记忆节点
        self.memories[memory.id] = memory
        
        # 3. 更新类型索引
        if memory.memory_type in self.memory_by_type:
            self.memory_by_type[memory.memory_type][memory.id] = memory
        
        # 4. 如果是对话记忆，更新场景索引
        if isinstance(memory, DialogueMemory) and memory.scene_id:
            self.dialogue_by_scene[memory.scene_id].append(memory.id)
            
            # 4.1. 如果对话记忆数量超过上限，执行遗忘
            scene_dialogues = self.dialogue_by_scene[memory.scene_id]
            if len(scene_dialogues) > self.max_dialogue_memories:
                self._forget_oldest_dialogues(memory.scene_id)
        
        # 5. 如果有轮次信息，更新轮次索引
        round_num = None
        if isinstance(memory, DialogueMemory):
            round_num = memory.round_num
        elif isinstance(memory, SummaryMemory):
            round_num = memory.round_num
            
        if round_num is not None:
            self.memory_by_round[round_num].append(memory.id)
        
        return memory.id
    
    def add_dialogue_memory(
        self, 
        content: str, 
        speaker_id: str, 
        speaker_name: str,
        scene_id: str = None,
        round_num: int = None,
        response_to: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加对话记忆
        
        Args:
            content: 对话内容
            speaker_id: 说话者ID
            speaker_name: 说话者名称
            scene_id: 场景ID
            round_num: 轮次编号
            response_to: 回复的消息ID
            metadata: 额外元数据
            
        Returns:
            str: 对话记忆ID
        """
        # 生成嵌入向量
        embedding = None
        if self.embedding_model:
            try:
                embedding = self._get_embedding(content)
            except Exception as e:
                print(f"生成对话嵌入向量失败: {str(e)}")
        
        # 评估对话的重要性
        importance = self._evaluate_dialogue_importance(content, speaker_id)
        
        # 创建对话记忆
        memory = DialogueMemory(
            content=content,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            scene_id=scene_id,
            round_num=round_num,
            response_to=response_to,
            embedding=embedding,
            metadata=metadata
        )
        memory.importance = importance
        
        # 添加到记忆库
        return self.add_memory(memory)
    
    def add_summary_memory(
        self,
        content: str,
        summary_type: str,
        author_id: str = None,
        round_num: int = None,
        related_dialogue_ids: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加总结记忆
        
        Args:
            content: 总结内容
            summary_type: 总结类型，'round', 'subtopic', 'final'
            author_id: 作者ID
            round_num: 轮次编号
            related_dialogue_ids: 相关对话ID列表
            metadata: 额外元数据
            
        Returns:
            str: 总结记忆ID
        """
        # 生成嵌入向量
        embedding = None
        if self.embedding_model:
            try:
                embedding = self._get_embedding(content)
            except Exception as e:
                print(f"生成总结嵌入向量失败: {str(e)}")
        
        # 提取关键点
        key_points = self._extract_key_points(content)
        
        # 创建总结记忆
        memory = SummaryMemory(
            content=content,
            summary_type=summary_type,
            author_id=author_id,
            round_num=round_num,
            related_dialogue_ids=related_dialogue_ids,
            key_points=key_points,
            embedding=embedding,
            metadata=metadata
        )
        
        # 添加到记忆库
        return self.add_memory(memory)
    
    def add_task_memory(
        self,
        title: str,
        description: str,
        assignees: List[str] = None,
        priority: str = "中",
        status: str = "待开始",
        due_date: str = None,
        dependencies: List[str] = None,
        source_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加任务记忆
        
        Args:
            title: 任务标题
            description: 任务描述
            assignees: 任务负责人ID列表
            priority: 优先级：'高', '中', '低'
            status: 任务状态
            due_date: 截止日期
            dependencies: 依赖任务ID列表
            source_id: 来源ID，可能是总结或讨论
            metadata: 额外元数据
            
        Returns:
            str: 任务记忆ID
        """
        # 生成嵌入向量
        content = f"{title}: {description}"
        embedding = None
        if self.embedding_model:
            try:
                embedding = self._get_embedding(content)
            except Exception as e:
                print(f"生成任务嵌入向量失败: {str(e)}")
        
        # 创建任务记忆
        memory = TaskMemory(
            title=title,
            description=description,
            assignees=assignees,
            priority=priority,
            status=status,
            due_date=due_date,
            dependencies=dependencies,
            source_id=source_id,
            embedding=embedding,
            metadata=metadata
        )
        
        # 添加到记忆库
        return self.add_memory(memory)
    
    def add_literature_memory(
        self,
        title: str,
        authors: List[str] = None,
        year: str = None,
        abstract: str = None,
        url: str = None,
        key_findings: List[str] = None,
        related_topics: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加文献记忆
        
        Args:
            title: 文献标题
            authors: 作者列表
            year: 出版年份
            abstract: 摘要
            url: 链接
            key_findings: 关键发现列表
            related_topics: 相关主题列表
            metadata: 额外元数据
            
        Returns:
            str: 文献记忆ID
        """
        # 生成嵌入向量
        content = f"{title}"
        if abstract:
            content += f"\n{abstract}"
        
        embedding = None
        if self.embedding_model:
            try:
                embedding = self._get_embedding(content)
            except Exception as e:
                print(f"生成文献嵌入向量失败: {str(e)}")
        
        # 如果没有提供关键发现，尝试提取
        if not key_findings and abstract:
            key_findings = self._extract_literature_key_findings(title, abstract)
        
        # 创建文献记忆
        memory = LiteratureMemory(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            url=url,
            key_findings=key_findings,
            related_topics=related_topics,
            embedding=embedding,
            metadata=metadata
        )
        
        # 添加到记忆库
        return self.add_memory(memory)
    
    def add_concept_memory(
        self,
        name: str,
        definition: str,
        related_concepts: List[str] = None,
        examples: List[str] = None,
        source_ids: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加概念记忆
        
        Args:
            name: 概念名称
            definition: 概念定义
            related_concepts: 相关概念列表
            examples: 概念示例列表
            source_ids: 来源ID列表
            metadata: 额外元数据
            
        Returns:
            str: 概念记忆ID
        """
        # 生成嵌入向量
        content = f"{name}: {definition}"
        embedding = None
        if self.embedding_model:
            try:
                embedding = self._get_embedding(content)
            except Exception as e:
                print(f"生成概念嵌入向量失败: {str(e)}")
        
        # 创建概念记忆
        memory = ConceptMemory(
            name=name,
            definition=definition,
            related_concepts=related_concepts,
            examples=examples,
            source_ids=source_ids,
            embedding=embedding,
            metadata=metadata
        )
        
        # 添加到记忆库
        return self.add_memory(memory)
    
    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """
        根据ID获取记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            Optional[MemoryNode]: 记忆对象，不存在则返回None
        """
        memory = self.memories.get(memory_id)
        if memory:
            memory.update_access()  # 更新访问记录
        return memory
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            updates: 要更新的属性字典
            
        Returns:
            bool: 更新是否成功
        """
        memory = self.memories.get(memory_id)
        if not memory:
            return False
            
        # 应用更新
        for key, value in updates.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
                # 同步更新元数据
                if key in memory.metadata:
                    memory.metadata[key] = value
            
        # 更新访问记录
        memory.update_access()
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 删除是否成功
        """
        memory = self.memories.get(memory_id)
        if not memory:
            return False
            
        # 从主存储中删除
        del self.memories[memory_id]
        
        # 从类型索引中删除
        if memory.memory_type in self.memory_by_type:
            if memory_id in self.memory_by_type[memory.memory_type]:
                del self.memory_by_type[memory.memory_type][memory_id]
        
        # 从场景索引中删除
        if isinstance(memory, DialogueMemory) and memory.scene_id:
            if memory_id in self.dialogue_by_scene.get(memory.scene_id, []):
                self.dialogue_by_scene[memory.scene_id].remove(memory_id)
        
        # 从轮次索引中删除
        round_num = None
        if isinstance(memory, DialogueMemory):
            round_num = memory.round_num
        elif isinstance(memory, SummaryMemory):
            round_num = memory.round_num
            
        if round_num is not None and round_num in self.memory_by_round:
            if memory_id in self.memory_by_round[round_num]:
                self.memory_by_round[round_num].remove(memory_id)
        
        return True
    
    def search_memories(
        self,
        query: str,
        memory_types: List[str] = None,
        top_k: int = 5,
        scene_id: str = None,
        round_num: int = None,
        recency_bias: float = 0.2
    ) -> List[Tuple[MemoryNode, float]]:
        """
        搜索记忆
        
        Args:
            query: 搜索查询
            memory_types: 搜索的记忆类型列表，None表示搜索所有类型
            top_k: 返回的最大结果数量
            scene_id: 限定搜索的场景ID
            round_num: 限定搜索的轮次
            recency_bias: 时间偏好因子，控制最近记忆的权重
            
        Returns:
            List[Tuple[MemoryNode, float]]: 匹配的记忆及其相似度分数
        """
        # 如果没有嵌入模型，使用关键词匹配
        if not self.embedding_model:
            return self._keyword_search(query, memory_types, top_k, scene_id, round_num)
        
        # 生成查询嵌入
        query_embedding = None
        try:
            query_embedding = self._get_embedding(query)
        except Exception as e:
            print(f"生成查询嵌入向量失败: {str(e)}")
            return self._keyword_search(query, memory_types, top_k, scene_id, round_num)
        
        # 确定目标记忆池
        target_memories = {}
        if memory_types:
            for mem_type in memory_types:
                if mem_type in self.memory_by_type:
                    target_memories.update(self.memory_by_type[mem_type])
        else:
            target_memories = self.memories
        
        # 进一步筛选记忆池
        filtered_memories = {}
        for mid, memory in target_memories.items():
            # 筛选场景
            if scene_id and isinstance(memory, DialogueMemory) and memory.scene_id != scene_id:
                continue
            
            # 筛选轮次
            if round_num is not None:
                if isinstance(memory, DialogueMemory) and memory.round_num != round_num:
                    continue
                if isinstance(memory, SummaryMemory) and memory.round_num != round_num:
                    continue
            
            # 添加到筛选后的记忆池
            filtered_memories[mid] = memory
        
        # 计算相似度并排序
        results = []
        current_time = time.time()
        
        for memory in filtered_memories.values():
            if memory.embedding is None:
                continue
                
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            
            # 应用时间衰减和重要性加权
            time_factor = 1.0
            if recency_bias > 0:
                age = current_time - memory.creation_time
                time_factor = max(0.5, 1.0 - recency_bias * min(1.0, age / (24 * 3600)))  # 最多1天的时间衰减
            
            # 最终分数 = 相似度 * 时间因子 * 重要性
            final_score = similarity * time_factor * memory.importance
            
            results.append((memory, final_score))
        
        # 筛选并排序结果
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # 更新访问记录
        for memory, _ in top_results:
            memory.update_access()
        
        return top_results
    
    def get_dialogue_history(
        self, 
        scene_id: str, 
        count: int = 10, 
        speaker_id: str = None,
        round_num: int = None
    ) -> List[DialogueMemory]:
        """
        获取对话历史
        
        Args:
            scene_id: 场景ID
            count: 返回的最大对话数量
            speaker_id: 筛选特定说话者的对话
            round_num: 筛选特定轮次的对话
            
        Returns:
            List[DialogueMemory]: 对话记忆列表
        """
        # 获取场景的所有对话ID
        dialogue_ids = self.dialogue_by_scene.get(scene_id, [])
        
        # 收集对话记忆
        dialogues = []
        for did in dialogue_ids:
            memory = self.memories.get(did)
            if not memory or not isinstance(memory, DialogueMemory):
                continue
                
            # 筛选说话者
            if speaker_id and memory.speaker_id != speaker_id:
                continue
                
            # 筛选轮次
            if round_num is not None and memory.round_num != round_num:
                continue
                
            dialogues.append(memory)
        
        # 按时间戳排序
        dialogues.sort(key=lambda x: x.timestamp)
        
        # 取最近的count条对话
        recent_dialogues = dialogues[-count:]
        
        # 更新访问记录
        for dialogue in recent_dialogues:
            dialogue.update_access()
            
        return recent_dialogues
    
    def get_round_summary(self, round_num: int) -> Optional[SummaryMemory]:
        """
        获取特定轮次的总结
        
        Args:
            round_num: 轮次编号
            
        Returns:
            Optional[SummaryMemory]: 轮次总结，如果不存在则返回None
        """
        # 获取轮次的所有记忆ID
        memory_ids = self.memory_by_round.get(round_num, [])
        
        # 查找轮次总结
        for mid in memory_ids:
            memory = self.memories.get(mid)
            if (memory and 
                isinstance(memory, SummaryMemory) and 
                memory.summary_type == 'round' and
                memory.round_num == round_num):
                
                # 更新访问记录
                memory.update_access()
                return memory
        
        return None
    
    def get_tasks_by_status(self, status: str = None) -> List[TaskMemory]:
        """
        按状态获取任务
        
        Args:
            status: 任务状态，None表示所有状态
            
        Returns:
            List[TaskMemory]: 任务记忆列表
        """
        tasks = []
        
        # 获取所有任务记忆
        for memory in self.memory_by_type[MemoryType.TASK].values():
            if not isinstance(memory, TaskMemory):
                continue
                
            # 筛选状态
            if status and memory.status != status:
                continue
                
            tasks.append(memory)
            
            # 更新访问记录
            memory.update_access()
        
        return tasks
    
    def extract_key_information(self, text: str, topic: str = None) -> Dict[str, Any]:
        """
        从文本中提取关键信息
        
        Args:
            text: 输入文本
            topic: 相关主题，可以帮助LLM理解上下文
            
        Returns:
            Dict[str, Any]: 提取的关键信息
        """
        # 构建提示
        topic_context = f"关于主题「{topic}」的" if topic else ""
        prompt = f"""
        请从以下{topic_context}文本中提取关键信息，并按以下JSON格式返回结果：
        
        文本内容:
        ```
        {text}
        ```
        
        请提取以下信息：
        1. key_points: 主要观点（最多5点）
        2. questions: 提出的问题（如有）
        3. conclusions: 得出的结论（如有）
        4. concepts: 提及的重要概念（如有）
        5. action_items: 提及的行动项（如有）
        
        请以JSON格式返回，确保格式正确。
        """
        
        messages = [
            {"role": "system", "content": "你是一个擅长从学术讨论中提取关键信息的AI助手。请精确提取信息，返回有效JSON格式。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM提取关键信息
        try:
            response_text = self.llm.get_response(messages)
            # 尝试解析JSON响应
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果无法识别JSON，返回原始文本作为key_points
                return {
                    "key_points": [response_text],
                    "questions": [],
                    "conclusions": [],
                    "concepts": [],
                    "action_items": []
                }
        except Exception as e:
            print(f"提取关键信息失败: {str(e)}")
            return {
                "key_points": ["无法提取关键信息"],
                "error": str(e)
            }
    
    def extract_concepts_from_text(self, text: str, topic: str = None) -> List[Dict[str, Any]]:
        """
        从文本中提取概念
        
        Args:
            text: 输入文本
            topic: 相关主题，可以帮助LLM理解上下文
            
        Returns:
            List[Dict[str, Any]]: 提取的概念列表
        """
        # 构建提示
        topic_context = f"关于「{topic}」的" if topic else ""
        prompt = f"""
        请从以下{topic_context}学术讨论文本中提取重要的专业概念，并为每个概念提供简短定义：
        
        文本内容:
        ```
        {text}
        ```
        
        返回格式示例:
        [
          {{"name": "概念1", "definition": "概念1的简要定义"}},
          {{"name": "概念2", "definition": "概念2的简要定义"}}
        ]
        
        请仅提取文本中明确提到的专业概念，不要添加未在文本中出现的概念。返回JSON格式。
        """
        
        messages = [
            {"role": "system", "content": "你是一个擅长从学术文本中识别和定义专业概念的AI助手。请提取关键概念并提供准确定义。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM提取概念
        try:
            response_text = self.llm.get_response(messages)
            # 尝试解析JSON响应
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                concepts = json.loads(json_str)
                return concepts
            else:
                return []
        except Exception as e:
            print(f"提取概念失败: {str(e)}")
            return []
    
    def generate_review_framework(
        self, 
        topic: str, 
        subtopics: List[str] = None,
        context: str = None
    ) -> Dict[str, Any]:
        """
        为综述报告生成框架
        
        Args:
            topic: 综述主题
            subtopics: 预定义的子话题列表
            context: 提供给LLM的额外上下文
            
        Returns:
            Dict[str, Any]: 综述框架，包含结构和子话题
        """
        # 构建提示
        subtopics_text = ""
        if subtopics:
            subtopics_text = "以下是预定义的子话题:\n" + "\n".join([f"- {s}" for s in subtopics])
        else:
            subtopics_text = "请生成3-5个合适的子话题。"
            
        context_text = f"\n以下是相关背景信息:\n{context}" if context else ""
            
        prompt = f"""
        请为"{topic}"学术综述报告设计一个详细的框架。
        
        {subtopics_text}
        {context_text}
        
        请返回以下格式的JSON：
        {{
          "title": "综述标题",
          "subtopics": [
            {{
              "title": "子话题1标题",
              "key_aspects": ["关键方面1", "关键方面2"],
              "research_questions": ["研究问题1", "研究问题2"]
            }},
            ...
          ],
          "overall_structure": ["第1部分: 简介", "第2部分: ...", ...],
          "research_gaps": ["研究空白1", "研究空白2"]
        }}
        
        请确保框架全面、结构合理，能够系统性地覆盖该主题的各个重要方面。
        """
        
        messages = [
            {"role": "system", "content": "你是一位经验丰富的学术编辑，擅长设计高质量的学术综述框架。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM生成框架
        try:
            response_text = self.llm.get_response(messages)
            # 尝试解析JSON响应
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                framework = json.loads(json_str)
                
                # 保存框架作为概念记忆
                framework_text = f"{topic}综述框架: " + ", ".join([st.get("title", "") for st in framework.get("subtopics", [])])
                self.add_concept_memory(
                    name=f"{topic}综述框架",
                    definition=framework_text,
                    related_concepts=[topic],
                    metadata={"framework": framework}
                )
                
                return framework
            else:
                print("生成综述框架失败: 无法解析JSON响应")
                return {
                    "title": topic,
                    "subtopics": [{"title": "概述", "key_aspects": [], "research_questions": []}],
                    "overall_structure": ["引言", "主体", "结论"],
                    "research_gaps": []
                }
        except Exception as e:
            print(f"生成综述框架失败: {str(e)}")
            return {
                "title": topic,
                "subtopics": [{"title": "概述", "key_aspects": [], "research_questions": []}],
                "overall_structure": ["引言", "主体", "结论"],
                "research_gaps": []
            }
    
    def summarize_dialogues(
        self, 
        dialogue_ids: List[str], 
        summary_type: str = "round",
        topic: str = None,
        round_num: int = None
    ) -> str:
        """
        基于对话生成总结
        
        Args:
            dialogue_ids: 对话记忆ID列表
            summary_type: 总结类型，'round', 'subtopic', 'final'
            topic: 总结主题
            round_num: 轮次编号
            
        Returns:
            str: 总结内容
        """
        # 收集对话
        dialogues = []
        for did in dialogue_ids:
            memory = self.get_memory(did)
            if memory and isinstance(memory, DialogueMemory):
                dialogues.append(memory)
        
        if not dialogues:
            return "无对话内容可总结"
            
        # 格式化对话历史
        formatted_dialogues = []
        for d in dialogues:
            formatted_dialogues.append(f"{d.speaker_name}: {d.content}")
        
        dialogue_text = "\n".join(formatted_dialogues)
        
        # 构建总结提示
        topic_text = f"关于「{topic}」的" if topic else ""
        round_text = f"第{round_num}轮" if round_num is not None else ""
        
        summary_prompts = {
            "round": f"请总结以下{topic_text}{round_text}学术讨论的要点，包括主要观点、讨论问题和达成的共识：",
            "subtopic": f"请全面总结以下关于子话题「{topic}」的学术讨论，包括核心概念、研究现状、关键技术、挑战和未来方向：",
            "final": f"请对整个「{topic}」的学术讨论进行综合总结，涵盖研究背景、主要发现、理论框架、应用场景、未来研究方向和建议："
        }
        
        prompt = summary_prompts.get(summary_type, summary_prompts["round"]) + f"\n\n{dialogue_text}"
        
        messages = [
            {"role": "system", "content": "你是一位擅长总结学术讨论的AI助手，能够提取关键观点并形成结构化总结。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM生成总结
        try:
            summary = self.llm.get_response(messages)
            
            # 提取关键点
            key_info = self.extract_key_information(summary, topic)
            
            # 保存总结记忆
            summary_id = self.add_summary_memory(
                content=summary,
                summary_type=summary_type,
                round_num=round_num,
                related_dialogue_ids=dialogue_ids,
                metadata={"key_info": key_info, "topic": topic}
            )
            
            # 从总结中提取概念
            concepts = self.extract_concepts_from_text(summary, topic)
            for concept in concepts:
                self.add_concept_memory(
                    name=concept.get("name", ""),
                    definition=concept.get("definition", ""),
                    related_concepts=[topic] if topic else [],
                    source_ids=[summary_id]
                )
            
            return summary
        except Exception as e:
            print(f"生成总结失败: {str(e)}")
            return "生成总结时出错"
    
    def extract_tasks_from_summary(self, summary_id: str) -> List[str]:
        """
        从总结中提取任务
        
        Args:
            summary_id: 总结记忆ID
            
        Returns:
            List[str]: 提取并添加的任务记忆ID列表
        """
        # 获取总结记忆
        memory = self.get_memory(summary_id)
        if not memory or not isinstance(memory, SummaryMemory):
            print(f"记忆 {summary_id} 不存在或不是总结类型")
            return []
            
        # 提取任务
        prompt = f"""
        请从以下学术讨论总结中提取具体的研究任务：
        
        {memory.content}
        
        请识别3-5个明确的研究任务，每个任务包括：
        1. 任务标题
        2. 详细描述
        3. 优先级（高/中/低）
        4. 适合执行该任务的角色类型（教授/博士生/硕士生）
        
        以JSON格式返回，例如：
        [
          {{
            "title": "任务标题1",
            "description": "详细描述...",
            "priority": "高/中/低",
            "role_type": "教授/博士生/硕士生"
          }},
          ...
        ]
        """
        
        messages = [
            {"role": "system", "content": "你是一个擅长从学术讨论中提取研究任务的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM提取任务
        try:
            response_text = self.llm.get_response(messages)
            
            # 尝试解析JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                tasks_data = json.loads(json_str)
                
                # 创建任务记忆
                task_ids = []
                for task_data in tasks_data:
                    task_id = self.add_task_memory(
                        title=task_data.get("title", "未命名任务"),
                        description=task_data.get("description", ""),
                        priority=task_data.get("priority", "中"),
                        source_id=summary_id,
                        metadata={"role_type": task_data.get("role_type", "")}
                    )
                    task_ids.append(task_id)
                
                return task_ids
            else:
                print("提取任务失败: 无法解析JSON响应")
                return []
        except Exception as e:
            print(f"提取任务失败: {str(e)}")
            return []
    
    def forget_memories(self, cutoff_time: float = None, threshold: float = 0.3, memory_type: str = None):
        """
        遗忘低重要性的记忆
        
        Args:
            cutoff_time: 截止时间戳，None表示基于重要性阈值
            threshold: 重要性阈值，低于此值的记忆会被遗忘
            memory_type: 限定遗忘的记忆类型
        """
        to_forget = []
        
        # 遍历所有记忆
        for memory_id, memory in self.memories.items():
            # 按类型筛选
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # 按时间筛选
            if cutoff_time and memory.creation_time > cutoff_time:
                continue
                
            # 计算当前有效重要性
            current_importance = self._calculate_effective_importance(memory)
            
            # 低于阈值则遗忘
            if current_importance < threshold:
                to_forget.append(memory_id)
        
        # 执行遗忘
        for memory_id in to_forget:
            self.delete_memory(memory_id)
            
        print(f"遗忘了 {len(to_forget)} 条记忆")
    
    def save_memories(self, filename: str = None):
        """
        保存记忆到文件
        
        Args:
            filename: 文件名，None则使用时间戳
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.persistence_dir, f"memories_{timestamp}.json")
        
        # 将记忆转换为可序列化格式
        serializable_memories = {
            "dialogue": [],
            "summary": [],
            "task": [],
            "literature": [],
            "concept": []
        }
        
        # 处理对话记忆
        for memory in self.memory_by_type.get(MemoryType.DIALOGUE, {}).values():
            if isinstance(memory, DialogueMemory):
                serializable_memories["dialogue"].append(memory.to_dict())
        
        # 处理总结记忆
        for memory in self.memory_by_type.get(MemoryType.SUMMARY, {}).values():
            if isinstance(memory, SummaryMemory):
                serializable_memories["summary"].append(memory.to_dict())
        
        # 处理任务记忆
        for memory in self.memory_by_type.get(MemoryType.TASK, {}).values():
            if isinstance(memory, TaskMemory):
                serializable_memories["task"].append(memory.to_dict())
        
        # 处理文献记忆
        for memory in self.memory_by_type.get(MemoryType.LITERATURE, {}).values():
            if isinstance(memory, LiteratureMemory):
                serializable_memories["literature"].append(memory.to_dict())
        
        # 处理概念记忆
        for memory in self.memory_by_type.get(MemoryType.CONCEPT, {}).values():
            if isinstance(memory, ConceptMemory):
                serializable_memories["concept"].append(memory.to_dict())
        
        # 保存到文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serializable_memories, f, ensure_ascii=False, indent=2)
            
        print(f"记忆已保存至 {filename}")
    
    def load_memories(self, filename: str):
        """
        从文件加载记忆
        
        Args:
            filename: 记忆文件名
            
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 处理对话记忆
            for item in data.get("dialogue", []):
                memory = DialogueMemory(
                    content=item.get("content", ""),
                    speaker_id=item.get("speaker_id", ""),
                    speaker_name=item.get("speaker_name", ""),
                    scene_id=item.get("scene_id"),
                    round_num=item.get("round_num"),
                    response_to=item.get("response_to"),
                    metadata=item.get("metadata", {})
                )
                memory.id = item.get("id", memory.id)
                memory.importance = item.get("importance", memory.importance)
                memory.timestamp = item.get("timestamp", memory.timestamp)
                memory.creation_time = item.get("creation_time", memory.creation_time)
                memory.last_access_time = item.get("last_access_time", memory.last_access_time)
                memory.access_count = item.get("access_count", memory.access_count)
                
                # 添加到记忆库
                self.memories[memory.id] = memory
                self.memory_by_type[MemoryType.DIALOGUE][memory.id] = memory
                if memory.scene_id:
                    self.dialogue_by_scene[memory.scene_id].append(memory.id)
                if memory.round_num is not None:
                    self.memory_by_round[memory.round_num].append(memory.id)
            
            # 处理总结记忆
            for item in data.get("summary", []):
                memory = SummaryMemory(
                    content=item.get("content", ""),
                    summary_type=item.get("summary_type", ""),
                    author_id=item.get("author_id"),
                    round_num=item.get("round_num"),
                    related_dialogue_ids=item.get("related_dialogue_ids", []),
                    key_points=item.get("key_points", []),
                    metadata=item.get("metadata", {})
                )
                memory.id = item.get("id", memory.id)
                memory.importance = item.get("importance", memory.importance)
                memory.timestamp = item.get("timestamp", memory.timestamp)
                memory.creation_time = item.get("creation_time", memory.creation_time)
                memory.last_access_time = item.get("last_access_time", memory.last_access_time)
                memory.access_count = item.get("access_count", memory.access_count)
                
                # 添加到记忆库
                self.memories[memory.id] = memory
                self.memory_by_type[MemoryType.SUMMARY][memory.id] = memory
                if memory.round_num is not None:
                    self.memory_by_round[memory.round_num].append(memory.id)
            
            # 处理任务记忆
            for item in data.get("task", []):
                memory = TaskMemory(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    assignees=item.get("assignees", []),
                    priority=item.get("priority", "中"),
                    status=item.get("status", "待开始"),
                    due_date=item.get("due_date"),
                    dependencies=item.get("dependencies", []),
                    source_id=item.get("source_id"),
                    metadata=item.get("metadata", {})
                )
                memory.id = item.get("id", memory.id)
                memory.importance = item.get("importance", memory.importance)
                memory.timestamp = item.get("timestamp", memory.timestamp)
                memory.creation_time = item.get("creation_time", memory.creation_time)
                memory.last_access_time = item.get("last_access_time", memory.last_access_time)
                memory.access_count = item.get("access_count", memory.access_count)
                
                # 添加到记忆库
                self.memories[memory.id] = memory
                self.memory_by_type[MemoryType.TASK][memory.id] = memory
            
            # 处理文献记忆
            for item in data.get("literature", []):
                memory = LiteratureMemory(
                    title=item.get("title", ""),
                    authors=item.get("authors", []),
                    year=item.get("year"),
                    abstract=item.get("abstract"),
                    url=item.get("url"),
                    key_findings=item.get("key_findings", []),
                    related_topics=item.get("related_topics", []),
                    metadata=item.get("metadata", {})
                )
                memory.id = item.get("id", memory.id)
                memory.importance = item.get("importance", memory.importance)
                memory.timestamp = item.get("timestamp", memory.timestamp)
                memory.creation_time = item.get("creation_time", memory.creation_time)
                memory.last_access_time = item.get("last_access_time", memory.last_access_time)
                memory.access_count = item.get("access_count", memory.access_count)
                
                # 添加到记忆库
                self.memories[memory.id] = memory
                self.memory_by_type[MemoryType.LITERATURE][memory.id] = memory
            
            # 处理概念记忆
            for item in data.get("concept", []):
                memory = ConceptMemory(
                    name=item.get("name", ""),
                    definition=item.get("definition", ""),
                    related_concepts=item.get("related_concepts", []),
                    examples=item.get("examples", []),
                    source_ids=item.get("source_ids", []),
                    metadata=item.get("metadata", {})
                )
                memory.id = item.get("id", memory.id)
                memory.importance = item.get("importance", memory.importance)
                memory.timestamp = item.get("timestamp", memory.timestamp)
                memory.creation_time = item.get("creation_time", memory.creation_time)
                memory.last_access_time = item.get("last_access_time", memory.last_access_time)
                memory.access_count = item.get("access_count", memory.access_count)
                
                # 添加到记忆库
                self.memories[memory.id] = memory
                self.memory_by_type[MemoryType.CONCEPT][memory.id] = memory
                if memory.source_ids:
                    for source_id in memory.source_ids:
                        if source_id not in self.memory_by_type[MemoryType.DIALOGUE]:
                            self.memory_by_type[MemoryType.DIALOGUE][source_id] = []
                        self.memory_by_type[MemoryType.DIALOGUE][source_id].append(memory.id)
            print(f"记忆已从 {filename} 加载")
            return True
        except Exception as e:
            print(f"加载记忆失败: {str(e)}")
            return False