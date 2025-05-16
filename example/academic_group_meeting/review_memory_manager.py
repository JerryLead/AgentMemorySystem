from collections import defaultdict
from datetime import datetime
import os
from typing import Any, Dict, List
from memory_manager import MemoryManager, MemoryType, DialogueMemory, SummaryMemory, TaskMemory, LiteratureMemory

class ReviewMemoryManager(MemoryManager):
    """专为学术综述设计的记忆管理器，对MemoryManager进行扩展"""
    
    def __init__(self, topic: str, subtopics: List[str] = None, **kwargs):
        """初始化综述记忆管理器
        
        Args:
            topic: 综述主题
            subtopics: 子话题列表
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.topic = topic
        self.subtopics = subtopics or []
        self.round_dialogues = defaultdict(list)  # 记录每轮讨论的对话ID
        self.scene_id = None  # 记录当前场景ID
        
        # 为综述创建基本框架概念
        self._initialize_review_framework()
    
    def _initialize_review_framework(self):
        """初始化综述框架，创建相关概念记忆"""
        # 生成综述框架
        framework = self.generate_review_framework(
            topic=self.topic,
            subtopics=self.subtopics
        )
        
        # 记录综述主题作为核心概念
        self.add_concept_memory(
            name=self.topic,
            definition=f"本综述的核心主题，探讨{self.topic}的各个方面。",
            related_concepts=self.subtopics
        )
        
        # 为每个子话题创建概念记忆
        for subtopic in self.subtopics:
            self.add_concept_memory(
                name=subtopic,
                definition=f"{self.topic}的一个重要方面：{subtopic}",
                related_concepts=[self.topic]
            )
    
    def track_discussion_round(self, round_num: int, subtopic: str):
        """跟踪讨论轮次
        
        Args:
            round_num: 当前轮次编号
            subtopic: 当前讨论的子话题
        """
        # 创建轮次概念记忆
        self.add_concept_memory(
            name=f"第{round_num}轮讨论：{subtopic}",
            definition=f"关于{subtopic}的学术讨论",
            related_concepts=[self.topic, subtopic]
        )
    
    def add_discussion_dialogue(self, speaker_id: str, speaker_name: str, 
                               content: str, round_num: int):
        """添加讨论对话记忆
        
        Args:
            speaker_id: 发言者ID
            speaker_name: 发言者名称
            content: 发言内容
            round_num: 当前轮次
            
        Returns:
            str: 对话记忆ID
        """
        dialogue_id = self.add_dialogue_memory(
            content=content,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            scene_id=self.scene_id,
            round_num=round_num
        )
        
        # 记录到轮次对话列表
        self.round_dialogues[round_num].append(dialogue_id)
        return dialogue_id
    
    def add_literature_reference(self, paper_info: Dict[str, Any], related_subtopic: str):
        """添加文献引用记忆
        
        Args:
            paper_info: 论文信息字典
            related_subtopic: 相关子话题
            
        Returns:
            str: 文献记忆ID
        """
        return self.add_literature_memory(
            title=paper_info.get("Title", "未知标题"),
            authors=paper_info.get("Authors", []),
            year=paper_info.get("Year", ""),
            abstract=paper_info.get("Abstract", ""),
            url=paper_info.get("URL", ""),
            key_findings=paper_info.get("KeyFindings", []),
            related_topics=[self.topic, related_subtopic],
            metadata={"subtopic": related_subtopic, "source": paper_info.get("Source", "")}
        )
    
    def add_open_source_system(self, system_info: Dict[str, Any], related_subtopic: str):
        """添加开源系统记忆
        
        Args:
            system_info: 系统信息字典
            related_subtopic: 相关子话题
            
        Returns:
            str: 概念记忆ID
        """
        return self.add_concept_memory(
            name=system_info.get("name", "未命名系统"),
            definition=system_info.get("description", ""),
            related_concepts=[self.topic, related_subtopic, "开源系统"],
            metadata={
                "url": system_info.get("url", ""),
                "subtopic": related_subtopic
            }
        )
    
    def summarize_round_discussion(self, round_num: int, subtopic: str):
        """总结某轮讨论内容
        
        Args:
            round_num: 轮次编号
            subtopic: 讨论的子话题
            
        Returns:
            str: 生成的总结内容
        """
        # 获取该轮讨论的所有对话ID
        dialogue_ids = self.round_dialogues.get(round_num, [])
        if not dialogue_ids:
            return f"第{round_num}轮关于{subtopic}的讨论暂无记录"
            
        # 调用总结方法
        summary = self.summarize_dialogues(
            dialogue_ids=dialogue_ids,
            summary_type="round",
            topic=subtopic,
            round_num=round_num
        )
        
        return summary
    
    def generate_review_summary(self):
        """生成综述总结，整合所有轮次的讨论
        
        Returns:
            str: 综述总结内容
        """
        # 收集所有轮次总结
        round_summaries = []
        for round_num in sorted(self.round_dialogues.keys()):
            summary = self.get_round_summary(round_num)
            if summary:
                round_summaries.append(summary.content)
        
        # 如果没有找到任何轮次总结，直接返回一个基本总结
        if not round_summaries:
            return f"关于{self.topic}的综述总结：尚未收集到足够的讨论信息。"
        
        # 构造总结提示
        prompt = f"""
        请基于以下各轮讨论的总结，生成一份关于"{self.topic}"的全面、系统的学术综述总结：
        
        {"\n\n".join([f"第{i+1}轮讨论总结:\n{summary}" for i, summary in enumerate(round_summaries)])}
        
        请确保总结内容:
        1. 系统全面地覆盖主题的各个关键方面
        2. 强调主要研究进展和关键发现
        3. 指出研究空白和未来方向
        4. 结构清晰，逻辑连贯
        5. 突出各子话题间的联系和影响
        """
        
        messages = [
            {"role": "system", "content": "你是一位经验丰富的学术编辑，善于整合多方面信息编写高质量的学术综述。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM生成最终总结
        try:
            final_summary = self.llm.get_response(messages)
            
            # 存储为最终总结记忆
            summary_id = self.add_summary_memory(
                content=final_summary,
                summary_type="final",
                metadata={"topic": self.topic}
            )
            
            # 提取可能的研究任务
            self.extract_tasks_from_summary(summary_id)
            
            return final_summary
        except Exception as e:
            print(f"生成综述总结失败: {str(e)}")
            return "无法生成综述总结"
    
    def generate_structured_review_report(self, output_dir: str = None):
        """生成结构化综述报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 包含报告内容和元数据的字典
        """
        # 收集所有记忆数据
        literature_memories = list(self.memory_by_type[MemoryType.LITERATURE].values())
        summary_memories = list(self.memory_by_type[MemoryType.SUMMARY].values())
        concept_memories = list(self.memory_by_type[MemoryType.CONCEPT].values())
        
        # 找到最终总结
        final_summary = None
        for memory in summary_memories:
            if isinstance(memory, SummaryMemory) and memory.summary_type == "final":
                final_summary = memory
                break
        
        # 如果没有最终总结，先生成一个
        if not final_summary:
            final_summary_content = self.generate_review_summary()
            final_summary_id = self.add_summary_memory(
                content=final_summary_content,
                summary_type="final",
                metadata={"topic": self.topic}
            )
            final_summary = self.get_memory(final_summary_id)
        
        # 构建报告结构
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"review_{timestamp}"
        report_filename = f"{self.topic.replace(' ', '_')}_{timestamp}.md"
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, report_filename)
        else:
            report_path = report_filename
        
        # 构造报告内容
        report_content = [
            f"# {self.topic} 综述报告",
            f"\n## 摘要\n",
            final_summary.content if final_summary else f"关于{self.topic}的综述",
            "\n## 目录\n",
        ]
        
        # 添加目录
        for i, subtopic in enumerate(self.subtopics):
            report_content.append(f"{i+1}. [{subtopic}](#{''.join(subtopic.lower().split(' '))})") 
        
        # 添加各子话题内容
        for i, subtopic in enumerate(self.subtopics):
            report_content.append(f"\n## {i+1}. {subtopic}<a name='{''.join(subtopic.lower().split(' '))}'></a>\n")
            
            # 查找该子话题的轮次总结
            subtopic_summary = None
            for memory in summary_memories:
                if (isinstance(memory, SummaryMemory) and 
                    memory.summary_type == "round" and
                    memory.metadata.get("topic") == subtopic):
                    subtopic_summary = memory
                    break
            
            # 添加子话题内容
            if subtopic_summary:
                report_content.append(subtopic_summary.content)
            else:
                report_content.append(f"关于{subtopic}的讨论内容待完善。")
            
            # 添加相关文献
            related_literature = []
            for memory in literature_memories:
                if isinstance(memory, LiteratureMemory) and subtopic in memory.related_topics:
                    related_literature.append(memory)
            
            if related_literature:
                report_content.append("\n### 相关文献\n")
                for lit in related_literature:
                    authors = ", ".join(lit.authors) if lit.authors else "未知作者"
                    report_content.append(f"- {lit.title} ({lit.year or '未知年份'}) - {authors}")
                    if lit.key_findings:
                        report_content.append(f"  - 主要发现: {'; '.join(lit.key_findings[:3])}")
        
        # 添加研究展望
        report_content.append("\n## 研究展望\n")
        
        # 获取任务作为研究展望
        tasks = list(self.memory_by_type[MemoryType.TASK].values())
        if tasks:
            for task in tasks[:5]:  # 取前5个任务
                if isinstance(task, TaskMemory):
                    report_content.append(f"- **{task.title}**: {task.description}")
        else:
            report_content.append(f"对于{self.topic}领域，未来研究可以进一步探索上述子话题中的开放问题。")
        
        # 添加总结
        report_content.append("\n## 总结\n")
        if final_summary:
            key_points = final_summary.key_points
            if key_points:
                report_content.append("本综述的主要贡献包括:")
                for point in key_points:
                    report_content.append(f"- {point}")
            else:
                # 从内容中提取最后一段作为结论
                conclusion_paragraphs = final_summary.content.split("\n\n")[-2:]
                report_content.extend(conclusion_paragraphs)
        
        # 保存报告
        full_report = "\n\n".join(report_content)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(full_report)
            print(f"综述报告已保存至: {report_path}")
        except Exception as e:
            print(f"保存综述报告失败: {str(e)}")
        
        # 保存所有记忆
        memory_path = os.path.join(os.path.dirname(report_path), f"memories_{timestamp}.json")
        self.save_memories(memory_path)
        
        # 返回报告信息
        return {
            "report_id": report_id,
            "filename": report_path,
            "topic": self.topic,
            "subtopics": self.subtopics,
            "content": full_report,
            "memory_file": memory_path,
            "timestamp": timestamp
        }