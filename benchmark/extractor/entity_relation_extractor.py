import logging
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from benchmark.llm_utils.llm_client import LLMClient
from dev.memory_unit import MemoryUnit
from dev.semantic_graph import SemanticGraph

@dataclass
class Entity:
    """实体数据类"""
    name: str
    entity_type: str
    description: str
    source_text: str = ""
    confidence: float = 1.0

@dataclass
class Relationship:
    """关系数据类"""
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    keywords: List[str]
    strength: float
    source_text: str = ""

class EntityRelationExtractor:
    """实体关系抽取器"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.entity_types = ["person", "organization", "geo", "event", "category"]
        self.record_delimiter = "##"
        self.completion_delimiter = "<|COMPLETE|>"
        
        # 加载提示词模板
        self.prompt_template = self._load_prompt_template()
        
    # def _load_prompt_template(self) -> str:
    #     """加载提示词模板"""
    #     template = """---目标---
    #     从给定的文本文档中识别所有指定类型的实体，并识别实体间的关系。
    #     使用中文作为输出语言。

    #     ---步骤---
    #     1. 识别所有实体。对于每个识别出的实体，提取以下信息：
    #     - entity_name: 实体名称，使用与输入文本相同的语言
    #     - entity_type: 实体类型，必须是以下类型之一: {entity_types}
    #     - entity_description: 实体属性和活动的综合描述
    #     格式化每个实体为 ("entity"<|><entity_name><|><entity_type><|><entity_description>)

    #     2. 从步骤1识别的实体中，识别所有明确相关的实体对(source_entity, target_entity)。
    #     对于每对相关实体，提取以下信息：
    #     - source_entity: 源实体名称，如步骤1中识别的
    #     - target_entity: 目标实体名称，如步骤1中识别的
    #     - relationship_description: 解释为什么源实体和目标实体相关
    #     - relationship_strength: 表示实体间关系强度的数值分数(0-1)
    #     - relationship_keywords: 总结关系整体性质的高级关键词
    #     格式化每个关系为 ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

    #     3. 识别总结整个文本主要概念、主题或话题的高级关键词。
    #     格式化内容级关键词为 ("content_keywords"<|><high_level_keywords>)

    #     4. 使用 **{record_delimiter}** 作为列表分隔符返回步骤1和2中识别的所有实体和关系的单一列表。

    #     5. 完成后输出 {completion_delimiter}

    #     ######################
    #     ---真实数据---
    #     ######################
    #     实体类型: {entity_types}
    #     文本: {input_text}
    #     ######################
    #     输出:"""
    #     return template
    def _load_prompt_template(self,input_text=None,entity_types=None,record_delimiter=None,completion_delimiter=None) -> str:
        """加载提示词模板"""
        if not entity_types:
            entity_types = "不限制实体类型"
        if not record_delimiter:
            record_delimiter = "##"
        if not completion_delimiter:
            completion_delimiter = "<|COMPLETE|>"
        template = f"""---目标---
            从给定的文本文档中识别所有指定类型的实体，并识别实体间的关系。
            使用中文作为输出语言。

            ---步骤---
            1. 识别所有实体。对于每个识别出的实体，提取以下信息：
            - entity_name: 实体名称，使用与输入文本相同的语言
            - entity_type: 实体类型，必须是以下类型之一: {entity_types}
            - entity_description: 实体属性和活动的综合描述
            格式化每个实体为 ("entity"<|><entity_name><|><entity_type><|><entity_description>)

            2. 从步骤1识别的实体中，识别所有明确相关的实体对(source_entity, target_entity)。
            对于每对相关实体，提取以下信息：
            - source_entity: 源实体名称，如步骤1中识别的
            - target_entity: 目标实体名称，如步骤1中识别的
            - relationship_type: 关系类型，建议使用以下类型之一: FAMILY, WORK, FRIEND, LOCATION, TEMPORAL, TOPIC, RELATED_TO
            - relationship_description: 解释为什么源实体和目标实体相关
            - relationship_strength: 表示实体间关系强度的数值分数(0.0-1.0)
            格式化每个关系为 ("relationship"<|><source_entity><|><target_entity><|><relationship_type><|><relationship_description><|><relationship_strength>)

            3. 识别总结整个文本主要概念、主题或话题的高级关键词。
            格式化内容级关键词为 ("content_keywords"<|><high_level_keywords>)

            4. 使用 **{record_delimiter}** 作为列表分隔符返回步骤1和2中识别的所有实体和关系的单一列表。

            5. 完成后输出 {completion_delimiter}

            ---示例---
            文本: "Caroline住在纽约，是一名心理咨询师。Melanie住在加州，是艺术家。她们是好朋友。"

            输出:
            ("entity"<|>Caroline<|>person<|>住在纽约的心理咨询师)##
            ("entity"<|>Melanie<|>person<|>住在加州的艺术家)##
            ("entity"<|>纽约<|>geo<|>城市地点)##
            ("entity"<|>加州<|>geo<|>州地点)##
            ("relationship"<|>Caroline<|>Melanie<|>FRIEND<|>她们是好朋友<|>0.9)##
            ("relationship"<|>Caroline<|>纽约<|>LOCATION<|>Caroline住在纽约<|>0.8)##
            ("relationship"<|>Melanie<|>加州<|>LOCATION<|>Melanie住在加州<|>0.8)##
            ("content_keywords"<|>朋友关系,职业,地理位置)##
            <|COMPLETE|>

            ######################
            ---真实数据---
            ######################
            实体类型: {entity_types}
            文本: {input_text}
            ######################
            输出:"""
        return template
    
    def extract_entities_and_relations(self, text: str, max_retries: int = 3) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """
        从文本中抽取实体和关系
        
        Args:
            text: 输入文本
            max_retries: 最大重试次数
            
        Returns:
            (entities, relationships, content_keywords)
        """
        if not text or not text.strip():
            logging.warning("输入文本为空，跳过实体关系抽取")
            return [], [], []
        
        # 构建提示词
        # prompt = self.prompt_template.format(
        #     input_text=text[:2000],  # 限制输入长度
        #     entity_types=str(self.entity_types),
        #     record_delimiter=self.record_delimiter,
        #     completion_delimiter=self.completion_delimiter
        # )
        prompt = self._load_prompt_template(
            input_text=text[:2000],  # 限制输入长度
            entity_types=str(self.entity_types),
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter
        )
        self.prompt_template = prompt  # 更新提示词模板
        
        logging.info(f"使用提示词: {prompt[:500]}...")  # 调试：打印提示词前500个字符
        
        for attempt in range(max_retries):
            try:
                # 调用LLM
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=2000
                )

                # 调试：打印LLM响应
                logging.info(f"LLM响应 (尝试 {attempt + 1}): {response[:500]}...")
                
                # 解析响应
                entities, relationships, content_keywords = self._parse_llm_response(response, text)
                
                if entities or relationships:  # 只要有一个成功就返回
                    logging.info(f"成功抽取 {len(entities)} 个实体, {len(relationships)} 个关系")
                    return entities, relationships, content_keywords
                    
            except Exception as e:
                logging.error(f"实体关系抽取第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    logging.error("实体关系抽取最终失败")
        
        return [], [], []
    
    def _parse_llm_response(self, response: str, source_text: str) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """解析LLM响应"""
        entities = []
        relationships = []
        content_keywords = []
        
        # 清理响应文本
        response = response.strip()
        if self.completion_delimiter in response:
            response = response.split(self.completion_delimiter)[0]
        
        # 按分隔符分割
        lines = response.split(self.record_delimiter)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                if line.startswith('("entity"'):
                    entity = self._parse_entity_line(line, source_text)
                    if entity:
                        entities.append(entity)
                        
                elif line.startswith('("relationship"'):
                    relationship = self._parse_relationship_line(line, source_text)
                    if relationship:
                        relationships.append(relationship)
                        
                elif line.startswith('("content_keywords"'):
                    keywords = self._parse_content_keywords_line(line)
                    content_keywords.extend(keywords)
                    
            except Exception as e:
                logging.warning(f"解析行失败: {line[:50]}... 错误: {e}")
                continue
        
        return entities, relationships, content_keywords
    
    def _parse_entity_line(self, line: str, source_text: str) -> Optional[Entity]:
        """解析实体行"""
        # 使用正则表达式提取实体信息
        pattern = r'\("entity"<\|>([^<]+)<\|>([^<]+)<\|>([^)]+)\)'
        match = re.search(pattern, line)
        
        if match:
            name = match.group(1).strip()
            entity_type = match.group(2).strip().lower()
            description = match.group(3).strip()
            
            # 验证实体类型
            if entity_type not in self.entity_types:
                logging.warning(f"无效的实体类型: {entity_type}")
                return None
            
            return Entity(
                name=name,
                entity_type=entity_type,
                description=description,
                source_text=source_text[:200],
                confidence=0.8
            )
        
        return None
    
    # def _parse_relationship_line(self, line: str, source_text: str) -> Optional[Relationship]:
    #     """解析关系行"""
    #     pattern = r'\("relationship"<\|>([^<]+)<\|>([^<]+)<\|>([^<]+)<\|>([^<]+)<\|>([^)]+)\)'
    #     match = re.search(pattern, line)
        
    #     if match:
    #         source_entity = match.group(1).strip()
    #         target_entity = match.group(2).strip()
    #         description = match.group(3).strip()
    #         keywords_str = match.group(4).strip()
    #         strength_str = match.group(5).strip()
            
    #         # 解析关键词
    #         keywords = [kw.strip() for kw in keywords_str.split(',')]
            
    #         # 解析强度
    #         try:
    #             strength = float(strength_str)
    #             strength = max(0.0, min(1.0, strength))  # 限制在0-1之间
    #         except:
    #             strength = 0.5  # 默认值
            
    #         # 推断关系类型
    #         relationship_type = self._infer_relationship_type(keywords, description)
            
    #         return Relationship(
    #             source_entity=source_entity,
    #             target_entity=target_entity,
    #             relationship_type=relationship_type,
    #             description=description,
    #             keywords=keywords,
    #             strength=strength,
    #             source_text=source_text[:200]
    #         )
        
    #     return None
    def _parse_relationship_line(self, line: str, source_text: str) -> Optional[Relationship]:
        """解析关系行"""
        # 新的模式：包含显式的关系类型
        pattern = r'\("relationship"<\|>([^<]+)<\|>([^<]+)<\|>([^<]+)<\|>([^<]+)<\|>([^)]+)\)'
        match = re.search(pattern, line)
        
        if match:
            source_entity = match.group(1).strip()
            target_entity = match.group(2).strip()
            relationship_type = match.group(3).strip()
            description = match.group(4).strip()
            strength_str = match.group(5).strip()
            
            # 解析强度
            try:
                strength = float(strength_str)
                strength = max(0.0, min(1.0, strength))  # 限制在0-1之间
            except:
                strength = 0.5  # 默认值
            
            # 验证关系类型
            valid_types = ['FAMILY', 'WORK', 'FRIEND', 'LOCATION', 'TEMPORAL', 'TOPIC', 'RELATED_TO']
            if relationship_type not in valid_types:
                relationship_type = 'RELATED_TO'  # 默认类型
            
            return Relationship(
                source_entity=source_entity,
                target_entity=target_entity,
                relationship_type=relationship_type,
                description=description,
                keywords=[relationship_type.lower()],  # 简化关键词
                strength=strength,
                source_text=source_text[:200]
            )
        
        return None
    
    def _parse_content_keywords_line(self, line: str) -> List[str]:
        """解析内容关键词行"""
        pattern = r'\("content_keywords"<\|>([^)]+)\)'
        match = re.search(pattern, line)
        
        if match:
            keywords_str = match.group(1).strip()
            keywords = [kw.strip() for kw in keywords_str.split(',')]
            return [kw for kw in keywords if kw]
        
        return []
    
    def _infer_relationship_type(self, keywords: List[str], description: str) -> str:
        """根据关键词和描述推断关系类型"""
        # 关系类型映射
        type_patterns = {
            'FAMILY': ['家庭', '父母', '子女', '夫妻', '兄弟', '姐妹', '亲属'],
            'WORK': ['工作', '同事', '合作', '职业', '公司', '组织'],
            'FRIEND': ['朋友', '友谊', '社交', '伙伴'],
            'LOCATION': ['位置', '地点', '居住', '访问', '旅行'],
            'TEMPORAL': ['时间', '日期', '之前', '之后', '同时'],
            'TOPIC': ['讨论', '话题', '主题', '内容', '关于'],
            'RELATED_TO': ['相关', '关联', '有关']
        }
        
        # 组合所有文本进行匹配
        combined_text = ' '.join(keywords) + ' ' + description
        combined_text = combined_text.lower()
        
        for rel_type, patterns in type_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return rel_type
        
        return 'RELATED_TO'  # 默认关系类型