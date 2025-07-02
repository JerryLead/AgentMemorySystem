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
        # self.record_delimiter = "##"
        # self.completion_delimiter = "<|COMPLETE|>"
        # 不在初始化时加载模板，而是在使用时动态生成
        # # 加载提示词模板
        # self.prompt_template = self._load_prompt_template()

    # 在 EntityRelationExtractor 类中添加以下方法

    def estimate_token_count(self, text: str) -> int:
        """估算文本的token数量"""
        # 对于中文和英文混合文本的粗略估算
        # 1个中文字符 ≈ 1 token
        # 1个英文单词 ≈ 1.3 tokens
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len(text.replace(' ', '').replace('\n', '')) - chinese_chars
        return chinese_chars + int(english_words * 1.3)

    def smart_text_chunking(self, text: str, max_tokens: int = 60000) -> List[str]:
        """智能文本分块，保持语义完整性"""
        if self.estimate_token_count(text) <= max_tokens:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # 检查添加这个段落是否会超过限制
            test_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
            
            if self.estimate_token_count(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 检查单个段落是否太长
                if self.estimate_token_count(paragraph) > max_tokens:
                    # 进一步分割长段落
                    sentences = self.split_long_paragraph(paragraph, max_tokens)
                    chunks.extend(sentences[:-1])
                    current_chunk = sentences[-1] if sentences else ""
                else:
                    current_chunk = paragraph
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def split_long_paragraph(self, paragraph: str, max_tokens: int) -> List[str]:
        """分割过长的段落"""
        sentences = []
        current_sentence = ""
        
        # 按句子分割（中英文标点）
        import re
        sentence_endings = re.split(r'([。！？.!?])', paragraph)
        
        for i in range(0, len(sentence_endings)-1, 2):
            sentence = sentence_endings[i] + (sentence_endings[i+1] if i+1 < len(sentence_endings) else "")
            test_sentence = current_sentence + sentence
            
            if self.estimate_token_count(test_sentence) <= max_tokens:
                current_sentence = test_sentence
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = sentence
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences

    def extract_entities_and_relations_chunked(self, text: str, max_retries: int = 3) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """
        从长文本中分块抽取实体和关系，然后合并结果
        """
        if not text or not text.strip():
            logging.warning("输入文本为空，跳过实体关系抽取")
            return [], [], []
        
        # 分块处理
        chunks = self.smart_text_chunking(text, max_tokens=60000)
        logging.info(f"将文本分为 {len(chunks)} 个块进行处理")
        
        all_entities = []
        all_relationships = []
        all_keywords = []
        
        for i, chunk in enumerate(chunks):
            logging.info(f"处理第 {i+1}/{len(chunks)} 块，长度: {len(chunk)} 字符")
            
            # 对每个块进行实体抽取
            entities, relationships, keywords = self.extract_entities_and_relations(
                chunk, max_retries=max_retries
            )
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            all_keywords.extend(keywords)
        
        # 合并和去重
        merged_entities, merged_relationships, merged_keywords = self.merge_extraction_results(
            all_entities, all_relationships, all_keywords
        )
        
        logging.info(f"合并后结果: {len(merged_entities)} 个实体, {len(merged_relationships)} 个关系")
        
        return merged_entities, merged_relationships, merged_keywords

    def merge_extraction_results(self, entities: List[Entity], relationships: List[Relationship], 
                            keywords: List[str]) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """合并多个块的抽取结果，去重和规范化"""
        
        # 实体去重（基于名称和类型）
        entity_dict = {}
        for entity in entities:
            key = (entity.name.lower().strip(), entity.entity_type)
            if key not in entity_dict:
                entity_dict[key] = entity
            else:
                # 合并描述信息
                existing = entity_dict[key]
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
                existing.confidence = max(existing.confidence, entity.confidence)
        
        # 关系去重（基于源实体、目标实体和关系类型）
        relationship_dict = {}
        for rel in relationships:
            key = (rel.source_entity.lower().strip(), 
                rel.target_entity.lower().strip(), 
                rel.relationship_type)
            if key not in relationship_dict:
                relationship_dict[key] = rel
            else:
                # 保留强度更高的关系
                existing = relationship_dict[key]
                if rel.strength > existing.strength:
                    relationship_dict[key] = rel
        
        # 关键词去重
        unique_keywords = list(set([kw.strip() for kw in keywords if kw.strip()]))
        
        return (list(entity_dict.values()), 
                list(relationship_dict.values()), 
                unique_keywords)
    
    def _load_prompt_template(self, input_text, entity_types=None) -> str:
        """优化的提示词模板，支持长文本处理"""
        # if not input_text:
        #     print("输入文本为空，无法生成提示词模板")
        #     return ""
        if not entity_types:
            entity_types = "不限制实体类型"

        template = f"""---目标---
        从给定的文本文档中识别所有指定类型的实体，并识别实体间的关系。

        ---要求---
        请严格按照以下JSON格式输出结果，不要添加任何额外的文字说明：

        {{
        "entities": [
            {{
            "name": "实体名称",
            "type": "实体类型(person/organization/geo/event/category)",
            "description": "实体描述"
            }}
        ],
        "relationships": [
            {{
            "source": "源实体名称",
            "target": "目标实体名称", 
            "type": "关系类型(FAMILY/WORK/FRIEND/LOCATION/TEMPORAL/TOPIC/RELATED_TO)",
            "description": "关系描述",
            "strength": 0.8
            }}
        ],
        "keywords": ["关键词1", "关键词2"]
        }}

        ---示例---
        文本: "Caroline住在纽约，是心理咨询师。Melanie住在加州，是艺术家。她们是好朋友。"

        输出:
        {{
        "entities": [
            {{"name": "Caroline", "type": "person", "description": "住在纽约的心理咨询师"}},
            {{"name": "Melanie", "type": "person", "description": "住在加州的艺术家"}},
            {{"name": "纽约", "type": "geo", "description": "城市"}},
            {{"name": "加州", "type": "geo", "description": "州"}}
        ],
        "relationships": [
            {{"source": "Caroline", "target": "Melanie", "type": "FRIEND", "description": "好朋友关系", "strength": 0.9}},
            {{"source": "Caroline", "target": "纽约", "type": "LOCATION", "description": "居住地", "strength": 0.8}},
            {{"source": "Melanie", "target": "加州", "type": "LOCATION", "description": "居住地", "strength": 0.8}}
        ],
        "keywords": ["朋友关系", "职业", "地理位置"]
        }}

        ---实际任务---
        文本: {input_text}
        实体类型限制: {entity_types}

        请输出JSON格式的结果:"""
            
        return template
 
    # 修改 extract_entities_and_relations 方法
    def extract_entities_and_relations_with_position(self, text: str, position_map: Dict[str, Tuple[int, int]] = None, max_retries: int = 3) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """
        从文本中抽取实体和关系，并追踪源位置
        """
        if not text or not text.strip():
            logging.warning("输入文本为空，跳过实体关系抽取")
            return [], [], []
        
        # 检查文本长度
        estimated_tokens = self.estimate_token_count(text)
        
        if estimated_tokens > 60000:
            logging.info(f"文本过长({estimated_tokens} tokens)，使用分块处理")
            return self.extract_entities_and_relations_chunked_with_position(text, position_map, max_retries)
        
        # 正常处理流程
        prompt = self._load_prompt_template(
            input_text=text,
            entity_types=str(self.entity_types)
        )
        
        for attempt in range(max_retries):
            try:
                max_tokens = min(4000, max(1000, estimated_tokens // 10))
                
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                # TODO: 如果使用deepseek-chat，可以启用JSON响应格式,但是也要相应修改处理response的逻辑
                # response = self.llm_client.generate_answer(
                #     prompt=prompt,
                #     temperature=0.1,
                #     max_tokens=max_tokens,
                #     json_response=True  # 使用JSON响应格式
                # )

                logging.info(f"LLM响应 (尝试 {attempt + 1}): {response[:500]}...")
                
                entities, relationships, content_keywords = self._parse_llm_response_with_position(
                    response, text, position_map
                )
                
                if entities or relationships:
                    logging.info(f"成功抽取 {len(entities)} 个实体, {len(relationships)} 个关系")
                    return entities, relationships, content_keywords
                    
            except Exception as e:
                logging.error(f"实体关系抽取第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    logging.error("实体关系抽取最终失败")
        
        return [], [], []
    
    def _parse_llm_response_with_position(self, response: str, source_text: str, position_map: Dict[str, Tuple[int, int]] = None) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """解析JSON格式的LLM响应，并追踪源位置"""
        entities = []
        relationships = []
        content_keywords = []
        
        try:
            # 清理响应，提取JSON部分
            cleaned_response = self._extract_json_from_response(response)
            
            # 解析JSON
            data = json.loads(cleaned_response)
            
            # 解析实体
            if "entities" in data:
                for entity_data in data["entities"]:
                    try:
                        entity_name = entity_data.get("name", "").strip()
                        
                        # 寻找实体在原文中的位置
                        source_info = self._find_entity_source_position(entity_name, source_text, position_map)
                        
                        entity = Entity(
                            name=entity_name,
                            entity_type=self._normalize_entity_type(entity_data.get("type", "").strip()),
                            description=entity_data.get("description", "").strip(),
                            source_text=source_info["source_text"],
                            confidence=0.8
                        )
                        if entity.name and entity.entity_type:
                            entities.append(entity)
                    except Exception as e:
                        logging.warning(f"解析实体失败: {entity_data} - {e}")
            
            # 解析关系
            if "relationships" in data:
                for rel_data in data["relationships"]:
                    try:
                        source_entity = rel_data.get("source", "").strip()
                        target_entity = rel_data.get("target", "").strip()
                        
                        # 寻找关系在原文中的位置（基于两个实体）
                        source_info = self._find_relationship_source_position(
                            source_entity, target_entity, source_text, position_map
                        )
                        
                        relationship = Relationship(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relationship_type=self._normalize_relationship_type(rel_data.get("type", "").strip()),
                            description=rel_data.get("description", "").strip(),
                            keywords=[rel_data.get("type", "").lower()],
                            strength=float(rel_data.get("strength", 0.5)),
                            source_text=source_info["source_text"]
                        )
                        if relationship.source_entity and relationship.target_entity:
                            relationships.append(relationship)
                    except Exception as e:
                        logging.warning(f"解析关系失败: {rel_data} - {e}")
            
            # 解析关键词
            if "keywords" in data:
                content_keywords = [kw.strip() for kw in data["keywords"] if isinstance(kw, str) and kw.strip()]
            
            logging.info(f"JSON解析成功: {len(entities)} 个实体, {len(relationships)} 个关系")
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}")
            logging.debug(f"原始响应: {response}")
            return self._parse_llm_response_fallback(response, source_text)
        except Exception as e:
            logging.error(f"响应解析异常: {e}")
            return [], [], []
        
        return entities, relationships, content_keywords

    def _find_entity_source_position(self, entity_name: str, full_text: str, position_map: Dict[str, Tuple[int, int]] = None) -> Dict[str, str]:
        """寻找实体在原文中的具体位置"""
        if not position_map:
            return {"source_text": full_text[:200] + "..." if len(full_text) > 200 else full_text}
        
        # 在完整文本中搜索实体名称
        entity_positions = []
        start = 0
        while True:
            pos = full_text.find(entity_name, start)
            if pos == -1:
                break
            entity_positions.append(pos)
            start = pos + 1
        
        if not entity_positions:
            return {"source_text": f"实体 '{entity_name}' 未在原文中找到"}
        
        # 为每个找到的位置，确定它属于哪个消息片段
        best_match = None
        for entity_pos in entity_positions:
            for segment_key, (start_pos, end_pos, metadata) in position_map.items():
                if start_pos <= entity_pos < end_pos:
                    # 提取包含实体的具体对话片段
                    segment_text = full_text[start_pos:end_pos].strip()
                    
                    source_info = {
                        "source_text": segment_text,
                        "segment_key": segment_key,
                        "position": entity_pos - start_pos,
                        **metadata
                    }
                    
                    # 优先选择会话对话而不是摘要
                    if segment_key.startswith('session_') and 'msg_' in segment_key:
                        best_match = source_info
                        break
                    elif not best_match:
                        best_match = source_info
            
            if best_match and best_match.get("segment_key", "").startswith('session_'):
                break
        
        if best_match:
            return best_match
        else:
            return {"source_text": full_text[:200] + "..." if len(full_text) > 200 else full_text}

    def _find_relationship_source_position(self, source_entity: str, target_entity: str, full_text: str, position_map: Dict[str, Tuple[int, int]] = None) -> Dict[str, str]:
        """寻找关系在原文中的具体位置"""
        if not position_map:
            return {"source_text": full_text[:200] + "..." if len(full_text) > 200 else full_text}
        
        # 寻找同时包含两个实体的文本片段
        best_match = None
        best_score = 0
        
        for segment_key, (start_pos, end_pos, metadata) in position_map.items():
            segment_text = full_text[start_pos:end_pos]
            
            # 计算匹配得分
            score = 0
            if source_entity.lower() in segment_text.lower():
                score += 1
            if target_entity.lower() in segment_text.lower():
                score += 1
            
            # 优先选择会话对话
            if segment_key.startswith('session_') and 'msg_' in segment_key:
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_match = {
                    "source_text": segment_text.strip(),
                    "segment_key": segment_key,
                    "match_score": score,
                    **metadata
                }
        
        if best_match and best_match.get("match_score", 0) >= 1:
            return best_match
        else:
            # 如果没找到好的匹配，返回包含任一实体的片段
            for segment_key, (start_pos, end_pos, metadata) in position_map.items():
                segment_text = full_text[start_pos:end_pos]
                if (source_entity.lower() in segment_text.lower() or 
                    target_entity.lower() in segment_text.lower()):
                    return {
                        "source_text": segment_text.strip(),
                        "segment_key": segment_key,
                        **metadata
                    }
            
            return {"source_text": full_text[:200] + "..." if len(full_text) > 200 else full_text}
        
    # 在 EntityRelationExtractor 类中添加以下方法（在 extract_entities_and_relations_chunked 方法之后）

    def extract_entities_and_relations_chunked_with_position(self, text: str, position_map: Dict[str, Tuple[int, int]] = None, max_retries: int = 3) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """
        从长文本中分块抽取实体和关系，并追踪源位置，然后合并结果
        """
        if not text or not text.strip():
            logging.warning("输入文本为空，跳过实体关系抽取")
            return [], [], []
        
        # 分块处理
        chunks = self.smart_text_chunking(text, max_tokens=60000)
        logging.info(f"将文本分为 {len(chunks)} 个块进行处理")
        
        all_entities = []
        all_relationships = []
        all_keywords = []
        
        # 为每个分块创建对应的位置映射
        chunk_position_maps = self._split_position_map_for_chunks(text, chunks, position_map)
        
        for i, (chunk, chunk_position_map) in enumerate(zip(chunks, chunk_position_maps)):
            logging.info(f"处理第 {i+1}/{len(chunks)} 块，长度: {len(chunk)} 字符")
            
            # 对每个块进行实体抽取（带位置追踪）
            entities, relationships, keywords = self._extract_from_single_chunk_with_position(
                chunk, chunk_position_map, max_retries=max_retries
            )
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            all_keywords.extend(keywords)
        
        # 合并和去重
        merged_entities, merged_relationships, merged_keywords = self.merge_extraction_results(
            all_entities, all_relationships, all_keywords
        )
        
        logging.info(f"合并后结果: {len(merged_entities)} 个实体, {len(merged_relationships)} 个关系")
        
        return merged_entities, merged_relationships, merged_keywords

    def _split_position_map_for_chunks(self, full_text: str, chunks: List[str], position_map: Dict[str, Tuple[int, int]] = None) -> List[Dict[str, Tuple[int, int]]]:
        """
        为每个文本块创建对应的位置映射
        
        Args:
            full_text: 完整文本
            chunks: 文本块列表
            position_map: 原始位置映射
            
        Returns:
            每个块对应的位置映射列表
        """
        if not position_map:
            return [None] * len(chunks)
        
        chunk_position_maps = []
        current_position = 0
        
        for chunk in chunks:
            chunk_map = {}
            chunk_start = full_text.find(chunk, current_position)
            chunk_end = chunk_start + len(chunk)
            
            # 找出与当前块重叠的位置映射项
            for segment_key, (start_pos, end_pos, metadata) in position_map.items():
                # 检查是否与当前块有重叠
                if not (end_pos <= chunk_start or start_pos >= chunk_end):
                    # 计算在块内的相对位置
                    relative_start = max(0, start_pos - chunk_start)
                    relative_end = min(len(chunk), end_pos - chunk_start)
                    
                    if relative_end > relative_start:
                        chunk_map[segment_key] = (relative_start, relative_end, metadata)
            
            chunk_position_maps.append(chunk_map if chunk_map else None)
            current_position = chunk_end
        
        return chunk_position_maps

    def _extract_from_single_chunk_with_position(self, text: str, position_map: Dict[str, Tuple[int, int]] = None, max_retries: int = 3) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """
        从单个文本块中抽取实体和关系，并追踪源位置
        """
        prompt = self._load_prompt_template(
            input_text=text,
            entity_types=str(self.entity_types)
        )
        
        estimated_tokens = self.estimate_token_count(text)
        
        for attempt in range(max_retries):
            try:
                max_tokens = min(4000, max(1000, estimated_tokens // 10))
                
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=max_tokens
                )

                logging.info(f"LLM响应 (尝试 {attempt + 1}): {response[:500]}...")
                
                entities, relationships, content_keywords = self._parse_llm_response_with_position(
                    response, text, position_map
                )
                
                if entities or relationships:
                    logging.info(f"成功抽取 {len(entities)} 个实体, {len(relationships)} 个关系")
                    return entities, relationships, content_keywords
                    
            except Exception as e:
                logging.error(f"实体关系抽取第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    logging.error("实体关系抽取最终失败")
        
        return [], [], []
    
    def extract_entities_and_relations(self, text: str, max_retries: int = 3) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """
        从文本中抽取实体和关系，自动处理长文本
        Args:
            text: 输入文本
            max_retries: 最大重试次数
        """
        if not text or not text.strip():
            logging.warning("输入文本为空，跳过实体关系抽取")
            return [], [], []
        
        # 检查文本长度，决定是否需要分块处理
        estimated_tokens = self.estimate_token_count(text)
        
        if estimated_tokens > 60000:  # 超过阈值时使用分块处理
            logging.info(f"文本过长({estimated_tokens} tokens)，使用分块处理")
            return self.extract_entities_and_relations_chunked(text, max_retries)
        
        # 正常处理流程
        prompt = self._load_prompt_template(
            input_text=text,
            entity_types=str(self.entity_types),
            # record_delimiter=self.record_delimiter,
            # completion_delimiter=self.completion_delimiter
        )
        
        for attempt in range(max_retries):
            try:
                # 根据文本长度调整参数
                max_tokens = min(4000, max(1000, estimated_tokens // 10))
                
                # response = self.llm_client.generate_answer(
                #     prompt=prompt,
                #     temperature=0.1,
                #     max_tokens=max_tokens,
                #     # json_response=True  # 使用JSON响应格式
                # )

                # TODO: 如果使用deepseek-chat，可以启用JSON响应格式,但是也要相应修改处理response的逻辑
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    json_response=True  # 使用JSON响应格式
                )

                logging.info(f"LLM响应 (尝试 {attempt + 1}): {response[:500]}...")
                
                # entities, relationships, content_keywords = self._parse_llm_response(response, text)
                entities, relationships, content_keywords = self._parse_llm_response(response, text)
                
                if entities or relationships:
                    logging.info(f"成功抽取 {len(entities)} 个实体, {len(relationships)} 个关系")
                    return entities, relationships, content_keywords
                    
            except Exception as e:
                logging.error(f"实体关系抽取第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    logging.error("实体关系抽取最终失败")
        
        return [], [], []
    
    def _parse_llm_response(self, response: str, source_text: str) -> Tuple[List[Entity], List[Relationship], List[str]]:
        """解析JSON格式的LLM响应"""
        entities = []
        relationships = []
        content_keywords = []
        
        try:
            # # 清理响应，提取JSON部分
            # 使用新接口后不需要提取JSON部分，直接使用响应内容
            # cleaned_response = self._extract_json_from_response(response)
            # # 解析JSON
            # data = json.loads(cleaned_response)

            # 解析JSON
            data = json.loads(response)
            
            # 解析实体
            if "entities" in data:
                for entity_data in data["entities"]:
                    try:
                        entity = Entity(
                            name=entity_data.get("name", "").strip(),
                            entity_type=self._normalize_entity_type(entity_data.get("type", "").strip()),
                            description=entity_data.get("description", "").strip(),
                            source_text=source_text[:200],
                            confidence=0.8
                        )
                        if entity.name and entity.entity_type:
                            entities.append(entity)
                    except Exception as e:
                        logging.warning(f"解析实体失败: {entity_data} - {e}")
            
            # 解析关系
            if "relationships" in data:
                for rel_data in data["relationships"]:
                    try:
                        relationship = Relationship(
                            source_entity=rel_data.get("source", "").strip(),
                            target_entity=rel_data.get("target", "").strip(),
                            relationship_type=self._normalize_relationship_type(rel_data.get("type", "").strip()),
                            description=rel_data.get("description", "").strip(),
                            keywords=[rel_data.get("type", "").lower()],
                            strength=float(rel_data.get("strength", 0.5)),
                            source_text=source_text[:200]
                        )
                        if relationship.source_entity and relationship.target_entity:
                            relationships.append(relationship)
                    except Exception as e:
                        logging.warning(f"解析关系失败: {rel_data} - {e}")
            
            # 解析关键词
            if "keywords" in data:
                content_keywords = [kw.strip() for kw in data["keywords"] if isinstance(kw, str) and kw.strip()]
            
            logging.info(f"JSON解析成功: {len(entities)} 个实体, {len(relationships)} 个关系")
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}")
            logging.debug(f"原始响应: {response}")
            # 回退到改进的正则表达式解析
            return self._parse_llm_response_fallback(response, source_text)
        except Exception as e:
            logging.error(f"响应解析异常: {e}")
            return [], [], []
        
        return entities, relationships, content_keywords

    def _extract_json_from_response(self, response: str) -> str:
        """从LLM响应中提取JSON部分"""
        # 查找JSON开始和结束位置
        start_markers = ['{', '```json\n{', '```\n{']
        end_markers = ['}', '}\n```', '}```']
        
        for start_marker in start_markers:
            start_idx = response.find(start_marker)
            if start_idx != -1:
                # 找到开始位置，现在找结束位置
                json_start = start_idx + len(start_marker) - 1  # 保留 {
                
                # 寻找匹配的闭合括号
                brace_count = 0
                json_end = json_start
                
                for i, char in enumerate(response[json_start:], json_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count == 0:
                    return response[json_start:json_end]
        
        # 如果没找到完整的JSON，尝试修复
        return self._try_fix_json(response)

    def _try_fix_json(self, response: str) -> str:
        """尝试修复不完整的JSON"""
        # 移除markdown代码块标记
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # 寻找第一个 { 和最后一个 }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return response[first_brace:last_brace + 1]
        
        raise ValueError("无法提取有效的JSON")
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """标准化实体类型"""
        type_mapping = {
            'location': 'geo',
            'place': 'geo',
            'concept': 'category',
            '人': 'person',
            '人物': 'person',
            '地点': 'geo',
            '组织': 'organization',
            '事件': 'event'
        }
        
        entity_type = entity_type.lower().strip()
        return type_mapping.get(entity_type, entity_type if entity_type in self.entity_types else 'category')

    def _normalize_relationship_type(self, rel_type: str) -> str:
        """标准化关系类型"""
        valid_types = ['FAMILY', 'WORK', 'FRIEND', 'LOCATION', 'TEMPORAL', 'TOPIC', 'RELATED_TO']
        rel_type = rel_type.upper().strip()
        return rel_type if rel_type in valid_types else 'RELATED_TO'
    