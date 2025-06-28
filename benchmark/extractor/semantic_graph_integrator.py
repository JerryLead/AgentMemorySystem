import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import List, Dict, Any, Optional
from dev.semantic_graph import SemanticGraph
from dev.memory_unit import MemoryUnit
from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor, Entity, Relationship
from benchmark.llm_utils.llm_client import LLMClient


class SemanticGraphIntegrator:
    """增强的语义图集成器 - 直接使用SemanticGraph接口"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        
        # 创建专用空间
        self._setup_entity_spaces()
        
    def _setup_entity_spaces(self):
        """设置实体和关系的专用空间"""
        # 为不同实体类型创建子空间
        entity_types = ["person", "organization", "geo", "event", "category"]
        for entity_type in entity_types:
            space_name = f"entities_{entity_type}"
            self.graph.create_memory_space_in_map(space_name)
        
        # 创建通用实体空间
        self.graph.create_memory_space_in_map("extracted_entities")
    
    def add_entities_to_graph(self, entities: List[Entity], source_unit_id: str) -> Dict[str, str]:
        """
        将实体添加到语义图中，直接使用SemanticGraph接口
        
        Args:
            entities: 实体列表
            source_unit_id: 源文档单元ID
            
        Returns:
            实体名称到单元ID的映射
        """
        entity_id_map = {}
        
        for entity in entities:
            try:
                # 生成规范化的实体单元ID
                entity_uid = f"entity_{entity.entity_type}_{self._normalize_name(entity.name)}"
                
                # 检查实体是否已存在
                existing_unit = self.graph.get_unit(entity_uid)
                if existing_unit:
                    # 如果已存在，更新映射并跳过
                    entity_id_map[entity.name] = entity_uid
                    logging.debug(f"实体 {entity.name} 已存在，跳过创建")
                    continue
                
                # 创建实体内存单元
                entity_unit = MemoryUnit(
                    uid=entity_uid,
                    raw_data={
                        "text_content": f"{entity.name}: {entity.description}",
                        "entity_name": entity.name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "source_text": entity.source_text,
                        "confidence": entity.confidence,
                        "content_type": "extracted_entity"
                    },
                    metadata={
                        "data_source": "entity_extraction",
                        "entity_type": entity.entity_type,
                        "entity_name": entity.name,
                        "source_unit_id": source_unit_id,
                        "extraction_confidence": entity.confidence,
                        "spaces": [f"entities_{entity.entity_type}", "extracted_entities"]
                    }
                )
                
                # 使用SemanticGraph接口添加单元
                space_names = [f"entities_{entity.entity_type}", "extracted_entities"]
                self.graph.add_unit(
                    unit=entity_unit,
                    explicit_content_for_embedding=entity_unit.raw_data["text_content"],
                    content_type_for_embedding="text",
                    space_names=space_names,
                    rebuild_semantic_map_index_immediately=False
                )
                
                # 建立与源文档的关系
                if source_unit_id and self.graph.get_unit(source_unit_id):
                    success = self.graph.add_relationship(
                        source_uid=source_unit_id,
                        target_uid=entity_uid,
                        relationship_name="CONTAINS_ENTITY",
                        entity_type=entity.entity_type,
                        confidence=entity.confidence,
                        bidirectional=False
                    )
                    if success:
                        logging.debug(f"已建立源文档与实体的关系: {source_unit_id} -> {entity_uid}")
                
                entity_id_map[entity.name] = entity_uid
                logging.debug(f"已添加实体: {entity.name} ({entity.entity_type}) -> {entity_uid}")
                
            except Exception as e:
                logging.error(f"添加实体失败: {entity.name} - {e}")
                continue
        
        logging.info(f"成功添加 {len(entity_id_map)} 个实体到语义图")
        return entity_id_map
    
    def add_relationships_to_graph(self, relationships: List[Relationship], entity_id_map: Dict[str, str], source_unit_id: str) -> int:
        """
        将关系添加到语义图中，直接使用SemanticGraph接口
        
        Args:
            relationships: 关系列表
            entity_id_map: 实体名称到单元ID的映射
            source_unit_id: 源文档单元ID
            
        Returns:
            成功添加的关系数量
        """
        success_count = 0
        
        for relationship in relationships:
            try:
                # 查找实体ID
                source_entity_id = entity_id_map.get(relationship.source_entity)
                target_entity_id = entity_id_map.get(relationship.target_entity)
                
                if not source_entity_id or not target_entity_id:
                    logging.warning(f"关系中的实体未找到: {relationship.source_entity} -> {relationship.target_entity}")
                    continue
                
                # 检查关系是否已存在
                existing_rel = self.graph.get_relationship(source_entity_id, target_entity_id, relationship.relationship_type)
                if existing_rel:
                    logging.debug(f"关系已存在，跳过: {relationship.source_entity} -[{relationship.relationship_type}]-> {relationship.target_entity}")
                    success_count += 1  # 视为成功
                    continue
                
                # 使用SemanticGraph接口添加关系
                success = self.graph.add_relationship(
                    source_uid=source_entity_id,
                    target_uid=target_entity_id,
                    relationship_name=relationship.relationship_type,
                    description=relationship.description,
                    keywords=",".join(relationship.keywords),
                    strength=relationship.strength,
                    source_text=relationship.source_text,
                    source_unit_id=source_unit_id,
                    bidirectional=False
                )
                
                if success:
                    success_count += 1
                    logging.debug(f"已添加关系: {relationship.source_entity} -[{relationship.relationship_type}]-> {relationship.target_entity}")
                
            except Exception as e:
                logging.error(f"添加关系失败: {relationship.source_entity} -> {relationship.target_entity} - {e}")
                continue
        
        logging.info(f"成功添加 {success_count} 个关系到语义图")
        return success_count
    
    def process_memory_unit_for_entities(self, unit: MemoryUnit) -> Dict[str, Any]:
        """
        处理单个内存单元的实体关系抽取
        
        Args:
            unit: 内存单元
            
        Returns:
            处理结果统计
        """
        # 检查是否已处理过
        if unit.metadata.get('entities_extracted', False):
            logging.debug(f"单元 {unit.uid} 已处理过实体抽取，跳过")
            return {"skipped": True}
        
        # 获取文本内容
        text_content = unit.raw_data.get('text_content', '')
        if not text_content or len(text_content.strip()) < 10:
            logging.debug(f"单元 {unit.uid} 文本内容太短，跳过实体抽取")
            return {"skipped": True, "reason": "text_too_short"}
        
        try:
            # 初始化抽取器
            llm_client = LLMClient(model_name="deepseek-chat")
            extractor = EntityRelationExtractor(llm_client)
            
            # 抽取实体和关系
            entities, relationships, content_keywords = extractor.extract_entities_and_relations(text_content)
            
            if not entities and not relationships:
                logging.debug(f"单元 {unit.uid} 未抽取到实体或关系")
                return {"skipped": True, "reason": "no_entities_extracted"}
            
            # 添加到图中
            entity_id_map = self.add_entities_to_graph(entities, unit.uid)
            relationship_count = self.add_relationships_to_graph(relationships, entity_id_map, unit.uid)
            
            # 更新原单元的元数据，标记已处理
            unit.metadata['entities_extracted'] = True
            unit.metadata['entity_count'] = len(entities)
            unit.metadata['relationship_count'] = relationship_count
            unit.metadata['content_keywords'] = content_keywords
            
            # 重建索引（如果有新实体添加）
            if entity_id_map:
                self.graph.build_semantic_map_index()
            
            result = {
                "success": True,
                "entities_count": len(entities),
                "relationships_count": relationship_count,
                "keywords_count": len(content_keywords),
                "entity_ids": list(entity_id_map.values())
            }
            
            logging.info(f"单元 {unit.uid} 实体抽取完成: {result}")
            return result
            
        except Exception as e:
            logging.error(f"处理单元 {unit.uid} 的实体抽取失败: {e}")
            return {"success": False, "error": str(e)}
    
    def batch_extract_entities_from_space(self, 
                                        space_name: str = "locomo_dialogs",
                                        max_units: int = 50,
                                        unit_filter: Optional[callable] = None) -> Dict[str, Any]:
        """
        批量处理指定空间中的内存单元进行实体关系抽取
        
        Args:
            space_name: 要处理的空间名称
            max_units: 最大处理单元数
            unit_filter: 单元过滤函数
            
        Returns:
            批量处理结果统计
        """
        logging.info(f"开始批量实体抽取，空间: {space_name}, 最大单元数: {max_units}")
        
        # 获取指定空间中的单元
        space = self.graph.semantic_map.get_memory_space(space_name)
        if not space:
            logging.error(f"空间 {space_name} 不存在")
            return {"error": "space_not_found"}
        
        unit_ids = list(space.get_memory_uids())
        
        # 应用过滤器
        units_to_process = []
        for unit_id in unit_ids[:max_units]:
            unit = self.graph.get_unit(unit_id)
            if unit and (not unit_filter or unit_filter(unit)):
                units_to_process.append(unit)
        
        logging.info(f"将处理 {len(units_to_process)} 个单元")
        
        # 批量处理
        results = {
            "total_units": len(units_to_process),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "errors": []
        }
        
        for i, unit in enumerate(units_to_process):
            logging.info(f"处理单元 {i+1}/{len(units_to_process)}: {unit.uid}")
            
            try:
                result = self.process_memory_unit_for_entities(unit)
                
                if result.get("skipped"):
                    results["skipped"] += 1
                elif result.get("success"):
                    results["processed"] += 1
                    results["total_entities"] += result.get("entities_count", 0)
                    results["total_relationships"] += result.get("relationships_count", 0)
                else:
                    results["failed"] += 1
                    if result.get("error"):
                        results["errors"].append(f"{unit.uid}: {result['error']}")
                        
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{unit.uid}: {str(e)}")
                logging.error(f"处理单元 {unit.uid} 时发生异常: {e}")
        
        logging.info(f"批量实体抽取完成: {results}")
        return results
    
    # def batch_extract_entities_from_space_optimized(self, space_name: str, 
    #                                           max_units: int = None,
    #                                           unit_filter=None,
    #                                           batch_size: int = 5) -> Dict[str, Any]:
    #     """
    #     优化的批量实体抽取，支持批处理和上下文合并
    #     """
    #     space = self.graph.semantic_map.get_memory_space(space_name)
    #     if not space:
    #         logging.warning(f"内存空间 '{space_name}' 不存在")
    #         return {"processed": 0, "extracted_entities": 0, "extracted_relationships": 0}
        
    #     # 获取待处理的单元
    #     all_units = []
    #     for uid in space.get_memory_uids():
    #         unit = self.graph.get_unit(uid)
    #         if unit and (not unit_filter or unit_filter(unit)):
    #             all_units.append(unit)
        
    #     if max_units:
    #         all_units = all_units[:max_units]
        
    #     logging.info(f"开始批量处理 {len(all_units)} 个单元")
        
    #     # 按会话分组批处理
    #     session_groups = {}
    #     for unit in all_units:
    #         session = unit.metadata.get('session', 'default')
    #         if session not in session_groups:
    #             session_groups[session] = []
    #         session_groups[session].append(unit)
        
    #     total_stats = {"processed": 0, "extracted_entities": 0, "extracted_relationships": 0}
        
    #     for session, units in session_groups.items():
    #         logging.info(f"处理会话 {session}，包含 {len(units)} 个单元")
            
    #         # 合并同一会话的文本，提供更好的上下文
    #         combined_text = self._combine_session_texts(units)
            
    #         # 批量抽取
    #         entities, relationships, keywords = self.extractor.extract_entities_and_relations(combined_text)
            
    #         # 为每个单元创建实体单元
    #         for unit in units:
    #             unit_stats = self._create_entity_units_for_unit(unit, entities, relationships, keywords)
    #             total_stats["processed"] += 1
    #             total_stats["extracted_entities"] += unit_stats["entities"]
    #             total_stats["extracted_relationships"] += unit_stats["relationships"]
                
    #             # 标记已处理
    #             unit.metadata["entities_extracted"] = True
        
    #     return total_stats

    # def _combine_session_texts(self, units: List[MemoryUnit]) -> str:
    #     """合并同一会话的文本，保持上下文"""
    #     texts = []
    #     for unit in sorted(units, key=lambda u: u.metadata.get('timestamp', '')):
    #         text_content = unit.raw_data.get('text_content', '')
    #         speaker = unit.raw_data.get('speaker', 'Unknown')
    #         texts.append(f"{speaker}: {text_content}")
        
    #     return '\n'.join(texts)
    
    def _normalize_name(self, name: str) -> str:
        """标准化名称用于ID生成"""
        import re
        # 移除特殊字符，替换为下划线
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '_', name)
        # 限制长度并确保唯一性
        normalized = normalized[:30]
        # 添加哈希确保唯一性
        import hashlib
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{normalized}_{hash_suffix}"
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """获取实体统计信息"""
        stats = {
            "total_entities": 0,
            "entity_types": {},
            "total_relationships": 0,
            "relationship_types": {}
        }
        
        # 统计实体
        entity_space = self.graph.semantic_map.get_memory_space("extracted_entities")
        if entity_space:
            entity_uids = entity_space.get_memory_uids()
            stats["total_entities"] = len(entity_uids)
            
            for uid in entity_uids:
                unit = self.graph.get_unit(uid)
                if unit:
                    entity_type = unit.metadata.get("entity_type", "unknown")
                    stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
        
        # 统计关系
        for source, target, data in self.graph.nx_graph.edges(data=True):
            if data.get("source_unit_id"):  # 只统计我们添加的关系
                stats["total_relationships"] += 1
                rel_type = data.get("type", "UNKNOWN")
                stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
        
        return stats
# class SemanticGraphIntegrator:
#     """Locomo语义图集成器 - 将实体和关系添加到语义图中"""
    
#     def __init__(self, semantic_graph: SemanticGraph):
#         self.graph = semantic_graph
#         self.entity_space_name = "extracted_entities"
#         self.relationship_space_name = "extracted_relationships"
        
#         # 创建专用空间
#         self._setup_entity_spaces()
        
#     def _setup_entity_spaces(self):
#         """设置实体和关系的专用空间"""
#         self.graph.create_memory_space_in_map(self.entity_space_name)
#         self.graph.create_memory_space_in_map(self.relationship_space_name)
        
#         # 为不同实体类型创建子空间
#         entity_types = ["person", "organization", "geo", "event", "category"]
#         for entity_type in entity_types:
#             space_name = f"entities_{entity_type}"
#             self.graph.create_memory_space_in_map(space_name)
    
#     def add_entities_to_graph(self, entities: List[Entity], source_unit_id: str) -> Dict[str, str]:
#         """
#         将实体添加到语义图中
        
#         Args:
#             entities: 实体列表
#             source_unit_id: 源文档单元ID
            
#         Returns:
#             实体名称到单元ID的映射
#         """
#         entity_id_map = {}
        
#         for entity in entities:
#             try:
#                 # 生成实体单元ID
#                 entity_uid = f"entity_{entity.entity_type}_{self._normalize_name(entity.name)}_{hash(entity.name) % 10000}"
                
#                 # 创建实体内存单元
#                 entity_unit = MemoryUnit(
#                     uid=entity_uid,
#                     raw_data={
#                         "text_content": f"{entity.name}: {entity.description}",
#                         "entity_name": entity.name,
#                         "entity_type": entity.entity_type,
#                         "description": entity.description,
#                         "source_text": entity.source_text,
#                         "confidence": entity.confidence
#                     },
#                     metadata={
#                         "data_source": "entity_extraction",
#                         "entity_type": entity.entity_type,
#                         "entity_name": entity.name,
#                         "source_unit_id": source_unit_id,
#                         "extraction_confidence": entity.confidence
#                     }
#                 )
                
#                 # 添加到图中
#                 self.graph.add_unit(entity_unit)
                
#                 # 添加到相应的空间
#                 self.graph.add_unit_to_space_in_map(entity_uid, self.entity_space_name)
#                 self.graph.add_unit_to_space_in_map(entity_uid, f"entities_{entity.entity_type}")
                
#                 # 建立与源文档的关系
#                 if source_unit_id and self.graph.get_unit(source_unit_id):
#                     self.graph.add_relationship(
#                         source_uid=source_unit_id,
#                         target_uid=entity_uid,
#                         relationship_name="CONTAINS_ENTITY",
#                         entity_type=entity.entity_type,
#                         confidence=entity.confidence
#                     )
                
#                 entity_id_map[entity.name] = entity_uid
#                 logging.debug(f"已添加实体: {entity.name} ({entity.entity_type}) -> {entity_uid}")
                
#             except Exception as e:
#                 logging.error(f"添加实体失败: {entity.name} - {e}")
#                 continue
        
#         logging.info(f"成功添加 {len(entity_id_map)} 个实体到语义图")
#         return entity_id_map
    
#     def add_relationships_to_graph(self, relationships: List[Relationship], entity_id_map: Dict[str, str], source_unit_id: str) -> int:
#         """
#         将关系添加到语义图中
        
#         Args:
#             relationships: 关系列表
#             entity_id_map: 实体名称到单元ID的映射
#             source_unit_id: 源文档单元ID
            
#         Returns:
#             成功添加的关系数量
#         """
#         success_count = 0
        
#         for relationship in relationships:
#             try:
#                 # 查找实体ID
#                 source_entity_id = entity_id_map.get(relationship.source_entity)
#                 target_entity_id = entity_id_map.get(relationship.target_entity)
                
#                 if not source_entity_id or not target_entity_id:
#                     logging.warning(f"关系中的实体未找到: {relationship.source_entity} -> {relationship.target_entity}")
#                     continue
                
#                 # 添加关系到图中
#                 success = self.graph.add_relationship(
#                     source_uid=source_entity_id,
#                     target_uid=target_entity_id,
#                     relationship_name=relationship.relationship_type,
#                     description=relationship.description,
#                     keywords=",".join(relationship.keywords),
#                     strength=relationship.strength,
#                     source_text=relationship.source_text,
#                     source_unit_id=source_unit_id
#                 )
                
#                 if success:
#                     success_count += 1
#                     logging.debug(f"已添加关系: {relationship.source_entity} -[{relationship.relationship_type}]-> {relationship.target_entity}")
                
#             except Exception as e:
#                 logging.error(f"添加关系失败: {relationship.source_entity} -> {relationship.target_entity} - {e}")
#                 continue
        
#         logging.info(f"成功添加 {success_count} 个关系到语义图")
#         return success_count
    
#     def process_memory_unit_for_entities(self, unit: MemoryUnit) -> Dict[str, Any]:
#         """
#         处理单个内存单元的实体关系抽取
        
#         Args:
#             unit: 内存单元
            
#         Returns:
#             处理结果统计
#         """

        
#         # 检查是否已处理过
#         if unit.metadata.get('entities_extracted', False):
#             logging.debug(f"单元 {unit.uid} 已处理过实体抽取，跳过")
#             return {"skipped": True}
        
#         # 获取文本内容
#         text_content = unit.raw_data.get('text_content', '')
#         if not text_content or len(text_content.strip()) < 10:
#             logging.debug(f"单元 {unit.uid} 文本内容太短，跳过实体抽取")
#             return {"skipped": True, "reason": "text_too_short"}
        
#         try:
#             # 初始化抽取器
#             llm_client = LLMClient(model_name="deepseek-chat")
#             extractor = EntityRelationExtractor(llm_client)
            
#             # 抽取实体和关系
#             entities, relationships, content_keywords = extractor.extract_entities_and_relations(text_content)
            
#             # 添加到图中
#             entity_id_map = self.add_entities_to_graph(entities, unit.uid)
#             relationship_count = self.add_relationships_to_graph(relationships, entity_id_map, unit.uid)
            
#             # 更新原单元的元数据，标记已处理
#             unit.metadata['entities_extracted'] = True
#             unit.metadata['entity_count'] = len(entities)
#             unit.metadata['relationship_count'] = relationship_count
#             unit.metadata['content_keywords'] = content_keywords
            
#             result = {
#                 "success": True,
#                 "entities_count": len(entities),
#                 "relationships_count": relationship_count,
#                 "keywords_count": len(content_keywords),
#                 "entity_ids": list(entity_id_map.values())
#             }
            
#             logging.info(f"单元 {unit.uid} 实体抽取完成: {result}")
#             return result
            
#         except Exception as e:
#             logging.error(f"处理单元 {unit.uid} 的实体抽取失败: {e}")
#             return {"success": False, "error": str(e)}
    
#     def batch_extract_entities_from_graph(self, 
#                                         space_name: Optional[str] = None,
#                                         max_units: int = 100,
#                                         unit_filter: Optional[callable] = None) -> Dict[str, Any]:
#         """
#         批量处理图中的内存单元进行实体关系抽取
        
#         Args:
#             space_name: 限制处理的空间名称
#             max_units: 最大处理单元数
#             unit_filter: 单元过滤函数
            
#         Returns:
#             批量处理结果统计
#         """
#         logging.info(f"开始批量实体抽取，空间: {space_name}, 最大单元数: {max_units}")
        
#         # 获取要处理的单元
#         if space_name:
#             space = self.graph.semantic_map.get_memory_space(space_name)
#             if not space:
#                 logging.error(f"空间 {space_name} 不存在")
#                 return {"error": "space_not_found"}
#             unit_ids = list(space.get_memory_uids())
#         else:
#             unit_ids = list(self.graph.semantic_map.memory_units.keys())
        
#         # 应用过滤器
#         units_to_process = []
#         for unit_id in unit_ids[:max_units]:
#             unit = self.graph.get_unit(unit_id)
#             if unit and (not unit_filter or unit_filter(unit)):
#                 units_to_process.append(unit)
        
#         logging.info(f"将处理 {len(units_to_process)} 个单元")
        
#         # 批量处理
#         results = {
#             "total_units": len(units_to_process),
#             "processed": 0,
#             "skipped": 0,
#             "failed": 0,
#             "total_entities": 0,
#             "total_relationships": 0,
#             "errors": []
#         }
        
#         for i, unit in enumerate(units_to_process):
#             logging.info(f"处理单元 {i+1}/{len(units_to_process)}: {unit.uid}")
            
#             try:
#                 result = self.process_memory_unit_for_entities(unit)
                
#                 if result.get("skipped"):
#                     results["skipped"] += 1
#                 elif result.get("success"):
#                     results["processed"] += 1
#                     results["total_entities"] += result.get("entities_count", 0)
#                     results["total_relationships"] += result.get("relationships_count", 0)
#                 else:
#                     results["failed"] += 1
#                     if result.get("error"):
#                         results["errors"].append(f"{unit.uid}: {result['error']}")
                        
#             except Exception as e:
#                 results["failed"] += 1
#                 results["errors"].append(f"{unit.uid}: {str(e)}")
#                 logging.error(f"处理单元 {unit.uid} 时发生异常: {e}")
        
#         logging.info(f"批量实体抽取完成: {results}")
#         return results
    
#     def _normalize_name(self, name: str) -> str:
#         """标准化名称用于ID生成"""
#         # 移除特殊字符，替换为下划线
#         import re
#         normalized = re.sub(r'[^\w\u4e00-\u9fff]', '_', name)
#         return normalized[:50]  # 限制长度
    
#     def get_entity_statistics(self) -> Dict[str, Any]:
#         """获取实体统计信息"""
#         stats = {
#             "total_entities": 0,
#             "entity_types": {},
#             "total_relationships": 0,
#             "relationship_types": {}
#         }
        
#         # 统计实体
#         entity_space = self.graph.semantic_map.get_memory_space(self.entity_space_name)
#         if entity_space:
#             entity_uids = entity_space.get_memory_uids()
#             stats["total_entities"] = len(entity_uids)
            
#             for uid in entity_uids:
#                 unit = self.graph.get_unit(uid)
#                 if unit:
#                     entity_type = unit.metadata.get("entity_type", "unknown")
#                     stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
        
#         # 统计关系
#         for source, target, data in self.graph.nx_graph.edges(data=True):
#             if data.get("source_unit_id"):  # 只统计我们添加的关系
#                 stats["total_relationships"] += 1
#                 rel_type = data.get("type", "UNKNOWN")
#                 stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
        
#         return stats