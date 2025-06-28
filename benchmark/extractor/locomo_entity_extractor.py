import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import networkx as nx
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from dev.semantic_graph import SemanticGraph
from dev.memory_unit import MemoryUnit
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor
from benchmark.extractor.semantic_graph_integrator import SemanticGraphIntegrator
from benchmark.task_eval.locomo_test_split import load_dataset, ingest_conversation_history

class LoCoMoEntityExtractor:
    """LoCoMo数据集实体关系抽取器 - 专门处理单个sample_id的完整实体关系抽取"""
    
    def __init__(self, dataset_path: str, output_base_dir: str = None):
        self.dataset_path = Path(dataset_path)
        # self.output_base_dir = Path(output_base_dir) if output_base_dir else Path(__file__).parent.parent / "results"
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path(__file__).parent / "results"
        self.raw_data = None
        
        # 初始化LLM客户端和抽取器
        self.llm_client = LLMClient(model_name="deepseek-chat")
        self.entity_extractor = EntityRelationExtractor(self.llm_client)
        
        # 加载数据集
        self._load_dataset()
        
    def _load_dataset(self):
        """加载数据集"""
        logging.info(f"加载数据集: {self.dataset_path}")
        self.raw_data = load_dataset(self.dataset_path)
        if not self.raw_data:
            raise ValueError("数据集加载失败")
        logging.info(f"成功加载 {len(self.raw_data)} 个样本")
    
    def list_available_samples(self) -> list:
        """列出所有可用的sample_id"""
        sample_ids = []
        for sample in self.raw_data:
            sample_id = sample.get('sample_id')
            if sample_id:
                sample_ids.append(sample_id)
        return sample_ids
    
    def get_sample_info(self, sample_id: str) -> Dict[str, Any]:
        """获取指定样本的基本信息"""
        sample = self._find_sample(sample_id)
        if not sample:
            return {}
        
        conversation = sample.get('conversation', {})
        speaker_a = conversation.get('speaker_a', 'Unknown')
        speaker_b = conversation.get('speaker_b', 'Unknown')
        
        # 统计会话数量
        session_count = sum(1 for key in conversation.keys() 
                           if key.startswith('session_') and not key.endswith('_date_time'))
        
        # 统计实际对话内容
        total_messages = 0
        for key, value in conversation.items():
            if key.startswith('session_') and not key.endswith('_date_time') and isinstance(value, list):
                total_messages += len(value)
        
        return {
            "sample_id": sample_id,
            "speakers": [speaker_a, speaker_b],
            "session_count": session_count,
            "total_messages": total_messages,
            "qa_count": len(sample.get('qa', [])),
            "event_summary_count": len(sample.get('event_summary', {})),
            "observation_count": len(sample.get('observation', {})),
            "session_summary_count": len(sample.get('session_summary', {}))
        }
    
    def _find_sample(self, sample_id: str) -> Optional[dict]:
        """查找指定的样本"""
        for sample in self.raw_data:
            if sample.get('sample_id') == sample_id:
                return sample
        return None
    
    def _extract_full_conversation_text(self, sample: dict) -> str:
        """提取样本的完整对话文本"""
        conversation = sample.get('conversation', {})
        conversation_texts = []
        
        # 获取说话者信息
        speaker_a = conversation.get('speaker_a', 'Speaker A')
        speaker_b = conversation.get('speaker_b', 'Speaker B')
        
        # 按会话顺序提取对话内容
        session_keys = [key for key in conversation.keys() 
                       if key.startswith('session_') and not key.endswith('_date_time')]
        session_keys.sort(key=lambda x: int(x.split('_')[1]))
        
        for session_key in session_keys:
            session_messages = conversation.get(session_key, [])
            session_datetime = conversation.get(f"{session_key}_date_time", "")
            
            if session_messages and session_datetime:
                conversation_texts.append(f"\n=== {session_key.upper()} ({session_datetime}) ===")
                
                for message in session_messages:
                    if isinstance(message, dict):
                        speaker = message.get('speaker', 'Unknown')
                        content = message.get('content', '')
                        if speaker and content:
                            conversation_texts.append(f"{speaker}: {content}")
                    elif isinstance(message, str):
                        # 有些格式可能是简单的字符串
                        conversation_texts.append(message)
        
        # 添加会话摘要信息
        session_summary = sample.get('session_summary', {})
        if session_summary:
            conversation_texts.append("\n=== SESSION SUMMARIES ===")
            for summary_key, summary_content in session_summary.items():
                if isinstance(summary_content, str):
                    conversation_texts.append(f"{summary_key}: {summary_content}")
        
        # 添加事件摘要
        event_summary = sample.get('event_summary', {})
        if event_summary:
            conversation_texts.append("\n=== EVENT SUMMARIES ===")
            for event_key, event_content in event_summary.items():
                if isinstance(event_content, dict):
                    for sub_key, sub_content in event_content.items():
                        if isinstance(sub_content, str):
                            conversation_texts.append(f"{event_key}.{sub_key}: {sub_content}")
        
        # 添加观察记录
        observations = sample.get('observation', {})
        if observations:
            conversation_texts.append("\n=== OBSERVATIONS ===")
            for obs_key, obs_content in observations.items():
                if isinstance(obs_content, dict):
                    for sub_key, sub_content in obs_content.items():
                        if isinstance(sub_content, str):
                            conversation_texts.append(f"{obs_key}.{sub_key}: {sub_content}")
        
        full_text = "\n".join(conversation_texts)
        
        # 添加基本上下文信息
        context_info = f"""
=== CONVERSATION CONTEXT ===
Sample ID: {sample.get('sample_id', 'Unknown')}
Participants: {speaker_a} and {speaker_b}
Total Sessions: {len(session_keys)}
Total Messages: {sum(len(conversation.get(key, [])) for key in session_keys)}

=== FULL CONVERSATION CONTENT ===
{full_text}
"""
        
        return context_info
    
    def extract_entities_and_relations_for_sample(self, 
                                                sample_id: str, 
                                                use_chunking: bool = True,
                                                save_intermediate: bool = True) -> Dict[str, Any]:
        """
        为指定样本进行完整的实体关系抽取
        
        Args:
            sample_id: 样本ID
            use_chunking: 是否使用文本分块处理
            save_intermediate: 是否保存中间结果
        """
        
        print(f"🚀 开始对样本 {sample_id} 进行实体关系抽取")
        
        # 1. 查找样本
        sample = self._find_sample(sample_id)
        if not sample:
            raise ValueError(f"未找到sample_id为{sample_id}的样本")
        
        # 2. 创建输出目录
        output_dir = self.output_base_dir / f"{sample_id}_entity_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging(output_dir)
        
        # 3. 获取样本信息
        sample_info = self.get_sample_info(sample_id)
        print(f"📊 样本信息: {sample_info}")
        
        # 4. 提取完整对话文本
        print("\n=== 提取完整对话文本 ===")
        full_conversation_text = self._extract_full_conversation_text(sample)
        
        # 保存原始文本
        if save_intermediate:
            text_file = output_dir / f"{sample_id}_full_conversation.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(full_conversation_text)
            print(f"💾 完整对话文本已保存到: {text_file}")
        
        print(f"✅ 提取的文本长度: {len(full_conversation_text)} 字符")
        
        # 5. 使用EntityRelationExtractor进行实体关系抽取
        print(f"\n=== 开始实体关系抽取 ===")
        
        try:
            if use_chunking:
                print("使用智能分块处理长文本...")
                entities, relationships, content_keywords = self.entity_extractor.extract_entities_and_relations_chunked(full_conversation_text)
            else:
                print("使用直接处理方式...")
                entities, relationships, content_keywords = self.entity_extractor.extract_entities_and_relations(full_conversation_text)
            
            print(f"✅ 实体关系抽取完成!")
            print(f"   抽取实体数: {len(entities)}")
            print(f"   抽取关系数: {len(relationships)}")
            print(f"   关键词数: {len(content_keywords)}")
            
        except Exception as e:
            logging.error(f"实体关系抽取失败: {e}")
            print(f"❌ 实体关系抽取失败: {e}")
            return {"success": False, "error": str(e)}
        
        # 6. 创建语义图并添加抽取结果
        print("\n=== 构建语义图 ===")
        graph = SemanticGraph()
        
        # 创建原始文档单元
        doc_unit = MemoryUnit(
            uid=f"{sample_id}_document",
            raw_data={
                "text_content": full_conversation_text,
                "sample_id": sample_id,
                "speakers": sample_info["speakers"],
                "session_count": sample_info["session_count"],
                "total_messages": sample_info["total_messages"]
            },
            metadata={
                "data_source": "locomo_full_conversation",
                "sample_id": sample_id,
                "conversation_type": "multi_session_dialog"
            }
        )
        
        # 添加文档单元到图中
        graph.add_unit(doc_unit, space_names=["locomo_documents"])
        
        # 7. 使用SemanticGraphIntegrator将实体和关系添加到图中
        integrator = SemanticGraphIntegrator(graph)
        
        print("=== 添加实体到语义图 ===")
        entity_id_map = integrator.add_entities_to_graph(entities, doc_unit.uid)
        print(f"✅ 成功添加 {len(entity_id_map)} 个实体")
        
        print("=== 添加关系到语义图 ===")
        relationship_count = integrator.add_relationships_to_graph(relationships, entity_id_map, doc_unit.uid)
        print(f"✅ 成功添加 {relationship_count} 个关系")
        
        # 8. 构建语义图索引
        print("=== 构建语义图索引 ===")
        graph.build_semantic_map_index()
        
        # 9. 显示图谱摘要
        print("\n=== 语义图摘要 ===")
        graph.display_graph_summary()
        
        # 10. 保存完整的语义图
        print("\n=== 保存语义图 ===")
        semantic_graph_path = output_dir / f"{sample_id}_semantic_graph"
        graph.save_graph(str(semantic_graph_path))
        print(f"✅ 语义图已保存到: {semantic_graph_path}")
        
        # 11. 获取并保存统计信息
        entity_stats = integrator.get_entity_statistics()
        
        # 12. 保存抽取结果
        extraction_results = {
            "sample_info": sample_info,
            "extraction_statistics": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "total_keywords": len(content_keywords),
                "processing_time": str(datetime.now())
            },
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                    "confidence": e.confidence,
                    "source_text": e.source_text[:200] + "..." if len(e.source_text) > 200 else e.source_text
                } for e in entities
            ],
            "relationships": [
                {
                    "source": r.source_entity,
                    "target": r.target_entity,
                    "type": r.relationship_type,
                    "description": r.description,
                    "keywords": r.keywords,
                    "strength": r.strength,
                    "source_text": r.source_text[:200] + "..." if len(r.source_text) > 200 else r.source_text
                } for r in relationships
            ],
            "content_keywords": content_keywords,
            "entity_statistics": entity_stats,
            "graph_structure": {
                "total_memory_units": len(graph.semantic_map.memory_units),
                "total_memory_spaces": len(graph.semantic_map.memory_spaces),
                "networkx_nodes": graph.nx_graph.number_of_nodes(),
                "networkx_edges": graph.nx_graph.number_of_edges(),
                "faiss_vectors": graph.semantic_map.faiss_index.ntotal if graph.semantic_map.faiss_index else 0
            }
        }
        
        # 保存详细结果
        results_file = output_dir / f"{sample_id}_extraction_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅ 抽取结果已保存到: {results_file}")
        
        # 13. 导出实体和关系的详细信息
        self._export_detailed_analysis(graph, output_dir, sample_id)
        
        print(f"\n🎉 样本 {sample_id} 的实体关系抽取完成!")
        print(f"📁 所有结果已保存到: {output_dir}")
        
        return {
            "success": True,
            "sample_id": sample_id,
            "output_directory": output_dir,
            "graph": graph,
            "extraction_results": extraction_results,
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "keywords_count": len(content_keywords)
        }
    
    def _export_detailed_analysis(self, graph: SemanticGraph, output_dir: Path, sample_id: str):
        """导出详细的实体和关系分析"""
        
        analysis = {
            "sample_id": sample_id,
            "analysis_timestamp": str(datetime.now()),
            "entity_analysis": {},
            "relationship_analysis": {},
            "network_analysis": {}
        }
        
        # 实体分析
        entity_space = graph.semantic_map.get_memory_space("extracted_entities")
        if entity_space:
            entity_details = []
            entity_type_counts = {}
            
            for uid in entity_space.get_memory_uids():
                unit = graph.get_unit(uid)
                if unit:
                    entity_type = unit.raw_data.get('entity_type', 'Unknown')
                    entity_details.append({
                        "uid": uid,
                        "name": unit.raw_data.get('entity_name', 'Unknown'),
                        "type": entity_type,
                        "description": unit.raw_data.get('description', ''),
                        "confidence": unit.raw_data.get('confidence', 0.0)
                    })
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            
            analysis["entity_analysis"] = {
                "total_entities": len(entity_details),
                "entity_type_distribution": entity_type_counts,
                "entities": entity_details
            }
        
        # 关系分析
        relationship_details = []
        relationship_type_counts = {}
        
        for source, target, data in graph.nx_graph.edges(data=True):
            if data.get("source_unit_id"):  # 只分析抽取出的关系
                rel_type = data.get("type", "UNKNOWN")
                
                # 获取实体名称
                source_unit = graph.get_unit(source)
                target_unit = graph.get_unit(target)
                source_name = source_unit.raw_data.get('entity_name', source) if source_unit else source
                target_name = target_unit.raw_data.get('entity_name', target) if target_unit else target
                
                relationship_details.append({
                    "source_uid": source,
                    "target_uid": target,
                    "source_name": source_name,
                    "target_name": target_name,
                    "relationship_type": rel_type,
                    "description": data.get("description", ""),
                    "strength": data.get("strength", 0.0)
                })
                
                relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        
        analysis["relationship_analysis"] = {
            "total_relationships": len(relationship_details),
            "relationship_type_distribution": relationship_type_counts,
            "relationships": relationship_details
        }
        
        # 网络分析
        analysis["network_analysis"] = {
            "graph_density": nx.density(graph.nx_graph),
            "number_of_nodes": graph.nx_graph.number_of_nodes(),
            "number_of_edges": graph.nx_graph.number_of_edges(),
            "is_connected": nx.is_connected(graph.nx_graph.to_undirected()),
            "number_of_connected_components": nx.number_connected_components(graph.nx_graph.to_undirected())
        }
        
        # 保存分析结果
        analysis_file = output_dir / f"{sample_id}_detailed_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 详细分析已保存到: {analysis_file}")
        
        # 生成简要报告
        report_lines = [
            f"=== {sample_id} 实体关系抽取报告 ===",
            f"处理时间: {analysis['analysis_timestamp']}",
            f"",
            f"📊 实体统计:",
            f"  - 总实体数: {analysis['entity_analysis'].get('total_entities', 0)}",
        ]
        
        for entity_type, count in analysis['entity_analysis'].get('entity_type_distribution', {}).items():
            report_lines.append(f"  - {entity_type}: {count} 个")
        
        report_lines.extend([
            f"",
            f"🔗 关系统计:",
            f"  - 总关系数: {analysis['relationship_analysis'].get('total_relationships', 0)}",
        ])
        
        for rel_type, count in analysis['relationship_analysis'].get('relationship_type_distribution', {}).items():
            report_lines.append(f"  - {rel_type}: {count} 个")
        
        report_lines.extend([
            f"",
            f"🕸️ 网络特征:",
            f"  - 节点数: {analysis['network_analysis']['number_of_nodes']}",
            f"  - 边数: {analysis['network_analysis']['number_of_edges']}",
            f"  - 图密度: {analysis['network_analysis']['graph_density']:.4f}",
            f"  - 连通性: {'是' if analysis['network_analysis']['is_connected'] else '否'}",
            f"  - 连通组件数: {analysis['network_analysis']['number_of_connected_components']}",
        ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_file = output_dir / f"{sample_id}_summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 摘要报告已保存到: {report_file}")
        print("\n" + report_content)
    
    def _setup_logging(self, output_dir: Path):
        """设置日志"""
        log_file = output_dir / "extraction.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def batch_extract_samples(self, 
                            sample_ids: list = None, 
                            use_chunking: bool = True,
                            save_intermediate: bool = False) -> Dict[str, Any]:
        """
        批量处理多个样本的实体关系抽取
        
        Args:
            sample_ids: 要处理的样本ID列表，如果为None则处理所有样本
            use_chunking: 是否使用文本分块处理
            save_intermediate: 是否保存中间结果
        """
        
        if sample_ids is None:
            sample_ids = self.list_available_samples()
        
        print(f"🚀 开始批量实体关系抽取，共 {len(sample_ids)} 个样本")
        
        batch_results = {
            "processed_samples": [],
            "failed_samples": [],
            "total_entities": 0,
            "total_relationships": 0,
            "total_keywords": 0,
            "start_time": str(datetime.now())
        }
        
        for i, sample_id in enumerate(sample_ids):
            print(f"\n--- 处理样本 {i+1}/{len(sample_ids)}: {sample_id} ---")
            
            try:
                result = self.extract_entities_and_relations_for_sample(
                    sample_id=sample_id,
                    use_chunking=use_chunking,
                    save_intermediate=save_intermediate
                )
                
                if result.get("success"):
                    batch_results["processed_samples"].append({
                        "sample_id": sample_id,
                        "entities_count": result.get("entities_count", 0),
                        "relationships_count": result.get("relationships_count", 0),
                        "keywords_count": result.get("keywords_count", 0),
                        "output_directory": str(result.get("output_directory", ""))
                    })
                    
                    batch_results["total_entities"] += result.get("entities_count", 0)
                    batch_results["total_relationships"] += result.get("relationships_count", 0)
                    batch_results["total_keywords"] += result.get("keywords_count", 0)
                    
                    print(f"✅ 样本 {sample_id} 处理完成")
                else:
                    batch_results["failed_samples"].append({
                        "sample_id": sample_id,
                        "error": result.get("error", "Unknown error")
                    })
                    print(f"❌ 样本 {sample_id} 处理失败")
                
            except Exception as e:
                print(f"❌ 样本 {sample_id} 处理失败: {e}")
                batch_results["failed_samples"].append({
                    "sample_id": sample_id,
                    "error": str(e)
                })
                logging.error(f"样本 {sample_id} 处理失败", exc_info=True)
        
        batch_results["end_time"] = str(datetime.now())
        
        # 保存批量处理结果
        batch_summary_file = self.output_base_dir / f"batch_extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n🎊 批量实体关系抽取完成!")
        print(f"✅ 成功处理: {len(batch_results['processed_samples'])} 个样本")
        print(f"❌ 失败: {len(batch_results['failed_samples'])} 个样本")
        print(f"📊 总实体数: {batch_results['total_entities']}")
        print(f"📊 总关系数: {batch_results['total_relationships']}")
        print(f"📊 总关键词数: {batch_results['total_keywords']}")
        print(f"📁 批量摘要已保存到: {batch_summary_file}")
        
        return batch_results

def main():
    """主函数"""
    import argparse
    import networkx as nx  # 添加这个导入
    
    parser = argparse.ArgumentParser(description="LoCoMo数据集实体关系抽取器")
    parser.add_argument("--dataset", required=True, help="LoCoMo数据集路径")
    parser.add_argument("--sample-id", help="要处理的样本ID")
    parser.add_argument("--list-samples", action="store_true", help="列出所有可用的样本ID")
    parser.add_argument("--batch", nargs="+", help="批量处理指定的样本ID")
    parser.add_argument("--batch-all", action="store_true", help="批量处理所有样本")
    parser.add_argument("--use-chunking", action="store_true", default=True, help="使用文本分块处理")
    parser.add_argument("--no-chunking", action="store_true", help="不使用文本分块处理")
    parser.add_argument("--save-intermediate", action="store_true", help="保存中间结果")
    parser.add_argument("--output", help="输出目录")
    
    args = parser.parse_args()
    
    try:
        # 初始化处理器
        extractor = LoCoMoEntityExtractor(
            dataset_path=args.dataset,
            output_base_dir=args.output
        )
        
        # 确定是否使用分块
        use_chunking = not args.no_chunking if args.no_chunking else args.use_chunking
        
        # 列出所有样本
        if args.list_samples:
            samples = extractor.list_available_samples()
            print(f"数据集中包含 {len(samples)} 个样本:")
            for i, sample_id in enumerate(samples, 1):
                info = extractor.get_sample_info(sample_id)
                print(f"  {i:2d}. {sample_id} - {info['speakers'][0]} & {info['speakers'][1]} "
                      f"({info['session_count']} sessions, {info['total_messages']} messages)")
            return 0
        
        # 处理单个样本
        if args.sample_id:
            start_time = datetime.now()
            print(f"⏰ 开始时间: {start_time}")
            
            result = extractor.extract_entities_and_relations_for_sample(
                sample_id=args.sample_id,
                use_chunking=use_chunking,
                save_intermediate=args.save_intermediate
            )
            
            end_time = datetime.now()
            if result.get("success"):
                print(f"\n🎉 样本 {args.sample_id} 处理完成!")
                print(f"⏰ 完成时间: {end_time}")
                print(f"⏱️  总耗时: {end_time - start_time}")
                print(f"📁 输出目录: {result['output_directory']}")
                print(f"📊 抽取统计: {result['entities_count']} 实体, {result['relationships_count']} 关系")
            else:
                print(f"❌ 样本 {args.sample_id} 处理失败: {result.get('error', '未知错误')}")
            
        # 批量处理指定样本
        elif args.batch:
            extractor.batch_extract_samples(
                sample_ids=args.batch,
                use_chunking=use_chunking,
                save_intermediate=args.save_intermediate
            )
            
        # 批量处理所有样本
        elif args.batch_all:
            extractor.batch_extract_samples(
                sample_ids=None,
                use_chunking=use_chunking,
                save_intermediate=args.save_intermediate
            )
            
        else:
            print("请指定要执行的操作:")
            print("  --list-samples: 列出所有样本")
            print("  --sample-id SAMPLE_ID: 处理单个样本")
            print("  --batch SAMPLE1 SAMPLE2 ...: 批量处理指定样本")
            print("  --batch-all: 批量处理所有样本")
            return 1
        
    except Exception as e:
        logging.error(f"执行失败: {e}", exc_info=True)
        print(f"❌ 执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# import sys
# import json
# import logging
# from pathlib import Path
# from typing import Dict, Any, Optional
# from datetime import datetime

# # 添加项目根目录到Python路径
# sys.path.append(str(Path(__file__).parent.parent.parent))

# from dev.semantic_graph import SemanticGraph
# from dev.memory_unit import MemoryUnit
# from benchmark.llm_utils.llm_client import LLMClient
# from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor
# from benchmark.extractor.semantic_graph_integrator import SemanticGraphIntegrator
# from benchmark.task_eval.locomo_test_split import load_dataset, ingest_conversation_history

# class SingleSampleProcessor:
#     """单个样本处理器 - 专门处理单个sample_id的知识图谱构建"""
    
#     def __init__(self, dataset_path: str, output_base_dir: str = None):
#         self.dataset_path = Path(dataset_path)
#         self.output_base_dir = Path(output_base_dir) if output_base_dir else Path(__file__).parent.parent / "results"
#         self.raw_data = None
        
#         # 加载数据集
#         self._load_dataset()
        
#     def _load_dataset(self):
#         """加载数据集"""
#         logging.info(f"加载数据集: {self.dataset_path}")
#         self.raw_data = load_dataset(self.dataset_path)
#         if not self.raw_data:
#             raise ValueError("数据集加载失败")
#         logging.info(f"成功加载 {len(self.raw_data)} 个样本")
    
#     def list_available_samples(self) -> list:
#         """列出所有可用的sample_id"""
#         sample_ids = []
#         for sample in self.raw_data:
#             sample_id = sample.get('sample_id')
#             if sample_id:
#                 sample_ids.append(sample_id)
#         return sample_ids
    
#     def get_sample_info(self, sample_id: str) -> Dict[str, Any]:
#         """获取指定样本的基本信息"""
#         sample = self._find_sample(sample_id)
#         if not sample:
#             return {}
        
#         conversation = sample.get('conversation', {})
#         speaker_a = conversation.get('speaker_a', 'Unknown')
#         speaker_b = conversation.get('speaker_b', 'Unknown')
        
#         # 统计会话数量
#         session_count = sum(1 for key in conversation.keys() 
#                            if key.startswith('session_') and not key.endswith('_date_time'))
        
#         # 统计实际对话内容
#         total_messages = 0
#         for key, value in conversation.items():
#             if key.startswith('session_') and not key.endswith('_date_time') and isinstance(value, list):
#                 total_messages += len(value)
        
#         return {
#             "sample_id": sample_id,
#             "speakers": [speaker_a, speaker_b],
#             "session_count": session_count,
#             "total_messages": total_messages,
#             "qa_count": len(sample.get('qa', [])),
#             "event_summary_count": len(sample.get('event_summary', {})),
#             "observation_count": len(sample.get('observation', {})),
#             "session_summary_count": len(sample.get('session_summary', {}))
#         }
    
#     def _find_sample(self, sample_id: str) -> Optional[dict]:
#         """查找指定的样本"""
#         for sample in self.raw_data:
#             if sample.get('sample_id') == sample_id:
#                 return sample
#         return None
    
#     def process_single_sample(self, 
#                             sample_id: str, 
#                             max_units: int = 50,
#                             save_intermediate: bool = True) -> Dict[str, Any]:
#         """
#         处理单个样本，构建知识图谱
        
#         Args:
#             sample_id: 样本ID
#             max_units: 最大处理单元数
#             save_intermediate: 是否保存中间结果
#         """
        
#         print(f"🚀 开始处理样本: {sample_id}")
        
#         # 1. 查找样本
#         sample = self._find_sample(sample_id)
#         if not sample:
#             raise ValueError(f"未找到sample_id为{sample_id}的样本")
        
#         # 2. 创建输出目录
#         output_dir = self.output_base_dir / f"{sample_id}_knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         output_dir.mkdir(parents=True, exist_ok=True)
#         self._setup_logging(output_dir)
        
#         # 3. 获取样本信息
#         sample_info = self.get_sample_info(sample_id)
#         print(f"📊 样本信息: {sample_info}")
        
#         # 4. 初始化语义图
#         print("\n=== 初始化语义图 ===")
#         graph = SemanticGraph()
        
#         # 5. 注入对话数据
#         print("=== 注入对话数据到语义图 ===")
#         total_messages = ingest_conversation_history(graph, [sample])
#         print(f"✅ 成功注入 {total_messages} 个对话消息")
        
#         # 保存原始对话图
#         if save_intermediate:
#             original_graph_path = output_dir / "01_original_conversations_graph"
#             graph.save_graph(str(original_graph_path))
#             print(f"💾 原始对话图已保存到: {original_graph_path}")
        
#         # 6. 构建初始索引
#         print("=== 构建语义图索引 ===")
#         graph.build_semantic_map_index()
        
#         # 7. 显示图谱基本信息
#         print("\n=== 语义图基本信息 ===")
#         graph.display_graph_summary()
        
#         # 8. 实体关系抽取
#         print(f"\n=== 开始实体关系抽取（最多处理{max_units}个单元）===")
#         integrator = SemanticGraphIntegrator(graph)
        
#         # 定义过滤器：只处理当前样本的对话内容
#         def sample_dialog_filter(unit):
#             return (unit.metadata.get('data_source') == 'locomo_dialog' 
#                     and unit.metadata.get('conversation_id') == sample_id
#                     and not unit.metadata.get('entities_extracted', False))
        
#         # 批量抽取实体关系
#         extraction_results = integrator.batch_extract_entities_from_space(
#             space_name="locomo_dialogs",
#             max_units=max_units,
#             unit_filter=sample_dialog_filter
#         )
        
#         print(f"✅ 实体关系抽取完成!")
#         print(f"   处理单元数: {extraction_results.get('processed', 0)}")
#         print(f"   跳过单元数: {extraction_results.get('skipped', 0)}")
#         print(f"   失败单元数: {extraction_results.get('failed', 0)}")
#         print(f"   总抽取实体数: {extraction_results.get('total_entities', 0)}")
#         print(f"   总抽取关系数: {extraction_results.get('total_relationships', 0)}")
        
#         # 9. 获取详细统计信息
#         print("\n=== 生成统计信息 ===")
#         entity_stats = integrator.get_entity_statistics()
#         print(f"实体类型分布: {entity_stats.get('entity_types', {})}")
#         print(f"关系类型分布: {entity_stats.get('relationship_types', {})}")
        
#         # 10. 保存最终知识图谱
#         print("\n=== 保存知识图谱 ===")
#         final_graph_path = output_dir / f"{sample_id}_final_knowledge_graph"
#         graph.save_graph(str(final_graph_path))
#         print(f"✅ 知识图谱已保存到: {final_graph_path}")
        
#         # 11. 保存详细统计报告
#         final_summary = {
#             "sample_info": sample_info,
#             "processing_info": {
#                 "timestamp": str(datetime.now()),
#                 "max_units_processed": max_units,
#                 "total_messages_ingested": total_messages
#             },
#             "extraction_results": extraction_results,
#             "entity_statistics": entity_stats,
#             "graph_structure": {
#                 "total_memory_units": len(graph.semantic_map.memory_units),
#                 "total_memory_spaces": len(graph.semantic_map.memory_spaces),
#                 "networkx_nodes": graph.nx_graph.number_of_nodes(),
#                 "networkx_edges": graph.nx_graph.number_of_edges(),
#                 "faiss_vectors": graph.semantic_map.faiss_index.ntotal if graph.semantic_map.faiss_index else 0
#             }
#         }
        
#         summary_file = output_dir / f"{sample_id}_summary.json"
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             json.dump(final_summary, f, ensure_ascii=False, indent=2, default=str)
#         print(f"✅ 处理摘要已保存到: {summary_file}")
        
#         # 12. 导出实体和关系示例
#         print("\n=== 导出实体和关系示例 ===")
#         self._export_samples(graph, output_dir, sample_id, num_samples=10)
        
#         # 13. 显示最终图谱摘要
#         print("\n=== 最终知识图谱摘要 ===")
#         graph.display_graph_summary()
        
#         return {
#             "sample_id": sample_id,
#             "graph": graph,
#             "extraction_results": extraction_results,
#             "entity_statistics": entity_stats,
#             "output_directory": output_dir,
#             "summary": final_summary
#         }
    
#     def _export_samples(self, graph: SemanticGraph, output_dir: Path, sample_id: str, num_samples: int = 10):
#         """导出实体和关系示例"""
        
#         # 导出实体示例
#         entities_sample = []
#         entity_space = graph.semantic_map.get_memory_space("extracted_entities")
#         if entity_space:
#             entity_uids = list(entity_space.get_memory_uids())[:num_samples]
#             for uid in entity_uids:
#                 unit = graph.get_unit(uid)
#                 if unit:
#                     entities_sample.append({
#                         "uid": uid,
#                         "name": unit.raw_data.get('entity_name', 'Unknown'),
#                         "type": unit.raw_data.get('entity_type', 'Unknown'),
#                         "description": unit.raw_data.get('description', ''),
#                         "confidence": unit.raw_data.get('confidence', 0.0),
#                         "source_unit_id": unit.metadata.get('source_unit_id', '')
#                     })
        
#         # 导出关系示例
#         relationships_sample = []
#         relationship_count = 0
#         for source, target, data in graph.nx_graph.edges(data=True):
#             if relationship_count >= num_samples:
#                 break
#             if data.get("source_unit_id"):  # 只导出抽取出的关系
#                 source_unit = graph.get_unit(source)
#                 target_unit = graph.get_unit(target)
                
#                 relationships_sample.append({
#                     "source_uid": source,
#                     "target_uid": target,
#                     "source_name": source_unit.raw_data.get('entity_name', source) if source_unit else source,
#                     "target_name": target_unit.raw_data.get('entity_name', target) if target_unit else target,
#                     "relationship_type": data.get("type", "UNKNOWN"),
#                     "description": data.get("description", ""),
#                     "strength": data.get("strength", 0.0),
#                     "source_unit_id": data.get("source_unit_id", "")
#                 })
#                 relationship_count += 1
        
#         # 保存示例
#         samples = {
#             "sample_id": sample_id,
#             "entities_sample": entities_sample,
#             "relationships_sample": relationships_sample,
#             "total_entities": len(entity_space.get_memory_uids()) if entity_space else 0,
#             "total_relationships": graph.nx_graph.number_of_edges()
#         }
        
#         samples_file = output_dir / f"{sample_id}_entities_relationships_samples.json"
#         with open(samples_file, 'w', encoding='utf-8') as f:
#             json.dump(samples, f, ensure_ascii=False, indent=2)
        
#         print(f"✅ 实体关系示例已保存到: {samples_file}")
#         print(f"📊 实体示例数: {len(entities_sample)}")
#         print(f"📊 关系示例数: {len(relationships_sample)}")
    
#     def _setup_logging(self, output_dir: Path):
#         """设置日志"""
#         log_file = output_dir / "processing.log"
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(log_file),
#                 logging.StreamHandler()
#             ]
#         )
    
#     def batch_process_samples(self, 
#                             sample_ids: list = None, 
#                             max_units_per_sample: int = 50) -> Dict[str, Any]:
#         """
#         批量处理多个样本
        
#         Args:
#             sample_ids: 要处理的样本ID列表，如果为None则处理所有样本
#             max_units_per_sample: 每个样本最大处理单元数
#         """
        
#         if sample_ids is None:
#             sample_ids = self.list_available_samples()
        
#         print(f"🚀 开始批量处理 {len(sample_ids)} 个样本")
        
#         batch_results = {
#             "processed_samples": [],
#             "failed_samples": [],
#             "total_entities": 0,
#             "total_relationships": 0,
#             "start_time": str(datetime.now())
#         }
        
#         for i, sample_id in enumerate(sample_ids):
#             print(f"\n--- 处理样本 {i+1}/{len(sample_ids)}: {sample_id} ---")
            
#             try:
#                 result = self.process_single_sample(
#                     sample_id=sample_id,
#                     max_units=max_units_per_sample,
#                     save_intermediate=False  # 批量处理时不保存中间结果
#                 )
                
#                 batch_results["processed_samples"].append({
#                     "sample_id": sample_id,
#                     "entities": result["extraction_results"].get("total_entities", 0),
#                     "relationships": result["extraction_results"].get("total_relationships", 0),
#                     "output_directory": str(result["output_directory"])
#                 })
                
#                 batch_results["total_entities"] += result["extraction_results"].get("total_entities", 0)
#                 batch_results["total_relationships"] += result["extraction_results"].get("total_relationships", 0)
                
#                 print(f"✅ 样本 {sample_id} 处理完成")
                
#             except Exception as e:
#                 print(f"❌ 样本 {sample_id} 处理失败: {e}")
#                 batch_results["failed_samples"].append({
#                     "sample_id": sample_id,
#                     "error": str(e)
#                 })
#                 logging.error(f"样本 {sample_id} 处理失败", exc_info=True)
        
#         batch_results["end_time"] = str(datetime.now())
        
#         # 保存批量处理结果
#         batch_summary_file = self.output_base_dir / f"batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         with open(batch_summary_file, 'w', encoding='utf-8') as f:
#             json.dump(batch_results, f, ensure_ascii=False, indent=2, default=str)
        
#         print(f"\n🎊 批量处理完成!")
#         print(f"✅ 成功处理: {len(batch_results['processed_samples'])} 个样本")
#         print(f"❌ 失败: {len(batch_results['failed_samples'])} 个样本")
#         print(f"📊 总实体数: {batch_results['total_entities']}")
#         print(f"📊 总关系数: {batch_results['total_relationships']}")
#         print(f"📁 批量摘要已保存到: {batch_summary_file}")
        
#         return batch_results

# def main():
#     """主函数"""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="单个样本知识图谱处理器")
#     parser.add_argument("--dataset", required=True, help="LoCoMo数据集路径")
#     parser.add_argument("--sample-id", help="要处理的样本ID")
#     parser.add_argument("--list-samples", action="store_true", help="列出所有可用的样本ID")
#     parser.add_argument("--batch", nargs="+", help="批量处理指定的样本ID")
#     parser.add_argument("--batch-all", action="store_true", help="批量处理所有样本")
#     parser.add_argument("--max-units", type=int, default=50, help="每个样本最大处理单元数")
#     parser.add_argument("--output", help="输出目录")
    
#     args = parser.parse_args()
    
#     try:
#         # 初始化处理器
#         processor = SingleSampleProcessor(
#             dataset_path=args.dataset,
#             output_base_dir=args.output
#         )
        
#         # 列出所有样本
#         if args.list_samples:
#             samples = processor.list_available_samples()
#             print(f"数据集中包含 {len(samples)} 个样本:")
#             for i, sample_id in enumerate(samples, 1):
#                 info = processor.get_sample_info(sample_id)
#                 print(f"  {i:2d}. {sample_id} - {info['speakers'][0]} & {info['speakers'][1]} "
#                       f"({info['session_count']} sessions, {info['total_messages']} messages)")
#             return 0
        
#         # 处理单个样本
#         if args.sample_id:
#             start_time = datetime.now()
#             print(f"⏰ 开始时间: {start_time}")
            
#             result = processor.process_single_sample(
#                 sample_id=args.sample_id,
#                 max_units=args.max_units
#             )
            
#             end_time = datetime.now()
#             print(f"\n🎉 样本 {args.sample_id} 处理完成!")
#             print(f"⏰ 完成时间: {end_time}")
#             print(f"⏱️  总耗时: {end_time - start_time}")
#             print(f"📁 输出目录: {result['output_directory']}")
            
#         # 批量处理指定样本
#         elif args.batch:
#             processor.batch_process_samples(
#                 sample_ids=args.batch,
#                 max_units_per_sample=args.max_units
#             )
            
#         # 批量处理所有样本
#         elif args.batch_all:
#             processor.batch_process_samples(
#                 sample_ids=None,
#                 max_units_per_sample=args.max_units
#             )
            
#         else:
#             print("请指定要执行的操作:")
#             print("  --list-samples: 列出所有样本")
#             print("  --sample-id SAMPLE_ID: 处理单个样本")
#             print("  --batch SAMPLE1 SAMPLE2 ...: 批量处理指定样本")
#             print("  --batch-all: 批量处理所有样本")
#             return 1
        
#     except Exception as e:
#         logging.error(f"执行失败: {e}", exc_info=True)
#         print(f"❌ 执行失败: {e}")
#         return 1
    
#     return 0

# if __name__ == "__main__":
#     exit(main())
