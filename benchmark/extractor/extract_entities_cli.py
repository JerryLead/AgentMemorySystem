import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from dev.semantic_graph import SemanticGraph
from dev.memory_unit import MemoryUnit
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor
from benchmark.extractor.semantic_graph_integrator import SemanticGraphIntegrator

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('entity_extraction.log'),
            logging.StreamHandler()
        ]
    )

def extract_entities_from_locomo_dataset(graph: SemanticGraph, sample_limit: int = 10) -> Dict[str, Any]:
    """
    从LoCoMo对话中批量抽取实体关系 - 适配新架构
    
    Args:
        graph: 已加载LoCoMo数据的语义图
        sample_limit: 限制处理的对话单元数量
        
    Returns:
        处理结果统计
    """
    integrator = SemanticGraphIntegrator(graph)
    
    # 定义过滤器：只处理对话内容，且未处理过的
    def dialog_filter(unit):
        return (unit.metadata.get('data_source') == 'locomo_dialog' 
                and not unit.metadata.get('entities_extracted', False))
    
    # 批量抽取实体
    results = integrator.batch_extract_entities_from_space(
        space_name="locomo_dialogs",
        max_units=sample_limit,
        unit_filter=dialog_filter
    )
    
    return results

def extract_from_locomo_dataset(dataset_path: str, sample_limit: int = None) -> Dict[str, Any]:
    """从LoCoMo数据集抽取实体关系 - 适配新架构"""
    from benchmark.task_eval.locomo_test_split import (
        load_dataset, ingest_conversation_history
    )
    
    logging.info(f"加载LoCoMo数据集: {dataset_path}")
    
    # 加载数据集
    raw_data = load_dataset(Path(dataset_path))
    if not raw_data:
        raise ValueError("数据集加载失败")
    
    if sample_limit:
        raw_data = raw_data[:sample_limit]
        logging.info(f"限制处理样本数量: {sample_limit}")
    
    # 初始化语义图 - 使用新架构
    graph = SemanticGraph()
    
    # 注入对话数据
    total_messages = ingest_conversation_history(graph, raw_data)
    logging.info(f"注入了 {total_messages} 个对话消息")
    
    # 构建索引
    graph.build_semantic_map_index()
    
    # 显示图谱摘要
    graph.display_graph_summary()
    
    # 批量抽取实体关系
    extraction_results = extract_entities_from_locomo_dataset(graph, sample_limit=100)
    
    # 获取统计信息
    integrator = SemanticGraphIntegrator(graph)
    entity_stats = integrator.get_entity_statistics()
    
    return {
        "extraction_results": extraction_results,
        "entity_statistics": entity_stats,
        "graph": graph,
        "total_messages": total_messages
    }

def extract_from_single_text(text: str, model_name: str = "deepseek-chat") -> Dict[str, Any]:
    """从单个文本抽取实体关系 - 适配新架构"""
    logging.info("开始从单个文本抽取实体关系")
    
    # 初始化LLM客户端和抽取器
    llm_client = LLMClient(model_name=model_name)
    extractor = EntityRelationExtractor(llm_client)
    
    # 抽取实体和关系
    entities, relationships, content_keywords = extractor.extract_entities_and_relations(text)
    
    # 创建语义图并添加结果
    graph = SemanticGraph()
    integrator = SemanticGraphIntegrator(graph)
    
    # 创建文档单元
    doc_unit = MemoryUnit(
        uid="single_text_document",
        raw_data={
            "text_content": text,
            "content_type": "single_text_input"
        },
        metadata={
            "data_source": "single_text",
            "created": datetime.now().isoformat()
        }
    )
    
    # 添加到图中
    graph.add_unit(doc_unit, space_names=["single_text_documents"])
    
    # 添加实体和关系
    entity_id_map = integrator.add_entities_to_graph(entities, doc_unit.uid)
    relationship_count = integrator.add_relationships_to_graph(relationships, entity_id_map, doc_unit.uid)
    
    # 构建索引
    graph.build_semantic_map_index()
    
    return {
        "entities": [
            {
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "confidence": e.confidence,
                "source_text": e.source_text[:100] + "..." if len(e.source_text) > 100 else e.source_text
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
                "source_text": r.source_text[:100] + "..." if len(r.source_text) > 100 else r.source_text
            } for r in relationships
        ],
        "content_keywords": content_keywords,
        "graph": graph,
        "entity_count": len(entities),
        "relationship_count": relationship_count,
        "semantic_graph_info": {
            "total_memory_units": len(graph.semantic_map.memory_units),
            "total_memory_spaces": len(graph.semantic_map.memory_spaces),
            "networkx_nodes": graph.nx_graph.number_of_nodes(),
            "networkx_edges": graph.nx_graph.number_of_edges(),
            "faiss_vectors": graph.semantic_map.faiss_index.ntotal if graph.semantic_map.faiss_index else 0
        }
    }

def extract_conv26_only(dataset_path: str, output_dir: str = None) -> Dict[str, Any]:
    """专门处理conv-26样本 - 适配新架构"""
    from benchmark.task_eval.locomo_test_split import load_dataset, ingest_conversation_history
    from datetime import datetime
    
    # 加载数据集
    raw_data = load_dataset(Path(dataset_path))
    if not raw_data:
        raise ValueError("数据集加载失败")
    
    # 找到conv-26样本
    conv26_sample = None
    for sample in raw_data:
        if sample.get('sample_id') == 'conv-26':
            conv26_sample = sample
            break
    
    if not conv26_sample:
        raise ValueError("未找到sample_id为conv-26的样本")
    
    logging.info("找到conv-26样本，开始处理...")
    
    # 初始化语义图 - 使用新架构
    graph = SemanticGraph()
    
    # 注入对话数据
    total_messages = ingest_conversation_history(graph, [conv26_sample])
    logging.info(f"注入了 {total_messages} 个对话消息")
    
    # 构建索引
    graph.build_semantic_map_index()
    
    # 显示图谱摘要
    graph.display_graph_summary()
    
    # 初始化抽取器
    integrator = SemanticGraphIntegrator(graph)
    
    # 定义过滤器
    def conv26_filter(unit):
        return (unit.metadata.get('data_source') == 'locomo_dialog' 
                and unit.metadata.get('conversation_id') == 'conv-26'
                and not unit.metadata.get('entities_extracted', False))
    
    # 执行实体关系抽取
    extraction_results = integrator.batch_extract_entities_from_space(
        space_name="locomo_dialogs",
        max_units=50,  # 可以根据需要调整
        unit_filter=conv26_filter
    )
    
    # 获取统计信息
    entity_stats = integrator.get_entity_statistics()
    
    # 设置输出目录
    if not output_dir:
        output_dir = Path(__file__).parent.parent / "results" / f"conv26_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存语义图
    graph_path = output_dir / "conv26_semantic_graph"
    graph.save_graph(str(graph_path))
    
    # 保存详细结果
    detailed_results = {
        "sample_id": "conv-26",
        "total_messages": total_messages,
        "extraction_results": extraction_results,
        "entity_statistics": entity_stats,
        "graph_structure": {
            "total_memory_units": len(graph.semantic_map.memory_units),
            "total_memory_spaces": len(graph.semantic_map.memory_spaces),
            "networkx_nodes": graph.nx_graph.number_of_nodes(),
            "networkx_edges": graph.nx_graph.number_of_edges(),
            "faiss_vectors": graph.semantic_map.faiss_index.ntotal if graph.semantic_map.faiss_index else 0
        },
        "memory_space_analysis": {
            space_name: len(space.get_unit_uids())
            for space_name, space in graph.semantic_map.memory_spaces.items()
        }
    }
    
    # 保存结果文件
    results_file = output_dir / "conv26_extraction_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 准备返回结果
    results = {
        "sample_id": "conv-26",
        "total_messages": total_messages,
        "extraction_results": extraction_results,
        "entity_statistics": entity_stats,
        "graph_saved_path": str(graph_path),
        "output_directory": str(output_dir),
        "results_file": str(results_file),
        "graph_structure": detailed_results["graph_structure"],
        "memory_space_analysis": detailed_results["memory_space_analysis"]
    }
    
    return results

def save_extraction_results(results: Dict[str, Any], output_path: str, mode: str):
    """保存抽取结果到文件 - 适配新架构"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备保存数据，排除不能序列化的对象
    save_data = {
        "metadata": {
            "mode": mode,
            "created_at": datetime.now().isoformat(),
            "semantic_graph_version": "new_architecture"
        }
    }
    
    # 根据模式保存不同的数据
    if mode == "locomo":
        save_data.update({
            "total_messages": results.get("total_messages", 0),
            "extraction_results": results.get("extraction_results", {}),
            "entity_statistics": results.get("entity_statistics", {}),
            "graph_summary": {
                "total_memory_units": len(results["graph"].semantic_map.memory_units) if results.get("graph") else 0,
                "total_memory_spaces": len(results["graph"].semantic_map.memory_spaces) if results.get("graph") else 0,
                "networkx_nodes": results["graph"].nx_graph.number_of_nodes() if results.get("graph") else 0,
                "networkx_edges": results["graph"].nx_graph.number_of_edges() if results.get("graph") else 0,
                "faiss_vectors": results["graph"].semantic_map.faiss_index.ntotal if results.get("graph") and results["graph"].semantic_map.faiss_index else 0
            }
        })
    elif mode == "text":
        save_data.update({
            "entities": results.get("entities", []),
            "relationships": results.get("relationships", []),
            "content_keywords": results.get("content_keywords", []),
            "entity_count": results.get("entity_count", 0),
            "relationship_count": results.get("relationship_count", 0),
            "semantic_graph_info": results.get("semantic_graph_info", {})
        })
    elif mode == "conv26":
        save_data.update({
            "sample_id": results.get("sample_id", ""),
            "total_messages": results.get("total_messages", 0),
            "extraction_results": results.get("extraction_results", {}),
            "entity_statistics": results.get("entity_statistics", {}),
            "graph_structure": results.get("graph_structure", {}),
            "memory_space_analysis": results.get("memory_space_analysis", {}),
            "output_directory": results.get("output_directory", ""),
            "graph_saved_path": results.get("graph_saved_path", "")
        })
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    
    logging.info(f"抽取结果已保存到: {output_path}")

def main():
    """主函数 - 适配新架构"""
    parser = argparse.ArgumentParser(description="实体关系抽取工具 (新架构版本)")
    parser.add_argument("--mode", choices=["locomo", "text", "conv26"], required=True,
                       help="抽取模式: locomo(整个数据集) 或 text(单个文本) 或 conv26(专门处理conv-26)")
    parser.add_argument("--input", required=True,
                       help="输入: LoCoMo数据集路径 或 文本内容")
    parser.add_argument("--output", help="输出目录路径")
    parser.add_argument("--model", default="deepseek-chat",
                       help="LLM模型名称")
    parser.add_argument("--sample-limit", type=int,
                       help="限制处理的样本数量(仅用于locomo模式)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--save-graph", action="store_true",
                       help="保存语义图到磁盘")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        if args.mode == "conv26":
            results = extract_conv26_only(args.input, args.output)
            
            # 保存结果摘要（已在函数内保存）
            print(f"\n🎉 Conv-26实体关系抽取完成!")
            print(f"📁 输出目录: {results['output_directory']}")
            print(f"📊 总消息数: {results['total_messages']}")
            print(f"📊 处理单元数: {results['extraction_results']['processed']}")
            print(f"📊 抽取实体数: {results['extraction_results']['total_entities']}")
            print(f"📊 抽取关系数: {results['extraction_results']['total_relationships']}")
            print(f"📊 实体类型分布: {results['entity_statistics']['entity_types']}")
            print(f"📊 关系类型分布: {results['entity_statistics']['relationship_types']}")
            print(f"📊 图结构信息:")
            print(f"   - 内存单元数: {results['graph_structure']['total_memory_units']}")
            print(f"   - 内存空间数: {results['graph_structure']['total_memory_spaces']}")
            print(f"   - NetworkX节点数: {results['graph_structure']['networkx_nodes']}")
            print(f"   - NetworkX边数: {results['graph_structure']['networkx_edges']}")
            print(f"   - FAISS向量数: {results['graph_structure']['faiss_vectors']}")
            
        elif args.mode == "locomo":
            results = extract_from_locomo_dataset(args.input, args.sample_limit)
            
            # 保存结果
            if args.output:
                output_file = Path(args.output) / f"locomo_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_extraction_results(results, output_file, "locomo")
            
            # 可选：保存语义图
            if args.save_graph and args.output:
                graph_path = Path(args.output) / f"locomo_semantic_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                results["graph"].save_graph(str(graph_path))
                print(f"📁 语义图已保存到: {graph_path}")
            
            print(f"\n🎉 LoCoMo数据集实体关系抽取完成!")
            print(f"📊 总消息数: {results['total_messages']}")
            print(f"📊 处理单元数: {results['extraction_results']['processed']}")
            print(f"📊 抽取实体数: {results['extraction_results']['total_entities']}")
            print(f"📊 抽取关系数: {results['extraction_results']['total_relationships']}")
            print(f"📊 实体类型分布: {results['entity_statistics']['entity_types']}")
            print(f"📊 关系类型分布: {results['entity_statistics']['relationship_types']}")
            
        elif args.mode == "text":
            results = extract_from_single_text(args.input, args.model)
            
            # 保存结果
            if args.output:
                output_file = Path(args.output) / f"text_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_extraction_results(results, output_file, "text")
            
            # 可选：保存语义图
            if args.save_graph and args.output:
                graph_path = Path(args.output) / f"text_semantic_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                results["graph"].save_graph(str(graph_path))
                print(f"📁 语义图已保存到: {graph_path}")
            
            print(f"\n🎉 单文本实体关系抽取完成!")
            print(f"📊 抽取实体数: {results['entity_count']}")
            print(f"📊 抽取关系数: {results['relationship_count']}")
            print(f"📊 关键词数: {len(results['content_keywords'])}")
            print(f"📊 语义图信息: {results['semantic_graph_info']}")
            
            # 显示前几个结果
            print("\n📋 实体示例:")
            for i, entity in enumerate(results['entities'][:3]):
                print(f"  {i+1}. {entity['name']} ({entity['type']}): {entity['description']}")
            
            print("\n🔗 关系示例:")
            for i, rel in enumerate(results['relationships'][:3]):
                print(f"  {i+1}. {rel['source']} -[{rel['type']}]-> {rel['target']}")
        
        logging.info("任务执行完成")
        
    except Exception as e:
        logging.error(f"执行失败: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()