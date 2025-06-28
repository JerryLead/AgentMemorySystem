import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from entity_relation_extractor import EntityRelationExtractor
from dev.semantic_graph import SemanticGraph
from benchmark.llm_utils.llm_client import LLMClient
# from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor
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
    从LoCoMo对话中批量抽取实体关系
    
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

# def extract_from_locomo_dataset(dataset_path: str, sample_limit: int = None) -> Dict[str, Any]:
#     """从LoCoMo数据集抽取实体关系"""
#     from benchmark.task_eval.locomo_test_split import (
#         load_dataset, ingest_conversation_history
#     )
    
#     logging.info(f"加载LoCoMo数据集: {dataset_path}")
    
#     # 加载数据集
#     raw_data = load_dataset(Path(dataset_path))
#     if not raw_data:
#         raise ValueError("数据集加载失败")
    
#     if sample_limit:
#         raw_data = raw_data[:sample_limit]
#         logging.info(f"限制处理样本数量: {sample_limit}")
    
#     # 初始化语义图
#     graph = SemanticGraph()
    
#     # 注入对话数据
#     total_messages = ingest_conversation_history(graph, raw_data)
#     logging.info(f"注入了 {total_messages} 个对话消息")
    
#     # 构建索引
#     graph.build_semantic_map_index()
    
#     # 初始化实体抽取集成器
#     integrator = SemanticGraphIntegrator(graph)
    
#     # 定义过滤器：只处理对话内容
#     def dialog_filter(unit):
#         return unit.metadata.get('data_source') == 'locomo_dialog'
    
#     # 批量抽取实体
#     results = integrator.batch_extract_entities_from_graph(
#         space_name="locomo_dialogs",
#         max_units=100,  # 限制处理数量
#         unit_filter=dialog_filter
#     )
    
#     # 获取统计信息
#     stats = integrator.get_entity_statistics()
    
#     # 显示图谱信息
#     graph.display_graph_summary()
    
#     return {
#         "extraction_results": results,
#         "entity_statistics": stats,
#         "graph": graph
#     }

def extract_from_single_text(text: str, model_name: str = "deepseek-chat") -> Dict[str, Any]:
    """从单个文本抽取实体关系"""
    logging.info("开始从单个文本抽取实体关系")
    
    # 初始化LLM客户端和抽取器
    llm_client = LLMClient(model_name=model_name)
    extractor = EntityRelationExtractor(llm_client)
    
    # 抽取实体和关系
    entities, relationships, content_keywords = extractor.extract_entities_and_relations(text)
    
    return {
        "entities": [
            {
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "confidence": e.confidence
            } for e in entities
        ],
        "relationships": [
            {
                "source": r.source_entity,
                "target": r.target_entity,
                "type": r.relationship_type,
                "description": r.description,
                "keywords": r.keywords,
                "strength": r.strength
            } for r in relationships
        ],
        "content_keywords": content_keywords
    }

def extract_conv26_only(dataset_path: str, output_dir: str = None) -> Dict[str, Any]:
    """专门处理conv-26样本"""
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
    
    # 初始化语义图
    graph = SemanticGraph()
    
    # 注入对话数据
    total_messages = ingest_conversation_history(graph, [conv26_sample])
    logging.info(f"注入了 {total_messages} 个对话消息")
    
    # 构建索引
    graph.build_semantic_map_index()
    
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
    
    # 准备返回结果
    results = {
        "sample_id": "conv-26",
        "total_messages": total_messages,
        "extraction_results": extraction_results,
        "entity_statistics": entity_stats,
        "graph_saved_path": str(graph_path),
        "output_directory": str(output_dir)
    }
    
    return results

# 修改main函数，添加conv-26选项
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实体关系抽取工具")
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
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        if args.mode == "conv26":
            results = extract_conv26_only(args.input, args.output)
            
            # 保存结果摘要
            summary_file = Path(results["output_directory"]) / "extraction_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n🎉 Conv-26实体关系抽取完成!")
            print(f"📁 输出目录: {results['output_directory']}")
            print(f"📊 总消息数: {results['total_messages']}")
            print(f"📊 处理单元数: {results['extraction_results']['processed']}")
            print(f"📊 抽取实体数: {results['extraction_results']['total_entities']}")
            print(f"📊 抽取关系数: {results['extraction_results']['total_relationships']}")
            print(f"📊 实体类型分布: {results['entity_statistics']['entity_types']}")
            print(f"📊 关系类型分布: {results['entity_statistics']['relationship_types']}")
            
        elif args.mode == "locomo":
            # 现有的locomo处理逻辑
            pass
        elif args.mode == "text":
            # 现有的text处理逻辑
            results = extract_from_single_text(args.input, args.model)
        
        logging.info("任务执行完成")
        
    except Exception as e:
        logging.error(f"执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()