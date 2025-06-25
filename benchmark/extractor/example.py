"""实体关系抽取使用示例"""

import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from dev.semantic_graph import SemanticGraph
from dev.memory_unit import MemoryUnit
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor
from benchmark.extractor.semantic_graph_integrator import SemanticGraphIntegrator

# 设置日志
logging.basicConfig(level=logging.INFO)

def example_extract_from_text():
    """示例：从文本抽取实体关系"""
    text = """
    Caroline和Melanie是好朋友。Caroline住在纽约，是一名心理咨询师，专门帮助LGBTQ+群体。
    她最近通过了收养机构的面试，希望能够建立自己的家庭。
    Melanie住在加州，是一位艺术家，经常画画和制作陶器。
    她们经常通过电话讨论生活和工作，互相支持。
    """
    
    print("=== 从文本抽取实体关系示例 ===")
    
    # 初始化
    llm_client = LLMClient(model_name="deepseek-chat")
    extractor = EntityRelationExtractor(llm_client)
    
    # 抽取
    entities, relationships, keywords = extractor.extract_entities_and_relations(text)
    
    # 显示结果
    print(f"\n抽取到 {len(entities)} 个实体:")
    for entity in entities:
        print(f"  - {entity.name} ({entity.entity_type}): {entity.description}")
    
    print(f"\n抽取到 {len(relationships)} 个关系:")
    for rel in relationships:
        print(f"  - {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity}")
        print(f"    描述: {rel.description}")
        print(f"    强度: {rel.strength}")
    
    print(f"\n内容关键词: {', '.join(keywords)}")

# def example_integrate_to_graph():
#     """示例：将抽取结果集成到语义图"""
#     print("\n=== 集成到语义图示例 ===")
    
#     # 创建语义图
#     graph = SemanticGraph()
#     integrator = SemanticGraphIntegrator(graph)
    
#     # 模拟一些对话数据
#     from dev.memory_unit import MemoryUnit
    
#     # 添加原始对话单元
#     dialog_unit = MemoryUnit(
#         uid="dialog_001",
#         raw_data={
#             "text_content": "Caroline: 我最近通过了收养机构的面试！Melanie: 太棒了，你一直想要自己的家庭。",
#             "speaker": "Caroline, Melanie",
#             "session": "session_1"
#         },
#         metadata={
#             "data_source": "locomo_dialog",
#             "session": "session_1"
#         }
#     )
    
#     graph.add_unit(dialog_unit)
    
#     # 对这个单元进行实体抽取
#     result = integrator.process_memory_unit_for_entities(dialog_unit)
    
#     print(f"处理结果: {result}")
    
#     # 显示图谱统计
#     graph.display_graph_summary()
    
#     # 显示实体统计
#     stats = integrator.get_entity_statistics()
#     print(f"\n实体统计: {stats}")

def example_integrate_to_graph():
    """示例：将抽取结果集成到语义图"""
    print("\n=== 集成到语义图示例 ===")
    
    # 创建语义图
    graph = SemanticGraph()
    integrator = SemanticGraphIntegrator(graph)

    
    # 添加原始对话单元
    dialog_unit = MemoryUnit(
        uid="dialog_001",
        raw_data={
            "text_content": "Caroline: 我最近通过了收养机构的面试！Melanie: 太棒了，你一直想要自己的家庭。Caroline住在纽约，是心理咨询师。Melanie住在加州，是艺术家。",
            "speaker": "Caroline, Melanie",
            "session": "session_1"
        },
        metadata={
            "data_source": "locomo_dialog",
            "session": "session_1"
        }
    )
    
    # 先添加对话单元到图中
    graph.add_unit(dialog_unit, space_names=["locomo_dialogs"])
    
    # 对这个单元进行实体抽取
    result = integrator.process_memory_unit_for_entities(dialog_unit)
    
    print(f"处理结果: {result}")
    
    # 显示图谱统计
    graph.display_graph_summary()
    
    # 显示实体统计
    stats = integrator.get_entity_statistics()
    print(f"\n实体统计: {stats}")
    
    # 显示一些抽取到的实体
    entity_space = graph.semantic_map.get_memory_space("extracted_entities")
    if entity_space:
        print(f"\n抽取到的实体:")
        for uid in list(entity_space.get_memory_uids())[:5]:  # 只显示前5个
            unit = graph.get_unit(uid)
            if unit:
                print(f"  - {unit.raw_data.get('entity_name')} ({unit.raw_data.get('entity_type')}): {unit.raw_data.get('description')}")

def example_batch_processing():
    """示例：批量处理LoCoMo数据集"""
    print("\n=== 批量处理示例 ===")
    
    # 这个示例需要真实的LoCoMo数据集
    dataset_path = "/path/to/locomo/dataset.json"
    
    if not Path(dataset_path).exists():
        print("数据集文件不存在，跳过批量处理示例")
        return
    
    try:
        from .extract_entities_cli import extract_from_locomo_dataset
        
        results = extract_from_locomo_dataset(dataset_path, sample_limit=2)
        
        print("批量处理完成!")
        print(f"处理结果: {results['extraction_results']}")
        print(f"实体统计: {results['entity_statistics']}")
        
    except Exception as e:
        print(f"批量处理示例失败: {e}")

if __name__ == "__main__":
    # 运行示例
    example_extract_from_text()
    example_integrate_to_graph()
    example_batch_processing()