import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from dev.semantic_graph import SemanticGraph
from dev.memory_unit import MemoryUnit
from dev.memory_space import MemorySpace

class LoCoMoDatasetMaker:
    """
    LoCoMo 数据集制作器 - 适配新的 semantic_map/graph 架构
    将所有样本的实体关系抽取结果整合成统一的数据集文件
    """
    
    def __init__(self, results_dir: str, output_dir: str = None):
        """
        初始化数据集制作器
        
        Args:
            results_dir: 抽取结果目录路径 (如 benchmark/extractor/results)
            output_dir: 输出目录路径 (默认为 benchmark/dataset/locomo/extraction)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark/dataset/locomo/extraction")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"数据集制作器初始化完成")
        self.logger.info(f"结果目录: {self.results_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
    
    def collect_extraction_results(self) -> Dict[str, Any]:
        """
        收集所有样本的抽取结果 - 适配新架构
        
        Returns:
            包含所有样本数据的字典
        """
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source_directory": str(self.results_dir),
                "total_samples": 0,
                "extraction_summary": {},
                "semantic_graph_version": "new_architecture"
            },
            "samples": {}
        }
        
        # 查找所有样本的抽取结果目录
        sample_dirs = [d for d in self.results_dir.iterdir() 
                      if d.is_dir() and ('conv-' in d.name or 'entity_extraction' in d.name)]
        
        self.logger.info(f"发现 {len(sample_dirs)} 个样本目录")
        
        total_entities = 0
        total_relationships = 0
        total_keywords = 0
        total_memory_units = 0
        total_memory_spaces = 0
        
        for sample_dir in sorted(sample_dirs):
            try:
                sample_data = self._process_sample_directory(sample_dir)
                if sample_data:
                    sample_id = sample_data["sample_info"]["sample_id"]
                    dataset["samples"][sample_id] = sample_data
                    
                    # 累计统计 - 适配新架构
                    stats = sample_data.get("extraction_statistics", {})
                    total_entities += stats.get("total_entities", 0)
                    total_relationships += stats.get("total_relationships", 0)
                    total_keywords += stats.get("total_keywords", 0)
                    
                    # 新架构特有的统计
                    graph_structure = sample_data.get("graph_structure", {})
                    total_memory_units += graph_structure.get("total_memory_units", 0)
                    total_memory_spaces += graph_structure.get("total_memory_spaces", 0)
                    
                    self.logger.info(f"✅ 成功处理样本: {sample_id}")
                else:
                    self.logger.warning(f"⚠️  跳过无效样本目录: {sample_dir.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ 处理样本目录 {sample_dir.name} 时出错: {e}")
        
        # 更新元数据 - 包含新架构信息
        dataset["metadata"]["total_samples"] = len(dataset["samples"])
        dataset["metadata"]["extraction_summary"] = {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "total_keywords": total_keywords,
            "total_memory_units": total_memory_units,
            "total_memory_spaces": total_memory_spaces,
            "avg_entities_per_sample": total_entities / len(dataset["samples"]) if dataset["samples"] else 0,
            "avg_relationships_per_sample": total_relationships / len(dataset["samples"]) if dataset["samples"] else 0,
            "avg_memory_units_per_sample": total_memory_units / len(dataset["samples"]) if dataset["samples"] else 0
        }
        
        return dataset
    
    def _process_sample_directory(self, sample_dir: Path) -> Optional[Dict[str, Any]]:
        """
        处理单个样本目录 - 适配新架构
        
        Args:
            sample_dir: 样本目录路径
            
        Returns:
            样本数据字典，如果处理失败返回None
        """
        # 查找抽取结果文件
        result_files = list(sample_dir.glob("*_extraction_results.json"))
        
        if not result_files:
            self.logger.warning(f"未找到抽取结果文件: {sample_dir}")
            return None
        
        # 使用最新的结果文件
        result_file = sorted(result_files)[-1]
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            # 添加文件路径信息
            sample_data["source_files"] = {
                "extraction_results": str(result_file.relative_to(self.results_dir)),
                "directory": str(sample_dir.relative_to(self.results_dir))
            }
            
            # 查找其他相关文件 - 适配新架构
            additional_files = {}
            file_patterns = [
                "*_full_conversation.txt",
                "*_detailed_analysis.json", 
                "*_summary_report.txt",
                "*semantic_graph*",
                "*semantic_map_data*",
                "*management_state.pkl"
            ]
            
            for pattern in file_patterns:
                files = list(sample_dir.glob(pattern))
                if files:
                    pattern_key = pattern.replace("*", "").replace(".", "").replace("/", "_")
                    additional_files[pattern_key] = [
                        str(f.relative_to(self.results_dir)) for f in files
                    ]
            
            if additional_files:
                sample_data["source_files"].update(additional_files)
            
            # 尝试加载语义图谱信息 - 新架构
            semantic_graph_dirs = list(sample_dir.glob("*semantic_graph*"))
            if semantic_graph_dirs:
                graph_info = self._extract_semantic_graph_info(semantic_graph_dirs[0])
                if graph_info:
                    sample_data["semantic_graph_info"] = graph_info
            
            return sample_data
            
        except Exception as e:
            self.logger.error(f"读取结果文件 {result_file} 时出错: {e}")
            return None
    
    def _extract_semantic_graph_info(self, graph_dir: Path) -> Optional[Dict[str, Any]]:
        """
        从语义图谱目录中提取信息 - 新架构
        
        Args:
            graph_dir: 语义图谱目录
            
        Returns:
            语义图谱信息字典
        """
        try:
            # 检查语义图谱文件
            semantic_map_dir = graph_dir / "semantic_map_data"
            if semantic_map_dir.exists():
                # 尝试读取语义图谱基本信息
                data_file = semantic_map_dir / "semantic_map_data.pkl"
                if data_file.exists():
                    # 由于是pickle文件，我们只记录存在信息
                    return {
                        "has_semantic_map": True,
                        "semantic_map_path": str(semantic_map_dir),
                        "data_file_size": data_file.stat().st_size,
                        "last_modified": datetime.fromtimestamp(data_file.stat().st_mtime).isoformat()
                    }
            
            # 检查NetworkX图文件
            nx_files = list(graph_dir.glob("*.gml")) + list(graph_dir.glob("*.pkl"))
            if nx_files:
                return {
                    "has_networkx_graph": True,
                    "graph_files": [str(f.relative_to(graph_dir)) for f in nx_files],
                    "graph_file_sizes": {f.name: f.stat().st_size for f in nx_files}
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"提取语义图谱信息失败: {e}")
            return None
    
    def create_entity_relationship_summary(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建实体关系汇总 - 适配新架构
        
        Args:
            dataset: 完整数据集
            
        Returns:
            实体关系汇总数据
        """
        summary = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_samples": len(dataset["samples"]),
                "semantic_graph_version": "new_architecture"
            },
            "entity_summary": {
                "all_entities": {},
                "entity_types": {},
                "entity_frequency": {},
                "entity_confidence_stats": {}
            },
            "relationship_summary": {
                "all_relationships": {},
                "relationship_types": {},
                "relationship_patterns": {},
                "relationship_strength_stats": {}
            },
            "memory_space_analysis": {
                "space_types": {},
                "space_unit_distribution": {},
                "average_units_per_space": 0
            },
            "cross_sample_analysis": {
                "common_entities": {},
                "common_relationships": {},
                "entity_co_occurrence": {},
                "semantic_similarity_clusters": {}
            }
        }
        
        all_entities = {}
        all_relationships = {}
        entity_types_count = {}
        relationship_types_count = {}
        entity_confidences = []
        relationship_strengths = []
        space_info = {}
        
        # 遍历所有样本
        for sample_id, sample_data in dataset["samples"].items():
            entities = sample_data.get("entities", [])
            relationships = sample_data.get("relationships", [])
            graph_structure = sample_data.get("graph_structure", {})
            
            # 处理实体 - 包含新架构的置信度信息
            for entity in entities:
                entity_name = entity.get("name", "").lower()
                entity_type = entity.get("type", "unknown")
                confidence = entity.get("confidence", 0.0)
                
                if entity_name:
                    # 实体频率统计
                    if entity_name not in all_entities:
                        all_entities[entity_name] = {
                            "name": entity.get("name", ""),
                            "type": entity_type,
                            "descriptions": [],
                            "samples": [],
                            "frequency": 0,
                            "avg_confidence": 0.0,
                            "source_texts": []
                        }
                    
                    all_entities[entity_name]["descriptions"].append({
                        "sample_id": sample_id,
                        "description": entity.get("description", ""),
                        "confidence": confidence,
                        "source_text": entity.get("source_text", "")[:100] + "..." if len(entity.get("source_text", "")) > 100 else entity.get("source_text", "")
                    })
                    all_entities[entity_name]["samples"].append(sample_id)
                    all_entities[entity_name]["frequency"] += 1
                    
                    # 置信度统计
                    entity_confidences.append(confidence)
                
                # 实体类型统计
                entity_types_count[entity_type] = entity_types_count.get(entity_type, 0) + 1
            
            # 处理关系 - 包含新架构的强度信息
            for relationship in relationships:
                source = relationship.get("source", "").lower()
                target = relationship.get("target", "").lower()
                rel_type = relationship.get("type", "unknown")
                strength = relationship.get("strength", 0.0)
                
                rel_key = f"{source}-{rel_type}-{target}"
                
                if rel_key not in all_relationships:
                    all_relationships[rel_key] = {
                        "source": relationship.get("source", ""),
                        "target": relationship.get("target", ""),
                        "type": rel_type,
                        "descriptions": [],
                        "samples": [],
                        "frequency": 0,
                        "avg_strength": 0.0,
                        "keywords": set()
                    }
                
                all_relationships[rel_key]["descriptions"].append({
                    "sample_id": sample_id,
                    "description": relationship.get("description", ""),
                    "strength": strength,
                    "keywords": relationship.get("keywords", [])
                })
                all_relationships[rel_key]["samples"].append(sample_id)
                all_relationships[rel_key]["frequency"] += 1
                all_relationships[rel_key]["keywords"].update(relationship.get("keywords", []))
                
                # 强度统计
                relationship_strengths.append(strength)
                
                # 关系类型统计
                relationship_types_count[rel_type] = relationship_types_count.get(rel_type, 0) + 1
            
            # 处理内存空间信息 - 新架构特有
            memory_spaces = graph_structure.get("total_memory_spaces", 0)
            memory_units = graph_structure.get("total_memory_units", 0)
            if memory_spaces > 0:
                space_info[sample_id] = {
                    "spaces": memory_spaces,
                    "units": memory_units,
                    "units_per_space": memory_units / memory_spaces if memory_spaces > 0 else 0
                }
        
        # 计算平均值和统计信息
        for entity_data in all_entities.values():
            confidences = [desc["confidence"] for desc in entity_data["descriptions"] if desc["confidence"] > 0]
            entity_data["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        for rel_data in all_relationships.values():
            strengths = [desc["strength"] for desc in rel_data["descriptions"] if desc["strength"] > 0]
            rel_data["avg_strength"] = sum(strengths) / len(strengths) if strengths else 0.0
            rel_data["keywords"] = list(rel_data["keywords"])
        
        # 填充汇总数据
        summary["entity_summary"]["all_entities"] = dict(sorted(
            all_entities.items(), key=lambda x: x[1]["frequency"], reverse=True
        ))
        summary["entity_summary"]["entity_types"] = dict(sorted(
            entity_types_count.items(), key=lambda x: x[1], reverse=True
        ))
        summary["entity_summary"]["entity_frequency"] = {
            "total_unique_entities": len(all_entities),
            "most_frequent": max(all_entities.values(), key=lambda x: x["frequency"]) if all_entities else None,
            "cross_sample_entities": {k: v for k, v in all_entities.items() if v["frequency"] > 1}
        }
        summary["entity_summary"]["entity_confidence_stats"] = {
            "avg_confidence": sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0.0,
            "min_confidence": min(entity_confidences) if entity_confidences else 0.0,
            "max_confidence": max(entity_confidences) if entity_confidences else 0.0
        }
        
        summary["relationship_summary"]["all_relationships"] = dict(sorted(
            all_relationships.items(), key=lambda x: x[1]["frequency"], reverse=True
        ))
        summary["relationship_summary"]["relationship_types"] = dict(sorted(
            relationship_types_count.items(), key=lambda x: x[1], reverse=True
        ))
        summary["relationship_summary"]["relationship_patterns"] = {
            "total_unique_relationships": len(all_relationships),
            "most_frequent": max(all_relationships.values(), key=lambda x: x["frequency"]) if all_relationships else None,
            "cross_sample_relationships": {k: v for k, v in all_relationships.items() if v["frequency"] > 1}
        }
        summary["relationship_summary"]["relationship_strength_stats"] = {
            "avg_strength": sum(relationship_strengths) / len(relationship_strengths) if relationship_strengths else 0.0,
            "min_strength": min(relationship_strengths) if relationship_strengths else 0.0,
            "max_strength": max(relationship_strengths) if relationship_strengths else 0.0
        }
        
        # 内存空间分析 - 新架构特有
        if space_info:
            total_spaces = sum(info["spaces"] for info in space_info.values())
            total_units = sum(info["units"] for info in space_info.values())
            
            summary["memory_space_analysis"] = {
                "total_spaces_across_samples": total_spaces,
                "total_units_across_samples": total_units,
                "average_spaces_per_sample": total_spaces / len(space_info) if space_info else 0,
                "average_units_per_sample": total_units / len(space_info) if space_info else 0,
                "average_units_per_space": total_units / total_spaces if total_spaces > 0 else 0,
                "sample_space_distribution": space_info
            }
        
        return summary
    
    def create_sample_index(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建样本索引文件 - 适配新架构
        
        Args:
            dataset: 完整数据集
            
        Returns:
            样本索引数据
        """
        index = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_samples": len(dataset["samples"]),
                "semantic_graph_version": "new_architecture"
            },
            "sample_index": {}
        }
        
        for sample_id, sample_data in dataset["samples"].items():
            sample_info = sample_data.get("sample_info", {})
            extraction_stats = sample_data.get("extraction_statistics", {})
            graph_structure = sample_data.get("graph_structure", {})
            semantic_graph_info = sample_data.get("semantic_graph_info", {})
            
            index["sample_index"][sample_id] = {
                # 基本信息
                "speakers": sample_info.get("speakers", []),
                "session_count": sample_info.get("session_count", 0),
                "total_messages": sample_info.get("total_messages", 0),
                
                # 抽取统计
                "entity_count": extraction_stats.get("total_entities", 0),
                "relationship_count": extraction_stats.get("total_relationships", 0),
                "keyword_count": extraction_stats.get("total_keywords", 0),
                "processing_time": extraction_stats.get("processing_time", ""),
                
                # 新架构特有信息
                "memory_units": graph_structure.get("total_memory_units", 0),
                "memory_spaces": graph_structure.get("total_memory_spaces", 0),
                "networkx_nodes": graph_structure.get("networkx_nodes", 0),
                "networkx_edges": graph_structure.get("networkx_edges", 0),
                "faiss_vectors": graph_structure.get("faiss_vectors", 0),
                
                # 语义图谱信息
                "has_semantic_graph": bool(semantic_graph_info),
                "semantic_graph_size": semantic_graph_info.get("data_file_size", 0) if semantic_graph_info else 0,
                
                # 文件路径
                "source_directory": sample_data.get("source_files", {}).get("directory", "")
            }
        
        return index
    
    def generate_dataset_files(self, output_prefix: str = "locomo_extracted") -> Dict[str, Path]:
        """
        生成完整的数据集文件 - 适配新架构
        
        Args:
            output_prefix: 输出文件前缀
            
        Returns:
            生成的文件路径字典
        """
        self.logger.info("🚀 开始生成 LoCoMo 抽取数据集（新架构版本）...")
        
        # 1. 收集所有抽取结果
        self.logger.info("📊 收集抽取结果...")
        dataset = self.collect_extraction_results()
        
        if not dataset["samples"]:
            raise ValueError("未找到任何有效的抽取结果")
        
        # 2. 生成完整数据集文件
        full_dataset_path = self.output_dir / f"{output_prefix}_full_dataset.json"
        with open(full_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 完整数据集已保存: {full_dataset_path}")
        
        # 3. 生成实体关系汇总
        self.logger.info("📈 生成实体关系汇总...")
        summary = self.create_entity_relationship_summary(dataset)
        summary_path = self.output_dir / f"{output_prefix}_entity_relationship_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 实体关系汇总已保存: {summary_path}")
        
        # 4. 生成样本索引
        self.logger.info("📋 生成样本索引...")
        index = self.create_sample_index(dataset)
        index_path = self.output_dir / f"{output_prefix}_sample_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 样本索引已保存: {index_path}")
        
        # 5. 生成轻量级版本（仅包含核心数据）
        self.logger.info("💾 生成轻量级数据集...")
        lightweight_dataset = self._create_lightweight_dataset(dataset)
        lightweight_path = self.output_dir / f"{output_prefix}_lightweight.json"
        with open(lightweight_path, 'w', encoding='utf-8') as f:
            json.dump(lightweight_dataset, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 轻量级数据集已保存: {lightweight_path}")
        
        # 6. 生成架构兼容性报告 - 新功能
        self.logger.info("🔄 生成架构兼容性报告...")
        compatibility_report = self._generate_compatibility_report(dataset)
        compatibility_path = self.output_dir / f"{output_prefix}_architecture_compatibility.json"
        with open(compatibility_path, 'w', encoding='utf-8') as f:
            json.dump(compatibility_report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 架构兼容性报告已保存: {compatibility_path}")
        
        # 7. 生成统计报告
        self.logger.info("📊 生成统计报告...")
        report = self._generate_statistics_report(dataset, summary)
        report_path = self.output_dir / f"{output_prefix}_statistics_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"✅ 统计报告已保存: {report_path}")
        
        generated_files = {
            "full_dataset": full_dataset_path,
            "entity_relationship_summary": summary_path,
            "sample_index": index_path,
            "lightweight_dataset": lightweight_path,
            "architecture_compatibility": compatibility_path,
            "statistics_report": report_path
        }
        
        self.logger.info(f"🎉 数据集生成完成！共生成 {len(generated_files)} 个文件")
        return generated_files
    
    def _generate_compatibility_report(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成架构兼容性报告 - 新功能
        
        Args:
            dataset: 完整数据集
            
        Returns:
            兼容性报告数据
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "architecture_version": "new_semantic_graph",
                "total_samples": len(dataset["samples"])
            },
            "compatibility_analysis": {
                "memory_space_usage": {},
                "memory_unit_distribution": {},
                "semantic_graph_features": {},
                "faiss_index_usage": {}
            },
            "migration_recommendations": [],
            "performance_metrics": {}
        }
        
        memory_space_counts = []
        memory_unit_counts = []
        faiss_vector_counts = []
        samples_with_semantic_graph = 0
        
        for sample_id, sample_data in dataset["samples"].items():
            graph_structure = sample_data.get("graph_structure", {})
            semantic_graph_info = sample_data.get("semantic_graph_info", {})
            
            # 内存空间使用情况
            memory_spaces = graph_structure.get("total_memory_spaces", 0)
            memory_units = graph_structure.get("total_memory_units", 0)
            faiss_vectors = graph_structure.get("faiss_vectors", 0)
            
            memory_space_counts.append(memory_spaces)
            memory_unit_counts.append(memory_units)
            faiss_vector_counts.append(faiss_vectors)
            
            if semantic_graph_info:
                samples_with_semantic_graph += 1
        
        # 统计分析
        if memory_space_counts:
            report["compatibility_analysis"]["memory_space_usage"] = {
                "average_spaces_per_sample": sum(memory_space_counts) / len(memory_space_counts),
                "min_spaces": min(memory_space_counts),
                "max_spaces": max(memory_space_counts),
                "total_spaces": sum(memory_space_counts)
            }
        
        if memory_unit_counts:
            report["compatibility_analysis"]["memory_unit_distribution"] = {
                "average_units_per_sample": sum(memory_unit_counts) / len(memory_unit_counts),
                "min_units": min(memory_unit_counts),
                "max_units": max(memory_unit_counts),
                "total_units": sum(memory_unit_counts)
            }
        
        if faiss_vector_counts:
            report["compatibility_analysis"]["faiss_index_usage"] = {
                "average_vectors_per_sample": sum(faiss_vector_counts) / len(faiss_vector_counts),
                "samples_with_vectors": len([c for c in faiss_vector_counts if c > 0]),
                "total_vectors": sum(faiss_vector_counts)
            }
        
        report["compatibility_analysis"]["semantic_graph_features"] = {
            "samples_with_semantic_graph": samples_with_semantic_graph,
            "semantic_graph_coverage": samples_with_semantic_graph / len(dataset["samples"]) if dataset["samples"] else 0
        }
        
        # 迁移建议
        if samples_with_semantic_graph < len(dataset["samples"]):
            report["migration_recommendations"].append({
                "issue": "Incomplete semantic graph coverage",
                "description": f"Only {samples_with_semantic_graph}/{len(dataset['samples'])} samples have semantic graph data",
                "recommendation": "Re-run entity extraction with semantic graph persistence enabled"
            })
        
        if sum(faiss_vector_counts) == 0:
            report["migration_recommendations"].append({
                "issue": "No FAISS vectors found",
                "description": "Samples do not contain FAISS index vectors",
                "recommendation": "Rebuild semantic map indexes to enable vector similarity search"
            })
        
        return report
    
    def _create_lightweight_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """创建轻量级数据集（移除source_text等大字段）- 适配新架构"""
        lightweight = {
            "metadata": dataset["metadata"],
            "samples": {}
        }
        
        for sample_id, sample_data in dataset["samples"].items():
            lightweight_sample = {
                "sample_info": sample_data.get("sample_info", {}),
                "extraction_statistics": sample_data.get("extraction_statistics", {}),
                "entities": [],
                "relationships": [],
                "content_keywords": sample_data.get("content_keywords", []),
                "entity_statistics": sample_data.get("entity_statistics", {}),
                "graph_structure": sample_data.get("graph_structure", {}),
                # 新架构特有字段
                "memory_space_summary": self._summarize_memory_spaces(sample_data),
                "semantic_graph_summary": self._summarize_semantic_graph(sample_data)
            }
            
            # 简化实体信息
            for entity in sample_data.get("entities", []):
                lightweight_sample["entities"].append({
                    "name": entity.get("name", ""),
                    "type": entity.get("type", ""),
                    "description": entity.get("description", ""),
                    "confidence": entity.get("confidence", 0.0)
                    # 移除 source_text 以减小文件大小
                })
            
            # 简化关系信息
            for relationship in sample_data.get("relationships", []):
                lightweight_sample["relationships"].append({
                    "source": relationship.get("source", ""),
                    "target": relationship.get("target", ""),
                    "type": relationship.get("type", ""),
                    "description": relationship.get("description", ""),
                    "strength": relationship.get("strength", 0.0),
                    "keywords": relationship.get("keywords", [])[:5]  # 限制关键词数量
                    # 移除 source_text
                })
            
            lightweight["samples"][sample_id] = lightweight_sample
        
        return lightweight
    
    def _summarize_memory_spaces(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """总结内存空间信息 - 新架构特有 - 修复版本"""
        graph_structure = sample_data.get("graph_structure", {})
        memory_space_analysis = sample_data.get("memory_space_analysis", {})
        
        return {
            "total_memory_spaces": graph_structure.get("total_memory_spaces", 0),
            "total_memory_units": graph_structure.get("total_memory_units", 0),
            "space_types_detected": list(memory_space_analysis.keys()) if memory_space_analysis else ["entities", "extracted_data", "conversations"],  # 基于实际数据或常见模式
            "space_unit_distribution": memory_space_analysis
        }
    
    def _summarize_semantic_graph(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """总结语义图谱信息 - 新架构特有"""
        graph_structure = sample_data.get("graph_structure", {})
        semantic_graph_info = sample_data.get("semantic_graph_info", {})
        
        return {
            "has_networkx_graph": graph_structure.get("networkx_nodes", 0) > 0,
            "has_faiss_index": graph_structure.get("faiss_vectors", 0) > 0,
            "has_persistent_storage": bool(semantic_graph_info),
            "graph_density": graph_structure.get("networkx_edges", 0) / max(1, graph_structure.get("networkx_nodes", 1)),
            "vector_coverage": min(1.0, graph_structure.get("faiss_vectors", 0) / max(1, graph_structure.get("total_memory_units", 1)))
        }
    
    def _generate_statistics_report(self, dataset: Dict[str, Any], summary: Dict[str, Any]) -> str:
        """生成统计报告 - 适配新架构"""
        metadata = dataset["metadata"]
        extraction_summary = metadata["extraction_summary"]
        
        report = f"""
        # LoCoMo 实体关系抽取数据集统计报告
        ## 语义图谱新架构版本

        ## 基本信息
        - 生成时间: {metadata['created_at']}
        - 架构版本: {metadata.get('semantic_graph_version', 'new_architecture')}
        - 源目录: {metadata['source_directory']}
        - 样本总数: {metadata['total_samples']}

        ## 抽取统计
        - 实体总数: {extraction_summary['total_entities']}
        - 关系总数: {extraction_summary['total_relationships']}
        - 关键词总数: {extraction_summary['total_keywords']}
        - 内存单元总数: {extraction_summary.get('total_memory_units', 0)}
        - 内存空间总数: {extraction_summary.get('total_memory_spaces', 0)}
        - 平均每样本实体数: {extraction_summary['avg_entities_per_sample']:.2f}
        - 平均每样本关系数: {extraction_summary['avg_relationships_per_sample']:.2f}
        - 平均每样本内存单元数: {extraction_summary.get('avg_memory_units_per_sample', 0):.2f}

        ## 实体类型分布
        """
        
        entity_types = summary["entity_summary"]["entity_types"]
        for entity_type, count in list(entity_types.items())[:10]:
            report += f"- {entity_type}: {count}\n"
        
        # 添加置信度统计
        confidence_stats = summary["entity_summary"].get("entity_confidence_stats", {})
        if confidence_stats:
            report += f"""
            ## 实体置信度统计
            - 平均置信度: {confidence_stats.get('avg_confidence', 0):.3f}
            - 最小置信度: {confidence_stats.get('min_confidence', 0):.3f}
            - 最大置信度: {confidence_stats.get('max_confidence', 0):.3f}
            """
        
        report += f"""
        ## 关系类型分布
        """
        
        rel_types = summary["relationship_summary"]["relationship_types"]
        for rel_type, count in list(rel_types.items())[:10]:
            report += f"- {rel_type}: {count}\n"
        
        # 添加关系强度统计
        strength_stats = summary["relationship_summary"].get("relationship_strength_stats", {})
        if strength_stats:
            report += f"""
            ## 关系强度统计
            - 平均强度: {strength_stats.get('avg_strength', 0):.3f}
            - 最小强度: {strength_stats.get('min_strength', 0):.3f}
            - 最大强度: {strength_stats.get('max_strength', 0):.3f}
            """
        
        # 添加内存空间分析
        memory_analysis = summary.get("memory_space_analysis", {})
        if memory_analysis:
            report += f"""
            ## 内存空间分析 (新架构特有)
            - 样本间总空间数: {memory_analysis.get('total_spaces_across_samples', 0)}
            - 样本间总单元数: {memory_analysis.get('total_units_across_samples', 0)}
            - 平均每样本空间数: {memory_analysis.get('average_spaces_per_sample', 0):.2f}
            - 平均每样本单元数: {memory_analysis.get('average_units_per_sample', 0):.2f}
            - 平均每空间单元数: {memory_analysis.get('average_units_per_space', 0):.2f}
            """
        
        report += f"""
        ## 跨样本分析
        - 唯一实体数: {summary['entity_summary']['entity_frequency']['total_unique_entities']}
        - 跨样本实体数: {len(summary['entity_summary']['entity_frequency']['cross_sample_entities'])}
        - 唯一关系数: {summary['relationship_summary']['relationship_patterns']['total_unique_relationships']}
        - 跨样本关系数: {len(summary['relationship_summary']['relationship_patterns']['cross_sample_relationships'])}

        ## 样本详情
        """
        
        for sample_id, sample_data in dataset["samples"].items():
            stats = sample_data.get("extraction_statistics", {})
            info = sample_data.get("sample_info", {})
            graph_structure = sample_data.get("graph_structure", {})
            semantic_graph_info = sample_data.get("semantic_graph_info", {})
            
            report += f"""
            ### {sample_id}
            - 说话者: {', '.join(info.get('speakers', []))}
            - 会话数: {info.get('session_count', 0)}
            - 消息数: {info.get('total_messages', 0)}
            - 实体数: {stats.get('total_entities', 0)}
            - 关系数: {stats.get('total_relationships', 0)}
            - 内存单元数: {graph_structure.get('total_memory_units', 0)}
            - 内存空间数: {graph_structure.get('total_memory_spaces', 0)}
            - NetworkX节点数: {graph_structure.get('networkx_nodes', 0)}
            - NetworkX边数: {graph_structure.get('networkx_edges', 0)}
            - FAISS向量数: {graph_structure.get('faiss_vectors', 0)}
            - 语义图谱: {'已保存' if semantic_graph_info else '未保存'}
            - 处理时间: {stats.get('processing_time', 'N/A')}
            """
        
        return report


def main():
    """命令行入口 - 适配新架构"""
    parser = argparse.ArgumentParser(description="LoCoMo 数据集制作器 (新架构版本)")
    parser.add_argument(
        "--results-dir", 
        required=True,
        help="抽取结果目录路径 (如 benchmark/extractor/results)"
    )
    parser.add_argument(
        "--output-dir", 
        default="benchmark/dataset/locomo/extraction",
        help="输出目录路径 (默认: benchmark/dataset/locomo/extraction)"
    )
    parser.add_argument(
        "--output-prefix", 
        default="locomo_extracted",
        help="输出文件前缀 (默认: locomo_extracted)"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建数据集制作器
        maker = LoCoMoDatasetMaker(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        
        # 生成数据集文件
        generated_files = maker.generate_dataset_files(
            output_prefix=args.output_prefix
        )
        
        print("\n🎉 数据集生成成功！")
        print("\n生成的文件:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path}")
        
        print(f"\n✨ 新架构特性:")
        print(f"  - 内存空间管理和统计")
        print(f"  - 语义图谱持久化支持")
        print(f"  - FAISS向量索引追踪")
        print(f"  - 架构兼容性分析")
        
    except Exception as e:
        print(f"❌ 数据集生成失败: {e}")
        raise


if __name__ == "__main__":
    main()
