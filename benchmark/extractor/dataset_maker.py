import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

class LoCoMoDatasetMaker:
    """
    LoCoMo 数据集制作器
    将所有样本的实体关系抽取结果整合成统一的数据集文件
    """
    
    def __init__(self, results_dir: str, output_dir: str = None):
        """
        初始化数据集制作器
        
        Args:
            results_dir: 抽取结果目录路径 (如 benchmark/results/fixed_full_extraction)
            output_dir: 输出目录路径 (默认为 benchmark/dataset)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark/dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_extraction_results(self) -> Dict[str, Any]:
        """
        收集所有样本的抽取结果
        
        Returns:
            包含所有样本数据的字典
        """
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source_directory": str(self.results_dir),
                "total_samples": 0,
                "extraction_summary": {}
            },
            "samples": {}
        }
        
        # 查找所有样本的抽取结果目录
        sample_dirs = [d for d in self.results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('conv-')]
        
        self.logger.info(f"发现 {len(sample_dirs)} 个样本目录")
        
        total_entities = 0
        total_relationships = 0
        total_keywords = 0
        
        for sample_dir in sorted(sample_dirs):
            try:
                sample_data = self._process_sample_directory(sample_dir)
                if sample_data:
                    sample_id = sample_data["sample_info"]["sample_id"]
                    dataset["samples"][sample_id] = sample_data
                    
                    # 累计统计
                    stats = sample_data.get("extraction_statistics", {})
                    total_entities += stats.get("total_entities", 0)
                    total_relationships += stats.get("total_relationships", 0)
                    total_keywords += stats.get("total_keywords", 0)
                    
                    self.logger.info(f"✅ 成功处理样本: {sample_id}")
                else:
                    self.logger.warning(f"⚠️  跳过无效样本目录: {sample_dir.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ 处理样本目录 {sample_dir.name} 时出错: {e}")
        
        # 更新元数据
        dataset["metadata"]["total_samples"] = len(dataset["samples"])
        dataset["metadata"]["extraction_summary"] = {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "total_keywords": total_keywords,
            "avg_entities_per_sample": total_entities / len(dataset["samples"]) if dataset["samples"] else 0,
            "avg_relationships_per_sample": total_relationships / len(dataset["samples"]) if dataset["samples"] else 0
        }
        
        return dataset
    
    def _process_sample_directory(self, sample_dir: Path) -> Optional[Dict[str, Any]]:
        """
        处理单个样本目录
        
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
            
            # 查找其他相关文件
            additional_files = {}
            for pattern in ["*_full_conversation.txt", "*_samples.json", "*_analysis.json", 
                           "*_summary_report.txt", "*semantic_graph*"]:
                files = list(sample_dir.glob(pattern))
                if files:
                    additional_files[pattern.replace("*", "").replace(".", "")] = [
                        str(f.relative_to(self.results_dir)) for f in files
                    ]
            
            if additional_files:
                sample_data["source_files"].update(additional_files)
            
            return sample_data
            
        except Exception as e:
            self.logger.error(f"读取结果文件 {result_file} 时出错: {e}")
            return None
    
    def create_entity_relationship_summary(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建实体关系汇总
        
        Args:
            dataset: 完整数据集
            
        Returns:
            实体关系汇总数据
        """
        summary = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_samples": len(dataset["samples"])
            },
            "entity_summary": {
                "all_entities": {},
                "entity_types": {},
                "entity_frequency": {}
            },
            "relationship_summary": {
                "all_relationships": {},
                "relationship_types": {},
                "relationship_patterns": {}
            },
            "cross_sample_analysis": {
                "common_entities": {},
                "common_relationships": {},
                "entity_co_occurrence": {}
            }
        }
        
        all_entities = {}
        all_relationships = {}
        entity_types_count = {}
        relationship_types_count = {}
        
        # 遍历所有样本
        for sample_id, sample_data in dataset["samples"].items():
            entities = sample_data.get("entities", [])
            relationships = sample_data.get("relationships", [])
            
            # 处理实体
            for entity in entities:
                entity_name = entity.get("name", "").lower()
                entity_type = entity.get("type", "unknown")
                
                if entity_name:
                    # 实体频率统计
                    if entity_name not in all_entities:
                        all_entities[entity_name] = {
                            "name": entity.get("name", ""),
                            "type": entity_type,
                            "descriptions": [],
                            "samples": [],
                            "frequency": 0
                        }
                    
                    all_entities[entity_name]["descriptions"].append({
                        "sample_id": sample_id,
                        "description": entity.get("description", ""),
                        "confidence": entity.get("confidence", 0.0)
                    })
                    all_entities[entity_name]["samples"].append(sample_id)
                    all_entities[entity_name]["frequency"] += 1
                
                # 实体类型统计
                entity_types_count[entity_type] = entity_types_count.get(entity_type, 0) + 1
            
            # 处理关系
            for relationship in relationships:
                source = relationship.get("source", "").lower()
                target = relationship.get("target", "").lower()
                rel_type = relationship.get("type", "unknown")
                
                rel_key = f"{source}-{rel_type}-{target}"
                
                if rel_key not in all_relationships:
                    all_relationships[rel_key] = {
                        "source": relationship.get("source", ""),
                        "target": relationship.get("target", ""),
                        "type": rel_type,
                        "descriptions": [],
                        "samples": [],
                        "frequency": 0,
                        "avg_strength": 0.0
                    }
                
                all_relationships[rel_key]["descriptions"].append({
                    "sample_id": sample_id,
                    "description": relationship.get("description", ""),
                    "strength": relationship.get("strength", 0.0)
                })
                all_relationships[rel_key]["samples"].append(sample_id)
                all_relationships[rel_key]["frequency"] += 1
                
                # 关系类型统计
                relationship_types_count[rel_type] = relationship_types_count.get(rel_type, 0) + 1
        
        # 计算平均强度
        for rel_data in all_relationships.values():
            strengths = [desc["strength"] for desc in rel_data["descriptions"] if desc["strength"]]
            rel_data["avg_strength"] = sum(strengths) / len(strengths) if strengths else 0.0
        
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
        
        return summary
    
    def create_sample_index(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建样本索引文件
        
        Args:
            dataset: 完整数据集
            
        Returns:
            样本索引数据
        """
        index = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_samples": len(dataset["samples"])
            },
            "sample_index": {}
        }
        
        for sample_id, sample_data in dataset["samples"].items():
            sample_info = sample_data.get("sample_info", {})
            extraction_stats = sample_data.get("extraction_statistics", {})
            
            index["sample_index"][sample_id] = {
                "speakers": sample_info.get("speakers", []),
                "session_count": sample_info.get("session_count", 0),
                "total_messages": sample_info.get("total_messages", 0),
                "entity_count": extraction_stats.get("total_entities", 0),
                "relationship_count": extraction_stats.get("total_relationships", 0),
                "keyword_count": extraction_stats.get("total_keywords", 0),
                "processing_time": extraction_stats.get("processing_time", ""),
                "source_directory": sample_data.get("source_files", {}).get("directory", "")
            }
        
        return index
    
    def generate_dataset_files(self, output_prefix: str = "locomo_extracted") -> Dict[str, Path]:
        """
        生成完整的数据集文件
        
        Args:
            output_prefix: 输出文件前缀
            
        Returns:
            生成的文件路径字典
        """
        self.logger.info("🚀 开始生成 LoCoMo 抽取数据集...")
        
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
        
        # 6. 生成统计报告
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
            "statistics_report": report_path
        }
        
        self.logger.info(f"🎉 数据集生成完成！共生成 {len(generated_files)} 个文件")
        return generated_files
    
    def _create_lightweight_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """创建轻量级数据集（移除source_text等大字段）"""
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
                "graph_structure": sample_data.get("graph_structure", {})
            }
            
            # 简化实体信息
            for entity in sample_data.get("entities", []):
                lightweight_sample["entities"].append({
                    "name": entity.get("name", ""),
                    "type": entity.get("type", ""),
                    "description": entity.get("description", ""),
                    "confidence": entity.get("confidence", 0.0)
                })
            
            # 简化关系信息
            for relationship in sample_data.get("relationships", []):
                lightweight_sample["relationships"].append({
                    "source": relationship.get("source", ""),
                    "target": relationship.get("target", ""),
                    "type": relationship.get("type", ""),
                    "description": relationship.get("description", ""),
                    "strength": relationship.get("strength", 0.0)
                })
            
            lightweight["samples"][sample_id] = lightweight_sample
        
        return lightweight
    
    def _generate_statistics_report(self, dataset: Dict[str, Any], summary: Dict[str, Any]) -> str:
        """生成统计报告"""
        metadata = dataset["metadata"]
        extraction_summary = metadata["extraction_summary"]
        
        report = f"""
        # LoCoMo 实体关系抽取数据集统计报告

        ## 基本信息
        - 生成时间: {metadata['created_at']}
        - 源目录: {metadata['source_directory']}
        - 样本总数: {metadata['total_samples']}

        ## 抽取统计
        - 实体总数: {extraction_summary['total_entities']}
        - 关系总数: {extraction_summary['total_relationships']}
        - 关键词总数: {extraction_summary['total_keywords']}
        - 平均每样本实体数: {extraction_summary['avg_entities_per_sample']:.2f}
        - 平均每样本关系数: {extraction_summary['avg_relationships_per_sample']:.2f}

        ## 实体类型分布
        """
        
        entity_types = summary["entity_summary"]["entity_types"]
        for entity_type, count in list(entity_types.items())[:10]:
            report += f"- {entity_type}: {count}\n"
        
        report += f"""
        ## 关系类型分布
        """
        
        rel_types = summary["relationship_summary"]["relationship_types"]
        for rel_type, count in list(rel_types.items())[:10]:
            report += f"- {rel_type}: {count}\n"
        
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
            report += f"""
            ### {sample_id}
            - 说话者: {', '.join(info.get('speakers', []))}
            - 会话数: {info.get('session_count', 0)}
            - 消息数: {info.get('total_messages', 0)}
            - 实体数: {stats.get('total_entities', 0)}
            - 关系数: {stats.get('total_relationships', 0)}
            - 处理时间: {stats.get('processing_time', 'N/A')}
            """
        
        return report


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="LoCoMo 数据集制作器")
    parser.add_argument(
        "--results-dir", 
        required=True,
        help="抽取结果目录路径 (如 benchmark/results/fixed_full_extraction)"
    )
    parser.add_argument(
        "--output-dir", 
        default="benchmark/dataset",
        help="输出目录路径 (默认: benchmark/dataset)"
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
        
    except Exception as e:
        print(f"❌ 数据集生成失败: {e}")
        raise


if __name__ == "__main__":
    main()