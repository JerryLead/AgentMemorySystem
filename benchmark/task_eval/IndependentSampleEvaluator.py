# 修改 IndependentSampleEvaluator.py 

import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from benchmark.task_eval.semantic_graph_qa_evaluator import SemanticGraphQAEvaluator
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.task_eval.evaluation import calculate_f1_score,calculate_f1_score_multi, calculate_rouge_score, calculate_bleu_score, exact_match_score


class IndependentSampleEvaluator:
    """
    独立样本评估器 - 简化输出格式，包含多种评分指标
    """
    
    def __init__(self, 
                 llm_client: LLMClient,
                 output_dir: str = "benchmark/task_eval/results"):
        """
        初始化独立样本评估器
        
        Args:
            llm_client: LLM客户端
            output_dir: 输出目录
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def evaluate_single_sample(self, 
                              sample_id: str,
                              raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
                              extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json") -> Dict[str, Any]:
        """独立评估单个样本 - 返回简化结果"""
        self.logger.info(f"🔄 开始独立评估样本: {sample_id}")
        
        try:
            # 步骤1-3: 创建存储、存储数据、提取QA
            storage = ConversationSemanticStorage()
            storage_stats = storage.store_conversation(
                sample_id=sample_id,
                raw_dataset_file=raw_dataset_file,
                extracted_dataset_file=extracted_dataset_file,
                include_raw=True,
                include_extracted=True
            )
            
            qa_test_data = storage.get_qa_test_data([sample_id], raw_dataset_file)
            if not qa_test_data or sample_id not in qa_test_data:
                raise ValueError(f"没有找到 {sample_id} 的QA测试数据")
            
            # 步骤4: 创建评估器并评估
            evaluator = SemanticGraphQAEvaluator(
                semantic_graph=storage.semantic_graph,
                llm_client=self.llm_client,
                output_dir=self.output_dir
            )
            
            # 步骤5: 逐个问题评估并收集详细结果
            detailed_results = []
            all_scores = []
            
            for qa in qa_test_data[sample_id]:
                result = self._evaluate_single_question(qa, evaluator, sample_id)
                detailed_results.append(result)
                all_scores.append(result["scores"])
            
            # 步骤6: 计算汇总统计
            summary_stats = self._calculate_summary_stats(detailed_results, sample_id)
            
            # 清理资源
            del storage
            del evaluator
            
            return {
                "sample_id": sample_id,
                "summary": summary_stats,
                "detailed_results": detailed_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ 评估 {sample_id} 失败: {e}")
            return {
                "sample_id": sample_id,
                "error": str(e),
                "summary": None,
                "detailed_results": []
            }
    
    def _evaluate_single_question(self, 
                            qa: Dict[str, Any], 
                            evaluator: SemanticGraphQAEvaluator, 
                            sample_id: str) -> Dict[str, Any]:
        """评估单个问题并返回完整结果 - 修复数据类型问题"""
        
        # 确定标准答案
        category = qa.get("category", 2)
        if category == 5:  # 对抗性问题
            standard_answer = qa.get("adversarial_answer", "")
        else:
            standard_answer = qa.get("answer", "")
        
        # 确保答案是字符串类型
        standard_answer = str(standard_answer) if standard_answer is not None else ""
        
        # 检索上下文
        context = evaluator.retrieve_context_for_question(
            question=qa["question"],
            category=category,
            conversation_id=sample_id,
            evidence=qa.get("evidence", [])
        )
        
        # 生成答案
        answer_result = evaluator.generate_answer(context)
        prediction = answer_result.get("generated_answer", "")
        
        # 确保预测也是字符串类型
        prediction = str(prediction) if prediction is not None else ""
        
        # 计算多种评分
        scores = self._calculate_all_scores(prediction, standard_answer, category)
        
        # 整理检索内容
        retrieval_info = self._format_retrieval_info(context)
        
        return {
            "question": qa["question"],
            "category": category,
            "category_name": evaluator.category_strategies.get(category, "unknown"),
            "answer": standard_answer,
            "prediction": prediction,
            "scores": scores,
            "retrieval": retrieval_info,
            "evidence": qa.get("evidence", []),
            "evidence_recall": context.get("evidence_recall", 0.0),
            "prompt": answer_result.get("prompt", ""),
            "context_nodes_count": len(context.get("retrieved_nodes", [])),
            "context_relations_count": len(context.get("retrieved_relations", []))
        }

    # 修复 _calculate_all_scores 方法，添加类型检查和转换

    def _calculate_all_scores(self, prediction: str, answer: str, category: int) -> Dict[str, float]:
        """计算所有评分指标 - 修复类型错误"""
        scores = {}
        
        try:
            # 1. 确保输入都是字符串类型
            prediction = str(prediction) if prediction is not None else ""
            answer = str(answer) if answer is not None else ""
            
            # 2. F1分数（词级别）
            scores["f1_score"] = calculate_f1_score(prediction, answer)
            
            # 3. 多答案F1分数（处理逗号分隔的答案）
            scores["f1_multi"] = calculate_f1_score_multi(prediction, answer)
            
            # 4. ROUGE-L分数
            scores["rouge_l"] = calculate_rouge_score(prediction, answer)
            
            # 5. ROUGE-L多答案分数
            scores["rouge_l_multi"] = calculate_bleu_score(prediction, [answer])
            
            # 6. 精确匹配
            scores["exact_match"] = float(exact_match_score(prediction, answer))
            
            # 7. 根据类别选择主要分数
            if category == 5:  # 对抗性问题
                pred_lower = prediction.lower()
                ans_lower = answer.lower()
                
                # 检查是否正确拒绝回答
                rejection_phrases = ['no information available', 'not mentioned', 'not provided', 'unknown']
                pred_is_rejection = any(phrase in pred_lower for phrase in rejection_phrases)
                ans_is_rejection = any(phrase in ans_lower for phrase in rejection_phrases)
                
                if pred_is_rejection and ans_is_rejection:
                    scores["main_score"] = 1.0
                elif not pred_is_rejection and not ans_is_rejection:
                    scores["main_score"] = scores["f1_score"]
                else:
                    scores["main_score"] = 0.0
                    
            elif category == 1:  # 多跳推理
                scores["main_score"] = scores["f1_multi"]
            else:  # 其他类别
                scores["main_score"] = scores["f1_score"]
            
        except Exception as e:
            self.logger.error(f"计算评分失败: {e}")
            self.logger.error(f"预测类型: {type(prediction)}, 值: {prediction}")
            self.logger.error(f"答案类型: {type(answer)}, 值: {answer}")
            
            # 返回默认分数
            scores = {
                "f1_score": 0.0,
                "f1_multi": 0.0,
                "rouge_l": 0.0,
                "rouge_l_multi": 0.0,
                "exact_match": 0.0,
                "main_score": 0.0
            }
        
        return scores
    

    def _format_retrieval_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """格式化检索信息"""
        
        # 节点信息
        nodes_info = []
        for node in context.get("retrieved_nodes", []):
            nodes_info.append({
                "uid": node.get("uid", ""),
                "data_type": node.get("data_type", "unknown"),
                "content": node.get("content", "")[:150],  # 限制长度
                "score": node.get("score", 0.0)
            })
        
        # 关系信息
        relations_info = []
        for rel in context.get("retrieved_relations", []):
            relations_info.append({
                "source": rel.get("source", ""),
                "target": rel.get("target", ""),
                "type": rel.get("type", "UNKNOWN"),
                "properties": rel.get("properties", {})
            })
        
        # 原始上下文内容（截断）
        raw_context = context.get("context_text", "")
        if len(raw_context) > 500:
            raw_context = raw_context[:500] + "...[TRUNCATED]"
        
        return {
            "strategy": context.get("strategy", "unknown"),
            "nodes": nodes_info,
            "relations": relations_info,
            "raw_context": raw_context,
            "nodes_count": len(nodes_info),
            "relations_count": len(relations_info)
        }
    
    def _calculate_summary_stats(self, detailed_results: List[Dict], sample_id: str) -> Dict[str, Any]:
        """计算汇总统计信息"""
        
        if not detailed_results:
            return {"error": "No results to summarize"}
        
        # 按类别分组
        by_category = {}
        all_scores = []
        
        for result in detailed_results:
            category = result["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
            all_scores.append(result["scores"])
        
        # 总体统计
        overall_stats = {
            "total_questions": len(detailed_results),
            "avg_f1_score": float(np.mean([s["f1_score"] for s in all_scores])),
            "avg_f1_multi": float(np.mean([s["f1_multi"] for s in all_scores])),
            "avg_rouge_l": float(np.mean([s["rouge_l"] for s in all_scores])),
            "avg_exact_match": float(np.mean([s["exact_match"] for s in all_scores])),
            "avg_main_score": float(np.mean([s["main_score"] for s in all_scores])),
            "avg_evidence_recall": float(np.mean([r["evidence_recall"] for r in detailed_results])),
            "avg_context_nodes": float(np.mean([r["context_nodes_count"] for r in detailed_results])),
            "avg_context_relations": float(np.mean([r["context_relations_count"] for r in detailed_results]))
        }
        
        # 按类别统计
        category_stats = {}
        for category, results in by_category.items():
            cat_scores = [r["scores"] for r in results]
            category_name = results[0]["category_name"]
            
            category_stats[f"category_{category}"] = {
                "category_name": category_name,
                "questions_count": len(results),
                "avg_f1_score": float(np.mean([s["f1_score"] for s in cat_scores])),
                "avg_f1_multi": float(np.mean([s["f1_multi"] for s in cat_scores])),
                "avg_rouge_l": float(np.mean([s["rouge_l"] for s in cat_scores])),
                "avg_exact_match": float(np.mean([s["exact_match"] for s in cat_scores])),
                "avg_main_score": float(np.mean([s["main_score"] for s in cat_scores])),
                "std_main_score": float(np.std([s["main_score"] for s in cat_scores]))
            }
        
        return {
            "sample_id": sample_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "llm_model": self.llm_client.model_name,
            "overall": overall_stats,
            "by_category": category_stats
        }
    
    def save_sample_results(self, sample_result: Dict[str, Any], output_format: str = "comprehensive") -> str:
        """保存样本结果到文件"""
        
        sample_id = sample_result["sample_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "comprehensive":
            # 完整格式：包含所有信息
            output_file = self.output_dir / f"evaluation_{sample_id}_{timestamp}.json"
            
            # 构建输出结构
            output_data = {
                "evaluation_summary": sample_result.get("summary", {}),
                "questions": sample_result.get("detailed_results", [])
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                
        elif output_format == "simple":
            # 简化格式：只包含核心信息
            output_file = self.output_dir / f"simple_{sample_id}_{timestamp}.json"
            
            simple_results = []
            for result in sample_result.get("detailed_results", []):
                simple_results.append({
                    "question": result["question"],
                    "category": result["category"],
                    "answer": result["answer"],
                    "prediction": result["prediction"],
                    "score": result["scores"]["main_score"],
                    "retrieve": {
                        "nodes_count": result["retrieval"]["nodes_count"],
                        "relations_count": result["retrieval"]["relations_count"],
                        "strategy": result["retrieval"]["strategy"]
                    },
                    "evidence": result["evidence"]
                })
            
            output_data = {
                "summary": sample_result.get("summary", {}),
                "results": simple_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 {sample_id} 结果已保存: {output_file}")
        return str(output_file)
    
    def evaluate_multiple_samples(self, 
                                 sample_ids: List[str],
                                 raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
                                 extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json",
                                 save_format: str = "comprehensive") -> Dict[str, Any]:
        """评估多个样本"""
        
        self.logger.info(f"🚀 开始独立评估 {len(sample_ids)} 个样本")
        
        all_results = []
        failed_samples = []
        
        for sample_id in sample_ids:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"评估样本 {sample_id} ({sample_ids.index(sample_id)+1}/{len(sample_ids)})")
            self.logger.info(f"{'='*50}")
            
            # 评估单个样本
            sample_result = self.evaluate_single_sample(
                sample_id=sample_id,
                raw_dataset_file=raw_dataset_file,
                extracted_dataset_file=extracted_dataset_file
            )
            
            if "error" not in sample_result:
                all_results.append(sample_result)
                
                # 保存单个样本结果
                self.save_sample_results(sample_result, save_format)
                
                # 显示简要结果
                if sample_result.get("summary"):
                    summary = sample_result["summary"]["overall"]
                    self.logger.info(f"✅ {sample_id} 完成:")
                    self.logger.info(f"   问题数: {summary['total_questions']}")
                    self.logger.info(f"   主要分数: {summary['avg_main_score']:.4f}")
                    self.logger.info(f"   F1分数: {summary['avg_f1_score']:.4f}")
                    self.logger.info(f"   精确匹配: {summary['avg_exact_match']:.4f}")
                    
            else:
                failed_samples.append(sample_id)
                self.logger.error(f"❌ {sample_id} 评估失败")
        
        # 计算跨样本汇总
        if all_results:
            aggregated_summary = self._calculate_cross_sample_summary(all_results)
            
            # 保存汇总结果
            self._save_aggregated_results(aggregated_summary, sample_ids, failed_samples, save_format)
            
            return {
                "aggregated_summary": aggregated_summary,
                "individual_results": all_results,
                "failed_samples": failed_samples
            }
        else:
            self.logger.error("❌ 没有成功的评估结果")
            return {"error": "No successful evaluations", "failed_samples": failed_samples}
    
    def _calculate_cross_sample_summary(self, all_results: List[Dict]) -> Dict[str, Any]:
        """计算跨样本汇总统计"""
        
        # 收集所有样本的总体指标
        overall_metrics = []
        category_metrics = {i: [] for i in range(1, 6)}
        
        for sample_result in all_results:
            if sample_result.get("summary") and sample_result["summary"].get("overall"):
                overall_metrics.append(sample_result["summary"]["overall"])
                
                # 按类别收集
                by_category = sample_result["summary"].get("by_category", {})
                for cat_key, cat_data in by_category.items():
                    category_num = int(cat_key.split("_")[1])
                    category_metrics[category_num].append(cat_data["avg_main_score"])
        
        # 计算跨样本统计
        cross_sample_stats = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "llm_model": self.llm_client.model_name,
                "total_samples": len(all_results),
                "evaluation_mode": "independent_samples"
            },
            "cross_sample_overall": {},
            "cross_sample_by_category": {}
        }
        
        # 总体跨样本统计
        if overall_metrics:
            cross_sample_stats["cross_sample_overall"] = {
                "avg_f1_score": float(np.mean([m["avg_f1_score"] for m in overall_metrics])),
                "std_f1_score": float(np.std([m["avg_f1_score"] for m in overall_metrics])),
                "avg_f1_multi": float(np.mean([m["avg_f1_multi"] for m in overall_metrics])),
                "std_f1_multi": float(np.std([m["avg_f1_multi"] for m in overall_metrics])),
                "avg_rouge_l": float(np.mean([m["avg_rouge_l"] for m in overall_metrics])),
                "std_rouge_l": float(np.std([m["avg_rouge_l"] for m in overall_metrics])),
                "avg_exact_match": float(np.mean([m["avg_exact_match"] for m in overall_metrics])),
                "std_exact_match": float(np.std([m["avg_exact_match"] for m in overall_metrics])),
                "avg_main_score": float(np.mean([m["avg_main_score"] for m in overall_metrics])),
                "std_main_score": float(np.std([m["avg_main_score"] for m in overall_metrics])),
                "total_questions": sum([m["total_questions"] for m in overall_metrics])
            }
        
        # 按类别跨样本统计
        strategy_names = {1: "multi_hop", 2: "single_hop", 3: "temporal", 4: "open_domain", 5: "adversarial"}
        for category, scores in category_metrics.items():
            if scores:
                cross_sample_stats["cross_sample_by_category"][f"category_{category}"] = {
                    "category_name": strategy_names.get(category, "unknown"),
                    "samples_count": len(scores),
                    "avg_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "min_score": float(np.min(scores)),
                    "max_score": float(np.max(scores))
                }
        
        return cross_sample_stats
    
    def _save_aggregated_results(self, 
                               aggregated_summary: Dict[str, Any],
                               sample_ids: List[str],
                               failed_samples: List[str],
                               save_format: str):
        """保存跨样本汇总结果"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON格式汇总
        summary_file = self.output_dir / f"cross_sample_summary_{timestamp}.json"
        
        summary_data = {
            **aggregated_summary,
            "sample_info": {
                "evaluated_samples": sample_ids,
                "failed_samples": failed_samples,
                "success_rate": len(sample_ids) - len(failed_samples) / len(sample_ids) if sample_ids else 0
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 2. 可读性报告
        report_file = self.output_dir / f"evaluation_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Independent Sample QA Evaluation Report ===\n\n")
            
            # 基本信息
            info = aggregated_summary["evaluation_info"]
            f.write(f"Timestamp: {info['timestamp']}\n")
            f.write(f"LLM Model: {info['llm_model']}\n")
            f.write(f"Total Samples: {info['total_samples']}\n")
            f.write(f"Failed Samples: {len(failed_samples)}\n")
            f.write(f"Sample IDs: {', '.join(sample_ids)}\n\n")
            
            # 总体结果
            if "cross_sample_overall" in aggregated_summary:
                overall = aggregated_summary["cross_sample_overall"]
                f.write("=== Cross-Sample Overall Results ===\n")
                f.write(f"Average F1 Score: {overall['avg_f1_score']:.4f} ± {overall['std_f1_score']:.4f}\n")
                f.write(f"Average F1 Multi: {overall['avg_f1_multi']:.4f} ± {overall['std_f1_multi']:.4f}\n")
                f.write(f"Average ROUGE-L: {overall['avg_rouge_l']:.4f} ± {overall['std_rouge_l']:.4f}\n")
                f.write(f"Average Exact Match: {overall['avg_exact_match']:.4f} ± {overall['std_exact_match']:.4f}\n")
                f.write(f"Average Main Score: {overall['avg_main_score']:.4f} ± {overall['std_main_score']:.4f}\n")
                f.write(f"Total Questions Evaluated: {overall['total_questions']}\n\n")
            
            # 按类别结果
            if "cross_sample_by_category" in aggregated_summary:
                f.write("=== Results by Category ===\n")
                for cat_key, cat_data in aggregated_summary["cross_sample_by_category"].items():
                    category_num = cat_key.split("_")[1]
                    f.write(f"Category {category_num} ({cat_data['category_name']}):\n")
                    f.write(f"  Samples: {cat_data['samples_count']}\n")
                    f.write(f"  Average: {cat_data['avg_score']:.4f} ± {cat_data['std_score']:.4f}\n")
                    f.write(f"  Range: [{cat_data['min_score']:.4f}, {cat_data['max_score']:.4f}]\n\n")
        
        self.logger.info(f"📊 跨样本汇总结果已保存:")
        self.logger.info(f"  - JSON: {summary_file}")
        self.logger.info(f"  - 报告: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Independent Sample QA Evaluation - Simplified Output")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model name")
    parser.add_argument("--samples", nargs="+", default=["conv-26", "conv-30"], 
                       help="Sample IDs to evaluate")
    parser.add_argument("--output_dir", default="benchmark/task_eval/results/simplified", 
                       help="Output directory")
    parser.add_argument("--format", choices=["comprehensive", "simple"], default="comprehensive",
                       help="Output format")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化LLM客户端
    llm_client = LLMClient(model_name=args.model)
    
    # 创建独立评估器
    evaluator = IndependentSampleEvaluator(
        llm_client=llm_client,
        output_dir=args.output_dir
    )
    
    # 执行评估
    results = evaluator.evaluate_multiple_samples(
        sample_ids=args.samples,
        save_format=args.format
    )
    
    if "aggregated_summary" in results:
        summary = results["aggregated_summary"]["cross_sample_overall"]
        print(f"\n🎉 评估完成！")
        print(f"📊 成功评估: {len(results['individual_results'])}/{len(args.samples)} 个样本")
        print(f"📈 跨样本平均分数: {summary['avg_main_score']:.4f} ± {summary['std_main_score']:.4f}")
        print(f"📈 跨样本F1分数: {summary['avg_f1_score']:.4f} ± {summary['std_f1_score']:.4f}")
        print(f"📈 跨样本精确匹配: {summary['avg_exact_match']:.4f} ± {summary['std_exact_match']:.4f}")


if __name__ == "__main__":
    main()