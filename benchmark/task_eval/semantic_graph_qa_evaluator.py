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

from dev.semantic_graph import SemanticGraph
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.task_eval.evaluation import eval_question_answering, calculate_f1_score


class SemanticGraphQAEvaluator:
    """
    基于语义图谱的QA评估器
    结合信息检索、提示词工程和答案生成的完整评估流程
    """
    
    def __init__(self, 
                 semantic_graph: SemanticGraph,
                 llm_client: LLMClient,
                 output_dir: str = "benchmark/task_eval/results"):
        """
        初始化评估器
        
        Args:
            semantic_graph: 已构建好的语义图谱
            llm_client: LLM客户端
            output_dir: 输出目录
        """
        self.semantic_graph = semantic_graph
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # QA类别对应的检索策略
        self.category_strategies = {
            1: "multi_hop",      # 多跳推理
            2: "single_hop",     # 单跳事实
            3: "temporal",       # 时间相关
            4: "open_domain",    # 开放域
            5: "adversarial"     # 对抗性
        }
        
        # 检索配置
        self.retrieval_config = {
            1: {"top_k": 8, "include_relations": True, "expand_neighbors": True},
            2: {"top_k": 5, "include_relations": False, "expand_neighbors": False},
            3: {"top_k": 6, "include_relations": True, "expand_neighbors": False},
            4: {"top_k": 10, "include_relations": True, "expand_neighbors": True},
            5: {"top_k": 3, "include_relations": False, "expand_neighbors": False}
        }

    def retrieve_context_for_question(self, 
                                     question: str, 
                                     category: int, 
                                     conversation_id: str,
                                     evidence: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        根据问题类别检索相关上下文
        
        Args:
            question: 问题文本
            category: QA类别 (1-5)
            conversation_id: 对话ID
            evidence: 证据列表（用于评估召回率）
            
        Returns:
            检索到的上下文信息
        """
        config = self.retrieval_config.get(category, self.retrieval_config[2])
        strategy = self.category_strategies.get(category, "single_hop")
        
        context = {
            "question": question,
            "category": category,
            "strategy": strategy,
            "conversation_id": conversation_id,
            "retrieved_nodes": [],
            "retrieved_relations": [],
            "context_text": "",
            "evidence_recall": 0.0
        }
        
        try:
            # 1. 基础语义检索
            search_results = self.semantic_graph.search_similarity_in_graph(
                query_text=question,
                k=config["top_k"],
                ms_names=[f"conversation_{conversation_id}"],
                recursive=True
            )
            
            context["retrieved_nodes"] = [
                {
                    "uid": unit.uid,
                    "score": float(score),
                    "content": unit.raw_data.get("text_content", "")[:200],
                    "data_type": unit.raw_data.get("data_type", "unknown"),
                    "data_source": unit.raw_data.get("data_source", "unknown")
                }
                for unit, score in search_results
            ]
            
            # 2. 根据类别进行特殊处理
            if strategy == "multi_hop" and config["expand_neighbors"]:
                # 多跳：扩展邻居节点
                expanded_nodes = self._expand_with_neighbors(search_results[:3], conversation_id)
                context["retrieved_nodes"].extend(expanded_nodes)
            
            elif strategy == "temporal":
                # 时间相关：优先检索时间相关信息
                temporal_nodes = self._retrieve_temporal_context(question, conversation_id)
                context["retrieved_nodes"].extend(temporal_nodes)
            
            # 3. 检索关系信息
            if config["include_relations"]:
                context["retrieved_relations"] = self._retrieve_relations(
                    [node["uid"] for node in context["retrieved_nodes"][:5]]
                )
            
            # 4. 构建上下文文本
            context["context_text"] = self._build_context_text(
                context["retrieved_nodes"], 
                context["retrieved_relations"],
                strategy
            )
            
            # 5. 计算证据召回率
            if evidence:
                context["evidence_recall"] = self._calculate_evidence_recall(
                    context["retrieved_nodes"], evidence
                )
                
        except Exception as e:
            self.logger.error(f"检索上下文失败: {e}")
            context["error"] = str(e)
        
        return context

    def _expand_with_neighbors(self, 
                              top_results: List[Tuple], 
                              conversation_id: str) -> List[Dict]:
        """扩展邻居节点（用于多跳推理）"""
        expanded_nodes = []
        
        for unit, score in top_results:
            # 获取显式邻居
            explicit_neighbors = self.semantic_graph.traverse_explicit_nodes(
                uid=unit.uid,
                space_name=f"conversation_{conversation_id}"
            )
            
            for neighbor in explicit_neighbors[:2]:  # 限制邻居数量
                expanded_nodes.append({
                    "uid": neighbor.uid,
                    "score": float(score * 0.8),  # 降低邻居分数
                    "content": neighbor.raw_data.get("text_content", "")[:200],
                    "data_type": neighbor.raw_data.get("data_type", "unknown"),
                    "data_source": neighbor.raw_data.get("data_source", "unknown"),
                    "relation_type": "neighbor"
                })
        
        return expanded_nodes

    def _retrieve_temporal_context(self, question: str, conversation_id: str) -> List[Dict]:
        """检索时间相关上下文"""
        temporal_keywords = ["when", "date", "time", "day", "week", "month", "year", "ago", "recently", "yesterday", "today"]
        
        # 检查问题是否包含时间关键词
        if not any(keyword in question.lower() for keyword in temporal_keywords):
            return []
        
        # 优先检索包含时间信息的节点
        time_results = self.semantic_graph.semantic_map.filter_memory_units(
            ms_names=[f"conversation_{conversation_id}"],
            filter_condition={
                "data_type": {"in": ["summary", "event", "observation"]}
            }
        )
        
        temporal_nodes = []
        for unit in time_results[:3]:
            temporal_nodes.append({
                "uid": unit.uid,
                "score": 0.9,
                "content": unit.raw_data.get("text_content", "")[:200],
                "data_type": unit.raw_data.get("data_type", "unknown"),
                "data_source": unit.raw_data.get("data_source", "unknown"),
                "relation_type": "temporal"
            })
        
        return temporal_nodes

    def _retrieve_relations(self, node_uids: List[str]) -> List[Dict]:
        """检索节点间的关系"""
        relations = []
        
        for i, source_uid in enumerate(node_uids):
            for target_uid in node_uids[i+1:]:
                # 检查是否存在显式关系
                if self.semantic_graph.nx_graph.has_edge(source_uid, target_uid):
                    edge_data = self.semantic_graph.nx_graph.get_edge_data(source_uid, target_uid)
                    relations.append({
                        "source": source_uid,
                        "target": target_uid,
                        "type": edge_data.get("type", "RELATED_TO"),
                        "properties": {k: v for k, v in edge_data.items() if k != "type"}
                    })
        
        return relations

    def _build_context_text(self, 
                           nodes: List[Dict], 
                           relations: List[Dict], 
                           strategy: str) -> str:
        """构建上下文文本"""
        context_parts = []
        
        # 添加节点信息
        if strategy == "multi_hop":
            context_parts.append("=== CONVERSATION CONTEXT (Multi-hop Analysis) ===")
        elif strategy == "temporal":
            context_parts.append("=== CONVERSATION CONTEXT (Temporal Information) ===")
        else:
            context_parts.append("=== CONVERSATION CONTEXT ===")
        
        for i, node in enumerate(nodes[:8]):  # 限制节点数量
            data_type = node.get("data_type", "content")
            content = node.get("content", "")
            
            if content.strip():
                context_parts.append(f"[{i+1}] {data_type.upper()}: {content}")
        
        # 添加关系信息
        if relations:
            context_parts.append("\n=== RELATIONSHIPS ===")
            for rel in relations[:5]:  # 限制关系数量
                context_parts.append(
                    f"- {rel['source']} --[{rel['type']}]--> {rel['target']}"
                )
        
        return "\n".join(context_parts)

    def _calculate_evidence_recall(self, retrieved_nodes: List[Dict], evidence: List[str]) -> float:
        """计算证据召回率"""
        if not evidence:
            return 1.0
        
        retrieved_content = " ".join([node.get("content", "") for node in retrieved_nodes])
        
        found_evidence = 0
        for ev in evidence:
            # 简单的字符串匹配检查
            if any(keyword in retrieved_content.lower() for keyword in ev.lower().split()):
                found_evidence += 1
        
        return found_evidence / len(evidence) if evidence else 0.0
    
    # 在 semantic_graph_qa_evaluator.py 中修复 generate_answer 方法

    def generate_answer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于检索到的上下文生成答案 - 使用增强版LLMClient"""
        question = context["question"]
        category = context["category"]
        context_text = context["context_text"]
        
        # 根据类别选择提示词模板
        prompt = self._build_prompt(question, context_text, category)
        
        try:
            # 使用增强版LLMClient，支持智能上下文管理
            if category == 5:  # 对抗性问题
                answer = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1, 
                    max_tokens=100,
                    json_response=False,
                    context_text=context_text,  # 提供分离的上下文
                    question=question           # 提供问题用于智能截断
                )
            else:
                answer = self.llm_client.generate_answer(
                    prompt=prompt,
                    temperature=0.1, 
                    max_tokens=150,
                    context_text=context_text,
                    question=question
                )
            
            return {
                "question": question,
                "category": category,
                "generated_answer": answer,  # 确保这里是 answer，不是其他键名
                "prompt": prompt,
                "context_nodes_count": len(context["retrieved_nodes"]),
                "context_relations_count": len(context["retrieved_relations"]),
                "evidence_recall": context.get("evidence_recall", 0.0),
                "generation_strategy": context["strategy"],
                "llm_context_info": self.llm_client.get_context_info()  # 添加LLM配置信息
            }
            
        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            return {
                "question": question,
                "category": category,
                "generated_answer": f"生成失败: {str(e)}",  # 确保错误情况下也有这个键
                "error": str(e)
            }

    def _build_prompt(self, question: str, context_text: str, category: int) -> str:
        """构建针对不同类别的提示词"""
        
        base_instruction = f"""Based on the following conversation context, answer the question accurately and concisely.

        {context_text}

        """
        
        if category == 1:  # 多跳推理
            instruction = base_instruction + """
            This question requires multi-step reasoning. Consider the relationships between different pieces of information in the context.

            Question: {question}

            Provide a concise answer based on the connected information in the context:"""
            
        elif category == 2:  # 单跳事实
            instruction = base_instruction + """
            This question asks for a specific fact. Look for direct information in the context.

            Question: {question}

            Provide a short, factual answer:"""
            
        elif category == 3:  # 时间相关
            instruction = base_instruction + """
            This question is about timing or dates. Pay attention to temporal information and dates mentioned in the context.

            Question: {question}

            Provide a specific date or time-related answer:"""
            
        elif category == 4:  # 开放域
            instruction = base_instruction + """
            This is an open-domain question. Use all available information in the context to provide a comprehensive answer.

            Question: {question}

            Provide a detailed answer based on the context:"""
            
        elif category == 5:  # 对抗性
            instruction = base_instruction + """
            This question tests if the information is actually mentioned in the conversation. Be very careful to only answer based on what is explicitly stated.

            Question: {question}

            If the answer is clearly stated in the context, provide it. If the information is not mentioned or unclear, respond with "No information available":"""
        
        else:
            instruction = base_instruction + f"""
            Question: {question}

            Answer:"""
        
        return instruction.format(question=question)
    
    def evaluate_single_conversation(self, 
                                conversation_id: str, 
                                qa_data: List[Dict]) -> Dict[str, Any]:
        """评估单个对话的所有QA - 修复数据格式问题"""
        results = {
            "conversation_id": conversation_id,
            "total_questions": len(qa_data),
            "results_by_category": {},
            "overall_metrics": {},
            "detailed_results": []
        }
        
        # 按类别分组处理
        qa_by_category = {}
        for qa in qa_data:
            category = qa.get("category", 2)
            if category not in qa_by_category:
                qa_by_category[category] = []
            qa_by_category[category].append(qa)
        
        all_predictions = []
        
        for category, qa_list in qa_by_category.items():
            self.logger.info(f"处理类别 {category}: {len(qa_list)} 个问题")
            
            category_results = {
                "category": category,
                "strategy": self.category_strategies.get(category, "single_hop"),
                "questions_count": len(qa_list),
                "predictions": [],
                "metrics": {}
            }
            
            for qa in tqdm(qa_list, desc=f"Category {category}"):
                try:
                    # 修复对抗性问题的数据格式
                    if category == 5:  # 对抗性问题
                        # 对抗性问题使用 adversarial_answer 作为标准答案
                        if "adversarial_answer" not in qa or "question" not in qa:
                            self.logger.warning(f"对抗性QA数据不完整，跳过: {qa}")
                            continue
                        standard_answer = qa["adversarial_answer"]
                    else:
                        # 其他类别使用 answer 字段
                        if not all(key in qa for key in ["question", "answer"]):
                            self.logger.warning(f"QA数据不完整，跳过: {qa}")
                            continue
                        standard_answer = qa["answer"]
                    
                    # 检索上下文
                    context = self.retrieve_context_for_question(
                        question=qa["question"],
                        category=category,
                        conversation_id=conversation_id,
                        evidence=qa.get("evidence", [])
                    )
                    
                    # 生成答案
                    answer_result = self.generate_answer(context)
                    
                    # 确保answer_result包含必要的键
                    if "generated_answer" not in answer_result:
                        self.logger.error(f"答案生成结果缺少 'generated_answer' 键: {answer_result}")
                        continue
                    
                    # 准备评估数据
                    prediction_data = {
                        "question": qa["question"],
                        "category": category,
                        "answer": standard_answer,  # 使用正确的标准答案字段
                        "prediction": answer_result["generated_answer"],
                        "evidence": qa.get("evidence", []),
                        "context_nodes_count": answer_result.get("context_nodes_count", 0),
                        "evidence_recall": answer_result.get("evidence_recall", 0.0)
                    }
                    
                    category_results["predictions"].append(prediction_data)
                    all_predictions.append(prediction_data)
                    
                    # 详细结果
                    results["detailed_results"].append({
                        **prediction_data,
                        "retrieval_context": context,
                        "generation_details": answer_result
                    })
                    
                except KeyError as e:
                    self.logger.error(f"处理QA时缺少必要键 {e}: {qa}")
                    continue
                except Exception as e:
                    self.logger.error(f"处理QA时发生错误: {e}, qa: {qa}")
                    continue
            
            # 计算类别指标
            if category_results["predictions"]:
                category_metrics = self._evaluate_predictions(category_results["predictions"])
                category_results["metrics"] = category_metrics
                results["results_by_category"][category] = category_results
        
        # 计算总体指标
        if all_predictions:
            results["overall_metrics"] = self._evaluate_predictions(all_predictions)
        
        return results
    
    # 在 semantic_graph_qa_evaluator.py 中修改评估方法

    def _evaluate_predictions(self, predictions: List[Dict]) -> Dict[str, float]:
        """评估预测结果 - 使用evaluation.py的评分函数"""
        if not predictions:
            return {}
        
        try:
            # 使用evaluation.py的评估函数
            scores, _, recall_scores = eval_question_answering(predictions, eval_key="prediction")
            
            metrics = {
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "std_score": float(np.std(scores)) if scores else 0.0,
                "avg_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
                "std_recall": float(np.std(recall_scores)) if recall_scores else 0.0,
                "total_questions": len(predictions),
                "individual_scores": [float(s) for s in scores],  # 保存每个问题的具体分数
                "individual_recalls": [float(r) for r in recall_scores]  # 保存每个问题的召回率
            }
            
            # 按类别统计详细分数
            by_category = {}
            for i, pred in enumerate(predictions):
                cat = pred["category"]
                if cat not in by_category:
                    by_category[cat] = {
                        "scores": [], 
                        "recalls": [], 
                        "questions": [],
                        "predictions": [],
                        "answers": []
                    }
                
                if i < len(scores):
                    by_category[cat]["scores"].append(float(scores[i]))
                    by_category[cat]["questions"].append(pred["question"])
                    by_category[cat]["predictions"].append(pred["prediction"])
                    by_category[cat]["answers"].append(pred["answer"])
                if i < len(recall_scores):
                    by_category[cat]["recalls"].append(float(recall_scores[i]))
            
            # 计算每个类别的统计信息
            for cat, data in by_category.items():
                if data["scores"]:
                    metrics[f"category_{cat}_avg_score"] = float(np.mean(data["scores"]))
                    metrics[f"category_{cat}_std_score"] = float(np.std(data["scores"]))
                    metrics[f"category_{cat}_individual_scores"] = data["scores"]
                    metrics[f"category_{cat}_count"] = len(data["scores"])
                    
                    # 保存具体的问题和答案用于分析
                    metrics[f"category_{cat}_details"] = {
                        "questions": data["questions"],
                        "predictions": data["predictions"], 
                        "answers": data["answers"],
                        "scores": data["scores"]
                    }
                
                if data["recalls"]:
                    metrics[f"category_{cat}_avg_recall"] = float(np.mean(data["recalls"]))
                    metrics[f"category_{cat}_std_recall"] = float(np.std(data["recalls"]))
                    metrics[f"category_{cat}_individual_recalls"] = data["recalls"]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {"error": str(e)}

    def _save_detailed_results(self, results: Dict[str, Any], output_file: str):
        """保存详细的评估结果，包含每个问题的具体分数"""
        detailed_output = {
            "evaluation_info": results["evaluation_info"],
            "sample_results": []
        }
        
        # 为每个对话保存详细结果
        for conv_id, conv_results in results["conversation_results"].items():
            sample_result = {
                "sample_id": conv_id,
                "overall_metrics": conv_results.get("overall_metrics", {}),
                "category_results": {},
                "question_level_results": []
            }
            
            # 按类别整理结果
            for category, cat_data in conv_results.get("results_by_category", {}).items():
                sample_result["category_results"][category] = {
                    "strategy": cat_data["strategy"],
                    "questions_count": cat_data["questions_count"],
                    "metrics": cat_data.get("metrics", {}),
                }
            
            # 整理问题级别的详细结果
            for detail in conv_results.get("detailed_results", []):
                question_result = {
                    "question": detail["question"],
                    "category": detail["category"],
                    "ground_truth": detail["answer"],
                    "prediction": detail["prediction"], 
                    "evidence": detail.get("evidence", []),
                    "context_nodes_count": detail.get("context_nodes_count", 0),
                    "evidence_recall": detail.get("evidence_recall", 0.0),
                    "retrieval_strategy": detail.get("generation_details", {}).get("generation_strategy", "unknown")
                }
                
                # 计算该问题的具体分数
                question_result["f1_score"] = self._calculate_question_f1_score(
                    detail["prediction"], detail["answer"], detail["category"]
                )
                
                sample_result["question_level_results"].append(question_result)
            
            detailed_output["sample_results"].append(sample_result)
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_output, f, ensure_ascii=False, indent=2)

    def _calculate_question_f1_score(self, prediction: str, answer: str, category: int) -> float:
        """计算单个问题的F1分数"""
        try:
            if category == 5:  # 对抗性问题
                if 'no information available' in prediction.lower() or 'not mentioned' in prediction.lower():
                    return 1.0 if 'not mentioned' in answer.lower() else 0.0
                else:
                    return 0.0 if 'not mentioned' in answer.lower() else calculate_f1_score(prediction, answer)
            # elif category == 1:  # 多跳推理
            #     return f1(prediction, answer)  # 使用多答案F1
            else:  # 其他类别
                return calculate_f1_score(prediction, answer)
        except Exception as e:
            self.logger.error(f"计算F1分数失败: {e}")
            return 0.0
    
    # def _evaluate_predictions(self, predictions: List[Dict]) -> Dict[str, float]:
    #     """评估预测结果"""
    #     if not predictions:
    #         return {}
        
    #     # 转换为evaluation.py需要的格式
    #     eval_data = []
    #     for pred in predictions:
    #         eval_data.append({
    #             "category": pred["category"],
    #             "answer": pred["answer"],
    #             "prediction": pred["prediction"],
    #             "evidence": pred.get("evidence", [])
    #         })
        
    #     try:
    #         # 使用evaluation.py的评估函数
    #         scores, _, recall_scores = eval_question_answering(eval_data, eval_key="prediction")
            
    #         metrics = {
    #             "avg_score": float(np.mean(scores)) if scores else 0.0,
    #             "avg_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
    #             "total_questions": len(predictions)
    #         }
            
    #         # 按类别统计
    #         by_category = {}
    #         for i, pred in enumerate(predictions):
    #             cat = pred["category"]
    #             if cat not in by_category:
    #                 by_category[cat] = {"scores": [], "recalls": []}
    #             if i < len(scores):
    #                 by_category[cat]["scores"].append(scores[i])
    #             if i < len(recall_scores):
    #                 by_category[cat]["recalls"].append(recall_scores[i])
            
    #         for cat, data in by_category.items():
    #             metrics[f"category_{cat}_score"] = float(np.mean(data["scores"])) if data["scores"] else 0.0
    #             metrics[f"category_{cat}_recall"] = float(np.mean(data["recalls"])) if data["recalls"] else 0.0
    #             metrics[f"category_{cat}_count"] = len(data["scores"])
            
    #         return metrics
            
    #     except Exception as e:
    #         self.logger.error(f"评估失败: {e}")
    #         return {"error": str(e)}

    def run_full_evaluation(self, 
                           qa_test_data: Dict[str, List[Dict]],
                           save_results: bool = True) -> Dict[str, Any]:
        """
        运行完整评估
        
        Args:
            qa_test_data: QA测试数据 {conversation_id: [qa_items]}
            save_results: 是否保存结果
            
        Returns:
            完整评估结果
        """
        self.logger.info(f"开始完整评估，共 {len(qa_test_data)} 个对话")
        
        full_results = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_conversations": len(qa_test_data),
                "llm_model": self.llm_client.model_name,
                "retrieval_strategies": self.category_strategies
            },
            "conversation_results": {},
            "overall_summary": {}
        }
        
        all_conversation_metrics = []
        
        for conv_id, qa_list in qa_test_data.items():
            self.logger.info(f"评估对话 {conv_id}: {len(qa_list)} 个问题")
            
            try:
                conv_results = self.evaluate_single_conversation(conv_id, qa_list)
                full_results["conversation_results"][conv_id] = conv_results
                
                if conv_results.get("overall_metrics"):
                    all_conversation_metrics.append(conv_results["overall_metrics"])
                
            except Exception as e:
                self.logger.error(f"评估对话 {conv_id} 失败: {e}")
                full_results["conversation_results"][conv_id] = {"error": str(e)}
        
        # 计算总体摘要
        if all_conversation_metrics:
            full_results["overall_summary"] = self._calculate_overall_summary(all_conversation_metrics)
        
        # 保存结果
        if save_results:
            self._save_results(full_results)
        
        return full_results

    def _calculate_overall_summary(self, conversation_metrics: List[Dict]) -> Dict[str, Any]:
        """计算总体摘要统计"""
        summary = {}
        
        # 平均分数
        avg_scores = [m.get("avg_score", 0) for m in conversation_metrics if "avg_score" in m]
        if avg_scores:
            summary["overall_avg_score"] = float(np.mean(avg_scores))
            summary["overall_std_score"] = float(np.std(avg_scores))
        
        # 平均召回率
        avg_recalls = [m.get("avg_recall", 0) for m in conversation_metrics if "avg_recall" in m]
        if avg_recalls:
            summary["overall_avg_recall"] = float(np.mean(avg_recalls))
            summary["overall_std_recall"] = float(np.std(avg_recalls))
        
        # 按类别统计
        for category in range(1, 6):
            cat_scores = [m.get(f"category_{category}_score", 0) for m in conversation_metrics 
                         if f"category_{category}_score" in m]
            if cat_scores:
                summary[f"category_{category}_avg_score"] = float(np.mean(cat_scores))
                summary[f"category_{category}_std_score"] = float(np.std(cat_scores))
        
        # 总问题数
        total_questions = sum(m.get("total_questions", 0) for m in conversation_metrics)
        summary["total_questions_evaluated"] = total_questions
        
        return summary

    def _save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        full_results_file = self.output_dir / f"semantic_graph_qa_evaluation_{timestamp}.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存摘要报告
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== Semantic Graph QA Evaluation Summary ===\n\n")
            f.write(f"Timestamp: {results['evaluation_info']['timestamp']}\n")
            f.write(f"LLM Model: {results['evaluation_info']['llm_model']}\n")
            f.write(f"Total Conversations: {results['evaluation_info']['total_conversations']}\n\n")
            
            if "overall_summary" in results:
                summary = results["overall_summary"]
                f.write("=== Overall Results ===\n")
                f.write(f"Average Score: {summary.get('overall_avg_score', 0):.4f} ± {summary.get('overall_std_score', 0):.4f}\n")
                f.write(f"Average Recall: {summary.get('overall_avg_recall', 0):.4f} ± {summary.get('overall_std_recall', 0):.4f}\n")
                f.write(f"Total Questions: {summary.get('total_questions_evaluated', 0)}\n\n")
                
                f.write("=== Results by Category ===\n")
                for category in range(1, 6):
                    cat_score = summary.get(f"category_{category}_avg_score")
                    if cat_score is not None:
                        cat_std = summary.get(f"category_{category}_std_score", 0)
                        strategy = self.category_strategies.get(category, "unknown")
                        f.write(f"Category {category} ({strategy}): {cat_score:.4f} ± {cat_std:.4f}\n")
        
        self.logger.info(f"评估结果已保存到: {full_results_file}")
        self.logger.info(f"摘要报告已保存到: {summary_file}")


def main():
    """主函数 - 演示语义图谱QA评估"""
    
    # 1. 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 2. 加载已存储的语义图谱
    print("🔄 加载语义图谱...")
    storage = ConversationSemanticStorage()
    
    # 假设您已经运行了dataset_inserter存储了数据
    # 这里我们使用已有的语义图谱
    semantic_graph = storage.semantic_graph
    
    # 3. 初始化LLM客户端
    print("🔄 初始化LLM客户端...")
    llm_client = LLMClient(model_name="deepseek-chat")  # 或您喜欢的模型
    
    # 4. 创建评估器
    evaluator = SemanticGraphQAEvaluator(
        semantic_graph=semantic_graph,
        llm_client=llm_client
    )
    
    # 5. 获取QA测试数据
    print("📋 获取QA测试数据...")
    qa_test_data = storage.get_qa_test_data(["conv-26", "conv-30"])  # 测试指定对话
    
    if not qa_test_data:
        print("❌ 没有找到QA测试数据")
        return
    
    print(f"📊 发现测试数据: {sum(len(qa_list) for qa_list in qa_test_data.values())} 个问题")
    
    # 6. 运行评估
    print("🚀 开始评估...")
    results = evaluator.run_full_evaluation(qa_test_data)
    
    # 7. 显示结果
    print("\n🎯 评估完成！")
    if "overall_summary" in results:
        summary = results["overall_summary"]
        print(f"📈 总体平均分数: {summary.get('overall_avg_score', 0):.4f}")
        print(f"📈 总体平均召回率: {summary.get('overall_avg_recall', 0):.4f}")
        print(f"📊 总计评估问题: {summary.get('total_questions_evaluated', 0)} 个")
        
        print("\n📋 按类别结果:")
        for category in range(1, 6):
            cat_score = summary.get(f"category_{category}_avg_score")
            if cat_score is not None:
                strategy = evaluator.category_strategies.get(category, "unknown")
                print(f"  类别 {category} ({strategy}): {cat_score:.4f}")


if __name__ == "__main__":
    main()