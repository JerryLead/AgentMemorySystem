import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# 添加父目录到路径以便导入dev模块
sys.path.append(str(Path(__file__).parent.parent.parent))
from dev import SemanticGraph, MemoryUnit
from benchmark.task_eval.evaluation import eval_question_answering, f1_score, exact_match_score
from benchmark.llm_utils import LLMClient, PromptBuilder, AnswerExtractor

# --- 配置 ---
LOGGING_LEVEL = logging.INFO
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "locomo"
DATASET_PATH = DATASET_DIR / "locomo10.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
TOP_K_RESULTS = 5
EVALUATION_MODEL = 'all-MiniLM-L6-v2'

# LLM配置
LLM_MODELS = [
    "deepseek-chat",

    # "gpt-3.5-turbo",
    # "gpt-4",
    # 可以添加更多模型
]

# 创建结果目录
RESULTS_DIR.mkdir(exist_ok=True)

# --- 日志设置 ---
logging.basicConfig(
    level=LOGGING_LEVEL, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / f'locomo_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# ... 保留之前的数据加载和处理函数 ...

def llm_enhanced_evaluation(graph: SemanticGraph, 
                           qa_pairs: List[Dict], 
                           eval_model: SentenceTransformer,
                           llm_models: List[str]) -> Dict[str, Any]:
    """使用LLM增强的评估函数"""
    
    # 初始化工具
    prompt_builder = PromptBuilder()
    answer_extractor = AnswerExtractor()
    
    results = {
        'total_questions': len(qa_pairs),
        'models': {},
        'category_results': {},
        'detailed_results': []
    }
    
    # 为每个模型创建结果存储
    for model in llm_models:
        results['models'][model] = {
            'all_scores': [],
            'f1_scores': [],
            'similarity_scores': [],
            'exact_match_scores': [],
            'category_results': {}
        }
    
    for qa_idx, qa in enumerate(qa_pairs):
        question = qa.get('question')
        golden_answer = qa.get('answer')
        category = qa.get('category', 0)
        evidence = qa.get('evidence', [])
        
        if not question or not golden_answer:
            continue
        
        logging.info(f"处理问题 {qa_idx + 1}/{len(qa_pairs)}: {question[:50]}...")
        
        # 1. 检索相关记忆
        retrieved_units = graph.search_similarity_in_graph(
            query_text=question, k=TOP_K_RESULTS
        )
        
        if not retrieved_units:
            logging.warning(f"问题 {qa_idx + 1} 没有检索到相关记忆")
            continue
        
        # 2. 构建提示词
        prompt = prompt_builder.build_prompt(
            question=question,
            context_units=retrieved_units,
            category=category,
            evidence=evidence
        )
        
        question_result = {
            'question_id': qa_idx,
            'question': question,
            'golden_answer': golden_answer,
            'category': category,
            'evidence': evidence,
            'retrieved_count': len(retrieved_units),
            'context_summary': _summarize_context(retrieved_units),
            'prompt': prompt,
            'model_results': {}
        }
        
        # 3. 对每个模型进行评估
        for model_name in llm_models:
            logging.info(f"  使用模型 {model_name} 生成答案...")
            
            try:
                # 创建LLM客户端
                llm_client = LLMClient(model_name=model_name)
                
                # 生成答案
                llm_response = llm_client.generate_answer(prompt)
                
                # 提取答案
                predicted_answer = answer_extractor.extract_answer(llm_response, category)
                
                # 评估答案质量
                scores = _evaluate_answer(
                    predicted_answer, golden_answer, category, eval_model
                )
                
                # 记录结果
                model_result = {
                    'llm_response': llm_response,
                    'predicted_answer': predicted_answer,
                    'scores': scores
                }
                
                question_result['model_results'][model_name] = model_result
                
                # 更新统计
                model_stats = results['models'][model_name]
                model_stats['all_scores'].append(scores['combined_score'])
                model_stats['f1_scores'].append(scores['f1_score'])
                model_stats['similarity_scores'].append(scores['similarity_score'])
                model_stats['exact_match_scores'].append(scores['exact_match'])
                
                # 按类别统计
                if category not in model_stats['category_results']:
                    model_stats['category_results'][category] = {
                        'count': 0,
                        'scores': [],
                        'f1_scores': [],
                        'similarity_scores': [],
                        'exact_match_scores': []
                    }
                
                cat_stats = model_stats['category_results'][category]
                cat_stats['count'] += 1
                cat_stats['scores'].append(scores['combined_score'])
                cat_stats['f1_scores'].append(scores['f1_score'])
                cat_stats['similarity_scores'].append(scores['similarity_score'])
                cat_stats['exact_match_scores'].append(scores['exact_match'])
                
                logging.info(f"    {model_name}: F1={scores['f1_score']:.3f}, "
                           f"相似度={scores['similarity_score']:.3f}, "
                           f"精确匹配={scores['exact_match']}")
                
            except Exception as e:
                logging.error(f"模型 {model_name} 处理问题 {qa_idx + 1} 失败: {e}")
                question_result['model_results'][model_name] = {
                    'error': str(e),
                    'predicted_answer': "生成失败",
                    'scores': {'f1_score': 0, 'similarity_score': 0, 'exact_match': 0, 'combined_score': 0}
                }
        
        results['detailed_results'].append(question_result)
        
        # 打印当前问题的结果摘要
        print(f"\n{'='*60}")
        print(f"问题 {qa_idx + 1}/{len(qa_pairs)} (类别 {category})")
        print(f"❓ 问题: {question}")
        print(f"✅ 标准答案: {golden_answer}")
        print(f"🔍 检索到 {len(retrieved_units)} 个相关记忆")
        
        for model_name in llm_models:
            if model_name in question_result['model_results']:
                result = question_result['model_results'][model_name]
                if 'error' not in result:
                    scores = result['scores']
                    print(f"🤖 {model_name}: {result['predicted_answer']}")
                    print(f"   📊 F1={scores['f1_score']:.3f}, 相似度={scores['similarity_score']:.3f}, EM={scores['exact_match']}")
    
    return results

def _summarize_context(retrieved_units: List[Tuple]) -> str:
    """总结检索到的上下文"""
    if not retrieved_units:
        return "无上下文"
    
    sources = []
    for unit, score in retrieved_units:
        data_source = unit.metadata.get('data_source', 'unknown')
        sources.append(f"{data_source}({score:.3f})")
    
    return ", ".join(sources[:3])

def _evaluate_answer(predicted_answer: str, 
                    golden_answer: str, 
                    category: int,
                    eval_model: SentenceTransformer) -> Dict[str, float]:
    """评估答案质量"""
    
    # F1得分
    f1_score_result = f1_score(predicted_answer, golden_answer)
    
    # 精确匹配
    exact_match = 1.0 if exact_match_score(predicted_answer, golden_answer) else 0.0
    
    # 语义相似度
    try:
        embedding1 = eval_model.encode(predicted_answer, convert_to_tensor=True)
        embedding2 = eval_model.encode(golden_answer, convert_to_tensor=True)
        similarity_score = util.cos_sim(embedding1, embedding2).item()
    except:
        similarity_score = 0.0
    
    # 综合得分
    combined_score = (f1_score_result + similarity_score + exact_match) / 3
    
    return {
        'f1_score': f1_score_result,
        'similarity_score': similarity_score,
        'exact_match': exact_match,
        'combined_score': combined_score
    }

def calculate_llm_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """计算LLM评估指标"""
    metrics = {}
    
    for model_name, model_results in results['models'].items():
        model_metrics = {}
        
        # 整体指标
        if model_results['all_scores']:
            model_metrics['avg_combined_score'] = np.mean(model_results['all_scores'])
            model_metrics['std_combined_score'] = np.std(model_results['all_scores'])
            model_metrics['avg_f1_score'] = np.mean(model_results['f1_scores'])
            model_metrics['avg_similarity_score'] = np.mean(model_results['similarity_scores'])
            model_metrics['exact_match_rate'] = np.mean(model_results['exact_match_scores'])
        
        # 按类别指标
        category_metrics = {}
        for category, cat_results in model_results['category_results'].items():
            if cat_results['scores']:
                category_metrics[f'category_{category}'] = {
                    'count': cat_results['count'],
                    'avg_score': np.mean(cat_results['scores']),
                    'avg_f1': np.mean(cat_results['f1_scores']),
                    'avg_similarity': np.mean(cat_results['similarity_scores']),
                    'exact_match_rate': np.mean(cat_results['exact_match_scores'])
                }
        
        model_metrics['categories'] = category_metrics
        metrics[model_name] = model_metrics
    
    return metrics

def print_llm_evaluation_summary(results: Dict[str, Any], metrics: Dict[str, Any]):
    """打印LLM评估总结"""
    print(f"\n{'='*80}")
    print("📊 LoCoMo LLM 增强评估总结")
    print(f"{'='*80}")
    
    print(f"总问题数: {results['total_questions']}")
    print(f"测试模型: {', '.join(results['models'].keys())}")
    
    # 模型对比
    print(f"\n🏆 模型性能对比:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  综合得分: {model_metrics.get('avg_combined_score', 0):.4f} ± {model_metrics.get('std_combined_score', 0):.4f}")
        print(f"  F1得分: {model_metrics.get('avg_f1_score', 0):.4f}")
        print(f"  语义相似度: {model_metrics.get('avg_similarity_score', 0):.4f}")
        print(f"  精确匹配率: {model_metrics.get('exact_match_rate', 0):.4f}")
        
        # 按类别显示
        categories = model_metrics.get('categories', {})
        if categories:
            print(f"  按类别统计:")
            for cat_key, cat_data in categories.items():
                print(f"    {cat_key}: {cat_data['count']} 问题, "
                      f"得分 {cat_data['avg_score']:.3f}, "
                      f"F1 {cat_data['avg_f1']:.3f}, "
                      f"EM {cat_data['exact_match_rate']:.3f}")

def save_llm_results(results: Dict[str, Any], metrics: Dict[str, Any], timestamp: str):
    """保存LLM评估结果"""
    result_file = RESULTS_DIR / f'locomo_llm_evaluation_{timestamp}.json'
    
    output_data = {
        'timestamp': timestamp,
        'config': {
            'dataset_path': str(DATASET_PATH),
            'top_k_results': TOP_K_RESULTS,
            'evaluation_model': EVALUATION_MODEL,
            'llm_models': LLM_MODELS
        },
        'results': results,
        'metrics': metrics
    }
    
    # 转换numpy类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def deep_convert(data):
        if isinstance(data, dict):
            return {k: deep_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [deep_convert(item) for item in data]
        else:
            return convert_numpy(data)
    
    output_data = deep_convert(output_data)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"LLM评估结果已保存到: {result_file}")

# 在现有的导入部分后添加这些函数：

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """加载LoCoMo数据集文件。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.info(f"成功加载数据集: {file_path}")
            logging.info(f"数据集包含 {len(data)} 个样本")
            return data
    except FileNotFoundError:
        logging.error(f"数据集文件未找到: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}")
        return []

def parse_locomo_data(data: List[Dict[str, Any]]) -> List[Dict]:
    """
    解析LoCoMo数据集，仅提取对话、event_summary、observation等记忆单元（不返回QA）。
    返回所有对话消息、event_summary、observation的列表，每条为一个dict。
    """
    memory_units = []
    for sample in data:
        sample_id = sample.get('sample_id', 'unknown')

        # 处理对话消息
        if 'conversation' in sample:
            conversation_data = sample['conversation']
            for key, value in conversation_data.items():
                if key.startswith('session_') and not key.endswith('_date_time') and isinstance(value, list):
                    session_time = conversation_data.get(f"{key}_date_time", "")
                    for msg_idx, message in enumerate(value):
                        if isinstance(message, dict):
                            memory_units.append({
                                'type': 'dialog',
                                'sample_id': sample_id,
                                'session': key,
                                'session_time': session_time,
                                'message_index': msg_idx,
                                'speaker': message.get('speaker', 'unknown'),
                                'content': message.get('message', message.get('text', ''))
                            })

        # 处理 event_summary
        if 'event_summary' in sample:
            for ev_key, ev in sample['event_summary'].items():
                memory_units.append({
                    'type': 'event_summary',
                    'sample_id': sample_id,
                    'event_key': ev_key,
                    'event': ev
                })

        # 处理 observation
        if 'observation' in sample:
            for ob_key, ob in sample['observation'].items():
                memory_units.append({
                    'type': 'observation',
                    'sample_id': sample_id,
                    'observation_key': ob_key,
                    'observation': ob
                })

        # 处理 session_summary
        if 'session_summary' in sample:
            for ss_key, ss in sample['session_summary'].items():
                memory_units.append({
                    'type': 'session_summary',
                    'sample_id': sample_id,
                    'session_summary_key': ss_key,
                    'summary': ss
                })

    return memory_units

def extract_all_speakers(data: List[Dict]) -> List[str]:
    """从所有对话消息中提取说话人（不依赖QA）"""
    speakers = set()
    for sample in data:
        if 'conversation' in sample:
            conversation_data = sample['conversation']
            for key, value in conversation_data.items():
                if key.startswith('session_') and not key.endswith('_date_time') and isinstance(value, list):
                    for message in value:
                        if isinstance(message, dict):
                            speaker = message.get('speaker')
                            if speaker:
                                speakers.add(speaker.lower())
    return sorted(list(speakers))

def setup_locomo_memory_spaces(graph: SemanticGraph, data: List[Dict]):
    graph.create_memory_space_in_map("locomo_dialogs")
    for sample in data:
        sample_id = sample.get('sample_id', 'unknown')
        graph.create_memory_space_in_map(f"session_summary_{sample_id}")
        graph.create_memory_space_in_map(f"event_summary_{sample_id}")
        graph.create_memory_space_in_map(f"observation_{sample_id}")
        # 为每个session建空间
        if 'conversation' in sample:
            for key in sample['conversation']:
                if key.startswith('session_') and not key.endswith('_date_time'):
                    graph.create_memory_space_in_map(f"session_{sample_id}_{key}")
    # 说话人空间
    speakers = extract_all_speakers(data)
    for speaker in speakers:
        graph.create_memory_space_in_map(f"speaker_{speaker}")
    # QA空间
    graph.create_memory_space_in_map("locomo_qa_pairs")
    for category in [1, 2, 3, 4, 5]:
        graph.create_memory_space_in_map(f"locomo_qa_category_{category}")


def ingest_conversation_history(graph: SemanticGraph, data: List[Dict]) -> int:
    logging.info("开始注入LoCoMo对话历史...")
    total_messages = 0
    setup_locomo_memory_spaces(graph, data)

    for sample in data:
        sample_id = sample.get('sample_id', f'sample_{total_messages}')

        # 处理session_summary
        if 'session_summary' in sample:
            for session_key, summary in sample['session_summary'].items():
                if isinstance(summary, str) and summary.strip():
                    unit_id = f"{sample_id}_{session_key}_summary"
                    summary_unit = MemoryUnit(
                        uid=unit_id,
                        raw_data={
                            "text_content": summary,
                            "content_type": "session_summary",
                            "session": session_key,
                            "sample_id": sample_id
                        },
                        metadata={
                            "data_source": "locomo_summary",
                            "sample_id": sample_id,
                            "session": session_key,
                            "content_type": "summary"
                        }
                    )
                    graph.add_unit(summary_unit)
                    graph.add_unit_to_space_in_map(unit_id, f"session_summary_{sample_id}")
                    total_messages += 1

        # 处理event_summary
        if 'event_summary' in sample:
            for ev_key, ev in sample['event_summary'].items():
                unit_id = f"{sample_id}_event_{ev_key}"
                event_unit = MemoryUnit(
                    uid=unit_id,
                    raw_data={
                        "text_content": ev,
                        "content_type": "event_summary",
                        "event_key": ev_key,
                        "sample_id": sample_id
                    },
                    metadata={
                        "data_source": "locomo_event_summary",
                        "sample_id": sample_id,
                        "event_key": ev_key,
                        "content_type": "event_summary"
                    }
                )
                graph.add_unit(event_unit)
                graph.add_unit_to_space_in_map(unit_id, f"event_summary_{sample_id}")
                total_messages += 1

        # 处理observation
        if 'observation' in sample:
            for ob_key, ob in sample['observation'].items():
                unit_id = f"{sample_id}_observation_{ob_key}"
                obs_unit = MemoryUnit(
                    uid=unit_id,
                    raw_data={
                        "text_content": ob,
                        "content_type": "observation",
                        "observation_key": ob_key,
                        "sample_id": sample_id
                    },
                    metadata={
                        "data_source": "locomo_observation",
                        "sample_id": sample_id,
                        "observation_key": ob_key,
                        "content_type": "observation"
                    }
                )
                graph.add_unit(obs_unit)
                graph.add_unit_to_space_in_map(unit_id, f"observation_{sample_id}")
                total_messages += 1

        # 处理对话消息
        if 'conversation' in sample:
            conversation_data = sample['conversation']
            for key, value in conversation_data.items():
                if key.startswith('session_') and not key.endswith('_date_time') and isinstance(value, list):
                    session_time = conversation_data.get(f"{key}_date_time", "")
                    for msg_idx, message in enumerate(value):
                        if isinstance(message, dict) and message.get('message'):
                            unit_id = f"{sample_id}_{key}_msg_{msg_idx}"
                            speaker = message.get('speaker', 'unknown')
                            content = message.get('message', '')
                            if content.strip():
                                unit = MemoryUnit(
                                    uid=unit_id,
                                    raw_data={
                                        "text_content": f"{speaker}: {content}",
                                        "speaker": speaker,
                                        "message_content": content,
                                        "session": key,
                                        "sample_id": sample_id
                                    },
                                    metadata={
                                        "timestamp": session_time,
                                        "conversation_id": sample_id,
                                        "session": key,
                                        "message_index": msg_idx,
                                        "data_source": "locomo_dialog",
                                        "speaker": speaker
                                    }
                                )
                                graph.add_unit(unit)
                                graph.add_unit_to_space_in_map(unit_id, "locomo_dialogs")
                                graph.add_unit_to_space_in_map(unit_id, f"session_{sample_id}_{key}")
                                graph.add_unit_to_space_in_map(unit_id, f"speaker_{speaker.lower().replace(' ', '_')}")
                                total_messages += 1

        # 处理QA对（只加入QA空间，不加入locomo_dialogs）
        if 'qa' in sample:
            for qa_idx, qa in enumerate(sample['qa']):
                if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                    qa_unit_id = f"{sample_id}_qa_{qa_idx}"
                    qa_unit = MemoryUnit(
                        uid=qa_unit_id,
                        raw_data={
                            "text_content": f"Q: {qa['question']} A: {qa['answer']}",
                            "question": qa['question'],
                            "answer": qa['answer'],
                            "evidence": qa.get('evidence', []),
                            "category": qa.get('category', 0)
                        },
                        metadata={
                            "data_source": "locomo_qa",
                            "sample_id": sample_id,
                            "qa_category": qa.get('category', 0),
                            "evidence_sources": qa.get('evidence', [])
                        }
                    )
                    graph.add_unit(qa_unit)
                    graph.add_unit_to_space_in_map(qa_unit_id, "locomo_qa_pairs")
                    category = qa.get('category', 0)
                    if category > 0:
                        graph.add_unit_to_space_in_map(qa_unit_id, f"locomo_qa_category_{category}")

    logging.info(f"数据注入完成。共添加 {total_messages} 个记忆单元")
    return total_messages

def format_enhanced_context(retrieved_units: List[Tuple]) -> str:
    """格式化增强的上下文信息"""
    if not retrieved_units:
        return "没有检索到相关记忆。"
    
    context_lines = ["--- 检索到的相关记忆上下文 ---"]
    
    for unit, score in retrieved_units:
        data_source = unit.metadata.get('data_source', 'unknown')
        
        if data_source == 'locomo_summary':
            session = unit.raw_data.get('session', 'unknown')
            content = unit.raw_data.get('text_content', '')
            speakers = unit.raw_data.get('speakers', 'Unknown')
            line = f"[摘要-{session}] {speakers}: {content[:200]}... (相似度: {score:.4f})"
            
        elif data_source == 'locomo_dialog':
            speaker = unit.raw_data.get('speaker', 'unknown')
            content = unit.raw_data.get('message_content', '')
            session = unit.metadata.get('session', 'unknown')
            line = f"[对话-{session}] {speaker}: {content} (相似度: {score:.4f})"
            
        elif data_source == 'locomo_qa':
            question = unit.raw_data.get('question', '')
            answer = unit.raw_data.get('answer', '')
            category = unit.metadata.get('qa_category', 0)
            line = f"[QA-类别{category}] Q: {question} A: {answer} (相似度: {score:.4f})"
            
        else:
            content = unit.raw_data.get('text_content', '')
            if not isinstance(content, str):
                content = str(content)
            line = f"[{data_source}] {content[:100]}... (相似度: {score:.4f})"
            # content = unit.raw_data.get('text_content', '')[:100]
            # line = f"[{data_source}] {content}... (相似度: {score:.4f})"
        
        context_lines.append(line)
    
    context_lines.append("-" * 60)
    return "\n".join(context_lines)

def extract_answer_from_context(question: str, context: str, retrieved_units: List[Tuple]) -> str:
    """
    从检索到的上下文中提取答案
    这是一个简化的答案提取逻辑，实际应用中会使用LLM
    """
    if not retrieved_units:
        return "无法找到相关信息"
    
    # 简单的答案提取：取相似度最高的内容作为答案的基础
    best_unit, best_score = retrieved_units[0]
    
    # 根据不同的数据源提取答案
    data_source = best_unit.metadata.get('data_source', 'unknown')
    
    if data_source == 'locomo_qa' and best_score > 0.3:
        # 如果检索到的是QA对，直接使用答案
        return best_unit.raw_data.get('answer', '未找到答案')
    elif data_source in ['locomo_dialog', 'locomo_summary'] and best_score > 0.3:
        # 如果检索到的是对话或摘要，提取相关内容
        content = best_unit.raw_data.get('text_content', '')
        return f"根据对话记录: {content[:200]}"
    else:
        return "未找到足够相关的信息"

def enhanced_search_and_evaluate(graph: SemanticGraph, qa_pairs: List[Dict], eval_model: SentenceTransformer) -> Dict[str, Any]:
    """增强的搜索和评估函数"""
    results = {
        'total_questions': len(qa_pairs),
        'category_results': {},
        'retrieval_results': [],
        'all_scores': []
    }
    
    for qa in qa_pairs:
        question = qa.get('question')
        golden_answer = qa.get('answer')
        category = qa.get('category', 0)
        evidence = qa.get('evidence', [])
        
        if not question or not golden_answer:
            continue
    
        # 1. 多策略检索（排除QA空间）
        search_results = {}
        
        # 全局搜索（需要过滤QA结果）
        all_global_results = graph.search_similarity_in_graph(query_text=question, k=10)
        search_results['global'] = [
            (unit, score) for unit, score in all_global_results 
            if unit.metadata.get('data_source') != 'locomo_qa'
        ][:5]
        
        # 在对话空间中搜索
        try:
            search_results['dialog_only'] = graph.search_similarity_in_graph(
                query_text=question, k=5, space_name="locomo_dialogs"
            )
        except:
            search_results['dialog_only'] = []
        
        # 在summary空间中搜索
        try:
            summary_results = []
            # 可以遍历所有 session_summary_* 空间
            # summary_results.extend(graph.search_similarity_in_graph(...))
            search_results['summary_only'] = summary_results
        except:
            search_results['summary_only'] = []
        
        # # 1. 多策略检索
        # search_results = {}
        # # 全局搜索
        # search_results['global'] = graph.search_similarity_in_graph(
        #     query_text=question, k=5
        # )
        
        # # 在对话空间中搜索
        # try:
        #     search_results['dialog_only'] = graph.search_similarity_in_graph(
        #         query_text=question, k=5, space_name="locomo_dialogs"
        #     )
        # except:
        #     search_results['dialog_only'] = []
        
        # QA空间不作为检索策略

        # # 在同类别QA中搜索
        # if category > 0:
        #     try:
        #         search_results[f'category_{category}'] = graph.search_similarity_in_graph(
        #             query_text=question, k=3, space_name=f"locomo_qa_category_{category}"
        #         )
        #     except:
        #         search_results[f'category_{category}'] = []
        
        # 2. 融合检索结果
        all_retrieved = []
        for strategy, units in search_results.items():
            for unit, score in units:
                all_retrieved.append((unit, score, strategy))
        
        # 按相似度排序并去重
        seen_uids = set()
        unique_retrieved = []
        for unit, score, strategy in sorted(all_retrieved, key=lambda x: x[1], reverse=True):
            if unit.uid not in seen_uids:
                unique_retrieved.append((unit, score, strategy))
                seen_uids.add(unit.uid)
        
        # 3. 生成上下文和答案
        top_retrieved = unique_retrieved[:TOP_K_RESULTS]
        context_str = format_enhanced_context([(u, s) for u, s, _ in top_retrieved])
        predicted_answer = extract_answer_from_context(question, context_str, [(u, s) for u, s, _ in top_retrieved])
        
        # 4. 评估
        try:
            # 使用F1得分评估
            f1_score_result = f1_score(predicted_answer, golden_answer)
            
            # 使用语义相似度评估
            embedding1 = eval_model.encode(predicted_answer, convert_to_tensor=True)
            embedding2 = eval_model.encode(golden_answer, convert_to_tensor=True)
            similarity_score = util.cos_sim(embedding1, embedding2).item()
            
            # 综合得分
            combined_score = (f1_score_result + similarity_score) / 2
            
        except Exception as e:
            logging.error(f"评估过程中出错: {e}")
            f1_score_result = 0
            similarity_score = 0
            combined_score = 0
        
        # 5. 记录结果
        result_entry = {
            'question': question,
            'golden_answer': golden_answer,
            'predicted_answer': predicted_answer,
            'category': category,
            'evidence': evidence,
            'retrieved_count': len(unique_retrieved),
            'f1_score': f1_score_result,
            'similarity_score': similarity_score,
            'combined_score': combined_score,
            'context': context_str,
            'search_strategies': list(search_results.keys())
        }
        
        results['retrieval_results'].append(result_entry)
        results['all_scores'].append(combined_score)
        
        # 按类别统计
        if category not in results['category_results']:
            results['category_results'][category] = {
                'count': 0,
                'scores': [],
                'f1_scores': [],
                'similarity_scores': [],
                'avg_retrieved': 0
            }
        
        results['category_results'][category]['count'] += 1
        results['category_results'][category]['scores'].append(combined_score)
        results['category_results'][category]['f1_scores'].append(f1_score_result)
        results['category_results'][category]['similarity_scores'].append(similarity_score)
        results['category_results'][category]['avg_retrieved'] += len(unique_retrieved)
        
        # 打印详细结果
        print(f"\n{'='*60}")
        print(f"类别 {category} | 证据: {evidence}")
        print(f"❓ 问题: {question}")
        print(f"✅ 标准答案: {golden_answer}")
        print(f"🤖 预测答案: {predicted_answer}")
        print(f"🔍 检索策略: {', '.join(search_results.keys())}")
        print(f"📄 检索到 {len(unique_retrieved)} 个相关单元")
        print(f"📊 F1得分: {f1_score_result:.4f}")
        print(f"📊 语义相似度: {similarity_score:.4f}")
        print(f"📊 综合得分: {combined_score:.4f}")
        print(context_str)
    
    return results

def calculate_final_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """计算最终评估指标"""
    total_questions = results['total_questions']
    metrics = {}
    
    if total_questions > 0:
        # 检索成功率
        successful_retrievals = sum(1 for r in results['retrieval_results'] if r['retrieved_count'] > 0)
        metrics['retrieval_success_rate'] = successful_retrievals / total_questions
        
        # 平均得分
        if results['all_scores']:
            metrics['avg_combined_score'] = np.mean(results['all_scores'])
            metrics['std_combined_score'] = np.std(results['all_scores'])
        
        # 按类别的指标
        for category, cat_results in results['category_results'].items():
            if cat_results['count'] > 0:
                cat_results['avg_retrieved'] = cat_results['avg_retrieved'] / cat_results['count']
                
                if cat_results['scores']:
                    metrics[f'category_{category}_avg_score'] = np.mean(cat_results['scores'])
                    metrics[f'category_{category}_avg_f1'] = np.mean(cat_results['f1_scores'])
                    metrics[f'category_{category}_avg_similarity'] = np.mean(cat_results['similarity_scores'])
    
    return metrics

def save_results(results: Dict[str, Any], metrics: Dict[str, float], timestamp: str):
    """保存评估结果"""
    result_file = RESULTS_DIR / f'locomo_evaluation_{timestamp}.json'
    
    output_data = {
        'timestamp': timestamp,
        'config': {
            'dataset_path': str(DATASET_PATH),
            'top_k_results': TOP_K_RESULTS,
            'evaluation_model': EVALUATION_MODEL
        },
        'results': results,
        'metrics': metrics
    }
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def deep_convert(data):
        if isinstance(data, dict):
            return {k: deep_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [deep_convert(item) for item in data]
        else:
            return convert_numpy(data)
    
    output_data = deep_convert(output_data)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"评估结果已保存到: {result_file}")

def print_evaluation_summary(results: Dict[str, Any], metrics: Dict[str, float]):
    """打印评估总结"""
    print(f"\n{'='*80}")
    print("📊 LoCoMo 评估总结")
    print(f"{'='*80}")
    
    if results['all_scores']:
        print(f"总问题数: {results['total_questions']}")
        print(f"检索成功率: {metrics.get('retrieval_success_rate', 0):.4f}")
        print(f"平均综合得分: {metrics.get('avg_combined_score', 0):.4f} ± {metrics.get('std_combined_score', 0):.4f}")
        
        print(f"\n按类别统计:")
        for category, cat_results in results['category_results'].items():
            if cat_results['scores']:
                cat_avg = metrics.get(f'category_{category}_avg_score', 0)
                cat_f1 = metrics.get(f'category_{category}_avg_f1', 0)
                cat_sim = metrics.get(f'category_{category}_avg_similarity', 0)
                print(f"  类别 {category}: {cat_results['count']} 问题, "
                      f"综合得分 {cat_avg:.4f}, "
                      f"F1得分 {cat_f1:.4f}, "
                      f"语义相似度 {cat_sim:.4f}, "
                      f"平均检索数 {cat_results['avg_retrieved']:.1f}")

def analyze_dataset_structure(data: List[Dict]) -> Dict[str, Any]:
    """分析数据集结构"""
    stats = {
        'total_samples': len(data),
        'speakers': {},
        'qa_categories': {},
        'sessions_per_sample': [],
        'qa_per_sample': []
    }
    
    all_speakers = set()
    all_categories = set()
    
    for sample in data:
        sample_id = sample.get('sample_id', 'unknown')
        
        # 分析对话
        if 'conversation' in sample:
            conversation_data = sample['conversation']
            speaker_a = conversation_data.get('speaker_a')
            speaker_b = conversation_data.get('speaker_b')
            
            if speaker_a:
                all_speakers.add(speaker_a)
            if speaker_b:
                all_speakers.add(speaker_b)
            
            # 统计session数量
            session_count = sum(1 for key in conversation_data.keys() 
                              if key.startswith('session_') and not key.endswith('_date_time'))
            stats['sessions_per_sample'].append(session_count)
        
        # 分析QA
        if 'qa' in sample:
            qa_count = len(sample['qa'])
            stats['qa_per_sample'].append(qa_count)
            
            for qa in sample['qa']:
                category = qa.get('category', 0)
                if category:
                    all_categories.add(category)
    
    stats['speakers'] = {
        'total_unique': len(all_speakers),
        'list': sorted(list(all_speakers))
    }
    
    stats['qa_categories'] = {
        'total_categories': len(all_categories),
        'list': sorted(list(all_categories))
    }
    
    if stats['sessions_per_sample']:
        stats['avg_sessions_per_sample'] = np.mean(stats['sessions_per_sample'])
    
    if stats['qa_per_sample']:
        stats['avg_qa_per_sample'] = np.mean(stats['qa_per_sample'])
    
    return stats

# 修改 main 函数，去掉LLM增强评估部分，使用基础评估
def main():
    """主执行函数（按 sample_id 分组逐个评测）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # 初始化系统
        logging.info("初始化AgentMemorySystem...")

        # 加载数据集
        logging.info(f"加载数据集: {DATASET_PATH}")
        raw_data = load_dataset(DATASET_PATH)
        if not raw_data:
            logging.error("数据集加载失败")
            return

        # 分组：sample_id -> sample
        sample_map = {sample.get('sample_id', f'sample_{i}'): sample for i, sample in enumerate(raw_data)}
        all_sample_ids = list(sample_map.keys())
        logging.info(f"共检测到 {len(all_sample_ids)} 个 sample_id: {all_sample_ids}")

        # 逐个 sample_id 测试
        for sample_id in all_sample_ids:
            logging.info(f"\n{'='*40}\n开始评测 sample_id: {sample_id}\n{'='*40}")
            sample = sample_map[sample_id]
            # 只处理当前 sample
            sample_data = [sample]

            # 分析结构
            dataset_stats = analyze_dataset_structure(sample_data)
            logging.info(f"sample_id={sample_id} 结构统计: {dataset_stats}")

            # 解析数据
            memory_unit = parse_locomo_data(sample_data)

            # 提取QA对
            qa_pairs = []
            if 'qa' in sample:
                for qa in sample['qa']:
                    if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                        qa_pairs.append(qa)

            # 初始化图谱
            hippo_graph = SemanticGraph()
            total_messages = ingest_conversation_history(hippo_graph, sample_data)
            if total_messages == 0:
                logging.warning(f"sample_id={sample_id} 没有成功注入任何对话数据")
                continue

            # 构建索引
            logging.info("构建向量索引...")
            hippo_graph.build_semantic_map_index()
            logging.info("索引构建完成")

            # 显示图谱统计信息
            hippo_graph.display_graph_summary()

            # 加载评估模型
            logging.info(f"加载评估模型: {EVALUATION_MODEL}...")
            eval_model = SentenceTransformer(EVALUATION_MODEL)

            # 执行基础评估
            results = enhanced_search_and_evaluate(hippo_graph, qa_pairs, eval_model)

            # 计算指标
            metrics = calculate_final_metrics(results)

            # 打印总结
            print_evaluation_summary(results, metrics)

            # 保存结果（每个 sample_id 单独保存）
            save_results(results, metrics, f"{sample_id}_{timestamp}")

    except Exception as e:
        logging.error(f"评估过程中发生错误: {e}", exc_info=True)

    logging.info("全部评估完成")

if __name__ == "__main__":
    main()
