import regex
import json
import string
import unicodedata
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import os
import asyncio
import time
import logging

# 评估相关库
from bert_score import score
from nltk.stem import PorterStemmer
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_utils.llm_client import LLMClient

# 尝试下载NLTK资源
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK resources: {e}")

# 初始化词干提取器
ps = PorterStemmer()

# 长度阈值
LENGTH_THRESHOLD = 5

# 全局语义模型
_sentence_model = None

def get_sentence_model():
    """获取全局语义模型"""
    global _sentence_model
    if _sentence_model is None:
        try:
            # 可以根据需要更改模型
            model_name = "all-MiniLM-L6-v2"  # 或 "Qwen/Qwen3-Embedding-0.6B"
            _sentence_model = SentenceTransformer(model_name)
            logging.info(f"SentenceTransformer model {model_name} loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}")
            _sentence_model = None
    return _sentence_model

# ================================
# LLM评估相关类和函数
# ================================

class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")

async def llm_grader_async(llm_client: LLMClient, question: str, gold_answer: str, response: str) -> bool:
    """
    LLM-as-a-Judge评估器（适配LLMClient）
    
    Args:
        llm_client: LLMClient实例
        question: 问题
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        bool: 是否正确
    """
    accuracy_prompt = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    You are an expert grader that determines if answers to questions match a gold standard answer.
    Be generous with your grading - focus on whether the core meaning and facts are correct.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    try:
        # 使用LLMClient的generate_answer方法
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0,
            max_tokens=100,  # 足够短的回复
            json_format=True  # 如果支持JSON格式
        )
        
        # 尝试提取JSON
        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().lower()
                return label == "correct"
            else:
                # 回退到简单文本匹配
                return "correct" in llm_response.lower()
        except json.JSONDecodeError:
            # JSON解析失败，使用文本匹配
            logging.warning(f"JSON解析失败，使用文本匹配: {llm_response}")
            return "correct" in llm_response.lower()
            
    except Exception as e:
        logging.error(f"LLM grader failed: {e}")
        return False

def llm_grader(llm_client: LLMClient, question: str, gold_answer: str, response: str) -> bool:
    """
    同步版本的LLM评估器（适配LLMClient）
    
    Args:
        llm_client: LLMClient实例
        question: 问题
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        bool: 是否正确
    """
    accuracy_prompt = f"""
    You are an expert grader that determines if answers to questions match a gold standard answer.
    Be generous with your grading - focus on whether the core meaning and facts are correct.

    Your task is to evaluate if the generated answer is CORRECT or WRONG compared to the gold answer.

    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    Consider the answer CORRECT if:
    - It contains the same key information as the gold answer
    - The meaning is equivalent even if wording differs
    - For time questions, the time period matches even if format differs
    - For factual questions, the core facts are accurate

    Return only a JSON with "label" key containing either "CORRECT" or "WRONG".
    Example: {{"label": "CORRECT"}} or {{"label": "WRONG"}}
    """

    try:
        # 使用LLMClient的generate_answer方法
        llm_response = llm_client.generate_answer(
            prompt=accuracy_prompt,
            temperature=0.0,  # 确保一致性
            max_tokens=50,    # 足够返回JSON
        )
        
        # 尝试提取JSON
        try:
            if '{' in llm_response and '}' in llm_response:
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                result = json.loads(json_str)
                label = result.get("label", "").strip().upper()
                return label == "CORRECT"
            else:
                # 回退到关键词匹配
                llm_response_upper = llm_response.upper()
                if "CORRECT" in llm_response_upper and "WRONG" not in llm_response_upper:
                    return True
                elif "WRONG" in llm_response_upper and "CORRECT" not in llm_response_upper:
                    return False
                else:
                    # 如果同时包含或都不包含，使用更宽松的匹配
                    return any(word in llm_response.lower() for word in ["correct", "right", "accurate", "yes"])
                    
        except json.JSONDecodeError as e:
            # JSON解析失败，使用文本匹配
            logging.warning(f"JSON解析失败，使用文本匹配: {llm_response}, 错误: {e}")
            llm_response_upper = llm_response.upper()
            
            # 更健壮的文本匹配
            if "CORRECT" in llm_response_upper:
                return True
            elif "WRONG" in llm_response_upper:
                return False
            else:
                # 最后的回退策略
                positive_words = ["correct", "right", "accurate", "yes", "true", "match"]
                negative_words = ["wrong", "incorrect", "false", "no", "mismatch"]
                
                response_lower = llm_response.lower()
                positive_count = sum(1 for word in positive_words if word in response_lower)
                negative_count = sum(1 for word in negative_words if word in response_lower)
                
                return positive_count > negative_count
                
    except Exception as e:
        logging.error(f"LLM grader sync failed: {e}")
        return False

def llm_grader_batch(llm_client: LLMClient, 
                    questions: List[str], 
                    gold_answers: List[str], 
                    responses: List[str]) -> List[bool]:
    """
    批量LLM评估器（适配LLMClient）
    
    Args:
        llm_client: LLMClient实例
        questions: 问题列表
        gold_answers: 标准答案列表
        responses: 生成答案列表
        
    Returns:
        List[bool]: 评估结果列表
    """
    if not (len(questions) == len(gold_answers) == len(responses)):
        raise ValueError("输入列表长度不一致")
    
    results = []
    
    for i, (question, gold_answer, response) in enumerate(zip(questions, gold_answers, responses)):
        logging.debug(f"批量评估 {i+1}/{len(questions)}")
        result = llm_grader(llm_client, question, gold_answer, response)
        results.append(result)
    
    return results

def calculate_llm_judgment(llm_client: LLMClient, 
                        question: str, 
                        gold_answer: str, 
                        response: str,
                        num_runs: int = 1) -> Dict[str, Any]:
    """
    计算LLM判断分数（适配LLMClient）
    
    Args:
        llm_client: LLMClient实例
        question: 问题
        gold_answer: 标准答案
        response: 生成的答案
        num_runs: 运行次数（用于一致性检查）
        
    Returns:
        包含LLM判断结果的字典
    """
    judgments = []
    
    for i in range(num_runs):
        try:
            judgment = llm_grader(llm_client, question, gold_answer, response)
            judgments.append(judgment)
            logging.debug(f"LLM判断 {i+1}/{num_runs}: {judgment}")
        except Exception as e:
            logging.error(f"LLM judgment {i+1} failed: {e}")
            judgments.append(False)
    
    if not judgments:
        return {
            "judgments": [],
            "accuracy": 0.0,
            "num_runs": num_runs,
            "consistency": False,
            "error": "所有判断都失败了"
        }
    
    accuracy = sum(judgments) / len(judgments)
    consistency = len(set(judgments)) == 1  # 所有判断是否一致
    
    return {
        "judgments": judgments,
        "accuracy": accuracy,
        "num_runs": num_runs,
        "consistency": consistency,
        "confidence": "high" if consistency else "low"
    }

# 更新综合评估函数以支持LLMClient
def evaluate_answer_comprehensive(question: str,
                                gold_answer: str,
                                predicted_answer: str,
                                context: str = "",
                                llm_client: Optional[LLMClient] = None,
                                include_llm_judgment: bool = False,
                                evaluation_options: Optional[List[str]] = None,
                                llm_runs: int = 1) -> Dict[str, Any]:
    """
    综合答案评估接口（适配LLMClient）
    
    Args:
        question: 问题
        gold_answer: 标准答案
        predicted_answer: 预测答案
        context: 上下文
        llm_client: LLMClient实例（可选）
        include_llm_judgment: 是否包含LLM判断
        evaluation_options: 评估选项
        llm_runs: LLM评估运行次数
        
    Returns:
        综合评估结果
    """
    if evaluation_options is None:
        evaluation_options = ["lexical", "semantic"]
    
    # 基础指标
    result = calculate_comprehensive_metrics(
        gold_answer, predicted_answer, context, evaluation_options
    )
    
    # LLM判断（如果启用且提供了客户端）
    if include_llm_judgment and llm_client is not None:
        try:
            llm_result = calculate_llm_judgment(
                llm_client, question, gold_answer, predicted_answer, llm_runs
            )
            result["llm_judgment"] = llm_result
        except Exception as e:
            logging.error(f"LLM判断失败: {e}")
            result["llm_judgment"] = {
                "error": str(e),
                "accuracy": 0.0,
                "num_runs": llm_runs,
                "consistency": False
            }
    
    # 转换numpy类型
    result = convert_numpy_types(result)
    
    return result

# 添加测试函数
def test_llm_grader(llm_client: LLMClient):
    """
    测试LLM评估器


    """
    test_cases = [
        {
            "question": "What is Caroline's relationship status?",
            "gold_answer": "single",
            "response": "Caroline is single",
            "expected": True
        },
        {
            "question": "What did they eat for dinner?",
            "gold_answer": "pizza",
            "response": "They had Chinese food",
            "expected": False
        },
        {
            "question": "When did they meet?",
            "gold_answer": "May 7, 2023",
            "response": "They met on 7 May 2023",
            "expected": True
        }
    ]
    
    print("🧪 测试LLM评估器...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}:")
        print(f"问题: {case['question']}")
        print(f"标准答案: {case['gold_answer']}")
        print(f"生成答案: {case['response']}")
        print(f"期望结果: {case['expected']}")
        
        result = llm_grader(
            llm_client, 
            case['question'], 
            case['gold_answer'], 
            case['response']
        )
        
        print(f"实际结果: {result}")
        print(f"匹配期望: {'✅' if result == case['expected'] else '❌'}")
    
    print("\n🎯 批量测试...")
    questions = [case['question'] for case in test_cases]
    gold_answers = [case['gold_answer'] for case in test_cases]
    responses = [case['response'] for case in test_cases]
    
    batch_results = llm_grader_batch(llm_client, questions, gold_answers, responses)
    print(f"批量结果: {batch_results}")

# async def llm_grader(llm_client:LLMClient, question: str, gold_answer: str, response: str) -> bool:
#     """
#     LLM-as-a-Judge评估器（LoCoMo风格）
    
#     Args:
#         llm_client: LLM客户端（OpenAI或其他）
#         question: 问题
#         gold_answer: 标准答案
#         response: 生成的答案
        
#     Returns:
#         bool: 是否正确
#     """
#     system_prompt = """
#     You are an expert grader that determines if answers to questions match a gold standard answer
#     """

#     accuracy_prompt = f"""
#     Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
#         (1) a question (posed by one user to another user),
#         (2) a 'gold' (ground truth) answer,
#         (3) a generated answer
#     which you will score as CORRECT/WRONG.

#     The point of the question is to ask about something one user should know about the other user based on their prior conversations.
#     The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
#     Question: Do you remember what I got the last time I went to Hawaii?
#     Gold answer: A shell necklace
#     The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

#     For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

#     Now it's time for the real question:
#     Question: {question}
#     Gold answer: {gold_answer}
#     Generated answer: {response}

#     First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
#     Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

#     Just return the label CORRECT or WRONG in a json format with the key as "label".
#     """

#     try:
#         response = await llm_client.chat.completions.create(
#             model="gpt-4o-mini",  # 可以根据需要调整模型
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": accuracy_prompt},
#             ],
#             temperature=0,
#         )
#         message_content = response.choices[0].message.content
#         label = json.loads(message_content)["label"]
#         parsed = LLMGrade(llm_judgment=label, llm_reasoning="")
        
#         return parsed.llm_judgment.strip().lower() == "correct"
#     except Exception as e:
#         logging.error(f"LLM grader failed: {e}")
#         return False

# def llm_grader(llm_client:LLMClient, question: str, gold_answer: str, response: str) -> bool:
#     """
#     同步版本的LLM评估器（用于非异步环境）
#     """
#     system_prompt = """
#     You are an expert grader that determines if answers to questions match a gold standard answer.
#     Be generous with your grading - focus on whether the core meaning and facts are correct.
#     """

#     accuracy_prompt = f"""
#     Question: {question}
#     Gold answer: {gold_answer}
#     Generated answer: {response}

#     Rate this as CORRECT or WRONG. Return only a JSON with "label" key.
#     """

#     try:
#         # 这里需要根据实际的LLM客户端调整
#         if hasattr(llm_client, 'generate_answer'):
#             # 适配我们的LLMClient
#             full_prompt = f"{system_prompt}\n\n{accuracy_prompt}"
#             llm_response = llm_client.generate_answer(full_prompt, temperature=0, max_tokens=50)
            
#             # 尝试提取JSON
#             try:
#                 if '{' in llm_response and '}' in llm_response:
#                     start = llm_response.find('{')
#                     end = llm_response.rfind('}') + 1
#                     json_str = llm_response[start:end]
#                     result = json.loads(json_str)
#                     return result.get("label", "").strip().lower() == "correct"
#                 else:
#                     # 回退到简单文本匹配
#                     return "correct" in llm_response.lower()
#             except:
#                 return "correct" in llm_response.lower()
#         else:
#             logging.warning("LLM client not compatible, skipping LLM evaluation")
#             return False
#     except Exception as e:
#         logging.error(f"Sync LLM grader failed: {e}")
#         return False

# ================================
# 现有函数保持不变（已经修复过的）
# ================================

class SimpleTokenizer(object):
    """简单的分词器类"""
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    """文本标准化"""
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    """答案标准化函数（已修复）"""
    if s is None:
        s = ""
    elif not isinstance(s, str):
        s = str(s)
    
    s = s.replace(',', "")
    
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(gold_answer: str, response: str) -> bool:
    """
    精确匹配得分（统一参数顺序）
    Args:
        gold_answer: 标准答案
        response: 生成的答案
    Returns:
        bool: 是否完全匹配
    """
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response = normalize_answer(response)
    gold_answer = normalize_answer(gold_answer)
    return set(response.split()) == set(gold_answer.split())

def calculate_f1_score(gold_answer: str, response: str) -> float:
    """
    F1得分（统一参数顺序）
    Args:
        gold_answer: 标准答案
        response: 生成的答案
    Returns:
        float: F1得分
    """
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    response_tokens = [ps.stem(w) for w in normalize_answer(response).split()]
    gold_answer_tokens = [ps.stem(w) for w in normalize_answer(gold_answer).split()]
    
    common = Counter(response_tokens) & Counter(gold_answer_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(response_tokens)
    recall = 1.0 * num_same / len(gold_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_f1_score_multi(gold_answer: str, response: str) -> float:
    """
    多答案F1得分（统一参数顺序）
    Args:
        gold_answer: 标准答案（可以是多个答案，用逗号分隔）
        response: 生成的答案（可以是多个答案，用逗号分隔）
    Returns:
        float: 多答案F1得分
    """
    response = str(response) if response is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    
    responses = [r.strip() for r in response.split(',')]
    gold_answers = [g.strip() for g in gold_answer.split(',')]
    
    return np.mean([max([calculate_f1_score(ga, resp) for resp in responses]) for ga in gold_answers])

# ================================
# 更新相关的向后兼容函数
# ================================

# def ems(prediction, ground_truths):
#     """保持向后兼容的多答案精确匹配函数"""
#     prediction = str(prediction) if prediction is not None else ""
    
#     safe_ground_truths = []
#     for gt in ground_truths:
#         if gt is not None:
#             safe_ground_truths.append(str(gt))
#         else:
#             safe_ground_truths.append("")
    
#     return max([exact_match_score(gt, prediction) for gt in safe_ground_truths])

# ================================
# 更新 eval_question_answering 函数中的调用
# ================================

def eval_question_answering(qas, eval_key='prediction', metric='f1'):
    """问答任务评估（更新函数调用）"""
    all_ems = []
    all_recall = []
    
    for i, line in enumerate(qas):
        if type(line[eval_key]) == list:
            answer = str(line['answer']) if line['answer'] is not None else ""
        else:
            answer = str(line['answer']) if line['answer'] is not None else ""
            
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = str(line[eval_key]) if line[eval_key] is not None else ""
        
        if line['category'] in [2, 3, 4]:
            all_ems.append(calculate_f1_score(answer, output))  # 更新参数顺序
        elif line['category'] in [1]:
            all_ems.append(calculate_f1_score_multi(answer, output))  # 更新参数顺序和函数名
        elif line['category'] in [5]:
            output_lower = output.lower()
            if 'no information available' in output_lower or 'not mentioned' in output_lower:
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            raise ValueError(f"未知的问题类别: {line['category']}")
        
        assert i+1 == len(all_ems)

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} 个QA样本已评估; {} 个准确率值".format(len(qas), len(all_ems)))
    return all_ems, 0.0, all_recall

def eval_dialogue_system(infile):
    """对话系统评估（更新函数调用）"""
    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(calculate_f1_score(answer, output))  # 更新参数顺序
        rl_scores.append(calculate_bleu_score(answer, output))  # 更新参数顺序
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens

def calculate_comprehensive_metrics(gold_answer: str, 
                                  response: str, 
                                  context: str = "", 
                                  options: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    计算综合评估指标（更新函数调用）
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        context: 上下文
        options: 评估选项 ["lexical", "semantic"]
        
    Returns:
        包含各种评估指标的字典
    """
    if options is None:
        options = ["lexical", "semantic"]

    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {
        "context_tokens": len(nltk.word_tokenize(context)) if context else 0,
        "response_tokens": len(nltk.word_tokenize(response)),
        "gold_tokens": len(nltk.word_tokenize(gold_answer))
    }

    if "lexical" in options:
        metrics["lexical"] = {}
        
        # 基础指标 - 更新参数顺序
        metrics["lexical"]["exact_match"] = float(exact_match_score(gold_answer, response))
        metrics["lexical"]["token_f1"] = calculate_f1_score(gold_answer, response)
        
        # ROUGE指标
        rouge_scores = calculate_rouge_score(gold_answer, response)
        metrics["lexical"].update(rouge_scores)
        
        # BLEU指标
        bleu_scores = calculate_bleu_score(gold_answer, response)
        metrics["lexical"].update(bleu_scores)
        
        # METEOR指标
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_answer, response)

    if "semantic" in options:
        metrics["semantic"] = {}
        
        # 语义相似度
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(gold_answer, response)
        
        # BERT F1
        metrics["semantic"]["bert_f1"] = calculate_bert_f1_score(gold_answer, response)

    return metrics


# ================================
# 新增和改进的评分函数（参考locomo_eval）
# ================================

def calculate_rouge_score(gold_answer: str, response: str) -> Dict[str, float]:
    """
    计算ROUGE分数（使用rouge_score库，更准确）
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        Dict包含rouge1_f, rouge2_f, rougeL_f
    """
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        logging.error(f"Failed to calculate ROUGE scores: {e}")
    
    return metrics

def calculate_bleu_score(gold_answer: str, response: str) -> Dict[str, float]:
    """
    计算BLEU分数
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        Dict包含bleu1, bleu2, bleu3, bleu4
    """
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except Exception as e:
        logging.error(f"Failed to calculate BLEU scores: {e}")

    return metrics

def calculate_meteor_score(gold_answer: str, response: str) -> float:
    """
    计算METEOR分数
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        METEOR分数
    """
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    try:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        logging.error(f"Failed to calculate METEOR score: {e}")
        return 0.0

def calculate_semantic_similarity(gold_answer: str, response: str) -> float:
    """
    计算语义相似度
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        语义相似度分数 (0-1)
    """
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    try:
        sentence_model = get_sentence_model()
        if sentence_model is None:
            return 0.0
            
        gold_embedding = sentence_model.encode([gold_answer], show_progress_bar=False)[0]
        response_embedding = sentence_model.encode([response], show_progress_bar=False)[0]
        return 1 - cosine(gold_embedding, response_embedding)
    except Exception as e:
        logging.error(f"Failed to calculate semantic similarity: {e}")
        return 0.0

def calculate_bert_f1_score(gold_answer: str, response: str) -> float:
    """
    计算BERT F1分数
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        
    Returns:
        BERT F1分数
    """
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    
    try:
        _, _, f1 = score([response], [gold_answer], lang="en", rescale_with_baseline=True, verbose=False)
        return f1.item() if f1 is not None else 0.0
    except Exception as e:
        logging.error(f"Failed to calculate BERT F1 score: {e}")
        return 0.0

# def calculate_token_f1_score(gold_answer: str, response: str) -> float:
#     """
#     计算基于token的F1分数（与原f1_score相同但名称更明确）
    
#     Args:
#         gold_answer: 标准答案
#         response: 生成的答案
        
#     Returns:
#         Token-level F1分数
#     """
#     return calculate_f1_score(response, gold_answer)

# ================================
# 综合评估函数
# ================================

def calculate_comprehensive_metrics(gold_answer: str, 
                                  response: str, 
                                  context: str = "", 
                                  options: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    计算综合评估指标（参考locomo_eval的calculate_nlp_metrics）
    
    Args:
        gold_answer: 标准答案
        response: 生成的答案
        context: 上下文
        options: 评估选项 ["lexical", "semantic"]
        
    Returns:
        包含各种评估指标的字典
    """
    if options is None:
        options = ["lexical", "semantic"]

    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {
        "context_tokens": len(nltk.word_tokenize(context)) if context else 0,
        "response_tokens": len(nltk.word_tokenize(response)),
        "gold_tokens": len(nltk.word_tokenize(gold_answer))
    }

    if "lexical" in options:
        metrics["lexical"] = {}
        
        # 基础指标
        metrics["lexical"]["exact_match"] = float(exact_match_score(response, gold_answer))
        metrics["lexical"]["token_f1"] = calculate_f1_score(gold_answer, response)
        
        # ROUGE指标
        rouge_scores = calculate_rouge_score(gold_answer, response)
        metrics["lexical"].update(rouge_scores)
        
        # BLEU指标
        bleu_scores = calculate_bleu_score(gold_answer, response)
        metrics["lexical"].update(bleu_scores)
        
        # METEOR指标
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_answer, response)

    if "semantic" in options:
        metrics["semantic"] = {}
        
        # 语义相似度
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(gold_answer, response)
        
        # BERT F1
        metrics["semantic"]["bert_f1"] = calculate_bert_f1_score(gold_answer, response)

    return metrics

def calculate_llm_judgment(llm_client, 
                         question: str, 
                         gold_answer: str, 
                         response: str,
                         num_runs: int = 1) -> Dict[str, Any]:
    """
    计算LLM判断分数
    
    Args:
        llm_client: LLM客户端
        question: 问题
        gold_answer: 标准答案
        response: 生成的答案
        num_runs: 运行次数
        
    Returns:
        包含LLM判断结果的字典
    """
    judgments = []
    
    for i in range(num_runs):
        try:
            judgment = llm_grader(llm_client, question, gold_answer, response)
            judgments.append(judgment)
        except Exception as e:
            logging.error(f"LLM judgment {i+1} failed: {e}")
            judgments.append(False)
    
    return {
        "judgments": judgments,
        "accuracy": sum(judgments) / len(judgments) if judgments else 0.0,
        "num_runs": num_runs,
        "consistency": len(set(judgments)) == 1  # 是否所有判断一致
    }

# ================================
# 工具函数
# ================================

def convert_numpy_types(obj):
    """
    转换numpy类型为Python原生类型（用于JSON序列化）
    """
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

def ems(prediction, ground_truths):
    """保持向后兼容的多答案精确匹配函数"""
    prediction = str(prediction) if prediction is not None else ""
    
    safe_ground_truths = []
    for gt in ground_truths:
        if gt is not None:
            safe_ground_truths.append(str(gt))
        else:
            safe_ground_truths.append("")
    
    return max([exact_match_score(prediction, gt) for gt in safe_ground_truths])

# ================================
# 原有文件级评估函数保持不变
# ================================

def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """检查文档是否包含答案字符串"""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def check_answer(example, tokenizer) -> List[bool]:
    """在所有顶部文档中搜索答案"""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []
    for _, doc in enumerate(ctxs):
        text = doc['text']
        if text is None:
            hits.append(False)
            continue
        hits.append(has_answer(answers, text, tokenizer))

    return hits

def eval_recall(infile):
    """评估召回率"""
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens

def eval_question_answering(qas, eval_key='prediction', metric='f1'):
    """问答任务评估（保持原有逻辑）"""
    all_ems = []
    all_recall = []
    
    for i, line in enumerate(qas):
        if type(line[eval_key]) == list:
            answer = str(line['answer']) if line['answer'] is not None else ""
        else:
            answer = str(line['answer']) if line['answer'] is not None else ""
            
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = str(line[eval_key]) if line[eval_key] is not None else ""
        
        if line['category'] in [2, 3, 4]:
            all_ems.append(calculate_f1_score(output, answer))
        elif line['category'] in [1]:
            all_ems.append(calculate_f1_score(output, answer))
        elif line['category'] in [5]:
            output_lower = output.lower()
            if 'no information available' in output_lower or 'not mentioned' in output_lower:
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            raise ValueError(f"未知的问题类别: {line['category']}")
        
        assert i+1 == len(all_ems)

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} 个QA样本已评估; {} 个准确率值".format(len(qas), len(all_ems)))
    return all_ems, 0.0, all_recall

def eval_fact_checking(infile):
    """事实检查任务评估"""
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens

def eval_dialogue_system(infile):
    """对话系统评估"""
    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(calculate_f1_score(output, answer))
        rl_scores.append(calculate_bleu_score(output, [answer]))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens

# ================================
# 新增：简化的评估接口
# ================================

def evaluate_answer_comprehensive(question: str,
                                gold_answer: str,
                                predicted_answer: str,
                                context: str = "",
                                llm_client=None,
                                include_llm_judgment: bool = False,
                                evaluation_options: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    综合答案评估接口
    
    Args:
        question: 问题
        gold_answer: 标准答案
        predicted_answer: 预测答案
        context: 上下文
        llm_client: LLM客户端（可选）
        include_llm_judgment: 是否包含LLM判断
        evaluation_options: 评估选项
        
    Returns:
        综合评估结果
    """
    if evaluation_options is None:
        evaluation_options = ["lexical", "semantic"]
    
    # 基础指标
    result = calculate_comprehensive_metrics(
        gold_answer, predicted_answer, context, evaluation_options
    )
    
    # LLM判断（如果启用）
    if include_llm_judgment and llm_client is not None:
        llm_result = calculate_llm_judgment(
            llm_client, question, gold_answer, predicted_answer
        )
        result["llm_judgment"] = llm_result
    
    # 转换numpy类型
    result = convert_numpy_types(result)
    
    return result

# ================================
# 使用示例
# ================================

def example_usage():
    """使用示例"""
    question = "What is Caroline's relationship status?"
    gold_answer = "single"
    predicted_answer = "Based on the conversation, Caroline appears to be single."
    
    # 基础评估
    basic_result = evaluate_answer_comprehensive(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        evaluation_options=["lexical", "semantic"]
    )
    
    print("Basic evaluation result:")
    print(json.dumps(basic_result, indent=2))
    
    # 使用LLM评估（需要LLM客户端）
    # llm_result = evaluate_answer_comprehensive(
    #     question=question,
    #     gold_answer=gold_answer,
    #     predicted_answer=predicted_answer,
    #     llm_client=your_llm_client,
    #     include_llm_judgment=True
    # )

# 使用示例
def example_llm_grader_usage():
    """LLM评估器使用示例"""
    
    # 创建LLMClient
    llm_client = LLMClient("deepseek-chat")  # 或者使用其他模型
    
    # 单个评估
    question = "What is Caroline's job?"
    gold_answer = "psychologist"
    predicted_answer = "She works as a therapist and counselor."
    
    # 基础LLM评估
    is_correct = llm_grader(llm_client, question, gold_answer, predicted_answer)
    print(f"LLM评估结果: {is_correct}")
    
    # 多次运行的一致性检查
    llm_judgment = calculate_llm_judgment(
        llm_client, question, gold_answer, predicted_answer, num_runs=3
    )
    print(f"LLM判断详情: {llm_judgment}")
    
    # 综合评估（包含LLM判断）
    comprehensive_result = evaluate_answer_comprehensive(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        llm_client=llm_client,
        include_llm_judgment=True,
        llm_runs=2
    )
    print(f"综合评估结果: {json.dumps(comprehensive_result, indent=2)}")

if __name__ == "__main__":
    # 运行测试
    print("🔍 测试LLM评估器...")
    client = LLMClient("deepseek-chat")
    test_llm_grader(client)
    
    # 运行示例
    example_llm_grader_usage()
    example_usage()
