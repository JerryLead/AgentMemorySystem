import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import os
from bert_score import score
from nltk.stem import PorterStemmer
from rouge import Rouge

# 初始化词干提取器，用于标准化词汇比较
ps = PorterStemmer()

# 长度阈值，用于某些评估逻辑
LENGTH_THRESHOLD = 5

class SimpleTokenizer(object):
    """
    简单的分词器类，用于文本标记化
    支持Unicode字符，包括多语言文本的处理
    """
    # 正则表达式模式：匹配字母、数字、标记字符
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    # 正则表达式模式：匹配非空白字符
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        初始化分词器
        Args:
            annotators: None 或空集合（仅进行分词）
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        """
        对文本进行分词
        Args:
            text: 输入文本
            uncased: 是否转换为小写
        Returns:
            tokens: 分词结果列表
        """
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """
    在所有顶部文档中搜索，查看它们是否包含任何答案
    Args:
        example: 包含答案和上下文的示例
        tokenizer: 分词器实例
    Returns:
        hits: 每个文档是否包含答案的布尔值列表
    """
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # 由于某种原因无法找到文档
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """
    检查文档是否包含答案字符串
    Args:
        answers: 候选答案列表
        text: 要搜索的文本
        tokenizer: 分词器实例
    Returns:
        bool: 是否找到答案
    """
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        # 滑动窗口搜索答案
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    """
    文本标准化：使用Unicode NFD标准化
    Args:
        text: 输入文本
    Returns:
        标准化后的文本
    """
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    """
    答案标准化函数：去除冠词、标点符号，转换为小写等
    Args:
        s: 输入字符串
    Returns:
        标准化后的字符串
    """
    s = s.replace(',', "")
    
    def remove_articles(text):
        """移除冠词和连词"""
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        """修复空白字符"""
        return ' '.join(text.split())

    def remove_punc(text):
        """移除标点符号"""
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        """转换为小写"""
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    精确匹配得分：比较预测答案和标准答案是否完全匹配
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
    Returns:
        bool: 是否精确匹配
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # 使用集合比较，忽略词序
    return set(prediction.split()) == set(ground_truth.split())


def bert_score(prediction, ground_truth):
    """
    使用BERT Score评估预测答案质量
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
    Returns:
        float: BERT F1得分
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    return max(0, F1[0].item())


def ems(prediction, ground_truths):
    """
    多个标准答案的精确匹配得分
    Args:
        prediction: 预测答案
        ground_truths: 标准答案列表
    Returns:
        float: 最高精确匹配得分
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    """
    计算F1得分：基于词级别的精确率和召回率
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
    Returns:
        float: F1得分
    """
    # 使用词干提取进行标准化
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    
    # 计算词汇重叠
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    # 计算精确率、召回率和F1得分
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def f1(prediction, ground_truth):
    """
    多答案F1得分：处理逗号分隔的多个答案
    Args:
        prediction: 预测答案（可能包含多个答案，逗号分隔）
        ground_truth: 标准答案（可能包含多个答案，逗号分隔）
    Returns:
        float: 平均F1得分
    """
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    
    # 对每个标准答案，找到最佳匹配的预测答案，然后求平均
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    """
    计算ROUGE-L得分
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
    Returns:
        float: ROUGE-L F1得分
    """
    rouge = Rouge()
    
    # 使用词干提取进行标准化
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    """
    多个标准答案的ROUGE-L得分
    Args:
        prediction: 预测答案
        ground_truths: 标准答案列表
    Returns:
        float: 最高ROUGE-L得分
    """
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## 文件级别的评估函数
def eval_recall(infile):
    """
    评估召回率：检查输出中是否包含答案
    Args:
        infile: 输入文件路径
    Returns:
        tuple: (召回率, 平均答案长度)
    """
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]  # 跳过头部行

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
    """
    问答任务评估：根据不同类别使用不同的评估策略
    Args:
        qas: 问答数据列表
        eval_key: 预测结果的键名
        metric: 评估指标类型
    Returns:
        tuple: (得分列表, 平均长度, 召回率列表)
    """
    all_ems = []        # 存储所有评估得分
    all_recall = []     # 存储所有召回率得分
    exact_match_count = 0
    f1_count = 0
    answer_lengths = []
    
    for i, line in enumerate(qas):
        # 处理答案格式
        if type(line[eval_key]) == list:
            answer = line['answer']
        else:
            answer = str(line['answer'])
            
        # 类别3的特殊处理：只取分号前的部分
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line[eval_key]
        
        # 根据不同类别采用不同的评估策略
        # 类别2,3,4: 单跳、时间、开放域评估，不拆分子答案
        if line['category'] in [2, 3, 4]:
            all_ems.append(f1_score(output, answer))
        
        # 类别1: 多跳评估，将整个短语拆分为子答案，计算每个的部分F1
        elif line['category'] in [1]:
            all_ems.append(f1(output, answer))

        # 类别5: 对抗性评估，检查是否选择了正确选项
        elif line['category'] in [5]:
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                all_ems.append(1)
            else:
                all_ems.append(0)
        else:
            print(line)
            raise ValueError("未知的问题类别")
        
        # 确保索引一致性
        assert i+1 == len(all_ems), all_ems

        # 计算召回率（如果有上下文和证据）
        if eval_key + '_context' in line and len(line['evidence']) > 0:
            # 对话的召回率准确性
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} 个QA样本已评估; {} 个准确率值".format(len(qas), len(all_ems)))
    lens = 0.0
    return all_ems, lens, all_recall


def eval_fact_checking(infile):
    """
    事实检查任务评估
    Args:
        infile: 输入文件路径
    Returns:
        tuple: (精确匹配率, 平均答案长度)
    """
    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        # 标准化事实检查答案
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
    """
    对话系统评估
    Args:
        infile: 输入文件路径
    Returns:
        tuple: (F1得分, ROUGE-L得分, 平均答案长度)
    """
    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens