# 任务评估工具 (Task Evaluation Tools)

本目录提供了完整的语义图谱问答评估框架，专门用于评测基于对话记忆的QA系统性能。支持独立样本评估、语义图谱检索、多种评分指标和详细的结果分析。

## 🎯 核心功能

- 🧠 **语义图谱QA评估**：基于语义图谱的智能问答评估
- 🔍 **多策略检索**：支持单跳、多跳、时间、开放域、对抗性检索策略
- 📊 **多维度评分**：F1、ROUGE-L、精确匹配、证据召回等多种指标
- 🎯 **独立样本评估**：每个样本独立建立语义图谱，避免数据干扰
- 📈 **详细分析报告**：按类别、按问题、按样本的详细统计分析
- ⚡ **快速测试脚本**：支持单问题快速验证和调试
- 🔄 **批量处理**：支持多样本、多模型的批量评估
- 📁 **多格式输出**：JSON、CSV、TXT等多种格式的结果文件

## 📁 目录结构

```
benchmark/task_eval/
├── evaluation.py                      # 核心评分函数库
├── semantic_graph_qa_evaluator.py     # 语义图谱QA评估器
├── IndependentSampleEvaluator.py      # 独立样本评估器
├── run_detailed_evaluation.py         # 详细评估执行脚本
├── run_semantic_evaluation.py         # 语义图谱评估脚本
├── test_light_script.py              # 快速测试脚本
├── locomo_test.py                     # LoCoMo完整测试（已有）
├── locomo_test_split.py              # LoCoMo分组测试（已有）
├── locomo_test_split.py              # LoCoMo分割测试（已有）
├── readme.md                         # 本文档
└── results/                          # 评估结果目录
    ├── detailed/                     # 详细评估结果
    ├── simplified/                   # 简化评估结果
    └── semantic/                     # 语义图谱评估结果
```

## 🚀 快速开始

### 1. 环境准备

确保已完成实体关系抽取：

```bash
# 运行实体关系抽取（如果尚未完成）
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch-all \
    --use-chunking \
    --output benchmark/results/full_extraction

# 制作数据集
python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/full_extraction \
    --output-dir benchmark/dataset/locomo/extraction
```

### 2. 快速测试单个问题

```bash
cd /home/zyh/code/AgentMemorySystem
python benchmark/task_eval/test_light_script.py
```

选择测试类型：
- 1: 单问题测试 - 快速验证系统功能
- 2: 类别策略测试 - 测试不同QA类别的处理策略

### 3. 独立样本评估（推荐）

```bash
# 评估单个样本
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model deepseek-chat \
    --samples conv-26 \
    --format comprehensive

# 评估多个样本
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model deepseek-chat \
    --samples conv-26 conv-30 conv-41 \
    --format comprehensive

# 详细评估脚本（推荐）
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat \
    --samples conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/detailed
```

### 4. 语义图谱评估

```bash
python benchmark/task_eval/run_semantic_evaluation.py \
    --model deepseek-chat \
    --conversations conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/semantic
```

## 📚 详细使用指南

### 核心评分函数 (evaluation.py)

提供标准的评分函数，兼容LoCoMo评估协议：

#### 主要函数

| 函数 | 用途 | 说明 |
|------|------|------|
| `f1_score(prediction, ground_truth)` | 词级别F1分数 | 基于词汇重叠的F1计算 |
| `f1(prediction, ground_truth)` | 多答案F1分数 | 处理逗号分隔的多答案 |
| `rougel_score(prediction, ground_truth)` | ROUGE-L分数 | 最长公共子序列评分 |
| `exact_match_score(prediction, ground_truth)` | 精确匹配 | 完全匹配评分 |
| `eval_question_answering(qas, eval_key)` | 问答评估 | 按类别使用不同评分策略 |

#### 评分策略

- **类别1（多跳推理）**: 使用 `f1()` 多答案F1分数
- **类别2-4（单跳/时间/开放域）**: 使用 `f1_score()` 标准F1分数
- **类别5（对抗性问题）**: 检查是否正确拒绝回答

#### 使用示例

```python
from benchmark.task_eval.evaluation import f1_score, f1, eval_question_answering

# 单个问题评分
score = f1_score("Alice is a doctor", "Alice works as a doctor")
print(f"F1 Score: {score:.4f}")

# 多答案评分
multi_score = f1("Alice, Bob", "Alice, Bob, Charlie")
print(f"Multi F1 Score: {multi_score:.4f}")

# 批量评估
qa_data = [
    {"question": "Who is Alice?", "category": 2, "answer": "doctor", "prediction": "Alice is a doctor"},
    {"question": "What does Bob do?", "category": 1, "answer": "teacher, coach", "prediction": "teacher"}
]
scores, lengths, recalls = eval_question_answering(qa_data, eval_key="prediction")
```

### 语义图谱QA评估器 (semantic_graph_qa_evaluator.py)

核心评估引擎，实现智能检索和答案生成。

#### 检索策略

| 类别 | 策略名 | 检索配置 | 说明 |
|------|---------|----------|------|
| 1 | multi_hop | top_k=8, 扩展邻居 | 多跳推理，扩展相关节点 |
| 2 | single_hop | top_k=5, 无扩展 | 直接事实查询 |
| 3 | temporal | top_k=6, 时间优先 | 时间相关信息检索 |
| 4 | open_domain | top_k=10, 全面检索 | 开放域问题，广泛检索 |
| 5 | adversarial | top_k=3, 保守检索 | 对抗性问题，谨慎回答 |

#### 基本使用

```python
from benchmark.task_eval.semantic_graph_qa_evaluator import SemanticGraphQAEvaluator
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage
from benchmark.llm_utils.llm_client import LLMClient

# 初始化
storage = ConversationSemanticStorage()
llm_client = LLMClient(model_name="deepseek-chat")
evaluator = SemanticGraphQAEvaluator(storage.semantic_graph, llm_client)

# 检索上下文
context = evaluator.retrieve_context_for_question(
    question="What LGBTQ support group did Caroline attend?",
    category=2,
    conversation_id="conv-26",
    evidence=["D10:3"]
)

# 生成答案
answer_result = evaluator.generate_answer(context)
print(f"Generated Answer: {answer_result['generated_answer']}")

# 评估单个对话
qa_data = [{"question": "...", "answer": "...", "category": 2}]
results = evaluator.evaluate_single_conversation("conv-26", qa_data)
```

#### 高级功能

**自定义检索策略**：
```python
# 修改检索配置
evaluator.retrieval_config[1] = {
    "top_k": 10, 
    "include_relations": True, 
    "expand_neighbors": True
}
```

**批量评估**：
```python
qa_test_data = {
    "conv-26": [qa1, qa2, qa3],
    "conv-30": [qa4, qa5, qa6]
}
full_results = evaluator.run_full_evaluation(qa_test_data, save_results=True)
```

### 独立样本评估器 (IndependentSampleEvaluator.py)

确保每个样本独立建立语义图谱的评估器，避免数据泄露。

#### 核心特性

- **独立语义图谱**：每个样本创建独立的图谱和存储
- **完整评分指标**：F1、ROUGE-L、精确匹配、召回率等
- **详细结果记录**：问题级别、类别级别、样本级别统计
- **简化输出格式**：清晰的结果结构

#### 基本命令

```bash
python benchmark/task_eval/IndependentSampleEvaluator.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--model` | str | ❌ | LLM模型名称 (默认: deepseek-chat) |
| `--samples` | list | ❌ | 要评估的样本ID列表 |
| `--output_dir` | str | ❌ | 输出目录路径 |
| `--format` | str | ❌ | 输出格式 (comprehensive/simple) |
| `--debug` | flag | ❌ | 启用调试日志 |

#### 使用示例

**1. 单样本评估**
```bash
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model deepseek-chat \
    --samples conv-26 \
    --format comprehensive \
    --output_dir benchmark/task_eval/results/test
```

**2. 多样本批量评估**
```bash
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model deepseek-chat \
    --samples conv-26 conv-30 conv-41 \
    --format comprehensive \
    --output_dir benchmark/task_eval/results/batch
```

**3. 不同模型对比**
```bash
# DeepSeek评估
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model deepseek-chat --samples conv-26 \
    --output_dir results/deepseek

# GPT评估
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model gpt-4o-mini --samples conv-26 \
    --output_dir results/gpt4o
```

#### 编程接口

```python
from benchmark.task_eval.IndependentSampleEvaluator import IndependentSampleEvaluator
from benchmark.llm_utils.llm_client import LLMClient

# 初始化
llm_client = LLMClient(model_name="deepseek-chat")
evaluator = IndependentSampleEvaluator(llm_client, output_dir="results")

# 评估单个样本
result = evaluator.evaluate_single_sample("conv-26")
if "error" not in result:
    summary = result["summary"]["overall"]
    print(f"Average F1 Score: {summary['avg_f1_score']:.4f}")
    print(f"Total Questions: {summary['total_questions']}")

# 评估多个样本
results = evaluator.evaluate_multiple_samples(
    sample_ids=["conv-26", "conv-30"],
    save_format="comprehensive"
)
```

### 详细评估脚本 (run_detailed_evaluation.py)

高级评估脚本，提供汇总分析和验证功能。

#### 主要功能

- **汇总分析**：跨样本的统计分析
- **多格式输出**：JSON、CSV、TXT格式结果
- **评估验证**：使用evaluation.py进行结果验证
- **自动发现样本**：支持评估所有可用样本

#### 基本命令

```bash
python benchmark/task_eval/run_detailed_evaluation.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--model` | str | ❌ | LLM模型名称 |
| `--samples` | list | ❌ | 样本ID列表 |
| `--output_dir` | str | ❌ | 输出目录 |
| `--all_samples` | flag | ❌ | 评估所有可用样本 |

#### 使用示例

**1. 指定样本评估**
```bash
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat \
    --samples conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/detailed
```

**2. 全样本评估**
```bash
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat \
    --all_samples \
    --output_dir benchmark/task_eval/results/full
```

**3. 多模型对比**
```bash
# DeepSeek Chat
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat --samples conv-26 conv-30

# DeepSeek Reasoner
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-reasoner --samples conv-26 conv-30

# GPT-4o-mini
python benchmark/task_eval/run_detailed_evaluation.py \
    --model gpt-4o-mini --samples conv-26 conv-30
```

#### 输出文件

生成的文件包括：

```
benchmark/task_eval/results/detailed/
├── evaluation_conv-26_20250708_143025.json       # 样本详细结果
├── evaluation_conv-30_20250708_143156.json       # 样本详细结果
├── cross_sample_summary_20250708_143300.json     # 跨样本汇总
├── evaluation_report_20250708_143300.txt         # 可读性报告
├── all_questions_detailed_20250708_143300.csv    # 问题级别CSV
├── sample_metrics_summary_20250708_143300.json   # 样本指标汇总
├── evaluation_dataset_20250708_143300.json       # evaluation.py兼容数据
└── evaluation_validation_20250708_143300.json    # 验证结果
```

### 快速测试脚本 (test_light_script.py)

用于快速验证和调试的轻量级测试工具。

#### 功能模式

**1. 单问题测试**
- 测试单个预设问题的完整流程
- 显示检索上下文、生成答案、评分结果
- 用于功能验证和调试

**2. 类别策略测试**
- 测试不同QA类别的处理策略
- 对比各类别的检索和生成效果
- 用于策略优化

#### 使用方法

```bash
python benchmark/task_eval/test_light_script.py
```

根据提示选择测试模式：
- 输入 `1`: 单问题测试
- 输入 `2`: 类别策略测试

#### 示例输出

```
🧪 Semantic Graph QA Testing
1. 单问题测试
2. 类别策略测试

选择测试类型 (1 或 2): 1

🔍 测试问题: What LGBTQ support group did Caroline attend?
📝 标准答案: Connected LGBTQ Activists
🏷️ 类别: 2

1️⃣ 检索相关上下文...
✅ 检索到 5 个节点
✅ 检索到 0 个关系
📊 证据召回率: 0.80

📋 检索到的主要内容:
  [1] dialog: Caroline mentioned joining a local LGBTQ support group called "Connected LGBTQ...
  [2] summary: Caroline discussed her involvement with LGBTQ activism...
  [3] event: Caroline attended her first LGBTQ support group meeting...

2️⃣ 生成答案...
🤖 生成答案: Connected LGBTQ Activists

3️⃣ 评估结果...
📊 F1分数: 1.0000
📊 精确匹配: 1
```

### 语义图谱评估脚本 (run_semantic_evaluation.py)

专门用于语义图谱QA评估的完整脚本。

#### 主要功能

- **完整评估流程**：从数据加载到结果保存
- **多对话支持**：同时评估多个对话
- **详细统计分析**：按类别、按对话的详细分析
- **可视化输出**：人类可读的评估报告

#### 基本命令

```bash
python benchmark/task_eval/run_semantic_evaluation.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--model` | str | ❌ | LLM模型名称 |
| `--conversations` | list | ❌ | 对话ID列表 |
| `--output_dir` | str | ❌ | 输出目录 |
| `--load_existing` | flag | ❌ | 加载已有语义图谱 |
| `--debug` | flag | ❌ | 启用调试日志 |

#### 使用示例

**1. 基本评估**
```bash
python benchmark/task_eval/run_semantic_evaluation.py \
    --model deepseek-chat \
    --conversations conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/semantic
```

**2. 调试模式**
```bash
python benchmark/task_eval/run_semantic_evaluation.py \
    --model deepseek-chat \
    --conversations conv-26 \
    --debug \
    --output_dir benchmark/task_eval/results/debug
```

**3. 使用已有图谱**
```bash
python benchmark/task_eval/run_semantic_evaluation.py \
    --model deepseek-chat \
    --conversations conv-26 conv-30 \
    --load_existing \
    --output_dir benchmark/task_eval/results/existing
```

## 📊 输出文件详细说明

### 评估结果文件结构

#### 1. 样本详细结果 (`evaluation_*.json`)

```json
{
  "evaluation_summary": {
    "sample_id": "conv-26",
    "evaluation_timestamp": "2025-01-08T14:30:15.123456",
    "llm_model": "deepseek-chat",
    "overall": {
      "total_questions": 45,
      "avg_f1_score": 0.3245,
      "avg_f1_multi": 0.3156,
      "avg_rouge_l": 0.3892,
      "avg_exact_match": 0.1556,
      "avg_main_score": 0.3245,
      "avg_evidence_recall": 0.7234,
      "avg_context_nodes": 6.2,
      "avg_context_relations": 2.8
    },
    "by_category": {
      "category_1": {
        "category_name": "multi_hop",
        "questions_count": 12,
        "avg_f1_score": 0.2156,
        "avg_main_score": 0.2156,
        "std_main_score": 0.1245
      }
    }
  },
  "questions": [
    {
      "question": "When did Caroline go to the LGBTQ support group?",
      "category": 2,
      "category_name": "single_hop",
      "answer": "7 May 2023",
      "prediction": "Caroline went to the LGBTQ support group yesterday.",
      "scores": {
        "f1_score": 0.2456,
        "f1_multi": 0.2456,
        "rouge_l": 0.3123,
        "rouge_l_multi": 0.3123,
        "exact_match": 0.0,
        "main_score": 0.2456
      },
      "retrieval": {
        "strategy": "single_hop",
        "nodes": [
          {
            "uid": "node_123",
            "data_type": "event",
            "content": "Caroline attended an LGBTQ support group...",
            "score": 0.8756
          }
        ],
        "relations": [],
        "raw_context": "=== CONVERSATION CONTEXT ===\n[1] EVENT: ...",
        "nodes_count": 5,
        "relations_count": 0
      },
      "evidence": ["D1:3"],
      "evidence_recall": 0.8,
      "prompt": "Based on the following conversation context...",
      "context_nodes_count": 5,
      "context_relations_count": 0
    }
  ]
}
```

#### 2. 跨样本汇总 (`cross_sample_summary_*.json`)

包含所有样本的汇总统计，支持多样本对比分析。

#### 3. 问题级别CSV (`all_questions_detailed_*.csv`)

| Sample_ID | Question | Category | Strategy | Ground_Truth | Prediction | F1_Score | Evidence_Recall | Context_Nodes | Evidence_Count |
|-----------|----------|----------|----------|--------------|------------|----------|-----------------|---------------|----------------|
| conv-26 | When did Caroline... | 2 | single_hop | 7 May 2023 | yesterday | 0.2456 | 0.8000 | 5 | 1 |

#### 4. 评估验证 (`evaluation_validation_*.json`)

使用evaluation.py验证评估结果的一致性。

### 文件命名规范

- `evaluation_{sample_id}_{timestamp}.json`: 单样本详细结果
- `cross_sample_summary_{timestamp}.json`: 跨样本汇总
- `evaluation_report_{timestamp}.txt`: 可读性报告
- `all_questions_detailed_{timestamp}.csv`: 问题级别结果
- `evaluation_dataset_{timestamp}.json`: 标准格式数据集

## 💡 评估策略说明

### QA类别和策略映射

| 类别 | 名称 | 策略特点 | 评分方法 | 适用场景 |
|------|------|----------|----------|----------|
| 1 | Multi-hop | 扩展邻居节点，关系推理 | 多答案F1 | 需要多步推理的复杂问题 |
| 2 | Single-hop | 直接检索，无扩展 | 标准F1 | 直接事实查询 |
| 3 | Temporal | 时间信息优先 | 标准F1 | 时间相关问题 |
| 4 | Open-domain | 广泛检索，全面分析 | 标准F1 | 开放性问题 |
| 5 | Adversarial | 保守检索，拒绝回答 | 拒绝检测 | 对抗性问题，测试系统边界 |

### 评分指标说明

#### 主要指标

- **F1 Score**: 词级别的精确率和召回率调和平均
- **F1 Multi**: 处理多答案的F1分数（逗号分隔）
- **ROUGE-L**: 基于最长公共子序列的评分
- **Exact Match**: 完全精确匹配
- **Evidence Recall**: 证据检索召回率

#### 选择策略

- **类别1**: 使用F1 Multi作为主要分数
- **类别2-4**: 使用标准F1作为主要分数
- **类别5**: 使用拒绝检测逻辑

### 检索策略配置

#### 可调参数

```python
retrieval_config = {
    1: {"top_k": 8, "include_relations": True, "expand_neighbors": True},
    2: {"top_k": 5, "include_relations": False, "expand_neighbors": False},
    3: {"top_k": 6, "include_relations": True, "expand_neighbors": False},
    4: {"top_k": 10, "include_relations": True, "expand_neighbors": True},
    5: {"top_k": 3, "include_relations": False, "expand_neighbors": False}
}
```

#### 优化建议

- **多跳推理**: 增加top_k和邻居扩展
- **单跳查询**: 减少检索数量，提高精确性
- **时间问题**: 优先检索时间相关节点
- **对抗性**: 减少检索，避免过度推测

## 🔧 性能优化和调试

### 性能优化建议

#### 1. 内存优化
```bash
# 限制样本数量
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --samples conv-26 conv-30 \
    --format simple

# 分批处理
for sample in conv-26 conv-30 conv-41; do
    python benchmark/task_eval/IndependentSampleEvaluator.py \
        --samples $sample --output_dir results/batch_$sample
done
```

#### 2. 速度优化
```bash
# 使用较小的检索数量
# 在代码中修改retrieval_config的top_k值

# 禁用复杂特性
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --samples conv-26 --format simple
```

#### 3. 模型选择
- **快速测试**: deepseek-chat (较快)
- **高质量**: deepseek-reasoner (较慢但推理能力强)
- **对比基准**: gpt-4o-mini

### 调试和故障排除

#### 1. 启用详细日志
```bash
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --samples conv-26 --debug
```

#### 2. 快速功能验证
```bash
python benchmark/task_eval/test_light_script.py
```

#### 3. 检查数据完整性
```bash
# 验证实体抽取结果
ls benchmark/results/full_extraction/

# 验证数据集制作结果
ls benchmark/dataset/locomo/extraction/

# 检查QA数据格式
python -c "
import json
with open('benchmark/dataset/locomo/locomo10.json') as f:
    data = json.load(f)
    sample = data[0]
    print('Sample keys:', sample.keys())
    if 'qa' in sample:
        print('QA keys:', sample['qa'][0].keys())
"
```

#### 4. 常见问题解决

**问题1: 检索结果为空**
```bash
# 检查语义图谱是否正确构建
python -c "
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage
storage = ConversationSemanticStorage()
storage.store_conversation('conv-26')
print(f'节点数: {len(storage.semantic_graph.semantic_map.memory_units)}')
"
```

**问题2: LLM生成失败**
```bash
# 测试LLM连接
python -c "
from benchmark.llm_utils.llm_client import LLMClient
client = LLMClient('deepseek-chat')
result = client.generate_answer('Test question')
print(f'Result: {result}')
"
```

**问题3: 评分异常**
```bash
# 验证评分函数
python -c "
from benchmark.task_eval.evaluation import f1_score, f1
print('F1 Score:', f1_score('test answer', 'test answer'))
print('F1 Multi:', f1('test, answer', 'test, answer'))
"
```

## 📈 完整工作流程示例

### 从零开始的完整评估流程

```bash
# 1. 设置环境
cd /home/zyh/code/AgentMemorySystem
export DEEPSEEK_API_KEY="your-api-key"

# 2. 数据准备（如果需要）
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch-all --use-chunking \
    --output benchmark/results/full_extraction

python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/full_extraction \
    --output-dir benchmark/dataset/locomo/extraction

# 3. 快速功能测试
python benchmark/task_eval/test_light_script.py

# 4. 单样本详细评估
python benchmark/task_eval/IndependentSampleEvaluator.py \
    --model deepseek-chat \
    --samples conv-26 \
    --format comprehensive \
    --output_dir benchmark/task_eval/results/single

# 5. 多样本批量评估
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat \
    --samples conv-26 conv-30 conv-41 \
    --output_dir benchmark/task_eval/results/batch

# 6. 全样本评估（如果需要）
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat \
    --all_samples \
    --output_dir benchmark/task_eval/results/full

# 7. 结果分析
cat benchmark/task_eval/results/batch/evaluation_report_*.txt
```

### 多模型对比评估

```bash
# DeepSeek Chat
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-chat \
    --samples conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/deepseek_chat

# DeepSeek Reasoner
python benchmark/task_eval/run_detailed_evaluation.py \
    --model deepseek-reasoner \
    --samples conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/deepseek_reasoner

# GPT-4o-mini
python benchmark/task_eval/run_detailed_evaluation.py \
    --model gpt-4o-mini \
    --samples conv-26 conv-30 \
    --output_dir benchmark/task_eval/results/gpt4o_mini

# 结果对比
echo "=== Model Comparison ==="
for model in deepseek_chat deepseek_reasoner gpt4o_mini; do
    echo "--- $model ---"
    grep "平均分数" benchmark/task_eval/results/$model/evaluation_report_*.txt
done
```

### 定制化评估流程

```bash
# 针对特定类别的评估
python -c "
import json
from benchmark.task_eval.IndependentSampleEvaluator import IndependentSampleEvaluator
from benchmark.llm_utils.llm_client import LLMClient

# 过滤特定类别的问题
# 在这里可以添加自定义的过滤和评估逻辑
"

# 针对特定策略的优化
# 修改retrieval_config后重新评估
```

## 📋 输出分析和解读

### 关键指标解读

#### 总体性能指标
- **avg_main_score**: 主要评分（根据类别选择最适合的评分方法）
- **avg_f1_score**: 平均F1分数
- **avg_exact_match**: 精确匹配率
- **avg_evidence_recall**: 证据召回率

#### 按类别性能
- **category_X_avg_score**: 各类别平均分数
- **category_X_count**: 各类别问题数量
- **category_X_std_score**: 各类别分数标准差

#### 检索质量指标
- **avg_context_nodes**: 平均检索节点数
- **avg_context_relations**: 平均检索关系数
- **retrieval_strategy**: 使用的检索策略

### 结果解读建议

#### 良好性能指标
- **F1 Score > 0.4**: 较好的答案质量
- **Evidence Recall > 0.7**: 良好的检索质量
- **Exact Match > 0.2**: 可接受的精确性

#### 问题诊断
- **F1 Score < 0.2**: 可能是检索质量差或LLM生成问题
- **Evidence Recall < 0.5**: 检索策略需要优化
- **High Std Score**: 性能不稳定，需要参数调优

## 🔗 相关工具和依赖

### 核心依赖
- **实体关系抽取**: [`benchmark/extractor/`](../extractor/) - 提供语义图谱构建
- **LLM工具**: [`benchmark/llm_utils/`](../llm_utils/) - 提供模型接口
- **语义图谱**: [`dev/semantic_graph.py`](../../dev/semantic_graph.py) - 核心图谱结构

### 数据依赖
- **原始数据**: `benchmark/dataset/locomo/locomo10.json`
- **抽取结果**: `benchmark/dataset/locomo/extraction/`
- **实体关系**: `benchmark/results/full_extraction/`

### 环境要求
- Python 3.8+
- DeepSeek API Key 或 OpenAI API Key
- 充足的内存（推荐16GB+用于大规模评估）

## 📄 版本历史

- **v1.0**: 基础语义图谱QA评估
- **v1.1**: 添加独立样本评估
- **v1.2**: 多策略检索和评分
- **v1.3**: 详细分析和报告生成
- **v1.4**: 快速测试和调试工具
- **v1.5**: 多模型支持和对比评估
- **v1.6**: 完整工作流程和性能优化

## 🤝 贡献指南

### 开发环境设置
```bash
git clone <repository-url>
cd AgentMemorySystem
pip install -r requirements.txt
```

### 添加新的评估策略
1. 在 `SemanticGraphQAEvaluator` 中添加新的 `category_strategies`
2. 更新 `retrieval_config` 配置
3. 在 `_build_prompt` 中添加对应的提示词模板
4. 更新评分逻辑

### 测试新功能
```bash
python benchmark/task_eval/test_light_script.py
```

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues]
- 邮箱: zhangyuhan25@otcaix.iscas.ac.cn

---

*最后更新: 2025年7月8日*