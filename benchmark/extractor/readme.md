# 实体关系抽取工具 (Entity Relation Extractor)

本工具集提供了从对话文本中抽取实体和关系的完整解决方案，特别针对LoCoMo数据集进行了优化。支持单个样本处理、批量处理、知识图谱构建和数据集制作。

## 🎯 核心功能

- 🤖 **LLM驱动的实体关系抽取**：使用大语言模型进行智能抽取
- 📊 **结构化输出**：支持JSON格式的结构化响应解析
- 🎯 **单样本处理**：针对LoCoMo数据集中的单个sample_id进行处理
- 🔄 **批量处理**：支持多个样本的批量处理
- 🕸️ **知识图谱构建**：构建完整的NetworkX知识图谱
- 💾 **多种存储格式**：支持SemanticMap和NetworkX图结构保存
- 📈 **详细统计**：提供完整的处理和抽取统计信息
- 🗂️ **数据集制作**：将抽取结果整合成统一的数据集文件
- 🔍 **智能分块**：支持长文本的智能分块处理
- 📋 **位置追踪**：追踪抽取内容在原文中的具体位置

## 📁 目录结构

```
benchmark/extractor/
├── entity_relation_extractor.py      # 核心实体关系抽取器
├── semantic_graph_integrator.py      # 语义图集成器
├── locomo_entity_extractor.py        # LoCoMo专用实体抽取器
├── dataset_inserter.py              # 对话语义存储器
├── dataset_maker.py                 # 数据集制作器
├── extract_entities_cli.py          # 命令行工具
├── example.py                       # 使用示例
├── quick_sample_test.py             # 快速测试脚本
└── readme.md                        # 本文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖：

```bash
pip install sentence-transformers
pip install networkx
pip install faiss-cpu  # 或 faiss-gpu
pip install numpy pandas
pip install tiktoken
```

### 2. 设置环境变量

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# 或者在根目录创建 .env 文件
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENAI_API_KEY=your-openai-api-key
```

### 3. 基本使用

#### 查看可用样本
```bash
cd /home/zyh/code/AgentMemorySystem
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --list-samples
```

#### 处理单个样本
```bash
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --sample-id conv-26 \
    --use-chunking \
    --save-intermediate \
    --output benchmark/results/conv26_extraction
```

#### 快速测试
```bash
python benchmark/extractor/example.py
```

## 📚 详细使用指南

### LoCoMo实体抽取器 (locomo_entity_extractor.py)

专门用于处理LoCoMo数据集的实体关系抽取，支持位置追踪和智能分块。

#### 基本命令

```bash
python benchmark/extractor/locomo_entity_extractor.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--dataset` | str | ✅ | LoCoMo数据集路径 |
| `--sample-id` | str | ❌ | 要处理的样本ID |
| `--list-samples` | flag | ❌ | 列出所有可用的样本ID |
| `--batch` | list | ❌ | 批量处理指定的样本ID |
| `--batch-all` | flag | ❌ | 批量处理所有样本 |
| `--use-chunking` | flag | ❌ | 使用智能分块处理长文本 |
| `--no-chunking` | flag | ❌ | 禁用分块处理 |
| `--save-intermediate` | flag | ❌ | 保存中间结果 |
| `--output` | str | ❌ | 输出目录 |

#### 使用示例

**1. 列出所有样本**
```bash
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --list-samples
```

**2. 处理单个样本（推荐）**
```bash
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --sample-id conv-26 \
    --use-chunking \
    --save-intermediate \
    --output benchmark/results/conv26_full_extraction
```

**3. 批量处理指定样本**
```bash
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch conv-26 conv-30 conv-41 \
    --use-chunking \
    --output benchmark/results/batch_extraction
```

**4. 批量处理所有样本**
```bash
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch-all \
    --use-chunking \
    --output benchmark/results/full_extraction
```

### 数据集制作器 (dataset_maker.py)

用于将所有样本的实体关系抽取结果整合成统一的数据集文件。

#### 基本功能

- **收集抽取结果**：自动扫描抽取结果目录，收集所有样本数据
- **实体关系汇总**：统计实体和关系的频率、类型分布等
- **跨样本分析**：识别共同实体和关系模式
- **多格式输出**：生成完整版、轻量级版本和统计报告
- **样本索引**：创建便于查询的样本索引文件

#### 基本命令

```bash
python benchmark/extractor/dataset_maker.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--results-dir` | str | ✅ | 抽取结果目录路径 |
| `--output-dir` | str | ❌ | 输出目录路径 (默认: benchmark/dataset) |
| `--output-prefix` | str | ❌ | 输出文件前缀 (默认: locomo_extracted) |

#### 使用示例

**1. 基本用法**
```bash
python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/full_extraction
```

**2. 指定输出目录和前缀**
```bash
python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/full_extraction \
    --output-dir benchmark/dataset/locomo/extraction \
    --output-prefix locomo_extracted
```

**3. 完整工作流示例**
```bash
# 1. 先进行实体关系抽取
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch-all \
    --use-chunking \
    --output benchmark/results/fixed_full_extraction

# 2. 然后制作数据集
python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/fixed_full_extraction \
    --output-dir benchmark/dataset/locomo/extraction \
    --output-prefix locomo_extracted
```

#### 输出文件说明

数据集制作器会生成以下文件：

```
benchmark/dataset/locomo/extraction/
├── locomo_extracted_full_dataset.json           # 完整数据集
├── locomo_extracted_entity_relationship_summary.json  # 实体关系汇总
├── locomo_extracted_sample_index.json           # 样本索引
├── locomo_extracted_lightweight.json            # 轻量级数据集
└── locomo_extracted_statistics_report.txt       # 统计报告
```

**文件详细说明：**

| 文件 | 描述 | 用途 |
|------|------|------|
| `*_full_dataset.json` | 包含所有样本的完整抽取结果 | 主要数据集文件 |
| `*_entity_relationship_summary.json` | 实体关系的统计汇总和跨样本分析 | 数据分析和质量评估 |
| `*_sample_index.json` | 样本快速索引，包含基本统计信息 | 快速查询和选择样本 |
| `*_lightweight.json` | 移除了source_text等大字段的轻量版本 | 快速加载和预览 |
| `*_statistics_report.txt` | 人类可读的统计报告 | 数据质量报告 |

### 对话语义存储器 (dataset_inserter.py)

用于将LoCoMo数据存储到语义图谱中，支持原始数据和抽取结果的混合存储。

#### 基本用法

```bash
# 测试所有存储模式
python benchmark/extractor/dataset_inserter.py

# 或单独测试某种模式
python -c "
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage
storage = ConversationSemanticStorage()
stats = storage.store_conversation('conv-26')
print(stats)
"
```

#### 编程接口

```python
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage

# 1. 创建存储器
storage = ConversationSemanticStorage()

# 2. 存储单个对话
stats = storage.store_conversation(
    sample_id="conv-26",
    include_raw=True,
    include_extracted=True
)

# 3. 获取QA测试数据
qa_test_data = storage.get_qa_test_data(["conv-26"])

# 4. 存储所有对话
all_stats = storage.store_all_conversations()
```

### 命令行通用工具 (extract_entities_cli.py)

通用的实体关系抽取命令行工具。

#### 基本命令

```bash
python benchmark/extractor/extract_entities_cli.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--mode` | str | ✅ | 抽取模式：locomo/text/conv26 |
| `--input` | str | ✅ | 输入：数据集路径或文本内容 |
| `--output` | str | ❌ | 输出目录路径 |
| `--model` | str | ❌ | LLM模型名称 (默认deepseek-chat) |
| `--sample-limit` | int | ❌ | 限制处理的样本数量 |
| `--log-level` | str | ❌ | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

#### 使用示例

**1. 处理单个文本**
```bash
python benchmark/extractor/extract_entities_cli.py \
    --mode text \
    --input "Caroline住在纽约，是心理咨询师。Melanie住在加州，是艺术家。"
```

**2. 专门处理conv-26**
```bash
python benchmark/extractor/extract_entities_cli.py \
    --mode conv26 \
    --input benchmark/dataset/locomo/locomo10.json \
    --output benchmark/results/conv26_extraction
```

## 📂 输出文件详细说明

### 单样本处理输出

每个样本处理完成后，会在输出目录生成以下文件结构：

```
{sample_id}_entity_extraction_{timestamp}/
├── extraction.log                                   # 处理日志
├── {sample_id}_full_conversation.txt               # 完整对话文本
├── {sample_id}_semantic_graph/                     # 语义图谱目录
│   ├── semantic_map_data/                         # SemanticMap数据
│   │   ├── memory_units.pkl                       # 内存单元数据
│   │   ├── memory_spaces.pkl                      # 内存空间数据
│   │   └── faiss_index.faiss                      # FAISS索引
│   ├── semantic_graph.pkl                         # NetworkX图结构
│   └── management_state.pkl                       # 内存管理状态
├── {sample_id}_extraction_results.json            # 抽取结果JSON
├── {sample_id}_detailed_analysis.json             # 详细分析结果
└── {sample_id}_summary_report.txt                 # 摘要报告
```

### 关键文件说明

| 文件 | 格式 | 说明 |
|------|------|------|
| `semantic_graph.pkl` | Pickle | NetworkX图结构，包含所有节点和边 |
| `semantic_map_data/` | 目录 | SemanticMap的完整状态数据 |
| `*_extraction_results.json` | JSON | 抽取的实体、关系和统计信息 |
| `*_detailed_analysis.json` | JSON | 详细的网络分析和实体关系分析 |
| `*_summary_report.txt` | Text | 人类可读的处理摘要报告 |
| `extraction.log` | Text | 详细的处理日志和错误信息 |

## 💡 编程接口使用

### 基本实体关系抽取

```python
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.extractor.entity_relation_extractor import EntityRelationExtractor

# 初始化
llm_client = LLMClient(model_name="deepseek-chat")
extractor = EntityRelationExtractor(llm_client)

# 抽取实体和关系
text = "Alice住在纽约，是一名医生。Bob住在加州，是她的朋友。"
entities, relationships, keywords = extractor.extract_entities_and_relations(text)

print(f"实体: {len(entities)} 个")
print(f"关系: {len(relationships)} 个")
print(f"关键词: {keywords}")
```

### LoCoMo专用处理器

```python
from benchmark.extractor.locomo_entity_extractor import LoCoMoEntityExtractor

# 初始化处理器
extractor = LoCoMoEntityExtractor("benchmark/dataset/locomo/locomo10.json")

# 列出可用样本
samples = extractor.list_available_samples()
print(f"可用样本: {samples}")

# 获取样本信息
info = extractor.get_sample_info("conv-26")
print(f"样本信息: {info}")

# 处理单个样本
result = extractor.extract_entities_and_relations_for_sample(
    sample_id="conv-26",
    use_chunking=True,
    save_intermediate=True
)

print(f"处理结果: {result['extraction_results']}")
```

### 语义图集成

```python
from dev.semantic_graph import SemanticGraph
from benchmark.extractor.semantic_graph_integrator import SemanticGraphIntegrator

# 创建语义图
graph = SemanticGraph()
integrator = SemanticGraphIntegrator(graph)

# 处理内存单元进行实体抽取
result = integrator.process_memory_unit_for_entities(unit)

# 批量处理
batch_results = integrator.batch_extract_entities_from_space(
    space_name="locomo_dialogs",
    max_units=50
)
```

## ⚙️ 配置说明

### LLM配置

支持多种LLM模型：

```python
# DeepSeek (推荐)
llm_client = LLMClient(model_name="deepseek-chat")
llm_client = LLMClient(model_name="deepseek-reasoner")

# OpenAI GPT
llm_client = LLMClient(model_name="gpt-4o-mini")
llm_client = LLMClient(model_name="gpt-4o")
```

### 实体类型配置

支持的实体类型：
- `person`: 人物
- `organization`: 组织机构
- `geo`: 地理位置
- `event`: 事件
- `category`: 类别/概念

### 关系类型配置

支持的关系类型：
- `FAMILY`: 家庭关系
- `WORK`: 工作关系
- `FRIEND`: 朋友关系
- `LOCATION`: 位置关系
- `TEMPORAL`: 时间关系
- `TOPIC`: 主题关系
- `RELATED_TO`: 一般关联关系

### 分块处理配置

```python
# 智能分块配置
max_tokens = 60000  # 最大token数
strategy = "intelligent"  # 分块策略

# 自定义分块
extractor.smart_text_chunking(text, max_tokens=50000)
```

## 🔧 性能优化建议

### 处理大量数据时

1. **使用分块处理**：启用`--use-chunking`参数处理长文本
2. **分批处理**：使用`--batch`指定部分样本，避免一次处理过多
3. **保存中间结果**：使用`--save-intermediate`防止意外中断
4. **调整日志级别**：使用`--log-level WARNING`减少日志输出

### 内存优化

```bash
# 限制处理样本数
python benchmark/extractor/locomo_entity_extractor.py \
    --batch conv-26 conv-30 \
    --use-chunking \
    --output results/small_batch

# 不保存中间结果（节省磁盘空间）
python benchmark/extractor/locomo_entity_extractor.py \
    --sample-id conv-26 \
    --use-chunking
```

### API调用优化

- 合理设置处理间隔，避免API限流
- 使用较小的样本集进行测试，确认效果后再扩大规模
- 监控API使用量和成本

## 🐛 故障排除

### 常见问题

**1. 模块导入错误**
```bash
ModuleNotFoundError: No module named 'benchmark'
```
解决方案：确保在项目根目录下运行命令
```bash
cd /home/zyh/code/AgentMemorySystem
```

**2. 数据集路径错误**
```bash
ValueError: 数据集加载失败
```
解决方案：检查数据集文件路径是否正确
```bash
ls benchmark/dataset/locomo/locomo10.json
```

**3. LLM API错误**
检查API密钥和网络连接：
```bash
echo $DEEPSEEK_API_KEY
echo $OPENAI_API_KEY
```

**4. 内存不足**
减少批量处理的样本数量，或使用分块处理。

**5. 抽取结果为空**
- 检查输入文本是否有效
- 确认LLM模型是否可用
- 查看日志文件了解详细错误信息

### 调试模式

开启详细日志进行调试：
```bash
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --sample-id conv-26 \
    --save-intermediate
```

查看日志文件：
```bash
tail -f benchmark/results/*/extraction.log
```

### 数据验证

验证抽取结果的质量：
```bash
# 1. 检查抽取结果文件
cat benchmark/results/conv-26_*/conv-26_extraction_results.json | jq '.extraction_statistics'

# 2. 查看统计报告
cat benchmark/results/conv-26_*/conv-26_summary_report.txt

# 3. 验证数据集制作结果
cat benchmark/dataset/locomo/extraction/locomo_extracted_statistics_report.txt
```

## 📈 完整工作流程示例

### 从零开始的完整处理流程

```bash
# 1. 设置环境
cd /home/zyh/code/AgentMemorySystem
export DEEPSEEK_API_KEY="your-api-key"

# 2. 查看可用样本
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --list-samples

# 3. 测试单个样本
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --sample-id conv-26 \
    --use-chunking \
    --save-intermediate \
    --output benchmark/results/test_extraction

# 4. 批量处理所有样本
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch-all \
    --use-chunking \
    --output benchmark/results/full_extraction

# 5. 制作数据集
python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/full_extraction \
    --output-dir benchmark/dataset/locomo/extraction \
    --output-prefix locomo_extracted

# 6. 验证结果
cat benchmark/dataset/locomo/extraction/locomo_extracted_statistics_report.txt
```

### 针对特定需求的定制流程

```bash
# 仅处理特定样本集
python benchmark/extractor/locomo_entity_extractor.py \
    --dataset benchmark/dataset/locomo/locomo10.json \
    --batch conv-26 conv-30 conv-41 \
    --use-chunking \
    --output benchmark/results/selected_samples

# 制作定制数据集
python benchmark/extractor/dataset_maker.py \
    --results-dir benchmark/results/selected_samples \
    --output-dir benchmark/dataset/custom \
    --output-prefix custom_extracted
```

## 🔗 相关工具

- **任务评估**: [`benchmark/task_eval/`](../task_eval/) - 使用抽取的数据进行QA评估
- **语义图谱**: [`dev/semantic_graph.py`](../../dev/semantic_graph.py) - 核心语义图谱实现
- **LLM工具**: [`benchmark/llm_utils/`](../llm_utils/) - LLM客户端和工具

## 📄 版本历史

- **v1.0**: 基础实体关系抽取功能
- **v1.1**: 添加JSON格式输出支持
- **v1.2**: LoCoMo专用处理器
- **v1.3**: 批量处理和知识图谱构建
- **v1.4**: 智能分块和位置追踪
- **v1.5**: 数据集制作器和完整工作流程
- **v1.6**: 对话语义存储和查询功能

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本工具。

### 开发环境设置

```bash
git clone <repository-url>
cd AgentMemorySystem
pip install -r requirements.txt
```

### 测试

运行快速测试确保功能正常：
```bash
python benchmark/extractor/example.py
```

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues]
- 邮箱: zhangyuhan25@otcaix.iscas.ac.cn

---

*最后更新: 2025年7月8日*