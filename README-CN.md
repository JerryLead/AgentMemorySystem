# Agent Memory System

<img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="版本" />
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="许可证" />
<img src="https://img.shields.io/badge/python-3.8+-orange.svg" alt="Python版本" />

基于语义数据结构的增强记忆系统，通过显式结构关系和隐式语义关联实现多模态知识表示和复杂高阶推理。

## 🌟 项目概述

Agent Memory System（AMS） 是一个高级记忆系统，将语义向量数据库与图数据库的功能相结合，打造一种更加智能、灵活的知识管理和检索方案。本系统通过 SemanticMap（语义地图）组织结构化数据，并在此基础上构建 SemanticGraph（语义图）以建立实体间的显式和隐式关联。

特点：

- **混合记忆结构**：结合向量化语义记忆和结构化关系图表示
- **多跳复杂查询**：支持沿着关系链进行多层次知识推理
- **显式与隐式边共存**：同时表达确定性关系和语义相似性关联
- **多模态支持**：处理文本、代码、图像等多种数据类型
- **插件化扩展**：支持自定义嵌入策略和搜索算法

## 🧩 系统架构

### 核心组件

#### 1. MemoryUnit

最基本的记忆单元，封装具体数据实体：

```python
unit = MemoryUnit(
    uid="issue_1618",  
    raw_data={"title": "Fix memory leak", "description": "System crashes after running for 24h"},
    metadata={"created_by": "user123", "tags": ["bug", "critical"]}
)
```

特性：
- 唯一标识符管理
- 原始数据与元数据分离
- 自动计算数据指纹
- 版本历史追踪
- 灵活的语义向量表示

#### 2. MemorySpace

按类型组织MemoryUnit的命名空间：

```python
smap.create_namespace(
    ms_name="github_issues",
    ms_type="github/issue",
    embedding_fields=["title", "description", "comments"]
)
```

特性：
- 类型化数据组织
- 配置化嵌入字段
- 高效检索索引
- 模式验证支持

#### 3. SemanticMap

管理多个MemorySpace并提供语义检索功能：

```python
# 创建全局语义地图
smap = SemanticMap()

# 注册自定义嵌入策略
smap.set_embedding_strategy("code", code_strategy)

# 插入记忆单元
smap.insert_unit("github_issues", issue_unit)

# 语义搜索
results = smap.find_similar_units("memory leak fixes", ms_names=["github_issues"], top_k=5)
```

特性：
- 多模态嵌入模型支持
- 跨空间语义检索
- 自定义嵌入策略
- 高效反向索引

#### 4. SemanticGraph

在SemanticMap基础上构建关系图网络：

```python
# 创建语义图
sgraph = SemanticGraph(smap)

# 添加显式结构边
sgraph._add_explicit_edge(
    source="pr_423",
    target="issue_1618",
    rel_type="PR修复Issue",
    metadata={"commit_hash": "a1b2c3d"}
)

# 生成隐式语义边
sgraph.infer_implicit_edges("github_issues", "github_code", similarity_threshold=0.9)

# 关系查询
relations = sgraph.find_relations(target="issue_1618", rel_type="PR修复Issue")
```

特性：
- 双边（显式/隐式）图结构
- 带权关系表示
- 多跳路径查询
- 语义相似度自动推断

## 💡 高级应用场景

### 1. [GitHub仓库增强记忆系统](example/issue_manager)

本项目的一个实际应用案例是构建GitHub仓库的增强记忆系统，将仓库中的各类信息（issues、PRs、commits、贡献者、代码文件）组织成一个知识图谱，支持复杂查询和洞察发现。

#### 数据组织

```
SemanticMap (smap)
├── MemorySpace: github_issues
├── MemorySpace: github_prs
├── MemorySpace: github_commits
├── MemorySpace: github_contributors
└── MemorySpace: github_code

SemanticGraph (sgraph)
├── 显式结构边:
│   ├── 开发者提交/审查commit
│   ├── commit修改代码文件
│   ├── 仓库的commit对应唯一PR
│   ├── commit按时间排序串为主链
│   ├── PR修复Issue
│   ├── 开发者提交/审查PR
│   ├── PR包含commit串链
│   └── PR修改代码文件
└── 隐式语义边:
    └── 语义相似度>0.9的实体间连接
```

#### 高阶应用示例

##### 1. [开发者专业知识画像](example/issue_manager/developer_profile.py)

结合显式和隐式边，构建开发者的专业技能图谱：

```json
{
  "developer_id": "dev_789",
  "creation_date": "2025-04-28",
  "core_expertise": {
    "primary_domains": ["图数据库", "搜索引擎", "缓存系统"],
    "expertise_level": "高级",
    "technical_strengths": ["性能优化", "系统设计", "并发控制"],
    "preferred_technologies": ["Python", "C++", "Redis"]
  },
  "code_proficiency": {
    "mastered_modules": ["内存管理", "索引器", "查询优化器"],
    "contribution_areas": ["核心引擎", "API设计", "测试框架"],
    "code_quality_traits": ["高效", "可读性强", "全面错误处理"],
    "complexity_handling": "能有效分解复杂系统，设计清晰的模块接口"
  },
   ...
  "confidence_score": 0.94
}
```

##### 2. [Issue修复模式分析](example/issue_manager/issue_fix_pattern.py)

结合显式和隐式边，分析bug修复的完整路径：

```json
{
  "修复策略": "重构存储工厂的参数传递机制，采用动态kwargs过滤模式，各存储实现类自行提取所需参数。",
  "修复步骤": [
    "将create_blob_storage的参数改为**kwargs接收所有配置参数",
    "在blob存储实现内部通过kwargs.get()提取所需参数",
    "更新工厂类文档说明参数传递机制",
    "添加集成测试验证各存储类型实例化",
    "增加Windows平台条件跳过cosmosdb测试"
  ],
  "修改的关键组件": [
    {
      "文件路径": "graphrag/storage/blob_pipeline_storage.py",
      "功能描述": "Azure Blob存储实现",
      "修改内容": "将函数签名改为**kwargs，内部通过kwargs提取connection_string等参数"
    },
    {
      "文件路径": "graphrag/storage/factory.py",
      "功能描述": "存储工厂基类",
      "修改内容": "添加文档说明参数传递策略，明确各实现类自行处理kwargs参数"
    },
    {
      "文件路径": "tests/integration/storage/test_factory.py",
      "功能描述": "存储工厂集成测试",
      "修改内容": "新增测试用例验证blob/cosmosdb/file/memory存储的实例化，包含参数传递测试"
    },
     ...
  ]
}
```


### 2. [学术组会智能管理系统](example/acdemic_group_meeting)
通过语义图实现学术讨论的全生命周期管理，支持自动会议纪要生成、知识关联检索和智能问答。

#### 数据组织
```
AcademicSemanticGraph
├── 节点类型:
│   ├── 参与者节点: 教授/博士生/硕士生
│   ├── 研究主题节点: 包含领域关键词和技术路线
│   ├── 讨论节点: 发言内容与时间戳
│   └── 附件节点: 论文/演示文稿/数据集
└── 关系类型:
    ├── 发言: 参与者 → 讨论节点
    ├── 引用: 讨论节点 → 论文节点
    ├── 总结: 讨论节点 → 结论节点
    └── 关联: 跨领域语义相似性连接
```

#### 核心功能
- **智能议程生成**：基于历史讨论自动规划会议议题
- **实时知识图谱**：动态构建讨论内容与学术资源的关联网络
- **多模态纪要输出**：
  ```python
  # 自动生成结构化会议纪要
  meeting_summary = agent.generate_summary(
      include=["关键结论", "待解决问题", "参考文献"],
      format="markdown"
  )
  ```
- **跨会议溯源**：通过时间线视图追踪研究思路演进
- **强大的数据库连接能力**：可选连接neo4j和milvus，分别存储知识图谱和对话信息

#### 使用指南

  这个示例包含一个前端，您可以很方便地通过网页访问并使用

- **环境配置**:
  ```bash
  pip install requirements.txt

- **运行前端**:
  ```bash
  cd ./example/academic_group_meeting/front_end
  streamlit run front_end.py

### 3. [Arxiv论文智能问答系统](example/arxiv_QA_system)

基于语义图的Arxiv论文分析系统，通过多层次语义表示和结构化查询规划，实现对科学论文的深度理解和复杂问答。

#### 数据组织

```
ArxivSemanticGraph
├── 节点类型:
│   ├── 根节点: {paper_id}_title_authors (标题与作者信息)
│   ├── 摘要节点: {paper_id}_abstract
│   ├── 章节节点: {paper_id}_chapter_{idx}
│   ├── 段落节点: {chapter_key}_paragraph_{para_idx}
│   ├── 图片节点: {chapter_key}_photo_{photo_idx}
│   └── 表格行节点: {paper_id}_table_{table_idx}_row_{row_idx}
└── 关系类型:
    ├── has_abstract: 论文 → 摘要
    ├── has_chapter: 论文 → 章节
    ├── has_paragraph: 章节 → 段落
    ├── has_photo: 章节 → 图片
    └── has_table_row: 论文 → 表格行
```

#### 系统架构

ArxivSemanticGraph与ArxivAgent组成的强大问答系统：

1. **数据采集与处理**
   - 从Arxiv下载论文HTML文件
   - 解析HTML提取结构化内容（标题、作者、摘要、章节、图片、表格）
   - 构建语义图表示论文知识

2. **多级查询处理**
   - 用户自然语言问题分析与优化
   - LLM生成结构化查询计划
   - 多步查询执行引擎
   - 结果汇总与答案生成

3. **增强功能**
   - 用户偏好记录与推荐
   - 语义相似度评估
   - 多步推理路径

#### 查询计划示例

```json
[
  {
    "step": 1,
    "target": ["paper"],
    "constraints": {
      "semantic_query": "papers about machine learning",
      "filter": {}
    },
    "input": null
  },
  {
    "step": 2,
    "target": ["title"],
    "constraints": {
      "semantic_query": "",
      "filter": {}
    },
    "input": [1]
  }
]
```

#### 高级查询能力

比传统RAG系统更强大的功能：

1. **结构化+语义化混合查询**
   - 可同时利用论文结构信息和语义相似性
   - 例如：找出与某主题相关的论文中具体章节

2. **多步推理**
   - 从论文找到作者→检索作者的其他论文→分析研究趋势
   - 对比不同论文在相同主题上的方法差异

3. **偏好学习**
   - 记录用户喜欢/不喜欢的论文
   - 基于历史偏好提供个性化推荐

4. **深度内容理解**
   - 从摘要到章节到段落的层次化表示
   - 支持针对论文特定部分的精确查询

5. **跨文档推理**
   - 比较不同论文对同一问题的处理方法
   - 综合多篇论文的信息回答复杂问题

#### 示例应用

```python
# 创建Arxiv论文语义图
graph = ArxivSemanticGraph()

# 解析论文并插入到图中
decoded_info = decode_html("arxiv_paper.html")
graph.insert(
    paper_id="2304.01234",
    title=decoded_info['title'],
    authors=decoded_info['authors'],
    abstract=decoded_info['abstract'],
    chapters=decoded_info['chapters'],
    references=decoded_info['references'],
    tables=decoded_info['tables']
)

# 创建问答代理
agent = ArxivAgent(graph, api_key="your-api-key")

# 复杂问题查询
question = "比较最新的两篇关于大型语言模型的论文在训练方法上的差异"
answer = agent.structured_semantic_query(question)
```

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基本用法

```python
from semantic_graph_memory import SemanticMap, SemanticGraph, MemoryUnit

# 创建语义地图
smap = SemanticMap()

# 创建命名空间
smap.create_namespace(
    ms_name="documents", 
    ms_type="text/document",
    embedding_fields=["title", "content"]
)

# 添加记忆单元
doc_unit = MemoryUnit(
    uid="doc_001",
    raw_data={
        "title": "向量数据库简介",
        "content": "向量数据库是一种专门存储高维向量的数据库系统..."
    }
)
smap.insert_unit("documents", doc_unit)

# 创建语义图
sgraph = SemanticGraph(smap)

# 添加显式关系
sgraph._add_explicit_edge(
    source="doc_001",
    target="doc_002",
    rel_type="references",
    metadata={"context": "技术背景"}
)

# 推断隐式关系
sgraph.infer_implicit_edges("documents", similarity_threshold=0.8)

# 进行显式查询
results = sgraph.find_relations(source="doc_001", rel_type="references")
```

### GitHub仓库集成示例

```python
# 创建GitHub仓库记忆系统
smap = SemanticMap()

# 创建相关命名空间
smap.create_namespace("github_issues", "github/issue", ["title", "body", "comments"])
smap.create_namespace("github_prs", "github/pr", ["title", "description", "diff"])
smap.create_namespace("github_commits", "github/commit", ["message", "diff"])
smap.create_namespace("github_contributors", "github/user", ["username", "bio"])
smap.create_namespace("github_code", "github/code", ["content", "path"])

# 添加代码文件自定义嵌入策略
code_strategy = CodeSemanticStrategy()
smap.set_embedding_strategy("github/code", code_strategy)

# 从GitHub API加载数据
load_github_repository_data("microsoft/graphrag", smap)

# 构建语义图谱
sgraph = SemanticGraph(smap)
build_explicit_relations(sgraph)  # 构建显式结构边
sgraph.infer_implicit_edges("github_issues", "github_code", 0.9)  # 构建隐式语义边

# 高级分析
dev_profile = build_developer_expertise_profile("dev_123")
issue_recommendation = assign_new_issue("issue_789")
fix_patterns = analyze_issue_fix_patterns("bug")
```

## 📊 性能与扩展性

- **内存优化**：延迟加载策略，按需生成嵌入
- **分布式支持**：可扩展到分布式存储后端
- **缓存机制**：策略结果LRU缓存
- **FAISS集成**：高效相似度搜索
- **并行处理**：大规模数据处理时的并行计算支持

## 📝 贡献指南

我们欢迎各种形式的贡献，包括功能建议、代码提交和文档改进。

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个新的 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🔗 相关资源

- [项目文档](https://semantic-graph-memory.readthedocs.io/)
- [API参考](https://semantic-graph-memory.readthedocs.io/api/)
- [示例代码](examples/)
- [相关论文](docs/papers.md)

---

## 联系我们

如有问题或建议，请通过 [GitHub Issues](https://github.com/yourusername/semantic-graph-memory/issues) 联系我们。