# Agent Memory System

<img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version" />
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License" />
<img src="https://img.shields.io/badge/python-3.8+-orange.svg" alt="Python Version" />

An enhanced memory system based on semantic data structures, enabling multi-modal knowledge representation and complex higher-order reasoning through explicit structural relationships and implicit semantic associations.

## 🌟 Project Overview

Agent Memory System (AMS) is an advanced memory system that combines the capabilities of semantic vector databases and graph databases to create a more intelligent and flexible knowledge management and retrieval solution. This system uses SemanticMap to organize structured data and builds a SemanticGraph on top of it to establish explicit and implicit associations between entities.

Features:

- **Hybrid Memory Structure**: Combines vectorized semantic memory and structured relational graph representation.
- **Multi-hop Complex Queries**: Supports multi-level knowledge reasoning along relationship chains.
- **Coexistence of Explicit and Implicit Edges**: Expresses both deterministic relationships and semantic similarity associations simultaneously.
- **Multi-modal Support**: Handles various data types such as text, code, images, etc.
- **Pluggable Extension**: Supports custom embedding strategies and search algorithms.

## 🧩 System Architecture

### Core Components

#### 1. MemoryUnit

The most basic memory unit, encapsulating specific data entities:

```python
unit = MemoryUnit(
    uid="issue_1618",
    raw_data={"title": "Fix memory leak", "description": "System crashes after running for 24h"},
    metadata={"created_by": "user123", "tags": ["bug", "critical"]}
)
```

Characteristics:

  - Unique identifier management
  - Separation of raw data and metadata
  - Automatic data fingerprint calculation
  - Version history tracking
  - Flexible semantic vector representation

#### 2. MemorySpace

A namespace for organizing MemoryUnits by type:

```python
smap.create_namespace(
    ms_name="github_issues",
    ms_type="github/issue",
    embedding_fields=["title", "description", "comments"]
)
```

Characteristics:

  - Typed data organization
  - Configurable embedding fields
  - Efficient retrieval indexing
  - Schema validation support

#### 3. SemanticMap

Manages multiple MemorySpaces and provides semantic retrieval functionality:

```python
# Create a global semantic map
smap = SemanticMap()

# Register a custom embedding strategy
smap.set_embedding_strategy("code", code_strategy)

# Insert a memory unit
smap.insert_unit("github_issues", issue_unit)

# Semantic search
results = smap.find_similar_units("memory leak fixes", ms_names=["github_issues"], top_k=5)
```

Characteristics:

  - Support for multi-modal embedding models
  - Cross-space semantic retrieval
  - Custom embedding strategies
  - Efficient inverted indexing

#### 4. SemanticGraph

Builds a relational graph network on top of SemanticMap:

```python
# Create a semantic graph
sgraph = SemanticGraph(smap)

# Add an explicit structural edge
sgraph._add_explicit_edge(
    source="pr_423",
    target="issue_1618",
    rel_type="PR fixes Issue",
    metadata={"commit_hash": "a1b2c3d"}
)

# Generate implicit semantic edges
sgraph.infer_implicit_edges("github_issues", "github_code", similarity_threshold=0.9)

# Relationship query
relations = sgraph.find_relations(target="issue_1618", rel_type="PR fixes Issue")
```

Characteristics:

  - Dual-edge (explicit/implicit) graph structure
  - Weighted relationship representation
  - Multi-hop path queries
  - Automatic inference of semantic similarity

## 💡 Advanced Application Scenarios

### 1. [Enhanced GitHub Repository Memory System](example/issue_manager)

A practical application case of this project is building an enhanced memory system for GitHub repositories. It organizes various types of information within a repository (issues, PRs, commits, contributors, code files) into a knowledge graph, supporting complex queries and insight discovery.

#### Data Organization

```
SemanticMap (smap)
├── MemorySpace: github_issues
├── MemorySpace: github_prs
├── MemorySpace: github_commits
├── MemorySpace: github_contributors
└── MemorySpace: github_code

SemanticGraph (sgraph)
├── Explicit Structural Edges:
│   ├── Developer submits/reviews commit
│   ├── Commit modifies code file
│   ├── Repository's commit corresponds to a unique PR
│   ├── Commits sorted by time form the main chain
│   ├── PR fixes Issue
│   ├── Developer submits/reviews PR
│   ├── PR contains commit chain
│   └── PR modifies code file
└── Implicit Semantic Edges:
    └── Connections between entities with semantic similarity > 0.9
```

#### Advanced Application Examples

##### 1. [Developer Expertise Profile](example/issue_manager/developer_profile.py)

Combines explicit and implicit edges to build a developer's professional skill graph:

```json
{
  "developer_id": "dev_789",
  "creation_date": "2025-04-28",
  "core_expertise": {
    "primary_domains": ["Graph Databases", "Search Engines", "Caching Systems"],
    "expertise_level": "Advanced",
    "technical_strengths": ["Performance Optimization", "System Design", "Concurrency Control"],
    "preferred_technologies": ["Python", "C++", "Redis"]
  },
  "code_proficiency": {
    "mastered_modules": ["Memory Management", "Indexer", "Query Optimizer"],
    "contribution_areas": ["Core Engine", "API Design", "Test Framework"],
    "code_quality_traits": ["Efficient", "Highly Readable", "Comprehensive Error Handling"],
    "complexity_handling": "Can effectively decompose complex systems and design clear module interfaces"
  },
   ...
  "confidence_score": 0.94
}
```

##### 2. [Issue Fix Pattern Analysis](example/issue_manager/issue_fix_pattern.py)

Combines explicit and implicit edges to analyze the complete path of bug fixes:

```json
{
  "Fix Strategy": "Refactor the parameter passing mechanism of the storage factory, adopting a dynamic kwargs filtering pattern where each storage implementation class extracts the required parameters itself.",
  "Fix Steps": [
    "Change the parameters of create_blob_storage to **kwargs to receive all configuration parameters",
    "Extract required parameters within the blob storage implementation using kwargs.get()",
    "Update the factory class documentation to explain the parameter passing mechanism",
    "Add integration tests to verify the instantiation of each storage type",
    "Add conditional skipping of cosmosdb tests on Windows platform"
  ],
  "Key Components Modified": [
    {
      "File Path": "graphrag/storage/blob_pipeline_storage.py",
      "Functional Description": "Azure Blob storage implementation",
      "Modification Content": "Changed function signature to **kwargs, extract connection_string etc. parameters internally via kwargs"
    },
    {
      "File Path": "graphrag/storage/factory.py",
      "Functional Description": "Storage factory base class",
      "Modification Content": "Added documentation explaining the parameter passing strategy, clarifying that each implementation class handles kwargs parameters itself"
    },
    {
      "File Path": "tests/integration/storage/test_factory.py",
      "Functional Description": "Storage factory integration tests",
      "Modification Content": "Added new test cases to verify instantiation of blob/cosmosdb/file/memory storage, including parameter passing tests"
    },
     ...
  ]
}
```

### 2. [Intelligent Academic Group Meeting Management System](example/academic_group_meeting)

Implements full lifecycle management of academic discussions through semantic graphs, supporting automatic meeting minute generation, knowledge association retrieval, and intelligent Q&A.

#### Data Organization

```
AcademicSemanticGraph
├── Node Types:
│   ├── Participant Node: Professor/PhD Student/Master Student
│   ├── Research Topic Node: Contains domain keywords and technical routes
│   ├── Discussion Node: Speech content and timestamp
│   └── Attachment Node: Paper/Presentation/Dataset
└── Relationship Types:
    ├── Speaks: Participant → Discussion Node
    ├── Cites: Discussion Node → Paper Node
    ├── Summarizes: Discussion Node → Conclusion Node
    └── Correlates: Cross-domain semantic similarity connection
```

#### Core Functions

  - **Intelligent Agenda Generation**: Automatically plans meeting topics based on historical discussions.
  - **Real-time Knowledge Graph**: Dynamically builds an association network between discussion content and academic resources.
  - **Multi-modal Minute Output**:
    ```python
    # Automatically generate structured meeting minutes
    meeting_summary = agent.generate_summary(
        include=["Key Conclusions", "Open Issues", "References"],
        format="markdown"
    )
    ```
  - **Cross-Meeting Traceability**: Tracks the evolution of research ideas through a timeline view.
  - **database connect**: connect to the milvus and neo4j.Store and remember your data while you perform academic group meeting. 

#### Usage Guide

  This example has a front end, you can view it on the website.

- **Installation**:
  ```bash
  pip install requirements.txt

- **Running the Frontend**:
  ```bash
  cd ./example/academic_group_meeting/front_end
  streamlit run front_end.py



### 3. [Arxiv Paper Intelligent Q&A System](example/arxiv_QA_system)

An Arxiv paper analysis system based on semantic graphs, achieving deep understanding and complex Q&A for scientific papers through multi-level semantic representation and structured query planning.

#### Data Organization

```
ArxivSemanticGraph
├── Node Types:
│   ├── Root Node: {paper_id}_title_authors (Title and author information)
│   ├── Abstract Node: {paper_id}_abstract
│   ├── Chapter Node: {paper_id}_chapter_{idx}
│   ├── Paragraph Node: {chapter_key}_paragraph_{para_idx}
│   ├── Image Node: {chapter_key}_photo_{photo_idx}
│   └── Table Row Node: {paper_id}_table_{table_idx}_row_{row_idx}
└── Relationship Types:
    ├── has_abstract: Paper → Abstract
    ├── has_chapter: Paper → Chapter
    ├── has_paragraph: Chapter → Paragraph
    ├── has_photo: Chapter → Image
    └── has_table_row: Paper → Table Row
```

#### System Architecture

A powerful Q\&A system composed of ArxivSemanticGraph and ArxivAgent:

1.  **Data Collection and Processing**

      * Download paper HTML files from Arxiv
      * Parse HTML to extract structured content (title, authors, abstract, chapters, images, tables)
      * Build a semantic graph to represent paper knowledge

2.  **Multi-level Query Processing**

      * Analysis and optimization of user natural language questions
      * LLM generates structured query plans
      * Multi-step query execution engine
      * Result aggregation and answer generation

3.  **Enhanced Features**

      * User preference recording and recommendation
      * Semantic similarity assessment
      * Multi-step reasoning paths

#### Query Plan Example

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

#### Advanced Query Capabilities

More powerful features than traditional RAG systems:

1.  **Hybrid Structured + Semantic Queries**

      * Can utilize both paper structure information and semantic similarity simultaneously
      * Example: Find specific chapters in papers related to a certain topic

2.  **Multi-step Reasoning**

      * Find author from paper → Retrieve author's other papers → Analyze research trends
      * Compare methods used in different papers on the same topic

3.  **Preference Learning**

      * Record papers the user likes/dislikes
      * Provide personalized recommendations based on historical preferences

4.  **Deep Content Understanding**

      * Hierarchical representation from abstract to chapters to paragraphs
      * Supports precise queries targeting specific parts of the paper

5.  **Cross-document Reasoning**

      * Compare how different papers handle the same problem
      * Synthesize information from multiple papers to answer complex questions

#### Example Application

```python
# Create Arxiv paper semantic graph
graph = ArxivSemanticGraph()

# Parse paper and insert into the graph
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

# Create Q&A agent
agent = ArxivAgent(graph, api_key="your-api-key")

# Complex question query
question = "Compare the differences in training methods between the two latest papers on large language models"
answer = agent.structured_semantic_query(question)
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from semantic_graph_memory import SemanticMap, SemanticGraph, MemoryUnit

# Create a semantic map
smap = SemanticMap()

# Create a namespace
smap.create_namespace(
    ms_name="documents",
    ms_type="text/document",
    embedding_fields=["title", "content"]
)

# Add a memory unit
doc_unit = MemoryUnit(
    uid="doc_001",
    raw_data={
        "title": "Introduction to Vector Databases",
        "content": "A vector database is a database system specialized for storing high-dimensional vectors..."
    }
)
smap.insert_unit("documents", doc_unit)

# Create a semantic graph
sgraph = SemanticGraph(smap)

# Add an explicit relationship
sgraph._add_explicit_edge(
    source="doc_001",
    target="doc_002",
    rel_type="references",
    metadata={"context": "Technical background"}
)

# Infer implicit relationships
sgraph.infer_implicit_edges("documents", similarity_threshold=0.8)

# Perform explicit query
results = sgraph.find_relations(source="doc_001", rel_type="references")
```

### GitHub Repository Integration Example

```python
# Create GitHub repository memory system
smap = SemanticMap()

# Create relevant namespaces
smap.create_namespace("github_issues", "github/issue", ["title", "body", "comments"])
smap.create_namespace("github_prs", "github/pr", ["title", "description", "diff"])
smap.create_namespace("github_commits", "github/commit", ["message", "diff"])
smap.create_namespace("github_contributors", "github/user", ["username", "bio"])
smap.create_namespace("github_code", "github/code", ["content", "path"])

# Add custom embedding strategy for code files
code_strategy = CodeSemanticStrategy()
smap.set_embedding_strategy("github/code", code_strategy)

# Load data from GitHub API
load_github_repository_data("microsoft/graphrag", smap)

# Build semantic graph
sgraph = SemanticGraph(smap)
build_explicit_relations(sgraph)  # Build explicit structural edges
sgraph.infer_implicit_edges("github_issues", "github_code", 0.9)  # Build implicit semantic edges

# Advanced analysis
dev_profile = build_developer_expertise_profile("dev_123")
issue_recommendation = assign_new_issue("issue_789")
fix_patterns = analyze_issue_fix_patterns("bug")
```

## 📊 Performance and Scalability

  - **Memory Optimization**: Lazy loading strategy, on-demand embedding generation
  - **Distributed Support**: Scalable to distributed storage backends
  - **Caching Mechanism**: LRU cache for strategy results
  - **FAISS Integration**: Efficient similarity search
  - **Parallel Processing**: Parallel computation support for large-scale data processing

## 📝 Contribution Guide

We welcome contributions of all forms, including feature suggestions, code submissions, and documentation improvements.

1.  Fork this repository
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add some amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Create a new Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Resources

  - [Project Documentation](https://semantic-graph-memory.readthedocs.io/)
  - [API Reference](https://semantic-graph-memory.readthedocs.io/api/)
  - [Example Code](examples/)
  - [Related Papers](docs/papers.md)

-----

## Contact Us

If you have questions or suggestions, please contact us via [GitHub Issues](https://github.com/yourusername/semantic-graph-memory/issues).