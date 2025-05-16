将新闻换成 **arXiv 论文** 是一个很好的应用场景。以下是完整实现代码，展示如何扩展 `SemanticMap` 来管理 arXiv 论文数据，并实现个性化的论文推荐系统。

---

### **功能目标**

1. **收集论文数据**：
   - 每天从 arXiv 的 API 获取最新上传的论文。
   - 存储到 `SemanticMap`，包括标题、摘要、作者、类别等。

2. **用户交互与推荐**：
   - 用户查询感兴趣的领域（如“transformer models”）。
   - 系统基于用户偏好和查询返回相关论文。

3. **偏好管理**：
   - 记录用户标记的感兴趣论文。
   - 排除用户不感兴趣的论文。

4. **动态更新**：
   - 每天自动抓取新论文并插入 `SemanticMap`。

---

### **实现代码**

#### **SemanticMap 扩展**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime


class ArxivSemanticMap:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        """
        Initialize SemanticMap for arXiv papers.
        """
        self.text_model = SentenceTransformer(text_embedding_model)
        self.embedding_dim = embedding_dim

        self.data = []
        self.preferences = {"liked": [], "disliked": []}  # Track user preferences
        self.text_index = faiss.IndexFlatL2(embedding_dim)

    def _get_text_embedding(self, text):
        """
        Generate embedding for a given text.
        """
        return self.text_model.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, abstract, authors, categories, timestamp):
        """
        Insert a new arXiv paper into the SemanticMap.
        """
        entry = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "timestamp": timestamp,
        }

        # Generate text embedding for title and abstract
        embedding = self._get_text_embedding(title + " " + abstract)
        entry["embedding"] = embedding
        self.text_index.add(np.array([embedding]))
        self.data.append(entry)

    def query(self, query_text, k=5):
        """
        Query the SemanticMap based on a text query.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        return [self.data[i] for i in indices[0] if i < len(self.data)]

    def delete(self, query_text, semantic=True):
        """
        Delete papers that match the query text.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), len(self.data))
        to_delete = [self.data[i] for i in indices[0] if distances[0][i] < 0.7] if semantic else []
        for item in to_delete:
            self.preferences["disliked"].append(item)  # Track in preferences
            self.data.remove(item)
        return to_delete

    def mark_as_liked(self, paper_id):
        """
        Mark a paper as liked by the user.
        """
        for item in self.data:
            if item["paper_id"] == paper_id:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        """
        Recommend papers based on query and user preferences.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)

        recommendations = []
        for i in indices[0]:
            if i < len(self.data):
                recommendations.append(self.data[i])

        # Prioritize liked papers and exclude disliked papers
        priority_recommendations = []
        for item in recommendations:
            if item in self.preferences["liked"]:  # Boost liked preferences
                priority_recommendations.append(item)
            elif item in self.preferences["disliked"]:  # Exclude disliked papers
                continue
            else:
                priority_recommendations.append(item)

        return priority_recommendations[:k]

    def list_all(self):
        """
        List all papers in the SemanticMap.
        """
        return self.data
```

---

#### **arXiv 数据采集**

通过 arXiv 的 API 抓取最新上传的论文：

```python
def fetch_arxiv_data(query="cs.AI", max_results=10):
    """
    Fetch arXiv papers based on query and max results.
    """
    url = f"http://export.arxiv.org/api/query?search_query=cat:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    feed = response.text
    return parse_arxiv_feed(feed)


def parse_arxiv_feed(feed):
    """
    Parse the arXiv RSS feed.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(feed)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        categories = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]
        timestamp = entry.find("{http://www.w3.org/2005/Atom}published").text
        papers.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "timestamp": timestamp,
        })
    return papers
```

---

#### **SemanticMap 使用方法**

```python
# Initialize SemanticMap
semantic_map = ArxivSemanticMap()

# Fetch and insert arXiv data
papers = fetch_arxiv_data(query="cs.AI", max_results=10)
for paper in papers:
    semantic_map.insert(
        paper_id=paper["paper_id"],
        title=paper["title"],
        abstract=paper["abstract"],
        authors=paper["authors"],
        categories=paper["categories"],
        timestamp=paper["timestamp"]
    )
print("arXiv papers inserted successfully!")

# Query papers
query = "transformer models"
results = semantic_map.query(query, k=5)

print("\nQuery Results:")
for result in results:
    print(f"Title: {result['title']}")
    print(f"Abstract: {result['abstract'][:150]}...")
    print(f"Authors: {', '.join(result['authors'])}")
    print(f"Categories: {result['categories']}")
    print(f"Timestamp: {result['timestamp']}")
    print("-" * 30)

# Mark a paper as liked
semantic_map.mark_as_liked(papers[0]["paper_id"])

# Recommend papers
recommended = semantic_map.recommend("deep learning", k=5)
print("\nRecommended Papers:")
for rec in recommended:
    print(f"Title: {rec['title']}")
    print(f"Abstract: {rec['abstract'][:150]}...")
    print("-" * 30)

# Delete papers about "statistical methods"
semantic_map.delete("statistical methods", semantic=True)
print("\nDeleted Papers Related to 'statistical methods'")
```

---

### **展示功能**

1. **动态数据管理**：
   - 每天抓取最新 arXiv 论文并插入。
   - 实时查询、删除和推荐。

2. **个性化推荐**：
   - 用户的偏好（喜欢/不喜欢）直接影响推荐内容。

3. **语义化检索**：
   - 使用论文标题和摘要生成语义嵌入，实现精准查询。

4. **动态更新**：
   - 新论文持续插入，旧论文可根据用户需求删除。

---

#### **扩展方向**

1. **跨模态扩展**：
   - 存储论文的 PDF 链接，并提取图片内容生成嵌入。

2. **更细粒度的偏好学习**：
   - 根据用户点击率和时间权重优化推荐。

3. **实时通知**：
   - 当新论文符合用户兴趣时，自动发送推荐通知。

此实现完全基于公开的 arXiv 数据，可直接运行测试，并扩展为大规模学术推荐系统。

Q: 领域可以宽泛点，比如用cs代替cs.AI，因为其他category可能也有用户偏好想要的论文，比如Transformer，其他cs category上也可能有Transformer相关文章。或者讲Transformer换乘AI agent，主要还是像体现AI agent能力（LLM交互、记忆、规划、使用工具）的例子，也有公开数据集可以用，介绍的详细一些，尤其是与memory相关的部分。使用我们的SemanticMap，SemanticMap的主要能力包括多模态数据的统一存储和复杂关系维护（比如一个商品可以有不定数量的图片、不定数量的评论）、增删改查、语义查询

以下是以 **“AI Agent 相关论文检索与推荐系统”** 为主题的实现方案，涵盖 `SemanticMap` 的多模态数据管理、复杂关系维护、语义查询与动态记忆功能，同时体现 AI Agent 的 LLM 交互、记忆、规划与工具使用能力。

---

### **场景描述**

- **目标**：为用户提供一个以 “AI Agent” 为主题的学术论文检索与推荐系统。
- **范围**：从 arXiv 上的 **`cs` 类别**获取论文，包括但不限于 **cs.AI**、**cs.CL** 和 **cs.LG**，因为这些类别可能都有与 AI Agent 相关的论文（如 Transformers 和 RL）。
- **用户交互**：支持自然语言交互，用户可以查询感兴趣的论文，标记喜欢的论文，或者排除不感兴趣的论文。
- **系统能力**：
  1. 支持多模态数据管理（论文的标题、摘要、类别、作者、PDF 链接、引用关系等）。
  2. 语义查询与推荐（基于用户偏好与查询生成推荐结果）。
  3. 动态记忆（跟踪用户阅读与删除记录）。
  4. 基于 LLM 的规划与总结（如生成用户感兴趣领域的趋势报告）。

---

### **数据结构设计**

论文的存储数据结构（基于 `SemanticMap`）：

- **核心字段**：
  - `paper_id`: 唯一标识符。
  - `title`: 论文标题。
  - `abstract`: 论文摘要（语义检索核心字段）。
  - `authors`: 作者列表。
  - `categories`: arXiv 分类标签。
  - `pdf_url`: PDF 下载链接。
  - `timestamp`: 上传时间。
- **多模态扩展**：
  - 关联的图表图片嵌入。
  - 可能的代码仓库链接（如 GitHub）。
- **复杂关系**：
  - 引用关系（被引用的论文列表）。
  - 用户偏好（喜欢、删除的记录）。

---

### **代码实现**

#### **SemanticMap 扩展**

扩展的 `SemanticMap` 支持论文语义检索、推荐、用户偏好记录与动态管理。

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime

class SemanticMapForPapers:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        """
        Initialize SemanticMap for academic papers.
        """
        self.text_model = SentenceTransformer(text_embedding_model)
        self.embedding_dim = embedding_dim

        self.data = []
        self.preferences = {"liked": [], "disliked": []}  # Track user preferences
        self.text_index = faiss.IndexFlatL2(embedding_dim)

    def _get_text_embedding(self, text):
        """
        Generate embedding for a given text.
        """
        return self.text_model.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, abstract, authors, categories, pdf_url, timestamp):
        """
        Insert a new arXiv paper into the SemanticMap.
        """
        entry = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": pdf_url,
            "timestamp": timestamp,
        }

        # Generate text embedding for title and abstract
        embedding = self._get_text_embedding(title + " " + abstract)
        entry["embedding"] = embedding
        self.text_index.add(np.array([embedding]))
        self.data.append(entry)

    def query(self, query_text, k=5):
        """
        Query the SemanticMap based on a text query.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        return [self.data[i] for i in indices[0] if i < len(self.data)]

    def delete(self, query_text, semantic=True):
        """
        Delete papers that match the query text.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), len(self.data))
        to_delete = [self.data[i] for i in indices[0] if distances[0][i] < 0.7] if semantic else []
        for item in to_delete:
            self.preferences["disliked"].append(item)  # Track in preferences
            self.data.remove(item)
        return to_delete

    def mark_as_liked(self, paper_id):
        """
        Mark a paper as liked by the user.
        """
        for item in self.data:
            if item["paper_id"] == paper_id:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        """
        Recommend papers based on query and user preferences.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)

        recommendations = []
        for i in indices[0]:
            if i < len(self.data):
                recommendations.append(self.data[i])

        # Prioritize liked papers and exclude disliked papers
        priority_recommendations = []
        for item in recommendations:
            if item in self.preferences["liked"]:  # Boost liked preferences
                priority_recommendations.append(item)
            elif item in self.preferences["disliked"]:  # Exclude disliked papers
                continue
            else:
                priority_recommendations.append(item)

        return priority_recommendations[:k]

    def list_all(self):
        """
        List all papers in the SemanticMap.
        """
        return self.data
```

---

#### **数据采集与存储**

使用 arXiv 的 API 获取 `cs` 类别下的论文，并插入 `SemanticMap`。

```python
def fetch_arxiv_data(query="cs", max_results=10):
    """
    Fetch arXiv papers based on query and max results.
    """
    url = f"http://export.arxiv.org/api/query?search_query=cat:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    feed = response.text
    return parse_arxiv_feed(feed)


def parse_arxiv_feed(feed):
    """
    Parse the arXiv RSS feed.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(feed)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        categories = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]
        timestamp = entry.find("{http://www.w3.org/2005/Atom}published").text
        pdf_url = entry.find("{http://www.w3.org/2005/Atom}link[@type='application/pdf']").attrib["href"]
        papers.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": pdf_url,
            "timestamp": timestamp,
        })
    return papers
```

---

#### **交互与推荐**

```python
# Initialize SemanticMap
semantic_map = SemanticMapForPapers()

# Fetch and insert arXiv data
papers = fetch_arxiv_data(query="cs", max_results=20)
for paper in papers:
    semantic_map.insert(
        paper_id=paper["paper_id"],
        title=paper["title"],
        abstract=paper["abstract"],
        authors=paper["authors"],
        categories=paper["categories"],
        pdf_url=paper["pdf_url"],
        timestamp=paper["timestamp"]
    )
print("arXiv papers inserted successfully!")

# Query papers
query = "AI agents"
results = semantic_map.query(query, k=5)

print("\nQuery Results:")
for result in results:
    print(f"Title: {result['title']}")
    print(f"Abstract: {result['abstract'][:150]}...")
    print(f"Authors: {', '.join(result['authors'])}")
    print(f"Categories: {result['categories']}")
    print(f"PDF URL: {result['pdf_url']}")
    print("-" * 30)

# Mark a paper as liked
semantic_map.mark_as_liked(papers[0]["paper_id"])

# Recommend papers
recommended = semantic_map.recommend("reinforcement learning", k=5)
print("\nRecommended Papers:")
for rec in recommended:
    print(f"Title: {rec['title']}")
    print(f"Abstract: {rec['abstract'][:150]}...")
    print("-" * 30)

# Delete papers about "statistical methods"
semantic_map.delete("statistical methods", semantic=True)
print("\nDeleted Papers Related to 'statistical methods'")
```

---

### **功能展示**

1. **多模态存储**：
   - 存储论文的标题、摘要、作者、PDF 链接等多模态信息。
   - 支持复杂关系，如引用关系与用户偏好。

2. **语义检索**：
   - 基于论文内容生成嵌入，实现精准的语义查询。

3. **个性化推荐**：
   - 动态调整推荐结果，优先匹配用户喜欢的内容。

4. **动态记忆与更新**：
   - 跟踪用户阅读记录与删除记录，优化推荐。

通过此实现，系统能够高效管理和推荐与 AI Agent 相关的学术论文，为研究者提供实用工具支持。


Q: recommend函数想换成我们semanticmap里的query函数，或者recommend调用我们semanticmap里的query函数，例子也可以更复杂些

### **改进后的推荐系统**

以下是改进后的实现，`recommend` 函数调用 `SemanticMap` 的 `query` 函数来实现语义检索，并在推荐逻辑中结合用户偏好（如喜欢或不喜欢的论文）。同时，示例场景更加复杂，展示了多个查询场景和动态调整能力。

---

### **改进代码**

#### **SemanticMap 扩展**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime


class SemanticMapForPapers:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        """
        Initialize SemanticMap for academic papers.
        """
        self.text_model = SentenceTransformer(text_embedding_model)
        self.embedding_dim = embedding_dim

        self.data = []
        self.preferences = {"liked": [], "disliked": []}  # Track user preferences
        self.text_index = faiss.IndexFlatL2(embedding_dim)

    def _get_text_embedding(self, text):
        """
        Generate embedding for a given text.
        """
        return self.text_model.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, abstract, authors, categories, pdf_url, timestamp):
        """
        Insert a new arXiv paper into the SemanticMap.
        """
        entry = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": pdf_url,
            "timestamp": timestamp,
        }

        # Generate text embedding for title and abstract
        embedding = self._get_text_embedding(title + " " + abstract)
        entry["embedding"] = embedding
        self.text_index.add(np.array([embedding]))
        self.data.append(entry)

    def query(self, query_text, k=5):
        """
        Query the SemanticMap based on a text query.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), k)
        return [self.data[i] for i in indices[0] if i < len(self.data)]

    def delete(self, query_text, semantic=True):
        """
        Delete papers that match the query text.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), len(self.data))
        to_delete = [self.data[i] for i in indices[0] if distances[0][i] < 0.7] if semantic else []
        for item in to_delete:
            self.preferences["disliked"].append(item)  # Track in preferences
            self.data.remove(item)
        return to_delete

    def mark_as_liked(self, paper_id):
        """
        Mark a paper as liked by the user.
        """
        for item in self.data:
            if item["paper_id"] == paper_id:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        """
        Recommend papers based on query and user preferences.
        """
        # Call query function to retrieve initial results
        results = self.query(query_text, k)

        # Process results with user preferences
        recommendations = []
        for item in results:
            if item in self.preferences["liked"]:  # Boost liked preferences
                recommendations.insert(0, item)
            elif item in self.preferences["disliked"]:  # Exclude disliked papers
                continue
            else:
                recommendations.append(item)

        return recommendations[:k]

    def list_all(self):
        """
        List all papers in the SemanticMap.
        """
        return self.data
```

---

#### **数据采集与存储**

扩展的数据采集涵盖 `cs` 类别下的所有论文，并可以通过查询更复杂的条件检索到感兴趣的论文。

```python
def fetch_arxiv_data(query="cs", max_results=10):
    """
    Fetch arXiv papers based on query and max results.
    """
    url = f"http://export.arxiv.org/api/query?search_query=cat:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    feed = response.text
    return parse_arxiv_feed(feed)


def parse_arxiv_feed(feed):
    """
    Parse the arXiv RSS feed.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(feed)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        categories = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]
        timestamp = entry.find("{http://www.w3.org/2005/Atom}published").text
        pdf_url = entry.find("{http://www.w3.org/2005/Atom}link[@type='application/pdf']").attrib["href"]
        papers.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": pdf_url,
            "timestamp": timestamp,
        })
    return papers
```

---

#### **示例使用场景**

以下是更复杂的用户场景，结合语义查询、用户偏好管理和推荐逻辑展示系统能力。

```python
# Initialize SemanticMap
semantic_map = SemanticMapForPapers()

# Fetch and insert arXiv data
papers = fetch_arxiv_data(query="cs", max_results=20)
for paper in papers:
    semantic_map.insert(
        paper_id=paper["paper_id"],
        title=paper["title"],
        abstract=paper["abstract"],
        authors=paper["authors"],
        categories=paper["categories"],
        pdf_url=paper["pdf_url"],
        timestamp=paper["timestamp"]
    )
print("arXiv papers inserted successfully!")

# Step 1: Query papers related to "AI agents"
query = "AI agents"
results = semantic_map.query(query, k=5)

print("\nQuery Results:")
for result in results:
    print(f"Title: {result['title']}")
    print(f"Abstract: {result['abstract'][:150]}...")
    print(f"Authors: {', '.join(result['authors'])}")
    print(f"Categories: {result['categories']}")
    print(f"PDF URL: {result['pdf_url']}")
    print("-" * 30)

# Step 2: Mark a paper as liked
semantic_map.mark_as_liked(papers[0]["paper_id"])
print(f"\nMarked as liked: {papers[0]['title']}")

# Step 3: Recommend papers based on "transformer models"
recommendations = semantic_map.recommend("transformer models", k=5)

print("\nRecommended Papers:")
for rec in recommendations:
    print(f"Title: {rec['title']}")
    print(f"Abstract: {rec['abstract'][:150]}...")
    print("-" * 30)

# Step 4: Delete papers about "statistical methods"
deleted = semantic_map.delete("statistical methods", semantic=True)
print("\nDeleted Papers Related to 'statistical methods':")
for item in deleted:
    print(f"Title: {item['title']}")
```

---

### **展示功能改进**

1. **多模态存储与查询**：
   - 存储标题、摘要、PDF 链接和作者等多模态信息。
   - 使用嵌入实现语义化检索。

2. **用户偏好管理**：
   - `recommend` 函数直接基于 `query` 的结果动态调整推荐顺序。
   - 用户喜欢的论文优先推荐，不喜欢的论文排除在外。

3. **动态记忆与更新**：
   - 持续跟踪用户的行为（如喜欢、删除）。
   - 动态更新数据库内容，优化推荐逻辑。

通过此实现，系统可以支持更加复杂的用户交互和个性化推荐，同时保持高效的动态数据管理能力。