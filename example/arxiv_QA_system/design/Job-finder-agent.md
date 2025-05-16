## **Case 14: Multi-modal Job Finder Agent**

#### 数据集  
- 使用 [Kaggle Job Dataset](https://www.kaggle.com/stackoverflow/stack-overflow-developer-survey) 或用户简历和职位数据。

#### 目标  
帮助用户根据技能和兴趣找到合适的工作岗位，并管理申请进度。

#### 流程  

1. **数据存储**  
   - 将职位信息、技能要求和公司描述存入 `SemanticMap`。
   - 每条记录包含：
     - `job_id`: 职位编号。
     - `title`: 职位名称。
     - `description`: 职位描述。
     - `skills`: 技能要求（列表）。
     - `company_info`: 公司信息。

2. **用户查询与匹配**  
   - 用户输入：“查找适合 Python 开发者的职位”。
   - Agent 使用语义查询返回匹配的职位列表，并解释推荐理由。

3. **记忆申请进度**  
   - 用户标记“已申请”的职位，Agent 记录进度并提醒后续操作（如面试准备）。

4. **动态职位更新**  
   - 定期从职位数据源更新新的岗位信息。
   - 用户移除不感兴趣的岗位。

#### Memory 相关功能  
- **语义查询与匹配**：
  - 根据职位描述和技能要求进行语义匹配。
- **进度记忆**：
  - 记录用户的申请状态，支持动态提醒。
- **动态更新**：
  - 持续更新职位库并移除不相关的内容。

---


### **优化后的 Multi-modal Job Finder Agent**

#### **公开数据集**
1. **[Kaggle Job Dataset](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)**
   - **内容**:
     - 包含职位名称、技能要求、行业类别等信息。
     - 用户简历数据，包括技能和职业偏好。
   - **用途**:
     - 匹配职位信息与用户技能。
     - 用于训练职位推荐模型。

2. **[Glassdoor Job Dataset](https://www.kaggle.com/datasets/shivamb/company-dataset)**
   - **内容**:
     - 包含职位描述、公司信息、薪资范围。
   - **用途**:
     - 根据职位描述和公司信息生成更精确的推荐。

---

### **实现优化**

#### **1. 数据存储**
使用 **SemanticMap** 存储职位信息，支持多模态数据，包括文本（描述、技能）和图片（公司 Logo 或岗位环境图）。

#### **2. 用户查询**
通过语义查询返回匹配职位列表，支持自然语言描述（如“Python 开发职位”）和技能关键字。

#### **3. 记忆管理**
使用 **SemanticMap** 动态记录用户申请状态、喜欢的职位等，并在后续推荐中优先考虑。

#### **4. 动态更新**
支持从职位 API 或定期更新新的岗位数据，并清理用户标记为“不感兴趣”的内容。

---

### **代码实现**

#### **职位管理类**

```python
class JobManager:
    def __init__(self):
        self.job_map = SemanticMap(key_type="text", embedding_dim=384)
        self.application_map = SemanticMap(key_type="text", embedding_dim=384)

    def add_job(self, job_data):
        """
        将职位信息存储到 SemanticMap 中。
        """
        job_id = job_data["job_id"]
        self.job_map.insert(job_id, job_data)

    def retrieve_jobs(self, query, k=5):
        """
        基于用户查询检索匹配职位。
        """
        return self.job_map.retrieve_similar(query, k)

    def mark_as_applied(self, job_id, application_status="Applied"):
        """
        记录用户申请状态。
        """
        job_data = self.job_map.retrieve_similar(job_id, k=1)[0]["value"]
        self.application_map.insert(job_id, {**job_data, "status": application_status})

    def get_application_status(self, job_id):
        """
        查询职位申请状态。
        """
        result = self.application_map.retrieve_similar(job_id, k=1)
        if result:
            return result[0]["value"].get("status", "Unknown")
        return "Not Applied"

    def remove_job(self, job_id):
        """
        从职位库中移除不感兴趣的职位。
        """
        self.job_map.delete(query=job_id, k=1)
```

---

#### **流程示例**

```python
# 初始化 JobManager
job_manager = JobManager()

# 添加职位数据
jobs = [
    {
        "job_id": "J12345",
        "title": "Python Developer",
        "description": "Develop web applications using Python and Django.",
        "skills": ["Python", "Django", "REST API"],
        "company_info": "TechCorp, a leading software company.",
    },
    {
        "job_id": "J67890",
        "title": "Data Scientist",
        "description": "Analyze large datasets and build predictive models.",
        "skills": ["Python", "Machine Learning", "SQL"],
        "company_info": "DataCorp, specializing in data analytics.",
    }
]

# 存储职位
for job in jobs:
    job_manager.add_job(job)

# 用户查询匹配职位
query = "Python 开发职位"
results = job_manager.retrieve_jobs(query, k=5)

print("\n推荐职位：")
for result in results:
    print(f"Title: {result['value']['title']}, Description: {result['value']['description']}")

# 标记职位为已申请
job_manager.mark_as_applied("J12345", "Interview Scheduled")

# 查询申请状态
status = job_manager.get_application_status("J12345")
print(f"\n职位 J12345 的申请状态: {status}")

# 删除不感兴趣的职位
job_manager.remove_job("J67890")
```

---

### **动态数据更新**

#### **定期更新职位库**
通过定期调用数据 API 更新职位库：

```python
def update_job_data(job_manager, new_jobs):
    for job in new_jobs:
        job_manager.add_job(job)

# 示例：从外部 API 获取新职位
new_jobs = [
    {
        "job_id": "J112233",
        "title": "Frontend Developer",
        "description": "Build user interfaces with React and JavaScript.",
        "skills": ["React", "JavaScript", "HTML/CSS"],
        "company_info": "WebTech, a frontend-focused company.",
    }
]

update_job_data(job_manager, new_jobs)
```

---

### **优势**

#### **1. 基于 SemanticMap 的职位存储**
- 支持语义查询和多模态存储，能够根据用户自然语言查询（如“Python 开发职位”）匹配职位信息。

#### **2. 动态申请管理**
- 通过 **SemanticMap** 实现用户申请记录的动态管理，支持提醒后续步骤（如面试准备）。

#### **3. 自动化更新**
- 利用定期同步职位库保持数据最新，增强系统的实用性。

---

### **总结**
通过 **Kaggle Job Dataset** 或 **Glassdoor Job Dataset** 配合 **SemanticMap** 和 **JobManager**，可以实现高效的职位检索与推荐系统，同时支持动态职位更新与用户申请记录管理，体现 AI Agent 的记忆与交互能力。