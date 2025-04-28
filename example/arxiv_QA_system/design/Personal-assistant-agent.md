---

### Case 2: **Personalized News Summarization Agent**

#### 目标
为用户构建个性化新闻摘要系统，过滤不感兴趣的内容，并支持图片、视频检索。

#### 流程

1. **Step 1: 收集新闻数据**  
   - 通过 RSS Feed 或新闻 API 获取新闻数据，包括标题、正文、图片和视频链接。
   - 将新闻条目存储到 `SemanticMap`，每条记录包括：
     - 新闻标题、正文（文本嵌入）。
     - 图片和视频（嵌入）。

2. **Step 2: 查询兴趣内容**  
   - 用户查询“关于人工智能进展的新闻”或“最新的科技趋势”。
   - 返回匹配的新闻标题、摘要和相关图片。

3. **Step 3: 排除不感兴趣的内容**  
   - 用户输入“删除所有与政治相关的内容”。
   - 通过语义搜索找到匹配内容并删除。

4. **Step 4: 生成个性化摘要**  
   - 根据剩余新闻，生成用户感兴趣的新闻摘要列表，并附加相关图片。

#### 展示功能
- **多模态检索**：结合文本、图片和视频嵌入进行语义检索。
- **动态更新**：支持实时数据插入和删除。
- **LLM 推理**：根据用户兴趣生成个性化摘要。

---

### **优化后的 Case 15: AI Personal Assistant for Daily Life**

#### **数据集**
1. **Google Calendar API**
   - 提供日历事件数据，包括时间、地点和描述。
   - 支持动态更新事件和提醒。

2. **用户生成任务数据**
   - 用户输入的任务列表，包括优先级和时间。
   - 结合语义化存储支持上下文分析。

---

### **目标**
帮助用户高效管理日常任务，通过多模态存储和动态优化实现智能日程管理，支持以下功能：
- **日程优化**：根据时间、优先级和冲突重新调整任务。
- **事件提醒**：主动提醒用户即将到来的事件。
- **反馈与记忆**：记录用户任务完成状态，支持连续任务。
- **动态调整**：添加新任务时，自动重新优化日程。

---

### **实现优化**

#### **数据存储**
使用 **SemanticMap** 存储任务数据，支持基于文本或时间的语义查询，记录复杂的任务关系。

#### **代码实现**

##### **任务管理类**

```python
class TaskManager:
    def __init__(self):
        # 使用 SemanticMap 存储任务和事件
        self.task_map = SemanticMap(key_type="text", embedding_dim=384)

    def add_task(self, task_data):
        """
        添加任务到 SemanticMap。
        """
        task_id = task_data["event_id"]
        self.task_map.insert(task_id, task_data)

    def retrieve_tasks(self, query, k=5):
        """
        基于用户查询检索任务或事件。
        """
        return self.task_map.retrieve_similar(query, k)

    def optimize_schedule(self):
        """
        优化任务日程，考虑时间和优先级冲突。
        """
        tasks = [task["value"] for task in self.task_map.data]
        # 按时间和优先级排序
        tasks.sort(key=lambda t: (t["time"], -t["priority"]))
        return tasks

    def update_task_status(self, event_id, status):
        """
        更新任务完成状态。
        """
        task = self.task_map.retrieve_similar(event_id, k=1)[0]["value"]
        task["status"] = status
        self.task_map.update(event_id, new_value=task)

    def adjust_schedule(self, new_task):
        """
        动态调整日程，添加新任务后重新优化。
        """
        self.add_task(new_task)
        return self.optimize_schedule()

    def remind_next_event(self):
        """
        提醒用户最近的事件。
        """
        tasks = self.optimize_schedule()
        if tasks:
            next_event = tasks[0]
            return f"下一事件：{next_event['title']}，时间：{next_event['time']}"
        return "没有即将到来的事件。"
```

---

#### **流程示例**

```python
# 初始化任务管理
task_manager = TaskManager()

# 添加任务数据
tasks = [
    {"event_id": "E001", "title": "开会", "time": "2024-12-25 09:00", "location": "办公室", "priority": 3, "status": "Pending"},
    {"event_id": "E002", "title": "医生预约", "time": "2024-12-25 11:00", "location": "医院", "priority": 5, "status": "Pending"},
    {"event_id": "E003", "title": "完成项目报告", "time": "2024-12-25 14:00", "location": "家", "priority": 4, "status": "Pending"},
]

# 将任务存储到 SemanticMap
for task in tasks:
    task_manager.add_task(task)

# 优化日程
optimized_schedule = task_manager.optimize_schedule()
print("\n优化后的日程：")
for task in optimized_schedule:
    print(f"{task['time']}: {task['title']} (优先级 {task['priority']})")

# 提醒用户下一个事件
reminder = task_manager.remind_next_event()
print("\n提醒：", reminder)

# 更新任务状态
task_manager.update_task_status("E003", "Completed")
print("\n更新任务状态后：")
completed_task = task_manager.task_map.retrieve_similar("E003", k=1)
print(completed_task[0]["value"])

# 动态调整日程
new_task = {"event_id": "E004", "title": "下午开会", "time": "2024-12-25 16:00", "location": "办公室", "priority": 2, "status": "Pending"}
adjusted_schedule = task_manager.adjust_schedule(new_task)
print("\n动态调整后的日程：")
for task in adjusted_schedule:
    print(f"{task['time']}: {task['title']} (优先级 {task['priority']})")
```

---

### **核心功能**

1. **任务存储与语义查询**
   - 使用 **SemanticMap** 支持复杂任务存储，任务查询支持自然语言描述。

2. **日程优化**
   - 根据时间冲突和优先级调整任务顺序。

3. **事件提醒**
   - 主动提醒最近事件，减少任务遗漏。

4. **动态调整**
   - 用户新增任务后，自动调整日程并提供建议。

---

### **优势**
- **动态优化**: 根据实时任务调整日程，提升用户效率。
- **任务记忆**: 记录用户完成状态，支持任务衔接。
- **自然语言交互**: 基于语义查询实现智能化日程管理。

通过结合 **Google Calendar API** 或用户生成的任务数据，这一优化后的 Personal Assistant 系统可实现智能日程管理，增强用户体验。


Q: 有没有公开数据集，是否还需要与LLM交互


### **公开数据集推荐**

#### 1. **[Google Calendar API](https://developers.google.com/calendar)**
   - **内容**:
     - 提供个人日历数据，包括事件标题、时间、地点和描述。
     - 支持事件创建、更新和提醒。
   - **适用场景**:
     - 用于获取用户的日程数据，动态更新事件状态。
     - 实现事件提醒和时间冲突优化。
   - **数据格式**:
     ```json
     {
       "id": "event1",
       "summary": "开会",
       "start": { "dateTime": "2024-12-25T09:00:00" },
       "end": { "dateTime": "2024-12-25T10:00:00" },
       "location": "办公室",
       "description": "讨论项目进度",
       "priority": 3
     }
     ```

#### 2. **[iCalender Format (ics)](https://icalendar.org/)**
   - **内容**:
     - 通用事件存储格式，可通过多种应用导出。
   - **适用场景**:
     - 用于集成第三方任务管理或日程数据。
   - **数据格式**:
     - 事件包含 `SUMMARY`, `DTSTART`, `DTEND`, `DESCRIPTION`, 等字段。

#### 3. **[Kaggle To-Do List Dataset](https://www.kaggle.com/datasets/kaushal289/to-do-dataset)**
   - **内容**:
     - 提供个人任务数据，包括任务标题、描述、优先级、截止日期等。
   - **适用场景**:
     - 存储用户日常任务，与日程优化相结合。
   - **数据格式**:
     ```csv
     Task, Description, Priority, Due Date
     完成项目报告, 编写项目计划书并提交, 高, 2024-12-25
     医生预约, 访问医生诊所, 中, 2024-12-25
     ```

---

### **是否需要与 LLM 交互**

是的，与 **LLM** 交互有以下好处：

#### **1. 自然语言解析**
   - **需求**:
     用户可能用自然语言输入任务或查询（如“添加明天上午 9 点与王先生的会议”）。
   - **LLM 作用**:
     - 解析自然语言为结构化任务数据。
     - 示例：
       ```plaintext
       用户输入: "添加下午 2 点的项目会议"
       LLM 输出: 
       {
         "event_id": "auto_generated_id",
         "title": "项目会议",
         "time": "2024-12-25T14:00:00",
         "location": null,
         "priority": 3
       }
       ```

#### **2. 动态调整与解释**
   - **需求**:
     用户需要 Agent 提供任务调整理由和优化建议。
   - **LLM 作用**:
     - 生成易理解的解释：
       ```plaintext
       "由于下午 2 点有会议，建议将上午 10 点的医生预约推迟到 11 点。"
       ```

#### **3. 增强记忆功能**
   - **需求**:
     - 用户反馈任务状态（如“项目报告完成得很好”）。
   - **LLM 作用**:
     - 将模糊的反馈转化为结构化状态更新。
     - 示例：
       ```plaintext
       用户输入: "我完成了今天的任务"
       LLM 输出: 更新任务状态为 "Completed"
       ```

---

### **是否必须与 LLM 交互**
不一定，以下是场景区分：

#### **无需 LLM 的场景**
- 用户提供高度结构化的数据。
- 任务以固定格式存储，无需自然语言解析。

#### **需要 LLM 的场景**
- 用户输入多样化、非结构化（如自由文本任务描述）。
- 需要动态解释任务调整或优化建议。

---

### **推荐整合方案**

1. **结构化数据管理**:
   - 使用 **SemanticMap** 存储任务。
   - 通过 `Google Calendar API` 或任务数据直接更新任务状态。

2. **自然语言处理**:
   - 集成 LLM（如 `OpenAI GPT` 或 `Google PaLM`）。
   - 实现任务解析、优化解释和反馈转化。

3. **公开数据与真实交互结合**:
   - 使用 **Google Calendar API** 测试真实用户日程数据。
   - 在无真实用户数据时，结合 Kaggle 数据集模拟任务。

通过将公开数据集与 LLM 集成，能够在多任务场景下实现智能日程管理和语义化推荐。