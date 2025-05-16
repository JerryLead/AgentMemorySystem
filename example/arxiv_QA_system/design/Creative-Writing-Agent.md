---

### Case 5: **Creative Writing Agent**

#### 数据集  
- 使用公开的文学作品数据集 [Project Gutenberg](https://www.gutenberg.org/)。
- 包含小说章节、标题、角色和描述。

#### 流程  

1. **用户任务**：
   - 用户希望创作一部小说，要求：
     - 从经典小说中提取灵感。
     - 根据用户描述生成角色设定和情节框架。

2. **Agent 步骤**：
   - **Step 1: 数据存储**：
     - 将经典文学作品的章节、角色和设定存入 `SemanticMap`。
   - **Step 2: 生成情节框架**：
     - 用户输入“我想写一部关于复仇的小说，主角是一位孤儿”。
     - Agent 检索相关灵感（如《基督山伯爵》），生成框架：
       ```
       - 主角：一位被迫离家的孤儿。
       - 目标：揭开家族秘密，完成复仇。
       ```
   - **Step 3: 提供建议**：
     - Agent 动态补充设定（如地理背景、冲突场景）。
   - **Step 4: 更新故事情节**：
     - 用户调整情节后，Agent 更新存储内容。

#### 体现功能  
- **记忆与交互**：根据用户描述生成动态内容。
- **工具使用**：结合 LLM 提供创意支持。
- **动态更新**：实时调整故事设定。

---

### **优化后的 Creative Writing Agent 案例设计**

以下是对 **Case 5: Creative Writing Agent** 的优化设计，使用 **SemanticMap** 提供强大的存储、检索和动态交互能力，结合 LLM 实现创意写作支持。

---

### **核心功能与扩展**

1. **多模态存储与语义检索**：
   - 存储经典文学作品的章节、角色设定、情节和描述。
   - 基于文本语义检索灵感来源。

2. **创意生成与动态交互**：
   - 用户提供初始设定，Agent 动态生成角色和情节框架。
   - 提供多样化的情节建议。

3. **实时更新与管理**：
   - 支持用户调整后的情节设定更新，保留完整的故事发展路径。

---

### **数据存储设计**

每部经典文学作品存储为以下格式：

```json
{
  "title": "The Count of Monte Cristo",
  "chapter": "Chapter 1: Marseilles",
  "characters": ["Edmond Dantès", "Fernand Mondego", "Danglars"],
  "setting": {
    "time": "1815",
    "place": "Marseilles, France"
  },
  "summary": "Edmond Dantès, a sailor, returns to Marseilles...",
  "themes": ["revenge", "betrayal", "justice"]
}
```

---

### **优化的 EnhancedSemanticMap**

#### **实现**

```python
class EnhancedSemanticMap(SemanticMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def insert_literature(self, literature_data):
        """
        插入文学作品数据。
        literature_data 格式示例：
        {
            "title": "The Count of Monte Cristo",
            "chapter": "Chapter 1: Marseilles",
            "characters": ["Edmond Dantès", "Fernand Mondego", "Danglars"],
            "setting": {"time": "1815", "place": "Marseilles, France"},
            "summary": "Edmond Dantès, a sailor, returns to Marseilles...",
            "themes": ["revenge", "betrayal", "justice"]
        }
        """
        key = f"{literature_data['title']} - {literature_data['chapter']}"
        self.insert(key, literature_data)

    def retrieve_inspiration(self, user_query, k=3):
        """
        根据用户描述检索文学灵感。
        """
        return self.retrieve_similar(user_query, k)

    def generate_framework(self, user_description, inspiration):
        """
        基于用户描述和检索结果生成情节框架。
        """
        prompt = f"""
        User wants to write a novel with the following description: {user_description}.
        Relevant inspirations:
        {json.dumps(inspiration, indent=2)}

        Based on these, generate a plot framework with:
        - Main Character
        - Objective
        - Key Challenges
        - Resolution
        """
        response = self.llm(prompt, max_length=512)
        return response[0]["generated_text"]

    def update_story(self, story_title, updates):
        """
        更新存储中的故事情节。
        """
        for idx, (key, value, _) in enumerate(self.data):
            if story_title in key:
                self.data[idx] = (key, {**value, **updates}, self._get_embedding(key))
        self.build_or_update_index()
```

---

### **Creative Writing Agent 案例实现**

#### **Step 1: 数据存储**

```python
# 初始化 EnhancedSemanticMap
literature_map = EnhancedSemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 示例文学作品数据
literatures = [
    {
        "title": "The Count of Monte Cristo",
        "chapter": "Chapter 1: Marseilles",
        "characters": ["Edmond Dantès", "Fernand Mondego", "Danglars"],
        "setting": {"time": "1815", "place": "Marseilles, France"},
        "summary": "Edmond Dantès, a sailor, returns to Marseilles...",
        "themes": ["revenge", "betrayal", "justice"],
    },
    {
        "title": "Great Expectations",
        "chapter": "Chapter 1: The Convict",
        "characters": ["Pip", "Magwitch", "Joe Gargery"],
        "setting": {"time": "Early 19th century", "place": "Kent, England"},
        "summary": "Young Pip encounters an escaped convict...",
        "themes": ["ambition", "class", "redemption"],
    },
]

# 插入文学数据
for literature in literatures:
    literature_map.insert_literature(literature)

# 构建索引
literature_map.build_or_update_index()
```

---

#### **Step 2: 生成情节框架**

##### **用户输入描述**

```python
# 用户输入：描述一部复仇小说
user_description = "我想写一部关于复仇的小说，主角是一位孤儿。"

# 检索灵感来源
inspirations = literature_map.retrieve_inspiration(user_description, k=2)

print("\n检索的灵感来源：")
for inspiration in inspirations:
    print(f"Title: {inspiration['key']}")
    print(f"Themes: {inspiration['value']['themes']}")
```

##### **生成情节框架**

```python
# 基于灵感生成情节框架
framework = literature_map.generate_framework(user_description, inspirations)

print("\n生成的情节框架：")
print(framework)
```

---

#### **Step 3: 提供动态建议**

```python
# 用户希望扩展背景设定
user_request = "Add more details about the geographical setting and challenges faced by the orphan."

# 基于当前框架生成建议
suggestions = literature_map.generate_framework(user_request, inspirations)

print("\n动态建议：")
print(suggestions)
```

---

#### **Step 4: 更新故事情节**

```python
# 更新故事情节
updates = {
    "characters": ["The Orphan", "Mysterious Benefactor", "Ruthless Antagonist"],
    "setting": {"time": "1870", "place": "Victorian London"},
    "summary": "An orphan uncovers a dark family secret and seeks revenge against those who wronged them."
}

literature_map.update_story("User Story: Revenge of the Orphan", updates)

# 检索更新后的故事
updated_story = literature_map.retrieve_similar("Revenge of the Orphan", k=1)
print("\n更新后的故事情节：")
print(updated_story[0]["value"])
```

---

### **功能输出示例**

#### **检索的灵感来源**

```plaintext
检索的灵感来源：
Title: The Count of Monte Cristo - Chapter 1: Marseilles
Themes: ['revenge', 'betrayal', 'justice']

Title: Great Expectations - Chapter 1: The Convict
Themes: ['ambition', 'class', 'redemption']
```

#### **生成的情节框架**

```plaintext
生成的情节框架：
- Main Character: An orphan forced to leave their home.
- Objective: Discover the truth about their family and seek revenge.
- Key Challenges: Overcome betrayal, gather allies, and evade enemies.
- Resolution: Achieve justice and find redemption.
```

#### **动态建议**

```plaintext
动态建议：
Expand on Victorian London as the setting, with narrow streets and a mysterious atmosphere.
Introduce conflicts involving social class and hidden identities.
```

#### **更新后的故事情节**

```plaintext
更新后的故事情节：
{
    "characters": ["The Orphan", "Mysterious Benefactor", "Ruthless Antagonist"],
    "setting": {"time": "1870", "place": "Victorian London"},
    "summary": "An orphan uncovers a dark family secret and seeks revenge against those who wronged them."
}
```

---

### **扩展与改进**

1. **灵感检索扩展**：
   - 支持多条件查询（如特定主题、时间或地理背景）。

2. **故事版本管理**：
   - 记录用户的每次更新，支持版本回溯和比较。

3. **高阶生成能力**：
   - 引入多步对话生成更复杂的角色和情节细节。

此优化方案展示了 **SemanticMap** 在创意写作领域的广泛应用能力，结合语义检索和动态生成，实现真正的 AI 辅助创作。

Q: 也就是说已有的小说可以用semanticmap存储，新写的小说也可以存semanticmap不断存储？如果是的话，重新设计优化，并详细介绍这个case

是的，基于 **SemanticMap** 的设计，可以存储已有的经典文学作品，也可以动态地存储用户创作的新小说，并随着创作过程不断更新和扩展。以下是对 **Creative Writing Agent** 案例的重新设计和优化，支持双向存储与动态创作。

---

### **优化设计**

#### **核心目标**

1. **存储与检索**：
   - 存储经典文学作品及其结构化内容（章节、角色、主题等）。
   - 存储用户创作的小说，支持情节动态更新和扩展。

2. **语义化灵感生成**：
   - 基于用户输入，检索类似的经典作品片段以激发灵感。
   - 为用户提供语义化建议，包括角色、背景和情节设定。

3. **动态扩展与创作**：
   - 用户可以实时更新已有创作内容。
   - 保存创作历史，支持版本化管理。

---

### **SemanticMap 优化实现**

#### **EnhancedSemanticMap**

该子类扩展了 SemanticMap，增加对双向存储和灵活管理的支持。

```python
class EnhancedSemanticMap(SemanticMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def insert_novel(self, novel_data, is_user_created=False):
        """
        插入文学作品或用户创作的小说。
        novel_data 格式示例：
        {
            "title": "User Story: Revenge of the Orphan",
            "chapter": "Prologue",
            "characters": ["The Orphan"],
            "setting": {"time": "1870", "place": "Victorian London"},
            "summary": "An orphan uncovers a dark family secret and seeks revenge.",
            "themes": ["revenge", "family", "justice"],
            "is_user_created": True  # 默认标记为用户创作
        }
        """
        key = f"{'User Story' if is_user_created else 'Classic'}: {novel_data['title']} - {novel_data['chapter']}"
        self.insert(key, novel_data)

    def retrieve_inspiration(self, user_query, k=3):
        """
        检索灵感，包括经典作品和用户创作的小说。
        """
        return self.retrieve_similar(user_query, k)

    def update_novel(self, novel_title, chapter, updates):
        """
        更新用户创作的小说章节。
        """
        key = f"User Story: {novel_title} - {chapter}"
        for idx, (stored_key, stored_value, _) in enumerate(self.data):
            if key == stored_key:
                self.data[idx] = (stored_key, {**stored_value, **updates}, self._get_embedding(stored_key))
                break
        self.build_or_update_index()

    def retrieve_versions(self, novel_title):
        """
        检索用户创作的所有章节版本。
        """
        return [item for item in self.data if f"User Story: {novel_title}" in item[0]]
```

---

### **Case 5: Creative Writing Agent**

以下是完整的案例设计和实现，展示如何利用 **SemanticMap** 支持经典作品存储与用户创作。

---

#### **Step 1: 存储经典作品与用户创作**

```python
# 初始化 EnhancedSemanticMap
literature_map = EnhancedSemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 示例经典文学作品数据
classic_literatures = [
    {
        "title": "The Count of Monte Cristo",
        "chapter": "Chapter 1: Marseilles",
        "characters": ["Edmond Dantès", "Fernand Mondego", "Danglars"],
        "setting": {"time": "1815", "place": "Marseilles, France"},
        "summary": "Edmond Dantès, a sailor, returns to Marseilles...",
        "themes": ["revenge", "betrayal", "justice"],
    },
    {
        "title": "Great Expectations",
        "chapter": "Chapter 1: The Convict",
        "characters": ["Pip", "Magwitch", "Joe Gargery"],
        "setting": {"time": "Early 19th century", "place": "Kent, England"},
        "summary": "Young Pip encounters an escaped convict...",
        "themes": ["ambition", "class", "redemption"],
    },
]

# 插入经典文学数据
for literature in classic_literatures:
    literature_map.insert_novel(literature, is_user_created=False)

# 插入用户创作的小说
user_novel = {
    "title": "Revenge of the Orphan",
    "chapter": "Prologue",
    "characters": ["The Orphan"],
    "setting": {"time": "1870", "place": "Victorian London"},
    "summary": "An orphan uncovers a dark family secret and seeks revenge.",
    "themes": ["revenge", "family", "justice"],
}
literature_map.insert_novel(user_novel, is_user_created=True)

# 构建索引
literature_map.build_or_update_index()
```

---

#### **Step 2: 用户描述生成情节框架**

##### **检索灵感**

```python
# 用户输入：描述一部复仇小说
user_description = "我想写一部关于复仇的小说，主角是一位孤儿。"

# 检索灵感来源
inspirations = literature_map.retrieve_inspiration(user_description, k=2)

print("\n检索的灵感来源：")
for inspiration in inspirations:
    print(f"Title: {inspiration['key']}")
    print(f"Themes: {inspiration['value']['themes']}")
```

##### **生成情节框架**

```python
# 基于灵感生成情节框架
framework = literature_map.generate_framework(user_description, inspirations)

print("\n生成的情节框架：")
print(framework)
```

---

#### **Step 3: 更新与扩展**

##### **扩展故事设定**

```python
# 用户希望扩展设定
user_request = "Add more details about the antagonist's motives and the orphan's allies."
suggestions = literature_map.generate_framework(user_request, inspirations)

print("\n扩展建议：")
print(suggestions)
```

##### **更新章节内容**

```python
# 更新用户小说章节
updates = {
    "characters": ["The Orphan", "Mysterious Benefactor", "Ruthless Antagonist"],
    "setting": {"time": "1870", "place": "Victorian London"},
    "summary": "An orphan uncovers a dark family secret and faces a powerful antagonist while finding unlikely allies.",
}
literature_map.update_novel("Revenge of the Orphan", "Prologue", updates)

# 检索更新后的章节
updated_chapter = literature_map.retrieve_versions("Revenge of the Orphan")
print("\n更新后的章节：")
for chapter in updated_chapter:
    print(chapter[1])  # 打印章节内容
```

---

### **功能展示**

#### **检索的灵感来源**

```plaintext
检索的灵感来源：
Title: Classic: The Count of Monte Cristo - Chapter 1: Marseilles
Themes: ['revenge', 'betrayal', 'justice']

Title: Classic: Great Expectations - Chapter 1: The Convict
Themes: ['ambition', 'class', 'redemption']
```

#### **生成的情节框架**

```plaintext
生成的情节框架：
- Main Character: An orphan forced to leave their home.
- Objective: Discover the truth about their family and seek revenge.
- Key Challenges: Overcome betrayal, gather allies, and evade enemies.
- Resolution: Achieve justice and find redemption.
```

#### **扩展建议**

```plaintext
扩展建议：
- Antagonist: A ruthless industrial magnate with hidden motives tied to the orphan's family.
- Allies: A mysterious benefactor and a street-smart companion who help the orphan navigate Victorian London.
```

#### **更新后的章节**

```plaintext
更新后的章节：
{
    "characters": ["The Orphan", "Mysterious Benefactor", "Ruthless Antagonist"],
    "setting": {"time": "1870", "place": "Victorian London"},
    "summary": "An orphan uncovers a dark family secret and faces a powerful antagonist while finding unlikely allies.",
    "themes": ["revenge", "family", "justice"],
}
```

---

### **优势与扩展**

1. **双向存储**：
   - 支持经典文学作品和用户创作的小说双向存储与管理。

2. **动态更新**：
   - 实现创作内容的实时调整与扩展，支持复杂情节的发展。

3. **灵活检索**：
   - 结合 LLM 提供语义增强的灵感生成和情节建议。

4. **版本管理**：
   - 支持用户创作的章节版本化管理，便于追踪修改历史。

该实现充分展示了 **SemanticMap** 在创意写作场景下的强大应用能力，为用户提供了一种高效的文学创作辅助工具。