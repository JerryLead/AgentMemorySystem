Q: 下面我们想展示一下semanticmap+semanticgraph的应用，比如创作一部小说，将主要人物，人物属性，关系等存放到semanticgraph里，然后情节、时间、创作的draft存放到semanticmap里，同时可以不断更改角色（如添加role、删除role），添加、删除关系，作者给一些基本的设定、LLM负责填充等，你想一下具体的场景、方法、体现我们这个语义数据结构的功能。

A: 构建一个 **SemanticMap + SemanticGraph** 应用来创作小说，是一个很有趣的场景。这不仅可以展示我们语义数据结构的灵活性，还能体现如何使用 LLM 在一个动态语义网络中创作内容并生成新情节。以下是一个完整的应用设计。

---

## **场景设定**：小说创作管理器

### **目标**
- **存储人物、属性和关系**：将角色、背景和他们的属性（如职业、性格、秘密）存储到 **SemanticGraph**。
- **存储情节、事件和时间线**：将情节和创作草稿存储到 **SemanticMap**，支持版本控制和实时编辑。
- **动态操作**：支持动态添加/删除角色、关系或情节。
- **辅助生成**：根据基本设定和动态变化，使用 LLM 填充故事细节。
- **语义检索与查询**：支持以下查询：
  - 查找所有与角色相关的关系或情节。
  - 基于关键词检索情节片段。
  - 自动生成角色的未来故事走向。

---

## **方法设计**

### **1. 结构设计**

#### **SemanticGraph**
- **节点**：
  - **角色**：`key = character_id`，`value = {name, traits, role, backstory}`。
    - 如：`"key": "char_1", "value": {"name": "Alice", "traits": ["brave", "curious"], "role": "protagonist"}`
  - **地点**：`key = location_id`，`value = {name, description}`。
    - 如：`"key": "loc_1", "value": {"name": "Mystic Forest", "description": "A dark, enchanted forest full of secrets."}`
- **关系**：
  - `char_1 -> char_2 (relation="friend")`：描述角色之间的关系。
  - `char_1 -> loc_1 (relation="visited")`：描述角色与地点的交互。

#### **SemanticMap**
- **键值对存储**：
  - **情节段落**：`key = plot_id`，`value = {"text": "Alice discovers a magical key in the Mystic Forest.", "timestamp": "2025-01-01", "version": 1}`。
  - **草稿版本**：`key = draft_id`，`value = {"text": "Initial draft of Alice's adventure.", "author": "AI", "timestamp": "2025-01-01"}`。
  - **时间线事件**：`key = event_id`，`value = {"description": "Battle at Mystic Forest", "characters": ["Alice", "Bob"], "time": "2025-01-02"}`。

---

### **2. 功能设计**

#### **1. 添加角色**
- 添加一个新角色节点到 **SemanticGraph**。
- 例：
  ```python
  graph.add_node(
      key="char_3",
      value={"name": "Bob", "traits": ["loyal", "strategist"], "role": "sidekick"}
  )
  ```

#### **2. 创建关系**
- 在角色之间添加关系或在角色与地点之间添加关系。
- 例：
  ```python
  graph.insert_edge("char_1", "char_2", relation="friend")
  graph.insert_edge("char_1", "loc_1", relation="visited")
  ```

#### **3. 添加情节片段**
- 将情节文本存储到 **SemanticMap**。
- 例：
  ```python
  map.insert(
      key="plot_1",
      value={"text": "Alice discovers a magical key in the Mystic Forest.", "timestamp": "2025-01-01"}
  )
  ```

#### **4. 查询功能**
- 检索所有与某角色相关的关系和情节：
  ```python
  related_entities = graph.get_links("char_1")  # 查询角色Alice的关系
  related_plots = map.retrieve_similar("Alice", k=5)  # 查询包含Alice的情节片段
  ```

#### **5. 自动生成情节**
- 通过 LLM 根据角色设定和当前情节生成新的故事片段。
- 例：
  ```python
  prompt = """
  Alice is brave and curious. She is in the Mystic Forest and just discovered a magical key. 
  What happens next?
  """
  new_plot = llm.generate(prompt)
  map.insert(key="plot_2", value={"text": new_plot, "timestamp": "2025-01-02"})
  ```

#### **6. 动态更新**
- 删除角色并更新相关情节：
  ```python
  graph.delete_node("char_3")  # 删除角色Bob
  map.delete(key="plot_3")  # 删除相关情节
  ```

#### **7. 情节分析**
- 生成时间线或事件关系图：
  ```python
  timeline = graph.retrieve_subtree("event_id")  # 按事件层次返回时间线
  ```

---

## **示例应用场景**

### **1. 创作小说的基本步骤**
1. 添加角色：Alice、Bob、Mystic Forest。
2. 创建关系：Alice 和 Bob 是朋友，Alice 访问了 Mystic Forest。
3. 添加情节：Alice 发现了一个魔法钥匙。
4. 生成后续情节：Alice 使用钥匙打开了一个秘密通道。
5. 查询：列出所有包含 Alice 的情节和她与 Bob 的关系。

---

### **2. 查询示例**

#### **查询 1：某角色的情节**
- **问题**：Alice 参与了哪些情节？
- **执行**：
  ```python
  graph.get_links("char_1")  # 返回Alice相关的关系
  map.retrieve_similar("Alice", k=5)  # 返回包含Alice的情节
  ```

#### **查询 2：角色的时间线**
- **问题**：Alice 在 Mystic Forest 的行动时间线是什么？
- **执行**：
  ```python
  graph.get_children("char_1")  # 返回Alice访问过的地点
  ```

#### **查询 3：预测情节**
- **问题**：Alice 下一步会做什么？
- **执行**：
  ```python
  prompt = """
  Alice just discovered a magical key in Mystic Forest. Generate a plausible next step for her.
  """
  next_plot = llm.generate(prompt)
  ```

---

## **代码示例**

下面是部分功能的实现代码：

```python
# 初始化
from semantic_graph import SemanticSimpleGraph, SemanticMap
from llm_tool import LLM

graph = SemanticSimpleGraph()
semantic_map = SemanticMap()
llm = LLM()

# 添加角色和地点
graph.add_node("char_1", {"name": "Alice", "traits": ["brave", "curious"], "role": "protagonist"})
graph.add_node("loc_1", {"name": "Mystic Forest", "description": "A dark forest full of secrets."})
graph.insert_edge("char_1", "loc_1", relation="visited")

# 添加情节
semantic_map.insert(
    key="plot_1",
    value={"text": "Alice discovers a magical key in the Mystic Forest.", "timestamp": "2025-01-01"}
)

# 自动生成新的情节
prompt = """
Alice is brave and curious. She is in the Mystic Forest and just discovered a magical key. 
What happens next?
"""
new_plot = llm.generate(prompt)
semantic_map.insert(key="plot_2", value={"text": new_plot, "timestamp": "2025-01-02"})

# 查询角色相关情节
related_plots = semantic_map.retrieve_similar("Alice", k=5)
for plot in related_plots:
    print(plot)

# 动态更新：删除角色
graph.delete_node("char_3")
semantic_map.delete("plot_2")
```

---

### **扩展功能**
- 支持导入已有的小说背景数据集（如 CharacterDB、MovieScript）。
- 提供基于 LangChain 或 LlamaIndex 的集成接口，用于与更复杂的 LLM 查询对接。
- 实现动态可视化（角色网络图、时间线展示）。 

这将成为一个可以灵活用于文学创作和语义查询的工具框架。


Q: 用一部经典小说来举例，比如白雪公主？详细列出每个角色，以及semanticmap里需要存的内容。

以下是使用《白雪公主》作为示例，构建 **SemanticMap + SemanticGraph** 的方法。这种结构可以清晰地描述角色、地点及其之间的关系，以及存储情节、时间线和动态内容。

---

## **SemanticGraph：角色、地点和关系**

### **角色和地点的节点**
- **角色**：
  - **白雪公主**（Snow White）
    ```python
    graph.add_node(
        "char_snow_white",
        {"name": "Snow White", "traits": ["kind", "innocent", "beautiful"], "role": "protagonist"}
    )
    ```
  - **王后**（The Evil Queen）
    ```python
    graph.add_node(
        "char_evil_queen",
        {"name": "The Evil Queen", "traits": ["jealous", "vain", "cruel"], "role": "antagonist"}
    )
    ```
  - **猎人**（The Huntsman）
    ```python
    graph.add_node(
        "char_huntsman",
        {"name": "The Huntsman", "traits": ["conflicted", "loyal"], "role": "supporting"}
    )
    ```
  - **七个小矮人**（The Seven Dwarfs）
    ```python
    graph.add_node(
        "char_dwarfs",
        {"name": "The Seven Dwarfs", "traits": ["hardworking", "caring"], "role": "allies"}
    )
    ```

- **地点**：
  - **王宫**（The Castle）
    ```python
    graph.add_node(
        "loc_castle",
        {"name": "The Castle", "description": "A grand castle ruled by the Evil Queen."}
    )
    ```
  - **森林**（The Forest）
    ```python
    graph.add_node(
        "loc_forest",
        {"name": "The Forest", "description": "A dark, magical forest where Snow White finds refuge."}
    )
    ```
  - **小矮人的家**（The Dwarfs' Cottage）
    ```python
    graph.add_node(
        "loc_cottage",
        {"name": "The Dwarfs' Cottage", "description": "A cozy cottage where Snow White lives with the dwarfs."}
    )
    ```

---

### **关系描述**
- **王后 -> 白雪公主**：`"jealous of"`
  ```python
  graph.insert_edge("char_evil_queen", "char_snow_white", relation="jealous of")
  ```
- **猎人 -> 白雪公主**：`"spared her"`
  ```python
  graph.insert_edge("char_huntsman", "char_snow_white", relation="spared her")
  ```
- **白雪公主 -> 森林**：`"escaped to"`
  ```python
  graph.insert_edge("char_snow_white", "loc_forest", relation="escaped to")
  ```
- **白雪公主 -> 七个小矮人**：`"befriended"`
  ```python
  graph.insert_edge("char_snow_white", "char_dwarfs", relation="befriended")
  ```
- **王后 -> 森林**：`"searched for"`
  ```python
  graph.insert_edge("char_evil_queen", "loc_forest", relation="searched for")
  ```

---

## **SemanticMap：情节存储**

### **存储内容**
- 每个情节作为一个段落或事件存储到 **SemanticMap** 中，包含情节文本、时间戳以及关联的角色和地点信息。

#### **关键情节**
1. **王后嫉妒白雪公主**
   ```python
   semantic_map.insert(
       key="plot_1",
       value={
           "text": "The Evil Queen grew jealous of Snow White's beauty after her magic mirror declared Snow White the fairest of them all.",
           "timestamp": "2025-01-01",
           "related_characters": ["char_evil_queen", "char_snow_white"]
       }
   )
   ```

2. **猎人放走白雪公主**
   ```python
   semantic_map.insert(
       key="plot_2",
       value={
           "text": "The Huntsman, ordered to kill Snow White, couldn't bring himself to do it. Instead, he spared her and told her to run into the forest.",
           "timestamp": "2025-01-02",
           "related_characters": ["char_huntsman", "char_snow_white"],
           "related_locations": ["loc_forest"]
       }
   )
   ```

3. **白雪公主遇见小矮人**
   ```python
   semantic_map.insert(
       key="plot_3",
       value={
           "text": "Snow White stumbled upon the Dwarfs' Cottage in the forest. She befriended the seven dwarfs and stayed with them.",
           "timestamp": "2025-01-03",
           "related_characters": ["char_snow_white", "char_dwarfs"],
           "related_locations": ["loc_cottage"]
       }
   )
   ```

4. **王后的毒苹果计划**
   ```python
   semantic_map.insert(
       key="plot_4",
       value={
           "text": "The Evil Queen, disguised as an old peddler, tricked Snow White into eating a poisoned apple, causing her to fall into a deep sleep.",
           "timestamp": "2025-01-04",
           "related_characters": ["char_evil_queen", "char_snow_white"],
           "related_locations": ["loc_cottage"]
       }
   )
   ```

5. **王子的吻**
   ```python
   semantic_map.insert(
       key="plot_5",
       value={
           "text": "A Prince, passing through the forest, found Snow White in her glass coffin. His kiss broke the spell, and Snow White awoke.",
           "timestamp": "2025-01-05",
           "related_characters": ["char_snow_white", "char_prince"],
           "related_locations": ["loc_forest"]
       }
   )
   ```

---

### **支持的查询**

#### **1. 查询角色关系**
**问题**：白雪公主有哪些关系？
```python
relationships = graph.get_links("char_snow_white")
print(relationships)
# 输出：{'char_evil_queen': 'jealous of', 'char_huntsman': 'spared her', 'char_dwarfs': 'befriended'}
```

#### **2. 查询包含角色的情节**
**问题**：包含白雪公主的情节有哪些？
```python
plots = semantic_map.retrieve_similar("Snow White", k=5)
print(plots)
# 输出：
# 1. "The Evil Queen grew jealous of Snow White's beauty after her magic mirror..."
# 2. "The Huntsman, ordered to kill Snow White, couldn't bring himself to do it..."
# 3. "Snow White stumbled upon the Dwarfs' Cottage in the forest..."
```

#### **3. 生成新的情节**
**问题**：基于当前情节，生成后续情节。
```python
prompt = """
Snow White just befriended the Seven Dwarfs and found refuge in their cottage. 
What happens next?
"""
next_plot = llm.generate(prompt)
semantic_map.insert(key="plot_6", value={"text": next_plot, "timestamp": "2025-01-06"})
print(next_plot)
# 输出（示例生成结果）：
# "The Evil Queen, using her dark magic, crafted a poison apple and disguised herself..."
```

#### **4. 时间线查询**
**问题**：按照时间顺序返回所有情节。
```python
timeline = semantic_map.retrieve_all_sorted_by("timestamp")
for event in timeline:
    print(event["text"])
```

---

## **扩展应用**

1. **动态添加新角色或情节**：
   - 添加白雪公主的父亲（国王）或王子的背景故事。
   - 插入更多小矮人互动情节。

2. **实时编辑**：
   - 修改白雪公主的性格为“更加坚强”：
     ```python
     graph.update_node("char_snow_white", new_value={"traits": ["kind", "strong", "beautiful"]})
     ```

3. **多层查询**：
   - 查找与白雪公主直接和间接相关的情节（多跳查询）。
   - 返回与某情节关联的所有角色。

---

这个示例展示了如何用 **SemanticMap + SemanticGraph** 表示复杂的故事设定，同时支持动态编辑和基于语义的查询。小说创作过程可以通过 LLM 辅助，自动生成细节情节和对话。


Q: 这个例子前后不一致，一会是猎人、一会是王子，关系“spared her”也不对，重新生成这个例子，与下面的白雪公主的情节一致：故事简介

1. 皇后问魔镜

2. 白雪公主在森林里

3. 小矮人发现白雪公主睡着了

4. 皇后探访白雪公主

5. 小矮人提醒白雪公主

6. 皇后毒死白雪公主

7. 王子把白雪公主唤醒

8. 皇后到达婚礼现场
在原本的格林童话版本中，白雪公主的故事是这样的：

很久很久以前，某个国家的皇后在冬季生下一个女孩，她皮肤纯白如雪，嘴唇赤红如血，头发黑如乌木一样漂亮，因此她被命名为白雪公主，皇后在生下公主不久后就过世了，国王另娶了一个美丽骄傲，狠毒邪恶的女人当皇后，同时她也成为了白雪公主的继母（初稿为白雪公主生母），一开始新皇后也非常疼爱白雪公主。新皇后有一面魔镜，她常常问魔镜：“魔镜魔镜，谁是世界上最美的女人？”魔镜总是回答：“皇后，当然是您了。” 但白雪公主越长越大，也变得愈来愈美丽。当她七岁时，她的容貌比皇后更漂亮了，有一天，魔镜回答皇后说：“白雪公主。”从此新皇后便开始视白雪公主为眼中钉，肉中刺，所以一心想把她除掉。

皇后非常嫉妒白雪公主的美貌，因此她命令一名猎人带白雪公主到森林中，并将她杀掉。为了确认白雪公主已死，皇后要猎人事成之后，带着白雪公主的肺和肝回来，作为证明。猎人带着白雪公主到森林中，却发现自己无法下手杀害这个女孩，猎人放了白雪公主，猎了野猪，取野猪的肺和肝向皇后交差。

在森林中，白雪公主发现一个小小的农舍，这个农舍属于七个小矮人，她在这个农舍中住了下来。此时，皇后又再度问魔镜：“魔镜魔镜，谁是世界上最美丽的女人？”魔镜回答：“白雪公主。”于是皇后伪装成一个农妇，到森林中拜访白雪公主，并给她一个毒苹果（有些版本中，一开始是王后先后假扮卖丝带以及卖梳子的商妇来谋害白雪公主，但是都以失败结束），当白雪公主咬下苹果，立即昏了过去。当七矮人发现她时，只能哀恸地将她放在一个玻璃棺材中。

时光流逝，有一个国家的王子经过这座森林，发现躺在玻璃棺中的白雪公主。王子被白雪公主的美丽所吸引并且爱上了她。他向矮人们要求，让他带走玻璃棺。王子和他的随从在搬运的过程中，有人不小心被绊倒，这一摇晃，让那片毒苹果从白雪公主的口中吐了出来，白雪公主也因此苏醒。王子向白雪公主表明了爱意，决定结婚，并订下婚期。

虚荣任性的皇后认为白雪公主已死，她再度问魔镜谁是这个世界上最美丽的女人，魔镜的回答使她十分愤怒：“你是这里最美丽的人；但更美丽的人是将要成为新皇后的人。”这时王子的婚宴邀请送来，不知道新皇后就是白雪公主的皇后应邀赴宴，发现继女仍然在生。惧怒交加的皇后企图在婚宴上制造混乱，结果王子下令强行为她穿上烧红的铁鞋不断跳舞至死，好使婚礼能顺利进行。


A: 以下是基于《白雪公主》故事重述的 **SemanticMap + SemanticGraph** 结构化存储与查询的完整实现示例。

---

## **SemanticGraph：角色、地点和关系**

### **角色和地点的节点**
- **角色**：
  - **白雪公主**（Snow White）
  - **王后**（The Evil Queen）
  - **猎人**（The Huntsman）
  - **七个小矮人**（The Seven Dwarfs）
  - **王子**（The Prince）

- **地点**：
  - **王宫**（The Castle）
  - **森林**（The Forest）
  - **小矮人的小屋**（The Dwarfs' Cottage）
  - **玻璃棺**（The Glass Coffin）
  - **婚礼大厅**（The Wedding Hall）

### **关系描述**
我们将故事的角色和地点通过图的关系进行建模：
- **王后** → **白雪公主**（嫉妒 `jealous of`）
- **猎人** → **白雪公主**（放走 `spared`）
- **白雪公主** → **森林**（逃入 `escaped to`）
- **白雪公主** → **七个小矮人**（寄居 `sheltered by`）
- **王后** → **森林**（搜索 `searched for`）
- **王子** → **白雪公主**（爱上 `fell in love with`）
- **婚礼大厅** → **皇后**（见证 `witnessed`）

### **代码实现**
```python
from semantic_graph import SemanticSimpleGraph

# 初始化图
graph = SemanticSimpleGraph()

# 添加角色节点
graph.add_node("char_snow_white", {"name": "Snow White", "traits": ["kind", "innocent", "beautiful"], "role": "protagonist"})
graph.add_node("char_evil_queen", {"name": "The Evil Queen", "traits": ["jealous", "cruel", "vain"], "role": "antagonist"})
graph.add_node("char_huntsman", {"name": "The Huntsman", "traits": ["conflicted", "loyal"], "role": "supporting"})
graph.add_node("char_dwarfs", {"name": "The Seven Dwarfs", "traits": ["hardworking", "caring"], "role": "allies"})
graph.add_node("char_prince", {"name": "The Prince", "traits": ["brave", "romantic"], "role": "protagonist"})

# 添加地点节点
graph.add_node("loc_castle", {"name": "The Castle", "description": "Home of the Evil Queen and Snow White."})
graph.add_node("loc_forest", {"name": "The Forest", "description": "A dark forest where Snow White escapes to."})
graph.add_node("loc_cottage", {"name": "The Dwarfs' Cottage", "description": "The cozy home of the Seven Dwarfs."})
graph.add_node("loc_glass_coffin", {"name": "The Glass Coffin", "description": "Where Snow White lies in enchanted sleep."})
graph.add_node("loc_wedding_hall", {"name": "The Wedding Hall", "description": "The place where Snow White and the Prince's wedding is held."})

# 定义关系
graph.insert_edge("char_evil_queen", "char_snow_white", relation="jealous of")
graph.insert_edge("char_huntsman", "char_snow_white", relation="spared")
graph.insert_edge("char_snow_white", "loc_forest", relation="escaped to")
graph.insert_edge("char_snow_white", "char_dwarfs", relation="sheltered by")
graph.insert_edge("char_evil_queen", "loc_forest", relation="searched for")
graph.insert_edge("char_prince", "char_snow_white", relation="fell in love with")
graph.insert_edge("loc_wedding_hall", "char_evil_queen", relation="witnessed")

# 查看结果
print("Relationship of Snow White:", graph.get_links("char_snow_white"))
```

---

## **SemanticMap：情节存储**

### **存储情节**
我们将故事的主要情节按照时间顺序存储为片段，添加相关角色和地点信息。

#### **关键情节存储**
```python
from semantic_map import SemanticMap

# 初始化语义地图
semantic_map = SemanticMap()

# 添加故事情节
semantic_map.insert(
    key="scene_1",
    value={
        "text": "The Evil Queen asks her magic mirror, 'Who is the fairest of them all?' The mirror answers, 'Snow White.'",
        "timestamp": "2025-01-01",
        "related_characters": ["char_evil_queen", "char_snow_white"],
        "related_locations": ["loc_castle"]
    }
)

semantic_map.insert(
    key="scene_2",
    value={
        "text": "The Huntsman spares Snow White's life and lets her escape into the forest.",
        "timestamp": "2025-01-02",
        "related_characters": ["char_huntsman", "char_snow_white"],
        "related_locations": ["loc_forest"]
    }
)

semantic_map.insert(
    key="scene_3",
    value={
        "text": "Snow White finds refuge in the cottage of the Seven Dwarfs.",
        "timestamp": "2025-01-03",
        "related_characters": ["char_snow_white", "char_dwarfs"],
        "related_locations": ["loc_cottage"]
    }
)

semantic_map.insert(
    key="scene_4",
    value={
        "text": "The Evil Queen, disguised as an old woman, gives Snow White a poisoned apple.",
        "timestamp": "2025-01-04",
        "related_characters": ["char_evil_queen", "char_snow_white"],
        "related_locations": ["loc_cottage"]
    }
)

semantic_map.insert(
    key="scene_5",
    value={
        "text": "Snow White is placed in a glass coffin by the dwarfs, where she remains in enchanted sleep.",
        "timestamp": "2025-01-05",
        "related_characters": ["char_snow_white", "char_dwarfs"],
        "related_locations": ["loc_glass_coffin"]
    }
)

semantic_map.insert(
    key="scene_6",
    value={
        "text": "The Prince discovers Snow White in the glass coffin, and his kiss awakens her.",
        "timestamp": "2025-01-06",
        "related_characters": ["char_prince", "char_snow_white"],
        "related_locations": ["loc_glass_coffin"]
    }
)

semantic_map.insert(
    key="scene_7",
    value={
        "text": "The Evil Queen attends Snow White's wedding and is punished by dancing in red-hot iron shoes.",
        "timestamp": "2025-01-07",
        "related_characters": ["char_evil_queen", "char_snow_white", "char_prince"],
        "related_locations": ["loc_wedding_hall"]
    }
)
```

---

### **支持的查询功能**

#### **1. 查询角色关系**
**问题**：白雪公主的相关关系？
```python
relationships = graph.get_links("char_snow_white")
print(relationships)
# 输出：{'char_evil_queen': 'jealous of', 'char_huntsman': 'spared', 'char_dwarfs': 'sheltered by', 'char_prince': 'fell in love with'}
```

#### **2. 查询某角色参与的情节**
**问题**：猎人参与了哪些情节？
```python
hunter_scenes = semantic_map.retrieve_similar("Huntsman", k=3)
print(hunter_scenes)
# 输出：
# 1. "The Huntsman spares Snow White's life and lets her escape into the forest."
```

#### **3. 时间线情节查询**
**问题**：按时间顺序获取完整故事情节。
```python
timeline = semantic_map.retrieve_all_sorted_by("timestamp")
for event in timeline:
    print(event["text"])
```

#### **4. 添加新情节**
**问题**：添加故事后续内容。
```python
semantic_map.insert(
    key="scene_8",
    value={
        "text": "Snow White and the Prince live happily ever after in the castle.",
        "timestamp": "2025-01-08",
        "related_characters": ["char_snow_white", "char_prince"],
        "related_locations": ["loc_castle"]
    }
)
```

---

### **应用示例**
通过 **SemanticMap + SemanticGraph** 的能力，可以：
1. **动态构建故事角色和关系**。
2. **精确存储情节与时间线**。
3. **支持角色和情节的多跳查询**。
4. **结合 LLM 自动扩展情节或生成新内容**。



Q: 把上面的描述都翻译为英文

A: ### Using **SemanticMap + SemanticGraph** for "Snow White" Story Management

Below is a structured example that uses **SemanticMap + SemanticGraph** to represent characters, relationships, plot, and events from the classic fairy tale "Snow White."

---

## **SemanticGraph: Characters, Locations, and Relationships**

### **Nodes for Characters and Locations**
- **Characters**:
  - **Snow White**
  - **The Evil Queen**
  - **The Huntsman**
  - **The Seven Dwarfs**
  - **The Prince**

- **Locations**:
  - **The Castle**
  - **The Forest**
  - **The Dwarfs' Cottage**
  - **The Glass Coffin**
  - **The Wedding Hall**

### **Relationships**
The story's characters and locations are connected using a graph structure:
- **The Evil Queen** → **Snow White** (`jealous of`)
- **The Huntsman** → **Snow White** (`spared`)
- **Snow White** → **The Forest** (`escaped to`)
- **Snow White** → **The Seven Dwarfs** (`sheltered by`)
- **The Evil Queen** → **The Forest** (`searched for`)
- **The Prince** → **Snow White** (`fell in love with`)
- **The Wedding Hall** → **The Evil Queen** (`witnessed`)

### **Code Implementation**
```python
from semantic_graph import SemanticSimpleGraph

# Initialize the graph
graph = SemanticSimpleGraph()

# Add character nodes
graph.add_node("char_snow_white", {"name": "Snow White", "traits": ["kind", "innocent", "beautiful"], "role": "protagonist"})
graph.add_node("char_evil_queen", {"name": "The Evil Queen", "traits": ["jealous", "cruel", "vain"], "role": "antagonist"})
graph.add_node("char_huntsman", {"name": "The Huntsman", "traits": ["conflicted", "loyal"], "role": "supporting"})
graph.add_node("char_dwarfs", {"name": "The Seven Dwarfs", "traits": ["hardworking", "caring"], "role": "allies"})
graph.add_node("char_prince", {"name": "The Prince", "traits": ["brave", "romantic"], "role": "protagonist"})

# Add location nodes
graph.add_node("loc_castle", {"name": "The Castle", "description": "Home of the Evil Queen and Snow White."})
graph.add_node("loc_forest", {"name": "The Forest", "description": "A dark forest where Snow White escapes to."})
graph.add_node("loc_cottage", {"name": "The Dwarfs' Cottage", "description": "The cozy home of the Seven Dwarfs."})
graph.add_node("loc_glass_coffin", {"name": "The Glass Coffin", "description": "Where Snow White lies in enchanted sleep."})
graph.add_node("loc_wedding_hall", {"name": "The Wedding Hall", "description": "The place where Snow White and the Prince's wedding is held."})

# Define relationships
graph.insert_edge("char_evil_queen", "char_snow_white", relation="jealous of")
graph.insert_edge("char_huntsman", "char_snow_white", relation="spared")
graph.insert_edge("char_snow_white", "loc_forest", relation="escaped to")
graph.insert_edge("char_snow_white", "char_dwarfs", relation="sheltered by")
graph.insert_edge("char_evil_queen", "loc_forest", relation="searched for")
graph.insert_edge("char_prince", "char_snow_white", relation="fell in love with")
graph.insert_edge("loc_wedding_hall", "char_evil_queen", relation="witnessed")

# View the results
print("Relationship of Snow White:", graph.get_links("char_snow_white"))
```

---

## **SemanticMap: Plot Storage**

### **Storing Key Events**
The major plot points are stored in chronological order, with relevant characters and locations linked to each event.

#### **Key Plot Storage**
```python
from semantic_map import SemanticMap

# Initialize SemanticMap
semantic_map = SemanticMap()

# Add story events
semantic_map.insert(
    key="scene_1",
    value={
        "text": "The Evil Queen asks her magic mirror, 'Who is the fairest of them all?' The mirror answers, 'Snow White.'",
        "timestamp": "2025-01-01",
        "related_characters": ["char_evil_queen", "char_snow_white"],
        "related_locations": ["loc_castle"]
    }
)

semantic_map.insert(
    key="scene_2",
    value={
        "text": "The Huntsman spares Snow White's life and lets her escape into the forest.",
        "timestamp": "2025-01-02",
        "related_characters": ["char_huntsman", "char_snow_white"],
        "related_locations": ["loc_forest"]
    }
)

semantic_map.insert(
    key="scene_3",
    value={
        "text": "Snow White finds refuge in the cottage of the Seven Dwarfs.",
        "timestamp": "2025-01-03",
        "related_characters": ["char_snow_white", "char_dwarfs"],
        "related_locations": ["loc_cottage"]
    }
)

semantic_map.insert(
    key="scene_4",
    value={
        "text": "The Evil Queen, disguised as an old woman, gives Snow White a poisoned apple.",
        "timestamp": "2025-01-04",
        "related_characters": ["char_evil_queen", "char_snow_white"],
        "related_locations": ["loc_cottage"]
    }
)

semantic_map.insert(
    key="scene_5",
    value={
        "text": "Snow White is placed in a glass coffin by the dwarfs, where she remains in enchanted sleep.",
        "timestamp": "2025-01-05",
        "related_characters": ["char_snow_white", "char_dwarfs"],
        "related_locations": ["loc_glass_coffin"]
    }
)

semantic_map.insert(
    key="scene_6",
    value={
        "text": "The Prince discovers Snow White in the glass coffin, and his kiss awakens her.",
        "timestamp": "2025-01-06",
        "related_characters": ["char_prince", "char_snow_white"],
        "related_locations": ["loc_glass_coffin"]
    }
)

semantic_map.insert(
    key="scene_7",
    value={
        "text": "The Evil Queen attends Snow White's wedding and is punished by dancing in red-hot iron shoes.",
        "timestamp": "2025-01-07",
        "related_characters": ["char_evil_queen", "char_snow_white", "char_prince"],
        "related_locations": ["loc_wedding_hall"]
    }
)
```

---

### **Query Examples**

#### **1. Query Relationships**
**Question**: What are the relationships involving Snow White?
```python
relationships = graph.get_links("char_snow_white")
print(relationships)
# Output: {'char_evil_queen': 'jealous of', 'char_huntsman': 'spared', 'char_dwarfs': 'sheltered by', 'char_prince': 'fell in love with'}
```

#### **2. Query Scenes Involving a Character**
**Question**: What scenes involve the Huntsman?
```python
hunter_scenes = semantic_map.retrieve_similar("Huntsman", k=3)
print(hunter_scenes)
# Output:
# 1. "The Huntsman spares Snow White's life and lets her escape into the forest."
```

#### **3. Query Timeline**
**Question**: Retrieve the full timeline of the story in chronological order.
```python
timeline = semantic_map.retrieve_all_sorted_by("timestamp")
for event in timeline:
    print(event["text"])
```

#### **4. Add New Scene**
**Question**: Add future events to the story.
```python
semantic_map.insert(
    key="scene_8",
    value={
        "text": "Snow White and the Prince live happily ever after in the castle.",
        "timestamp": "2025-01-08",
        "related_characters": ["char_snow_white", "char_prince"],
        "related_locations": ["loc_castle"]
    }
)
```

---

### **Applications**
By leveraging **SemanticMap + SemanticGraph**, we can:
1. **Dynamically build story characters and relationships.**
2. **Accurately store plot events and timelines.**
3. **Support multi-hop queries for deeper insights.**
4. **Integrate with LLMs to extend storylines or generate new content dynamically.**