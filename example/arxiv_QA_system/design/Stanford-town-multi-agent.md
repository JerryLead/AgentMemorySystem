Q: 关于SemanticMap+SemanticGraph的应用，是不是可以举一个多agent协同的例子，比如多agent对话（如斯坦福小镇每个角色是一个agent，可以与其他agent交流，对话等），每个agent可以有一个自己的semanticmap来存自己的信息、别人发给自己的信息、自己做的事情等。斯坦福小镇可以参考https://arxiv.org/pdf/2304.03442 ,可以的话详细描述场景，semanticmap/graph里存的内容，数据集，代码等。也可以参考其代码，https://github.com/joonspk-research/generative_agents



A：以下是一个结合了 **[Generative Agents (斯坦福小镇场景)](https://arxiv.org/pdf/2304.03442)** 与 **SemanticMap + SemanticGraph** 的多智能体系统设计示例。该示例参考了论文与代码（[joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents)）中关于**内存流（memory stream）**和**内存检索（memory retrieve）**的思想，为每个智能体提供“**记忆存储**”和“**关系图**”，以便在模拟环境中进行多轮对话、协作、行为规划等活动。

---

## **一、背景：Generative Agents 与多智能体模拟**

在论文 [“Generative Agents: Interactive Simulacra of Human Behavior” (Park et al., 2023)](https://arxiv.org/pdf/2304.03442) 中，作者提出了模拟多人角色（agent）的方法，每个角色拥有：
1. **Memory Stream**：记录角色过去的对话、事件、行动，以实现对过去信息的记忆和可检索性。  
2. **Reflection**：角色可在关键时刻对过去记忆进行总结、分析，从而更新自我认知或目标。  
3. **Planning**：角色能基于过去记忆进行短期或长期计划，实现更逼真的角色行为。

斯坦福小镇（Stanford Small Town）是一个示例模拟环境，里面的多名角色（如 John、Alice 等）能根据各自的记忆、属性和目标进行互动。

---

## **二、结合 SemanticMap + SemanticGraph 的思路**

我们要将上面提到的**记忆（Memory Stream）**与**关系（Relationship）**融合到 **SemanticMap** 与 **SemanticGraph**：

1. **SemanticMap**：  
   - 用于存储每个智能体的记忆内容（memory entries）及其嵌入（embedding）。  
   - 可以提供**向量检索**（memory retrieve）——在需要反思或对话时，检索出最相关的记忆片段。

2. **SemanticGraph**：  
   - 用于表示智能体之间的关系（社交图）、事件/地点等节点，以及多跳关系查询。  
   - 也可以存储更高级的“谁与谁最近对话过、谁对于谁的印象”等信息，供多智能体交互和行为推理时使用。

通过这种设计，每个智能体都拥有：
- **一个 SemanticMap**：专门管理自己的记忆流（Memory Stream）。  
- **一个共享/全局或者部分共享**的 **SemanticGraph**：表示小镇中所有角色、地点、事件等节点，以及角色间的社会关系、对话链接等。

---

## **三、系统设计与数据结构**

### **1. Memory Stream 存储 (SemanticMap)**

根据论文，Memory Stream 中的每条记忆通常包含：
- 时间戳（timestamp）  
- 触发事件/内容（text）  
- 重要性/情感标签（可选）  
- 主题或标签（可选）

在 **SemanticMap** 中，每个记忆条目可以这样表示：

```python
{
  "text": "Bob 在集市上看到 Alice 并打了招呼。",
  "timestamp": "2025-01-10 09:15",
  "importance": 0.3,
  "emotions": ["friendly"],
  "tags": ["market", "greeting"]
}
```

插入到**SemanticMap**时，可指定 `key = "memory_{agent_name}_{index}"`，并在 `value` 中存储如上信息；另外加上 `text_for_embedding=...` 用于做embedding检索。

### **2. 关系图 (SemanticGraph)**

在斯坦福小镇，多智能体共享或部分共享的**SemanticGraph**可能包含：
- **Agent 节点**：每个角色都是一个节点，如 `agent_alice`, `agent_bob`。  
- **地点 节点**：如 `loc_coffee_shop`, `loc_town_square`。  
- **事件 节点**：如 `event_festival_planning`。  
- **边**：表示角色之间的关系（friends、colleagues），角色对事件的参与（参加、组织），角色与地点的关联（常去、路过）。

例如：
```python
graph.add_node("agent_alice", {"type": "agent", "name": "Alice", ...})
graph.add_node("agent_bob", {"type": "agent", "name": "Bob", ...})

graph.insert_edge("agent_alice", "agent_bob", relation="friends")
graph.insert_edge("agent_alice", "loc_coffee_shop", relation="frequent_visitor")
```

---

## **四、示例代码**

以下是一个示例脚本 `multi_agent_demo.py`，示范如何在**斯坦福小镇**风格的多智能体环境下创建若干Agent，每个Agent持有自己的SemanticMap，用于记忆流；同时共享一个SemanticGraph来表示全局关系。

### **1. 定义 Agent 类（带 Memory Stream）**

```python
from semantic_map import SemanticMap
from semantic_graph import SemanticSimpleGraph

class GenerativeAgent:
    def __init__(self, name):
        self.name = name
        # 每个Agent自己拥有一个 memory_map
        self.memory_map = SemanticMap(embedding_model="all-MiniLM-L6-v2", index_type="flat")
        # 也可以有一个独立的 semantic_graph 或指向全局graph
        self.graph = None  # 指向全局graph

        # 额外属性
        self.goals = []
        self.reflections = []

    def add_memory(self, memory_text, timestamp, importance=0.5, emotions=None):
        """
        将一条新的记忆插入 memory_map
        """
        key = f"memory_{self.name}_{timestamp}"
        value = {
            "text": memory_text,
            "timestamp": timestamp,
            "importance": importance,
            "emotions": emotions or []
        }
        self.memory_map.insert(key, value, text_for_embedding=memory_text)

    def build_memory_index(self):
        self.memory_map.build_index()

    def retrieve_relevant_memories(self, query, k=3):
        """
        在自己的 memory_map 中根据 query 做embedding检索
        """
        return self.memory_map.retrieve_similar(query, k)

    def reflect(self):
        """
        简化的反思(Reflection)操作：检索最近和高重要度的记忆，做摘要
        """
        # 只做演示
        memory_chunks = self.retrieve_relevant_memories("recent", k=5)
        summary = f"Agent {self.name} reflection on top memories: "
        for chunk in memory_chunks:
            summary += chunk["value"]["text"] + " | "
        self.reflections.append(summary)
        return summary

    def send_message(self, other_agent, message_text):
        # 将对话信息也视作记忆
        timestamp = "2025-01-10 10:00"
        my_memory_text = f"I said to {other_agent.name}: {message_text}"
        self.add_memory(my_memory_text, timestamp)
        # 同时对方也记录收到消息
        other_agent.receive_message(self.name, message_text)

    def receive_message(self, sender_name, message_text):
        timestamp = "2025-01-10 10:00"
        mem_text = f"{sender_name} told me: {message_text}"
        self.add_memory(mem_text, timestamp)
```

### **2. 全局图 (SemanticGraph) 表示关系**

```python
class TownGraph(SemanticSimpleGraph):
    def __init__(self):
        super().__init__()

    def add_agent_node(self, agent: GenerativeAgent):
        # 在图中添加一个Agent节点
        self.add_node(
            key=agent.name,
            value={"type":"agent", "name": agent.name}
        )

    def link_agents(self, agentA, agentB, relation):
        self.insert_edge(agentA.name, agentB.name, relation=relation)

    def add_location(self, loc_key, loc_info):
        self.add_node(loc_key, loc_info)

    def agent_goes_to_location(self, agent:GenerativeAgent, loc_key):
        self.insert_edge(agent.name, loc_key, relation="visits")
```

### **3. Demo：多Agent场景**

```python
def demo_stanford_town():
    # 1) 创建全局graph
    town_graph = TownGraph()

    # 2) 创建若干Agent
    alice = GenerativeAgent("Alice")
    bob = GenerativeAgent("Bob")
    charlie = GenerativeAgent("Charlie")

    # 把agent节点加入town_graph
    town_graph.add_agent_node(alice)
    town_graph.add_agent_node(bob)
    town_graph.add_agent_node(charlie)

    # 3) Agent之间建立初始关系
    town_graph.link_agents(alice, bob, "friends")
    town_graph.link_agents(bob, charlie, "neighbors")
    town_graph.link_agents(alice, charlie, "colleagues")

    # 4) 为Agent附加graph引用 (可选,若每agent要访问全局关系)
    alice.graph = town_graph
    bob.graph = town_graph
    charlie.graph = town_graph

    # 5) Agent添加初始记忆
    alice.add_memory("I love teaching art classes here in town.", "2025-01-09 09:00")
    bob.add_memory("I'm thinking of new bread recipes for the local market.", "2025-01-09 09:05")
    charlie.add_memory("I manage the local library and enjoy collecting interesting stories.", "2025-01-09 09:10")

    # 构建memory索引
    alice.build_memory_index()
    bob.build_memory_index()
    charlie.build_memory_index()

    # 6) 互动：Alice发送消息给Bob
    alice.send_message(bob, "Hey Bob, could you bake some pastries for my after-class event?")

    # 7) 互动：Bob回应Alice
    bob.send_message(alice, "Sure! Any specific theme in mind?")

    # 8) 互动：Charlie介入
    charlie.send_message(alice, "I have some great storybooks if you need them for your class!")
    # 9) 反思 (Reflection)
    print("Alice Reflection:", alice.reflect())
    print("Bob Reflection:", bob.reflect())

    # 10) 查询: Alice检索她最近的记忆
    recent_mem = alice.retrieve_relevant_memories("Alice said", k=3)
    print("Alice's relevant memory snippet:", recent_mem)

    # 11) 查看TownGraph
    print("TownGraph for Bob:", town_graph.get_links("Bob"))

if __name__=="__main__":
    demo_stanford_town()
```

---

### **四、Memory Stream / Retrieve 对应关系**

- **Memory Stream**：每个智能体的 `semantic_map` 中记录多条记忆条目（如对话、事件、情绪等），在**Generative Agents**中称为“Memory Stream”。
- **Memory Retrieve**：在**需要反思**（reflection）或**需要上下文**时，调用 `retrieve_relevant_memories(query, k)` 在**Agent**自己的 memory_map 中做**向量检索**，找出最符合的记忆片段。

---

### **五、参考与数据集**

1. **论文**: [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442)
2. **代码**: [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents)
3. **本示例**可使用**模拟数据**(如上)或**实际日志/对话数据**来填充每个智能体的记忆条目。

---

### **六、优势与扩展**

- **多智能体对话**：每个Agent都有独立`SemanticMap`储存自身“Memory Stream”，并可随时retrieve最相关的过去对话或事件。
- **关系图**：`SemanticGraph`维护所有角色和地点/事件的关联，可多跳查询（如“找Alice的朋友、再找他们在做什么”）。
- **对接LLM**：在**reflection**或**action**时，Agent可将检索到的记忆上下文拼成Prompt给LLM做推理/生成。
- **可扩展**：添加更多Agent、事件或真实数据；或在**Graph**中维护更细节的社交网络和事件网络。

---

### **总结**

通过这种**SemanticMap + SemanticGraph**方式，便可如论文所述，为**每个智能体**构建“**记忆流（Memory Stream）**”并利用“**向量检索（Memory Retrieve）**”来完成**自我反思**与**行为决策**；同时通过**Graph**管理多智能体在斯坦福小镇风格环境下的**社交与事件关系**，实现真正**多智能体协同**与**对话**的模拟。



A: ### **基于 SemanticMap 和 SemanticGraph 的多智能体协同系统**

本文设计一个多智能体协同系统，灵感来源于斯坦福小镇（Stanford Small Town）和 **[Generative Agents](https://github.com/joonspk-research/generative_agents)**。系统使用 **SemanticMap** 和 **SemanticGraph** 来实现智能体的数据存储、关系建模和复杂查询能力。每个智能体都有自己的 **SemanticMap** 存储个体信息，并通过 **SemanticGraph** 表示智能体之间的关系及交互记录。

---

### **场景：斯坦福小镇**
斯坦福小镇是一个模拟环境，多个智能体（agents）在其中交互。每个智能体都有独立的身份、目标，并可以通过对话或任务合作，与其他智能体建立关系。

---

### **案例：三个智能体协同举办小镇故事会**
- **Alice（老师）**：热爱教育和艺术，负责组织小镇活动。
- **Bob（面包师）**：喜欢烘焙，专注于创新面包设计。
- **Charlie（图书管理员）**：喜欢收集历史故事和帮助别人获取知识。

#### **场景示例**
1. Alice 希望举办一个面向儿童的“故事会”，并邀请 Bob 提供点心，Charlie 提供历史故事背景。
2. Bob 与 Charlie 合作，获取历史面包配方，并为活动制作特别点心。
3. 在活动结束后，Alice 记录了本次活动中学到的经验，Bob 和 Charlie 更新了合作记录。

---



### **SemanticMap 和 SemanticGraph 的设计**

#### **SemanticMap：存储个体数据**
每个智能体都有独立的 SemanticMap，用来存储：
1. **个人信息**：智能体的特性、目标和当前任务。
2. **接收的消息**：从其他智能体接收到的交互内容。
3. **生成的行为**：智能体自己做出的行为记录。

#### **SemanticGraph：存储关系和交互历史**
1. **节点**：包括智能体（Alice、Bob、Charlie）和事件（如“故事会活动”）。
2. **边**：描述智能体之间的关系（如“合作”）和交互内容（如“请求点心”）。

---

### **实现代码**

#### **1. 定义智能体类**
```python
from semantic_graph import SemanticSimpleGraph
from semantic_map import SemanticMap

class Agent:
    def __init__(self, name):
        self.name = name
        self.semantic_map = SemanticMap()
        self.graph = SemanticSimpleGraph()

    def add_personal_info(self, key, value):
        self.semantic_map.insert(key, value)

    def send_message(self, recipient, message):
        recipient.receive_message(self.name, message)
        self.graph.insert_edge(self.name, recipient.name, relation="sent message")

    def receive_message(self, sender, message):
        self.semantic_map.insert(f"message_from_{sender}", message)
        self.graph.insert_edge(sender, self.name, relation="received message")

    def perform_action(self, action_description):
        self.semantic_map.insert(f"action_{self.name}", action_description)

    def add_relationship(self, other_agent, relation):
        self.graph.insert_edge(self.name, other_agent.name, relation=relation)

    def get_relationships(self):
        return self.graph.get_links(self.name)
```

---

#### **2. 创建智能体和关系**

```python
# 创建智能体
alice = Agent("Alice")
bob = Agent("Bob")
charlie = Agent("Charlie")

# 添加个人信息
alice.add_personal_info("role", "teacher")
bob.add_personal_info("role", "baker")
charlie.add_personal_info("role", "librarian")

# 定义关系
alice.add_relationship(bob, "collaborates with")
bob.add_relationship(charlie, "collaborates with")
charlie.add_relationship(alice, "shares knowledge with")
```

---

#### **3. 模拟交互**

```python
# Alice 发起活动请求
alice.send_message(bob, "Can you bake themed pastries for the storytelling event?")
bob.send_message(alice, "Of course! I'll work on a design.")

# Bob 向 Charlie 请求帮助
bob.send_message(charlie, "Do you have any historical pastry designs?")
charlie.send_message(bob, "Yes, I can share a collection of medieval recipes.")

# 行为记录
alice.perform_action("Organized a storytelling event")
bob.perform_action("Baked pastries for Alice's event")
charlie.perform_action("Shared historical recipes with Bob")
```

---

#### **4. 查询 SemanticGraph 和 SemanticMap**

```python
# 查询 Alice 的关系
alice_relationships = alice.get_relationships()
print("Alice's Relationships:", alice_relationships)

# 查询 Bob 接收到的消息
messages_to_bob = bob.semantic_map.retrieve_similar("message_from", k=5)
print("Messages to Bob:", messages_to_bob)

# 查看 Charlie 的所有行为
charlie_actions = charlie.semantic_map.retrieve_similar("action", k=5)
print("Charlie's Actions:", charlie_actions)
```

---

### **数据存储结构**

#### **1. SemanticMap（Alice 示例）**
```json
{
    "personal_info": {
        "role": "teacher"
    },
    "received_messages": [
        {"from": "Bob", "message": "Of course! I'll work on a design."}
    ],
    "actions": [
        "Organized a storytelling event"
    ]
}
```

#### **2. SemanticGraph**
- **Alice → Bob**: "collaborates with"
- **Bob → Charlie**: "collaborates with"
- **Charlie → Alice**: "shares knowledge with"

---

### **展示多智能体的功能**

#### **1. 事件链查询**
**问题**：从 Alice 发起活动开始，所有智能体的行为链条是什么？

```python
chain = alice.graph.retrieve_subtree("Alice")
print("Interaction Chain:", chain)
```

#### **2. 动态协作**
**问题**：如果 Charlie 新增了一项活动记录，如何更新？

```python
charlie.perform_action("Planned a library workshop")
```

---

### **对接 LangChain**

#### **自然语言查询**
将 SemanticMap 的上下文传递给 LangChain 实现问答：
```python
from langchain import OpenAI

llm = OpenAI(temperature=0)
query = "Summarize Alice's role and recent activities."
context = alice.semantic_map.retrieve_all()

response = llm(context + query)
print("LLM Response:", response)
```

---

### **数据集来源**
1. **斯坦福小镇（Stanford Small Town）**：使用 [Generative Agents](https://github.com/joonspk-research/generative_agents) 数据。
2. **模拟生成数据**：根据预定义智能体角色生成对话和行为。

---

### **亮点和优势**
1. **灵活建模**：SemanticMap 和 SemanticGraph 可扩展为复杂的语义查询和关系查询。
2. **强大的上下文整合**：可用于多跳推理和动态更新，支持复杂协作场景。
3. **多框架兼容性**：可与 LangChain、LlamaIndex 等结合，提高生成式 AI 的智能性。

这种设计展示了 **SemanticMap + SemanticGraph** 的强大功能，特别是在多智能体环境中的应用潜力。



### **Multi-Agent Collaboration Example: Stanford Small Town (Inspired by Generative Agents)**

This example demonstrates how **SemanticMap + SemanticGraph** can be applied to model and enhance a multi-agent system inspired by Stanford Small Town. Each agent has its own **SemanticMap** to store personal and interactional data and utilizes **SemanticGraph** to maintain relationships and interaction history with other agents.

---

### **Scenario: Stanford Small Town**
Stanford Small Town is a simulation where multiple agents interact, exchange information, and collaborate dynamically. Agents have unique identities, goals, and the ability to engage in conversations, share tasks, and build relationships.

### **Agents**
- **Alice**: A teacher passionate about art and education.
- **Bob**: A local baker who loves experimenting with recipes.
- **Charlie**: The town librarian who enjoys storytelling and helping others learn.

### **Example Use Case**
Agents perform daily routines, exchange messages, collaborate on events (e.g., organizing a town festival), and remember interactions over time. For instance:
1. **Alice asks Bob** for cakes for her art students.
2. **Bob discusses recipe ideas with Alice** and collaborates with **Charlie** to source historical cake designs from books.
3. **Charlie shares historical recipes with Bob** and plans a storytelling event with Alice.

---

### **SemanticMap and SemanticGraph Structure**

#### **SemanticMap for Each Agent**
Stores the following for each agent:
1. **Personal Information**: Preferences, tasks, history.
2. **Received Messages**: Information sent by other agents.
3. **Generated Actions**: Things the agent has decided or done.

#### **SemanticGraph for Agent Interactions**
Tracks relationships and events:
1. **Nodes**:
   - Agents (Alice, Bob, Charlie).
   - Events (e.g., "Town Festival Planning").
2. **Edges**:
   - Relationships (e.g., "collaborates with").
   - Information exchange (e.g., "requested from").

---

### **Implementation**

#### **1. Define SemanticMap and SemanticGraph**
```python
from semantic_graph import SemanticSimpleGraph
from semantic_map import SemanticMap

class Agent:
    def __init__(self, name):
        self.name = name
        self.semantic_map = SemanticMap()
        self.graph = SemanticSimpleGraph()

    def add_personal_info(self, key, value):
        self.semantic_map.insert(key, value)

    def send_message(self, recipient, message):
        recipient.receive_message(self.name, message)

    def receive_message(self, sender, message):
        self.semantic_map.insert(f"message_from_{sender}", message)

    def perform_action(self, action_description):
        self.semantic_map.insert(f"action_{self.name}", action_description)

    def add_relationship(self, other_agent, relation):
        self.graph.insert_edge(self.name, other_agent.name, relation=relation)

    def get_relationships(self):
        return self.graph.get_links(self.name)
```

---

#### **2. Create Agents and Relationships**

```python
# Initialize Agents
alice = Agent("Alice")
bob = Agent("Bob")
charlie = Agent("Charlie")

# Add Personal Information
alice.add_personal_info("hobbies", "art and teaching")
bob.add_personal_info("hobbies", "baking and recipe creation")
charlie.add_personal_info("hobbies", "storytelling and research")

# Define Relationships
alice.add_relationship(bob, "collaborates with")
bob.add_relationship(charlie, "collaborates with")
charlie.add_relationship(alice, "shares knowledge with")
```

---

#### **3. Simulate Interaction**

```python
# Alice sends a request to Bob
alice.send_message(bob, "Can you bake some cakes for my art students?")
bob.send_message(alice, "Sure! Do you have any specific theme in mind?")

# Bob collaborates with Charlie for historical recipe ideas
bob.send_message(charlie, "Do you have any historical cake designs?")
charlie.send_message(bob, "Yes, I’ll find some references for you!")

# Actions
alice.perform_action("Planned an art class with Bob's cakes")
bob.perform_action("Baked historical-themed cakes for Alice's students")
charlie.perform_action("Shared historical cake designs with Bob")
```

---

#### **4. Query SemanticGraph**

```python
# View relationships for Alice
alice_relationships = alice.get_relationships()
print("Alice's Relationships:", alice_relationships)

# Query specific messages or actions
messages_to_alice = alice.semantic_map.retrieve_similar("message_from", k=5)
print("Messages to Alice:", messages_to_alice)
```

---

### **Potential Data Storage in Semantic Structures**

#### **SemanticMap (Per Agent)**
Example for Alice:
```json
{
    "personal_info": {
        "hobbies": "art and teaching"
    },
    "received_messages": [
        {"from": "Bob", "message": "Sure! Do you have any specific theme in mind?"}
    ],
    "actions": [
        "Planned an art class with Bob's cakes"
    ]
}
```

#### **SemanticGraph**
Relationships:
- Alice → Bob (`collaborates with`)
- Bob → Charlie (`collaborates with`)
- Charlie → Alice (`shares knowledge with`)

---

### **Example Use Cases**

#### **1. Query Interaction Chains**
**Question**: How did Alice's request for cakes propagate through the town?
```python
relationships_chain = alice.graph.retrieve_subtree("Alice")
print(relationships_chain)
```

#### **2. Summarize All Actions**
**Question**: What actions were taken by all agents during the festival preparation?
```python
all_actions = []
for agent in [alice, bob, charlie]:
    actions = agent.semantic_map.retrieve_similar("action", k=10)
    all_actions.extend(actions)
print(all_actions)
```

---

### **Integration with LangChain or LlamaIndex**

Agents can use LangChain for:
1. **Natural Language Processing**: Summarizing received messages or actions.
2. **Context Management**: Dynamically building prompts with data from SemanticMap + SemanticGraph.

Example:
```python
from langchain import OpenAI

llm = OpenAI(temperature=0)
query = "Summarize Alice's interactions and outcomes."
context = alice.semantic_map.retrieve_all()

response = llm(context + query)
print(response)
```

---

### **Benchmarking with Public Datasets**
- **Stanford Small Town Dataset**: Use existing synthetic datasets.
- **AI Town Simulations**: Extendable with custom scenarios.
- **Agent Conversation Logs**: Simulated interaction datasets for multi-agent benchmarks.

---

### **Advantages**
1. **Flexible Data Storage**: Store agent-specific data and inter-agent relationships.
2. **Rich Querying**: Retrieve and analyze multi-hop relationships or temporal interactions.
3. **Scalability**: Integrate with real-time simulations and large-scale multi-agent systems.

This setup demonstrates how **SemanticMap + SemanticGraph** can power dynamic, multi-agent environments like Stanford Small Town.



Q：仔细阅读这篇文章https://arxiv.org/pdf/2304.03442 和我提的需求：关于SemanticMap+SemanticGraph的应用，是不是可以举一个多agent协同的例子，比如多agent对话（如斯坦福小镇每个角色是一个agent，可以与其他agent交流，对话等），每个agent可以有一个自己的semanticmap来存自己的信息、别人发给自己的信息、自己做的事情等。斯坦福小镇可以参考https://arxiv.org/pdf/2304.03442 ,可以的话详细描述场景，semanticmap/graph里存的内容，数据集，代码等。也可以参考其代码，https://github.com/joonspk-research/generative_agents。这篇论文也提到memory stream，memory retrieve等，我们的semanticmap/graph是不是可以做这个memory stream/retrieve，写出详细的设计和代码


A：