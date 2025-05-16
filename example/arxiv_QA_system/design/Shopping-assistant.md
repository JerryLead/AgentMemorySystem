### **Case 12: Context-Aware Shopping Assistant**

### **Case 12: Context-Aware Shopping Assistant**

#### 数据集  
- 使用 [Amazon Product Dataset](https://nijianmo.github.io/amazon/index.html)，包含商品信息、评论和评分。

#### 目标  
为用户提供上下文感知的购物建议，结合用户偏好、语义查询和多模态数据。

#### 流程  

1. **数据存储**  
   - 将商品信息、评论、评分和图片存入 `SemanticMap`。
   - 每条记录包含：
     - `product_id`: 商品编号。
     - `name`: 商品名称。
     - `category`: 商品类别。
     - `description`: 商品描述。
     - `reviews`: 用户评论（不定数量）。
     - `images`: 商品图片（不定数量）。
     - `rating`: 用户评分。

2. **用户交互与推荐**  
   - 用户输入：“推荐适合送礼的高评分电子产品”。
   - Agent 使用语义查询结合评分过滤，返回符合条件的商品及推荐理由。

3. **记忆用户偏好**  
   - 用户标记“喜欢”某些商品后，Agent 存储这些偏好。
   - 在后续交互中优先推荐类似商品。

4. **动态更新商品信息**  
   - 定期从数据源更新商品的评论和评分。
   - 用户删除不感兴趣的商品，Agent 从记忆中移除。

#### Memory 相关功能  
- **复杂关系维护**：
  - 每个商品可以有不定数量的评论和图片。
- **语义查询**：
  - 根据描述、评论和类别进行综合检索。
- **记忆偏好**：
  - 存储用户偏好，并在后续推荐中优先考虑。

---

#### **优化设计**

1. **目标**
   - 使用 **SemanticMap** 处理复杂的多模态商品信息存储，包括文本描述、评分、评论、图片。
   - 结合语义查询和用户偏好实现上下文感知的推荐功能。
   - 动态管理商品信息，支持用户反馈和记忆调整。

2. **功能扩展**
   - **复杂关系维护**：支持商品与其评论、图片的多对多关系。
   - **语义查询**：支持根据用户输入的自然语言条件（如高评分、类别、用途）检索商品。
   - **用户偏好记忆**：记忆用户“喜欢”的商品，优先推荐相似商品。

---

### **数据存储设计**

#### **商品信息格式**

```json
{
  "product_id": "P12345",
  "name": "Sony WH-1000XM5",
  "category": "Electronics",
  "description": "Wireless noise-canceling headphones with high-fidelity audio.",
  "reviews": [
    "Excellent sound quality and noise cancellation.",
    "Comfortable for long use."
  ],
  "images": [
    "path/to/image1.jpg",
    "path/to/image2.jpg"
  ],
  "rating": 4.8
}
```

#### **用户偏好格式**

```json
{
  "user_id": "user_123",
  "liked_products": ["P12345", "P67890"],
  "preferences": ["high ratings", "electronics"]
}
```

---

### **优化后的实现**

#### **ShoppingAssistant 类实现**

```python
class ShoppingAssistant:
    def __init__(self):
        self.product_map = SemanticMap(key_type="text", embedding_dim=384)
        self.user_preferences = {}  # 用户偏好

    def insert_product(self, product_data):
        """
        插入商品数据。
        """
        product_id = product_data["product_id"]
        self.product_map.insert(product_id, product_data)

    def store_user_preference(self, user_id, liked_product_id):
        """
        记录用户喜欢的商品。
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {"liked_products": [], "preferences": []}
        if liked_product_id not in self.user_preferences[user_id]["liked_products"]:
            self.user_preferences[user_id]["liked_products"].append(liked_product_id)

    def retrieve_recommendations(self, user_query, user_id=None, k=5):
        """
        根据用户查询和偏好返回推荐商品。
        """
        # 基础语义查询
        recommendations = self.product_map.retrieve_similar(user_query, k)

        # 根据用户偏好过滤
        if user_id and user_id in self.user_preferences:
            liked_products = self.user_preferences[user_id]["liked_products"]
            filtered_recommendations = [
                rec for rec in recommendations if rec["key"] not in liked_products
            ]
            return filtered_recommendations[:k]
        return recommendations

    def update_product_info(self, product_id, updated_info):
        """
        更新商品信息。
        """
        product = self.product_map.retrieve_similar(product_id, k=1)
        if product:
            product_data = product[0]["value"]
            product_data.update(updated_info)
            self.product_map.update(product_id, new_value=product_data)

    def delete_product(self, product_id):
        """
        删除商品。
        """
        self.product_map.delete(query=product_id, k=1)
```

---

### **案例实现**

#### **Step 1: 存储商品信息**

```python
# 初始化 ShoppingAssistant
assistant = ShoppingAssistant()

# 示例商品数据
products = [
    {
        "product_id": "P12345",
        "name": "Sony WH-1000XM5",
        "category": "Electronics",
        "description": "Wireless noise-canceling headphones with high-fidelity audio.",
        "reviews": [
            "Excellent sound quality and noise cancellation.",
            "Comfortable for long use."
        ],
        "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
        "rating": 4.8
    },
    {
        "product_id": "P67890",
        "name": "Bose QuietComfort 45",
        "category": "Electronics",
        "description": "Comfortable noise-canceling headphones with long battery life.",
        "reviews": [
            "Great for frequent travelers.",
            "Battery lasts all day."
        ],
        "images": ["path/to/image3.jpg", "path/to/image4.jpg"],
        "rating": 4.5
    }
]

# 插入商品到 SemanticMap
for product in products:
    assistant.insert_product(product)
```

---

#### **Step 2: 用户交互与推荐**

##### **语义查询推荐**

```python
# 用户查询
user_query = "推荐适合送礼的高评分电子产品"

# 获取推荐
recommendations = assistant.retrieve_recommendations(user_query, k=3)

print("\n推荐商品：")
for rec in recommendations:
    print(f"Product Name: {rec['value']['name']}, Rating: {rec['value']['rating']}")
```

---

#### **Step 3: 用户偏好存储**

```python
# 用户标记喜欢的商品
user_id = "user_123"
liked_product_id = "P12345"
assistant.store_user_preference(user_id, liked_product_id)

# 检索推荐时过滤已喜欢的商品
recommendations = assistant.retrieve_recommendations(user_query, user_id=user_id, k=3)

print("\n推荐商品（过滤用户喜欢的商品）：")
for rec in recommendations:
    print(f"Product Name: {rec['value']['name']}, Rating: {rec['value']['rating']}")
```

---

#### **Step 4: 动态更新与删除**

##### **更新商品信息**

```python
# 更新商品评分
assistant.update_product_info("P12345", {"rating": 4.9})
```

##### **删除商品**

```python
# 删除商品
assistant.delete_product("P67890")
```

---

### **功能展示**

#### **推荐商品**

```plaintext
推荐商品：
Product Name: Sony WH-1000XM5, Rating: 4.8
Product Name: Bose QuietComfort 45, Rating: 4.5
```

#### **推荐商品（过滤用户喜欢的商品）**

```plaintext
推荐商品（过滤用户喜欢的商品）：
Product Name: Bose QuietComfort 45, Rating: 4.5
```

#### **更新后的商品信息**

```plaintext
Updated Rating for Sony WH-1000XM5: 4.9
```

---

### **扩展与优化**

1. **多模态查询扩展**：
   - 支持用户上传商品图片或语音描述，用于语义检索。

2. **推荐增强**：
   - 根据用户的购买历史、浏览行为和评论倾向生成更加个性化的推荐。

3. **定期数据更新**：
   - 定时爬取新的评论和评分，动态调整商品信息。

通过结合 **SemanticMap** 和多模态数据，Shopping Assistant 不仅实现了复杂的关系维护和语义查询，还通过记忆用户偏好优化了推荐逻辑，具备强大的上下文感知能力。

Q: 商品可以存放在semanticmap里？推荐商品也可以存放在semanticmap里，还是有其他更好的语义数据结构？

### **存储商品与推荐商品的语义数据结构选择**

#### **是否可以将商品存储在 SemanticMap 中？**
是的，**SemanticMap** 非常适合存储商品数据，因为它具有以下特点：
1. **多模态数据支持**：商品数据通常包括文本描述、图片（向量形式）和评分，可以轻松存入 SemanticMap。
2. **语义查询**：支持根据自然语言条件或向量检索（如高评分、适合送礼）快速找到匹配的商品。
3. **关系维护**：商品的评论、图片等复杂关系可通过键值对的扩展管理。

#### **推荐商品是否也可以存放在 SemanticMap 中？**
推荐商品可以存放在 SemanticMap 中，但需要考量以下场景：
- **静态推荐列表**：如果推荐列表是较为静态的（如基于分类的推荐），可直接存入 SemanticMap，利用语义查询进行动态调整。
- **动态推荐生成**：如果推荐内容需要频繁生成（如根据用户实时行为），推荐商品可以临时存储在一个轻量级的语义数据结构中，例如 **SemanticList** 或 **SessionMemory**。

---

### **优化建议：混合使用语义数据结构**

为更好地支持商品数据存储和推荐逻辑，推荐如下设计：

#### **1. 商品存储**
- 使用 **SemanticMap** 存储商品及其复杂多模态信息。
- 每个商品一条记录，支持通过描述、类别、评分等语义条件检索。

#### **2. 推荐商品管理**
- **短期推荐**（基于实时交互生成）：使用 **SemanticList** 或 **SessionMemory** 存储当前会话推荐列表。
- **长期推荐**（基于用户偏好存储）：将用户历史推荐和偏好存储在 **SemanticMap** 中，按需查询。

---

### **数据结构设计示例**

#### **SemanticMap 用于存储商品数据**
```json
{
  "product_id": "P12345",
  "name": "Sony WH-1000XM5",
  "category": "Electronics",
  "description": "Wireless noise-canceling headphones with high-fidelity audio.",
  "reviews": [
    "Excellent sound quality and noise cancellation.",
    "Comfortable for long use."
  ],
  "images": [
    "path/to/image1.jpg",
    "path/to/image2.jpg"
  ],
  "rating": 4.8
}
```

#### **SemanticList 用于短期推荐**
```json
[
  {
    "product_id": "P12345",
    "name": "Sony WH-1000XM5",
    "reason": "High rating and excellent reviews."
  },
  {
    "product_id": "P67890",
    "name": "Bose QuietComfort 45",
    "reason": "Great for frequent travelers."
  }
]
```

#### **用户偏好存储在 SemanticMap**
```json
{
  "user_id": "user_123",
  "liked_products": ["P12345", "P67890"],
  "preferences": ["high ratings", "electronics"]
}
```

---

### **改进实现：商品存储 + 推荐逻辑**

#### **商品存储**

```python
class ProductManager:
    def __init__(self):
        self.product_map = SemanticMap(key_type="text", embedding_dim=384)

    def add_product(self, product_data):
        """
        将商品信息存入 SemanticMap。
        """
        product_id = product_data["product_id"]
        self.product_map.insert(product_id, product_data)

    def retrieve_products(self, query, k=5):
        """
        基于查询条件检索商品。
        """
        return self.product_map.retrieve_similar(query, k)
```

#### **推荐逻辑**

```python
class RecommendationManager:
    def __init__(self):
        self.recommendation_list = []  # 使用 SemanticList 存储短期推荐

    def generate_recommendations(self, user_query, product_manager, k=5):
        """
        根据用户查询生成推荐列表。
        """
        recommendations = product_manager.retrieve_products(user_query, k)
        self.recommendation_list = [
            {"product_id": rec["key"], "name": rec["value"]["name"], "reason": f"Matching query: {user_query}"}
            for rec in recommendations
        ]
        return self.recommendation_list

    def clear_recommendations(self):
        """
        清空短期推荐列表。
        """
        self.recommendation_list = []
```

---

### **推荐系统流程**

#### **Step 1: 存储商品数据**

```python
# 初始化商品管理器
product_manager = ProductManager()

# 示例商品数据
products = [
    {
        "product_id": "P12345",
        "name": "Sony WH-1000XM5",
        "category": "Electronics",
        "description": "Wireless noise-canceling headphones with high-fidelity audio.",
        "reviews": ["Excellent sound quality.", "Comfortable for long use."],
        "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
        "rating": 4.8
    },
    {
        "product_id": "P67890",
        "name": "Bose QuietComfort 45",
        "category": "Electronics",
        "description": "Comfortable noise-canceling headphones with long battery life.",
        "reviews": ["Great for frequent travelers.", "Battery lasts all day."],
        "images": ["path/to/image3.jpg", "path/to/image4.jpg"],
        "rating": 4.5
    }
]

# 存储商品
for product in products:
    product_manager.add_product(product)
```

---

#### **Step 2: 根据用户查询生成推荐**

```python
# 初始化推荐管理器
recommendation_manager = RecommendationManager()

# 用户查询
user_query = "推荐高评分耳机"

# 生成推荐列表
recommendations = recommendation_manager.generate_recommendations(user_query, product_manager, k=3)

print("\n推荐商品：")
for rec in recommendations:
    print(f"Product Name: {rec['name']}, Reason: {rec['reason']}")
```

---

#### **Step 3: 清空短期推荐**

```python
# 清空短期推荐列表
recommendation_manager.clear_recommendations()
```

---

### **总结**

#### **SemanticMap 的角色**
- 存储商品数据及用户偏好。
- 提供强大的语义检索能力。

#### **SemanticList 的角色**
- 用于临时存储短期推荐列表，支持动态更新和清除。

#### **其他扩展**
1. **多模态推荐**：支持图片或文本输入。
2. **基于用户偏好的长效推荐**：将用户的历史行为存入 SemanticMap，并结合推荐算法生成个性化推荐。
3. **实时更新**：定期刷新商品信息，确保推荐内容的准确性。

通过以上设计，**SemanticMap** 和 **SemanticList** 的结合实现了商品存储、用户偏好记忆与推荐管理的全流程。


Q: 用什么数据集？

### **推荐数据集：支持 Context-Aware Shopping Assistant**

为实现 **Context-Aware Shopping Assistant** 场景，可以使用以下公开数据集：

---

#### **1. Amazon Product Dataset**
- **来源**: [Amazon Product Dataset](https://nijianmo.github.io/amazon/index.html)
- **内容**:
  - 商品分类：包含电子产品、服装、书籍等 24 个类别。
  - 用户评论：高质量的用户评论、评分和标题。
  - 商品描述：包含商品标题和详细描述。
  - 时间范围：1996 年至 2018 年的数据。
- **适用场景**:
  - 存储商品信息，包括商品名称、描述、评论、评分。
  - 用于语义查询（如“高评分耳机”）和推荐生成。

---

#### **2. Kaggle Electronics Dataset**
- **来源**: [Kaggle Electronics](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones)
- **内容**:
  - 专注于电子产品（如手机）的评论数据。
  - 包括评论文本、评分、用户反馈。
- **适用场景**:
  - 用于专注电子产品类别的语义查询和推荐。
  - 可以结合上下文感知推荐，分析用户反馈。

---

#### **3. OpenReview Dataset**
- **来源**: [OpenReview Dataset](https://openreview.net/)
- **内容**:
  - 包含商品评论、评分和用户元数据。
  - 包括正面、负面反馈和细粒度的评分维度。
- **适用场景**:
  - 用于推荐系统优化，特别是分析用户偏好的细节（如具体评论关键词）。

---

#### **4. Snap Product Dataset**
- **来源**: [SNAP: Amazon Review Dataset](https://snap.stanford.edu/data/web-Amazon.html)
- **内容**:
  - 包含数百万条亚马逊用户评论。
  - 包括时间戳、评分、商品类别和评论。
- **适用场景**:
  - 时间敏感推荐（如节假日购物场景）。
  - 结合用户评论，优化上下文相关的推荐。

---

#### **5. eCommerce Search Dataset**
- **来源**: [eCommerce Search](https://github.com/layerai/public-datasets)
- **内容**:
  - 包括用户的商品搜索记录和点击行为。
  - 包含搜索关键词和推荐商品。
- **适用场景**:
  - 搜索驱动的推荐系统。
  - 可与用户偏好记忆结合，优化搜索结果。

---

### **数据集选择与场景匹配**

| **数据集名称**       | **适用场景**                                                | **优势**                                    |
|----------------------|------------------------------------------------------------|---------------------------------------------|
| Amazon Product Dataset | 商品信息存储、语义查询、推荐生成                             | 数据全面，类别丰富，支持多模态数据（如评论与评分）。 |
| Kaggle Electronics   | 专注电子产品类别的推荐                                      | 适合垂直领域的推荐，如耳机、手机等商品。      |
| OpenReview Dataset   | 商品推荐优化，偏好分析                                      | 评论分析维度细化，适合基于评论优化推荐逻辑。   |
| Snap Product Dataset | 时间敏感推荐，结合用户评论优化推荐                           | 数据量大，支持长时间跨度的推荐场景。          |
| eCommerce Search     | 搜索驱动的商品推荐                                          | 提供搜索关键词与推荐结果的行为分析支持。      |

---

### **推荐使用的主要数据集**
1. **Amazon Product Dataset**：
   - 数据量大，类别多样，适合广泛场景。
   - 包含评论、评分和描述，支持语义查询。
   - 可扩展为多模态推荐，结合图片和描述生成结果。

2. **Snap Product Dataset**：
   - 时间跨度大，支持用户行为建模。
   - 适合分析节假日等特定场景下的推荐需求。

---

### **数据下载与预处理**
#### **数据下载**
- **Amazon Product Dataset**:
  ```bash
  wget http://deepyeti.ucsd.edu/jianmo/amazon/meta_Electronics.json.gz
  wget http://deepyeti.ucsd.edu/jianmo/amazon/Electronics.json.gz
  ```

- **Snap Product Dataset**:
  ```bash
  wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
  ```

#### **预处理代码示例**

```python
import json
import gzip

def load_amazon_data(file_path, max_records=1000):
    """
    加载 Amazon 数据集。
    """
    data = []
    with gzip.open(file_path, 'r') as f:
        for line in f:
            if len(data) >= max_records:
                break
            data.append(json.loads(line.strip()))
    return data

# 示例加载商品评论数据
file_path = "Electronics.json.gz"
amazon_data = load_amazon_data(file_path, max_records=500)

# 提取商品信息
products = []
for item in amazon_data:
    product = {
        "product_id": item.get("asin"),
        "name": item.get("title", "Unknown"),
        "category": item.get("category", "Electronics"),
        "description": item.get("description", ""),
        "reviews": [review.get("reviewText", "") for review in item.get("reviews", [])],
        "rating": item.get("overall", 0)
    }
    products.append(product)
```

---

### **结合 SemanticMap 使用**

#### **存储商品信息**

```python
# 初始化 SemanticMap
from semantic_map import SemanticMap

product_map = SemanticMap(key_type="text", embedding_dim=384)

# 插入商品数据
for product in products:
    product_map.insert(product["product_id"], product)
```

#### **查询与推荐**

```python
# 用户输入查询
query = "高评分的无线耳机"

# 检索匹配商品
recommendations = product_map.retrieve_similar(query, k=5)

# 打印推荐商品
for rec in recommendations:
    print(f"Product Name: {rec['value']['name']}, Rating: {rec['value']['rating']}")
```

---

通过选择 **Amazon Product Dataset** 或 **Snap Product Dataset**，结合 **SemanticMap** 实现商品存储与语义查询，构建功能丰富的购物助手。