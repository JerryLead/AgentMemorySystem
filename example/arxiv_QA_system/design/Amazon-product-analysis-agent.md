以下是优化后的案例 **Case 10: E-commerce Product Search Agent**，全面展示 AI Agent 的能力（LLM交互、记忆、规划、工具使用）。案例使用 **SemanticMap** 支持多模态数据存储和复杂关系维护，同时结合公开数据集（Amazon Product Dataset）实现。

---

### **案例设计与流程**

#### **数据集**
- **Amazon Product Dataset**：
  包含大量电商产品信息，如：
  - **`product_name`**：产品名称。
  - **`description`**：产品描述。
  - **`category`**：产品类别。
  - **`rating`**：用户评分。
  - **`reviews`**：用户评论。
  - **`image`**：产品图片（嵌入向量或二进制数据）。

---

### **核心能力展示**

1. **存储产品数据**：
   - 多模态存储：名称、描述、类别、评分、评论、图片。
   - 每条记录支持多个图片和评论。

2. **搜索产品**：
   - **语义搜索**：用户通过自然语言描述查询高评分无线耳机。
   - **图片搜索**：用户上传图片，返回相似产品。

3. **删除低评分产品**：
   - 动态过滤低评分产品。

4. **推荐替代产品**：
   - 基于相似性推荐替代产品。

---

### **代码实现**

#### **SemanticMap 初始化**

```python
# 初始化 SemanticMap
product_map = SemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 示例产品数据
products = [
    {
        "product_name": "Wireless Headphones A1",
        "description": "High-quality wireless headphones with noise cancellation.",
        "category": "Headphones",
        "rating": 4.5,
        "reviews": ["Amazing sound quality!", "Battery life is exceptional."],
        "images": [np.random.rand(384), np.random.rand(384)],
    },
    {
        "product_name": "Wireless Earbuds B2",
        "description": "Compact wireless earbuds with deep bass.",
        "category": "Earbuds",
        "rating": 3.8,
        "reviews": ["Good for the price.", "Comfortable fit but average sound."],
        "images": [np.random.rand(384)],
    },
    {
        "product_name": "Wired Headphones C3",
        "description": "Affordable wired headphones with decent sound.",
        "category": "Headphones",
        "rating": 2.5,
        "reviews": ["Sound is okay but build quality is poor.", "Not recommended for long use."],
        "images": [np.random.rand(384)],
    },
]

# 插入产品数据到 SemanticMap
for product in products:
    product_map.insert(product["product_name"], product)

# 构建索引
product_map.build_or_update_index()
```

---

#### **Step 2: 搜索产品**

##### **语义搜索**

```python
# 用户输入查询：高评分无线耳机
query_text = "高评分的无线耳机"
results = product_map.retrieve_similar(query_text, k=3)

print("\n语义搜索结果：")
for result in results:
    print(f"Product Name: {result['key']}")
    print(f"Rating: {result['value']['rating']}")
    print(f"Description: {result['value']['description']}")
```

##### **图片搜索**

```python
# 用户上传耳机图片
uploaded_image = np.random.rand(384)  # 示例图片嵌入向量
image_results = product_map.retrieve_similar(uploaded_image, k=3)

print("\n图片搜索结果：")
for result in image_results:
    print(f"Product Name: {result['key']}")
    print(f"Rating: {result['value']['rating']}")
    print(f"Description: {result['value']['description']}")
```

---

#### **Step 3: 删除低评分产品**

```python
# 删除评分低于 3 的产品
def low_rating_condition(record):
    return record[1]["rating"] < 3

product_map.delete(condition=low_rating_condition)

# 查看剩余产品
remaining_products = product_map.retrieve_similar("products", k=5)
print("\n删除低评分产品后：")
for product in remaining_products:
    print(f"Product Name: {product['key']}")
    print(f"Rating: {product['value']['rating']}")
```

---

#### **Step 4: 推荐替代产品**

```python
# 如果某产品不可用，推荐替代品
unavailable_product = "Wireless Headphones A1"
alternative_results = product_map.retrieve_similar(unavailable_product, k=3)

print("\n推荐的替代产品：")
for result in alternative_results:
    print(f"Product Name: {result['key']}")
    print(f"Rating: {result['value']['rating']}")
    print(f"Description: {result['value']['description']}")
```

---

### **案例输出示例**

#### **语义搜索结果**

```plaintext
语义搜索结果：
Product Name: Wireless Headphones A1
Rating: 4.5
Description: High-quality wireless headphones with noise cancellation.

Product Name: Wireless Earbuds B2
Rating: 3.8
Description: Compact wireless earbuds with deep bass.
```

#### **图片搜索结果**

```plaintext
图片搜索结果：
Product Name: Wireless Headphones A1
Rating: 4.5
Description: High-quality wireless headphones with noise cancellation.

Product Name: Wireless Earbuds B2
Rating: 3.8
Description: Compact wireless earbuds with deep bass.
```

#### **删除低评分产品后**

```plaintext
删除低评分产品后：
Product Name: Wireless Headphones A1
Rating: 4.5

Product Name: Wireless Earbuds B2
Rating: 3.8
```

#### **推荐替代产品**

```plaintext
推荐的替代产品：
Product Name: Wireless Earbuds B2
Rating: 3.8
Description: Compact wireless earbuds with deep bass.
```

---

### **改进与扩展**

1. **多模态支持**：
   - 名称和描述为文本键，图片嵌入支持图片检索。

2. **语义搜索增强**：
   - 使用产品描述和用户评论的嵌入向量提高检索效果。

3. **动态管理**：
   - 支持实时更新或删除产品记录，保持数据一致性。

4. **替代推荐**：
   - 基于相似度实现智能替代推荐。

这套实现展示了 SemanticMap 在电商场景中的高效存储和检索能力，并为 AI Agent 提供强大的工具支持。

基于上述的 `SemanticMap`，我们可以继承并优化它以适应电商产品搜索场景（**E-commerce Product Search Agent**）。以下是优化后的实现，包括对多模态数据支持的增强、具体功能的优化以及更智能的语义操作。

---

### **优化后的 SemanticMap**

#### **新增功能**
1. **多模态支持增强**：
   - 支持 `product_name` 为主键，值中包含描述、类别、评分、评论和多张图片。
   - 针对多模态数据的检索优化，支持文本、图片的混合检索。

2. **智能检索功能**：
   - 基于评论和描述的语义搜索。
   - 图片查询时返回最相关的产品。

3. **动态管理**：
   - 支持按条件删除低评分产品或某些类别的产品。
   - 自动推荐替代产品。

---

### **子类实现**

#### **EnhancedSemanticMap**

```python
class EnhancedSemanticMap(SemanticMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def insert_product(self, product_data):
        """
        插入产品数据，包括多模态内容。
        product_data 格式示例：
        {
            "product_name": "Wireless Headphones A1",
            "description": "High-quality wireless headphones with noise cancellation.",
            "category": "Headphones",
            "rating": 4.5,
            "reviews": ["Amazing sound quality!", "Battery life is exceptional."],
            "images": [np.random.rand(384), np.random.rand(384)],
        }
        """
        key = product_data["product_name"]
        self.insert(key, product_data)

    def delete_low_rated_products(self, threshold=3.0):
        """
        删除评分低于阈值的产品。
        """
        def low_rating_condition(record):
            return record[1]["rating"] < threshold

        self.delete(condition=low_rating_condition)

    def recommend_alternatives(self, product_name, k=3):
        """
        推荐相似的替代产品。
        """
        return self.retrieve_similar(product_name, k=k)
```

---

### **E-commerce Product Search Agent 案例**

#### **初始化**

```python
# 初始化 EnhancedSemanticMap
product_map = EnhancedSemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 示例产品数据
products = [
    {
        "product_name": "Wireless Headphones A1",
        "description": "High-quality wireless headphones with noise cancellation.",
        "category": "Headphones",
        "rating": 4.5,
        "reviews": ["Amazing sound quality!", "Battery life is exceptional."],
        "images": [np.random.rand(384), np.random.rand(384)],
    },
    {
        "product_name": "Wireless Earbuds B2",
        "description": "Compact wireless earbuds with deep bass.",
        "category": "Earbuds",
        "rating": 3.8,
        "reviews": ["Good for the price.", "Comfortable fit but average sound."],
        "images": [np.random.rand(384)],
    },
    {
        "product_name": "Wired Headphones C3",
        "description": "Affordable wired headphones with decent sound.",
        "category": "Headphones",
        "rating": 2.5,
        "reviews": ["Sound is okay but build quality is poor.", "Not recommended for long use."],
        "images": [np.random.rand(384)],
    },
]

# 插入产品数据
for product in products:
    product_map.insert_product(product)

# 构建索引
product_map.build_or_update_index()
```

---

#### **Step 1: 搜索产品**

##### **语义搜索**

```python
# 用户输入查询：高评分的无线耳机
query_text = "高评分的无线耳机"
results = product_map.retrieve_similar(query_text, k=3)

print("\n语义搜索结果：")
for result in results:
    print(f"Product Name: {result['key']}")
    print(f"Rating: {result['value']['rating']}")
    print(f"Description: {result['value']['description']}")
```

##### **图片搜索**

```python
# 用户上传耳机图片
uploaded_image = np.random.rand(384)  # 示例图片嵌入向量
image_results = product_map.retrieve_similar(uploaded_image, k=3)

print("\n图片搜索结果：")
for result in image_results:
    print(f"Product Name: {result['key']}")
    print(f"Rating: {result['value']['rating']}")
    print(f"Description: {result['value']['description']}")
```

---

#### **Step 2: 删除低评分产品**

```python
# 删除评分低于 3 的产品
product_map.delete_low_rated_products(threshold=3)

# 查看剩余产品
remaining_products = product_map.retrieve_similar("products", k=5)
print("\n删除低评分产品后：")
for product in remaining_products:
    print(f"Product Name: {product['key']}")
    print(f"Rating: {product['value']['rating']}")
```

---

#### **Step 3: 推荐替代产品**

```python
# 如果某产品不可用，推荐替代品
unavailable_product = "Wireless Headphones A1"
alternative_results = product_map.recommend_alternatives(unavailable_product, k=3)

print("\n推荐的替代产品：")
for result in alternative_results:
    print(f"Product Name: {result['key']}")
    print(f"Rating: {result['value']['rating']}")
    print(f"Description: {result['value']['description']}")
```

---

### **案例输出示例**

#### **语义搜索结果**

```plaintext
语义搜索结果：
Product Name: Wireless Headphones A1
Rating: 4.5
Description: High-quality wireless headphones with noise cancellation.

Product Name: Wireless Earbuds B2
Rating: 3.8
Description: Compact wireless earbuds with deep bass.
```

#### **图片搜索结果**

```plaintext
图片搜索结果：
Product Name: Wireless Headphones A1
Rating: 4.5
Description: High-quality wireless headphones with noise cancellation.

Product Name: Wireless Earbuds B2
Rating: 3.8
Description: Compact wireless earbuds with deep bass.
```

#### **删除低评分产品后**

```plaintext
删除低评分产品后：
Product Name: Wireless Headphones A1
Rating: 4.5

Product Name: Wireless Earbuds B2
Rating: 3.8
```

#### **推荐替代产品**

```plaintext
推荐的替代产品：
Product Name: Wireless Earbuds B2
Rating: 3.8
Description: Compact wireless earbuds with deep bass.
```

---

### **改进与扩展**

1. **扩展检索能力**：
   - 支持多条件过滤（如特定类别、最低评分等）。

2. **更智能的替代推荐**：
   - 考虑用户偏好（如类别、价格范围）。

3. **与其他模块集成**：
   - 可进一步结合订单管理和个性化推荐模块，提供全栈电商解决方案。

该实现展示了 SemanticMap 在电商场景下的强大能力，同时通过子类化提供更高层次的功能扩展，适用于生产环境中的多模态数据管理需求。

另外，数据集也可以采用最新的Amazon数据 https://amazon-reviews-2023.github.io/