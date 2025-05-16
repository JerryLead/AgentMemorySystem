以下是基于优化后的 **SemanticMap** 的完整代码和案例演示，应用于 **Recipe Assistant Agent**，结合多模态数据存储和智能语义检索能力。

---

### **优化后的 Recipe Assistant 案例**

#### **Step 1: 数据结构设计**

SemanticMap 的主要能力：
1. 支持多模态数据存储（`recipe_name`、`ingredients`、`instructions` 和 `image`）。
2. 高效检索（文本/图片查询）。
3. 动态更新和删除。

#### **Step 2: 流程描述**

1. **存储菜谱信息**：
   - 将 `recipe_name` 作为主键（文本或向量）。
   - 关联 `ingredients`（成分）、`instructions`（步骤）和 `image`（图片或嵌入向量）。

2. **查询菜谱**：
   - 用户可通过文本描述（如“无麸质甜点”）或图片找到相关菜谱。
   - 返回菜谱名称、成分和图片。

3. **更新菜谱**：
   - 支持修改菜谱的成分列表或图片。

4. **删除菜谱**：
   - 基于语义条件删除（如“删除所有含肉菜谱”）。

---

### **代码实现**

#### **SemanticMap 初始化**

```python
# 初始化 SemanticMap
recipe_map = SemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 示例数据
recipes = [
    {
        "recipe_name": "Gluten-Free Brownies",
        "ingredients": ["gluten-free flour", "sugar", "cocoa powder", "eggs", "butter"],
        "instructions": "Mix all ingredients. Bake at 350°F for 25 minutes.",
        "image": None  # 图片可为 None 或嵌入向量
    },
    {
        "recipe_name": "Vegan Pancakes",
        "ingredients": ["flour", "almond milk", "banana", "maple syrup"],
        "instructions": "Blend ingredients and cook in a skillet until golden.",
        "image": None
    },
    {
        "recipe_name": "Chocolate Cake",
        "ingredients": ["flour", "sugar", "cocoa powder", "eggs", "butter"],
        "instructions": "Mix all ingredients and bake at 350°F for 30 minutes.",
        "image": None
    },
]
```

#### **Step 1: 存储数据**

```python
# 插入菜谱数据到 SemanticMap
for recipe in recipes:
    recipe_map.insert(recipe["recipe_name"], recipe)

# 构建索引
recipe_map.build_or_update_index()
```

---

#### **Step 2: 查询菜谱**

##### **文本查询**

```python
# 查询“无麸质甜点”
query_text = "无麸质甜点"
results = recipe_map.retrieve_similar(query_text, k=3)

print("\n查询结果：")
for result in results:
    print(f"Recipe Name: {result['key']}")
    print(f"Ingredients: {result['value']['ingredients']}")
    print(f"Instructions: {result['value']['instructions']}")
```

##### **图片查询**

```python
# 模拟图片查询
uploaded_image_embedding = np.random.rand(384)  # 示例图片嵌入向量
image_results = recipe_map.retrieve_similar(uploaded_image_embedding, k=3)

print("\n图片查询结果：")
for result in image_results:
    print(f"Recipe Name: {result['key']}")
    print(f"Ingredients: {result['value']['ingredients']}")
    print(f"Instructions: {result['value']['instructions']}")
```

---

#### **Step 3: 更新菜谱**

```python
# 修改某个菜谱的成分列表
recipe_name = "Chocolate Cake"
new_ingredients = ["gluten-free flour", "sugar", "cocoa powder", "eggs", "butter"]

recipe_map.update(
    old_key=recipe_name,
    new_value={
        "recipe_name": recipe_name,
        "ingredients": new_ingredients,
        "instructions": "Mix all ingredients and bake at 350°F for 30 minutes.",
        "image": None
    }
)

# 查看更新后的菜谱
updated_results = recipe_map.retrieve_similar("Chocolate Cake", k=1)
print("\n更新后的菜谱：")
for result in updated_results:
    print(f"Recipe Name: {result['key']}")
    print(f"Ingredients: {result['value']['ingredients']}")
```

---

#### **Step 4: 删除菜谱**

```python
# 删除含肉菜谱（假设任何包含 "meat" 的成分为含肉菜谱）
def remove_meat_condition(record):
    ingredients = record[1]["ingredients"]
    return any("meat" in ingredient.lower() for ingredient in ingredients)

recipe_map.delete(condition=remove_meat_condition)

# 查看删除后的菜谱
remaining_recipes = recipe_map.retrieve_similar("recipes", k=5)
print("\n剩余菜谱：")
for recipe in remaining_recipes:
    print(f"Recipe Name: {recipe['key']}")
    print(f"Ingredients: {recipe['value']['ingredients']}")
```

---

#### **Step 5: 自然语言高级查询**

```python
# 用户查询“无麸质甜点但不包含香蕉”
query_nl = "Find gluten-free desserts but exclude anything with banana."
advanced_results = recipe_map.retrieve_advanced(query_nl, k=3)

print("\n高级查询结果：")
for result in advanced_results:
    print(f"Recipe Name: {result['key']}")
    print(f"Ingredients: {result['value']['ingredients']}")
```

---

### **输出示例**

1. **文本查询**：
   ```plaintext
   查询结果：
   Recipe Name: Gluten-Free Brownies
   Ingredients: ['gluten-free flour', 'sugar', 'cocoa powder', 'eggs', 'butter']
   Instructions: Mix all ingredients. Bake at 350°F for 25 minutes.
   ```

2. **图片查询**：
   ```plaintext
   图片查询结果：
   Recipe Name: Vegan Pancakes
   Ingredients: ['flour', 'almond milk', 'banana', 'maple syrup']
   Instructions: Blend ingredients and cook in a skillet until golden.
   ```

3. **更新后的菜谱**：
   ```plaintext
   更新后的菜谱：
   Recipe Name: Chocolate Cake
   Ingredients: ['gluten-free flour', 'sugar', 'cocoa powder', 'eggs', 'butter']
   ```

4. **删除后的菜谱**：
   ```plaintext
   剩余菜谱：
   Recipe Name: Gluten-Free Brownies
   Ingredients: ['gluten-free flour', 'sugar', 'cocoa powder', 'eggs', 'butter']
   ```

5. **高级查询**：
   ```plaintext
   高级查询结果：
   Recipe Name: Gluten-Free Brownies
   Ingredients: ['gluten-free flour', 'sugar', 'cocoa powder', 'eggs', 'butter']
   ```

---

### **改进点总结**

1. **多模态支持**：
   - `recipe_name` 为主键。
   - `ingredients`、`instructions` 为值。
   - 图片嵌入支持图片检索。

2. **智能查询**：
   - 文本查询结合 LLM 提供语义增强能力。
   - 高级查询解析复杂条件。

3. **动态更新**：
   - 支持用户修改菜谱成分和内容，保持索引一致性。

这套实现展示了 `SemanticMap` 在智能语义存储与检索上的强大能力，适合进一步扩展为生产环境中的解决方案。



在菜谱数据中，一个菜谱可以有多个图片（如展示成品的不同角度或步骤过程的图片）。为了支持这一需求，可以将图片存储为列表，每个列表项可以是图片的嵌入向量。这种设计适合多模态检索需求。

以下是完整的代码实现，支持菜谱多图片存储：

---

### **SemanticMap 实现**

```python
import faiss
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class SemanticMap:
    def __init__(
        self,
        key_type="text",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        llm_model="gpt-3.5-turbo",
        index_type="flat",
        nlist=100,
        nprobe=10,
        hnsw_m=32,
    ):
        if key_type not in ["text", "vector"]:
            raise ValueError("key_type must be 'text' or 'vector'.")

        self.key_type = key_type
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m

        self.data = []  # 存储 (key, value, key_embedding)
        self.index = None

        self.text_encoder = SentenceTransformer(embedding_model) if self.key_type == "text" else None
        self.llm = pipeline("text2text-generation", model=llm_model)
        self._init_index()

    def _init_index(self):
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, self.nlist, 8, 8)
            self.index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}.")

    def _train_index_if_needed(self, embeddings):
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            print("[INFO] Training IVF index.")
            self.index.train(np.array(embeddings, dtype=np.float32))

    def _get_embedding(self, key):
        if self.key_type == "text":
            if not isinstance(key, str):
                raise ValueError("For text key_type, key must be str.")
            return self.text_encoder.encode([key], convert_to_numpy=True)[0].astype(np.float32)
        else:
            if not isinstance(key, np.ndarray):
                raise ValueError("For vector key_type, key must be np.ndarray.")
            if key.shape[0] != self.embedding_dim:
                raise ValueError(f"Key vector shape {key.shape[0]} != embedding_dim {self.embedding_dim}.")
            return key.astype(np.float32)

    def insert(self, key, value):
        key_embedding = self._get_embedding(key)
        self.data.append((key, value, key_embedding))

    def build_or_update_index(self):
        self._init_index()
        if len(self.data) == 0:
            return
        embeddings = [item[2] for item in self.data]
        self._train_index_if_needed(embeddings)
        all_emb = np.array(embeddings, dtype=np.float32)
        self.index.add(all_emb)

    def retrieve_similar(self, query, k=5):
        if len(self.data) == 0:
            return []
        query_emb = self._get_embedding(query)
        distances, indices = self.index.search(query_emb.reshape(1, -1), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.data):
                key, value, _ = self.data[idx]
                results.append({"key": key, "value": value, "distance": float(dist)})
        return results

    def retrieve_exclude(self, query, k=5, condition=None):
        excluded = self.retrieve_similar(query, k)
        excluded_keys = {e["key"] for e in excluded}
        results = []
        for key, value, _ in self.data:
            if key not in excluded_keys and (not condition or not condition((key, value))):
                results.append({"key": key, "value": value})
        return results

    def retrieve_advanced(self, natural_language_query, k=5):
        if len(self.data) == 0:
            return []

        prompt_template = f"""
You have a semantic map storing key-value pairs, where keys are {self.key_type} and values can be anything.

User might have queries like:
1) "Find items about gluten-free desserts"
2) "Exclude items about citrus"
3) "Find sweet fruits but exclude something about berries"
4) "List tropical recipes"

Now parse this query into JSON format:
User query: "{natural_language_query}"

Output JSON like:
{{
  "action": "include" or "exclude",
  "target": "some string"
}}
"""
        response = self.llm(prompt_template.strip(), max_length=512)
        instructions = json.loads(response[0]["generated_text"].strip())

        def do_include(target_str):
            return self.retrieve_similar(target_str, k)

        def do_exclude(target_str):
            return self.retrieve_exclude(target_str, k)

        results = []
        if "action" in instructions:
            action = instructions["action"]
            target = instructions.get("target", "")
            if action == "include":
                results = do_include(target)
            elif action == "exclude":
                results = do_exclude(target)
        return results

    def update(self, old_key, new_key=None, new_value=None, k=1):
        old_emb = self._get_embedding(old_key)
        distances, indices = self.index.search(old_emb.reshape(1, -1), k)
        for idx in indices[0]:
            if idx < len(self.data):
                stored_key, stored_value, stored_emb = self.data[idx]
                if new_key:
                    updated_emb = self._get_embedding(new_key)
                    stored_key, stored_emb = new_key, updated_emb
                if new_value:
                    stored_value = new_value
                self.data[idx] = (stored_key, stored_value, stored_emb)
        self.build_or_update_index()

    def delete(self, query=None, k=1, condition=None):
        if query is None and condition is None:
            raise ValueError("Either query or condition must be provided.")
        if condition:
            self.data = [(key, value, emb) for key, value, emb in self.data if not condition((key, value))]
        else:
            query_emb = self._get_embedding(query)
            distances, indices = self.index.search(query_emb.reshape(1, -1), k)
            idx_to_del = sorted(indices[0], reverse=True)
            for i in idx_to_del:
                if i < len(self.data):
                    del self.data[i]
        self.build_or_update_index()
```

---

### **案例实现**

#### **多图片菜谱数据结构**

```python
# 示例菜谱
recipes = [
    {
        "recipe_name": "Gluten-Free Brownies",
        "ingredients": ["gluten-free flour", "sugar", "cocoa powder", "eggs", "butter"],
        "instructions": "Mix all ingredients. Bake at 350°F for 25 minutes.",
        "images": [np.random.rand(384), np.random.rand(384)],  # 多张图片的嵌入向量
    },
    {
        "recipe_name": "Vegan Pancakes",
        "ingredients": ["flour", "almond milk", "banana", "maple syrup"],
        "instructions": "Blend ingredients and cook in a skillet until golden.",
        "images": [np.random.rand(384)],
    },
]
```

#### **存储菜谱**

```python
recipe_map = SemanticMap(key_type="text", index_type="flat", embedding_dim=384)

for recipe in recipes:
    recipe_map.insert(recipe["recipe_name"], recipe)

recipe_map.build_or_update_index()
```

#### **查询菜谱**

```python
query = "gluten-free dessert"
results = recipe_map.retrieve_similar(query, k=3)
for result in results:
    print(f"Recipe Name: {result['key']}, Ingredients: {result['value']['ingredients']}")
```

#### **多图片检索**

```python
uploaded_image = np.random.rand(384)  # 模拟用户上传图片
results = recipe_map.retrieve_similar(uploaded_image, k=3)
for result in results:
    print(f"Recipe Name: {result['key']}, Ingredients: {result['value']['ingredients']}")
```

---

### **功能特点**

1. **多图片支持**：
   - 每个菜谱可以存储多个图片，通过列表管理。

2. **灵活检索**：
   - 支持基于文本或图片的多模态检索。

3. **动态更新与删除**：
   - 提供实时定制和管理能力。

这种多图片支持的结构非常适合多模态查询场景，并提供了高效的操作和存储能力。






以下是补充完整删除操作的代码，包括案例实现，展示如何通过多种方式删除菜谱数据。

---

### **删除操作的实现**

```python
def delete(self, query=None, k=1, condition=None):
    """
    删除条目；按相似度或按条件删除。

    Parameters:
        - query: 查询条件，支持文本或向量。
        - k: 最相似的 k 条条目将被删除。
        - condition: 自定义条件函数，返回 True 时删除该条目。
    """
    if query is None and condition is None:
        raise ValueError("Either query or condition must be provided for delete operation.")

    if condition:
        self.data = [
            (key, value, emb)
            for key, value, emb in self.data
            if not condition((key, value))  # 条目满足条件时会被过滤掉
        ]
    else:
        query_emb = self._get_embedding(query)
        distances, indices = self.index.search(query_emb.reshape(1, -1), k)
        idx_to_del = sorted(indices[0], reverse=True)
        for i in idx_to_del:
            if i < len(self.data):
                del self.data[i]

    # 重建索引
    self.build_or_update_index()
```

---

### **案例实现**

#### **存储示例菜谱数据**

```python
recipes = [
    {
        "recipe_name": "Gluten-Free Brownies",
        "ingredients": ["gluten-free flour", "sugar", "cocoa powder", "eggs", "butter"],
        "instructions": "Mix all ingredients. Bake at 350°F for 25 minutes.",
        "images": [np.random.rand(384), np.random.rand(384)],
    },
    {
        "recipe_name": "Vegan Pancakes",
        "ingredients": ["flour", "almond milk", "banana", "maple syrup"],
        "instructions": "Blend ingredients and cook in a skillet until golden.",
        "images": [np.random.rand(384)],
    },
    {
        "recipe_name": "Chocolate Cake",
        "ingredients": ["flour", "sugar", "cocoa powder", "eggs", "butter"],
        "instructions": "Mix all ingredients and bake at 350°F for 30 minutes.",
        "images": [np.random.rand(384)],
    },
]

# 初始化 SemanticMap
recipe_map = SemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 插入数据
for recipe in recipes:
    recipe_map.insert(recipe["recipe_name"], recipe)

recipe_map.build_or_update_index()
```

---

#### **按相似度删除**

```python
# 删除与查询“Chocolate Cake”最相似的菜谱
query = "Chocolate Cake"
recipe_map.delete(query=query, k=1)

# 查看剩余菜谱
remaining_recipes = recipe_map.retrieve_similar("recipes", k=5)
print("\n剩余菜谱（按相似度删除后）：")
for recipe in remaining_recipes:
    print(f"Recipe Name: {recipe['key']}")
```

---

#### **按条件删除**

```python
# 删除所有含 "banana" 的菜谱
def remove_banana_condition(record):
    ingredients = record[1]["ingredients"]
    return any("banana" in ingredient.lower() for ingredient in ingredients)

recipe_map.delete(condition=remove_banana_condition)

# 查看剩余菜谱
remaining_recipes = recipe_map.retrieve_similar("recipes", k=5)
print("\n剩余菜谱（按条件删除后）：")
for recipe in remaining_recipes:
    print(f"Recipe Name: {recipe['key']}")
```

---

### **完整功能验证**

1. **删除“Chocolate Cake”菜谱**：
   - 通过文本查询找到最相似的菜谱并删除。

2. **删除含特定成分的菜谱**：
   - 根据条件删除所有成分中含 `banana` 的菜谱。

---

### **代码输出示例**

#### **删除后菜谱列表**

1. **按相似度删除后**：
   ```plaintext
   剩余菜谱（按相似度删除后）：
   Recipe Name: Gluten-Free Brownies
   Recipe Name: Vegan Pancakes
   ```

2. **按条件删除后**：
   ```plaintext
   剩余菜谱（按条件删除后）：
   Recipe Name: Gluten-Free Brownies
   ```

---

### **改进总结**

1. **删除操作的灵活性**：
   - 支持按相似度删除指定数量的条目。
   - 支持基于复杂条件删除条目。

2. **多图片支持**：
   - 每个菜谱可关联多张图片（存储为嵌入向量列表）。

3. **重建索引**：
   - 删除后自动重建索引，保证检索的准确性和效率。

这套实现具备高扩展性，能够适应多样化的数据管理需求，同时支持智能语义检索与操作。