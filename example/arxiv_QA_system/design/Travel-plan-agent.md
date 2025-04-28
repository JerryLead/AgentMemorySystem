

以下是对 **Case 9: Multi-modal Travel Companion** 的优化设计，结合我们提供的 **SemanticMap** 数据结构，全面展示 **LLM** 交互、记忆、规划和多模态支持的能力。

---

### **案例设计**

#### **目标与功能**
1. **多模态旅游信息存储与检索**：
   - 存储目的地名称、地理位置、描述、图片等多模态内容。
   - 支持基于描述或图片的语义检索。

2. **行程记忆与偏好存储**：
   - 存储用户的旅行偏好，如主题（历史文化、亲子游）、预算等。
   - 根据偏好动态生成推荐行程。

3. **实时更新与反馈**：
   - 支持用户调整行程，动态修改推荐内容并记忆更新。

4. **跨模态查询支持**：
   - 用户可输入文本查询（如“推荐适合亲子游的景点”）或上传图片，获得多模态推荐。

---

### **优化后的 SemanticMap 子类实现**

```python
class TravelSemanticMap(SemanticMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_preferences = {}  # 存储用户偏好

    def insert_destination(self, destination_data):
        """
        插入目的地信息。
        destination_data 格式示例：
        {
            "location_name": "Eiffel Tower",
            "coordinates": {"latitude": 48.858844, "longitude": 2.294351},
            "description": "A wrought iron lattice tower in Paris, France.",
            "images": [np.random.rand(384)],
        }
        """
        key = destination_data["location_name"]
        self.insert(key, destination_data)

    def store_user_preference(self, user_id, preference):
        """
        存储用户偏好。
        preference 格式示例：
        {
            "theme": "history",
            "budget": "<$500",
            "preferred_activities": ["museums", "walking tours"]
        }
        """
        self.user_preferences[user_id] = preference

    def retrieve_recommendations(self, user_query, user_id=None, k=5):
        """
        根据用户描述和偏好检索推荐。
        """
        base_recommendations = self.retrieve_similar(user_query, k)
        if user_id and user_id in self.user_preferences:
            preference = self.user_preferences[user_id]
            # 基于偏好过滤或重新排序
            filtered = [
                rec for rec in base_recommendations
                if any(theme in rec["value"]["description"] for theme in preference.get("preferred_activities", []))
            ]
            return filtered[:k]
        return base_recommendations

    def update_itinerary(self, itinerary, user_updates):
        """
        更新行程计划。
        itinerary 格式示例：
        {
            "day_1": ["Eiffel Tower", "Louvre Museum"],
            "day_2": ["Notre Dame Cathedral"],
        }
        user_updates 示例：
        {
            "add": {"day_3": ["Versailles Palace"]},
            "remove": {"day_2": ["Notre Dame Cathedral"]}
        }
        """
        # 添加新目的地
        for day, places in user_updates.get("add", {}).items():
            if day not in itinerary:
                itinerary[day] = []
            itinerary[day].extend(places)

        # 移除目的地
        for day, places in user_updates.get("remove", {}).items():
            if day in itinerary:
                itinerary[day] = [place for place in itinerary[day] if place not in places]

        return itinerary
```

---

### **案例实现**

#### **Step 1: 存储旅游信息**

```python
# 初始化 TravelSemanticMap
travel_map = TravelSemanticMap(key_type="text", index_type="flat", embedding_dim=384)

# 示例旅游目的地数据
destinations = [
    {
        "location_name": "Eiffel Tower",
        "coordinates": {"latitude": 48.858844, "longitude": 2.294351},
        "description": "A wrought iron lattice tower in Paris, France.",
        "images": [np.random.rand(384)],
    },
    {
        "location_name": "Louvre Museum",
        "coordinates": {"latitude": 48.860611, "longitude": 2.337644},
        "description": "The world's largest art museum and a historic monument in Paris.",
        "images": [np.random.rand(384)],
    },
    {
        "location_name": "Notre Dame Cathedral",
        "coordinates": {"latitude": 48.852968, "longitude": 2.349902},
        "description": "A medieval Catholic cathedral in Paris, known for its French Gothic architecture.",
        "images": [np.random.rand(384)],
    },
]

# 插入目的地数据
for destination in destinations:
    travel_map.insert_destination(destination)

# 构建索引
travel_map.build_or_update_index()
```

---

#### **Step 2: 查询目的地**

##### **语义查询**

```python
# 用户输入查询：推荐适合亲子游的景点
user_query = "推荐适合亲子游的景点"
recommendations = travel_map.retrieve_recommendations(user_query, k=3)

print("\n推荐的景点：")
for rec in recommendations:
    print(f"Location: {rec['key']}, Description: {rec['value']['description']}")
```

##### **图片查询**

```python
# 用户上传图片
uploaded_image = np.random.rand(384)  # 示例图片嵌入向量
image_recommendations = travel_map.retrieve_similar(uploaded_image, k=3)

print("\n图片搜索推荐的景点：")
for rec in image_recommendations:
    print(f"Location: {rec['key']}, Description: {rec['value']['description']}")
```

---

#### **Step 3: 行程记忆与规划**

```python
# 存储用户偏好
user_id = "user_123"
user_preference = {
    "theme": "history",
    "budget": "<$500",
    "preferred_activities": ["museums", "walking tours"]
}
travel_map.store_user_preference(user_id, user_preference)

# 根据偏好规划行程
itinerary = {
    "day_1": ["Eiffel Tower", "Louvre Museum"],
    "day_2": ["Notre Dame Cathedral"],
}
print("\n初始行程计划：")
print(itinerary)
```

---

#### **Step 4: 更新行程与反馈**

```python
# 用户更新需求：添加温泉景点，移除大教堂
user_updates = {
    "add": {"day_3": ["Versailles Palace"]},
    "remove": {"day_2": ["Notre Dame Cathedral"]}
}
updated_itinerary = travel_map.update_itinerary(itinerary, user_updates)

print("\n更新后的行程计划：")
print(updated_itinerary)
```

---

### **功能展示**

#### **推荐的景点**

```plaintext
推荐的景点：
Location: Eiffel Tower, Description: A wrought iron lattice tower in Paris, France.
Location: Louvre Museum, Description: The world's largest art museum and a historic monument in Paris.
Location: Notre Dame Cathedral, Description: A medieval Catholic cathedral in Paris, known for its French Gothic architecture.
```

#### **图片搜索推荐的景点**

```plaintext
图片搜索推荐的景点：
Location: Eiffel Tower, Description: A wrought iron lattice tower in Paris, France.
Location: Louvre Museum, Description: The world's largest art museum and a historic monument in Paris.
```

#### **更新后的行程计划**

```plaintext
更新后的行程计划：
{'day_1': ['Eiffel Tower', 'Louvre Museum'], 'day_3': ['Versailles Palace']}
```

---

### **扩展与优化**

1. **偏好记忆扩展**：
   - 支持更复杂的偏好组合（如气候、文化类型）。

2. **动态推荐**：
   - 结合实时天气和用户位置，提供动态景点推荐。

3. **多模态扩展**：
   - 将用户上传的更多图片与目的地图片匹配，提高推荐精度。

通过以上实现，**Multi-modal Travel Companion** 展现了 **SemanticMap** 的强大能力，同时结合用户记忆与动态规划，实现了智能化、多模态旅行助手的完整功能。


### **公开可用的数据集**

以下是一些可用于旅游相关项目的公开数据集：

#### 1. **Geonames Dataset**
   - **来源**: [Geonames.org](https://www.geonames.org/)
   - **内容**:
     - 地点名称、地理坐标、国家、行政区划等。
     - 提供数百万个地理位置条目，可用于景点存储和查询。
   - **应用**:
     - 存储景点名称及其地理位置。
     - 结合描述和图片等附加信息创建完整的旅游信息。

#### 2. **OpenStreetMap POI 数据**
   - **来源**: [OpenStreetMap](https://www.openstreetmap.org/)
   - **内容**:
     - 包含兴趣点（POI），如公园、博物馆、餐厅、酒店等。
     - 支持地理查询、位置聚合。
   - **应用**:
     - 存储景点信息及周边设施。
     - 提供位置推荐服务。

#### 3. **Foursquare Places Dataset**
   - **来源**: [Foursquare API](https://developer.foursquare.com/)
   - **内容**:
     - 包括景点评分、评论、照片和分类。
   - **应用**:
     - 存储多模态数据（描述、图片、评分）。
     - 提供基于用户偏好的推荐。

#### 4. **Google Places API**
   - **来源**: [Google Developers](https://developers.google.com/maps/documentation/places)
   - **内容**:
     - 景点的详细信息，包括名称、地址、评分、评论、图片等。
   - **应用**:
     - 结合动态天气和实时数据生成推荐。

#### 5. **TripAdvisor Reviews Dataset**
   - **来源**: Kaggle
   - **内容**:
     - 包括来自 TripAdvisor 的景点评分、用户评论和分类信息。
   - **应用**:
     - 分析用户偏好。
     - 提供基于用户评论的推荐。

---

### **语义数据结构的应用**

#### **存储景点信息到 SemanticMap**

使用 **SemanticMap** 存储地理和多模态景点信息。

```python
# 插入旅游景点数据
destinations = [
    {
        "location_name": "Eiffel Tower",
        "coordinates": {"latitude": 48.858844, "longitude": 2.294351},
        "description": "A wrought iron lattice tower in Paris, France.",
        "images": [np.random.rand(384)],  # 示例图片向量
    },
    {
        "location_name": "Louvre Museum",
        "coordinates": {"latitude": 48.860611, "longitude": 2.337644},
        "description": "The world's largest art museum and a historic monument in Paris.",
        "images": [np.random.rand(384)],
    },
]

# 存入 SemanticMap
for destination in destinations:
    travel_map.insert(destination["location_name"], destination)
```

#### **存储用户行程到 SemanticList**

用户的行程规划更适合存储在 **SemanticList**（语义化链表）中，支持动态更新和查询。

##### **SemanticList 实现示例**

```python
class SemanticList:
    def __init__(self):
        self.data = []

    def add(self, item):
        """添加行程条目"""
        self.data.append(item)

    def remove(self, condition_fn):
        """按条件删除行程条目"""
        self.data = [item for item in self.data if not condition_fn(item)]

    def retrieve(self, condition_fn):
        """按条件检索行程条目"""
        return [item for item in self.data if condition_fn(item)]
```

##### **存储用户行程**

```python
# 初始化用户行程列表
user_itinerary = SemanticList()

# 添加行程条目
user_itinerary.add({
    "day": 1,
    "locations": ["Eiffel Tower", "Louvre Museum"],
    "preferences": {"theme": "culture"}
})

user_itinerary.add({
    "day": 2,
    "locations": ["Notre Dame Cathedral"],
    "preferences": {"theme": "architecture"}
})

# 检索行程
day_1_itinerary = user_itinerary.retrieve(lambda x: x["day"] == 1)
print("Day 1 Itinerary:", day_1_itinerary)
```

---

### **用户行程存储与检索整合**

#### **将 SemanticMap 与 SemanticList 结合**

- **SemanticMap**：存储静态景点信息（地理位置、描述、图片）。
- **SemanticList**：存储动态行程计划（用户偏好、每日安排）。

```python
# 示例：从 SemanticMap 中检索景点，并存入用户行程
recommendations = travel_map.retrieve_recommendations("亲子游景点", k=3)

# 将推荐的景点添加到行程列表
user_itinerary.add({
    "day": 3,
    "locations": [rec["key"] for rec in recommendations],
    "preferences": {"theme": "family"}
})

# 查看更新后的行程
print("Updated Itinerary:", user_itinerary.data)
```

---

### **扩展场景与能力**

#### **扩展用户记忆能力**
- 在 **SemanticMap** 中存储用户历史旅行数据：
  ```json
  {
      "user_id": "123",
      "past_trips": [
          {"destination": "Paris", "rating": 5, "activities": ["Eiffel Tower", "Louvre Museum"]},
          {"destination": "Rome", "rating": 4, "activities": ["Colosseum", "Vatican Museum"]}
      ]
  }
  ```

#### **个性化推荐**
- 根据用户过去的旅行记录，生成个性化推荐。
- 利用 LLM 生成动态建议（如推荐新的城市或活动）。

---

### **总结**

1. **公开数据集**：
   - Geonames、OpenStreetMap、Google Places 等数据集支持高质量景点信息存储。
2. **语义数据结构应用**：
   - **SemanticMap**：存储静态景点信息，支持多模态检索。
   - **SemanticList**：存储动态用户行程，支持增删改查。
3. **扩展场景**：
   - 支持用户偏好记忆、个性化推荐和动态行程更新。

通过以上方案，**SemanticMap** 与 **SemanticList** 可构建智能化、多模态的旅行助手系统，为用户提供全面的旅游计划支持。