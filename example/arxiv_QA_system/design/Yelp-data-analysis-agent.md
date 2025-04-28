Q: 下面根据我们的semanticmap+semanticgraph设计，来对具体数据进行应用。对于下面的Yelp数据集，我们可以进行很多查询，如“
1. Recommend some restaurants in California with good environment and spicy Chinese food.”，

这样可以体现语义查询，因为需要查询photo embeddings，review embeddings，还有business.metadata，并经过LLM聚合得到结果。你帮我多设计 一些这样的查询，最好能够利用上semanticmap embeddings+semanticgraph关系。


A: 下面给出一个示例脚本，演示如何对 **Yelp** 数据进行“**混合索引**”+“**丰富查询**”的设计：既利用 **`SemanticMap`** 做**向量检索**（对 `review`、`photo.caption`、`business.name` 等文字嵌入），又利用 **`SemanticGraph`** 存**业务结构**（`business -> review`、`business -> photo`、`review -> user` 等）。然后，我们给出**多种查询例子**，其中既要做**semantic(embedding)**搜索，也要做**graph**关系拓展，从而能一次返回更多上下文给 RAG Agent 调用（如“spicy Chinese food + environment + photo embeddings”）。

> **说明**：  
> - 以下代码仅演示**数据结构**与**查询接口**的设计原型，未做大规模实际数据导入。真实的 Yelp JSON（business.json, review.json, photo.json 等）可能数十万行，需要批量插入并构建索引。  
> - 重点在如何**设计**查询函数，返回“相似查询结果 + 关联节点/上下文”，而不是一次性回答自然语言问题。后续可将这些函数封装成**Tool**提供给 LLM/RAG。  

---

# YelpQA_Agent.py

```python
import os
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

############################################
# 1) SemanticMap (向量检索)
############################################
class SemanticMap:
    """
    简化: 负责 <key, value, embedding> 的增删改查 + 向量检索(Faiss/HNSW/IVF...).
    """
    def __init__(self, embedding_dim=384, embedding_model="all-MiniLM-L6-v2", index_type="flat"):
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.index_type = index_type

        self.data = []  # list of (key, value, embedding)
        self.index = None

        self.encoder = SentenceTransformer(self.embedding_model)
        self._init_index()

    def _init_index(self):
        if self.index_type=="flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type=="hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError("Unsupported index_type")

    def _get_emb(self, text):
        emb = self.encoder.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        return emb

    def insert(self, key, value, text_for_embedding=None):
        if text_for_embedding is None:
            text_for_embedding = str(value)
        emb = self._get_emb(text_for_embedding)
        self.data.append((key, value, emb))

    def build_index(self):
        self._init_index()
        if len(self.data)==0:
            return
        arr_emb = np.array([d[2] for d in self.data], dtype=np.float32)
        self.index.add(arr_emb)

    def retrieve_similar(self, query, k=5):
        if len(self.data)==0:
            return []
        q_emb = self._get_emb(query).reshape(1, -1)
        distances, indices = self.index.search(q_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.data):
                key, val, emb = self.data[idx]
                results.append({
                    "key": key,
                    "value": val,
                    "distance": float(dist)
                })
        return results


############################################
# 2) SemanticSimpleGraph (业务关系)
############################################
class SemanticSimpleGraph:
    """
    记录Yelp业务关系: business -> review, business -> photo, review -> user, etc.
    同时具备一个 semantic_map 用于文本embedding检索(分层).
    """
    def __init__(self, semantic_map=None):
        self.graph_relations = {}  # { nodeKey: {"parents":{}, "children":{}, "links":{}} }
        self.semantic_map = semantic_map if semantic_map else SemanticMap()

    def _ensure_node(self, key):
        if key not in self.graph_relations:
            self.graph_relations[key] = {
                "parents": {},
                "children": {},
                "links": {}
            }

    def add_node(self, key, value, parent_keys=None, parent_relation="contains", text_for_embedding=None):
        """
        Insert (key, value) into semantic_map, also record relation parent->child
        """
        self.semantic_map.insert(key, value, text_for_embedding)
        self._ensure_node(key)
        if parent_keys:
            for p in parent_keys:
                self._ensure_node(p)
                self.graph_relations[p]["children"][key] = parent_relation
                self.graph_relations[key]["parents"][p] = parent_relation

    def insert_edge(self, src_key, dst_key, relation="link"):
        self._ensure_node(src_key)
        self._ensure_node(dst_key)
        self.graph_relations[src_key]["links"][dst_key] = relation

    def link_nodes(self, key1, key2, relation="link"):
        self.insert_edge(key1, key2, relation)
        self.insert_edge(key2, key1, relation)

    def build_index(self):
        self.semantic_map.build_index()

    def retrieve_similar_nodes(self, query, k=5):
        return self.semantic_map.retrieve_similar(query, k)

    # graph getters
    def get_parents(self, key):
        if key in self.graph_relations:
            return self.graph_relations[key]["parents"]
        return {}
    def get_children(self, key):
        if key in self.graph_relations:
            return self.graph_relations[key]["children"]
        return {}
    def get_links(self, key):
        if key in self.graph_relations:
            return self.graph_relations[key]["links"]
        return {}

############################################
# 3) 构建Yelp结构 (demo)
############################################
def build_demo_yelp_graph():
    """
    简化示例: 构建 business / review / photo / user 节点, 以及 basic edges.
    """
    s_map = SemanticMap(index_type="flat")  # or "hnsw"
    s_graph = SemanticSimpleGraph(s_map)

    # 1) 伪造 business
    #   key="business_XXX", value里包含 {type="business", name, city, categories, ...}
    for i in range(3):
        bid = f"business_{i}"
        bval = {
            "type":"business",
            "name":f"Spicy Palace {i}",
            "city":"San Francisco" if i<2 else "Los Angeles",
            "categories":["Chinese","Spicy","CasualDining"] if i<2 else ["Bar","Cocktails"],
            "stars": 4.5 if i<2 else 4.0
        }
        s_graph.add_node(bid, bval, text_for_embedding=f"{bval['name']} {bval['city']} {','.join(bval['categories'])}")
        # 2) 伪造 review
        for r_idx in range(2):
            rid = f"review_{i}_{r_idx}"
            rtext = f"This is a review about the environment and spicy food index={r_idx}"
            r_val = {
                "type":"review",
                "text":rtext,
                "business_id":bid
            }
            s_graph.add_node(rid, r_val, parent_keys=[bid], text_for_embedding=rtext)
        # 3) 伪造 photo
        for p_idx in range(1):
            pid = f"photo_{i}_{p_idx}"
            pcap = f"Spicy chili sauce dish photo for {bval['name']}"
            p_val = {
                "type":"photo",
                "caption": pcap,
                "business_id":bid
            }
            s_graph.add_node(pid, p_val, parent_keys=[bid], text_for_embedding=pcap)

    # build index
    s_graph.build_index()
    return s_graph

############################################
# 4) 设计新的查询接口: YelpGraphQuery
############################################
class YelpGraphQuery:
    """
    提供多种查询函数, 返回 embedding相似节点 + 关联数据(父business, child review, etc.)
    让RAG Agent可拿更多上下文
    """
    def __init__(self, yelp_graph: SemanticSimpleGraph):
        self.graph = yelp_graph
        self.s_map = yelp_graph.semantic_map

    def query_similar(self, user_query, k=5):
        """
        最简单: 只做embedding相似
        """
        return self.s_map.retrieve_similar(user_query, k)

    def query_with_context(self, user_query, k=5, expand_level=1):
        """
        Step1: embedding top-k
        Step2: 对每个结果, gather 父/子/links => 形成上下文
        """
        top_res = self.s_map.retrieve_similar(user_query, k)
        results = []
        for r in top_res:
            nkey = r["key"]
            dist = r["distance"]
            val = r["value"]

            # gather parents, children
            parents = self.graph.get_parents(nkey)
            children = self.graph.get_children(nkey)
            links = self.graph.get_links(nkey)
            results.append({
                "node_key": nkey,
                "distance": dist,
                "value": val,
                "parents": parents,
                "children": children,
                "links": links
            })
        return results

    def query_business_in_city(self, city, user_query, k=5):
        """
        先symbol filter city, 然后embedding search in those node embeddings
        (可以先filter business节点, 再embedding搜索 photo/review? 这里仅演示.)
        """
        # gather all business matching city
        biz_keys = []
        for (key, val, emb) in self.s_map.data:
            if val.get("type")=="business" and val.get("city","")==city:
                biz_keys.append(key)

        # 取出embedding search for user_query
        # 这里 demo: 先embedding top=20 from all,再保留business in city
        top_res = self.s_map.retrieve_similar(user_query, k=20)
        final = []
        for r in top_res:
            if r["key"] in biz_keys:
                final.append(r)
            if len(final)>=k:
                break
        return final

    def multi_hop_search(self, user_query, hop=2):
        """
        Example: 先embedding search top=3 => each node => BFS hop=2, gather child or photo
        """
        top_res = self.s_map.retrieve_similar(user_query, 3)
        expanded_info = []
        for r in top_res:
            nkey = r["key"]
            dist = r["distance"]
            sub_nodes = self._bfs(nkey, hop)
            expanded_info.append({
                "root":r,
                "sub_nodes":sub_nodes
            })
        return expanded_info

    def _bfs(self, start_key, max_depth=2):
        from collections import deque
        visited = set([start_key])
        queue = deque([(start_key, 0)])
        results = []
        while queue:
            cur, depth = queue.popleft()
            if depth<max_depth:
                # explore children & links
                children = self.graph.get_children(cur)
                for c, rel in children.items():
                    if c not in visited:
                        visited.add(c)
                        queue.append((c, depth+1))
                        results.append((c, rel, "child"))
                links = self.graph.get_links(cur)
                for l, rel in links.items():
                    if l not in visited:
                        visited.add(l)
                        queue.append((l, depth+1))
                        results.append((l, rel, "link"))
        return results

############################################
# 5) Demo usage
############################################
def main_demo():
    # 构建demo YelpGraph
    yelp_graph = build_demo_yelp_graph()
    query_api = YelpGraphQuery(yelp_graph)

    # 一些查询例子

    # 1) "Recommend some restaurants in California with good environment and spicy Chinese food."
    # -> 先embedding,再context
    q1 = "restaurants in California with good environment and spicy Chinese food"
    r1 = query_api.query_with_context(q1, k=5)
    print(f"\n=== Query1: {q1} ===")
    for item in r1:
        print(item)

    # 2) "Find a place to celebrate birthday, with photo showing cozy or big group seating"
    q2 = "photo showing big group seating birthday"
    r2 = query_api.query_with_context(q2, k=3)
    print(f"\n=== Query2: {q2} ===")
    for item in r2:
        print(item)

    # 3) "Multi-hop search: from environment mention => gather child"
    q3 = "environment mention"
    r3 = query_api.multi_hop_search(q3, hop=2)
    print(f"\n=== Query3: multi-hop BFS for '{q3}' ===")
    for block in r3:
        print("Root:", block["root"])
        print("SubNodes:", block["sub_nodes"])

    # 4) "Symbol filter: city=San Francisco + user_query='spicy' => get top business"
    q4 = "spicy"
    r4 = query_api.query_business_in_city("San Francisco", q4, k=5)
    print(f"\n=== Query4: city=SF + '{q4}' =>")
    for x in r4:
        print(x)

if __name__ == "__main__":
    main_demo()
```

---

## 说明

1. **Yelp 数据：**  
   - 我们在 `build_demo_yelp_graph()` 里简化地插入了一些**business**节点（带 `name`, `city`, `categories` 等），**review**节点（`text`），**photo**节点（`caption`），并用**parent_keys**存**business->review**或**business->photo**关系。  
   - 实际中你需要**遍历** `business.json`, `review.json`, `photo.json`, 并调用 `add_node(...)`、`insert_edge(...)` 去插入数据和关系。

2. **查询接口**：  
   - **`query_with_context`**：先**embedding** top-k => 再**graph**拿parents/children/links => 返回更多信息；  
   - **`query_business_in_city`**：先**symbol filter**(city) => 再embedding => 也可以反过来；  
   - **`multi_hop_search`**：一次embedding找根节点 -> BFS多跳收集子节点或links => 让**RAG**有更多上下文(例如 business->photo->review->... )。

3. **查询例子**：  
   - **(1) "recommend restaurants in CA with environment + spicy Chinese"** => show top-k matches (embedding from business name/cat/review/photo) + parents/children => RAG可拿**review**里描述的东西, **photo**里`s caption`, etc.  
   - **(2) "photo showing big group seating birthday"** => 可能embedding hits `photo.caption` or `review.text` mention "birthday" => graph expansions.  
   - **(3) multi-hop BFS** => 先embedding => BFS 2-hop => return business + reviews + maybe user or photo.

4. **如何对接 RAG Agent**：  
   - 这个脚本不会直接回答**自然语言问题**。它只提供**(相似节点 + 关联节点)**；  
   - RAG Agent可把**(embedding match + parents children)**拼到**LLM**上下文 => LLM回答“Which place best fits user’s request?”  
   - 亦可让 LLM**多次**调用**`multi_hop_search`**或**`query_business_in_city`**进行思考链检索。

---

## 更多查询例子

- **"Find a place in CA that has good cocktails, with a big group seating environment, at least 4 star average, with reviews that mention 'birthday party'."**  
  - 先**symbol**: city=CA, stars>=4 => get candidate business => embedding check "cocktails big group" => find photo or review => BFS to see if "birthday party" mention => Done.  
- **"Check a user's reviews to see if they mention 'spicy' or 'Sichuan' flavor, gather those business"?** => (embedding "Sichuan" => reviews => business => maybe city=??).  
- **"Look up photo that has 'fried noodles' in the caption, then get that business reviews"?** => embed "fried noodles" => find photo => BFS: photo->business->reviews => gather.

这样就能充分展示**semantic(embedding)** + **structural(关系)**的查找组合，让**LLM-based** Agent能**拿到更多上下文**。

---

### 结语

以上便是一个**Yelp 数据结构**下**`SemanticMap + SemanticSimpleGraph`**的**原型示例**：我们提供了**多种 Query 函数**，帮助**RAG** or **LLM**获得**更多信息**（相似节点 + 父节点 + 多跳关联）以放进**对话上下文**。真正落地时，需要**批量插入** Yelp 的 business、review、photo、user 节点及 embedding，并可能在**LLM**层做**最后回答**。这样就能在**“Recommend some restaurants in California with good environment and spicy Chinese food.”**这样的请求下，不只返回embedding top-k，还能**深度**收集更多相关**评论**、**照片**、**metadata** 等一次性提供给**RAG Agent**。祝你在项目中一切顺利!