# 使用SimpleGraph保存paper:
# 将paper的元数据分开保存在SemanticMap中,其中title&timestamp作为paper的根节点
# 实现query(根据query_text查询相关的5篇论文),delete(根据query_text删除最相关的一篇Paper),marked_as_liked(将某一根节点标记为liked),
# recommend(根据query_text推荐相关的5篇Paper,如果被标记为disliked,则不推荐),list_all(列出所有根节点)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime
from semantic_data_structure.semantic_simple_graph import SemanticSimpleGraph  
from semantic_data_structure.semantic_map import SemanticMap
import time


class ArxivSemanticGraph:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        self.semantic_graph = SemanticSimpleGraph(SemanticMap(text_embedding_model, embedding_dim))
        self.preferences = {"liked": [], "disliked": []}

    def _get_text_embedding(self, text):
        return self.semantic_graph.semantic_map.text_encoder.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, abstract, authors, categories, timestamp):
        # 插入标题&timestamp作为根节点
        root_key = f"{paper_id}_root"
        root_value = {
            "title": title,
            "timestamp": timestamp
        }
        self.semantic_graph.add_node(root_key, root_value, text_for_embedding=title)

        # 插入摘要节点
        abstract_key = f"{paper_id}_abstract"
        self.semantic_graph.add_node(abstract_key, abstract, text_for_embedding=abstract)
        self.semantic_graph.insert_edge(root_key, abstract_key, relation="has_abstract")

        # 插入作者节点
        for index, author in enumerate(authors):
            author_key = f"{paper_id}_author_{index}"
            self.semantic_graph.add_node(author_key, author, text_for_embedding=author)
            self.semantic_graph.insert_edge(root_key, author_key, relation="has_author")

        # 插入类别节点
        category_key = f"{paper_id}_category"
        self.semantic_graph.add_node(category_key, categories, text_for_embedding=categories)
        self.semantic_graph.insert_edge(root_key, category_key, relation="has_category")

    def query(self, query_text, k=5):
        """
        根据查询文本，检索与之相关的 k 个节点。

        此方法首先通过语义图的相似节点检索方法获取与查询文本相似的节点，
        然后通过遍历这些相似节点，向上追溯到根节点，收集所有相关的节点信息。

        :param query_text: 查询文本。
        :param k: 需要返回的相似节点数量，默认为 5。
        :return: 包含相关节点信息的列表，每个元素是一个字典，包含 "key" 和 "value" 等信息。
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, k)
        all_related_nodes = []
        for node in similar_nodes:
            key = node["key"]
            related_nodes = [key]
            current_key = key
            while self.semantic_graph.graph_relations[current_key]["parents"]:
                current_key = list(self.semantic_graph.graph_relations[current_key]["parents"].keys())[0]
                related_nodes.append(current_key)
                if self.semantic_graph.graph_relations[current_key]["parents"] == {}:
                    break
            all_related_nodes.extend(related_nodes)
        unique_nodes = []
        for node in all_related_nodes:
            if node not in unique_nodes:
                unique_nodes.append(node)
        result_nodes = []
        for node in unique_nodes:
            value = self.semantic_graph.semantic_map.data[self.semantic_graph.semantic_map.data.index(next((item for item in self.semantic_graph.semantic_map.data if item[0] == node), None))][1]
            result_nodes.append({"key": node, "value": value})
        return result_nodes

    def delete(self, query_text):
        """
        根据查询文本删除相关的论文信息。

        此方法首先检索与查询文本相似的所有节点，然后找到距离小于阈值（0.7）的节点。
        对于找到的节点，向上追溯到根节点，将根节点标记为不喜欢，并删除根节点及其所有子节点。

        :param query_text: 用于确定要删除内容的查询文本。
        :return: 被删除的根节点的数据，如果没有找到符合条件的节点则返回 None。
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, len(self.semantic_graph.semantic_map.data))
        distance_threshold = 0.7
        for node in similar_nodes:
            if node["distance"] < distance_threshold:
                current_key = node["key"]
                while self.semantic_graph.graph_relations[current_key]["parents"]:
                    current_key = list(self.semantic_graph.graph_relations[current_key]["parents"].keys())[0]
                root_key = current_key
                self.preferences["disliked"].append(self.semantic_graph.semantic_map.data[self.semantic_graph.semantic_map.data.index(next((item for item in self.semantic_graph.semantic_map.data if item[0] == root_key), None))])
                children_keys = list(self.semantic_graph.graph_relations[root_key]["children"].keys())
                for child_key in children_keys:
                    self.semantic_graph.delete_node(child_key)
                self.semantic_graph.delete_node(root_key)
                target_item = next((item for item in self.semantic_graph.semantic_map.data if item[0] == root_key), None)
                if target_item:
                    return target_item
                else:
                    return None
        return None
     

    def mark_as_liked(self, root_key):
        """
        将指定的根节点标记为喜欢。

        :param root_key: 要标记为喜欢的根节点的键。
        """
        for item in self.semantic_graph.semantic_map.data:
            if item[0] == root_key:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        """
        根据查询文本推荐相关的 k 篇论文。

        此方法首先检索与查询文本相似的节点，然后根据用户的喜欢和不喜欢偏好对这些节点进行排序。
        喜欢的节点优先推荐，不喜欢的节点被排除。

        :param query_text: 查询文本。
        :param k: 需要推荐的论文数量，默认为 5。
        :return: 推荐的论文节点信息列表，每个元素是一个字典，包含 "key" 和 "value" 等信息。
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, k)
        priority_recommendations = []
        for item in similar_nodes:
            if item not in self.preferences["disliked"]:
                if item in self.preferences["liked"]:  # Boost liked preferences
                    priority_recommendations.insert(0, item)
                else:
                    priority_recommendations.append(item)
        return priority_recommendations[:k]

    def list_all(self):
        """
        列出所有的根节点。

        此方法遍历语义图的所有数据，筛选出键以 "_root" 结尾的节点，这些节点即为根节点。

        :return: 包含所有根节点信息的列表，每个元素是一个字典，包含 "key" 和 "value" 等信息。
        """
        root_nodes = []
        for item in self.semantic_graph.semantic_map.data:
            if item[0].endswith("_root"):
                root_node = {
                    "key": item[0],
                    "value": item[1]
                }
                root_nodes.append(root_node)
        return root_nodes


# 示例使用 ArxivSemanticGraph
# if __name__ == "__main__":
#     arxiv_semantic_graph = ArxivSemanticGraph()

#     arxiv_semantic_graph.insert(
#         paper_id="paper1",
#         title="Title of Paper 1",
#         abstract="Abstract of Paper 1",
#         authors=["Author 1", "Author 2"],
#         categories="Category 1",
#         timestamp="2023-10-01"
#     )

#     arxiv_semantic_graph.semantic_graph.build_index()

#     query = "Abstract of a paper"
#     query_results = arxiv_semantic_graph.query(query, k=3)
#     print(f"Query results for '{query}':")
#     for result in query_results:
#         print(result)

#     delete_query = "Author 1"
#     deleted_paper = arxiv_semantic_graph.delete(delete_query)
#     print(f"Deleted paper for query '{delete_query}':")
#     if deleted_paper:
#         print(deleted_paper)

#     arxiv_semantic_graph.mark_as_liked("paper1_root")
#     recommend_query = "Paper related query"
#     recommended_papers = arxiv_semantic_graph.recommend(recommend_query, k=3)
#     print(f"Recommended papers for query '{recommend_query}':")
#     for paper in recommended_papers:
#         print(paper)

#     all_root_nodes = arxiv_semantic_graph.list_all()
#     print(f"All root nodes:")
#     for root_node in all_root_nodes:
#         print(root_node)



#实时从Arixv上爬取最新论文并推荐
class RealTimeArxivSemanticGraph(ArxivSemanticGraph):
    def update_from_arxiv(self, query="cs.AI", max_results=10):
        papers = fetch_arxiv_data(query, max_results)
        for paper in papers:
            self.insert(
                paper_id=paper["paper_id"],
                title=paper["title"],
                abstract=paper["abstract"],
                authors=paper["authors"],
                categories=paper["categories"],
                timestamp=paper["timestamp"]
            )
        self.semantic_graph.build_index()

    def enhanced_recommend(self, query_text, k=5):
        # 先更新数据
        self.update_from_arxiv()
        return super().recommend(query_text, k)


def fetch_arxiv_data(query="cs.AI", max_results=10):
    """
    Fetch arXiv papers based on query and max results.
    """
    url = f"http://export.arxiv.org/api/query?search_query=cat:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    feed = response.text
    return parse_arxiv_feed(feed)


def parse_arxiv_feed(feed):
    """
    Parse the arXiv RSS feed.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(feed)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        categories = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]
        timestamp = entry.find("{http://www.w3.org/2005/Atom}published").text
        papers.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "timestamp": timestamp,
        })
    return papers


def test_query_method():
    real_time_graph = RealTimeArxivSemanticGraph()
    real_time_graph.update_from_arxiv()
    query_text = "Personal Assistant"
    query_results = real_time_graph.query(query_text, k=3)
    print(f"Query results for '{query_text}':")
    for result in query_results:
        print(result)



def test_mark_as_liked_method():
    real_time_graph = RealTimeArxivSemanticGraph()
    real_time_graph.update_from_arxiv()
    all_nodes = real_time_graph.list_all()
    if all_nodes:
        root_key = all_nodes[0]["key"]
        real_time_graph.mark_as_liked(root_key)
        print(f"Marked {root_key} as liked.")
    else:
        print("No nodes available to mark as liked.")


def test_list_all_method():
    real_time_graph = RealTimeArxivSemanticGraph()
    real_time_graph.update_from_arxiv()
    all_root_nodes = real_time_graph.list_all()
    print("All root nodes:")
    for root_node in all_root_nodes:
        print(root_node)


if __name__ == "__main__":

    test_query_method()
    test_mark_as_liked_method()
    test_list_all_method()

#每60s推荐3个paper
if __name__ == "__main__":
    real_time_graph = RealTimeArxivSemanticGraph()
    query = "AI research"
    while True:
        recommended_papers = real_time_graph.enhanced_recommend(query, k=3)
        print(f"Recommended papers for '{query}' at {time.ctime()}:")
        for paper in recommended_papers:
            print(paper)
        time.sleep(60)


