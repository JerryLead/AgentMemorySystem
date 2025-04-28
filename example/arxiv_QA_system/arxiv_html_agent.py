import sys
import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # 上一级目录
sys.path.append(project_root)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime
from semantic_data_structure.semantic_simple_graph import SemanticSimpleGraph  
from semantic_data_structure.semantic_graph import SemanticMap
import time



class ArxivSemanticGraph:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        self.semantic_graph = SemanticSimpleGraph(SemanticMap(text_embedding_model, embedding_dim))
        self.preferences = {"liked": [], "disliked": []}

    def _get_text_embedding(self, text):
        return self.semantic_graph.semantic_map.text_encoder.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, authors, abstract, chapters, references):
        # 将 title 和 authors 作为一个根节点插入
        root_key = f"{paper_id}_title_authors"
        root_value = {
            "title": title,
            "authors": authors
        }
        print(root_value)
        self.semantic_graph.add_node(root_key, root_value, text_for_embedding=f"{title} {' '.join(authors)}")

        # 插入其他元数据节点
        abstract_key = f"{paper_id}_abstract"
        self.semantic_graph.add_node(abstract_key, abstract, parent_keys=[root_key], parent_relation="has_abstract", text_for_embedding=abstract)
        print(abstract)
        for idx, chapter in enumerate(chapters):
            chapter_key = f"{paper_id}_chapter_{idx}"
            self.semantic_graph.add_node(chapter_key, chapter.get('title'), parent_keys=[root_key], parent_relation="has_chapter", text_for_embedding=chapter.get('title'))
            print(f"{idx}+{chapter.get('title')}")
            # 插入章节中的段落作为章节的子节点
            for para_idx, para in enumerate(chapter.get('paragraphs', [])):
                para_key = f"{chapter_key}_paragraph_{para_idx}"
                self.semantic_graph.add_node(para_key, para, parent_keys=[chapter_key], parent_relation="has_paragraph", text_for_embedding=para)
                print(f"{para_idx}+{para}")
            # 插入章节中的图片作为章节的子节点
            for photo_idx, photo in enumerate(chapter.get('images', [])):
                photo_key = f"{chapter_key}_photo_{photo_idx}"
                self.semantic_graph.add_node(photo_key, photo, parent_keys=[chapter_key], parent_relation="has_photo", text_for_embedding=photo)
                print(f"{photo_idx}")

        for ref_idx, ref in enumerate(references):
            ref_key = f"{paper_id}_reference_{ref_idx}"
            self.semantic_graph.add_node(ref_key, ref, parent_keys=[root_key], parent_relation="cite", text_for_embedding=ref)
            print(f"{ref_idx}+{ref}")


    def query(self, query_text, k=5):
        """
        返回查询节点及其根节点
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, k)
        result_nodes = []
        for node in similar_nodes:
            key = node["key"]
            if key.endswith("_title_authors"):
                result_nodes.append(node)
            else:
                current_key = key
                while "parents" in self.semantic_graph.graph_relations[current_key] and self.semantic_graph.graph_relations[current_key]["parents"]:
                    current_key = list(self.semantic_graph.graph_relations[current_key]["parents"].keys())[0]
                root_node = self._get_root_node(current_key)
                result_nodes.append(node)
                result_nodes.append(root_node)
        return result_nodes

    def _get_root_node(self, key):
        root_value = self.semantic_graph.semantic_map.data[self.semantic_graph.semantic_map.data.index(next((item for item in self.semantic_graph.semantic_map.data if item[0] == key), None))][1]
        return {
            "key": key,
            "value": root_value
        }

    def delete(self, query_text):
        """
         如果查询结果是根节点,则删除,否则不删除
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, 1)
        if not similar_nodes:
            return None
        key = similar_nodes[0]["key"]
        if key.endswith("_title_authors"):
            try:
                item_to_delete = next((item for item in self.semantic_graph.semantic_map.data if item[0] == key), None)
                if item_to_delete:
                    self.preferences["disliked"].append(item_to_delete)
                    self.semantic_graph.delete_node(key)
                    return similar_nodes[0]
            except ValueError:
                pass
        return None

    def mark_as_liked(self, root_key):
        """
        只能标记根节点
        """
        for item in self.semantic_graph.semantic_map.data:
            if item[0] == root_key:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        query_results = self.query(query_text, len(self.semantic_graph.semantic_map.data))
        root_nodes = set()
        for node in query_results:
            if node["key"].endswith("_title_authors"):
                root_nodes.add(node["key"])

        liked_nodes = []
        other_nodes = []
        for root_key in root_nodes:
            if root_key in [liked[0] for liked in self.preferences["liked"]]:
                liked_node = next((item for item in query_results if item["key"] == root_key), None)
                if liked_node:
                    liked_nodes.append(liked_node)
            elif root_key not in [disliked[0] for disliked in self.preferences["disliked"]]:
                other_node = next((item for item in query_results if item["key"] == root_key), None)
                if other_node:
                    other_nodes.append(other_node)

        recommended_nodes = liked_nodes + other_nodes
        return recommended_nodes[:k]


    def list_all(self):
        """
        列举所有根节点
        """
        all_nodes = self.semantic_graph.semantic_map.data
        root_nodes = []
        for item in all_nodes:
            key = item[0]
            if key.endswith("_title_authors"):
                root_nodes.append({
                    "key": key,
                    "value": item[1]
                })
        return root_nodes
    

import requests
from bs4 import BeautifulSoup
import os
import arxiv


def fetch_arxiv_html(query, max_results=5, output_dir='arxiv_html'):
    """
    在 Arxiv 数据集上根据查询词爬取相关论文，并以 HTML 格式保存。

    参数:
    query -- 搜索查询词
    num_results -- 要获取的论文数量，默认为 5
    output_dir -- 保存 HTML 文件的目录，默认为 'arxiv_html'

    返回:
    保存的 HTML 文件的路径列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    html_paths = []
    for result in client.results(search):
        html_url = result.entry_id.replace("abs", "html")
        file_name = os.path.join(output_dir, result.entry_id.split('/')[-1] + '.html')
        response = requests.get(html_url)
        if response.status_code == 200:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(response.text)
            html_paths.append(file_name)
        else:
            print(f"Failed to fetch HTML for {result.entry_id}. Status code: {response.status_code}")
    return html_paths



def decode_html(html_file_path, image_dir='arxiv_images'):
    """
    解析 HTML 文件，提取论文的元数据，并下载图片。
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取标题
    title = soup.find('title').text

    # 提取作者
    authors = []
    author_elements = soup.select('.ltx_creator.ltx_role_author.ltx_personname')
    for author_element in author_elements:
        authors.append(author_element.text.strip())

    # 提取摘要
    abstract = soup.find('div', class_='ltx_abstract').find('p').text

    # 提取章节信息
    chapters = []
    toc_entries = soup.select('.ltx_tocentry')
    # 获取文件名
    file_name_without_ext = os.path.splitext(os.path.basename(html_file_path))[0]
    base_url = f"https://arxiv.org/html/{file_name_without_ext}/"
    for entry in toc_entries:
        chapter_title = entry.find('span', class_='ltx_text ltx_ref_title').text
        chapter_href = entry.find('a')['href']
        chapter_id = chapter_href.split('#')[-1]
        chapter_section = soup.find('section', id=chapter_id)
        paragraphs = []
        images = []
        if chapter_section:
            # 提取段落信息
            paragraph_elements = chapter_section.find_all('p')
            for paragraph in paragraph_elements:
                paragraphs.append(paragraph.text)
            # 提取图片信息
            image_elements = chapter_section.find_all('img')
            for image in image_elements:
                if 'src' in image.attrs:
                    image_url = base_url + image['src']
                    # 下载图片
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        # 存储图片的二进制数据
                        images.append(response.content)
                    else:
                        print(f"Failed to download image from {image_url}. Status code: {response.status_code}")

        chapters.append({
            'title': chapter_title,
            'paragraphs': paragraphs,
            'images': images
        })


    # 提取参考文献
    references = []
    biblist = soup.find('ul', class_='ltx_biblist')
    if biblist:
        bibitems = biblist.find_all('li', class_='ltx_bibitem')
        for bibitem in bibitems:
            references.append(bibitem.text.strip())

    return {
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'chapters': chapters,
        'references': references
    }


# #decode测试
# if __name__ == "__main__":
#     html_file_path = "arxiv_html/2411.12761v1.html"
#     arxiv_graph = decode_html(html_file_path)
#     print(arxiv_graph)


def parse_arxiv(query, max_results=5, output_dir='arxiv_html'):
    """
    解析 arXiv 论文的方法，先获取 HTML 文件，再解析文件，最后使用 ArxivSemanticGraph 实例表示信息。
    """
    html_paths = fetch_arxiv_html(query, max_results, output_dir)
    arxiv_graphs = []

    for html_path in html_paths:
        decoded_info = decode_html(html_path)
        paper_id = os.path.basename(html_path).replace('.html', '')

        # 创建 ArxivSemanticGraph 实例
        graph = ArxivSemanticGraph()
        graph.insert(
            paper_id,
            decoded_info['title'],
            decoded_info['authors'],
            decoded_info['abstract'],
            decoded_info['chapters'],
            decoded_info['references']
        )
        arxiv_graphs.append(graph)

    return arxiv_graphs



def recommend_papers(query):
    """
    根据用户输入的查询词，定期从 arXiv 上更新论文并推荐给用户
    """
    while True:
        print(f"Updating papers for query: {query}")
        arxiv_graphs = parse_arxiv(query,1)
        print("Sleeping for 1 hour...")
        time.sleep(3600)  # 暂停 1 小时


def main():
    # 用户输入查询词
    query = input("Please enter your query: ")
    recommend_papers(query)


if __name__ == "__main__":
    main()