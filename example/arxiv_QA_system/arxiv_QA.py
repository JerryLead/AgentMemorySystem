from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime
from semantic_data_structuresemantic_simple_graph import SemanticSimpleGraph  
from semantic_data_structure.semantic_map import SemanticMap
import time


class ArxivSemanticGraph:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        self.semantic_graph = SemanticSimpleGraph(SemanticMap(text_embedding_model, embedding_dim))
        self.preferences = {"liked": [], "disliked": []}

    def _get_text_embedding(self, text):
        return self.semantic_graph.semantic_map.text_encoder.encode([text], convert_to_numpy=True)


    def insert(self, paper_id, title, authors, abstract, chapters, references):
        root_key = f"{paper_id}_title_authors"
        root_value = {
            "title": title,
            "authors": authors
        }
        # 检查是否存在相同 value 的根节点
        existing_root = next((item for item in self.semantic_graph.semantic_map.data if item[1] == root_value), None)
        if existing_root is None:
            self.semantic_graph.add_node(root_key, root_value, text_for_embedding=f"{title} {' '.join(authors)}")
        else:
            print(f"根节点 {root_key} 的内容已存在，跳过插入。")
        print(root_value)
        abstract_key = f"{paper_id}_abstract"
        # 检查是否存在相同 value 的摘要节点
        existing_abstract = next((item for item in self.semantic_graph.semantic_map.data if item[1] == abstract), None)
        if existing_abstract is None:
            self.semantic_graph.add_node(abstract_key, abstract, parent_keys=[root_key], parent_relation="has_abstract", text_for_embedding=abstract)
        else:
            print(f"摘要节点 {abstract_key} 的内容已存在，跳过插入。")
        print(abstract)
        for idx, chapter in enumerate(chapters):
            chapter_title = chapter.get('title')
            chapter_key = f"{paper_id}_chapter_{idx}"
            # 检查是否存在相同 value 的章节节点
            existing_chapter = next((item for item in self.semantic_graph.semantic_map.data if item[1] == chapter_title), None)
            if existing_chapter is None:
                self.semantic_graph.add_node(chapter_key, chapter_title, parent_keys=[root_key], parent_relation="has_chapter", text_for_embedding=chapter_title)
                for para_idx, para in enumerate(chapter.get('paragraphs', [])):
                    para_key = f"{chapter_key}_paragraph_{para_idx}"
                    # 检查是否存在相同 value 的段落节点
                    existing_para = next((item for item in self.semantic_graph.semantic_map.data if item[1] == para), None)
                    if existing_para is None:
                        self.semantic_graph.add_node(para_key, para, parent_keys=[chapter_key], parent_relation="has_paragraph", text_for_embedding=para)
                for photo_idx, photo in enumerate(chapter.get('images', [])):
                    photo_key = f"{chapter_key}_photo_{photo_idx}"
                    # 检查是否存在相同 value 的图片节点
                    existing_photo = next((item for item in self.semantic_graph.semantic_map.data if item[1] == photo), None)
                    if existing_photo is None:
                        self.semantic_graph.add_node(photo_key, photo, parent_keys=[chapter_key], parent_relation="has_photo", text_for_embedding=photo)
            else:
                print(f"章节节点 {chapter_key} 的内容已存在，跳过插入。")

        for ref_idx, ref in enumerate(references):
            ref_key = f"{paper_id}_reference_{ref_idx}"
            # 检查是否存在相同 value 的参考文献节点
            existing_ref = next((item for item in self.semantic_graph.semantic_map.data if item[1] == ref), None)
            if existing_ref is None:
                self.semantic_graph.add_node(ref_key, ref, parent_keys=[root_key], parent_relation="cite", text_for_embedding=ref)
                similar_nodes = self.semantic_graph.retrieve_similar_nodes(ref, k=1)
                if similar_nodes:
                    similar_key = similar_nodes[0]["key"]
                    if similar_key.endswith("_title_authors"):
                        self.semantic_graph.insert_edge(ref_key, similar_key, relation="point")
            else:
                print(f"参考文献节点 {ref_key} 的内容已存在，跳过插入。")

        self.semantic_graph.build_index()

    def query(self, query_text, k=5):
        """
        返回查询节点及其根节点，返回 k 个不同的结果
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, len(self.semantic_graph.semantic_map.data))
        result_nodes = []
        result_keys = set()  

        for node in similar_nodes:
            key = node["key"]
            if key.endswith("_title_authors"):
                if key not in result_keys:
                    result_nodes.append(node)
                    result_keys.add(key)
            else:
                current_key = key
                while "parents" in self.semantic_graph.graph_relations[current_key] and self.semantic_graph.graph_relations[current_key]["parents"]:
                    current_key = list(self.semantic_graph.graph_relations[current_key]["parents"].keys())[0]
                root_node = self._get_root_node(current_key)
                root_key = root_node["key"]
                if root_key not in result_keys:
                    result_nodes.append(root_node)
                    result_keys.add(root_key)
                if key not in result_keys:
                    result_nodes.append(node)
                    result_keys.add(key)

            if len(result_nodes) >= k:
                break

        return result_nodes[:k]

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
        self.semantic_graph.build_index()
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
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    title = soup.find('title').text

    authors = []
    author_elements = soup.select('.ltx_personname')
    for author_element in author_elements:
        author = author_element.text.strip()
        if author not in authors:
            authors.append(author)

    abstract = soup.find('div', class_='ltx_abstract').find('p').text

    chapters = []
    toc_entries = soup.select('.ltx_tocentry')
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
            paragraph_elements = chapter_section.find_all('p')
            for paragraph in paragraph_elements:
                para_text = paragraph.text
                if para_text not in paragraphs:
                    paragraphs.append(para_text)
            image_elements = chapter_section.find_all('img')
            for image in image_elements:
                if 'src' in image.attrs:
                    image_url = base_url + image['src']
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        images.append(response.content)
                    else:
                        print(f"Failed to download image from {image_url}. Status code: {response.status_code}")

        chapter = {
            'title': chapter_title,
            'paragraphs': paragraphs,
            'images': images
        }
        if chapter not in chapters:
            chapters.append(chapter)

    references = []
    biblist = soup.find('ul', class_='ltx_biblist')
    if biblist:
        bibitems = biblist.find_all('li', class_='ltx_bibitem')
        for bibitem in bibitems:
            ref = bibitem.text.strip()
            if ref not in references:
                references.append(ref)

    return {
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'chapters': chapters,
        'references': references
    }



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


def parse_local_paper(html_dir='arxiv_html'):
    graph = ArxivSemanticGraph()

    if not os.path.exists(html_dir):
        print(f"指定的目录 {html_dir} 不存在。")
        return graph

    for filename in os.listdir(html_dir):
        if filename.endswith('.html'):
            html_file_path = os.path.join(html_dir, filename)
            decoded_info = decode_html(html_file_path)
            if decoded_info:
                paper_id = os.path.basename(html_file_path).replace('.html', '')
                try:
                    graph.insert(
                        paper_id,
                        decoded_info['title'],
                        decoded_info['authors'],
                        decoded_info['abstract'],
                        decoded_info['chapters'],
                        decoded_info['references']
                    )
                except Exception as e:
                    print(f"插入 {paper_id} 的数据到图中时出现错误: {e}")

    return graph


from openai import OpenAI


class GenerativeArxivSemanticGraph(ArxivSemanticGraph):
    def __init__(self, arxiv_graph_instance=None, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        if arxiv_graph_instance:
            self.semantic_graph = arxiv_graph_instance.semantic_graph
            self.preferences = arxiv_graph_instance.preferences
        else:
            super().__init__(text_embedding_model, embedding_dim)
        self.client = OpenAI(api_key="sk-c4e7f78ec6fe48379839541971a2bfc7", base_url="https://api.deepseek.com")

    def generate_answer(self, question, k=5):
        
        results = self.query(question, k)
        if not results:
            return "未找到相关信息。"

        
        relevant_info = []
        for result in results:
            value = result["value"]
            if isinstance(value, dict):
                for v in value.values():
                    relevant_info.append(str(v))
            else:
                relevant_info.append(str(value))

       
        graph_structure_info = self.get_graph_structure_info(results)

        
        context = " ".join(relevant_info)
        prompt = f"问题: {question}\n相关信息: {context}\n图结构信息: {graph_structure_info}\n回答:"

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use the provided graph structure and relevant information to answer the question accurately."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"请求失败，错误信息: {str(e)}"

    def get_graph_structure_info(self, nodes):
        structure_info = []
        for node in nodes:
            key = node["key"]
            parents = self.semantic_graph.graph_relations[key].get("parents", {})
            children = self.semantic_graph.graph_relations[key].get("children", {})
            links = self.semantic_graph.graph_relations[key].get("links", {})

            parent_info = [f"{key} 的父节点为 {p}，关系为 {r}" for p, r in parents.items()]
            child_info = [f"{key} 的子节点为 {c}，关系为 {r}" for c, r in children.items()]
            link_info = [f"{key} 与 {l} 有 {r} 关系" for l, r in links.items()]

            structure_info.extend(parent_info)
            structure_info.extend(child_info)
            structure_info.extend(link_info)

        return "\n".join(structure_info)


if __name__ == "__main__":
    
    arxiv_graph = parse_local_paper()
    generative_graph = GenerativeArxivSemanticGraph(arxiv_graph_instance=arxiv_graph)

    while True:
        question = input("请输入查询内容（输入 end 结束）：")
        if question.lower() == 'end':
            break
        answer = generative_graph.generate_answer(question)
        print(answer)


