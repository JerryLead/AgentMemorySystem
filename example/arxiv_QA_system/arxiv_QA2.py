from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime
from semantic_simple_graph import SemanticSimpleGraph  
from semantic_graph import SemanticMap
import time


class ArxivSemanticGraph:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        self.semantic_graph = SemanticSimpleGraph(SemanticMap(text_embedding_model, embedding_dim))
        self.preferences = {"liked": [], "disliked": []}

    def _get_text_embedding(self, text):
        return self.semantic_graph.semantic_map.text_encoder.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, authors, abstract, chapters, references, tables=None):
        root_key = f"{paper_id}_title_authors"
        root_value = {
            "title": title,
            "authors": authors
        }
        existing_root = next((item for item in self.semantic_graph.semantic_map.data if item[1] == root_value), None)
        if existing_root is None:
            self.semantic_graph.add_node(root_key, root_value, text_for_embedding=f"{title} {' '.join(authors)}")
        else:
            print(f"根节点 {root_key} 的内容已存在，跳过插入。")
        # print(root_value)
        abstract_key = f"{paper_id}_abstract"
        existing_abstract = next((item for item in self.semantic_graph.semantic_map.data if item[1] == abstract), None)
        if existing_abstract is None:
            self.semantic_graph.add_node(abstract_key, abstract, parent_keys=[root_key],
                                         parent_relation="has_abstract", text_for_embedding=abstract)
        else:
            print(f"摘要节点 {abstract_key} 的内容已存在，跳过插入。")
        # print(abstract)
        for idx, chapter in enumerate(chapters):
            chapter_title = chapter.get('title')
            chapter_key = f"{paper_id}_chapter_{idx}"
            # print(f"chapter_key: {chapter_key}, type: {type(chapter_key)}")  # 添加调试信息

            existing_chapter = next((item for item in self.semantic_graph.semantic_map.data if item[1] == chapter_title),
                                    None)
            if existing_chapter is None:
                self.semantic_graph.add_node(chapter_key, chapter_title, parent_keys=[root_key],
                                             parent_relation="has_chapter", text_for_embedding=chapter_title)
                for para_idx, para in enumerate(chapter.get('paragraphs', [])):
                    para_key = f"{chapter_key}_paragraph_{para_idx}"
                    # print(f"para_key: {para_key}, type: {type(para_key)}")  # 添加调试信息

                    existing_para = next((item for item in self.semantic_graph.semantic_map.data if item[1] == para),
                                         None)
                    if existing_para is None:
                        self.semantic_graph.add_node(para_key, para, parent_keys=[chapter_key],
                                                     parent_relation="has_paragraph", text_for_embedding=para)
                for photo_idx, photo in enumerate(chapter.get('images', [])):
                    photo_key = f"{chapter_key}_photo_{photo_idx}"
                    # print(f"photo_key: {photo_key}, type: {type(photo_key)}")  # 添加调试信息

                    existing_photo = next((item for item in self.semantic_graph.semantic_map.data if item[1] == photo),
                                          None)
                    if existing_photo is None:
                        self.semantic_graph.add_node(photo_key, photo, parent_keys=[chapter_key],
                                                     parent_relation="has_photo", text_for_embedding=photo)
            else:
                print(f"章节节点 {chapter_key} 的内容已存在，跳过插入。")

        if tables:
            for table_idx, table in enumerate(tables):
                for row_idx, row in enumerate(table):
                    row_text = " ".join([f"{key}: {value}" for key, value in row.items()])
                    table_row_key = f"{paper_id}_table_{table_idx}_row_{row_idx}"
                    # print(f"table_row_key: {table_row_key}, type: {type(table_row_key)}")  # 添加调试信息

                    existing_table_row = next(
                        (item for item in self.semantic_graph.semantic_map.data if item[1] == row), None)
                    if existing_table_row is None:
                        self.semantic_graph.add_node(table_row_key, row, parent_keys=[root_key],
                                                     parent_relation="has_table_row", text_for_embedding=row_text)
                    else:
                        print(f"表格行节点 {table_row_key} 的内容已存在，跳过插入。")

        self.semantic_graph.build_index()

    def query(self, query_text, k=5):
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text,
                                                                   len(self.semantic_graph.semantic_map.data))
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
                while "parents" in self.semantic_graph.graph_relations[current_key] and \
                        self.semantic_graph.graph_relations[current_key]["parents"]:
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
        root_value = self.semantic_graph.semantic_map.data[
            self.semantic_graph.semantic_map.data.index(
                next((item for item in self.semantic_graph.semantic_map.data if item[0] == key), None))][1]
        return {
            "key": key,
            "value": root_value
        }

    def delete(self, query_text):
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

    # 解析表格数据
    tables = []
    table_elements = soup.find_all('table', class_='ltx_tabular')
    for table_element in table_elements:
        table = []
        # 检查 <thead> 标签是否存在
        thead = table_element.find('thead')
        if thead:
            header_row = thead.find('tr')
            headers = []
            for th in header_row.find_all('th'):
                header_text = ''.join(th.stripped_strings)
                headers.append(header_text)
        else:
            # 如果没有 <thead>，可以考虑取第一行作为表头
            first_row = table_element.find('tr')
            if first_row:
                headers = []
                for th in first_row.find_all(['th', 'td']):
                    header_text = ''.join(th.stripped_strings)
                    headers.append(header_text)
            else:
                headers = []

        body_rows = table_element.find_all('tr')[1:] if headers else table_element.find_all('tr')
        for row_element in body_rows:
            row = {}
            cells = row_element.find_all('td')
            for i, cell in enumerate(cells):
                cell_text = ''.join(cell.stripped_strings)
                if i < len(headers):
                    row[headers[i]] = cell_text
            if row:
                table.append(row)
        if table:
            tables.append(table)

    return {
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'chapters': chapters,
        'references': references,
        'tables': tables
    }


def parse_arxiv(query, max_results=5, output_dir='arxiv_html'):
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
            decoded_info['references'],
            decoded_info['tables']
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
                        decoded_info['references'],
                        decoded_info['tables']
                    )
                except Exception as e:
                    print(f"插入 {paper_id} 的数据到图中时出现错误: {e}")

    return graph

# def test_load_local_paper_to_graph():
#     try:
#         # 调用 parse_local_paper 方法将本地论文加载成图
#         graph = parse_local_paper()

#         # 检查返回的对象是否为 ArxivSemanticGraph 类的实例
#         if not isinstance(graph, ArxivSemanticGraph):
#             print("加载失败：返回对象不是 ArxivSemanticGraph 类的实例。")
#             return

#         # 检查图中是否包含节点数据（这里简单通过检查 semantic_map 的数据列表长度）
#         node_count = len(graph.semantic_graph.semantic_map.data)
#         if node_count == 0:
#             print("加载失败：图中未包含任何节点数据。")
#             return

#         print("加载成功：本地论文已成功加载成 ArxivSemanticGraph 类实例。")
#     except Exception as e:
#         print(f"加载失败：出现异常 {e}")


# if __name__ == "__main__":
#     test_load_local_paper_to_graph()

from openai import OpenAI
import hashlib
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import json


nltk.download('stopwords')


class ArxivAgent:
    def __init__(self, arxiv_graph, api_key, base_url="https://api.deepseek.com"):
        self.arxiv_graph = arxiv_graph
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.query_cache = {}
        self.stop_words = set(stopwords.words('english'))
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess_question(self, question):
        question = question.lower()
        question = re.sub(r'[^\w\s]', '', question)
        words = question.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)

    def analyze_question_with_llm(self, question):
        try:
            prompt = f"请对以下问题进行语义理解和转换，输出最能准确表达问题核心的表述：{question}"
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的问题分析助手，能精准理解问题语义并转换为合适的表述。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM分析问题失败，错误信息: {e}")
            return question

    # def evaluate_similarity(self, result, question):
    #     result_text = str(result["value"]) if isinstance(result["value"], dict) else result["value"]
    #     result_words = set(result_text.lower().split())
    #     question_words = set(question.lower().split())
    #     overlap = len(result_words.intersection(question_words))
    #     similarity = overlap / (len(result_words) + len(question_words) - overlap)
    #     return similarity

    def postprocess_results(self, results, question, similarity_threshold=0.05):
        filtered_results = []
        for result in results:
            similarity = self.evaluate_similarity(result, question)
            if similarity >= similarity_threshold:
                filtered_results.append(result)
        return filtered_results

    def extract_answer(self, results, question):
        print(f"步骤的输出结果: {results}")
        preprocessed_question = self.preprocess_question(question)

        if "author" in preprocessed_question or "authors" in preprocessed_question:
            for result in results:
                value = result["value"]
                if isinstance(value, list):
                    return ", ".join(value)
                elif isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "authors" in value:
                    return ", ".join(value["authors"])
            return "未找到相关作者信息。"
        elif "title" in preprocessed_question:
            for result in results:
                value = result["value"]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "title" in value:
                    return value["title"]
            return "未找到相关标题信息。"
        elif "abstract" in preprocessed_question:
            for result in results:
                value = result["value"]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "abstract" in value:
                    return value["abstract"]
            return "未找到相关摘要信息。"
        elif "reference" in preprocessed_question or "references" in preprocessed_question:
            reference_list = []
            for result in results:
                value = result["value"]
                if isinstance(value, list):
                    reference_list.extend(value)
                elif isinstance(value, str):
                    reference_list.append(value)
                elif isinstance(value, dict) and "references" in value:
                    reference_list.extend(value["references"])
            if reference_list:
                return "\n".join(reference_list)
            return "未找到相关参考文献信息。"

        # 若未匹配到特定关键词，进行通用结果总结
        summary = []
        for result in results:
            value = result["value"]
            if isinstance(value, dict):
                for v in value.values():
                    summary.append(str(v))
            else:
                summary.append(str(value))
        if summary:
            return "\n".join(summary)
        return "未找到相关信息。"

    def generate_prompt(self, query):
        arxiv_graph_structure = """
ArxivGraph 的结构如下：
- 根节点：以 "{paper_id}_title_authors" 命名，包含论文的标题（title）和作者（authors）信息。
- 摘要节点：以 "{paper_id}_abstract" 命名，包含论文的摘要（abstract），与根节点通过 "has_abstract" 关系相连。
- 章节节点：以 "{paper_id}_chapter_{idx}" 命名，包含章节标题（title），与根节点通过 "has_chapter" 关系相连。每个章节节点下可能包含段落（paragraphs）和图片（images）子节点。
- 表格行节点：以 "{paper_id}_table_{table_idx}_row_{row_idx}" 命名，包含表格行的数据，与根节点通过 "has_table_row" 关系相连。

ArxivGraph 可以有多个实例，每个实例代表一篇独立的 arXiv 论文，通过不同的 paper_id 进行区分。
"""

        paper_examples = """
以下是几个 paper 的样例：
- paper: {
    "paper_id": "2304.01234",
    "title": "Advances in Machine Learning Techniques",
    "authors": ["Alice Smith", "Bob Johnson"],
    "abstract": "This paper explores the latest advancements in machine learning techniques and their applications.",
    "chapters": [
        {
            "title": "Introduction",
            "paragraphs": ["Machine learning has become an important field in recent years.", "It has various applications in different industries."],
            "images": ["intro_fig1.png"]
        },
        {
            "title": "Related Work",
            "paragraphs": ["Many researchers have contributed to the development of machine learning algorithms.", "This section reviews some of the key works."],
            "images": []
        },
        {
            "title": "Methodology",
            "paragraphs": ["We propose a novel algorithm in this section.", "The algorithm is based on deep learning principles."],
            "images": ["method_fig1.png"]
        }
    ],
    "references": ["Smith, A. (2022). Machine Learning Basics. Journal of AI Research.", "Johnson, B. (2023). Advanced Machine Learning Models. AI Today."],
    "tables": [
        {
            "Column1": "Model Type",
            "Column2": "Accuracy",
            "Column3": "Training Time"
        },
        {
            "Column1": "Neural Network",
            "Column2": "0.9",
            "Column3": "2 hours"
        },
        {
            "Column1": "Decision Tree",
            "Column2": "0.85",
            "Column3": "1 hour"
        }
    ]
}
- paper: {
    "paper_id": "2305.05678",
    "title": "Natural Language Processing in Healthcare",
    "authors": ["Charlie Brown", "David Davis"],
    "abstract": "This research focuses on the application of natural language processing in the healthcare industry.",
    "chapters": [
        {
            "title": "Overview",
            "paragraphs": ["Natural language processing can be used to analyze medical records.", "It has the potential to improve patient care."],
            "images": []
        },
        {
            "title": "Case Studies",
            "paragraphs": ["Several case studies are presented to demonstrate the effectiveness of NLP in healthcare.", "These studies show promising results."],
            "images": ["case_study_fig1.png"]
        },
        {
            "title": "Future Directions",
            "paragraphs": ["There are still many challenges in applying NLP in healthcare.", "Future research should focus on improving accuracy and efficiency."],
            "images": []
        }
    ],
    "references": ["Brown, C. (2023). NLP for Medical Text Analysis. Healthcare Informatics Journal.", "Davis, D. (2023). NLP Applications in Patient Diagnosis. Medical AI Review."],
    "tables": [
        {
            "Column1": "Task",
            "Column2": "Performance",
            "Column3": "Complexity"
        },
        {
            "Column1": "Medical Text Classification",
            "Column2": "0.85",
            "Column3": "High"
        },
        {
            "Column1": "Named Entity Recognition",
            "Column2": "0.9",
            "Column3": "Medium"
        }
    ]
}
"""

        query_examples = """
以下是一些查询示例及对应的查询计划：
查询：查找所有关于机器学习的论文标题
查询计划：
[
    {
        "step": 1,
        "target": ["paper"],
        "constraints": {
            "semantic_query": "papers about machine learning",
            "filter": {}
        },
        "input": null
    },
    {
        "step": 2,
        "target": ["title"],
        "constraints": {
            "semantic_query": "",
            "filter": {}
        },
        "input": [1]
    }
]

查询：查找作者 Alice Smith 所写论文的摘要
查询计划：
[
    {
        "step": 1,
        "target": ["paper"],
        "constraints": {
            "semantic_query": "papers written by Alice Smith",
            "filter": {}
        },
        "input": null
    },
    {
        "step": 2,
        "target": ["abstract"],
        "constraints": {
            "semantic_query": "",
            "filter": {}
        },
        "input": [1]
    }
]
"""

        prompt = f"""
        {query}

        Please convert the above natural language query into a JSON format query plan. The query plan should follow this structure:
        [
            {{
                "step": 1,
                "target": ["paper", "abstract", ...],
                "constraints": {{
                    "semantic_query": "your semantic query here",
                    "filter": {{"attribute": "value"}}
                }},
                "input": null
            }},
            ...
        ]

        Use the following data structure information for reference:
        - paper: {{
            "paper_id": "xxxx.xxxxx",
            "title": "Paper Title",
            "authors": ["Author1", "Author2"],
            "abstract": "Abstract text",
            "chapters": [{{"title": "Chapter 1", "paragraphs": ["Para 1", "Para 2"]}}],
            "references": ["Reference 1", "Reference 2"],
            "tables": [{{"Column1": "Value1", "Column2": "Value2"}}]
        }}

        {arxiv_graph_structure}
        {paper_examples}
        {query_examples}
        """
        print(f"生成的提示信息: {prompt}")
        return prompt

    def query_llm_for_plan(self, query):
        prompt = self.generate_prompt(query)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert in generating query plans for arXiv dataset."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            result = response.choices[0].message.content
            print("LLM 返回的原始内容:", result)  
            structured_query = self.parse_query_plan(result)
            if structured_query:
                for step in structured_query:
                    semantic_query = step["constraints"].get("semantic_query", "")
                    print(f"步骤 {step['step']} 的语义查询字符串: {semantic_query}")
            return result
        except Exception as e:
            print(f"Failed to get query plan from LLM, 详细错误信息: {str(e)}")
            return None

    def parse_query_plan(self, result):
        if not result:
            print("LLM 返回内容为空，无法解析。")
            return None
        json_pattern = r'```json(.*?)```'
        match = re.search(json_pattern, result, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            print("未找到有效的 JSON 部分。")
            return None
        try:
            structured_query = json.loads(json_str)
            return structured_query
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}，提取的 JSON 内容: {json_str}")
            return None

    def get_related_nodes(self, results, question):
        preprocessed_question = self.preprocess_question(question)
        related_nodes = []
        comparison_words = ["compared to", "similar to", "comparison", "compare","similarity"]
        need_table_query = any(word in preprocessed_question for word in comparison_words)

        for result in results:
            value = result["value"]
            if isinstance(value, dict):
                if "title" in preprocessed_question:
                    related_nodes.append(value.get("title", ""))
                if "author" in preprocessed_question or "authors" in preprocessed_question:
                    related_nodes.extend(value.get("authors", []))
                if "abstract" in preprocessed_question:
                    related_nodes.append(value.get("abstract", ""))
                if "reference" in preprocessed_question or "references" in preprocessed_question:
                    related_nodes.extend(value.get("references", []))

                if need_table_query and "tables" in value:
                    for table in value["tables"]:
                        for key, val in table.items():
                            related_nodes.append(f"{key}: {val}")

        return related_nodes
    
    def evaluate_similarity(self, text1, text2):
        embedding1 = self.sbert_model.encode(text1)
        embedding2 = self.sbert_model.encode(text2)
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    
    def execute_plan(self, plan, question):
        cache = {}
        all_step_results = []  # 用于记录每一步的所有结果

        for step in plan:
            step_num = step["step"]
            targets = step["target"]
            constraints = step["constraints"]
            inputs = step["input"]

            step_inputs = []
            if inputs:
                for input_num in inputs:
                    step_inputs.extend(cache.get(f"step{input_num}", []))

            if step_num > 1:
                previous_results = cache.get(f"step{step_num - 1}", [])
                related_nodes = self.get_related_nodes(previous_results, question)
                if related_nodes:
                    if "semantic_query" in constraints:
                        relevant_nodes = [node for node in related_nodes if isinstance(node, str)]
                        new_semantic_query = constraints["semantic_query"] + " " + " ".join(relevant_nodes)
                        constraints["semantic_query"] = new_semantic_query
                    else:
                        relevant_nodes = [node for node in related_nodes if isinstance(node, str)]
                        constraints["semantic_query"] = " ".join(relevant_nodes)

            candidates = []
            if step_inputs:
                for target in targets:
                    for input_result in step_inputs:
                        value = input_result["value"]
                        if isinstance(value, dict):
                            if target == "paper":
                                semantic_query = constraints.get("semantic_query", "")
                                for key, val in value.items():
                                    if isinstance(val, str):
                                        similarity = self.evaluate_similarity(val, semantic_query)
                                        if similarity > 0.3:
                                            candidates.append(input_result)
                            elif target == "title":
                                if "title" in value:
                                    title = value["title"]
                                    semantic_query = constraints.get("semantic_query", "")
                                    similarity = self.evaluate_similarity(title, semantic_query)
                                    if similarity > 0.3:
                                        candidates.append({"value": title})
                            elif target == "authors":
                                if "authors" in value:
                                    authors = value["authors"]
                                    semantic_query = constraints.get("semantic_query", "")
                                    for author in authors:
                                        similarity = self.evaluate_similarity(author, semantic_query)
                                        if similarity > 0.3:
                                            candidates.append({"value": authors})
                                            break
                            elif target == "abstract":
                                if "abstract" in value:
                                    abstract = value["abstract"]
                                    semantic_query = constraints.get("semantic_query", "")
                                    similarity = self.evaluate_similarity(abstract, semantic_query)
                                    if similarity > 0.3:
                                        candidates.append({"value": abstract})
                            elif target == "reference":
                                if "reference" in value:
                                    candidates.append({"value": value["reference"]})
            else:
                for target in targets:
                    if target == "paper":
                        if constraints.get("semantic_query"):
                            results = self.arxiv_graph.query(constraints["semantic_query"])
                            candidates.extend(results)

            if constraints.get("filter"):
                filtered_candidates = []
                for candidate in candidates:
                    match = True
                    for filter_key, filter_value in constraints["filter"].items():
                        if isinstance(filter_value, dict) and "$gt" in filter_value:
                            value = candidate["value"].get(filter_key, 0)
                            if value <= filter_value["$gt"]:
                                match = False
                                break
                    if match:
                        filtered_candidates.append(candidate)
                candidates = filtered_candidates

            cache[f"step{step_num}"] = candidates
            all_step_results.extend(candidates)  # 记录当前步骤的结果

        final_results = cache.get(f"step{len(plan)}", [])

        if not final_results:
            print("最终结果为空，汇总之前步骤的所有结果。")
            final_results = all_step_results

        return final_results
        
    def structured_semantic_query(self, question):
        plan_text = self.query_llm_for_plan(question)
        if not plan_text:
            return "无法生成查询计划。"
        plan = self.parse_query_plan(plan_text)
        if not plan:
            return "无法解析查询计划。"

        results = self.execute_plan(plan, question)
        if not results:
            answer = "未找到相关信息。"
        else:
            answer = self.extract_answer(results, question)

        return answer


if __name__ == "__main__":
    arxiv_graph = parse_local_paper()
    api_key = "sk-c4e7f78ec6fe48379839541971a2bfc7"
    agent = ArxivAgent(arxiv_graph, api_key)

    while True:
        question = input("请输入查询内容（输入 end 结束）：")
        if question.lower() == 'end':
            break
        answer = agent.structured_semantic_query(question)
        print(answer)
