import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
intel_omp_path = r'C:\Program Files\Dell\DTP\IPDT'
os.environ['PATH'] = intel_omp_path + os.pathsep + os.environ['PATH']

from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
import arxiv
from bs4 import BeautifulSoup 
from semantic_data_structure.semantic_simple_graph import SemanticSimpleGraph
from semantic_data_structure.semantic_map import SemanticMap
import time
from typing import List, Dict, Any,Optional,Union
import re
from collections import defaultdict
import traceback
import networkx as nx
import matplotlib.pyplot as plt

class ArxivSemanticGraph:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        self.semantic_graph = SemanticSimpleGraph(SemanticMap(text_embedding_model, embedding_dim))
        self.preferences = {"liked": [], "disliked": []}
        self.similarity_threshold = 0.8
        self.embedding_model = SentenceTransformer(text_embedding_model)
        
        # 初始化FAISS索引
        self.reference_index = faiss.IndexFlatL2(embedding_dim)
        self.reference_embeddings = []
        self.root_node_embeddings = []
        self.root_node_keys = []


    def _get_text_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text], convert_to_numpy=True)

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def _node_exists(self, search_key: str, content: Any, node_type: str) -> bool:
        for item in self.semantic_graph.semantic_map.data:
            key, value, _ = item
            if key.startswith(search_key):
                if node_type == "reference":
                    ref_embedding = self._get_text_embedding(content)
                    existing_embedding = self._get_text_embedding(value)
                    if self._calculate_similarity(ref_embedding, existing_embedding) >= self.similarity_threshold:
                        return True
                elif value == content:
                    return True
        return False

    def insert(self, paper_id: str, title: str, authors: list, abstract: str,
               chapters: list, references: list, tables: list = None):
        # 插入根节点
        root_key = f"{paper_id}_title_authors"
        root_value = {
            "title": title,
            "authors": sorted(authors),
            "root_key": root_key,
            "paper_id": paper_id
        }
        if not self._node_exists(root_key, root_value, "root"):
            self.semantic_graph.add_node(
                root_key, root_value,
                text_for_embedding=f"{title} {' '.join(authors)}"
            )
            root_embedding = self._get_text_embedding(f"{title} {' '.join(authors)}")
            self.root_node_embeddings.append(root_embedding)
            self.root_node_keys.append(root_key)

        # 插入摘要节点
        abstract_key = f"{paper_id}_abstract"
        if not self._node_exists(abstract_key, abstract, "abstract"):
            self.semantic_graph.add_node(
                abstract_key, abstract,
                parent_keys=[root_key],
                parent_relation="has_abstract",
                text_for_embedding=abstract
            )

        prev_chapter_key = None
        # 处理章节
        prev_chapter_key = None
        for idx, chapter in enumerate(chapters):
            chapter_title = chapter.get('title', f"Untitled Chapter {idx}")
            chapter_key = f"{paper_id}_chapter_{idx}"

            if not self._node_exists(chapter_key, chapter_title, "chapter"):
                self.semantic_graph.add_node(
                    chapter_key, chapter_title,
                    parent_keys=[root_key],
                    parent_relation="has_chapter",
                    text_for_embedding=chapter_title
                )

                # 添加章节之间的双向上下文关系
                if prev_chapter_key:
                    # 前章节指向当前章节为 "next"
                    self.semantic_graph.insert_edge(prev_chapter_key, chapter_key, "next")
                    # 当前章节指向前章节为 "previous"
                    self.semantic_graph.insert_edge(chapter_key, prev_chapter_key, "previous")

                prev_chapter_key = chapter_key

                prev_para_key = None
                # 处理段落
                for para_idx, para in enumerate(chapter.get('paragraphs', [])):
                    para_key = f"{chapter_key}_para_{para_idx}"
                    if not self._node_exists(para_key, para, "paragraph"):
                        self.semantic_graph.add_node(
                            para_key, para,
                            parent_keys=[chapter_key],
                            parent_relation="has_paragraph",
                            text_for_embedding=para
                        )

                        # 添加段落之间的双向上下文关系
                        if prev_para_key:
                            # 前段落指向当前段落为 "next"
                            self.semantic_graph.insert_edge(prev_para_key, para_key, "next")
                            # 当前段落指向前段落为 "previous" 
                            self.semantic_graph.insert_edge(para_key, prev_para_key, "previous")

                        prev_para_key = para_key

                # 处理图片
                for img_idx, img_data in enumerate(chapter.get('images', [])):
                    img_key = f"{chapter_key}_img_{img_idx}"
                    if not self._node_exists(img_key, f"Image in {chapter_title}", "image"):
                        self.semantic_graph.add_node(
                            img_key, img_data,
                            parent_keys=[chapter_key],
                            parent_relation="has_image",
                            text_for_embedding=f"Figure in {chapter_title}"
                        )

        # 处理参考文献
        for ref_idx, ref in enumerate(references):
            ref_key = f"{paper_id}_ref_{ref_idx}"
            ref_embedding = self._get_text_embedding(ref)

            # FAISS相似性检查
            if len(self.reference_embeddings) > 0:
                D, I = self.reference_index.search(ref_embedding.reshape(1, -1), 1)
                if D[0][0] < (1 - self.similarity_threshold):
                    continue

            similar_root_key = None
            if len(self.root_node_embeddings) > 0:
                root_embeddings_array = np.vstack(self.root_node_embeddings)
                similarities = np.dot(ref_embedding, root_embeddings_array.T)
                max_similarity_idx = np.argmax(similarities)
                if similarities[0][max_similarity_idx] >= self.similarity_threshold:
                    similar_root_key = self.root_node_keys[max_similarity_idx]

            if not self._node_exists(ref_key, ref, "reference"):
                parents = [root_key]
                relations = "has_reference" 
                if similar_root_key:
                    parents.append(similar_root_key)
                    relations="is_referenced_by"

                self.semantic_graph.add_node(
                    ref_key, ref,
                    parent_keys=parents,
                    parent_relation=relations,
                    text_for_embedding=ref
                )
                self.reference_embeddings.append(ref_embedding)
                self.reference_index.add(ref_embedding.reshape(1, -1))

        # 处理表格
        if tables:
            for tbl_idx, table in enumerate(tables):
                tbl_key = f"{paper_id}_table_{tbl_idx}"
                table_meta = {
                    "columns": [],
                    "row_count": 0,
                    "root_key": tbl_key
                }

                if isinstance(table, list) and table:
                    first_item = table[0]
                    if isinstance(first_item, dict):
                        table_meta["columns"] = list(first_item.keys())
                        table_meta["row_count"] = len(table)
                    elif isinstance(first_item, (list, tuple)):
                        table_meta["columns"] = [f"Col{i}" for i in range(len(table[0]))]
                        table_meta["row_count"] = len(table)
                elif isinstance(table, dict):
                    table_meta["columns"] = list(table.keys())
                    table_meta["row_count"] = 1

                self.semantic_graph.add_node(
                    tbl_key, table_meta,
                    parent_keys=[root_key],
                    parent_relation="has_table",
                    text_for_embedding=f"Table with columns: {', '.join(table_meta['columns'])}"
                )

                if isinstance(table, list):
                    for row_idx, row in enumerate(table):
                        if isinstance(row, dict):
                            row_text = " | ".join([f"{k}:{v}" for k, v in row.items()])
                        elif isinstance(row, (list, tuple)):
                            row_text = " | ".join([str(item) for item in row])
                        else:
                            row_text = str(row)
                        row_key = f"{tbl_key}_row_{row_idx}"
                        if not self._node_exists(row_key, row_text, "table_row"):
                            self.semantic_graph.add_node(
                                row_key, row,
                                parent_keys=[tbl_key],
                                parent_relation="has_row",
                                text_for_embedding=row_text
                            )
                elif isinstance(table, dict):
                    row_text = " | ".join([f"{k}:{v}" for k, v in table.items()])
                    row_key = f"{tbl_key}_row_0"
                    if not self._node_exists(row_key, row_text, "table_row"):
                        self.semantic_graph.add_node(
                            row_key, table,
                            parent_keys=[tbl_key],
                            parent_relation="has_row",
                            text_for_embedding=row_text
                        )

        self.semantic_graph.build_index()


    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        query_embedding = self._get_text_embedding(query_text)
        scores, indices = self.semantic_graph.semantic_map.index.search(query_embedding, k*3)
        
        results = []
        seen_papers = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.semantic_graph.semantic_map.data):
                continue
                
            node = self.semantic_graph.semantic_map.data[idx]
            if not node:
                continue
                
            node_key = node[0]
            
            paper_root = self._get_root_node(node_key)
            if not paper_root:
                continue
                
            if paper_root["key"] in seen_papers:
                continue
                
            full_paper = self._get_full_paper(paper_root["key"])
            if full_paper:
                full_paper["similarity_score"] = float(1 - score)
                results.append(full_paper)
                seen_papers.add(paper_root["key"])
            
            if len(results) >= k:
                break
                
        return results

    def _get_root_node(self, key: str) -> Optional[Dict]:
        current_key = key
        while True:
            parents = self.semantic_graph.graph_relations.get(current_key, {}).get("parents", {})
            if not parents:
                break
            current_key = next(iter(parents.keys()))
        
        node = next((n for n in self.semantic_graph.semantic_map.data if n[0] == current_key), None)
        if node:
            return {
                "key": node[0],
                "value": node[1],
                "embedding": node[2]
            }
        return None

    def _get_full_paper(self, root_key: str) -> Optional[Dict]:
        root_node = next(
            (n for n in self.semantic_graph.semantic_map.data if n[0] == root_key), 
            None
        )
        if not root_node:
            return None

        paper_id = root_key.split("_")[0]
        paper = {
            "metadata": {
                "root_key": root_key,
                "paper_id": paper_id,
                "title": root_node[1].get("title", ""),
                "authors": root_node[1].get("authors", [])
            },
            "abstract": "",
            "chapters": [],
            "references": [],
            "tables": []
        }
        
        # 获取摘要
        abstract_node = next((n for n in self.semantic_graph.semantic_map.data 
                             if n[0].startswith(paper_id) and "_abstract" in n[0]), None)
        if abstract_node:
            paper["abstract"] = abstract_node[1]
        
        # 获取章节
        chapter_nodes = [n for n in self.semantic_graph.semantic_map.data 
                        if n[0].startswith(paper_id) and "_chapter_" in n[0]]
        for chap in chapter_nodes:
            chapter = {
                "title": chap[1] if isinstance(chap[1], str) else chap[1].get("title", ""),
                "paragraphs": [],
                "images": []
            }
            
            # 获取段落
            paragraph_nodes = [n for n in self.semantic_graph.semantic_map.data 
                             if n[0].startswith(chap[0]) and "_para_" in n[0]]
            for para in paragraph_nodes:
                chapter["paragraphs"].append(para[1])
            
            # 获取图片
            image_nodes = [n for n in self.semantic_graph.semantic_map.data 
                          if n[0].startswith(chap[0]) and "_img_" in n[0]]
            chapter["images"] = [n[1] for n in image_nodes]
            
            paper["chapters"].append(chapter)
        
        # 获取参考文献
        ref_nodes = [n for n in self.semantic_graph.semantic_map.data 
                    if n[0].startswith(paper_id) and "_ref_" in n[0]]
        paper["references"] = [n[1] for n in ref_nodes]
        
        # 获取表格
        table_nodes = [n for n in self.semantic_graph.semantic_map.data 
                      if n[0].startswith(paper_id) and "_table_" in n[0]]
        for tbl in table_nodes:
            if "_row_" not in tbl[0]:
                table = {
                    "metadata": tbl[1],
                    "rows": []
                }
                row_nodes = [n for n in self.semantic_graph.semantic_map.data 
                            if n[0].startswith(tbl[0]) and "_row_" in n[0]]
                for row in row_nodes:
                    table["rows"].append(row[1])
                paper["tables"].append(table)
        
        return paper


    def delete(self, paper_id: str, cascade: bool = True) -> bool:
        """
        删除指定论文及其所有关联节点
        :param paper_id: 要删除的论文ID（如"2305.12345"）
        :param cascade: 是否级联删除所有子节点
        :return: 是否删除成功
        """
        root_key = f"{paper_id}_title_authors"
        
        # 检查是否存在该论文
        if not any(node[0] == root_key for node in self.semantic_graph.semantic_map.data):
            print(f"论文 {paper_id} 不存在")
            return False

        # 级联删除所有子节点
        if cascade:
            children = self._get_all_children(root_key)
            for child_key in reversed(children + [root_key]):  # 反向删除从叶子节点开始
                self._safe_delete_node(child_key)

        # 更新参考文献索引
        self._update_reference_index()

        # 从偏好列表中移除
        self.preferences["liked"] = [n for n in self.preferences["liked"] if n[0] != root_key]
        self.preferences["disliked"] = [n for n in self.preferences["disliked"] if n[0] != root_key]

        self.semantic_graph.build_index()
        return True

    def _safe_delete_node(self, key: str):
        """安全删除节点并维护索引"""
        try:
            # 从语义图中删除
            self.semantic_graph.delete_node(key)
            
            # 如果是参考文献节点，从FAISS索引移除
            if "_ref_" in key:
                ref_index = next((i for i, (k, _, _) in enumerate(self.semantic_graph.semantic_map.data) 
                                if k == key), -1)
                if ref_index != -1 and ref_index < len(self.reference_embeddings):
                    self.reference_index.remove_ids(np.array([ref_index], dtype=np.int64))
                    del self.reference_embeddings[ref_index]
        except ValueError:
            pass

    def _get_all_children(self, root_key: str) -> List[str]:
        """递归获取所有子节点"""
        children = []
        to_process = [root_key]
        
        while to_process:
            current_key = to_process.pop()
            relations = self.semantic_graph.graph_relations.get(current_key, {})
            for child_key in relations.get("children", {}).keys():
                children.append(child_key)
                to_process.append(child_key)
                
        return children

    def _update_reference_index(self):
        """完全重建参考文献索引"""
        self.reference_index.reset()
        self.reference_embeddings = []
        
        # 重新添加所有参考文献嵌入
        ref_nodes = [n for n in self.semantic_graph.semantic_map.data if "_ref_" in n[0]]
        for node in ref_nodes:
            embedding = self._get_text_embedding(node[1])
            self.reference_embeddings.append(embedding)
        
        if self.reference_embeddings:
            embeddings_array = np.vstack(self.reference_embeddings)
            self.reference_index.add(embeddings_array)


    def mark_as_liked(self, paper_id: str) -> bool:
        """标记论文为喜欢"""
        root_key = f"{paper_id}_title_authors"
        node = next((n for n in self.semantic_graph.semantic_map.data if n[0] == root_key), None)
        
        if not node:
            print(f"论文 {paper_id} 不存在")
            return False
        
        # 如果已经存在则移除旧记录
        self.preferences["liked"] = [n for n in self.preferences["liked"] if n[0] != root_key]
        self.preferences["liked"].append(node)
        
        # 如果同时在不喜欢列表中则移除
        self.preferences["disliked"] = [n for n in self.preferences["disliked"] if n[0] != root_key]
        
        return True

    def mark_as_disliked(self, paper_id: str) -> bool:
        """标记论文为不喜欢"""
        root_key = f"{paper_id}_title_authors"
        node = next((n for n in self.semantic_graph.semantic_map.data if n[0] == root_key), None)
        
        if not node:
            print(f"论文 {paper_id} 不存在")
            return False
        
        # 如果已经存在则移除旧记录
        self.preferences["disliked"] = [n for n in self.preferences["disliked"] if n[0] != root_key]
        self.preferences["disliked"].append(node)
        
        # 如果同时在喜欢列表中则移除
        self.preferences["liked"] = [n for n in self.preferences["liked"] if n[0] != root_key]
        
        return True

    def recommend(self, query_text: str, k: int = 5, preference_weight: float = 2.0) -> List[Dict]:
        """
        个性化推荐论文
        :param query_text: 查询文本
        :param k: 返回数量
        :param preference_weight: 偏好权重（喜欢论文的分数倍增值）
        """
        # 获取基础查询结果
        base_results = self.query(query_text, k*3)
        
        # 计算偏好分数
        scored_results = []
        for paper in base_results:
            root_key = paper["metadata"]["root_key"]
            score = paper["similarity_score"]
            
            # 应用偏好权重
            if any(n[0] == root_key for n in self.preferences["liked"]):
                score *= preference_weight
            elif any(n[0] == root_key for n in self.preferences["disliked"]):
                score *= 0.2  # 降低不喜欢论文的分数
                
            scored_results.append((score, paper))
        
        # 按分数排序并去重
        seen_papers = set()
        final_results = []
        for score, paper in sorted(scored_results, key=lambda x: -x[0]):
            root_key = paper["metadata"]["root_key"]
            if root_key not in seen_papers:
                final_results.append(paper)
                seen_papers.add(root_key)
            if len(final_results) >= k:
                break
                
        return final_results

    def get_preferences(self) -> Dict[str, List[Dict]]:
        """获取当前偏好论文的完整信息"""
        return {
            "liked": [self._get_full_paper(n[0]) for n in self.preferences["liked"]],
            "disliked": [self._get_full_paper(n[0]) for n in self.preferences["disliked"]]
        }

    def batch_delete(self, paper_ids: List[str]) -> Dict[str, int]:
        """批量删除论文"""
        results = {"success": 0, "failures": 0}
        for pid in paper_ids:
            if self.delete(pid):
                results["success"] += 1
            else:
                results["failures"] += 1
        return results

    def export_preferences(self, file_path: str):
        """导出偏好列表到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump({
                "liked": [n[0] for n in self.preferences["liked"]],
                "disliked": [n[0] for n in self.preferences["disliked"]]
            }, f)

    def import_preferences(self, file_path: str):
        """从文件导入偏好列表"""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.preferences["liked"] = [n for n in self.semantic_graph.semantic_map.data 
                                       if n[0] in data["liked"]]
            self.preferences["disliked"] = [n for n in self.semantic_graph.semantic_map.data 
                                          if n[0] in data["disliked"]]

    def list_all(self):
        """统计图谱中论文的数量"""
        root_nodes = [node for node in self.semantic_graph.semantic_map.data if node[0].endswith("_title_authors")]
        return root_nodes
    
   

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# 全局配置
MAX_RETRIES = 3
TIMEOUT = 60  # 超时时间延长到60秒

# 创建带重试机制的Session
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

def fetch_arxiv_html(query, max_results=5, output_dir='arxiv_html'):
    """改进后的下载函数，添加重试机制"""
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
        paper_id = result.get_short_id()
        html_url = result.entry_id.replace("abs", "html")
        file_name = os.path.join(output_dir, f"{paper_id}.html")
        
        try:
            # 使用带重试的Session
            response = http.get(html_url, timeout=TIMEOUT)
            response.raise_for_status()
            
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(response.text)
            html_paths.append(file_name)
            print(f"成功下载：{paper_id}")
        except Exception as e:
            print(f"下载失败 [{paper_id}]: {str(e)}")
            continue  # 跳过失败项继续下载其他论文
            
    return html_paths
# fetch_arxiv_html("agent memory",100)
def decode_html(html_file_path, image_dir='arxiv_images'):
    paper_id = os.path.splitext(os.path.basename(html_file_path))[0]
    result = {
        'title': '',
        'authors': [],
        'abstract': '',
        'chapters': [],
        'references': [],
        'tables': []
    }

    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            if not html_content:
                raise ValueError("空文件内容")

            soup = BeautifulSoup(html_content, 'html.parser')

            # 标题解析
            try:
                title_tag = soup.find('title')
                if title_tag is None:
                    raise AttributeError("未找到标题标签")
                result['title'] = title_tag.text.strip()
            except AttributeError as e:
                print(f"[{paper_id}] 标题解析失败: {str(e)}")

            # 作者解析
            try:
                author_elements = soup.select('.ltx_personname')
                if not author_elements:
                    raise ValueError("未找到作者元素")
                result['authors'] = list({e.text.strip() for e in author_elements})
            except Exception as e:
                print(f"[{paper_id}] 作者解析失败: {str(e)}")

            # 摘要解析
            try:
                abstract_div = soup.find('div', class_='ltx_abstract')
                if abstract_div is None:
                    raise ValueError("未找到摘要部分")
                abstract_p = abstract_div.find('p')
                if abstract_p is None:
                    raise ValueError("摘要部分未找到段落")
                result['abstract'] = abstract_p.text
            except Exception as e:
                print(f"[{paper_id}] 摘要解析失败: {str(e)}")

            # 章节解析（带错误隔离）
            toc_entries = soup.select('.ltx_tocentry') or []
            for entry in toc_entries:
                try:
                    chapter = process_chapter(entry, paper_id, image_dir, soup)
                    result['chapters'].append(chapter)
                except Exception as e:
                    import traceback
                    print(f"[{paper_id}] 章节解析失败: 章节 {entry} 发生错误，具体信息：{traceback.format_exc()}")
                    continue

            # 参考文献解析
            try:
                result['references'] = process_references(soup)
            except Exception as e:
                import traceback
                print(f"[{paper_id}] 参考文献解析失败: {traceback.format_exc()}")

            # #表格解析
            # try:
            #     result['tables'] = process_tables(soup)
            # except Exception as e:
            #     import traceback
            #     print(f"[{paper_id}] 表格解析失败: {traceback.format_exc()}")

    except Exception as e:
        import traceback
        print(f"[{paper_id}] 严重错误: 文件 {html_file_path} 解析时发生错误，具体信息：{traceback.format_exc()}")
        return None  # 返回None表示解析完全失败

    return result

def process_chapter(entry, paper_id, image_dir, soup):
    """处理单个章节的辅助函数"""
    chapter = {'title': '', 'paragraphs': [], 'images': []}

    try:
        # 标题处理
        title_span = entry.find('span', class_='ltx_text ltx_ref_title')
        chapter['title'] = title_span.text.strip() if title_span else "未命名章节"

        # 内容定位
        chapter_id = entry.find('a')['href'].split('#')[-1]
        section = soup.find('section', id=chapter_id)
        if not section:
            return chapter

        # 段落处理
        paragraphs = []
        for p in section.find_all('p'):
            try:
                text = p.get_text(separator=' ', strip=True)
                if len(text) > 30:  # 过滤短文本
                    paragraphs.append(text)
            except:
                continue
        chapter['paragraphs'] = paragraphs

        # # 图片处理（带重试）
        # for img in section.find_all('img'):
        #     try:
        #         src = img.get('src')
        #         if not src:
        #             continue

        #         img_url = f"https://arxiv.org/html/{paper_id}/{src}"
        #         img_data = download_image(img_url)

        #         if img_data:
        #             chapter['images'].append({
        #                 'data': img_data,
        #                 'description': img.get('alt', '')
        #             })
        #     except Exception as e:
        #         print(f"图片处理失败: {str(e)}")
        #         continue

    except Exception as e:
        raise RuntimeError(f"章节处理失败: {str(e)}")

    return chapter

def process_references(soup):
    """处理参考文献解析"""
    references = []
    if soup.find("ul", class_="ltx_biblist"):
        for item in soup.find_all("li", class_="ltx_bibitem"):
            ref_text = " ".join(item.stripped_strings)
            if len(ref_text) > 20:  # 过滤无效引用
                references.append(ref_text)
    return references

def process_tables(soup):
    """处理表格解析"""
    tables = []
    for table in soup.find_all("table", class_="ltx_tabular"):
        table_data = []
        headers = []

        # 表头检测
        header_row = table.find("tr", class_="ltx_thead")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

        # 表格内容
        for row in table.find_all("tr", class_=lambda x: x != "ltx_thead"):
            cells = [cell.get_text(separator=" ", strip=True) for cell in row.find_all(["td", "th"])]
            if headers and len(cells) == len(headers):
                table_data.append(dict(zip(headers, cells)))
            elif cells:
                table_data.append(cells)

        if table_data:
            tables.append({
                "metadata": {"columns": headers} if headers else {},
                "rows": table_data
            })
    return tables


def download_image(url):
    """带重试机制的图片下载"""
    for attempt in range(MAX_RETRIES):
        try:
            response = http.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"图片下载失败 [{url}] 重试{MAX_RETRIES}次后仍然失败")
                return None
            time.sleep(2 ** attempt)  # 指数退避

def parse_arxiv(query, max_results=5, output_dir='arxiv_html'):
    """改进后的主解析函数"""
    html_paths = fetch_arxiv_html(query, max_results, output_dir)
    graph = ArxivSemanticGraph() 
    
    success_count = 0
    for idx, html_path in enumerate(html_paths, 1):
        paper_id = os.path.basename(html_path).replace('.html', '')
        try:
            decoded_info = decode_html(html_path)
            if not decoded_info:
                continue
                
            graph.insert(
                paper_id,
                decoded_info['title'],
                decoded_info['authors'],
                decoded_info['abstract'],
                decoded_info['chapters'],
                decoded_info['references'],
                decoded_info.get('tables', [])
            )
            success_count += 1
            print(f"({idx}/{len(html_paths)}) 成功处理 {paper_id}")
        except Exception as e:
            print(f"({idx}/{len(html_paths)}) 处理失败: {str(e)}")
            continue  # 跳过失败论文
            
    print(f"处理完成，成功导入 {success_count}/{len(html_paths)} 篇论文")
    graph.semantic_graph.build_index()  # 最后统一构建索引
    return graph

def parse_local_paper(html_dir='arxiv_html'):
    """改进后的本地解析函数"""
    graph = ArxivSemanticGraph()

    if not os.path.exists(html_dir):
        print(f"指定的目录 {html_dir} 不存在。")
        return graph

    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    total_files = len(html_files)
    success_count = 0

    for idx, filename in enumerate(html_files, 1):
        html_path = os.path.join(html_dir, filename)
        paper_id = filename.replace('.html', '')

        try:
            decoded_info = decode_html(html_path)
            if not decoded_info:
                continue

            graph.insert(
                paper_id,
                decoded_info['title'],
                decoded_info['authors'],
                decoded_info['abstract'],
                decoded_info['chapters'],
                decoded_info['references'],
                decoded_info.get('tables', [])
            )
            success_count += 1
            print(f"({idx}/{total_files}) 成功处理 {paper_id}")
        except Exception as e:
            import traceback
            print(f"({idx}/{total_files}) 处理失败: 文件 {html_path} 发生错误，具体信息：{traceback.format_exc()}")
            continue

    print(f"本地处理完成，成功导入 {success_count}/{total_files} 篇论文")
    graph.semantic_graph.build_index()
    return graph




def show_paper_structure(paper):
    print(f"标题: {paper['metadata']['title']}")
    print(f"作者: {', '.join(paper['metadata']['authors'])}")
    print(f"摘要: {paper['abstract']}")

    print("\n章节信息:")
    for i, chapter in enumerate(paper["chapters"], start=1):
        print(f"  章节 {i}: {chapter['title']}")
        if chapter["paragraphs"]:
            print("    段落信息:")
            for j, para in enumerate(chapter["paragraphs"], start=1):
                print(f"      段落 {j}: {para[:200]}...")
        if chapter["images"]:
            print("    图片信息:")
            for j, img in enumerate(chapter["images"], start=1):
                print(f"      图片 {j}: {img.get('description', '无描述')}")

    print("\n参考文献信息:")
    for i, ref in enumerate(paper["references"], start=1):
        print(f"  参考文献 {i}: {ref}")

    print("\n表格信息:")
    for i, table in enumerate(paper["tables"], start=1):
        print(f"  表格 {i}:")
        print(f"    列名: {', '.join(table['metadata']['columns'])}")
        print("    行数据:")
        for j, row in enumerate(table["rows"], start=1):
            if isinstance(row, dict):
                row_str = " | ".join([f"{k}: {v}" for k, v in row.items()])
            else:
                row_str = " | ".join(map(str, row))
            print(f"      行 {j}: {row_str}")


import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import requests
import logging
import os
from datetime import datetime
import sys

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-c4e7f78ec6fe48379839541971a2bfc7"
DEEPSEEK_MODEL = "deepseek-chat"
TIMEOUT = 60  


class ArxivAgent:
    def __init__(self, graph,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.4):
        """
        初始化智能代理
        :param graph: 已构建的ArxivSemanticGraph实例
        :param embedding_model: 嵌入模型名称
        :param similarity_threshold: 相似度阈值
        """
        self.setup_logging()
        self.graph = graph
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold

        # 初始化语义索引
        self.node_embeddings = []
        self.node_info = []
        self._build_hnsw_index()
        self.type_mapping = {
            "paper": "paper_root",
            "papers": "paper_root",
            "abstract": "abstract",
            "chapter": "chapter",
            "reference": "reference",
            "table": "table",
            "paragraph": "paragraph",

            "image": "image",
            "figure": "image",
            "table_row": "table_row",
            "row": "table_row",

            "root": "paper_root",
            "chapters": "chapter",
            "references": "reference",
            "tables": "table",
            "paragraphs": "paragraph",
            "images": "image"
        }
        self.relation_mapping = {
            "has_chapter": ["has_section", "contains_chapter", "chapter"],
            "has_abstract": ["abstract", "paper_abstract"],
            "has_paragraph": ["para", "paragraph", "text_section","has_paragraph"]
        }
        
        
    def _normalize_relation(self, rel: str) -> str:
        original = rel
        rel = rel.lower().strip().replace(" ", "_")
        for std_rel, variants in self.relation_mapping.items():
            if rel == std_rel or rel in variants:
                self.logger.debug(f"[NORMALIZE] 关系标准化: {original} -> {std_rel}")
                return std_rel
        self.logger.debug(f"[NORMALIZE] 未知关系类型: {original} 保持原样")
        return rel

    def setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger('ArxivAgent')
        self.logger.setLevel(logging.INFO)

        # 防止重复添加处理器
        if not self.logger.handlers:
            # 文件处理器（按天分割）
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'arxiv_agent_{datetime.now().strftime("%Y%m%d")}.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            ))

            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _build_hnsw_index(self):
        """重建语义索引（包含所有节点）"""
        # 初始化数据结构
        self.node_embeddings = []
        self.node_info = []
        
        # HNSW参数配置
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexHNSWFlat(dim, 32)

        # 节点过滤和验证
        valid_nodes = []
        for node in self.graph.semantic_graph.semantic_map.data:
            # 基本结构验证
            if not isinstance(node, tuple) or len(node) < 2:
                print(f"⚠️ 无效节点结构: {node}")
                continue
                
            node_key = node[0]
            node_content = node[1]
            
            # 键格式验证
            if not isinstance(node_key, str) or len(node_key) < 3:
                print(f"⚠️ 非法节点键: {node_key}")
                continue
                
            valid_nodes.append((node_key, node_content))

        print(f"开始索引构建，有效节点数: {len(valid_nodes)}")

        # 处理有效节点
        for node_key, node_content in valid_nodes:
            try:
                # 生成文本描述
                text = self._generate_node_text(node_key, node_content)
                if not text.strip():
                    print(f"🈳 空内容节点: {node_key}")
                    continue

                # 生成嵌入
                embedding = self.embedder.encode(text)
                if embedding.ndim == 2:
                    embedding = embedding.squeeze(0)

                # 收集元数据
                self.node_embeddings.append(embedding)
                self.node_info.append({
                    "key": node_key,
                    "type": self._get_node_type(node_key),
                    "content": node_content,
                    "paper_id": node_key.split("_")[0] if '_' in node_key else 'unknown'
                })

            except Exception as e:
                print(f"❌ 处理节点 {node_key} 失败: {str(e)}")
                traceback.print_exc()
                continue

        # 转换为numpy数组
        if len(self.node_embeddings) == 0:
            raise ValueError("无有效节点可供索引")
            
        self.node_embeddings = np.array(self.node_embeddings)
        self.index.add(self.node_embeddings)
        
        # 健康检查
        print(f"\n索引构建完成状态:")
        print(f"- 嵌入向量数: {self.index.ntotal}")
        print(f"- 节点信息数: {len(self.node_info)}")
        print(f"- 嵌入维度: {self.node_embeddings.shape[1]}")
        
        return True

    def _get_node_type(self, key: str) -> str:
        """根据节点键识别类型"""
        if "_abstract" in key:
            return "abstract"
        elif "_title_authors" in key:
            return "paper_root"
        elif "_para_" in key:
            return "paragraph"
        elif "_img_" in key:
            return "image"
        elif "_chapter_" in key:
            return "chapter"
        elif "_ref_" in key:
            return "reference"
        elif "_table_" in key:
            return "table"
        return "unknown"

    def _generate_node_text(self, key: str, content: Any) -> str:
        node_type = self._get_node_type(key)

        if isinstance(content, str):
            return content

        if node_type == "paper_root":
            return (
                f"Paper: {content.get('title', 'Unknown Title')} by {', '.join(content.get('authors', []))}"
                f" ({content.get('publication_date', 'unknown')})"
            )
        elif node_type == "chapter":
            return (
                f"Chapter {content.get('section_number', 'X')}: {content.get('title', 'Unknown Title')}"
                f" (parent chapter: {content.get('parent_chapter', 'None')})"
            )
        elif node_type == "paragraph":
            return (
                f"Paragraph {content.get('position_in_chapter', '?')}: {content.get('text', '').strip()[:200]}..."
                f" (in chapter: {content.get('chapter_title', 'Unknown')})"
            )
        elif node_type == "reference":
            return (
                f"Reference: {content.get('title', 'No Title')}"
                f" by {', '.join(content.get('authors', []))} ({content.get('publication_year', '?')})"
            )
        elif node_type == "table":
            return (
                f"Table with columns: {', '.join(content.get('columns', []))}"
                f" containing {content.get('row_count', 0)} rows"
            )
        return str(content)

    def _semantic_search(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """参数添加top_k并实现正确搜索逻辑"""
        if not candidates:
            return []

        # 获取候选节点的索引
        candidate_indices = [i for i, n in enumerate(self.node_info) if n in candidates]
        candidate_embeddings = self.node_embeddings[candidate_indices]

        # 创建临时索引
        temp_index = faiss.IndexFlatL2(candidate_embeddings.shape[1])
        temp_index.add(candidate_embeddings)

        # 执行搜索
        query_embedding = self.embedder.encode(query)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze(0)

        top_k = min(top_k, len(candidates))
        distances, indices = temp_index.search(query_embedding.reshape(1, -1), top_k)

        # 转换结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            original_idx = candidate_indices[idx]
            results.append({
                **self.node_info[original_idx],
                "score": 1 - distances[0][i]
            })

        return sorted(results, key=lambda x: -x['score'])

    def _structural_search(self, target: str, input_data: List) -> List[Dict]:
        """结构化搜索"""
        results = []
        paper_ids = [item["paper_id"] for item in input_data]

        for node in self.node_info:
            if node["type"] == target and node["paper_id"] in paper_ids:
                results.append({
                    "key": node["key"],
                    "type": node["type"],
                    "paper_id": node["paper_id"],
                    "content": node["content"]
                })
        return results

    def _resolve_input_data(self, step: Dict, context: Dict) -> List:
        """解析输入依赖"""
        input_data = []
        for inp in step.get("input", []):
            if "from step" in inp:
                ref_step = int(inp.split()[-1])
                input_data.extend(context.get(f"step_{ref_step}", []))
        return list({item["paper_id"] for item in input_data})

    def generate_answer(self, context: Dict, query: str) -> str:
        """使用LLM生成最终回答"""
        prompt = f"""
        根据以下查询执行的上下文信息，生成一个详细的回答。
        查询请求：{query}
        上下文信息：{context}
        """
        response = self._call_llm(prompt)
        return response

    def _call_llm(self, prompt: str) -> str:
        """改进的API调用方法，添加重试机制"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        data = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt[:3000]}],
            "temperature": 0.7,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    DEEPSEEK_API_URL,
                    headers=headers,
                    json=data,
                    timeout=TIMEOUT  # 延长超时时间
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"API调用失败，正在重试... ({attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                self.logger.error("无法生成报告：API请求失败")
                raise
        return "无法生成报告：API请求失败"

    def print_index_status(self):
        """打印索引状态"""
        self.logger.info("当前索引状态:")
        self.logger.info(f"索引包含 {self.index.ntotal} 个向量")
        self.logger.info(f"节点信息数量: {len(self.node_info)}")
        self.logger.info(f"嵌入矩阵形状: {self.node_embeddings.shape}")
        self.logger.info("前5个节点类型:")
        for info in self.node_info[:5]:
            self.logger.info(f"- {info['type']}: {info['key']}")

    def print_query_results(self, context: Dict[str, Any]):
        """打印查询计划每一步的搜索结果"""
        for step, results in context.items():
            self.logger.info(f"\n{'=' * 30} {step} 搜索结果 {'=' * 30}")
            if not results:
                self.logger.info(f"{step} 未找到相关结果。")
                continue
            for i, result in enumerate(results, start=1):
                self.logger.info(f"结果 {i}:")
                self.logger.info(f"  节点类型: {result['type']}")
                self.logger.info(f"  论文 ID: {result['paper_id']}")
                self.logger.info(f"  节点键: {result['key']}")
                if 'similarity' in result:
                    self.logger.info(f"  相似度: {result['similarity']:.4f}")
                self.logger.info(f"  内容: {str(result['content'])[:200]}...")

    def check_index_health(self):
        """检查索引健康状况"""
        self.logger.info("\n=== 索引诊断报告 ===")

        # 基础统计
        self.logger.info(f"索引包含向量数: {self.index.ntotal}")
        self.logger.info(f"节点信息记录数: {len(self.node_info)}")
        self.logger.info(f"嵌入矩阵形状: {self.node_embeddings.shape}")

        # 类型分布统计
        type_counts = {}
        for info in self.node_info:
            t = info['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        self.logger.info("\n节点类型分布:")
        for t, cnt in type_counts.items():
            self.logger.info(f"- {t}: {cnt}")

        # 随机采样检查
        self.logger.info("\n随机采样检查:")
        for _ in range(3):
            idx = np.random.randint(0, len(self.node_info))
            node = self.node_info[idx]
            self.logger.info(f"Key: {node['key']}")
            self.logger.info(f"Type: {node['type']}")
            self.logger.info(f"Embedding Norm: {np.linalg.norm(self.node_embeddings[idx]):.2f}")
            self.logger.info(f"Content Sample: {str(node['content'])[:100]}...\n")

    def validate_query_plan(self, plan: List[Dict]):
        """验证查询计划有效性"""
        valid_types = {'paper_root', 'abstract', 'chapter', 'reference', 'table'}

        for step in plan:
            # 校验步骤编号
            if 'step' not in step or not isinstance(step['step'], int):
                self.logger.error(f"步骤缺少有效编号: {step}")
                raise ValueError(f"步骤缺少有效编号: {step}")

            # 校验目标类型
            target = step.get('target', '').lower()
            if target not in valid_types:
                self.logger.error(f"无效目标类型: {target}，有效类型为: {valid_types}")
                raise ValueError(f"无效目标类型: {target}，有效类型为: {valid_types}")

            # 校验约束条件
            constraints = step.get('constraints', {})
            if not constraints:
                self.logger.error(f"步骤 {step['step']} 缺少约束条件")
                raise ValueError(f"步骤 {step['step']} 缺少约束条件")

            # 校验输入依赖
            inputs = step.get('input', [])
            for inp in inputs:
                if not inp.startswith('step_'):
                    self.logger.error(f"无效输入依赖格式: {inp}，应类似 'step_1'")
                    raise ValueError(f"无效输入依赖格式: {inp}，应类似 'step_1'")

    def validate_query_plan(self, plan: List[Dict]):
        """查询计划验证"""
        valid_types = set(self.type_mapping.values())  # 使用映射后的类型

        for step in plan:
            # 校验目标类型
            target_type = step.get('target', '')
            if target_type not in valid_types:
                self.logger.error(
                    f"无效目标类型: '{target_type}'\n"
                    f"有效类型应为: {', '.join(valid_types)}\n"
                    f"提示：可以使用类似'paper_root'等系统定义的类型名称"
                )
                raise ValueError(
                    f"无效目标类型: '{target_type}'\n"
                    f"有效类型应为: {', '.join(valid_types)}\n"
                    f"提示：可以使用类似'paper_root'等系统定义的类型名称"
                )

            # 校验约束条件
            if "semantic_query" not in step["constraints"] and "selected_papers" not in step["constraints"]:
                self.logger.warning(f"警告：步骤 {step['step']} 缺少有效约束条件")

    # function call
    def search_units(self,
                     entity_type: str,
                     attribute_filters: Dict = None,
                     relation_filters: Dict = None,
                     keywords: str = None,
                     top_k: int = 15) -> List[Dict]:
        """
        多条件组合搜索函数
        :param entity_type: 目标实体类型（paper_root/abstract/chapter等）
        :param attribute_filters: 属性过滤条件 {"authors": ["John"], "year": 2023}
        :param relation_filters: 关系过滤 {"has_reference": True}
        :param keywords: 关键词语义搜索
        :param top_k: 返回结果数量
        """
        if isinstance(top_k, list):
            top_k = int(top_k[0]) if len(top_k) > 0 else 5
        top_k = int(top_k)

        # 确保entity_type是字符串
        if isinstance(entity_type, list):
            entity_type = entity_type[0] if len(entity_type) > 0 else "paper_root"

        # 确保keywords是字符串
        if keywords and isinstance(keywords, list):
            keywords = " ".join(keywords)
        # 步骤1：按类型过滤
        candidates = [n for n in self.node_info if n['type'] == entity_type]

        # 步骤2：属性过滤
        if attribute_filters:
            candidates = self._apply_attribute_filters(candidates, attribute_filters)

        # 步骤3：语义搜索
        if keywords:
            candidates = self._semantic_search(keywords, candidates, top_k=top_k * 2)

        # 步骤4：关系过滤
        if relation_filters:
            candidates = self._apply_relation_filters(candidates, relation_filters)

        # 最终排序和截取
        results = sorted(
            candidates,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:top_k]
        self.logger.info(f"search_units 找到 {len(results)} 条结果")
        return results

    def _apply_attribute_filters(self, candidates, filters):
        filtered = []
        for node in candidates:
            match = True
            content = node['content']
            for attr, values in filters.items():
                # 处理不同节点类型的属性提取
                value = None
                if node['type'] == 'paper_root':
                    value = content.get(attr, '')
                elif node['type'] == 'reference':
                    value = self._extract_ref_attr(content, attr)
                # 其他类型处理

                if not self._check_value_match(value, values):
                    match = False
                    break
            if match:
                filtered.append(node)
        return filtered

    def _extract_ref_attr(self, ref_text, attr):
        # 实现参考文献属性提取逻辑
        if attr == 'year':
            return re.findall(r'\b(19|20)\d{2}\b', ref_text)[-1]  # 简单实现
        return ''

    def _check_value_match(self, actual, expected_values):
        if isinstance(actual, list):
            return any(v in actual for v in expected_values)
        return actual in expected_values

    def _apply_relation_filters(self, candidates, filters):
        filtered = []
        for node in candidates:
            relations = self.graph.graph_relations.get(node["key"], {})
            match = True
            for rel_type, required in filters.items():
                # 检查是否存在指定类型的关系
                if required and not any(
                        r == rel_type
                        for r in relations.get("children", {}).values()
                ):
                    match = False
                    break
            if match:
                filtered.append(node)
        return filtered
    
    def _normalize_node_type(self, node_type: str) -> str: 
        node_type = node_type.lower().replace(" ", "_")
        return self.type_mapping.get(node_type, node_type)

    def traverse_relations(self, 
                         source_keys: List[str],
                         relation_types: List[str],
                         target_types: List[str],
                         max_depth: int = 1) -> Dict[str, List[Dict]]:
        # 关系类型标准化
        # std_relations = [self._normalize_relation(rt) for rt in relation_types]
        std_targets = [self._normalize_node_type(nt) for nt in target_types]
        # print(f"[DEBUG] 开始遍历 源节点: {source_keys}")
        # print(f"[DEBUG] 标准化前关系类型: {relation_types}")
        std_relations = [self._normalize_relation(rt) for rt in relation_types]
        # print(f"[DEBUG] 标准化后关系类型: {std_relations}")
        # 调试日志
        # self.logger.info(
        #     f"[TRAVERSE] 开始遍历关系类型: {relation_types}->{std_relations} "
        #     f"目标类型: {target_types}->{std_targets} 最大深度: {max_depth}"
        # )

        results = defaultdict(list)
        visited = set()

        def dfs(current_key: str, current_depth: int):
            # print(f"[DFS] 当前节点: {current_key}, 深度: {current_depth}")
            if current_key in visited or current_depth > max_depth:
                return
            visited.add(current_key)

            # 节点信息验证
            node = self._get_node_info(current_key)
            if not node:
                self.logger.warning(f"[TRAVERSE] 无效节点: {current_key}")
                return

            node_type = self._normalize_node_type(node["type"])
            # print("node_type"+node_type)
            # 结果收集
            if node_type in std_targets:
                results[node_type].append(node)
                self.logger.info(
                    f"[TRAVERSE] 发现目标节点 {current_key} "
                    f"类型: {node_type} 深度: {current_depth}"
                )

            # 关系遍历
            child_relations = self.graph.semantic_graph.graph_relations.get(current_key, {})
            # print(f"[DFS] 节点 {current_key} 的子关系: {child_relations}")
            for raw_rel, children in child_relations.get("children", {}).items():
                std_rel = self._normalize_relation(raw_rel)
                # print(f"[DFS] 处理关系 {raw_rel} -> {std_rel}")
                if std_rel not in std_relations:
                    continue

                for child_key in children:
                    dfs(child_key, current_depth + 1)

        # 执行遍历
        for key in source_keys:
            if not self._validate_node_key(key):
                continue
            dfs(key, 0)

        return dict(results)
    
    def _validate_node_key(self, key: str) -> bool:
        """验证节点键有效性"""
        # 处理空字符串
        if not key:
            self.logger.error("[VALIDATE] 跳过空节点键")
            return False
        # 处理字典类型
        if isinstance(key, dict):
            key = key.get("key", "")
        exists = any(n[0] == key for n in self.graph.semantic_graph.semantic_map.data)
        if not exists:
            self.logger.error(f"[VALIDATE] 节点不存在: {key}")
        return exists
    
    def _get_node_info(self, key: str) -> Optional[Dict]:
        """带缓存的节点信息获取"""
        if not hasattr(self, "_node_cache"):
            self._node_cache = {n["key"]: n for n in self.node_info}
        
        if info := self._node_cache.get(key):
            return info
            
        # 实时回退查询
        node = next((n for n in self.graph.semantic_graph.semantic_map.data if n[0] == key), None)
        if node:
            return {
                "key": node[0],
                "type": self._get_node_type(node[0]),
                "content": node[1],
                "paper_id": node[0].split("_")[0]
            }
        return None
    
    def _post_process_results(self, raw_results: Dict) -> Dict:
        """结果后处理"""
        processed = {}
        for result_type, nodes in raw_results.items():
            # 去重处理
            seen = set()
            unique_nodes = []
            for node in nodes:
                if node["key"] not in seen:
                    seen.add(node["key"])
                    unique_nodes.append(node)
            # 按论文ID排序
            processed[result_type] = sorted(
                unique_nodes, 
                key=lambda x: x["paper_id"]
            )
        return processed

    def _resolve_param_item(self, item, context):
        """返回基础类型或字典/列表结构"""
        # 处理变量引用
        if isinstance(item, str) and item.startswith("$"):
            referred = context.get(item[1:], [])
            return self._resolve_param_item(referred, context)

        # 处理列表
        if isinstance(item, list):
            return [self._resolve_param_item(elem, context) for elem in item]

        # 保留字典结构
        if isinstance(item, dict):
            return {k: self._resolve_param_item(v, context) for k, v in item.items()}

        # 基础类型直接返回
        return item

    def _resolve_params(self, params: Dict, context: Dict) -> Dict:
        """参数解析，确保类型安全"""
        resolved = {}
        for k, v in params.items():
            if k in ["source_keys", "node_keys"]:
                raw_items = self._resolve_param_item(v, context)
                resolved[k] = []
                # 递归展平嵌套结构并提取节点键
                self._extract_node_keys(raw_items, resolved[k])
                # 去重并过滤空值
                resolved[k] = [k for k in list(set(resolved[k])) if k]
            else:
                resolved_item = self._resolve_param_item(v, context)
                # 处理数值参数
                if k in ["top_k", "max_depth", "depth"]:
                    resolved[k] = int(resolved_item) if resolved_item else 0
                # 处理实体类型
                elif k == "entity_type":
                    resolved[k] = resolved_item[0] if isinstance(resolved_item, list) else str(resolved_item)
                # 处理关键词
                elif k == "keywords":
                    resolved[k] = " ".join(resolved_item) if isinstance(resolved_item, list) else str(resolved_item)
                else:
                    resolved[k] = resolved_item
        return resolved

    def _extract_node_keys(self, item, resolved_list):
        """递归提取节点键"""
        if isinstance(item, list):
            for elem in item:
                self._extract_node_keys(elem, resolved_list)
        elif isinstance(item, dict):
            # 优先提取 key 字段
            key = item.get("key", "")
            if key:
                resolved_list.append(key)
            else:
                # 递归处理所有值
                for value in item.values():
                    self._extract_node_keys(value, resolved_list)
        elif isinstance(item, str):
            resolved_list.append(item)
    
    def generate_report(self, context: Dict, query: str) -> str:
        """生成最终分析报告"""
        prompt = f"""
        根据以下分析结果生成中文报告：
        原始查询：{query}
        上下文数据：{json.dumps(context, indent=2)}

        """

        report = self._call_llm(prompt)
        self.logger.info("成功生成分析报告")
        return report

    def semantic_traversal(self, source_keys: List[str], top_k: int = 3, depth: int = 2) -> Dict[str, List]:
        """语义遍历"""
        # 参数预处理
        def validate_key(key):
            """严格校验键格式并转换为字符串"""
            if isinstance(key, dict):
                return str(key.get("key", ""))
            return str(key)

        # 使用正则表达式验证arxiv ID格式
        arxiv_id_pattern = re.compile(r"^\d+\.\d+[vV]\d+_.+")
        validated_keys = [
            key_str
            for key in source_keys
            if (key_str := validate_key(key))
               and arxiv_id_pattern.match(key_str)
               and any(n["key"] == key_str for n in self.node_info)
        ]

        self.logger.info(f"[DEBUG] 语义遍历有效节点键 ({len(validated_keys)}): {validated_keys[:3]}...")

        results = defaultdict(list)
        visited = set()

        def dfs(current_key: str, current_depth: int):
            # 类型安全断言
            if not isinstance(current_key, str):
                raise TypeError(f"非法节点键类型: {type(current_key)} -> {current_key}")

            if current_depth > depth or current_key in visited:
                return
            visited.add(current_key)

            try:
                # 获取当前节点的索引
                node_index = next(i for i, n in enumerate(self.node_info) if n["key"] == current_key)
                query_embedding = self.node_embeddings[node_index]

                # 搜索相似节点
                distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k + 1)

                for idx in indices[0]:
                    if 0 <= idx < len(self.node_info):
                        neighbor = self.node_info[idx]
                        if neighbor["key"] != current_key:
                            results[current_key].append(neighbor)
                            dfs(neighbor["key"], current_depth + 1)
            except StopIteration:
                self.logger.warning(f"[WARN] 节点 {current_key} 不存在于索引中")
            except Exception as e:
                self.logger.error(f"[ERROR] 处理节点 {current_key} 时出错: {str(e)}")
                traceback.print_exc()

        for key in validated_keys:
            dfs(key, 0)

        total_nodes = sum(len(v) for v in results.values())
        self.logger.info(f"semantic_traversal 找到 {total_nodes} 个相似节点")
        return dict(results)
  
    def refine_knowledge(self, node_keys: List[Union[str, dict]]) -> List[str]:
        """知识提炼方法"""
        # 参数预处理和类型校验
        validated_keys = []
        for item in node_keys:
            try:
                if isinstance(item, dict):
                    key = str(item.get("key", ""))
                elif isinstance(item, str):
                    key = item.strip()
                else:
                    self.logger.warning(f"非法参数类型: {type(item)}")
                    continue
                
                # 键校验
                if not re.match(r"^[\w\.\-:_]+$", key):
                    self.logger.warning(f"非法键格式: {key}")
                    continue
                    
                validated_keys.append(key)
            except Exception as e:
                self.logger.error(f"参数处理错误: {str(e)}")
                continue

        new_nodes = []
        for key in validated_keys:
            self.logger.info(f"\n处理节点: {key}")
            try:
                # 清理旧摘要并保留关系结构
                self._clean_old_summaries(key)
                
                # 获取并校验节点内容
                node_content = self._get_node_content(key)
                if not node_content or not isinstance(node_content, dict):
                    self.logger.warning("无效节点内容，跳过")
                    continue
                    
                # 生成摘要
                summary = self._generate_summary(node_content)
                if not summary or len(summary) < 50:
                    self.logger.warning("摘要生成失败或过短")
                    continue
                    
                # 创建带时间戳的摘要节点
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                summary_key = f"{key}_summary_{timestamp}"
                
                # 关系绑定
                self.graph.semantic_graph.add_node(
                    summary_key,
                    {"type": "summary", "content": summary},
                    parent_keys=[key],
                    parent_relation="has_summary"
                )
                parent_relations = self.graph.semantic_graph.graph_relations.get(key, {}).get("children", {})
                if "has_summary" not in parent_relations:
                    self.logger.error(f"关系建立失败！父节点 {key} 无 has_summary 关系")
                else:
                    self.logger.info(f"成功建立关系: {key} → {summary_key}")
                
                # 立即更新索引
                self._update_index_with_new_node(summary_key, summary)
                
                new_nodes.append(summary_key)
                self.logger.info(f"创建摘要节点: {summary_key}")
                self.logger.info(f"content:{summary}")
            except Exception as e:
                self.logger.error(f"处理失败: {str(e)}")
                continue

        return new_nodes

    def _update_index_with_new_node(self, key: str, content: dict):
        """动态更新索引"""
        text = self._generate_node_text(key, content)
        embedding = self.embedder.encode(text)
        
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
            
        # 更新索引数据结构
        self.node_embeddings = np.vstack([self.node_embeddings, embedding])
        self.node_info.append({
            "key": key,
            "type": "summary",
            "content": content,
            "paper_id": key.split("_")[0] if '_' in key else 'unknown'
        })
        
        # 重建HNSW索引
        self.index.add(np.array([embedding]))
        self.logger.info("动态更新索引成功")
    def _get_node_content(self, node_key: str) -> dict:
        """安全获取节点内容"""
        try:
            # 在semantic_map中精确查找
            node = next(n for n in self.graph.semantic_graph.semantic_map.data 
                        if n[0] == node_key)
                        
            # 内容规范化
            if isinstance(node[1], dict):
                return node[1]
            else:
                return {"raw_content": node[1]}  # 包装非字典内容
        except StopIteration:
            print(f"🔍 节点不存在: {node_key}")
            return {}
        except Exception as e:
            print(f"获取内容失败: {str(e)}")
            return {}


    def _generate_summary(self, node_content: dict) -> Optional[dict]:
        # 内容提取
        content_type = node_content.get("type", "unknown")
        raw_text = node_content.get("text") or str(node_content.get("raw_content", ""))

        # 动态长度校验
        min_length = {
            "paper_root": 30,
            "chapter": 50,
            "paragraph": 80,
            "reference": 20,
            "summary": 0  # 避免重复生成
        }.get(content_type, 50)

        if len(raw_text) < min_length:
            self.logger.warning(f"内容过短 ({len(raw_text)} < {min_length}): {raw_text[:50]}...")
            return None

        # 构造提示词，明确要求输出为JSON格式
        prompt = f"""
        请根据以下{self._type_mapping_zh(content_type)}内容生成结构化摘要，输出格式为JSON：
        {{
            "核心观点": "用20 - 50字概括核心观点",
            "关键方法/数据": ["列出3 - 5项关键方法或数据"],
            "研究价值与局限": "简要阐述研究价值与局限，各1 - 2项"
        }}

        原始内容：
        {raw_text[:3000]}
        """

        # 调用LLM生成摘要
        return self._call_llm(prompt)
    def _type_mapping_zh(self, en_type: str) -> str:
            """类型中英映射"""
            mapping = {
                "paper_root": "论文元数据",
                "chapter": "章节",
                "paragraph": "段落",
                "reference": "参考文献",
                "table": "表格",
                "image": "图片",
                "summary": "摘要"
            }
            return mapping.get(en_type, "内容")
    def _has_valid_summary(self, source_key: str, summary_key: str) -> bool:
        """添加类型校验"""
        # 校验键类型
        if not isinstance(source_key, str) or not isinstance(summary_key, str):
            return False

        # 存在性检查
        if not any(n[0] == summary_key for n in self.graph.semantic_graph.semantic_map.data):
            return False

        # 关系检查（添加异常捕获）
        try:
            relations = self.graph.semantic_graph.graph_relations.get(str(source_key), {})
            return any(
                rel == "has_summary" and summary_key in children
                for rel, children in relations.get("children", {}).items()
            )
        except TypeError:  # 处理不可哈希类型
            return False

    def _clean_old_summaries(self, source_key: str):
        """清理旧摘要节点"""
        # 查找所有历史摘要
        old_summaries = [
            n[0] for n in self.graph.semantic_graph.semantic_map.data
            if n[0].startswith(f"{source_key}_summary")
        ]
        
        # 删除节点和关系
        for skey in old_summaries:
            try:
                self.graph.semantic_graph.delete_node(skey)
                print(f"♻️ 已清理旧摘要: {skey}")
            except KeyError:
                continue
                
        # 更新内存索引
        self.node_info = [n for n in self.node_info if n["key"] not in old_summaries]
        self.node_embeddings = np.array([
            emb for emb, n in zip(self.node_embeddings, self.node_info)
        ])

    def generate_advanced_plan(self, query: str) -> Dict:
        schema_description = """
        节点类型 (Node Types):
        1. paper_root: 论文根节点（包含标题、作者）
        2. abstract: 论文摘要
        3. chapter: 论文章节（包含段落和图片）
        4. paragraph: 章节段落
        5. reference: 参考文献
        6. table: 表格数据
        7. image: 图表信息

        关系类型 (Relation Types):
        1. has_abstract: 论文→摘要
        2. has_chapter: 论文→章节
        3. has_paragraph: 章节→段落
        4. has_image: 章节→图片
        5. has_table: 论文→表格
        6. references: 论文→参考文献
        7. similar: 节点之间的语义相似关系
        8. has_summary: 通过refine_knowledge方法生成的摘要关系（仅当调用refine_knowledge后存在）
        """

        function_descriptions = """
        可用函数：
        1. search_units(
            entity_type: 实体类型,
            [attribute_filters]: 属性过滤（字典）,
            [relation_filters]: 关系过滤（字典）,
            [keywords]: 关键词语义搜索,
            [top_k]: 返回数量（默认15）
        ) -> 匹配节点列表

        2. traverse_relations(
            source_keys: 起始节点列表,
            relation_types: 要遍历的关系类型列表,
            target_types: 目标节点类型列表,
            [max_depth]: 最大遍历深度（默认2）
        ) -> 按类型分组的节点字典

        3. semantic_traversal(
            source_keys: 起始节点列表,
            [top_k]: 每层扩展数量（默认3）,
            [depth]: 遍历深度（默认2）
        ) -> 相似节点字典（key为源节点）

        4. refine_knowledge(
            node_keys: 要提炼的节点列表
        ) -> 新生成的摘要节点列表（会自动创建has_summary关系）
        """

        full_example = """
        完整示例：
        用户查询："找关于深度学习的论文，发现相关研究并生成摘要"

        响应计划：
        {
            "steps": [
                {
                    "function": "search_units",
                    "params": {
                        "entity_type": "paper_root",
                        "keywords": "深度学习 神经网络",
                        "top_k": 15
                    },
                    "output_key": "core_papers"
                },
                {
                    "function": "traverse_relations",
                    "params": {
                        "source_keys": "$core_papers",
                        "relation_types": ["references"],
                        "target_types": ["reference"],
                        "max_depth": 1
                    },
                    "output_key": "related_refs"
                },
                {
                    "function": "semantic_traversal",
                    "params": {
                        "source_keys": "$core_papers",
                        "top_k": 10,
                        "depth": 1
                    },
                    "output_key": "similar_studies"
                },
                {
                    "function": "refine_knowledge",
                    "params": {
                        "node_keys": "$core_papers"
                    },
                    "output_key": "paper_summaries"
                }
            ]
        }
        """

        prompt = f'''
        # Task
        Generate execution plan strictly following these rules:

        ## Language Rules
        1. Accept both Chinese and English queries
        2. Always respond with valid JSON format
        3. Never include explanations or comments

        ## Knowledge Graph Structure
        {schema_description}

        ## Function Examples
        {function_descriptions}

        ## User Query
        {query}

        ## Response Example
        {full_example}
    '''
        try:
            response = self._call_llm(prompt).strip()
            plan = self._parse_plan(response)
            self.logger.info(f"成功为查询 '{query}' 生成高级计划，包含 {len(plan.get('steps', []))} 个步骤")
            return plan
        except Exception as e:
            self.logger.error(f"生成计划失败: {str(e)}")
            return {"steps": []}

    def _parse_plan(self, response: str) -> Dict:
        """解析方法"""
        try:
            cleaned = re.sub(r'[“”]', '"', response)  # 统一引号
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)  # 移除注释
            cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', cleaned)  # 清理转义

            # 尝试多种方式提取JSON
            for candidate in re.findall(r'\{.*\}', cleaned, re.DOTALL):
                try:
                    plan = json.loads(candidate)
                    if "steps" in plan and isinstance(plan["steps"], list):
                        self.logger.info("成功解析查询计划")
                        return plan
                except:
                    continue
            raise ValueError("No valid JSON found")

        except Exception as e:
            self.logger.error(f"解析失败，原始响应：\n{response[:300]}...")
            raise ValueError(f"计划解析失败: {str(e)}")

    def _print_step_result(self, step: Dict, result: Any):
        """安全打印结果"""
        self.logger.info(f"🔍 {step['output_key']} ({step['function']})")

        if isinstance(result, dict):
            total_nodes = sum(len(v) for v in result.values())
            self.logger.info(f"发现 {total_nodes} 个节点")
            for k, v in result.items():
                self.logger.info(f"  {k}:")
                for i, item in enumerate(v[:3], 1):
                    # 类型安全检查
                    if isinstance(item, dict):
                        self.logger.info(f"    {i}. [{item.get('type', '?')}] {item.get('key', '未知键')}")
                        self.logger.info(f"      内容: {str(item.get('content', '无内容'))[:80]}...")
                    else:
                        self.logger.info(f"    {i}. [无效格式] {str(item)[:80]}")
        elif isinstance(result, list):
            self.logger.info(f"获得 {len(result)} 个结果")
            for i, item in enumerate(result[:5], 1):
                if isinstance(item, dict):
                    self.logger.info(f"  {i}. [{item.get('type', '?')}] {item.get('key', '未知键')}")
                else:
                    self.logger.info(f"  {i}. [无效格式] {str(item)[:80]}")
        else:
            self.logger.info("未知结果格式")
        self.logger.info("━" * 50)

    def execute_advanced_plan(self, plan: Dict) -> Dict:
        """计划执行方法"""
        context = {}
        for step in plan.get("steps", []):
            try:
                self.logger.info(f"\n▶️ 开始执行步骤：{step['output_key']}")
                self.logger.info(f"  原始参数: {json.dumps(step['params'], indent=2)}")

                # 参数解析
                params = self._resolve_params(step["params"], context)
                self.logger.info(f"  解析后参数类型: { {k: type(v) for k, v in params.items()} }")
                self.logger.info(f"  解析后参数值样例: { {k: v[:3] if isinstance(v, list) else v for k, v in params.items()} }")

                # 执行函数
                if step["function"] == "traverse_relations":
                    self.logger.info("执行前source_keys样本:", params["source_keys"][:2] if params["source_keys"] else "空列表")

                if step["function"] == "search_units":
                    result = self.search_units(**params)
                elif step["function"] == "traverse_relations":
                    result = self.traverse_relations(**params)
                elif step["function"] == "semantic_traversal":
                    result = self.semantic_traversal(**params)
                elif step["function"] == "refine_knowledge":
                    result = self.refine_knowledge(**params)
                else:
                    raise ValueError(f"未知函数: {step['function']}")

                # 打印结果
                self._print_step_result(step, result)
                context[step["output_key"]] = result

            except Exception as e:
                self.logger.error(f"❌ 步骤执行失败: {str(e)}")
                context[step["output_key"]] = []
                # 部分继续执行
                if isinstance(e, KeyError):
                    self.logger.warning("⚠️ 忽略该步骤继续执行...")
                else:
                    import traceback
                    traceback.print_exc()
                    break  # 严重错误终止

        self.logger.info("高级计划执行完成")
        return context
    
    ##############################测试方法################################
    def validate_traverse_relations(self,
                                  source_key: str,
                                  relation_type: str,
                                  target_type: str,
                                  expected_min: int=1) -> Dict:
        """
        验证方法
        
        参数：
        - source_key: 源节点键（如 "2404.13501v1_title_authors"）
        - relation_type: 预期的关系类型（如 "has_abstract"）
        - target_type: 预期的目标节点类型（如 "abstract"）
        - expected_min: 期望找到的最小节点数
        
        返回：
        {
            "total_nodes": 总数,
            "valid_nodes": 有效节点列表,
            "invalid_nodes": 无效节点列表,
            "type_mismatch": 类型不匹配列表,
            "relation_issues": 关系问题列表,
            "expectation_met": 是否满足期望节点数,
            "validation_details": 详细验证数据
        }
        """
        # ====================== 初始化阶段 ======================
        self.logger.info(
            f"[VALIDATE] 启动验证 | 源: {source_key} "
            f"关系: {relation_type}->{self._normalize_relation(relation_type)} "
            f"目标类型: {target_type}->{self._normalize_node_type(target_type)}"
        )

        # ====================== 执行遍历 ======================
        traversal_results = self.traverse_relations(
            source_keys=[source_key],
            relation_types=[relation_type],
            target_types=[target_type],
            max_depth=2
        )
        found_nodes = traversal_results.get(self._normalize_node_type(target_type), [])
        
        result = {
            "total_nodes": len(found_nodes),
            "valid_nodes": [],
            "invalid_nodes": [],
            "type_mismatch": [],
            "relation_issues": [],
            "expectation_met": len(found_nodes) >= expected_min,
            "validation_details": []
        }
        if not self._validate_node_exists(source_key):
            self.logger.error(f"源节点不存在: {source_key}")
            return {
                "error": f"Source node {source_key} not found",
                **result
            }
        # ====================== 节点验证循环 ======================
        for idx, node in enumerate(found_nodes, 1):
            node_key = node["key"]
            entry = {
                "node": node_key,
                "content": node.get("content"),
                "validation_steps": {}
            }

            try:
                # ------------------- 节点存在性验证 -------------------
                if not self._validate_node_exists(node_key):
                    result["invalid_nodes"].append(node_key)
                    entry["validation_steps"]["existence"] = False
                    continue
                entry["validation_steps"]["existence"] = True

                # ------------------- 类型匹配验证 -------------------
                actual_type = self._normalize_node_type(node["type"])
                expected_type = self._normalize_node_type(target_type)
                type_valid = actual_type == expected_type
                
                entry["validation_steps"].update({
                    "type_check": {
                        "expected": expected_type,
                        "actual": actual_type,
                        "valid": type_valid
                    }
                })
                
                if not type_valid:
                    result["type_mismatch"].append(node_key)
                    self.logger.warning(
                        f"[VALIDATE] 类型不匹配 | 节点: {node_key} "
                        f"预期: {expected_type} 实际: {actual_type}"
                    )
                    continue

                # ------------------- 关系路径验证 -------------------
                path_valid, path_details = self._validate_relation_path_with_details(
                    source_key, node_key, relation_type
                )
                entry["validation_steps"]["path_check"] = {
                    "valid": path_valid,
                    "path": path_details
                }

                if not path_valid:
                    result["relation_issues"].append({
                        "node": node_key,
                        "path": path_details
                    })
                    self.logger.warning(
                        f"[VALIDATE] 路径验证失败 | 节点: {node_key}\n"
                        f"实际路径: {json.dumps(path_details, indent=2)}"
                    )
                    continue

                # ------------------- 最终有效节点 -------------------
                result["valid_nodes"].append(node_key)
                entry["validation_steps"]["final_result"] = "VALID"
                
            except Exception as e:
                entry["validation_steps"]["error"] = str(e)
                self.logger.error(f"验证节点 {node_key} 时出错: {str(e)}")
            
            result["validation_details"].append(entry)

        # ====================== 生成最终报告 ======================
        validation_rate = len(result["valid_nodes"]) / result["total_nodes"] if result["total_nodes"] > 0 else 0
        self.logger.info(
            f"[VALIDATE] 验证完成 | 有效率: {validation_rate:.2%} "
            f"有效/总数: {len(result['valid_nodes'])}/{result['total_nodes']} "
            f"达标: {result['expectation_met']}"
        )
        
        return result
    def _get_child_relations(self, node_key: str) -> Dict[str, List]:
        """获取子节点关系"""
        return self.graph.semantic_graph.graph_relations.get(
            node_key, {}
        ).get("children", {})
    
    def _validate_relation_path_with_details(self, source: str, target: str, expected_rel: str) -> tuple[bool, List]:
        """路径验证"""
        expected_rel = self._normalize_relation(expected_rel)
        valid_paths = []
        
        # 正向路径搜索（从source到target）
        def dfs(current: str, path: List[Dict], found_rel: bool):
            if current == target:
                if found_rel:
                    valid_paths.append(path.copy())
                return
            
            for raw_rel, children in self._get_child_relations(current).items():
                for child in children:
                    if child in [p["to"] for p in path]:  # 避免循环
                        continue
                        
                    # 标准化当前关系
                    norm_rel = self._normalize_relation(raw_rel)
                    new_found = found_rel or (norm_rel == expected_rel)
                    
                    new_step = {
                        "from": current,
                        "to": child,
                        "relation": norm_rel,
                        "original_relation": raw_rel
                    }
                    
                    dfs(child, path + [new_step], new_found)

        dfs(source, [], False)
        return (len(valid_paths) > 0, valid_paths)

    def _validate_node_exists(self, node_key: str) -> bool:
        """节点存在性检查"""
        return any(
            node[0] == node_key
            for node in self.graph.semantic_graph.semantic_map.data
        )

    def _get_relation_types(self, child: str, parent: str) -> List[str]:
        """获取两个节点间的所有关系类型"""
        return [
            rel for rel, children 
            in self.graph.semantic_graph.graph_relations.get(parent, {}).get("children", {}).items()
            if child in children
        ]

    def _get_relation_path(self, source: str, target: str) -> List[Dict]:
        """获取详细关系路径（用于日志）"""
        path = []
        current = target
        while current != source:
            parents = self.graph.semantic_graph.graph_relations.get(current, {}).get("parents", {})
            if not parents:
                break
            # 取第一个父节点
            for rel, parent_list in parents.items():
                if parent_list:
                    current = parent_list[0]
                    path.append({
                        "from": current,
                        "to": path[-1]["from"] if path else target,
                        "relation": rel
                    })
                    break
        return path[::-1]  # 反向显示路径

    def validate_refine_knowledge(self, 
                                node_key: str, 
                                min_summary_length: int = 50) -> Dict:
        """
        验证知识提炼功能
        :param node_key: 要测试的节点键（如"2301.12345v1_chapter_1"）
        :param min_summary_length: 摘要最小长度要求
        :return: 包含验证结果的字典
        """
        result = {
            "original_exists": False,
            "new_nodes": [],
            "relations_added": False,
            "index_updated": False,
            "content_valid": False,
            "errors": []
        }

        try:
            # 检查原始节点是否存在
            original_node = next((n for n in self.graph.semantic_graph.semantic_map.data if n[0] == node_key), None)
            if not original_node:
                result["errors"].append(f"原始节点 {node_key} 不存在")
                return result
            result["original_exists"] = True

            # 执行知识提炼
            new_nodes = self.refine_knowledge([node_key])
            result["new_nodes"] = new_nodes

            # 验证新节点
            if not new_nodes:
                result["errors"].append("未生成任何摘要节点")
                return result

            summary_key = new_nodes[0]
            summary_node = next((n for n in self.graph.semantic_graph.semantic_map.data if n[0] == summary_key), None)

            # 存在性检查
            if not summary_node:
                result["errors"].append(f"摘要节点 {summary_key} 未插入")
                return result

            # 关系验证
            relations = self.graph.semantic_graph.graph_relations.get(node_key, {})
            result["relations_added"] = "has_summary" in relations.get("children", {})
            if not result["relations_added"]:
                result["errors"].append(f"缺失 {node_key} -> {summary_key} 的 has_summary 关系")

            # 索引验证
            result["index_updated"] = any(n["key"] == summary_key for n in self.node_info)
            if not result["index_updated"]:
                result["errors"].append("摘要节点未加入索引")

            # 内容验证
            summary_content = str(summary_node[1])
            result["content_valid"] = len(summary_content) >= min_summary_length
            if not result["content_valid"]:
                result["errors"].append(f"摘要内容过短 ({len(summary_content)}<{min_summary_length})")

        except Exception as e:
            result["errors"].append(f"验证异常: {str(e)}")
            traceback.print_exc()

        return result
    def validate_relation_traversal_extended(self):
        """扩展关系遍历测试"""
        test_cases = [
            {
                "source_key": "2404.13501v1_title_authors",
                "relation_type": "has_chapter",
                "target_type": "chapter",
                "expected_min": 3
            },
            {
                "source_key": "2404.13501v1_chapter_1",
                "relation_type": "has_paragraph",
                "target_type": "paragraph",
                "expected_min": 5
            }
        ]
        
        for case in test_cases:
            result = self.validate_traverse_relations(
                case["source_key"],
                case["relation_type"],
                case["target_type"],
                case["expected_min"]
            )
            print(f"\n测试案例 {case['relation_type']}:")
            print(json.dumps(result, indent=2))

    def validate_summary_generation_diverse(self):
        """多样化摘要生成测试"""
        test_nodes = [
            "2404.13501v1_chapter_0_para_0",
            "2404.13501v1_chapter_0_para_1"
        ]
        
        for node_key in test_nodes:
            result = self.validate_refine_knowledge(node_key)
            print(f"\n测试节点 {node_key}:")
            print(json.dumps(result, indent=2))


    def visualize_graph(self):
        """
        将内部的 graph 进行图示化
        """
        # 创建一个有向图对象
        G = nx.DiGraph()

        # 添加节点
        for node in self.graph.semantic_graph.semantic_map.data:
            node_key = node[0]
            node_type = self._get_node_type(node_key)
            G.add_node(node_key, type=node_type)

        # 添加边
        for node_key, relations in self.graph.semantic_graph.graph_relations.items():
            children = relations.get("children", {})
            for rel_type, child_nodes in children.items():
                std_rel = self._normalize_relation(rel_type)
                for child_node in child_nodes:
                    G.add_edge(node_key, child_node, relation=std_rel)

            for dst, rel in relations.get("links", {}).items():
                normalized_rel = self._normalize_relation(rel)
                G.add_edge(node_key, dst,relation=normalized_rel)

        # 设置节点颜色和大小
        node_colors = []
        node_sizes = []
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if node_type == 'paper_root':
                node_colors.append('lightblue')
                node_sizes.append(800)
            elif node_type == 'abstract':
                node_colors.append('lightgreen')
                node_sizes.append(600)
            elif node_type == 'chapter':
                node_colors.append('orange')
                node_sizes.append(700)
            elif node_type == 'paragraph':
                node_colors.append('pink')
                node_sizes.append(500)
            elif node_type == 'reference':
                node_colors.append('purple')
                node_sizes.append(500)
            elif node_type == 'table':
                node_colors.append('yellow')
                node_sizes.append(500)
            elif node_type == 'image':
                node_colors.append('gray')
                node_sizes.append(500)
            elif node_type == 'summary':
                node_colors.append('brown')
                node_sizes.append(500)
            else:
                node_colors.append('lightgray')
                node_sizes.append(500)

        # 绘制图
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # 设置图形标题
        plt.title('Arxiv Semantic Graph Visualization')

        # 显示图形
        plt.show()

    
def create_simple_graph():
    graph = ArxivSemanticGraph()
    sample_data = {
    "paper_id": "2404.13501v1",
    "title": "Research on AI in Healthcare",
    "authors": ["Alice Smith", "Bob Johnson"],
    "abstract": "Large language model (LLM) based agents have recently attracted much attention from the research and industry communities. Compared with original LLMs, LLM-based agents are featured in their self-evolving capability, which is the basis for solving real-world problems that need long-term and complex agent-environment interactions. The key component to support agent-environment interactions is the memory of the agents. While previous studies have proposed many promising memory mechanisms, they are scattered in different papers, and there lacks a systematical review to summarize and compare these works from a holistic perspective, failing to abstract common and effective designing patterns for inspiring future studies. To bridge this gap, in this paper, we propose a comprehensive survey on the memory mechanism of LLM-based agents. In specific, we first discuss “what is” and “why do we need” the memory in LLM-based agents. Then, we systematically review previous studies on how to design and evaluate the memory module. In addition, we also present many agent applications, where the memory module plays an important role. At last, we analyze the limitations of existing work and show important future directions. To keep up with the latest advances in this field, we create a repository at https://github.com/nuster1128/LLM_Agent_Memory_Survey",
    "chapters": [
        {
            "title": "Introduction",
            "paragraphs": [
                "Recently, large language models (LLMs) have achieved remarkable success in a large number of domains, ranging from artificial intelligence and software engineering to education and social science [1, 2, 3]. Original LLMs usually accomplish different tasks without interacting with environments. However, to achieve the final goal of artificial general intelligence (AGI), intelligent machines should be able to improve themselves by autonomously exploring and learning from the real world. For example, if a trip-planning agent intends to book a ticket, it should send an order request to the ticket website, and observe the response before taking the next action. A personal assistant agent should adjust its behaviors according to the user’s feedback, providing personalized responses to improve user’s satisfaction. To further push the boundary of LLMs towards AGI, recent years have witnessed a large number of studies on LLM-based agents [3, 4], where the key is to equip LLMs with additional modules to enhance their self-evolving capability in real-world environments.",
                "Among all the added modules, memory is a key component that differentiates the agents from original LLMs, making an agent truly an agent (see Figure 1). It plays an extremely important role in determining how the agent accumulates knowledge, processes historical experience, retrieves informative knowledge to support its actions, and so on. Around the memory module, people have devoted much effort to designing its information sources, storage forms, and operation mechanisms. For example, Shinn et al. [5] incorporate both in-trial and cross-trial information to build the memory module for enhancing the agent’s reasoning capability. Zhong et al. [6] store memory information in the form of natural languages, which is explainable and friendly to the users. Modarressi et al. [7] design both memory reading and writing operations to interact with environments for task solving."
            ],
            "images": []
        }   
    ],
    "references": [
        "Smith, A. (2023). Recent Advances in Medical AI. Journal of Healthcare AI.",
        "Johnson, B. (2022). Deep Learning for Image Analysis. Cambridge University Press."
    ]
    }

    graph.insert(**sample_data)
    return graph

# 初始化测试数据
# graph = parse_local_paper()
# 如果进行图示化,使用该测试数据
# agent = ArxivAgent(create_simple_graph())

# 测试关系遍历功能
# traverse_result = agent.validate_traverse_relations(
#     source_key="2404.13501v1_title_authors",
#     relation_type="has_abstract",
#     target_type="abstract"
# )
# print("关系遍历验证结果：")
# print(json.dumps(traverse_result, indent=2, ensure_ascii=False))
# agent.validate_relation_traversal_extended()
#测试知识提炼功能
# refine_result = agent.validate_refine_knowledge(
#     node_key="2404.13501v1_abstract"
# )
# print("\n知识提炼验证结果：")
# print(json.dumps(refine_result, indent=2, ensure_ascii=False))
# agent.validate_summary_generation_diverse()
# agent.visualize_graph()
#################以上是测试traverse_relation及refine_knowledge方法的代码###################################################


if __name__ == "__main__":
    graph = parse_local_paper()
    agent = ArxivAgent(graph)

    while True:
        try:
            query = input("\n请输入查询（输入q退出）: ")
            if query.lower() == 'q':
                break

            agent.logger.info("\n生成查询计划中...")
            plan = agent.generate_advanced_plan(query)
            agent.logger.info("生成的计划: " + json.dumps(plan, indent=2, ensure_ascii=False))

            agent.logger.info("\n执行查询计划...")
            results = agent.execute_advanced_plan(plan)

            agent.logger.info("\n生成分析报告...")
            report = agent.generate_report(results, query)
            agent.logger.info("\n=== 最终报告 ===")
            agent.logger.info(report)

            with open(f"report_{int(time.time())}.txt", "w") as f:
                f.write(report)

        except KeyboardInterrupt:
            agent.logger.info("\n操作已取消")
            continue
        except Exception as e:
            agent.logger.error(f"发生错误：{str(e)}")