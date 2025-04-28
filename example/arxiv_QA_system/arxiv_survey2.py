import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
intel_omp_path = r'C:\Program Files\Dell\DTP\IPDT'
os.environ['PATH'] = intel_omp_path + os.pathsep + os.environ['PATH']

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
from typing import List, Dict, Any,Optional
import re


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
                    self.semantic_graph.link_nodes(prev_chapter_key, chapter_key, "preceded_by")

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
                            self.semantic_graph.link_nodes(prev_para_key, para_key, "preceded_by")

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
                relations = ["has_reference"]
                if similar_root_key:
                    parents.append(similar_root_key)
                    relations.append("is_referenced_by")

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
TIMEOUT = 20  # 超时时间延长到20秒

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
    """改进后的解析函数，增强错误处理"""
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


# # 下载论文
# graph = parse_local_paper()

# # 执行查询
# results = graph.query("machine learning", k=100)
# for paper in results:
#     show_paper_structure(paper)
#     print("-" * 80)

# # 保存图谱
# graph.export_preferences("my_preferences.json")


import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import requests


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-c4e7f78ec6fe48379839541971a2bfc7"
DEEPSEEK_MODEL = "deepseek-chat"


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


    def _build_hnsw_index(self):
        """构建HNSW索引并记录节点元数据"""
        print("Building HNSW index...")
        
        # 重置之前的数据
        self.node_embeddings = []
        self.node_info = []
        
        # HNSW参数配置
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexHNSWFlat(dim, 32)  # 每次创建新索引
        
        # 遍历所有节点生成嵌入
        valid_nodes = 0
        for node in self.graph.semantic_graph.semantic_map.data:
            try:
                node_key = node[0]
                node_content = node[1]
                
                # 生成文本描述
                text = self._generate_node_text(node_key, node_content)
                if not text.strip():
                    continue
                    
                # 生成嵌入并确保维度正确
                embedding = self.embedder.encode(text)
                if embedding.ndim == 2:
                    embedding = embedding.squeeze(0)  # 从(1,384)转为(384,)
                    
                self.node_embeddings.append(embedding)
                self.node_info.append({
                    "key": node_key,
                    "type": self._get_node_type(node_key),
                    "content": node_content,
                    "paper_id": node_key.split("_")[0]
                })
                valid_nodes += 1
            except Exception as e:
                print(f"处理节点 {node_key} 时出错: {str(e)}")
                continue

        # 转换为numpy数组并检查维度
        if len(self.node_embeddings) == 0:
            raise ValueError("没有有效的节点数据可供索引")
            
        self.node_embeddings = np.array(self.node_embeddings)
        print(f"嵌入矩阵形状: {self.node_embeddings.shape}")
        
        # 验证维度一致性
        if self.node_embeddings.shape[0] != len(self.node_info):
            raise ValueError("嵌入向量数量与节点信息数量不一致")
        
        # 添加到索引
        self.index.add(self.node_embeddings)
        print(f"索引构建完成，有效节点数: {valid_nodes}")

    

    def _get_node_type(self, key: str) -> str:
        """根据节点键识别类型"""
        if "_title_authors" in key:
            return "paper_root"
        elif "_abstract" in key:
            return "abstract"
        elif "_chapter_" in key:
            return "chapter"
        elif "_ref_" in key:
            return "reference"
        elif "_img_" in key:
            return "image"
        elif "_table_" in key:
            return "table"
        elif "_para_" in key:
            return "paragraph"
        return "unknown"

    # def generate_query_plan(self, query: str) -> List[Dict]:
    #     """使用LLM生成查询计划"""
    #     schema_description = """
    #     实体类型 (Entity Types):
    #     - Paper: 学术论文
    #     - Abstract: 论文摘要
    #     - Chapter: 论文章节
    #     - Paragraph: 段落
    #     - Reference: 参考文献
    #     - Table: 表格

    #     关键属性 (Key Attributes):
    #     - Paper: title, authors, publication_date
    #     - Abstract: content
    #     - Chapter: title, section_number, parent_chapter
    #     - Paragraph: text, position_in_chapter
    #     - Reference: title, authors, publication_year
    #     - Table: columns, rows

    #     主要关系类型 (Main Relation Types):
    #     - Paper has_abstract -> Abstract
    #     - Paper contains -> Chapter
    #     - Chapter contains -> Paragraph
    #     - Paragraph preceded_by -> Paragraph (顺序关系)
    #     - Chapter similar_to -> Chapter (语义相似)
    #     - Paper references -> Reference
    #     - Paper has_table -> Table
    #     """

    #     example_query_plan = """
    #     [
    #         {
    #             "step": 1,
    #             "target": "Paper",
    #             "constraints": {
    #                 "semantic_query": "agent memory mechanisms",
    #                 "top_k": 5
    #             },
    #             "input": [],
    #             "query_scope": {
    #                 "node_type": "Paper",
    #                 "query_type": "semantic_query",
    #                 "has_input": false
    #             }
    #         },
    #         {
    #             "step": 2,
    #             "target": "Chapter",
    #             "constraints": {
    #                 "selected_papers": ["STEP_1_PAPER_IDS"],
    #                 "semantic_query": "memory architecture"
    #             },
    #             "input": ["step_1"],
    #             "query_scope": {
    #                 "node_type": "Chapter",
    #                 "query_type": "combined_query",
    #                 "has_input": true
    #             }
    #         },
    #         {
    #             "step": 3,
    #             "target": "Paragraph",
    #             "constraints": {
    #                 "selected_chapters": ["STEP_2_CHAPTER_IDS"],
    #                 "semantic_query": "attention mechanism"
    #             },
    #             "input": ["step_2"],
    #             "query_scope": {
    #                 "node_type": "Paragraph",
    #                 "query_type": "combined_query",
    #                 "has_input": true
    #             }
    #         }
    #     ]
    #     """

    #     prompt = f"""
    #     # 任务
    #     根据用户的查询请求，生成一个分步的JSON格式查询计划。严格遵守以下规则：

    #     1. 使用以下实体类型名称：
    #     {schema_description.split('实体类型')[1].split('关键属性')[0].strip()}

    #     2. 支持以下查询类型：
    #     - semantic_query: 基于文本内容的语义搜索
    #     - structural_query: 基于结构关系的导航搜索
    #     - combined_query: 结合语义和结构的混合搜索

    #     3. 每个步骤必须包含以下字段：
    #     - step: 步骤序号 (从1开始)
    #     - target: 目标实体类型
    #     - constraints: 包含至少一个约束条件
    #     - input: 输入数据来源 (可选)
    #     - query_scope: 查询范围描述

    #     # 示例
    #     {example_query_plan}

    #     # 当前查询
    #     {query}
    #     """

    #     try:
    #         response = self._call_llm(prompt).strip()
    #         response = response.replace("```json", "").replace("```", "")
    #         plan = json.loads(response)
    #         for step in plan:
    #             if "target" in step:
    #                 original_type = step["target"].lower()
    #                 step["target"] = self.type_mapping.get(original_type, original_type)
    #         return plan
    #     except json.JSONDecodeError as e:
    #         print(f"JSON解析失败，原始响应内容：\n{response}")
    #         raise ValueError("生成的查询计划格式不正确，请重试")
    #     except Exception as e:
    #         print(f"生成查询计划时发生错误：{str(e)}")
    #         raise

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

    def execute_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        """查询计划执行"""
        context = {}
        for step in sorted(plan, key=lambda x: x["step"]):
            print(f"\n{'=' * 30} 执行步骤 {step['step']} {'=' * 30}")
            current_target = step["target"]
            constraints = step["constraints"]
            input_data = self._resolve_input_data(step, context)

            # 根据约束条件选择搜索类型
            if "semantic_query" in constraints:
                results = self._semantic_search(
                    query=constraints["semantic_query"],
                    target=current_target,
                    input_data=input_data,
                    k=constraints.get("top_k", 5)
                )
            elif "selected_papers" in constraints:
                results = self._structural_search(
                    target=current_target,
                    input_data=input_data
                )
            else:
                results = []

            context[f"step_{step['step']}"] = results
            print(f"步骤 {step['step']} 找到 {len(results)} 条结果")
        return context

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
                    timeout=45  # 延长超时时间
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"API调用失败，正在重试... ({attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                raise
        return "无法生成报告：API请求失败"
    
    def print_index_status(self):
        """打印索引状态"""
        print(f"当前索引状态:")
        print(f"索引包含 {self.index.ntotal} 个向量")
        print(f"节点信息数量: {len(self.node_info)}")
        print(f"嵌入矩阵形状: {self.node_embeddings.shape}")
        print("前5个节点类型:")
        for info in self.node_info[:5]:
            print(f"- {info['type']}: {info['key']}")

    def print_query_results(self, context: Dict[str, Any]):
        """打印查询计划每一步的搜索结果"""
        for step, results in context.items():
            print(f"\n{'=' * 30} {step} 搜索结果 {'=' * 30}")
            if not results:
                print(f"{step} 未找到相关结果。")
                continue
            for i, result in enumerate(results, start=1):
                print(f"结果 {i}:")
                print(f"  节点类型: {result['type']}")
                print(f"  论文 ID: {result['paper_id']}")
                print(f"  节点键: {result['key']}")
                if 'similarity' in result:
                    print(f"  相似度: {result['similarity']:.4f}")
                print(f"  内容: {str(result['content'])[:200]}...")

    def check_index_health(self):
        """检查索引健康状况"""
        print("\n=== 索引诊断报告 ===")
        
        # 基础统计
        print(f"索引包含向量数: {self.index.ntotal}")
        print(f"节点信息记录数: {len(self.node_info)}")
        print(f"嵌入矩阵形状: {self.node_embeddings.shape}")
        
        # 类型分布统计
        type_counts = {}
        for info in self.node_info:
            t = info['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        print("\n节点类型分布:")
        for t, cnt in type_counts.items():
            print(f"- {t}: {cnt}")

        # 随机采样检查
        print("\n随机采样检查:")
        for _ in range(3):
            idx = np.random.randint(0, len(self.node_info))
            node = self.node_info[idx]
            print(f"Key: {node['key']}")
            print(f"Type: {node['type']}")
            print(f"Embedding Norm: {np.linalg.norm(self.node_embeddings[idx]):.2f}")
            print(f"Content Sample: {str(node['content'])[:100]}...\n")

    def validate_query_plan(self, plan: List[Dict]):
        """验证查询计划有效性"""
        valid_types = {'paper_root', 'abstract', 'chapter', 'reference', 'table'}
        
        for step in plan:
            # 校验步骤编号
            if 'step' not in step or not isinstance(step['step'], int):
                raise ValueError(f"步骤缺少有效编号: {step}")
            
            # 校验目标类型
            target = step.get('target', '').lower()
            if target not in valid_types:
                raise ValueError(f"无效目标类型: {target}，有效类型为: {valid_types}")
            
            # 校验约束条件
            constraints = step.get('constraints', {})
            if not constraints:
                raise ValueError(f"步骤 {step['step']} 缺少约束条件")
                
            # 校验输入依赖
            inputs = step.get('input', [])
            for inp in inputs:
                if not inp.startswith('step_'):
                    raise ValueError(f"无效输入依赖格式: {inp}，应类似 'step_1'")

    def validate_query_plan(self, plan: List[Dict]):
        """增强版查询计划验证"""
        valid_types = set(self.type_mapping.values())  # 使用映射后的类型
        
        for step in plan:
            # 校验目标类型
            target_type = step.get('target', '')
            if target_type not in valid_types:
                raise ValueError(
                    f"无效目标类型: '{target_type}'\n"
                    f"有效类型应为: {', '.join(valid_types)}\n"
                    f"提示：可以使用类似'paper_root'等系统定义的类型名称"
                )
            
            # 校验约束条件
            if "semantic_query" not in step["constraints"] and "selected_papers" not in step["constraints"]:
                print(f"警告：步骤 {step['step']} 缺少有效约束条件")

    #function call
    def search_units(self, 
                    entity_type: str, 
                    attribute_filters: Optional[Dict] = None,
                    relation_filters: Optional[Dict] = None, 
                    keywords: Optional[str] = None,
                    top_k: int = 5) -> List[Dict]:
        """
        多条件组合搜索函数
        :param entity_type: 目标实体类型（paper_root/abstract/chapter等）
        :param attribute_filters: 属性过滤条件 {"authors": ["John"], "year": 2023}
        :param relation_filters: 关系过滤 {"has_reference": True, "in_chapter": "3"}
        :param keywords: 关键词语义搜索
        :param top_k: 返回结果数量
        """
        # 步骤1：按类型过滤
        candidates = [n for n in self.node_info if n['type'] == entity_type]
        
        # 步骤2：属性过滤
        if attribute_filters:
            candidates = self._apply_attribute_filters(candidates, attribute_filters)
            
        # 步骤3：语义搜索
        if keywords:
            candidates = self._semantic_search(keywords, candidates, top_k=top_k*2)
            
        # 步骤4：关系过滤
        if relation_filters:
            candidates = self._apply_relation_filters(candidates, relation_filters)
            
        # 最终排序和截取
        return sorted(
            candidates,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:top_k]

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

    def traverse_relations(self,
                          source_keys: List[str],
                          relation_types: List[str],
                          target_types: List[str],
                          max_depth: int = 2) -> Dict[str, List]:
        """
        图关系遍历函数
        :param source_keys: 起始节点键列表
        :param relation_types: 要遍历的关系类型 ["has_reference", "has_chapter"]
        :param target_types: 目标节点类型 ["reference", "chapter"]
        :param max_depth: 最大遍历深度
        """
        if not all(isinstance(k, str) for k in source_keys):
        # 尝试从字典中提取key
            source_keys = [k["key"] if isinstance(k, dict) else str(k) for k in source_keys]
    
        results = {t: [] for t in target_types}
        visited = set()
        
        def dfs(current_key: str, depth: int):
            if depth > max_depth or current_key in visited:
                return
            visited.add(current_key)
            
            relations = self.graph.semantic_graph.graph_relations.get(current_key, {})
            for rel_type, children in relations.get("children", {}).items():
                if rel_type not in relation_types:
                    continue
                
                for child_key in children:
                    child_node = next(
                        (n for n in self.node_info if n["key"] == child_key),
                        None
                    )
                    if not child_node:
                        continue
                        
                    if child_node['type'] in target_types:
                        results[child_node['type']].append(child_node)
                        
                    dfs(child_key, depth+1)
        
        for key in source_keys:
            dfs(key, 0)
            
        return results

    def generate_advanced_plan(self, query: str) -> Dict:
        schema_description = """
        节点类型 (Node Types):
        - paper_root: 论文根节点（包含标题、作者、发表日期）
        - chapter: 论文章节
        - paragraph: 段落
        - reference: 参考文献
        - table: 表格
        - image: 图片

        关系类型 (Relation Types):
        - has_chapter: 论文包含章节
        - has_paragraph: 章节包含段落
        - references: 论文引用参考文献
        - has_table: 论文包含表格
        - has_image: 章节包含图片
        """

        function_descriptions = """
        可用函数：
        1. search_units(
            entity_type: 实体类型,
            attribute_filters?: 属性过滤,
            relation_filters?:关系过滤,
            keywords?:关键词,
            top_k?:返回数量
        ) -> 搜索结果列表

        2. traverse_relations(
            source_keys: 起始节点列表,
            relation_types: 关系类型列表,
            target_types: 目标类型列表,
            max_depth?:遍历深度
        ) -> 关系遍历结果
        """

        prompt = f'''
        根据用户查询生成JSON格式的查询计划，使用上述函数。遵循以下规则：
        1. 使用标准实体类型：{list(self.type_mapping.values())}
        2. 参数值使用中文
        3. 输出键使用英文
        4. 优先使用关系类型进行关联查询

        示例计划：
        {{
            "steps": [
                {{
                    "function": "search_units",
                    "params": {{
                        "entity_type": "paper_root",
                        "keywords": "memory mechanism",
                        "top_k": 3
                    }},
                    "output_key": "papers"
                }},
                {{
                    "function": "traverse_relations",
                    "params": {{
                        "source_keys": "$papers",
                        "relation_types": ["references"],
                        "target_types": ["reference"],
                        "max_depth": 1
                    }},
                    "output_key": "refs"
                }}
            ]
        }}

        当前查询：{query}
        {function_descriptions}
        '''

        try:
            response = self._call_llm(prompt)
            return self._parse_plan(response)
        except Exception as e:
            print(f"生成计划失败: {str(e)}")
            return {"steps": []}

    def _parse_plan(self, response: str) -> Dict:
        """解析LLM响应为查询计划"""
        try:
            # 去除可能的代码块标记
            cleaned = response.replace("```json", "").replace("```", "").strip()
            plan = json.loads(cleaned)
            
            # 验证基本结构
            if "steps" not in plan:
                raise ValueError("计划缺少steps字段")
                
            for step in plan["steps"]:
                if not all(k in step for k in ["function", "params", "output_key"]):
                    raise ValueError("步骤缺少必要字段")
                    
            return plan
        except json.JSONDecodeError:
            print(f"无效的JSON响应：\n{response}")
            return {"steps": []}

    def execute_advanced_plan(self, plan: Dict) -> Dict:
        context = {}
        for step in plan.get("steps", []):
            try:
                # 参数预处理
                params = self._resolve_params(step["params"], context)
                
                # 确保source_keys为字符串列表
                if step["function"] == "traverse_relations":
                    params["source_keys"] = [
                        item["key"] if isinstance(item, dict) else str(item)
                        for item in params.get("source_keys", [])
                    ]
                
                # 执行函数
                if step["function"] == "search_units":
                    result = self.search_units(**params)
                elif step["function"] == "traverse_relations":
                    result = self.traverse_relations(**params)
                else:
                    raise ValueError(f"未知函数: {step['function']}")
                
                context[step["output_key"]] = result
                print(f"执行 {step['function']} 成功，获得 {len(result)} 条结果")
                
            except Exception as e:
                print(f"执行 {step['function']} 失败: {str(e)}")
                context[step["output_key"]] = []
        
        return context
   

    def _resolve_params(self, params: Dict, context: Dict) -> Dict:
        """解析参数中的上下文引用"""
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$"):
                # 处理嵌套引用（例如 $result.papers）
                keys = v[1:].split('.')
                value = context
                for key in keys:
                    value = value.get(key, {})
                resolved[k] = value
            elif isinstance(v, dict):
                # 递归处理字典参数
                resolved[k] = self._resolve_params(v, context)
            elif isinstance(v, list):
                # 处理列表中的引用
                resolved[k] = [
                    self._resolve_params(item, context) if isinstance(item, dict) else
                    self._resolve_param_item(item, context)
                    for item in v
                ]
            else:
                resolved[k] = v
        return resolved

    def _resolve_param_item(self, item, context):
        if isinstance(item, str) and item.startswith("$"):
            return context.get(item[1:], [])
        return item

    def generate_report(self, context: Dict, query: str) -> str:
        """生成最终分析报告"""
        prompt = f"""
        根据以下分析结果生成中文报告：
        原始查询：{query}
        上下文数据：{json.dumps(context, indent=2)}
        
        """
        
        return self._call_llm(prompt)


def _result_size(result):
    if isinstance(result, dict):
        return sum(len(v) for v in result.values())
    elif isinstance(result, list):
        return len(result)
    return 0


if __name__ == "__main__":
    graph = parse_local_paper()
    agent = ArxivAgent(graph)
    
    while True:
        try:
            query = input("\n请输入查询（输入q退出）: ")
            if query.lower() == 'q':
                break
                
            print("\n生成查询计划中...")
            plan = agent.generate_advanced_plan(query)
            print("生成的计划:", json.dumps(plan, indent=2, ensure_ascii=False))
            
            print("\n执行查询计划...")
            results = agent.execute_advanced_plan(plan)
           
            print("\n生成分析报告...")
            report = agent.generate_report(results, query)
            print("\n=== 最终报告 ===")
            print(report)
            
            with open(f"report_{int(time.time())}.txt", "w") as f:
                f.write(report)
                
        except KeyboardInterrupt:
            print("\n操作已取消")
            continue
        except Exception as e:
            print(f"发生错误：{str(e)}")
