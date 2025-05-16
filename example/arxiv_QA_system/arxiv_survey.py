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
fetch_arxiv_html("agent memory",100)
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

            # 表格解析
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

        # 图片处理（带重试）
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
    graph = ArxivSemanticGraph()  # 只创建一个图谱实例
    
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

# # 下载论文
# graph = parse_local_paper()


# # 执行查询
# results = graph.query("agent memory", k=3)
# for paper in results:
#     print(f"标题: {paper['metadata']['title']}")
#     print(f"摘要: {paper['abstract'][:100]}...\n")

# # 保存图谱
# graph.export_preferences("my_preferences.json")

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-c4e7f78ec6fe48379839541971a2bfc7"
DEEPSEEK_MODEL = "deepseek-chat"

class ArxivAgent:
    def __init__(self,
                 graph: ArxivSemanticGraph,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7):
        """
        初始化智能代理
        :param graph: 已构建的ArxivSemanticGraph实例
        :param embedding_model: 嵌入模型名称
        """
        self.graph = graph
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold

        # 初始化语义索引
        self._build_semantic_index()

    def _build_semantic_index(self):
        self.node_embeddings = []
        self.node_info = []
        for node in self.graph.semantic_graph.semantic_map.data:
            node_key = node[0]
            node_content = node[1]
            node_type = self._get_node_type(node_key)

            if node_type == "image":
                # 提取图片描述信息
                chapter_key = node_key.rsplit("_", 2)[0]
                chapter_title = self._get_chapter_title(chapter_key)
                text = f"Figure in {chapter_title}"
            else:
                if isinstance(node_content, dict):
                    text = node_content.get("title", str(node_content))
                else:
                    text = str(node_content)

            # 对文本进行编码
            embedding = self.embedder.encode(text)
            self.node_embeddings.append(embedding)

            # 记录节点信息
            self.node_info.append({
                "key": node_key,
                "type": node_type,
                "content": node_content
            })

        self.node_embeddings = np.array(self.node_embeddings)

    def _get_node_type(self, node_key):
        if "_title_authors" in node_key:
            return "paper"
        elif "_abstract" in node_key:
            return "abstract"
        elif "_chapter_" in node_key:
            return "chapter"
        elif "_para_" in node_key:
            return "paragraph"
        elif "_ref_" in node_key:
            return "reference"
        elif "_table_" in node_key:
            return "table"
        elif "_img_" in node_key:
            return "image"
        return "other"

    def _get_chapter_title(self, chapter_key):
        # 根据章节键查找章节标题
        for node in self.graph.semantic_graph.semantic_map.data:
            if node[0] == chapter_key:
                return node[1] if isinstance(node[1], str) else node[1].get("title", "Unknown Chapter")
        return "Unknown Chapter"

    def _call_deepseek(self, prompt):
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        session = requests.Session()

        # 配置重试策略
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # 添加 model 字段到请求体
        request_body = {
            "messages": [{"role": "user", "content": prompt}],
            "model": DEEPSEEK_MODEL
        }
        print(f"请求体: {request_body}")

        try:
            # 增加超时时间
            response = session.post(
                DEEPSEEK_API_URL,
                json=request_body,
                headers=headers,
                timeout=60  # 增加超时时间到 60 秒
            )
            response.raise_for_status()
            response_data = response.json()
            print(f"响应内容: {response_data}")

            # 提取助手的回复内容
            content = response_data["choices"][0]["message"]["content"]

            # 新增：提取JSON部分
            json_match = re.search(
                r'```(?:json)?\n(.*?)\n```',
                content,
                re.DOTALL
            )
            if json_match:
                return json_match.group(1).strip()
            return content
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP 错误: {http_err}，响应内容: {response.text}")
            return self._local_keyword_extract(prompt)
        except requests.exceptions.ConnectionError as conn_err:
            print(f"连接错误: {conn_err}，使用本地回退策略")
            return self._local_keyword_extract(prompt)
        except Exception as e:
            print(f"API请求失败，使用本地回退策略: {str(e)}")
            return self._local_keyword_extract(prompt)
    def _local_keyword_extract(self, text):
        """本地关键词提取备用方案"""
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=20
        )
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out().tolist()

        # 针对 generate_query_plan 生成模拟的查询计划 JSON
        if "转换为分步查询计划" in text:
            # 提取原始查询内容
            original_query = re.search(r'原始查询：(.*)', text).group(1).strip()
            plan = [
                {
                    "step": 1,
                    "target": "paper",
                    "constraints": {
                        "semantic_query": original_query
                    },
                    "input": []
                }
            ]
            return json.dumps(plan)
        # 针对 understand_intent 生成模拟的意图理解 JSON
        else:
            return json.dumps({
                "intent": "review_generation",
                "key_concepts": keywords
            })
    def understand_intent(self, query: str) -> Dict:
        """理解用户意图并提取关键概念（增强容错版）"""
        prompt = f"""请严格按以下格式输出JSON：
        {{
            "intent": "<classification>",
            "key_concepts": ["concept1", "concept2"]
        }}
        分类选项：review_generation, paper_search, methodology_compare
        查询内容：{query}"""
        try:
            response_text = self._call_deepseek(prompt)
            return json.loads(response_text) 
        except json.JSONDecodeError:
            print(f"理解意图时，无法解析响应为JSON: {response_text}")
            return {
                "intent": "review_generation",
                "key_concepts": [query]
            }


    def generate_query_plan(self, query: str) -> List[Dict]:
        """生成结构化查询计划"""
        prompt = f"""将以下科研查询转换为分步查询计划：
        原始查询：{query}

        要求：
        1. 使用JSON格式输出，包含step、target、constraints、input字段
        2. target字段使用[paper, abstract, methodology, application_case, conclusion]中的值
        3. 第一步必须是论文检索，后续步骤基于前序结果"""
        response = self._call_deepseek(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"生成查询计划时，无法解析响应为JSON: {response}")
            # 可以返回一个默认的查询计划，避免程序崩溃
            return [
                {
                    "step": 1,
                    "target": "paper",
                    "constraints": {
                        "semantic_query": query
                    },
                    "input": []
                }
            ]

    def expand_keywords(self, initial_keywords: List[str], max_docs: int = 1000, max_keywords: int = 10) -> List[str]:
        """关键词扩展算法"""
        keyword_pool = initial_keywords.copy()
        retrieved_docs = []

        # 空输入处理
        if not keyword_pool:
            print("警告：初始关键词为空")
            return []

        # 初始查询验证
        try:
            initial_results = self.graph.query(keyword_pool[0], k=1)
            if not initial_results:
                print("警告：初始查询无结果，返回原始关键词")
                return keyword_pool
        except Exception as e:
            print(f"初始查询失败: {str(e)}")
            return keyword_pool

        # 主循环
        iteration = 0
        while len(retrieved_docs) < max_docs and len(keyword_pool) < max_keywords:
            iteration += 1
            print(f"第 {iteration} 次迭代，当前关键词数量: {len(keyword_pool)}, 当前文档数量: {len(retrieved_docs)}")
            new_docs = []

            # 阶段1：使用当前关键词检索
            for keyword in keyword_pool:
                try:
                    results = self.graph.query(keyword, k=50)
                    if results:
                        new_docs.extend(results)
                except KeyError as e:
                    print(f"查询异常（关键词：{keyword}）: {str(e)}")
                    continue

            # 阶段2：数据清洗
            valid_docs = []
            for doc in new_docs:
                try:
                    if (
                        doc and
                        isinstance(doc, dict) and
                        doc.get("metadata") and
                        doc["metadata"].get("root_key") and
                        doc.get("abstract")
                    ):
                        valid_docs.append(doc)
                except Exception as e:
                    print(f"文档过滤异常: {str(e)}")
            new_docs = valid_docs

            # 阶段3：去重处理
            seen = set()
            unique_docs = []
            for doc in new_docs:
                try:
                    root_key = doc["metadata"]["root_key"]
                    if root_key not in seen:
                        seen.add(root_key)
                        unique_docs.append(doc)
                except KeyError:
                    continue
            new_docs = unique_docs

            # 阶段4：直接提取关键词
            new_keywords = []
            for doc in new_docs:
                try:
                    keywords = self._extract_keywords(doc["abstract"])
                    new_keywords.extend(keywords)
                except Exception as e:
                    print(f"关键词提取失败: {str(e)}")
                    continue

            # 阶段5：关键词筛选
            if new_keywords:
                try:
                    # 计算语义距离
                    existing_embeddings = self.embedder.encode(keyword_pool)
                    new_embeddings = self.embedder.encode(new_keywords)
                    similarity_matrix = np.dot(new_embeddings, existing_embeddings.T)

                    # 筛选条件
                    selected_keywords = []
                    for i, kw in enumerate(new_keywords):
                        avg_sim = np.mean(similarity_matrix[i])
                        max_sim = np.max(similarity_matrix[i])
                        if avg_sim > 0.4 and max_sim < 0.98:
                            selected_keywords.append(kw)

                    # 保留新关键词直到达到最大关键词数量
                    remaining_slots = max_keywords - len(keyword_pool)
                    keyword_pool.extend(selected_keywords[:remaining_slots])
                    keyword_pool = list(set(keyword_pool))  # 去重

                except Exception as e:
                    print(f"关键词筛选异常: {str(e)}")

            # 阶段6：更新文档池
            retrieved_docs.extend(new_docs)
            retrieved_docs = list({d["metadata"]["root_key"]: d for d in retrieved_docs}.values())  # 最终去重

            # 提前退出条件
            if not new_docs and not new_keywords:
                print("提前终止：没有新文档或关键词")
                break

        # 返回前max_keywords个最相关关键词（按出现频率）
        keyword_counts = {}
        for kw in keyword_pool:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        sorted_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: (-x[1], x[0])  # 先按频率降序，再按字母序
        )[:max_keywords]

        return [kw for kw, _ in sorted_keywords]

    def _extract_keywords(self, text: str) -> List[str]:
        """使用LLM提取关键词"""
        prompt = f"请从以下文本中提取5个最重要的科研关键词：\n{text}"
        response = self._call_deepseek(prompt)
        keywords = [kw.strip() for kw in response.split(",")]
        print(f"从文本中提取的关键词: {keywords}")  # 添加日志输出
        return keywords

    def retrieve_data(self, query_plan):
        """执行完整的数据检索流程"""
        # 存储各步骤结果
        step_results = {}

        for step in sorted(query_plan, key=lambda x: x["step"]):
            # 获取输入数据
            input_data = []
            if step["input"]:
                if isinstance(step["input"], str) and "from step" in step["input"]:
                    step_num = int(step["input"].split(" ")[-1])
                    input_data.extend(step_results.get(step_num, []))
                else:
                    input_data.extend(step["input"])

            constraints = step["constraints"]
            if "keywords" in constraints:
                # 执行基于关键词的语义查询
                keyword = constraints["keywords"][0]
                results = self._semantic_search(
                    keyword,
                    step["target"],
                    input_data
                )
            elif "selected_papers" in constraints:
                # 执行基于选定论文的查询
                results = self._structural_search(
                    step["target"],
                    input_data
                )
            else:
                results = []

            step_results[step["step"]] = results

        return step_results

    def _semantic_search(self, query: str, target_types: List[str], context: List[Dict]) -> List[Dict]:
        """语义查询"""
        query_embedding = self.embedder.encode(query)
        similarities = np.dot(self.node_embeddings, query_embedding)

        results = []
        for idx in np.argsort(similarities)[::-1]:
            if similarities[idx] < self.similarity_threshold:
                continue

            node = self.node_info[idx]
            if node["type"] in target_types:
                results.append({
                    "type": node["type"],
                    "content": node["content"],
                    "paper_id": node["key"].split("_")[0],
                    "similarity": float(similarities[idx])
                })

        return results[:500]  # 返回前500个结果

    def _structural_search(self, target_types: List[str], context: List[Dict]) -> List[Dict]:
        """结构查询"""
        results = []
        for item in context:
            # 获取关联节点
            related_nodes = self.graph.semantic_graph.graph_relations[item["key"]].get("children", {})
            for node_key, relation in related_nodes.items():
                node_type = self._get_node_type(node_key)
                if node_type in target_types:
                    node = next(n for n in self.node_info if n["key"] == node_key)
                    results.append({
                        "type": node_type,
                        "content": node["content"],
                        "paper_id": item["paper_id"],
                        "relation": relation
                    })
        return results

    def generate_review(self, data: Dict, query: str) -> str:
        """生成并优化综述"""
        # 整合数据
        structured_data = self._organize_data(data)

        # 初稿生成
        draft = self._generate_draft(structured_data, query)

        # RAG优化
        final_review = self._rag_enhancement(draft)

        # 生成参考文献
        references = self._generate_references(data)

        return f"{final_review}\n\n## References\n{references}"

    def _organize_data(self, data: Dict) -> Dict:
        """组织数据结构"""
        organized = {
            "background": [],
            "methodology": [],
            "applications": [],
            "conclusions": []
        }

        for step, results in data.items():
            for item in results:
                if item["type"] == "abstract":
                    organized["background"].append(item["content"])
                elif item["type"] == "methodology":
                    organized["methodology"].append(item["content"])
                elif item["type"] == "application_case":
                    organized["applications"].append(item["content"])
                elif item["type"] == "conclusion":
                    organized["conclusions"].append(item["content"])

        return organized

    def _generate_draft(self, data: Dict, query: str) -> str:
        """生成初稿"""
        prompt_template = f"""
        请根据以下研究数据生成综述，结构包含：
        1. 引言（背景）
        2. 研究方法
        3. 应用案例
        4. 结论与展望

        研究主题：{query}
        研究数据：
        {json.dumps(data, indent=2)}
        """
        return self._call_deepseek(prompt_template)

    def _rag_enhancement(self, draft: str) -> str:
        """检索增强的优化"""
        sections = draft.split("\n\n")
        enhanced = []

        for section in tqdm(sections, desc="优化章节"):
            # 检索补充信息
            results = self.graph.query(section, k=3)
            supplement = "\n".join([r["abstract"] for r in results])

            # 重写章节
            prompt = f"""请根据以下补充信息优化文本：
            原始文本：{section}

            补充信息：
            {supplement}

            要求：
            1. 保持原有结构
            2. 增强内容准确性
            3. 添加文献引用标记[1][2]"""
            enhanced_section = self._call_deepseek(prompt)
            enhanced.append(enhanced_section)

        return "\n\n".join(enhanced)

    def _generate_references(self, data: Dict) -> str:
        """生成参考文献列表"""
        papers = set()
        for step_results in data.values():
            for item in step_results:
                if paper := self.graph._get_full_paper(item["paper_id"]):
                    papers.add(paper)

        ref_list = []
        for idx, paper in enumerate(papers, 1):
            ref_list.append(
                f"[{idx}] {paper['metadata']['title']} "
                f"by {', '.join(paper['metadata']['authors'])}"
            )

        return "\n".join(ref_list)


if __name__ == "__main__":
    graph = parse_local_paper()
    agent = ArxivAgent(graph)

    while True:
        user_query = input("请输入查询内容（输入 'quit' 退出）：")
        if user_query.lower() == 'quit':
            break

        intent = agent.understand_intent(user_query)
        print(f"识别意图：{intent}")

        query_plan = agent.generate_query_plan(user_query)
        print("生成查询计划：")
        print(json.dumps(query_plan, indent=2))

        keywords = agent.expand_keywords(intent["key_concepts"])
        print(f"扩展关键词：{keywords}")

        retrieved_data = agent.retrieve_data(query_plan)

        review = agent.generate_review(retrieved_data, user_query)
        print("\n生成的综述：")
        print(review) 