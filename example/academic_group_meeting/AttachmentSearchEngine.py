import requests
import urllib.parse
import json
import base64
import os
import uuid
from serpapi import GoogleSearch
import arxiv
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from academic_group_meeting_graph import NamespaceType, AcademicDataType


class AttachmentSearchEngine:
    """学术附件检索引擎，用于检索和管理与学术主题相关的论文和图片"""
    
    def __init__(self, 
                 google_api_key: str = "8f64071da03e8c43e87abbeda944fa11d8111ee823294121ad91380c938c1717",
                 attachment_dir: str = "./attachments",
                 max_papers: int = 5,
                 max_images: int = 3):
        """
        初始化附件检索引擎
        
        Args:
            google_api_key: Google搜索API密钥
            attachment_dir: 附件存储目录
            max_papers: 每次检索最大论文数量
            max_images: 每次检索最大图片数量
        """
        self.google_api_key = google_api_key
        self.attachment_dir = attachment_dir
        self.max_papers = max_papers
        self.max_images = max_images
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 确保附件目录存在
        os.makedirs(attachment_dir, exist_ok=True)
        os.makedirs(os.path.join(attachment_dir, "papers"), exist_ok=True)
        os.makedirs(os.path.join(attachment_dir, "images"), exist_ok=True)
    
    def search_papers(self, query: str, max_results: int = None) -> List[Dict]:
        """
        使用arXiv API搜索相关学术论文
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量，None则使用默认设置
            
        Returns:
            List[Dict]: 论文信息列表
        """
        if max_results is None:
            max_results = self.max_papers
            
        try:
            # 使用arXiv API搜索
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                # 提取论文信息
                paper_info = {
                    "id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "url": paper.pdf_url,
                    "categories": paper.categories,
                    "source": "arXiv"
                }
                results.append(paper_info)
                
            return results
        
        except Exception as e:
            print(f"arXiv搜索失败: {str(e)}")
            
            # 如果arXiv API失败，回退到使用Google Scholar搜索
            try:
                return self._search_papers_via_google(query, max_results)
            except Exception as e2:
                print(f"Google Scholar搜索也失败: {str(e2)}")
                return []
    
    def _search_papers_via_google(self, query: str, max_results: int) -> List[Dict]:
        """使用Google Scholar搜索学术论文"""
        try:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.google_api_key,
                "num": str(max_results),
                "hl": "zh-cn"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            papers = []
            if "organic_results" in results:
                for result in results["organic_results"][:max_results]:
                    paper_info = {
                        "id": str(uuid.uuid4())[:8],
                        "title": result.get("title", "Unknown Title"),
                        "authors": result.get("publication_info", {}).get("authors", []),
                        "summary": result.get("snippet", ""),
                        "published": result.get("publication_info", {}).get("summary", ""),
                        "url": result.get("link", ""),
                        "source": "Google Scholar"
                    }
                    papers.append(paper_info)
            
            return papers
            
        except Exception as e:
            print(f"Google Scholar搜索失败: {str(e)}")
            return []
    
    def search_images(self, query: str, max_results: int = None) -> List[Dict]:
        """
        搜索与学术主题相关的图片
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量，None则使用默认设置
            
        Returns:
            List[Dict]: 图片信息列表
        """
        if max_results is None:
            max_results = self.max_images
            
        try:
            # 构建Google图片搜索参数
            params = {
                "engine": "google_images",
                "q": query + " academic research diagram",
                "api_key": self.google_api_key,
                "num": str(max_results),
                "safe": "active",  # 安全搜索
                "ijn": "0"         # 页码
            }
            
            # 执行搜索
            search = GoogleSearch(params)
            results = search.get_dict()
            
            images = []
            if "images_results" in results:
                for i, image in enumerate(results["images_results"][:max_results]):
                    image_info = {
                        "id": f"img_{uuid.uuid4().hex[:8]}",
                        "title": image.get("title", f"Image {i+1}"),
                        "url": image.get("original", ""),
                        "thumbnail": image.get("thumbnail", ""),
                        "source_url": image.get("source", ""),
                        "width": image.get("original_width", 0),
                        "height": image.get("original_height", 0),
                        "query": query
                    }
                    images.append(image_info)
            
            return images
            
        except Exception as e:
            print(f"图片搜索失败: {str(e)}")
            return []
    
    def download_paper(self, paper_info: Dict) -> Optional[str]:
        """
        下载论文并保存到本地
        
        Args:
            paper_info: 论文信息字典
            
        Returns:
            Optional[str]: 论文保存路径，失败则返回None
        """
        try:
            paper_id = paper_info.get("id", "unknown")
            paper_url = paper_info.get("url")
            
            if not paper_url:
                print(f"论文 {paper_id} 没有下载URL")
                return None
                
            # 构建保存路径
            filename = f"{paper_id}.pdf"
            save_path = os.path.join(self.attachment_dir, "papers", filename)
            
            # 检查文件是否已存在
            if os.path.exists(save_path):
                print(f"论文 {paper_id} 已存在，跳过下载")
                return save_path
                
            # 下载文件
            response = requests.get(paper_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # 保存文件
            with open(save_path, "wb") as f:
                f.write(response.content)
                
            print(f"论文 {paper_id} 下载成功: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"论文下载失败: {str(e)}")
            return None
    
    def download_image(self, image_info: Dict) -> Optional[str]:
        """
        下载图片并保存到本地
        
        Args:
            image_info: 图片信息字典
            
        Returns:
            Optional[str]: 图片保存路径，失败则返回None
        """
        try:
            image_id = image_info.get("id", f"img_{uuid.uuid4().hex[:8]}")
            image_url = image_info.get("url") or image_info.get("original")
            
            if not image_url:
                print(f"图片 {image_id} 没有下载URL")
                return None
                
            # 构建保存路径
            filename = f"{image_id}.jpg"
            save_path = os.path.join(self.attachment_dir, "images", filename)
            
            # 检查文件是否已存在
            if os.path.exists(save_path):
                print(f"图片 {image_id} 已存在，跳过下载")
                return save_path
                
            # 下载图片
            response = requests.get(image_url, headers=self.headers, timeout=20)
            response.raise_for_status()
            
            # 使用PIL处理图片
            image = Image.open(BytesIO(response.content))
            
            # 保存为JPEG格式
            image.save(save_path, "JPEG")
                
            print(f"图片 {image_id} 下载成功: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"图片下载失败: {str(e)}")
            return None
    
    def add_paper_to_semantic_graph(self, 
                                    backend, 
                                    paper_info: Dict, 
                                    topic_key: str = None,
                                    local_path: str = None) -> str:
        """
        将论文添加到语义图中
        
        Args:
            backend: 学术组会后端系统
            paper_info: 论文信息
            topic_key: 关联的主题节点ID
            local_path: 论文本地路径
            
        Returns:
            str: 添加的论文节点ID
        """
        paper_id = f"paper_{paper_info.get('id')}"
        
        # 准备论文节点数据
        paper_data = {
            "Title": paper_info.get("title", "Unknown Title"),
            "Authors": ", ".join(paper_info.get("authors", ["Unknown"])),
            "Abstract": paper_info.get("summary", ""),
            "Published": paper_info.get("published", ""),
            "URL": paper_info.get("url", ""),
            "Categories": ", ".join(paper_info.get("categories", [])),
            "Source": paper_info.get("source", "Unknown"),
            "LocalPath": local_path,
            "AddedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到语义图，如果有关联主题则建立关系
        parent_keys = [topic_key] if topic_key else None
        backend.add_paper(paper_id, paper_data, parent_keys)
        
        print(f"论文 '{paper_data['Title']}' 已添加到语义图")
        return paper_id
    
    def add_image_to_semantic_graph(self,
                                   backend,
                                   image_info: Dict,
                                   topic_key: str = None,
                                   local_path: str = None) -> str:
        """
        将图片添加到语义图中
        
        Args:
            backend: 学术组会后端系统
            image_info: 图片信息
            topic_key: 关联的主题节点ID
            local_path: 图片本地路径
            
        Returns:
            str: 添加的图片节点ID
        """
        # 创建图片ID
        image_id = image_info.get("id", f"img_{uuid.uuid4().hex[:8]}")
        
        # 准备图片节点数据
        image_data = {
            "Title": image_info.get("title", "Unknown Image"),
            "URL": image_info.get("url", ""),
            "SourceURL": image_info.get("source_url", ""),
            "Width": image_info.get("width", 0),
            "Height": image_info.get("height", 0),
            "Query": image_info.get("query", ""),
            "LocalPath": local_path,
            "AddedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到语义图中的Reference类型
        parent_keys = [topic_key] if topic_key else None
        backend.semantic_graph.add_node(
            image_id, 
            image_data, 
            AcademicDataType.Reference,  # 使用Reference类型存储图片
            parent_keys
        )
        
        print(f"图片 '{image_data['Title']}' 已添加到语义图")
        return image_id
    
    def search_and_add_attachments(self, 
                                  backend, 
                                  topic: str,
                                  topic_key: str = None,
                                  download_files: bool = True,
                                  papers_count: int = None,
                                  images_count: int = None) -> Dict[str, List[str]]:
        """
        搜索并添加与主题相关的附件到语义图
        
        Args:
            backend: 学术组会后端系统
            topic: 搜索主题
            topic_key: 主题节点ID
            download_files: 是否下载文件到本地
            papers_count: 论文数量
            images_count: 图片数量
            
        Returns:
            Dict[str, List[str]]: 添加的附件节点ID列表，按类型分组
        """
        result = {
            "papers": [],
            "images": []
        }
        
        # 修复: 使用英文引号而非中文引号
        print(f"\n=== 开始搜索主题'{topic}'的相关附件 ===\n")
        
        # 1. 搜索论文
        print(f"正在搜索相关论文...")
        papers = self.search_papers(topic, max_results=papers_count)
        
        if papers:
            print(f"找到 {len(papers)} 篇相关论文")
            for paper in papers:
                # 下载论文（可选）
                local_path = None
                if download_files:
                    local_path = self.download_paper(paper)
                
                # 添加到语义图
                paper_id = self.add_paper_to_semantic_graph(
                    backend=backend, 
                    paper_info=paper, 
                    topic_key=topic_key,
                    local_path=local_path
                )
                result["papers"].append(paper_id)
        else:
            print("未找到相关论文")
        
        # 2. 搜索图片
        print(f"\n正在搜索相关图片...")
        images = self.search_images(topic, max_results=images_count)
        
        if images:
            print(f"找到 {len(images)} 张相关图片")
            for image in images:
                # 下载图片（可选）
                local_path = None
                if download_files:
                    local_path = self.download_image(image)
                
                # 添加到语义图
                image_id = self.add_image_to_semantic_graph(
                    backend=backend,
                    image_info=image,
                    topic_key=topic_key,
                    local_path=local_path
                )
                result["images"].append(image_id)
        else:
            print("未找到相关图片")
        
        # 更新命名空间子图
        backend.semantic_graph.auto_generate_subgraphs()
        
        print(f"\n=== 附件搜索完成：添加了 {len(result['papers'])} 篇论文和 {len(result['images'])} 张图片 ===\n")
        return result
    
    def search_and_get_resources(self, topic: str) -> Dict:
        """
        搜索主题相关资源但不添加到图中，仅返回结果
        
        Args:
            topic: 搜索主题
            
        Returns:
            Dict: 包含论文和图片的搜索结果
        """
        result = {
            "papers": self.search_papers(topic),
            "images": self.search_images(topic)
        }
        
        paper_count = len(result["papers"])
        image_count = len(result["images"])
        
        print(f"找到 {paper_count} 篇相关论文和 {image_count} 张图片")
        
        return result