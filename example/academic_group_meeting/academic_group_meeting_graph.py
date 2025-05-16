import sys
import os
import faiss
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import re
import requests
from semantic_map.deepseek_client import deepseek_remote, deepseek_local
from sentence_transformers import SentenceTransformer
import uuid
import time
from datetime import datetime
from typing import List, Dict , Set, Tuple, Any
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties
from semantic_map.data_type import BaseDataType
from semantic_map.semantic_map import BaseSemanticMap
from semantic_map.semantic_simple_graph import BaseSemanticSimpleGraph
from enum import auto
import numpy as np
# from academic_group_meeting_backend import AcademicMeetingSystem
# from semantic_map import MilvusDialogueStorage
# from semantic_map import Neo4jInterface
from matplotlib import pyplot as plt
from enum import Enum, auto

try:
    plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：可能无法正确显示中文，请安装相应字体")

# 在AcademicDataType类之后添加
class NamespaceType(Enum):
    """学术组会系统的命名空间分类"""
    DIALOGUE = auto()   # 对话命名空间：包含问题、讨论、结论等
    ATTACHMENT = auto() # 附件命名空间：包含论文、研究资料等
    USER = auto()       # 用户命名空间：包含教授、博士生、硕士生等人员
    TASK = auto()       # 任务命名空间：包含研究主题、任务等
    SUMMARY = auto()    # 总结命名空间：包含教授的轮次总结、最终总结等

# plt.rc('font',family='YouYuan') # 设置中文显示
# plt.rc('SimHei') # 设置中文显示
try:
    plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：可能无法正确显示中文，请安装相应字体")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConversationScene:
    """对话场景类，管理一个多人对话场景及其历史记录"""
    def __init__(self, name: str, description: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.messages = []  # 该场景的所有对话消息
        self.participants = set()  # 参与者集合

    def add_message(self, speaker_id: str, content: str, message_id: str = None, timestamp: str = None):
        """添加一条消息到对话历史"""
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.messages.append({
            "id": message_id,
            "speaker_id": speaker_id,
            "content": content,
            "timestamp": timestamp
        })
        self.participants.add(speaker_id)
        return message_id

    def get_history(self, last_n=5) -> List[Dict[str, Any]]:
        """获取最近的 N 条对话历史"""
        return self.messages[-last_n:] if len(self.messages) > 0 else []

    def get_all_participants(self) -> set:
        """获取所有参与者"""
        return self.participants

class AcademicDataType(BaseDataType):
    """学术组会的数据类型定义"""
    Professor = auto()     # 教授
    PhD = auto()           # 博士生
    Master = auto()        # 硕士生
    ResearchTopic = auto() # 研究主题
    Paper = auto()         # 论文
    Question = auto()      # 问题
    Discussion = auto()    # 讨论内容
    Reference = auto()     # 参考文献
    Conclusion = auto()    # 结论
    Presentation = auto()  # 演讲/报告
    Summary = auto()       # 教授的总结（轮次总结、最终总结）
    Task = auto()          # 任务（后续工作、研究方向）
    Attachment = auto()   # 附件（论文、研究资料等）

class AcademicRole:
    """学术角色定义，包含角色特点和专业领域"""
    def __init__(self, role_type: str, specialty: List[str], personality: str):
        self.role_type = role_type  # 教授、博士生、硕士生
        self.specialty = specialty  # 专业领域列表
        self.personality = personality  # 性格特点
    
    def get_role_prompt(self) -> str:
        """获取角色提示信息，用于指导LLM生成符合角色特点的回复"""
        specialty_str = ", ".join(self.specialty)
        return f"{self.role_type}，专业领域：{specialty_str}。{self.personality}"


# class AcademicSearchEngine:
#     """学术搜索引擎，提供网络检索功能"""
#     def __init__(self, google_api_key: str = "8f64071da03e8c43e87abbeda944fa11d8111ee823294121ad91380c938c1717",
#                  llm_api_key:str="sk-fbab400c86184b0daf9bd59467d35772",):
#         self.google_api_key = google_api_key
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#         }
#         self.llm_client = deepseek_remote(llm_api_key)
        
#     def search(self, query: str, num_results: int = 3) -> List[Dict]:
#         """进行学术检索，返回搜索结果"""
#         try:
#             params = {
#                 "engine": "google",
#                 "q": query,
#                 "api_key": self.google_api_key,
#                 "num": str(num_results),
#                 "hl": "zh-cn",
#                 "gl": "cn"
#             }
#             search = GoogleSearch(params)
#             results = search.get_dict()
            
#             # 提取有用的搜索结果
#             if "organic_results" in results:
#                 return results["organic_results"]
#             else:
#                 return []
#         except Exception as e:
#             print(f"搜索API调用失败: {str(e)}")
#             return []
            
#     def fetch_webpage_content(self, url: str, max_length: int = 10000) -> str:
#         """从URL获取网页内容，改进版本增强了错误处理
        
#         Args:
#             url: 网页URL
#             max_length: 提取内容的最大字符数
            
#         Returns:
#             str: 提取的网页文本内容
#         """
#         try:
#             # URL检查和规范化
#             if not url or not url.startswith(('http://', 'https://')):
#                 return f"无效URL: {url}"
                
#             # 过滤不支持的文件类型
#             if url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
#                 return f"不支持直接提取文档格式: {url}，请访问链接查看内容。"
            
#             # 更长的超时设置
#             response = requests.get(
#                 url, 
#                 headers=self.headers, 
#                 timeout=15, 
#                 stream=True,  # 使用流式请求
#                 verify=True   # 验证SSL证书
#             )
#             response.raise_for_status()  # 检查请求是否成功
            
#             # 检查内容类型
#             content_type = response.headers.get('Content-Type', '')
#             if 'text/html' not in content_type and 'application/json' not in content_type:
#                 return f"不支持的内容类型: {content_type}，请直接访问链接查看内容。"
            
#             # 使用BeautifulSoup解析HTML
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # 移除可能干扰的元素
#             for tag in soup(['script', 'style', 'nav', 'footer', 'iframe', 'head', 'aside', 'noscript']):
#                 tag.decompose()
            
#             # 提取主要文本内容
#             text = soup.get_text(separator=' ', strip=True)
            
#             # 清理文本：移除多余空白和特殊字符
#             text = re.sub(r'\s+', ' ', text).strip()
#             text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)  # 移除控制字符
            
#             # 限制文本长度
#             if len(text) > max_length:
#                 text = text[:max_length] + "..."
            
#             # 检查内容是否为空
#             if not text.strip():
#                 return f"从{url}提取的内容为空，请直接访问链接查看。"
                
#             return text
#         except requests.exceptions.Timeout:
#             return f"获取网页内容超时 ({url})：请考虑直接访问网页查看内容。"
#         except requests.exceptions.SSLError:
#             return f"SSL证书验证失败 ({url})：可能是安全连接问题。"
#         except requests.exceptions.ConnectionError:
#             return f"连接失败 ({url})：无法连接到服务器。"
#         except requests.exceptions.HTTPError as e:
#             return f"HTTP错误 ({url}): {e.response.status_code} - {e.response.reason}"
#         except Exception as e:
#             return f"抓取网页内容时发生错误 ({url}): {str(e)}"
    
#     def enrich_search_results(self, results: List[Dict], fetch_content: bool = True) -> List[Dict]:
#         """增强搜索结果，可选择性地获取网页内容
        
#         Args:
#             results: 原始搜索结果列表
#             fetch_content: 是否获取完整网页内容
            
#         Returns:
#             List[Dict]: 增强后的搜索结果
#         """
#         enriched_results = []
        
#         for result in results:
#             enriched_result = {
#                 "title": result.get("title", ""),
#                 "url": result.get("link", ""),
#                 "snippet": result.get("snippet", ""),
#                 "date": result.get("date", ""),
#                 "source": result.get("source", "")
#             }
            
#             if fetch_content and "link" in result:
#                 # 添加短暂延迟避免过快发送请求
#                 time.sleep(1)
#                 content = self.fetch_webpage_content(result["link"])
#                 enriched_result["full_content"] = content
                
#             enriched_results.append(enriched_result)
            
#         return enriched_results
    
#     def prepare_content_for_llm(self, query: str, include_full_content: bool = False) -> str:
#         """准备适合LLM处理的内容
        
#         Args:
#             query: 搜索查询
#             include_full_content: 是否包含完整网页内容
            
#         Returns:
#             str: 格式化后适合LLM的文本
#         """
#         search_results = self.search(query)
#         enriched_results = self.enrich_search_results(
#             search_results, 
#             fetch_content=include_full_content
#         )
        
#         prompt = f'以下是关于"{query}"的搜索结果:\n\n'
        
#         for i, result in enumerate(enriched_results, 1):
#             prompt += f"【搜索结果 {i}】\n"
#             prompt += f"标题: {result['title']}\n"
#             prompt += f"来源: {result.get('source', '未知')} {result.get('date', '')}\n"
#             prompt += f"URL: {result['url']}\n"
#             prompt += f"摘要: {result['snippet']}\n"
            
#             if include_full_content and "full_content" in result:
#                 # 如果内容太长，可能需要截断
#                 content_preview = result["full_content"][:2000] + "..." if len(result["full_content"]) > 2000 else result["full_content"]
#                 prompt += f"网页内容: {content_preview}\n"
                
#             prompt += "\n" + "-"*50 + "\n\n"
            
#         prompt += "请基于以上信息，提供关于该主题的全面分析和见解。"
#         return prompt
    
#     def search_and_analyze(self, query: str, include_full_content: bool = False) -> Dict:
#         """搜索并分析结果
        
#         Args:
#             query: 搜索查询
#             include_full_content: 是否包含完整网页内容
#             llm_client: LLM客户端对象
            
#         Returns:
#             Dict: 包含原始搜索结果和LLM分析的字典
#         """
#         if not self.llm_client:
#             return {"error": "未提供LLM客户端"}
            
#         # 准备适合LLM的内容
#         prompt = self.prepare_content_for_llm(query, include_full_content)
        
#         # 使用LLM处理
#         messages = [
#             {"role": "system", "content": "你是一位专业的学术研究助手，擅长整合网络信息并提供深入分析。"},
#             {"role": "user", "content": prompt}
#         ]
        
#         # 调用LLM获取响应
#         response = self.llm_client.get_response(messages)
        
#         return {
#             "query": query,
#             "prompt": prompt,
#             "analysis": response
#         }
class AcademicSearchEngine:
    """学术搜索引擎，提供网络检索功能"""
    def __init__(self, google_api_key: str = "8f64071da03e8c43e87abbeda944fa11d8111ee823294121ad91380c938c1717",
                 llm_api_key:str="sk-fbab400c86184b0daf9bd59467d35772",
                 timeout: int = 3):  # 添加默认超时参数
        self.google_api_key = google_api_key
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.llm_client = deepseek_remote(llm_api_key)
        self.timeout = timeout  # 存储默认超时时间
        
    def search(self, query: str, num_results: int = 3, timeout: int = None) -> List[Dict]:
        """进行学术检索，返回搜索结果
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            timeout: 超时时间(秒)，默认使用实例初始化时设置的timeout值
            
        Returns:
            List[Dict]: 搜索结果列表，超时时返回包含超时信息的列表
        """
        # 使用指定的超时时间，若未指定则使用实例默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.google_api_key,
                "num": str(num_results),
                "hl": "zh-cn",
                "gl": "cn",
                "timeout": actual_timeout  # 添加超时参数
            }
            
            # 使用自定义超时进行API调用
            import socket
            # 保存原始超时设置
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(actual_timeout)
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # 提取有用的搜索结果
                if "organic_results" in results:
                    return results["organic_results"]
                else:
                    return []
            finally:
                # 恢复原始超时设置
                socket.setdefaulttimeout(original_timeout)
                
        except Exception as e:
            print(f"搜索API调用失败 (超时:{actual_timeout}秒): {str(e)}")
            return [{"title": "搜索超时", 
                     "link": "", 
                     "snippet": f"搜索请求超时({actual_timeout}秒)，请稍后再试。"}]
            
    def fetch_webpage_content(self, url: str, max_length: int = 10000, timeout: int = None) -> str:
        """从URL获取网页内容，改进版本增强了错误处理和超时设置
        
        Args:
            url: 网页URL
            max_length: 提取内容的最大字符数
            timeout: 超时时间(秒)，默认使用实例初始化时设置的timeout值
            
        Returns:
            str: 提取的网页文本内容，超时返回超时信息
        """
        # 使用指定的超时时间，若未指定则使用实例默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        
        try:
            # URL检查和规范化
            if not url or not url.startswith(('http://', 'https://')):
                return f"无效URL: {url}"
                
            # 过滤不支持的文件类型
            if url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
                return f"不支持直接提取文档格式: {url}，请访问链接查看内容。"
            
            # 使用设置的超时时间
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=actual_timeout,  # 使用设置的超时时间
                stream=True,  # 使用流式请求
                verify=True   # 验证SSL证书
            )
            response.raise_for_status()  # 检查请求是否成功
            
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'application/json' not in content_type:
                return f"不支持的内容类型: {content_type}，请直接访问链接查看内容。"
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除可能干扰的元素
            for tag in soup(['script', 'style', 'nav', 'footer', 'iframe', 'head', 'aside', 'noscript']):
                tag.decompose()
            
            # 提取主要文本内容
            text = soup.get_text(separator=' ', strip=True)
            
            # 清理文本：移除多余空白和特殊字符
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)  # 移除控制字符
            
            # 限制文本长度
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            # 检查内容是否为空
            if not text.strip():
                return f"从{url}提取的内容为空，请直接访问链接查看。"
                
            return text
        except requests.exceptions.Timeout:
            return f"获取网页内容超时 ({url})：请求超过{actual_timeout}秒，未能获取内容。"
        except requests.exceptions.SSLError:
            return f"SSL证书验证失败 ({url})：可能是安全连接问题。"
        except requests.exceptions.ConnectionError:
            return f"连接失败 ({url})：无法连接到服务器。"
        except requests.exceptions.HTTPError as e:
            return f"HTTP错误 ({url}): {e.response.status_code} - {e.response.reason}"
        except Exception as e:
            return f"抓取网页内容时发生错误 ({url}): {str(e)}"
    
    def enrich_search_results(self, results: List[Dict], fetch_content: bool = True, timeout: int = None) -> List[Dict]:
        """增强搜索结果，可选择性地获取网页内容
        
        Args:
            results: 原始搜索结果列表
            fetch_content: 是否获取完整网页内容
            timeout: 超时时间(秒)，默认使用实例初始化时设置的timeout值
            
        Returns:
            List[Dict]: 增强后的搜索结果
        """
        enriched_results = []
        
        for result in results:
            enriched_result = {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "date": result.get("date", ""),
                "source": result.get("source", "")
            }
            
            if fetch_content and "link" in result:
                # 添加短暂延迟避免过快发送请求
                time.sleep(1)
                content = self.fetch_webpage_content(result["link"], timeout=timeout)
                enriched_result["full_content"] = content
                
            enriched_results.append(enriched_result)
            
        return enriched_results
    
    def prepare_content_for_llm(self, query: str, include_full_content: bool = False, timeout: int = None) -> str:
        """准备适合LLM处理的内容
        
        Args:
            query: 搜索查询
            include_full_content: 是否包含完整网页内容
            timeout: 超时时间(秒)，默认使用实例初始化时设置的timeout值
            
        Returns:
            str: 格式化后适合LLM的文本
        """
        search_results = self.search(query, timeout=timeout)
        enriched_results = self.enrich_search_results(
            search_results, 
            fetch_content=include_full_content,
            timeout=timeout
        )
        
        prompt = f'以下是关于"{query}"的搜索结果:\n\n'
        
        for i, result in enumerate(enriched_results, 1):
            prompt += f"【搜索结果 {i}】\n"
            prompt += f"标题: {result['title']}\n"
            prompt += f"来源: {result.get('source', '未知')} {result.get('date', '')}\n"
            prompt += f"URL: {result['url']}\n"
            prompt += f"摘要: {result['snippet']}\n"
            
            if include_full_content and "full_content" in result:
                # 如果内容太长，可能需要截断
                content_preview = result["full_content"][:2000] + "..." if len(result["full_content"]) > 2000 else result["full_content"]
                prompt += f"网页内容: {content_preview}\n"
                
            prompt += "\n" + "-"*50 + "\n\n"
            
        prompt += "请基于以上信息，提供关于该主题的全面分析和见解。"
        return prompt
    
    def search_and_analyze(self, query: str, include_full_content: bool = False, timeout: int = None) -> Dict:
        """搜索并分析结果
        
        Args:
            query: 搜索查询
            include_full_content: 是否包含完整网页内容
            timeout: 超时时间(秒)，默认使用实例初始化时设置的timeout值
            
        Returns:
            Dict: 包含原始搜索结果和LLM分析的字典
        """
        if not self.llm_client:
            return {"error": "未提供LLM客户端"}
            
        # 准备适合LLM的内容
        prompt = self.prepare_content_for_llm(query, include_full_content, timeout=timeout)
        
        # 使用LLM处理
        messages = [
            {"role": "system", "content": "你是一位专业的学术研究助手，擅长整合网络信息并提供深入分析。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM获取响应
        response = self.llm_client.get_response(messages)
        
        return {
            "query": query,
            "prompt": prompt,
            "analysis": response
        }

class AcademicGroupMeetingMap(BaseSemanticMap):
    """学术组会语义图实现，继承自BaseSemanticMap"""
    def __init__(self, 
                use_local_model=True,
                image_embedding_model="clip-ViT-B-32",
                text_embedding_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                local_text_model_path="/home/zyh/model/clip-ViT-B-32-multilingual-v1",
                local_image_model_path="/home/zyh/model/clip-ViT-B-32",
                embedding_dim=512,
                index_type="flat"):
        """
        初始化学术组会语义地图
        
        Args:
            use_local_model: 是否使用本地模型
            image_embedding_model: 远程图像嵌入模型名称
            text_embedding_model: 远程文本嵌入模型名称
            local_text_model_path: 本地文本嵌入模型路径
            local_image_model_path: 本地图像嵌入模型路径
            embedding_dim: 向量维度，通常CLIP为512
            index_type: 索引类型，flat或hnsw
        """
        # 初始化基本参数
        self.use_local_model = use_local_model
        self.local_text_model_path = local_text_model_path
        self.local_image_model_path = local_image_model_path
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.data = []
        self.index = None
        
        # 初始化索引
        self._init_index()
        
        # 初始化文本和图像嵌入模型
        self._init_embedding_models(
            use_local_model, 
            image_embedding_model, 
            text_embedding_model, 
            local_text_model_path,
            local_image_model_path
        )
    
    def _init_index(self):
        """初始化向量索引"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "hnsw":
            self.index = faiss.index_factory(self.embedding_dim, "HNSW32,Flat")
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
    
    def _init_embedding_models(self, 
                              use_local_model, 
                              image_embedding_model, 
                              text_embedding_model,
                              local_text_model_path,
                              local_image_model_path):
        """初始化文本和图像嵌入模型，优先使用本地模型"""
        
        # 加载图像模型
        try:
            if use_local_model and os.path.exists(local_image_model_path):
                print(f"加载本地图像嵌入模型: {local_image_model_path}")
                # # 检查模型文件完整性
                # if not os.path.exists(os.path.join(local_image_model_path, "config_sentence_transformers.json")):
                #     print("警告: 模型目录似乎不完整，可能是git-lfs问题")
                #     print("提示: 请确保完整下载模型文件")
                #     raise FileNotFoundError("模型文件不完整")
                    
                self.image_model = SentenceTransformer(local_image_model_path)
                print("成功加载本地图像模型")
            else:
                print(f"加载远程图像嵌入模型: {image_embedding_model}")
                self.image_model = SentenceTransformer(image_embedding_model)
        except Exception as e:
            print(f"加载图像嵌入模型失败: {str(e)}")
            print("回退到使用远程图像模型")
            try:
                self.image_model = SentenceTransformer(image_embedding_model)
            except Exception as e2:
                print(f"远程图像模型也加载失败: {str(e2)}")
                self.image_model = None
        
        # 加载文本模型
        try:
            if use_local_model and os.path.exists(local_text_model_path):
                print(f"加载本地文本嵌入模型: {local_text_model_path}")
                # 检查模型文件完整性
                config_path = os.path.join(local_text_model_path, "config.json")
                if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
                    self.text_model = SentenceTransformer(local_text_model_path)
                    print("成功加载本地文本模型")
                else:
                    print("警告: 模型文件可能损坏")
                    raise FileNotFoundError("模型配置文件不完整或损坏")
            else:
                print(f"加载远程文本嵌入模型: {text_embedding_model}")
                self.text_model = SentenceTransformer(text_embedding_model)
        except Exception as e:
            print(f"加载文本嵌入模型失败: {str(e)}")
            print("回退到远程模型")
            try:
                self.text_model = SentenceTransformer(text_embedding_model)
                print("成功加载远程文本模型")
            except Exception as e2:
                print(f"远程文本模型也加载失败: {str(e2)}")
                # 尝试备用模型
                try:
                    print("尝试加载备用文本模型: paraphrase-multilingual-MiniLM-L12-v2")
                    self.text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    print("成功加载备用文本模型")
                except Exception as e3:
                    print(f"所有文本模型加载均失败: {str(e3)}")
                    raise RuntimeError("无法加载任何文本嵌入模型")
    
    def _get_text_embedding(self, text: str):
        """获取文本的向量表示，优化错误处理"""
        # 验证输入文本
        if not text or not isinstance(text, str):
            print("警告：嵌入生成收到无效文本，使用默认文本代替")
            text = "默认文本内容用于生成嵌入向量"
        
        # 截断超长文本，模型通常有输入长度限制
        max_length = 8000
        if len(text) > max_length:
            print(f"文本长度超过限制({len(text)} > {max_length})，将被截断")
            text = text[:max_length]
        
        try:
            # 使用加载的模型获取嵌入
            emb = self.text_model.encode(text)
            return emb.astype(np.float32)
        except Exception as e:
            print(f"文本嵌入生成失败: {str(e)}")
            # 如果出错，返回零向量
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
    def insert(self, key: str, value: Dict[str, Any], datatype: BaseDataType):
        """
        插入学术相关数据到语义图谱中
        根据不同的数据类型构建嵌入向量
        
        Args:
            key: 数据唯一标识
            value: 数据内容
            datatype: 数据类型
        """
        # 提取文本，用于计算嵌入向量
        text = ""
        if isinstance(value, dict):
            # 根据不同类型处理
            if datatype == AcademicDataType.Professor or datatype == AcademicDataType.PhD or datatype == AcademicDataType.Master:
                # 处理人员类型
                text = f"{value.get('Nickname', '')} {value.get('Specialty', '')} {value.get('Personality', '')} {value.get('Background', '')}"
            elif datatype == AcademicDataType.ResearchTopic:
                # 处理研究主题
                text = f"{value.get('Title', '')} {value.get('Description', '')} {value.get('Keywords', '')}"
            elif datatype == AcademicDataType.Paper:
                # 处理论文
                text = f"{value.get('Title', '')} {value.get('Abstract', '')} {value.get('Authors', '')} {value.get('Keywords', '')}"
            elif datatype == AcademicDataType.Discussion:
                # 处理讨论内容
                text = f"{value.get('Content', '')} {value.get('Speaker', '')} {value.get('Topic', '')}"
            elif datatype == AcademicDataType.Conclusion:
                # 处理结论
                text = f"{value.get('Summary', '')} {value.get('Implications', '')} {value.get('FutureDirections', '')}"
            else:
                # 默认处理：连接所有字符串值
                for k, v in value.items():
                    if isinstance(v, str):
                        text += v + " "
        
        # 获取嵌入向量
        embedding = self._get_text_embedding(text)
        
        # 保存数据和嵌入向量
        self.data.append((key, value, datatype, embedding))
        
        # 更新索引
        if self.index is not None:
            self.index.add(embedding.reshape(1, -1))
        
        return True
    
    def retrieve_papers(self, topic: str, k=5):
        """
        根据主题检索相关论文
        
        Args:
            topic: 查询主题
            k: 返回的结果数量
            
        Returns:
            list: 相关论文列表
        """
        # 获取主题的嵌入向量
        topic_embedding = self._get_text_embedding(topic)
        
        # 保存所有论文数据
        papers = []
        for key, value, dt, embedding in self.data:
            if dt == AcademicDataType.Paper:
                # 计算相似度
                similarity = np.dot(topic_embedding, embedding) / (np.linalg.norm(topic_embedding) * np.linalg.norm(embedding))
                papers.append((key, value, similarity))
        
        # 按相似度排序
        papers.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前k个结果
        return papers[:k]
    
    def get_research_network(self, researcher_key: str):
        """
        获取研究者的研究网络（相关主题、论文等）
        
        Args:
            researcher_key: 研究者的键
            
        Returns:
            dict: 研究网络数据
        """
        network = {
            "topics": [],
            "papers": [],
            "collaborators": []
        }
        
        # 查找该研究者的所有相关数据
        researcher = None
        for key, value, dt, _ in self.data:
            if key == researcher_key:
                researcher = value
                break
        
        if not researcher:
            return network
        
        # 找出相关主题、论文和合作者
        for key, value, dt, embedding in self.data:
            if dt == AcademicDataType.ResearchTopic and "Authors" in value and researcher_key in value.get("Authors", []):
                network["topics"].append(value)
            elif dt == AcademicDataType.Paper and "Authors" in value:
                authors = value.get("Authors", [])
                if researcher_key in authors:
                    network["papers"].append(value)
                    # 添加合作者
                    for author in authors:
                        if author != researcher_key:
                            for k, v, d, _ in self.data:
                                if k == author:
                                    network["collaborators"].append(v)
                                    break
        
        return network

class AcademicGroupMeetingGraph(BaseSemanticSimpleGraph):

    """学术组会语义简单图实现，继承自BaseSemanticSimpleGraph"""
    def __init__(self, semantic_map):
        super().__init__(semantic_map)
        # 命名空间相关的数据结构
        self.namespace_nodes = {
            NamespaceType.DIALOGUE: set(),
            NamespaceType.ATTACHMENT: set(),
            NamespaceType.USER: set(),
            NamespaceType.TASK: set(),
            NamespaceType.SUMMARY: set()
        }
        
        # 数据类型到命名空间的映射
        self.datatype_namespace_map = {
            # 用户命名空间
            AcademicDataType.Professor: NamespaceType.USER,
            AcademicDataType.PhD: NamespaceType.USER,
            AcademicDataType.Master: NamespaceType.USER,
            
            # 任务命名空间
            AcademicDataType.ResearchTopic: NamespaceType.TASK,
            AcademicDataType.Task: NamespaceType.TASK,
            
            # 对话命名空间
            AcademicDataType.Question: NamespaceType.DIALOGUE,
            AcademicDataType.Discussion: NamespaceType.DIALOGUE,
            AcademicDataType.Conclusion: NamespaceType.DIALOGUE,
            
            # 附件命名空间
            AcademicDataType.Paper: NamespaceType.ATTACHMENT,
            AcademicDataType.Reference: NamespaceType.ATTACHMENT,
            AcademicDataType.Presentation: NamespaceType.ATTACHMENT,
            
            # 总结命名空间
            AcademicDataType.Summary: NamespaceType.SUMMARY
        }
        
        # 子图数据结构
        self.sub_graphs = {
            NamespaceType.DIALOGUE: {"nodes": set(), "edges": []},
            NamespaceType.ATTACHMENT: {"nodes": set(), "edges": []},
            NamespaceType.USER: {"nodes": set(), "edges": []}, 
            NamespaceType.TASK: {"nodes": set(), "edges": []},
            NamespaceType.SUMMARY: {"nodes": set(), "edges": []}
        }

    def add_node(self, key: str, value: dict, datatype: BaseDataType, 
            parent_keys: List[str] = None, parent_relation="link"):
        """
        添加节点并建立父子关系，支持命名空间分类
        
        Args:
            key: 节点唯一标识
            value: 节点值/数据
            datatype: 数据类型
            parent_keys: 父节点列表
            parent_relation: 与父节点的关系类型
        """
        # 调用原始方法添加节点到语义图
        self.semantic_map.insert(key, value, datatype)
        self._ensure_node(key)
        
        # 分配节点到对应命名空间
        namespace = self.datatype_namespace_map.get(datatype)
        if namespace:
            self.namespace_nodes[namespace].add(key)
            self.sub_graphs[namespace]["nodes"].add(key)
        
        if parent_keys:
            for parent in parent_keys:
                self._ensure_node(parent)
                
                # 获取父节点数据类型
                parent_data = None
                parent_type = None
                for k, v, dt, _ in self.semantic_map.data:
                    if k == parent:
                        parent_data = v
                        parent_type = dt
                        break
                        
                # 智能确定关系类型（如果parent_relation是默认的"link"）
                relation = parent_relation
                if relation == "link" and parent_type and datatype:
                    relation = self._determine_relation_type(parent_type, datatype, parent, key)
                    
                # 建立父子关系
                self.graph_relations[parent]["children"][key] = relation
                self.graph_relations[key]["parents"][parent] = relation
                
                # 获取父节点的命名空间
                parent_namespace = self.datatype_namespace_map.get(parent_type)
                
                # 记录边到对应子图
                if namespace and parent_namespace:
                    if namespace == parent_namespace:
                        # 同一命名空间内的边
                        self.sub_graphs[namespace]["edges"].append((parent, key, relation))
                    else:
                        # 跨命名空间的边，添加到两个子图
                        self.sub_graphs[parent_namespace]["edges"].append((parent, key, relation))
                        self.sub_graphs[namespace]["edges"].append((parent, key, relation))
                        
    def _determine_relation_type(self, parent_type, child_type, parent_id, child_id):
        """根据节点类型确定更具体的关系类型"""
        # 用户 -> 对话关系
        if parent_type in [AcademicDataType.Professor, AcademicDataType.PhD, AcademicDataType.Master]:
            if child_type == AcademicDataType.Discussion:
                return "发言"
            elif child_type == AcademicDataType.Question:
                return "提问"
            elif child_type == AcademicDataType.Conclusion:
                return "总结"
            # 添加任务分配关系
            elif child_type == AcademicDataType.Task:
                return "分配任务"
        # 任务 -> 用户关系
        if parent_type == AcademicDataType.Task:
            if child_type in [AcademicDataType.Professor, AcademicDataType.PhD, AcademicDataType.Master]:
                return "执行者"   
                 
        # 话题 -> 用户关系
        if parent_type == AcademicDataType.ResearchTopic:
            if child_type in [AcademicDataType.Professor, AcademicDataType.PhD, AcademicDataType.Master]:
                return "参与者"
                
        # 话题 -> 对话关系
        if parent_type == AcademicDataType.ResearchTopic:
            if child_type in [AcademicDataType.Discussion, AcademicDataType.Question]:
                return "相关讨论"
                
        # 附件 -> 任务关系
        if parent_type == AcademicDataType.Paper:
            if child_type == AcademicDataType.ResearchTopic:
                return "支持"
        
        # 对话 -> 对话关系
        if parent_type in [AcademicDataType.Discussion, AcademicDataType.Question]:
            if child_type == AcademicDataType.Discussion:
                return "回应"
            elif child_type == AcademicDataType.Conclusion:
                return "引出"
                
        # 总结 -> 对话关系
        if parent_type == AcademicDataType.Summary:
            if child_type in [AcademicDataType.Discussion, AcademicDataType.Question]:
                return "总结"
        
        # 对话 -> 总结关系
        if child_type == AcademicDataType.Summary:
            if parent_type in [AcademicDataType.Discussion, AcademicDataType.Question]:
                return "被总结"
                
        # 任务 -> 总结关系
        if parent_type == AcademicDataType.Summary:
            if child_type == AcademicDataType.Task:
                return "产生任务"
                
        # 总结 -> 任务关系
        if child_type == AcademicDataType.Summary:
            if parent_type == AcademicDataType.Task:
                return "来源于总结"
            
        # 任务之间的依赖关系
        if parent_type == AcademicDataType.Task and child_type == AcademicDataType.Task:
            return "依赖于"
                
        return "关联"
    
    def get_namespace_nodes(self, namespace: NamespaceType) -> List[str]:
        """获取指定命名空间中的所有节点ID"""
        return list(self.namespace_nodes.get(namespace, set()))
    
    def get_node_namespace(self, node_id: str) -> NamespaceType:
        """获取节点所属的命名空间"""
        for namespace, nodes in self.namespace_nodes.items():
            if (node_id in nodes):
                return namespace
        return None
    
    def get_namespace_subgraph(self, namespace: NamespaceType):
        """获取指定命名空间的子图"""
        return self.sub_graphs.get(namespace)
    
    def get_cross_namespace_edges(self):
        """获取所有跨命名空间的边"""
        cross_edges = []
        for src_key in self.graph_relations:
            src_namespace = self.get_node_namespace(src_key)
            if not src_namespace:
                continue
                
            for dst_key, relation in self.graph_relations[src_key]["children"].items():
                dst_namespace = self.get_node_namespace(dst_key)
                if dst_namespace and src_namespace != dst_namespace:
                    cross_edges.append((src_key, dst_key, relation, src_namespace, dst_namespace))
        
        return cross_edges
    
    def update_namespace_subgraphs(self):
        """更新各命名空间的子图结构"""
        # 重置子图
        for namespace in self.sub_graphs:
            self.sub_graphs[namespace] = {"nodes": set(), "edges": []}
            
        # 分配节点到子图
        for namespace, nodes in self.namespace_nodes.items():
            self.sub_graphs[namespace]["nodes"] = nodes.copy()
            
        # 分配边到子图 
        for src_key in self.graph_relations:
            src_namespace = self.get_node_namespace(src_key)
            if not src_namespace:
                continue
                
            for dst_key, relation in self.graph_relations[src_key]["children"].items():
                dst_namespace = self.get_node_namespace(dst_key)
                if not dst_namespace:
                    continue
                    
                edge = (src_key, dst_key, relation)
                
                # 同命名空间边
                if src_namespace == dst_namespace:
                    self.sub_graphs[src_namespace]["edges"].append(edge)
                else:
                    # 跨命名空间边，记录在两个子图中
                    self.sub_graphs[src_namespace]["edges"].append(edge)
                    self.sub_graphs[dst_namespace]["edges"].append(edge)
    
    def visualize_namespaces(self, title="学术组会命名空间关系图", fontpath=None, output_filename="academic_meeting_namespaces.png"):
        """改进的命名空间可视化方法"""
        # 使用通用设置
        G = self._common_visualization_setup(title)
        
        # 命名空间颜色映射
        namespace_colors = {
            NamespaceType.USER: "skyblue",
            NamespaceType.DIALOGUE: "lightgreen",
            NamespaceType.ATTACHMENT: "salmon",
            NamespaceType.TASK: "gold",
            NamespaceType.SUMMARY: "orchid"  # 为总结命名空间添加颜色
        }
        
        # 命名空间形状映射
        namespace_shapes = {
            NamespaceType.USER: "o",      # 圆形
            NamespaceType.DIALOGUE: "s",  # 方形
            NamespaceType.ATTACHMENT: "^", # 三角形
            NamespaceType.TASK: "d",      # 钻石形
            NamespaceType.SUMMARY: "h"    # 六边形，用于总结
        }
        
        # 添加节点
        node_colors = []
        node_shapes = []
        node_labels = {}
        node_namespaces = {}
        
        for key in self.graph_relations:
            # 获取节点数据
            node_data = None
            node_type = None
            for k, value, datatype, _ in self.semantic_map.data:
                if k == key:
                    node_data = value
                    node_type = datatype
                    break
            
            if not node_data or not node_type:
                continue
                
            # 获取命名空间
            namespace = self.datatype_namespace_map.get(node_type)
            if not namespace:
                continue
                
            # 添加到图
            G.add_node(key)
            node_namespaces[key] = namespace
            
            # 记录颜色和形状
            node_colors.append(namespace_colors.get(namespace, "gray"))
            node_shapes.append(namespace_shapes.get(namespace, "o"))
            
            # 设置节点标签
            label = ""
            if isinstance(node_data, dict):
                if "Nickname" in node_data:
                    label = node_data["Nickname"]
                elif "Title" in node_data:
                    label = node_data["Title"]
                elif "Content" in node_data:
                    content = node_data["Content"]
                    label = content[:20] + "..." if len(content) > 20 else content
            
            node_labels[key] = label or key
        
        # 添加边
        edges = []
        edge_colors = []
        edge_styles = []
        edge_labels = {}
        
        for src in self.graph_relations:
            if src not in node_namespaces:
                continue
                
            src_namespace = node_namespaces[src]
            
            for dst, relation in self.graph_relations[src]["children"].items():
                if dst not in node_namespaces:
                    continue
                    
                dst_namespace = node_namespaces[dst]
                
                # 添加边
                G.add_edge(src, dst)
                edges.append((src, dst))
                
                # 设置边样式
                if src_namespace == dst_namespace:
                    # 同命名空间内的边
                    edge_colors.append(namespace_colors.get(src_namespace, "gray"))
                    edge_styles.append("solid")
                else:
                    # 跨命名空间的边
                    edge_colors.append("red")
                    edge_styles.append("dashed")
                    
                # 边标签
                edge_labels[(src, dst)] = relation
        
        # 计算布局
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # 绘制节点
        self._setup_node_visualization(G, node_colors, node_shapes, pos)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5,
                        edge_color=edge_colors, style=edge_styles,
                        connectionstyle='arc3,rad=0.1', 
                        arrowsize=15, alpha=0.7)
        
        # 绘制节点标签
        if fontpath:
            font_prop = FontProperties(fname=fontpath)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, 
                                font_family=font_prop.get_family())
        else:
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # 绘制边标签
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                    font_size=8, alpha=0.7)
        
        # 添加图例
        legend_elements = []
        for ns, color in namespace_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10,
                                        label=ns.name))
        
        # 添加边类型图例
        legend_elements.append(plt.Line2D([0], [0], color='gray', 
                                    linestyle='solid', linewidth=2,
                                    label='同命名空间关系'))
        legend_elements.append(plt.Line2D([0], [0], color='red',
                                    linestyle='dashed', linewidth=2,
                                    label='跨命名空间关系'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 设置布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_filename, dpi=300)
        plt.close()
        
        print(f"已保存命名空间分组的学术组会图到: {output_filename}")

    def auto_generate_subgraphs(self):
        """根据命名空间自动生成子图，优化后的版本"""
        # 重置命名空间节点分配
        for namespace in self.namespace_nodes:
            self.namespace_nodes[namespace] = set()
            
        # 遍历所有节点分配到命名空间
        for key, value, datatype, _ in self.semantic_map.data:
            namespace = self.datatype_namespace_map.get(datatype)
            if namespace:
                self.namespace_nodes[namespace].add(key)
        
        # 更新子图结构
        self.update_namespace_subgraphs()
        
        # 打印命名空间统计
        print("\n命名空间节点统计:")
        for namespace, nodes in self.namespace_nodes.items():
            print(f"  {namespace.name}: {len(nodes)}个节点")
        
        # 统计跨命名空间边
        cross_edges = self.get_cross_namespace_edges()
        print(f"跨命名空间关系: {len(cross_edges)}个")
        
        return self.sub_graphs
    
    def _common_visualization_setup(self, title, figsize=(18, 14)):
        """提取通用的可视化设置代码"""
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=16)
        plt.axis('off')
        return nx.DiGraph()
    
    def _setup_node_visualization(self, G, node_colors, node_shapes, pos, node_size=800, alpha=0.8):
        """设置节点可视化的通用方法"""
        for shape in set(node_shapes):
            # 获取该形状的节点索引
            node_idx = [i for i, s in enumerate(node_shapes) if s == shape]
            if not node_idx:
                continue
                
            # 获取节点子集
            node_subset = [list(G.nodes())[i] for i in node_idx]
            node_color_subset = [node_colors[i] for i in node_idx]
            
            # 根据不同形状调整节点大小
            size = node_size
            if shape == "s":  # 方形稍小
                size = node_size * 0.9
            elif shape == "^":  # 三角形稍大
                size = node_size * 1.1
                
            # 绘制节点
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=node_subset, 
                node_color=node_color_subset, 
                node_shape=shape,
                node_size=size, 
                alpha=alpha
            )
    
    def visualize_academic_meeting(self, title="学术组会关系图", fontpath=None):
        """
        可视化学术组会的互动关系，改进版
        
        Args:
            title: 图表标题
            fontpath: 字体路径，支持中文
        """
        plt.figure(figsize=(18, 14))  # 增大图形尺寸
        
        # 创建 NetworkX 图
        G = nx.DiGraph()
        
        # 节点类型映射字典，用于后续生成有意义的关系
        node_types = {}
        node_data = {}
        
        # 添加节点和属性
        for key in self.graph_relations:
            data = self.semantic_map.get(key)
            if not data:
                continue
                
            value = data.get("value", {})
            datatype = data.get("datatype")
            node_types[key] = datatype  # 存储节点类型
            node_data[key] = value      # 存储节点数据
            
            # 设置节点标签
            if datatype in [AcademicDataType.Professor, AcademicDataType.PhD, AcademicDataType.Master]:
                G.add_node(key, label=value.get("Nickname", key), type=str(datatype))
            elif datatype == AcademicDataType.ResearchTopic:
                G.add_node(key, label=value.get("Title", key), type=str(datatype))
            elif datatype == AcademicDataType.Paper:
                G.add_node(key, label=value.get("Title", key)[:20] + "...", type=str(datatype))
            else:
                # 处理其他类型节点，截取内容前20个字符
                content = ""
                if isinstance(value, dict) and "Content" in value:
                    content = value["Content"]
                elif isinstance(value, str):
                    content = value
                    
                if len(content) > 20:
                    content = content[:20] + "..."
                    
                G.add_node(key, label=content or key, type=str(datatype))
        
        # 添加边并根据节点类型生成有意义的关系
        for src in self.graph_relations:
            for dst in self.graph_relations[src]["children"]:
                # 根据源和目标节点类型生成有意义的关系
                relation = self._generate_meaningful_relation(
                    src, dst, 
                    node_types.get(src), 
                    node_types.get(dst), 
                    self.graph_relations[src]["children"][dst],
                    node_data.get(src, {}),
                    node_data.get(dst, {})
                )
                G.add_edge(src, dst, relation=relation)
        
        # 计算节点位置，增加k值以增大节点间距
        pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), seed=42, iterations=100)
        
        # 设置字体
        font = FontProperties(fname=fontpath) if fontpath else None
        
        # 获取节点属性
        node_types = nx.get_node_attributes(G, 'type')
        node_labels = nx.get_node_attributes(G, 'label')
        
        # 定义不同类型节点的颜色和大小
        colors = {
            str(AcademicDataType.Professor): 'red',
            str(AcademicDataType.PhD): 'orange',
            str(AcademicDataType.Master): 'yellow',
            str(AcademicDataType.ResearchTopic): 'green',
            str(AcademicDataType.Paper): 'blue',
            str(AcademicDataType.Discussion): 'purple',
            str(AcademicDataType.Question): 'cyan',
            str(AcademicDataType.Conclusion): 'magenta',
            str(AcademicDataType.Reference): 'teal',
            str(AcademicDataType.Presentation): 'brown'
        }
        
        sizes = {
            str(AcademicDataType.Professor): 2500,
            str(AcademicDataType.PhD): 2000,
            str(AcademicDataType.Master): 1800,
            str(AcademicDataType.ResearchTopic): 3000,
            str(AcademicDataType.Paper): 2200,
            str(AcademicDataType.Discussion): 1500,
            str(AcademicDataType.Question): 1000,
            str(AcademicDataType.Conclusion): 2000,
            str(AcademicDataType.Reference): 1800,
            str(AcademicDataType.Presentation): 2300
        }
        
        # 按类型分类节点
        for node_type in set(node_types.values()):
            nodes = [n for n, t in node_types.items() if t == node_type]
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=nodes,
                node_color=colors.get(node_type, 'lightgray'),
                node_size=sizes.get(node_type, 1500),
                alpha=0.8
            )
        
        # 根据不同的关系类型绘制不同样式的边
        edge_styles = {
            "发言": {"color": "blue", "style": "solid", "width": 2.0},
            "提问": {"color": "red", "style": "dashed", "width": 1.5},
            "回答": {"color": "green", "style": "solid", "width": 2.0},
            "指导": {"color": "purple", "style": "solid", "width": 2.5},
            "研究": {"color": "orange", "style": "dashed", "width": 1.5},
            "引用": {"color": "brown", "style": "dotted", "width": 1.0},
            "总结": {"color": "black", "style": "solid", "width": 3.0},
            "link": {"color": "gray", "style": "dotted", "width": 1.0}  # 默认样式
        }
        
        # 按关系类型绘制边
        for relation_type, style in edge_styles.items():
            # 筛选当前关系类型的边
            edges = [(u, v) for u, v, d in G.edges(data=True) 
                    if relation_type in d.get("relation", "").lower()]
            
            if edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    width=style["width"],
                    edge_color=style["color"],
                    style=style["style"],
                    alpha=0.7,
                    arrowsize=15
                )
        
        # 绘制节点标签，优化位置避免重叠
        label_pos = {}
        for node in G.nodes():
            # 随机微调标签位置，避免重叠
            offset = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
            label_pos[node] = pos[node] + offset
        
        # 绘制标签
        for node, (x, y) in label_pos.items():
            label = node_labels.get(node, "")
            node_type = node_types.get(node)
            
            # 根据节点类型调整字体大小和颜色
            fontsize = 10
            color = 'black'
            
            if node_type in [str(AcademicDataType.Professor), str(AcademicDataType.ResearchTopic)]:
                fontsize = 12
                color = 'darkred'
            elif node_type in [str(AcademicDataType.PhD), str(AcademicDataType.Master)]:
                fontsize = 11
                color = 'darkblue'
            elif node_type == str(AcademicDataType.Conclusion):
                fontsize = 11
                color = 'darkgreen'
                
            plt.text(x, y, label, fontsize=fontsize, fontproperties=font,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'),
                    color=color)
        
        # 绘制边标签，错开位置避免与节点标签重叠
        for u, v, data in G.edges(data=True):
            relation = data.get('relation', '')
            # 计算边的中间点，并添加随机偏移
            edge_x = (pos[u][0] + pos[v][0]) / 2 + np.random.uniform(-0.05, 0.05)
            edge_y = (pos[u][1] + pos[v][1]) / 2 + np.random.uniform(-0.05, 0.05)
            
            # 使用带背景的文本框显示关系
            plt.text(edge_x, edge_y, relation, fontsize=8, fontproperties=font,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, boxstyle='round'),
                color='navy')
        
        # 添加图例
        legend_elements = []
        for node_type, color in colors.items():
            if node_type in [str(t) for t in AcademicDataType]:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    label=node_type.split('.')[-1],
                                    markerfacecolor=color, markersize=10))
        
        # 添加边类型图例
        for rel_type, style in edge_styles.items():
            if rel_type != "link":  # 排除默认类型
                legend_elements.append(plt.Line2D([0], [0], 
                                    label=rel_type,
                                    color=style["color"], 
                                    linestyle=style["style"],
                                    linewidth=style["width"]))
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 设置标题和显示
        if font:
            plt.title(title, fontproperties=font, fontsize=16)
        else:
            plt.title(title, fontsize=16)
            
        plt.axis('off')
        plt.tight_layout()
        
        # 保存和显示
        plt.savefig(f"./semantic_map_case/academic_group_meeting/output/picture/academic_meeting_graph.png", dpi=300)
        # plt.show()
        print(f"学术组会图已保存为 academic_meeting_graph.png")

    def _generate_meaningful_relation(self, src_key, dst_key, src_type, dst_type, default_relation, src_data, dst_data):
        """
        根据节点类型生成有意义的关系描述
        
        Args:
            src_key: 源节点键
            dst_key: 目标节点键
            src_type: 源节点类型
            dst_type: 目标节点类型
            default_relation: 默认关系
            src_data: 源节点数据
            dst_data: 目标节点数据
            
        Returns:
            str: 有意义的关系描述
        """
        # 如果默认关系不是link，直接使用
        if default_relation != "link":
            return default_relation
            
        # 根据节点类型组合生成关系
        if src_type == AcademicDataType.Professor:
            if dst_type == AcademicDataType.Discussion:
                return "发言"
            elif dst_type == AcademicDataType.Question:
                return "提问"
            elif dst_type == AcademicDataType.PhD or dst_type == AcademicDataType.Master:
                return "指导"
            elif dst_type == AcademicDataType.Conclusion:
                return "总结"
            elif dst_type == AcademicDataType.ResearchTopic:
                return "研究"
                
        elif src_type == AcademicDataType.PhD or src_type == AcademicDataType.Master:
            if dst_type == AcademicDataType.Discussion:
                return "发言"
            elif dst_type == AcademicDataType.Question:
                return "提问"
            elif dst_type == AcademicDataType.Professor:
                return "汇报"
            elif dst_type == AcademicDataType.ResearchTopic:
                return "研究"
                
        elif src_type == AcademicDataType.Discussion:
            if dst_type == AcademicDataType.Discussion:
                return "延续"
            elif dst_type == AcademicDataType.Question:
                return "引发"
            elif dst_type == AcademicDataType.Conclusion:
                return "导致"
                
        elif src_type == AcademicDataType.Question:
            if dst_type == AcademicDataType.Discussion:
                return "回答"
                
        # 特殊处理：检查src_data和dst_data中的内容生成更精确的关系
        if "Speaker" in src_data and "Speaker" in dst_data:
            if src_data["Speaker"] == dst_data["Speaker"]:
                return "继续发言"
        
        # 如果是研究主题连接到人，返回"参与"
        if src_type == AcademicDataType.ResearchTopic and(dst_type == AcademicDataType.Professor or
                                                        dst_type == AcademicDataType.PhD or
                                                        dst_type == AcademicDataType.Master):
            return "参与者"
            
        # 默认返回节点类型组合
        if src_type and dst_type:
            return f"{src_type.name}-{dst_type.name}"
            
        # 最后的默认值
        return "关联"
    
    def get_discussion_flow(self):
        """获取讨论流程图，按时间顺序排列讨论内容"""
        # 获取所有讨论节点
        discussion_nodes = []
        for key in self.graph_relations:
            node_data = self.semantic_map.get(key)
            if not node_data:
                continue
                
            dt = node_data.get("datatype")
            if dt in [AcademicDataType.Discussion, AcademicDataType.Question, AcademicDataType.Conclusion]:
                data = node_data.get("value", {})
                timestamp = data.get("Timestamp", "")
                discussion_nodes.append((key, timestamp))
        
        # 按时间戳排序
        discussion_nodes.sort(key=lambda x: x[1])
        
        # 创建流程图
        flow = []
        for key, _ in discussion_nodes:
            node_data = self.semantic_map.get(key)
            if not node_data:
                continue
                
            data = node_data.get("value", {})
            speaker = data.get("Speaker", "")
            content = data.get("Content", "")
            topic = data.get("Topic", "")
            
            flow.append({
                "key": key,
                "speaker": speaker,
                "content": content,
                "topic": topic
            })
        
        return flow

