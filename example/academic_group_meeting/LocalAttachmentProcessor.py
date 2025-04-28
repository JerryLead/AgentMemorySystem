#!/usr/bin/env python3
import os
import sys
import uuid
import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# PDF处理
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
from academic_group_meeting_graph import NamespaceType, AcademicDataType

class LocalAttachmentProcessor:
    """处理本地附件文件（PDF和图像）并添加到语义图中的类"""
    
    def __init__(self, attachment_dir: str = "./semantic_map_case/academic_group_meeting/attachment"):
        """
        初始化本地附件处理器
        
        Args:
            attachment_dir: 附件存储目录路径
        """
        self.attachment_dir = attachment_dir
        
        # 确保目录存在
        if not os.path.exists(attachment_dir):
            print(f"警告: 附件目录 '{attachment_dir}' 不存在")
        else:
            print(f"找到附件目录: '{attachment_dir}'")
        
        # 支持的文件类型
        self.pdf_extensions = ['.pdf']
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    def scan_attachments(self) -> Dict[str, List[str]]:
        """
        扫描附件目录中的所有支持文件
        
        Returns:
            Dict[str, List[str]]: 按类型分组的文件路径列表
        """
        result = {
            "pdfs": [],
            "images": []
        }
        
        if not os.path.exists(self.attachment_dir):
            return result
        
        for file in os.listdir(self.attachment_dir):
            file_path = os.path.join(self.attachment_dir, file)
            if not os.path.isfile(file_path):
                continue
                
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in self.pdf_extensions:
                result["pdfs"].append(file_path)
            elif file_ext in self.image_extensions:
                result["images"].append(file_path)
        
        return result
    
    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        从PDF文件中提取元数据
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Dict[str, Any]: 包含PDF元数据的字典
        """
        metadata = {
            "Title": os.path.basename(pdf_path),
            "Authors": "",
            "Abstract": "",
            "Published": "",
            "Pages": 0,
            "Keywords": "",
            "Source": "Local",
            "LocalPath": pdf_path,
            "AddedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Content": ""
        }
        
        try:
            # 使用PyMuPDF提取PDF元数据和内容
            doc = fitz.open(pdf_path)
            
            # 提取页数
            metadata["Pages"] = doc.page_count
            
            # 提取元数据
            pdf_metadata = doc.metadata
            if pdf_metadata:
                if pdf_metadata.get("title"):
                    metadata["Title"] = pdf_metadata.get("title")
                if pdf_metadata.get("author"):
                    metadata["Authors"] = pdf_metadata.get("author")
                if pdf_metadata.get("subject"):
                    metadata["Abstract"] = pdf_metadata.get("subject")
                if pdf_metadata.get("creationDate"):
                    # 处理PDF日期格式
                    date_str = pdf_metadata.get("creationDate")
                    if date_str.startswith("D:"):
                        date_str = date_str[2:10]  # 提取YYYYMMDD
                        try:
                            date_obj = datetime.strptime(date_str, "%Y%m%d")
                            metadata["Published"] = date_obj.strftime("%Y-%m-%d")
                        except:
                            pass
                if pdf_metadata.get("keywords"):
                    metadata["Keywords"] = pdf_metadata.get("keywords")
            
            # 提取前两页的文本作为摘要
            text = ""
            for i in range(min(2, doc.page_count)):
                page = doc[i]
                text += page.get_text()
            
            # 限制文本长度
            if len(text) > 1000:
                text = text[:1000] + "..."
                
            metadata["Content"] = text
            
            # 如果没有提取到标题，尝试从文件名中获取
            if metadata["Title"] == os.path.basename(pdf_path):
                filename = os.path.basename(pdf_path)
                # 移除扩展名和常见前缀
                clean_name = os.path.splitext(filename)[0]
                clean_name = re.sub(r'^\d+\.\d+v\d+_*', '', clean_name)  # 移除arXiv风格的前缀
                # 将下划线转换为空格并规范化标题
                clean_name = clean_name.replace('_', ' ')
                # 如果提取的内容合理，使用它作为标题
                if len(clean_name) > 3:
                    metadata["Title"] = clean_name
            
            # 关闭文档
            doc.close()
            
        except Exception as e:
            print(f"提取PDF元数据时出错 ({pdf_path}): {str(e)}")
        
        return metadata
    
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        从图像文件中提取元数据
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Dict[str, Any]: 包含图像元数据的字典
        """
        metadata = {
            "Title": os.path.basename(image_path),
            "Width": 0,
            "Height": 0,
            "Format": "",
            "Source": "Local",
            "LocalPath": image_path,
            "AddedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Description": ""
        }
        
        try:
            # 使用PIL提取图像信息
            with Image.open(image_path) as img:
                metadata["Width"] = img.width
                metadata["Height"] = img.height
                metadata["Format"] = img.format
        
        except Exception as e:
            print(f"提取图像元数据时出错 ({image_path}): {str(e)}")
        
        return metadata
    
    def add_pdf_to_semantic_graph(self, 
                                  backend, 
                                  pdf_path: str, 
                                  topic_key: str = None) -> str:
        """
        将本地PDF添加到语义图中
        
        Args:
            backend: 学术组会后端系统
            pdf_path: PDF文件路径
            topic_key: 关联的主题节点ID
            
        Returns:
            str: 添加的PDF节点ID
        """
        # 提取PDF元数据
        metadata = self.extract_pdf_metadata(pdf_path)
        
        # 创建唯一ID
        pdf_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        # 添加到语义图
        parent_keys = [topic_key] if topic_key else None
        backend.add_paper(pdf_id, metadata, parent_keys)
        
        print(f"论文 '{metadata['Title']}' 已添加到语义图 (ID: {pdf_id})")
        return pdf_id
    
    def add_image_to_semantic_graph(self, 
                                   backend, 
                                   image_path: str, 
                                   topic_key: str = None) -> str:
        """
        将本地图像添加到语义图中
        
        Args:
            backend: 学术组会后端系统
            image_path: 图像文件路径
            topic_key: 关联的主题节点ID
            
        Returns:
            str: 添加的图像节点ID
        """
        # 提取图像元数据
        metadata = self.extract_image_metadata(image_path)
        
        # 创建唯一ID
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        
        # 添加到语义图中的Reference类型
        parent_keys = [topic_key] if topic_key else None
        backend.semantic_graph.add_node(
            image_id, 
            metadata, 
            AcademicDataType.Reference,  # 使用Reference类型存储图片
            parent_keys
        )
        
        print(f"图片 '{metadata['Title']}' 已添加到语义图 (ID: {image_id})")
        return image_id
    
    def process_all_attachments(self, 
                               backend, 
                               topic_key: str = None) -> Dict[str, List[str]]:
        """
        处理所有本地附件并添加到语义图
        
        Args:
            backend: 学术组会后端系统
            topic_key: 关联的主题节点ID
            
        Returns:
            Dict[str, List[str]]: 添加的附件节点ID列表，按类型分组
        """
        result = {
            "pdfs": [],
            "images": []
        }
        
        # 扫描附件
        attachments = self.scan_attachments()
        
        # 处理PDF文件
        if attachments["pdfs"]:
            print(f"\n处理 {len(attachments['pdfs'])} 个PDF文件...")
            for pdf_path in attachments["pdfs"]:
                pdf_id = self.add_pdf_to_semantic_graph(backend, pdf_path, topic_key)
                result["pdfs"].append(pdf_id)
        
        # 处理图像文件
        if attachments["images"]:
            print(f"\n处理 {len(attachments['images'])} 个图像文件...")
            for image_path in attachments["images"]:
                image_id = self.add_image_to_semantic_graph(backend, image_path, topic_key)
                result["images"].append(image_id)
        
        # 更新命名空间子图
        backend.semantic_graph.auto_generate_subgraphs()
        
        print(f"\n附件处理完成：添加了 {len(result['pdfs'])} 篇论文和 {len(result['images'])} 张图片")
        return result