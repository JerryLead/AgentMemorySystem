import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import networkx as nx

class MemorySpace:
    """
    内存空间 (MemorySpace) 用于组织和管理一组相关的内存单元。
    一个内存单元可以属于多个内存空间。
    """
    def __init__(self, name: str):
        """
        初始化一个内存空间。
        参数:
            name (str): 内存空间的名称，应唯一。
        """
        if not isinstance(name, str) or not name.strip():
            raise TypeError("内存空间名称 (name) 不能为空字符串。")
        self.name: str = name
        self.memory_uids: Set[str] = set() # 存储属于此空间的 MemoryUnit 的 ID

    def add_unit(self, uid: str):
        """向此内存空间添加一个内存单元的ID。"""
        if not isinstance(uid, str) or not uid.strip():
            logging.warning(f"尝试向内存空间 '{self.name}' 添加无效的单元ID。")
            return
        self.memory_uids.add(uid)
        logging.debug(f"内存单元 '{uid}' 已添加到内存空间 '{self.name}'。")

    def remove_unit(self, uid: str):
        """从此内存空间移除一个内存单元的ID。"""
        if uid in self.memory_uids:
            self.memory_uids.remove(uid)
            logging.debug(f"内存单元 '{uid}' 已从内存空间 '{self.name}' 移除。")
        else:
            logging.warning(f"尝试从内存空间 '{self.name}' 移除不存在的内存单元ID '{uid}'。")

    def add_units(self, uids: List[str]):
        """向此内存空间添加多个内存单元的ID。"""
        for uid in uids:
            if not isinstance(uid, str) or not uid.strip():
                logging.warning(f"尝试向内存空间 '{self.name}' 添加无效的单元ID '{uid}'。")
                continue
            self.add_unit(uid)
            logging.debug(f"内存单元 '{uid}' 已添加到内存空间 '{self.name}'。")

    def remove_units(self, uids: List[str]):
        """从此内存空间移除多个内存单元的ID。"""
        for uid in uids:
            if uid in self.memory_uids:
                self.remove_unit(uid)
            else:
                logging.warning(f"尝试从内存空间 '{self.name}' 移除不存在的内存单元ID '{uid}'。")

    def get_memory_uids(self) -> Set[str]:
        """获取此内存空间中所有内存单元的ID集合。"""
        return self.memory_uids

    def __repr__(self) -> str:
        return f"MemorySpace(name='{self.name}', unit_count={len(self.memory_uids)})"