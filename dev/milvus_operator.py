import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from pymilvus import (
    connections, 
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType,
    Collection,
    MilvusException
)

from Hippo import MemoryUnit, MemorySpace

class MilvusOperator:
    """Milvus数据库操作类，提供向量存储和检索功能"""
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: str = "19530",
        user: str = "", 
        password: str = "",
        collection_name: str = "hippo_memory_units",
        embedding_dim: int = 512
    ):
        """
        初始化Milvus操作类
        
        参数:
            host: Milvus服务器地址
            port: Milvus服务器端口
            user: 用户名（如果需要认证）
            password: 密码（如果需要认证）
            collection_name: 集合名称
            embedding_dim: 向量维度
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.is_connected = False
        
        try:
            connections.connect(
                alias="default", 
                host=host, 
                port=port,
                user=user,
                password=password
            )
            self.is_connected = True
            logging.info(f"成功连接到Milvus服务器: {host}:{port}")
        except Exception as e:
            logging.error(f"连接到Milvus服务器失败: {e}")
            self.is_connected = False
    
    def create_collection(self) -> bool:
        """创建集合"""
        if not self.is_connected:
            logging.error("未连接到Milvus服务器")
            return False
        
        if utility.has_collection(self.collection_name):
            logging.info(f"集合 '{self.collection_name}' 已存在")
            self.collection = Collection(self.collection_name)
            self.collection.load()
            return True
        
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="space_names", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=100),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(
                fields, 
                description=f"Hippo memory units with {self.embedding_dim}-dimensional vectors"
            )
            
            self.collection = Collection(
                name=self.collection_name, 
                schema=schema,
                using='default'
            )
            
            # 创建向量索引
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            self.collection.create_index("vector", index_params)
            self.collection.load()
            
            logging.info(f"成功创建集合 '{self.collection_name}'")
            return True
        
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False
    
    def add_memory_unit(self, unit: MemoryUnit, space_names: Optional[List[str]] = None) -> bool:
        """
        添加MemoryUnit到Milvus
        
        参数:
            unit: 要添加的MemoryUnit对象
            space_names: 该单元所属的空间名称列表
        
        返回:
            添加是否成功
        """
        if not self.is_connected:
            logging.error("未连接到Milvus服务器")
            return False
        
        if not hasattr(self, 'collection'):
            self.create_collection()
        
        if unit.vector is None:
            logging.warning(f"单元 '{unit.id}' 没有向量表示，无法添加到Milvus")
            return False
        
        try:
            # 删除可能存在的旧实体
            self.delete_memory_unit(unit.id)
            
            # 准备数据
            data = [
                [unit.id],  # id
                [unit.vector.astype(np.float32).tolist()],  # vector
                [space_names or []],  # space_names
                [unit.value]  # metadata
            ]
            
            # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            logging.info(f"成功添加内存单元 '{unit.id}' 到Milvus")
            return True
            
        except Exception as e:
            logging.error(f"添加内存单元 '{unit.id}' 到Milvus失败: {e}")
            return False
    
    def delete_memory_unit(self, unit_id: str) -> bool:
        """删除Milvus中的MemoryUnit"""
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return False
        
        try:
            expr = f'id == "{unit_id}"'
            self.collection.delete(expr)
            logging.info(f"成功从Milvus删除内存单元 '{unit_id}'")
            return True
            
        except MilvusException as e:
            # 如果是因为实体不存在而失败，视为成功
            if "entity not found" in str(e).lower():
                return True
            logging.error(f"从Milvus删除内存单元 '{unit_id}' 失败: {e}")
            return False
    
    def search_similarity(
        self, 
        query_vector: np.ndarray, 
        k: int = 5, 
        space_name: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        在Milvus中进行相似性搜索
        
        参数:
            query_vector: 查询向量
            k: 返回的最相似结果数量
            space_name: 如果提供，则仅在特定空间内搜索
            
        返回:
            List[Tuple[str, Dict[str, Any], float]]: [(unit_id, metadata, distance), ...]
        """
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return []
        
        try:
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 64}
            }
            
            # 如果指定了空间，添加过滤条件
            expr = None
            if space_name:
                expr = f'array_contains(space_names, "{space_name}")'
            
            results = self.collection.search(
                data=[query_vector.astype(np.float32).tolist()],
                anns_field="vector",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=["id", "metadata"]
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    unit_id = hit.entity.get('id')
                    metadata = hit.entity.get('metadata')
                    distance = hit.distance
                    search_results.append((unit_id, metadata, distance))
            
            return search_results
            
        except Exception as e:
            logging.error(f"Milvus相似性搜索失败: {e}")
            return []
    
    def get_memory_unit(self, unit_id: str) -> Optional[Tuple[str, Dict[str, Any], Optional[np.ndarray]]]:
        """通过ID从Milvus获取MemoryUnit"""
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return None
        
        try:
            expr = f'id == "{unit_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "metadata", "vector"]
            )
            
            if not results:
                return None
            
            unit_data = results[0]
            vector = None
            if "vector" in unit_data:
                vector = np.array(unit_data["vector"], dtype=np.float32)
            
            return (unit_data["id"], unit_data["metadata"], vector)
            
        except Exception as e:
            logging.error(f"从Milvus获取内存单元 '{unit_id}' 失败: {e}")
            return None
    
    def get_all_memory_units(self) -> List[Tuple[str, Dict[str, Any], Optional[np.ndarray]]]:
        """获取Milvus中的所有MemoryUnit"""
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return []
        
        try:
            results = self.collection.query(
                expr="id != ''",
                output_fields=["id", "metadata", "vector"]
            )
            
            units = []
            for unit_data in results:
                vector = None
                if "vector" in unit_data:
                    vector = np.array(unit_data["vector"], dtype=np.float32)
                
                units.append((unit_data["id"], unit_data["metadata"], vector))
            
            return units
            
        except Exception as e:
            logging.error(f"从Milvus获取所有内存单元失败: {e}")
            return []
    
    def get_memory_units_by_space(self, space_name: str) -> List[Tuple[str, Dict[str, Any], Optional[np.ndarray]]]:
        """获取特定空间中的所有MemoryUnit"""
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return []
        
        try:
            expr = f'array_contains(space_names, "{space_name}")'
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "metadata", "vector"]
            )
            
            units = []
            for unit_data in results:
                vector = None
                if "vector" in unit_data:
                    vector = np.array(unit_data["vector"], dtype=np.float32)
                
                units.append((unit_data["id"], unit_data["metadata"], vector))
            
            return units
            
        except Exception as e:
            logging.error(f"从Milvus获取空间 '{space_name}' 的内存单元失败: {e}")
            return []
    
    def close(self):
        """关闭Milvus连接"""
        if self.is_connected:
            try:
                if hasattr(self, 'collection'):
                    self.collection.release()
                connections.disconnect("default")
                logging.info("已关闭Milvus连接")
            except Exception as e:
                logging.error(f"关闭Milvus连接失败: {e}")

if __name__ == "__main__":
    # 示例用法
    milvus_operator = MilvusOperator()
    milvus_operator.create_collection()
    
    # 添加内存单元
    unit = MemoryUnit(id="unit_1", vector=np.random.rand(512), value={"key": "value"})
    milvus_operator.add_memory_unit(unit, space_names=["space_1"])
    
    # 搜索相似内存单元
    results = milvus_operator.search_similarity(query_vector=np.random.rand(512), k=5)
    print("搜索结果:", results)
    
    # 获取特定内存单元
    unit_data = milvus_operator.get_memory_unit("unit_1")
    print("获取的内存单元:", unit_data)
    
    # 关闭连接
    milvus_operator.close()