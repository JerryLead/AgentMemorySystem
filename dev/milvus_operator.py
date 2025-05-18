from datetime import datetime
import json
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

from dev.Hippo import MemoryUnit

# class DateTimeEncoder(json.JSONEncoder):
#     """处理datetime对象的JSON编码器"""
#     def default(self, obj):
#         if isinstance(obj, datetime):
#             return obj.isoformat()  # 将datetime转换为ISO格式字符串
#         return super().default(obj)

class MilvusOperator:
    """Milvus数据库操作类，专注于向量和数据存储与检索功能"""
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: str = "19530",
        user: str = "", 
        password: str = "",
        collection_name: str = "hippo_memory_units",
        embedding_dim: int = 512
    ):
        """初始化Milvus操作类"""
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

    # def _parse_datetime_strings(self, obj):
    #     """递归解析对象中的ISO格式日期字符串，转换回datetime对象"""
    #     if isinstance(obj, dict):
    #         for key, value in obj.items():
    #             if isinstance(value, str) and len(value) > 18:
    #                 try:
    #                     dt = datetime.fromisoformat(value)
    #                     obj[key] = dt
    #                 except (ValueError, TypeError):
    #                     pass
    #             elif isinstance(value, (dict, list)):
    #                 obj[key] = self._parse_datetime_strings(value)
    #     elif isinstance(obj, list):
    #         for i, item in enumerate(obj):
    #             obj[i] = self._parse_datetime_strings(item)
    #     return obj
    
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
                FieldSchema(name="uid", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="space_names", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=100),
                FieldSchema(name="raw_data", dtype=DataType.JSON),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            
            schema = CollectionSchema(
                fields, 
                description=f"Hippo memory units with {self.embedding_dim}-dimensional embeddings"
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
            self.collection.create_index("embedding", index_params)
            self.collection.load()
            
            logging.info(f"成功创建集合 '{self.collection_name}'")
            return True
        
        except Exception as e:
            logging.error(f"创建集合失败: {e}")
            return False
    
    def add_unit(self, unit: MemoryUnit, space_names: Optional[List[str]] = None) -> bool:
        """
        添加/更新MemoryUnit到Milvus
        
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
        
        if unit.embedding is None:
            logging.warning(f"单元 '{unit.uid}' 没有向量表示，无法添加到Milvus")
            return False
        
        try:
            # 删除可能存在的旧实体
            self.delete_unit(unit.uid)
            # 处理raw_data和metadata为JSON字符串
            raw_data_json = json.dumps(unit.raw_data)
            metadata_json = json.dumps(unit.metadata if hasattr(unit, 'metadata') and unit.metadata else {})
            # raw_data_json = json.dumps(unit.raw_data, cls=DateTimeEncoder)
            # metadata_json = json.dumps(unit.metadata if hasattr(unit, 'metadata') and unit.metadata else {}, cls=DateTimeEncoder)

            # 准备数据
            data = [
                [unit.uid],  # uid
                [space_names or []],  # space_names
                [raw_data_json],  # raw_data
                [metadata_json],  # metadata
                [unit.embedding.astype(np.float32).tolist()]  # embedding
            ]
            
            # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            logging.info(f"成功添加内存单元 '{unit.uid}' 到Milvus")
            return True
            
        except Exception as e:
            logging.error(f"添加内存单元 '{unit.uid}' 到Milvus失败: {e}")
            return False
    
    def delete_unit(self, unit_id: str) -> bool:
        """删除Milvus中的MemoryUnit"""
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return False
        
        try:
            expr = f'uid == "{unit_id}"'
            self.collection.delete(expr)
            logging.info(f"成功从Milvus删除内存单元 '{unit_id}'")
            return True
            
        except MilvusException as e:
            # 如果是因为实体不存在而失败，视为成功
            if "entity not found" in str(e).lower():
                return True
            logging.error(f"从Milvus删除内存单元 '{unit_id}' 失败: {e}")
            return False
    
    def get_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """
        通过ID从Milvus获取完整的MemoryUnit对象
        
        返回:
            Optional[MemoryUnit]: 完整的MemoryUnit对象，包含ID、raw_data和向量
        """
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return None
        
        try:
            expr = f'uid == "{unit_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["uid", "raw_data", "embedding", "metadata"]
            )
            
            if not results:
                return None
            
            unit_data = results[0]
            
            # 获取向量数据
            embedding = None
            if "embedding" in unit_data:
                embedding = np.array(unit_data["embedding"], dtype=np.float32)

            # 解析raw_data
            raw_data = {}
            if "raw_data" in unit_data:
                if isinstance(unit_data["raw_data"], str):
                    try:
                        raw_data = json.loads(unit_data["raw_data"])
                    except:
                        raw_data = {}
                else:
                    raw_data = unit_data["raw_data"]

            # 解析metadata
            metadata = {}
            if "metadata" in unit_data:
                if isinstance(unit_data["metadata"], str):
                    try:
                        metadata = json.loads(unit_data["metadata"])
                    except:
                        metadata = {}
                else:
                    metadata = unit_data["metadata"]

            # 创建MemoryUnit对象
            return MemoryUnit(
                uid=unit_data["uid"],
                raw_data=raw_data,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as e:
            logging.error(f"从Milvus获取内存单元 '{unit_id}' 失败: {e}")
            return None
    
    def get_units_batch(self, unit_ids: List[str]) -> List[MemoryUnit]:
        """
        批量获取多个MemoryUnit
        
        参数:
            unit_ids: 要获取的MemoryUnit ID列表
            
        返回:
            List[MemoryUnit]: MemoryUnit对象列表
        """
        if not unit_ids:
            return []
            
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return []
        
        try:
            id_list_str = ", ".join([f'"{id}"' for id in unit_ids])
            expr = f'uid in [{id_list_str}]'  # 使用uid而不是id
            results = self.collection.query(
                expr=expr,
                output_fields=["uid", "raw_data", "embedding", "metadata"]  # 使用uid而不是id
            )
                        
            units = []
            for unit_data in results:
                # 获取向量数据
                embedding = None
                if "embedding" in unit_data:
                    embedding = np.array(unit_data["embedding"], dtype=np.float32)
                
                # 创建MemoryUnit对象
                unit = MemoryUnit(
                    uid=unit_data["uid"],
                    raw_data=unit_data.get("raw_data", {}),
                    metadata=unit_data.get("metadata", {}),  # 添加这行
                    embedding=embedding
                )
                units.append(unit)
            
            return units
            
        except Exception as e:
            logging.error(f"批量获取内存单元失败: {e}")
            return []
    
    def search_similarity(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5, 
        space_name: Optional[str] = None,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        在Milvus中进行相似性搜索，返回完整的MemoryUnit对象
        
        参数:
            query_embedding: 查询向量
            k: 返回的最相似结果数量
            space_name: 如果提供，则仅在特定空间内搜索
            filter_ids: 如果提供，则只从这些ID中搜索
            
        返回:
            List[Tuple[MemoryUnit, float]]: [(memory_unit, distance), ...]
        """
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return []
        
        try:
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 64}
            }
            expr = None
            
            # 修改过滤表达式构建部分
            if space_name and filter_ids:
                id_list_str = ", ".join([f'"{id}"' for id in filter_ids])
                expr = f'array_contains(space_names, "{space_name}") and uid in [{id_list_str}]'  # 改为uid
            elif space_name:
                expr = f'array_contains(space_names, "{space_name}")'
            elif filter_ids:
                id_list_str = ", ".join([f'"{id}"' for id in filter_ids])
                expr = f'uid in [{id_list_str}]'  # 改为uid
            
            # 修改search调用 - 只有当expr不为None时才传入expr参数
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 64}
            }

            if expr:
                results = self.collection.search(
                    data=[query_embedding.astype(np.float32).tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=k,
                    expr=expr,
                    output_fields=["uid", "raw_data", "embedding", "metadata"]
                )
            else:
                # 没有过滤条件时省略expr参数
                results = self.collection.search(
                    data=[query_embedding.astype(np.float32).tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=k,
                    output_fields=["uid", "raw_data", "embedding", "metadata"]
                )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                        # 修改处理搜索结果的部分
                    try:
                        # 直接访问Hit对象的属性
                        unit_id = hit.id
                        
                        # 获取raw_data
                        raw_data = {}
                        if hasattr(hit, 'entity') and hit.entity is not None:
                            if isinstance(hit.entity, dict) and "raw_data" in hit.entity:
                                raw_data = hit.entity["raw_data"]
                        elif hasattr(hit, 'raw_data'):
                            raw_data = hit.raw_data
                        
                        # 获取metadata - 添加这部分代码
                        metadata = {}
                        if hasattr(hit, 'entity') and hit.entity is not None:
                            if isinstance(hit.entity, dict) and "metadata" in hit.entity:
                                metadata = hit.entity["metadata"]
                        elif hasattr(hit, 'metadata'):
                            metadata = hit.metadata
                        
                        # 确保metadata是字典类型
                        if not isinstance(metadata, dict):
                            metadata = {}
                        
                        # 获取向量
                        embedding = None
                        if hasattr(hit, 'entity') and hit.entity is not None:
                            if isinstance(hit.entity, dict) and "embedding" in hit.entity:
                                embedding = np.array(hit.entity["embedding"], dtype=np.float32)
                        elif hasattr(hit, 'embedding') and hit.embedding is not None:
                            embedding = np.array(hit.embedding, dtype=np.float32)
                        
                        # 创建MemoryUnit对象 - 添加metadata参数
                        unit = MemoryUnit(
                            uid=unit_id,
                            raw_data=raw_data,
                            metadata=metadata,  # 添加这个参数
                            embedding=embedding
                        )
                        
                        # 获取距离/分数
                        distance = 0.0
                        if hasattr(hit, 'distance'):
                            distance = hit.distance
                        elif hasattr(hit, 'score'):
                            distance = hit.score
                        
                        search_results.append((unit, distance))
                    except Exception as e:
                        logging.error(f"处理搜索结果项时出错: {e}")
                        continue
            
            return search_results
            
        except Exception as e:
            logging.error(f"Milvus相似性搜索失败: {e}")
            return []
    
    def get_units_by_space(self, space_name: str) -> List[MemoryUnit]:
        """
        获取特定空间中的所有MemoryUnit
        
        参数:
            space_name: 空间名称
            
        返回:
            List[MemoryUnit]: MemoryUnit对象列表
        """
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return []
        
        try:
            expr = f'array_contains(space_names, "{space_name}")'
            results = self.collection.query(
                expr=expr,
                output_fields=["uid", "raw_data", "embedding", "metadata"]
            )
            
            units = []
            for unit_data in results:
                # 获取向量数据
                embedding = None
                if "embedding" in unit_data:
                    embedding = np.array(unit_data["embedding"], dtype=np.float32)
                
                # 创建MemoryUnit对象
                unit = MemoryUnit(
                    uid=unit_data["uid"],
                    raw_data=unit_data.get("raw_data", {}),
                    metadata=unit_data.get("metadata", {}),  # 添加这行
                    embedding=embedding
                )
                units.append(unit)
            
            return units
            
        except Exception as e:
            logging.error(f"从Milvus获取空间 '{space_name}' 的内存单元失败: {e}")
            return []
    
    def update_space_membership(self, unit_id: str, space_names: List[str], operation: str = "add") -> bool:
        """
        更新内存单元的空间成员资格
        
        参数:
            unit_id: 内存单元ID
            space_names: 空间名称列表
            operation: "add"(添加到这些空间) 或 "remove"(从这些空间移除) 或 "set"(设置为这些空间)
            
        返回:
            更新是否成功
        """
        if not self.is_connected or not hasattr(self, 'collection'):
            logging.error("未连接到Milvus服务器或集合未初始化")
            return False
        
        try:
            # 先获取当前的空间成员资格
            expr = f'uid == "{unit_id}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["space_names"]
            )
            
            if not results:
                logging.warning(f"内存单元 '{unit_id}' 不存在")
                return False
            
            current_spaces = results[0].get("space_names", [])
            
            # 根据操作更新空间列表
            if operation == "add":
                new_spaces = list(set(current_spaces).union(set(space_names)))
            elif operation == "remove":
                new_spaces = list(set(current_spaces).difference(set(space_names)))
            elif operation == "set":
                new_spaces = space_names
            else:
                logging.error(f"不支持的操作: {operation}")
                return False
            
            # 更新空间成员资格
            self.collection.update(
                expr=expr,
                data={"space_names": new_spaces}
            )
            
            logging.info(f"成功更新内存单元 '{unit_id}' 的空间成员资格")
            return True
            
        except Exception as e:
            logging.error(f"更新内存单元 '{unit_id}' 的空间成员资格失败: {e}")
            return False
    
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
    
    metadata = {
        "created": datetime.now(),
        "updated": datetime.now(),
    }

    # 添加内存单元
    unit = MemoryUnit(uid="unit_1",metadata=metadata, embedding=np.random.rand(512), raw_data={"key": "value"})
    milvus_operator.add_unit(unit, space_names=["space_1"])
    
    # 搜索相似内存单元
    results = milvus_operator.search_similarity(query_embedding=np.random.rand(512), k=5)
    print("搜索结果:", results)
    
    # 获取特定内存单元
    unit_data = milvus_operator.get_unit("unit_1")
    print("获取的内存单元:", unit_data)
    
    # 关闭连接
    milvus_operator.close()